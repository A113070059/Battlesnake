from collections import deque
from dataclasses import dataclass, field
import heapq
import numpy as np
from typing import List, Tuple

from battlesnake_types import Food, GameState, MoveAction, Direction, BaseAgent, Point

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_obstacle_map(game_state: GameState, include_tails: bool = False):
    obstacle_map = np.zeros((game_state.board.height, game_state.board.width), dtype=bool)
    
    for snake in game_state.board.snakes:
        body = snake.body if include_tails else snake.body[:-1]
        for body_part in body:
            if body_part is None:
                # we don't see this body section, could be many parts long
                continue
            obstacle_map[body_part.y, body_part.x] = 1
    
    return obstacle_map

def get_vision_mask(width: int, height: int, center: Point, radius: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    dist_sq = abs(x - center.x) + abs(y - center.y)
    return dist_sq <= radius


def get_legal_directions(game_state: GameState, obstacle_map: np.ndarray) -> list[Direction]:
    """Return directions that stay in bounds and avoid known body segments."""
    head = game_state.you.head
    assert head is not None

    legal_directions = []
    for direction in Direction:
        next_x = head.x + direction.dx
        next_y = head.y + direction.dy

        if not (0 <= next_x < game_state.board.width):
            continue
        if not (0 <= next_y < game_state.board.height):
            continue
        if obstacle_map[next_y, next_x]:
            continue

        legal_directions.append(direction)

    return legal_directions


def flood_fill_area(grid: np.ndarray, start: tuple[int, int]) -> int:
    """Count the known free cells reachable from start."""
    height, width = grid.shape
    start_y, start_x = start
    if not (0 <= start_y < height and 0 <= start_x < width):
        return 0
    if grid[start_y, start_x]:
        return 0

    visited = {start}
    queue = deque([start])
    while queue:
        current_y, current_x = queue.popleft()
        for delta_y, delta_x in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_y = current_y + delta_y
            next_x = current_x + delta_x
            next_cell = (next_y, next_x)

            if not (0 <= next_y < height and 0 <= next_x < width):
                continue
            if grid[next_y, next_x] or next_cell in visited:
                continue

            visited.add(next_cell)
            queue.append(next_cell)

    return len(visited)


def get_future_space(
    game_state: GameState,
    direction: Direction,
    will_grow: bool,
) -> int:
    """Estimate how much space remains after making a candidate move."""
    head = game_state.you.head
    assert head is not None

    next_x = head.x + direction.dx
    next_y = head.y + direction.dy
    future_obstacles = get_obstacle_map(game_state, include_tails=True)

    # Our tail normally moves away. If we eat this turn, it stays in place.
    if not will_grow and game_state.you.body:
        tail = game_state.you.body[-1]
        if tail is not None:
            future_obstacles[tail.y, tail.x] = False

    # The new head is the starting cell for the flood fill, not an obstacle.
    future_obstacles[next_y, next_x] = False
    return flood_fill_area(future_obstacles, (next_y, next_x))


def get_hazard_positions(game_state: GameState) -> set[tuple[int, int]]:
    return {(hazard.x, hazard.y) for hazard in game_state.board.hazards}


def get_food_positions(game_state: GameState) -> set[tuple[int, int]]:
    return {(food.x, food.y) for food in game_state.board.food}


def get_enemy_risk(
    game_state: GameState,
    candidate: tuple[int, int],
    remembered_enemy_cells: dict[tuple[int, int], int],
) -> float:
    """Score danger from visible heads and recently remembered hidden cells."""
    candidate_x, candidate_y = candidate
    risk = 0.0

    # A remembered body cell is uncertain, not a guaranteed obstacle. Penalize
    # it heavily so the agent avoids it when another route exists.
    if candidate in remembered_enemy_cells:
        risk += 180.0

    you_length = game_state.you.length
    for snake in game_state.board.snakes:
        if snake.id == game_state.you.id or snake.head is None:
            continue

        distance = abs(candidate_x - snake.head.x) + abs(candidate_y - snake.head.y)
        if distance == 0:
            # Moving onto an enemy head is never a safe default under fog of war.
            risk += 2000.0
        elif distance == 1:
            # The enemy can potentially move into our candidate cell next turn.
            # If its length is unknown, treating it as at least our length is safer.
            if snake.length >= you_length:
                risk += 450.0
            else:
                risk += 140.0
        elif distance == 2:
            risk += 35.0

    return risk


def get_food_direction_score(
    game_state: GameState,
    remembered_food: list[Food],
    obstacle_map: np.ndarray,
    candidate: Direction,
    reachable_area: int,
) -> float:
    """Reward food only when its path is compatible with survival."""
    head = game_state.you.head
    assert head is not None
    health = game_state.you.health if game_state.you.health is not None else 100
    body_length = len([part for part in game_state.you.body if part is not None])
    best_score = 0.0

    for food in remembered_food:
        food_direction, path_length = a_star_wrapper(obstacle_map, head, food)
        if food_direction != candidate or path_length >= 9999999:
            continue

        steps_to_food = max(0, path_length - 1)
        score = max(0.0, 120.0 - steps_to_food * 7.0)

        # Food becomes urgent when the route may outlast our health.
        if steps_to_food >= health:
            score -= 220.0
        elif health <= 40:
            score += 180.0 + (40 - health) * 4.0
        elif health > 75:
            # When healthy, space is more valuable than risky growth.
            score *= 0.45

        if reachable_area <= body_length:
            score -= 400.0
        elif reachable_area < body_length + 3:
            score -= 160.0

        best_score = max(best_score, score)

    return best_score


def get_move_score(
    game_state: GameState,
    agent_state: "AgentState",
    obstacle_map: np.ndarray,
    direction: Direction,
) -> float:
    head = game_state.you.head
    assert head is not None
    next_x = head.x + direction.dx
    next_y = head.y + direction.dy
    candidate = (next_x, next_y)

    visible_food = get_food_positions(game_state)
    will_grow = candidate in visible_food
    reachable_area = get_future_space(game_state, direction, will_grow)
    body_length = len([part for part in game_state.you.body if part is not None])

    score = min(reachable_area, 100) * 6.0
    if reachable_area <= body_length:
        score -= 1200.0
    elif reachable_area < body_length + 3:
        score -= 450.0

    score -= get_enemy_risk(game_state, candidate, agent_state.remembered_enemy_cells)

    if candidate in get_hazard_positions(game_state):
        health = game_state.you.health if game_state.you.health is not None else 100
        hazard_damage = game_state.game.ruleset.settings.hazardDamagePerTurn
        if health <= hazard_damage + 1 and not will_grow:
            score -= 1200.0
        else:
            score -= 50.0 + hazard_damage * 2.0

    # A small centrality bonus reduces unnecessary edge and corner commitment.
    center_x = (game_state.board.width - 1) / 2.0
    center_y = (game_state.board.height - 1) / 2.0
    center_distance = abs(next_x - center_x) + abs(next_y - center_y)
    score += max(0.0, 24.0 - center_distance * 2.0)

    score += get_food_direction_score(
        game_state,
        agent_state.possible_food,
        obstacle_map,
        direction,
        reachable_area,
    )
    return score

# ---------------------------------------------------------
# Battlesnake Agent Implementation
# ---------------------------------------------------------
@dataclass
class AgentState:
    possible_food: list[Food] = field(default_factory=list)
    remembered_enemy_cells: dict[tuple[int, int], int] = field(default_factory=dict)
    last_move: Direction | None = None

class HungryAgent(BaseAgent):
    def __init__(self):
        self.agent_states: dict[str, AgentState] = {}

    def get_name(self):
        return "Hungry Caterpillar"

    def get_color(self):
        return "#FFC13C"

    def get_author(self):
        return "Gluttony"

    def start(self, game_state: GameState):
        """start is called when the battlesnake begins a game"""
        self.agent_states[game_state.game.id] = AgentState()

    def update_enemy_memory(self, game_state: GameState, agent_state: AgentState, vision_mask: np.ndarray):
        """Remember recently seen enemy cells, while forgetting disproven cells."""
        visible_enemy_cells: set[tuple[int, int]] = set()
        current_turn = game_state.turn

        for snake in game_state.board.snakes:
            if snake.id == game_state.you.id:
                continue
            for body_part in snake.body:
                if body_part is None:
                    continue
                position = (body_part.x, body_part.y)
                visible_enemy_cells.add(position)
                agent_state.remembered_enemy_cells[position] = current_turn

        for position, last_seen_turn in list(agent_state.remembered_enemy_cells.items()):
            x, y = position
            too_old = current_turn - last_seen_turn > 8
            now_visible_and_empty = vision_mask[y, x] and position not in visible_enemy_cells
            if too_old or now_visible_and_empty:
                del agent_state.remembered_enemy_cells[position]

    def move(self, game_state: GameState) -> MoveAction:
        """move is called on every turn and returns your next move"""
        # update the food memory (saved across move steps)
        agent_state = self.agent_states[game_state.game.id]
        head = game_state.you.head
        assert head is not None
        view_radius = game_state.game.ruleset.settings.viewRadius
        if view_radius is None:
            view_radius = max(game_state.board.width, game_state.board.height)
        vision_mask = get_vision_mask(width=game_state.board.width, height=game_state.board.height, center=head, radius=view_radius)

        updated_food = []
        # keep food that is not in vision range in memory
        for food in agent_state.possible_food:
            if not vision_mask[food.y, food.x]:
                updated_food.append(food)
        # add visible food
        visible_food = game_state.board.food
        for food in visible_food:
            if food not in updated_food:
                updated_food.append(food)
        agent_state.possible_food = updated_food

        self.update_enemy_memory(game_state, agent_state, vision_mask)

        # build an obstacle map
        obstacle_map = get_obstacle_map(game_state)

        legal_directions = get_legal_directions(game_state, obstacle_map)
        if legal_directions:
            scored_directions = [
                (get_move_score(game_state, agent_state, obstacle_map, direction), direction)
                for direction in legal_directions
            ]
            highest_score = max(score for score, _ in scored_directions)
            best_directions = [
                direction
                for score, direction in scored_directions
                if score >= highest_score - 1.0
            ]
            result_direction = best_directions[int(np.random.randint(len(best_directions)))]
        else:
            result_direction = self.random_fallback_move(game_state, obstacle_map)

        agent_state.last_move = result_direction

        return MoveAction(move=result_direction)
    
    def random_fallback_move(self, game_state: GameState, obstacle_map: np.ndarray) -> Direction:
        head = game_state.you.head
        assert head is not None
        clear_directions = get_legal_directions(game_state, obstacle_map)
        
        result_direction = (
            clear_directions[int(np.random.randint(len(clear_directions)))]
            if clear_directions
            else Direction.UP
        )

        return result_direction

    def end(self, game_state: GameState):
        """end is called when the battlesnake finishes a game"""
        if game_state.game.id in self.agent_states:
            del self.agent_states[game_state.game.id]


# ---------------------------------------------------------
# A* Algorithm
# ---------------------------------------------------------
def a_star_wrapper(grid: np.ndarray, start: Point, goal: Point) -> tuple[Direction | None, int]:
    """Converts from battlesnake x-y coords to i-j index-tuples used by a_star()."""
    path = a_star(grid, (start.y, start.x), (goal.y, goal.x))
    if path is None:
        return None, 9999999

    if len(path) < 2:
        return None, 0

    next_pos = path[1]
    result_direction = Direction.from_board_delta((next_pos[1] - start.x, next_pos[0] - start.y))
    return result_direction, len(path) 

def a_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]] | None:
    h, w = grid.shape
    open_set, g_score, came_from = [(0, start)], {start: 0}, {}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not grid[nr, nc]:
                neighbor, new_g = (nr, nc), g_score[current] + 1

                if new_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = new_g
                    f_score = new_g + abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(open_set, (f_score, neighbor))

    return None


if __name__ == "__main__":
    import sys
    from battlesnake_server import start_server

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <port>")
        sys.exit(1)

    agent = HungryAgent()
    port = int(sys.argv[1])

    start_server(agent=agent, port=port)
