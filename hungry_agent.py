from dataclasses import dataclass
import heapq
import numpy as np
import traceback
from typing import List, Tuple

from battlesnake_types import Food, GameState, MoveAction, Direction, BaseAgent, Point

# Safe coordinate offsets for our directions
DIRECTION_OFFSETS = {
    Direction.UP: (0, 1),
    Direction.DOWN: (0, -1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0)
}

# ---------------------------------------------------------
# Battlesnake Agent Implementation
# ---------------------------------------------------------
@dataclass
class AgentState:
    possible_food: list[Food]

class HungryAgent(BaseAgent):
    def __init__(self):
        self.agent_states: dict[str, AgentState] = {}

    def get_name(self):
        return "MeinKraft"

    def get_color(self):
        return "#FFC13C"

    def get_author(self):
        return "MeinKraft"

    def start(self, game_state: GameState):
        self.agent_states[game_state.game.id] = AgentState(possible_food=[])

    def get_manhattan_distance(coord1, coord2):
        """Calculates the grid distance between two coordinates."""
        return abs(coord1['x'] - coord2['x']) + abs(coord1['y'] - coord2['y'])

    def move(self, game_state: GameState) -> MoveAction:
        try:
            return self._calculate_move(game_state)
        except Exception as e:
            print(f"CRASH ON TURN {game_state.turn}: {str(e)}")
            traceback.print_exc()
            return MoveAction(move=Direction.UP)

    def _calculate_move(self, game_state: GameState) -> MoveAction:
        if game_state.game.id not in self.agent_states:
            self.start(game_state)

        agent_state = self.agent_states[game_state.game.id]
        head = game_state.you.head
        assert head is not None

        # 1. Update Food Memory (Simplified)
        for food in game_state.board.food:
            if food not in agent_state.possible_food:
                agent_state.possible_food.append(food)

        # 2. Build the Maps
        danger_map = self.get_obstacle_map(game_state, use_danger_zones=True)
        strict_map = self.get_obstacle_map(game_state, use_danger_zones=False)

        # 3. Assess Hunger
        my_health = game_state.you.health
        my_length = game_state.you.length
        
        largest_enemy_length = max([s.length for s in game_state.board.snakes if s.id != game_state.you.id], default=0)
        is_hungry = (my_health < 40) or (my_length <= largest_enemy_length + 1)
        
        result_direction = None
        
# STRATEGY 1: Hunt for food (Using Danger Map)
        if is_hungry:
            min_distance = float('inf')
            for food in agent_state.possible_food:
                direction, length = self.a_star_wrapper(danger_map, head, food)
                if length < min_distance:
                    result_direction = direction
                    min_distance = length

        # ==========================================
        # STRATEGY 2: Hunt Smaller Snakes 
        # ==========================================
        # If we aren't hungry, we have the size advantage to hunt!
        if result_direction is None and not is_hungry:
            min_hunt_dist = float('inf')
            best_hunt_dir = None
            
            for enemy in game_state.board.snakes:
                # Find enemies strictly smaller than us (buffer of 1)
                if enemy.id != game_state.you.id and my_length > (enemy.length + 1):
                    
                    # Check all 4 possible next steps using your existing DIRECTION_OFFSETS
                    for d, (dx, dy) in DIRECTION_OFFSETS.items():
                        nx = head.x + dx
                        ny = head.y + dy
                        
                        # 1. Is the step physically on the board?
                        if 0 <= nx < game_state.board.width and 0 <= ny < game_state.board.height:
                            
                            # 2. Is the step safe according to the Danger Map?
                            # (If it's safe, the array value should be False/0)
                            if not danger_map[ny, nx]: 
                                
                                # 3. Calculate Manhattan distance from this step to enemy head
                                dist = abs(nx - enemy.head.x) + abs(ny - enemy.head.y)
                                
                                if dist < min_hunt_dist:
                                    min_hunt_dist = dist
                                    best_hunt_dir = d
                                    
            # If we found a safe path toward prey, lock it in!
            if best_hunt_dir is not None:
                result_direction = best_hunt_dir
        # ==========================================

        # STRATEGY 3: Flood fill survival (Using Danger Map)
        # We only do this now if we aren't hungry AND there are no smaller snakes to hunt
        if result_direction is None:
            result_direction = self.get_best_survival_move(game_state, danger_map)
            
        # STRATEGY 4: Panic Mode Survival (Using Strict Map - ignores danger zones)
        if result_direction is None:
            result_direction = self.get_best_survival_move(game_state, strict_map)

        my_name = game_state.you.name
        move_str = getattr(result_direction, 'name', str(result_direction))
        print(f"Turn {game_state.turn:03d} | {my_name} | Hungry: {is_hungry} | Move: {move_str}")    

        return MoveAction(move=result_direction)

    def get_obstacle_map(self, game_state: GameState, use_danger_zones: bool) -> np.ndarray:
        board_height = game_state.board.height
        board_width = game_state.board.width
        obstacle_map = np.zeros((board_height, board_width), dtype=bool)
        my_snake = game_state.you
        
        # Mark OUR OWN body explicitly just to be 100% safe
        for body_part in my_snake.body[:-1]:
            if 0 <= body_part.x < board_width and 0 <= body_part.y < board_height:
                obstacle_map[body_part.y, body_part.x] = 1

        # Mark all enemy bodies
        for snake in game_state.board.snakes:
            if snake.id == my_snake.id:
                continue
                
            for body_part in snake.body[:-1]:
                if 0 <= body_part.x < board_width and 0 <= body_part.y < board_height:
                    obstacle_map[body_part.y, body_part.x] = 1
            
            # Danger Zones: ONLY apply if the enemy is STRICTLY LARGER than us
            if use_danger_zones and snake.length > my_snake.length:
                head = snake.head
                if head is not None:
                    danger_zones = [
                        (head.x, head.y + 1), 
                        (head.x, head.y - 1), 
                        (head.x - 1, head.y), 
                        (head.x + 1, head.y)  
                    ]
                    for dx, dy in danger_zones:
                        if 0 <= dx < board_width and 0 <= dy < board_height:
                            obstacle_map[dy, dx] = 1

        return obstacle_map

    def get_available_space(self, start_x: int, start_y: int, obstacle_map: np.ndarray, max_space: int) -> int:
        board_height, board_width = obstacle_map.shape
        
        if start_x < 0 or start_x >= board_width or start_y < 0 or start_y >= board_height:
            return 0
        if obstacle_map[start_y, start_x]:
            return 0
            
        visited = set()
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))
        space_count = 0
        
        while queue and space_count < max_space:
            curr_x, curr_y = queue.pop(0)
            space_count += 1
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr_x + dx, curr_y + dy
                if 0 <= nx < board_width and 0 <= ny < board_height:
                    if not obstacle_map[ny, nx] and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                        
        return space_count

    def get_best_survival_move(self, game_state: GameState, obstacle_map: np.ndarray) -> Direction | None:
        head = game_state.you.head
        my_length = game_state.you.length
        
        best_direction = None
        max_space_found = -1
        
        for d, (dx, dy) in DIRECTION_OFFSETS.items():
            next_x = head.x + dx
            next_y = head.y + dy
            
            space = self.get_available_space(next_x, next_y, obstacle_map, max_space=my_length)
            
            if space > max_space_found and space > 0:
                max_space_found = space
                best_direction = d
                
        return best_direction

    def end(self, game_state: GameState):
        if game_state.game.id in self.agent_states:
            del self.agent_states[game_state.game.id]

    def a_star_wrapper(self, obstacle_map, start_node, target_node):
        start_tuple = (start_node.x, start_node.y)
        target_tuple = (target_node.x, target_node.y)
        
        path = self.a_star(obstacle_map, start_tuple, target_tuple)
        
        if not path or len(path) < 2:
            return None, float('inf')

        next_pos = path[1]
        dx = next_pos[0] - start_node.x
        dy = next_pos[1] - start_node.y
        
        if dx == 0 and dy == 1: return Direction.UP, len(path)
        elif dx == 0 and dy == -1: return Direction.DOWN, len(path)
        elif dx == -1 and dy == 0: return Direction.LEFT, len(path)
        elif dx == 1 and dy == 0: return Direction.RIGHT, len(path)
            
        return None, float('inf')

    def a_star(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]] | None:
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

            x, y = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= ny < h and 0 <= nx < w and not grid[ny, nx]:
                    neighbor, new_g = (nx, ny), g_score[current] + 1
                    if new_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = new_g
                        f_score = new_g + abs(nx - goal[0]) + abs(ny - goal[1])
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