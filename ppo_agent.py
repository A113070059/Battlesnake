"""Deploy the trained PPO policy as a Battlesnake Blackout HTTP agent.

The PPO policy is used as the primary decision maker.  A small safety
supervisor rejects moves that are known to collide with a wall/body or to die
from the currently visible health/hazard state, then asks HungryAgent for a
fallback.  Hidden enemy cells are still uncertain: no HTTP agent can prove
that a move is safe from an enemy that is outside its view.
"""

from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import torch

try:
    from stable_baselines3 import PPO
except ImportError as error:  # pragma: no cover - deployment dependency
    raise RuntimeError(
        "PPO deployment requires stable-baselines3 and PyTorch. "
        "Install the PPO requirements before starting the server."
    ) from error

try:
    import hisss
    from hisss.game.state import BattleSnakeState
except ImportError as error:  # pragma: no cover - deployment dependency
    raise RuntimeError("PPO deployment requires hisss==1.2.0") from error

from battlesnake_types import BaseAgent, Direction, GameState, MoveAction
from hungry_agent import (
    HungryAgent,
    get_hazard_positions,
    get_legal_directions,
    get_obstacle_map,
)
from ppo_training.config import ExperimentConfig
from ppo_training.observation import ExplicitMemory, ObservationBuilder

# The Render instance does not need PyTorch parallelism for one move at a time.
torch.set_num_threads(1)

logger = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    def __init__(self, model_path: str, config_path: str):
        super().__init__()
        self.model_path = model_path
        self.config_path = config_path

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Config file not found: {os.path.basename(self.config_path)}"
            )
        self.config = ExperimentConfig.load(self.config_path)

        self.obs_builder = ObservationBuilder(self.config)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {os.path.basename(self.model_path)}"
            )

        # The checkpoint contains the custom BlackoutCNN class.  Importing the
        # module before PPO.load makes its class path resolvable on Render.
        from ppo_training import network as _network  # noqa: F401

        print(f"Loading PPO model: {os.path.basename(self.model_path)}...")
        self.model = PPO.load(self.model_path, device="cpu")
        expected_shape = (
            self.config.input_channels,
            self.config.observation_size,
            self.config.observation_size,
        )
        actual_shape = tuple(self.model.observation_space.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                "Model/config observation mismatch: "
                f"model={actual_shape}, config={expected_shape}"
            )
        print("Model loaded successfully.")

        # One game is normally active on the Battlesnake server, but keeping
        # these keyed by game id also makes accidental overlapping requests
        # safe.
        self.memories: dict[str, ExplicitMemory] = {}
        self.slot_ids: dict[str, dict[str, int]] = {}
        self.safety_agent = HungryAgent()

    def get_name(self):
        return "PPO Blackout Agent"

    def get_color(self):
        return "#800080"

    def get_author(self):
        return "MeinKraft"

    @staticmethod
    def _snake_is_alive(snake) -> bool:
        """Treat API elimination markers as authoritative when present."""
        return (
            getattr(snake, "elimination_event", None) is None
            and getattr(snake, "elimination", None) is None
        )

    def _register_snake_slots(self, game_state: GameState) -> dict[str, int]:
        """Keep a stable Hisss player slot for each Battlesnake id."""
        game_id = game_state.game.id
        slots = self.slot_ids.setdefault(game_id, {})
        slots[game_state.you.id] = 0

        for snake in game_state.board.snakes:
            if snake.id == game_state.you.id or snake.id in slots:
                continue
            used = set(slots.values())
            free_slots = [
                slot
                for slot in range(1, self.config.num_players)
                if slot not in used
            ]
            if not free_slots:
                logger.warning(
                    "Ignoring snake %s: more than %d snakes were reported",
                    snake.id,
                    self.config.num_players,
                )
                continue
            slots[snake.id] = free_slots[0]
        return slots

    @staticmethod
    def _valid_point(point, width: int, height: int) -> bool:
        return (
            point is not None
            and 0 <= point.x < width
            and 0 <= point.y < height
        )

    def _has_hidden_body(self, snake, width: int, height: int) -> bool:
        """Whether the API body cannot establish the snake's true length."""
        if not snake.body:
            return True
        return any(
            not self._valid_point(part, width, height)
            for part in snake.body
        )

    @staticmethod
    def _find_hidden_placeholder(
        width: int,
        height: int,
        head,
        view_radius: int | None,
        occupied: set[tuple[int, int]],
    ) -> tuple[int, int]:
        """Find a valid fake cell outside our view for unknown body segments."""
        candidates = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if (x, y) not in occupied
        ]
        if head is not None and view_radius is not None:
            outside_view = [
                coordinate
                for coordinate in candidates
                if abs(coordinate[0] - head.x) + abs(coordinate[1] - head.y)
                > view_radius
            ]
            if outside_view:
                return outside_view[0]
        if candidates:
            return candidates[0]
        # A completely full board is already terminal in practice.  Returning
        # a valid coordinate is still safer than passing an empty body to C++.
        return (0, 0)

    def _body_for_hisss(
        self,
        snake,
        game_state: GameState,
        placeholder: tuple[int, int],
    ) -> tuple[list[tuple[int, int]], int]:
        """Convert a partial Blackout body without treating None as a death.

        Hisss requires concrete coordinates, while Blackout deliberately sends
        None for hidden segments.  Known coordinates are retained and unknown
        segments are represented by an off-screen placeholder, so Hisss can
        build a correctly shaped observation without inventing visible body
        cells from the last known segment.
        """
        width = game_state.board.width
        height = game_state.board.height
        body: list[tuple[int, int]] = []
        for point in snake.body:
            if self._valid_point(point, width, height):
                body.append((int(point.x), int(point.y)))
            else:
                body.append(placeholder)

        reported_length = int(snake.length or 0)
        if self._has_hidden_body(snake, width, height):
            # The API length is not trustworthy when any body segment is
            # hidden.  Do not create a long row of duplicate placeholders.
            target_length = max(1, len(body))
        else:
            target_length = max(1, len(body), reported_length)
        if not body:
            body = [placeholder]
        if len(body) < target_length:
            body.extend([placeholder] * (target_length - len(body)))
        return body, target_length

    def _known_safe_directions(
        self, game_state: GameState
    ) -> list[Direction]:
        """Return moves that are safe according to currently visible facts."""
        obstacle_map = get_obstacle_map(game_state)
        candidates = get_legal_directions(game_state, obstacle_map)
        head = game_state.you.head
        if not self._valid_point(
            head, game_state.board.width, game_state.board.height
        ):
            return []

        food = {
            (item.x, item.y)
            for item in game_state.board.food
            if self._valid_point(item, game_state.board.width, game_state.board.height)
        }
        hazards = get_hazard_positions(game_state)
        health = (
            int(game_state.you.health)
            if game_state.you.health is not None
            else 100
        )
        hazard_damage = int(
            game_state.game.ruleset.settings.hazardDamagePerTurn or 0
        )
        own_tail = game_state.you.body[-1] if game_state.you.body else None
        safe: list[Direction] = []

        for direction in candidates:
            target = (head.x + direction.dx, head.y + direction.dy)

            # HungryAgent intentionally allows a tail move because the tail
            # normally vacates.  Do not allow it when food keeps the tail in
            # place, since that collision is immediate.
            if (
                own_tail is not None
                and target == (own_tail.x, own_tail.y)
                and target in food
            ):
                continue

            # Health reaches zero after a non-food move at health 1.  The
            # hazard check mirrors HungryAgent's conservative rule.
            grows = target in food
            if health <= 1 and not grows:
                continue
            if target in hazards and health <= hazard_damage + 1 and not grows:
                continue

            # Avoid visible enemy heads and conservative head-to-head ties.
            unsafe_head = False
            for snake in game_state.board.snakes:
                if (
                    snake.id == game_state.you.id
                    or not self._snake_is_alive(snake)
                    or not self._valid_point(
                        snake.head,
                        game_state.board.width,
                        game_state.board.height,
                    )
                ):
                    continue
                enemy_head = (snake.head.x, snake.head.y)
                distance = abs(target[0] - enemy_head[0]) + abs(
                    target[1] - enemy_head[1]
                )
                if distance == 0 or (
                    distance == 1
                    and (
                        self._has_hidden_body(
                            snake,
                            game_state.board.width,
                            game_state.board.height,
                        )
                        or int(snake.length or 0)
                        >= int(game_state.you.length or 0)
                    )
                ):
                    unsafe_head = True
                    break
            if not unsafe_head:
                safe.append(direction)
        return safe

    def _sanitize_snake_for_safety(self, snake, width: int, height: int):
        """Replace hidden/invalid points before HungryAgent sees the state."""
        sanitized_body = [
            part if self._valid_point(part, width, height) else None
            for part in (snake.body or [])
        ]
        sanitized_head = (
            snake.head
            if self._valid_point(snake.head, width, height)
            else None
        )
        return snake.model_copy(
            update={
                "body": sanitized_body,
                "head": sanitized_head,
            }
        )

    def _safety_state(self, game_state: GameState) -> GameState:
        """Remove dead snakes and sanitize all coordinates for HungryAgent."""
        width = game_state.board.width
        height = game_state.board.height
        live_snakes = [
            self._sanitize_snake_for_safety(snake, width, height)
            for snake in game_state.board.snakes
            if self._snake_is_alive(snake)
        ]
        board = game_state.board.model_copy(update={"snakes": live_snakes})
        sanitized_you = self._sanitize_snake_for_safety(
            game_state.you,
            width,
            height,
        )
        return game_state.model_copy(
            update={
                "board": board,
                "you": sanitized_you,
            }
        )

    def _head_to_head_directions(
        self, game_state: GameState
    ) -> list[Direction]:
        """Find moves where a live enemy head could contest our destination.

        Hisss resolves a head-to-head when both snakes move onto the same
        destination cell.  Therefore the useful fallback case is an empty
        cell adjacent to an enemy head, not simply moving into the enemy's
        current head cell.  A known length advantage is preferred; hidden or
        equal lengths remain possible but less certain.
        """
        board = game_state.board
        own_head = game_state.you.head
        if not self._valid_point(own_head, board.width, board.height):
            return []

        own_length = int(game_state.you.length or 0)
        occupied: set[tuple[int, int]] = set()
        if self._snake_is_alive(game_state.you):
            for part in game_state.you.body:
                if self._valid_point(part, board.width, board.height):
                    occupied.add((int(part.x), int(part.y)))
        enemies = []
        for snake in board.snakes:
            if not self._snake_is_alive(snake):
                continue
            for part in snake.body:
                if self._valid_point(part, board.width, board.height):
                    occupied.add((int(part.x), int(part.y)))
            if snake.id == game_state.you.id:
                continue
            if self._valid_point(snake.head, board.width, board.height):
                enemies.append(snake)

        ranked: list[tuple[int, Direction]] = []
        for direction in Direction:
            target = (own_head.x + direction.dx, own_head.y + direction.dy)
            if not (
                0 <= target[0] < board.width
                and 0 <= target[1] < board.height
            ):
                continue

            # A head-to-head requires an open destination.  Moving into a
            # known body is still an ordinary body collision, not a useful
            # head-to-head opportunity.
            if target in occupied:
                continue

            for enemy in enemies:
                enemy_head = (int(enemy.head.x), int(enemy.head.y))
                distance = abs(target[0] - enemy_head[0]) + abs(
                    target[1] - enemy_head[1]
                )
                if distance != 1:
                    continue

                if self._has_hidden_body(enemy, board.width, board.height):
                    priority = 1  # Possible, but true enemy length is unknown.
                elif own_length > int(enemy.length or 0):
                    priority = 0  # Best chance: known length advantage.
                elif own_length == int(enemy.length or 0):
                    priority = 2  # Both may be eliminated in a tie.
                else:
                    priority = 3  # Still preferable to a certain wall/body hit.
                ranked.append((priority, direction))
                break

        if not ranked:
            return []
        best_priority = min(priority for priority, _ in ranked)
        return [
            direction
            for priority, direction in ranked
            if priority == best_priority
        ]

    def _supervised_move(
        self, game_state: GameState, ppo_direction: Direction | None
    ) -> MoveAction:
        """Prefer PPO when known-safe, otherwise use HungryAgent."""
        safety_state = self._safety_state(game_state)
        # Calling HungryAgent every turn keeps its food/enemy memory current,
        # even when PPO's move is accepted.
        hungry_action = self.safety_agent.move(safety_state).move
        known_safe = self._known_safe_directions(safety_state)

        if ppo_direction is not None and ppo_direction in known_safe:
            chosen = ppo_direction
            source = "PPO"
        elif hungry_action in known_safe:
            chosen = hungry_action
            source = "HungryAgent"
        elif known_safe:
            chosen = random.choice(known_safe)
            source = "random known-safe fallback"
        else:
            head_to_head = self._head_to_head_directions(safety_state)
            if head_to_head:
                chosen = random.choice(head_to_head)
                source = "random head-to-head fallback"
            else:
                # Every direction is known to be unsafe and there is no
                # head-to-head chance, so randomise across all four instead
                # of always forcing one fixed direction such as UP.
                chosen = random.choice(list(Direction))
                source = "random no-known-safe fallback"

        if ppo_direction is not None and chosen != ppo_direction:
            logger.warning(
                "Rejected PPO move %s on turn %d; using %s (%s)",
                ppo_direction.value,
                game_state.turn,
                chosen.value,
                source,
            )
        else:
            logger.info(
                "Accepted %s move %s on turn %d",
                source,
                chosen.value,
                game_state.turn,
            )
        return MoveAction(move=chosen)

    def start(self, game_state: GameState):
        game_id = game_state.game.id
        self.obs_builder.clear_cache()
        self.memories[game_id] = ExplicitMemory()
        self.slot_ids[game_id] = {game_state.you.id: 0}
        self._register_snake_slots(game_state)
        self.safety_agent.start(game_state)

    def end(self, game_state: GameState):
        game_id = game_state.game.id
        self.obs_builder.clear_cache()
        self.memories.pop(game_id, None)
        self.slot_ids.pop(game_id, None)
        self.safety_agent.end(game_state)

    def _emergency_move(self, game_state: GameState) -> MoveAction:
        """Return a valid move without depending on PPO or HungryAgent."""
        directions = [
            Direction.UP,
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
        ]
        try:
            board = game_state.board
            head = game_state.you.head
            if not self._valid_point(head, board.width, board.height):
                return MoveAction(move=random.choice(list(Direction)))

            blocked: set[tuple[int, int]] = set()
            for snake in board.snakes:
                if not self._snake_is_alive(snake):
                    continue
                for part in snake.body:
                    if self._valid_point(part, board.width, board.height):
                        blocked.add((part.x, part.y))

            for direction in directions:
                target = (head.x + direction.dx, head.y + direction.dy)
                if (
                    0 <= target[0] < board.width
                    and 0 <= target[1] < board.height
                    and target not in blocked
                ):
                    return MoveAction(move=direction)
        except Exception:
            logger.exception("Emergency move calculation failed")
        return MoveAction(move=random.choice(list(Direction)))

    def move(self, game_state: GameState) -> MoveAction:
        """Return a move while keeping the HTTP endpoint alive on bad states."""
        try:
            return self._move_impl(game_state)
        except Exception:
            game = getattr(game_state, "game", None)
            game_id = getattr(game, "id", "unknown")
            turn = getattr(game_state, "turn", "unknown")
            logger.exception(
                "Unexpected move failure on game=%s turn=%s", game_id, turn
            )
            return self._emergency_move(game_state)

    def _move_impl(self, game_state: GameState) -> MoveAction:
        game_id = game_state.game.id
        if game_id not in self.memories:
            self.start(game_state)
        memory = self.memories[game_id]
        slots = self._register_snake_slots(game_state)

        game_config = hisss.restricted_standard_config()
        game_config.h = int(game_state.board.height)
        game_config.w = int(game_state.board.width)
        game_config.num_players = int(self.config.num_players)
        game_config.min_food = int(self.config.minimum_food)
        game_config.food_spawn_chance = int(self.config.food_spawn_chance)
        game_config.all_actions_legal = True

        settings = game_state.game.ruleset.settings
        configured_view_radius = getattr(settings, "viewRadius", None)
        game_config.view_radius = (
            self.config.view_radius
            if configured_view_radius is None
            else int(configured_view_radius)
        )

        current_snakes = {game_state.you.id: game_state.you}
        current_snakes.update(
            {
                snake.id: snake
                for snake in game_state.board.snakes
                if snake.id != game_state.you.id
            }
        )
        snakes_by_slot = {
            slots[snake_id]: snake
            for snake_id, snake in current_snakes.items()
            if snake_id in slots and slots[snake_id] < game_config.num_players
        }

        alive_snakes = [
            snake
            for snake in snakes_by_slot.values()
            if self._snake_is_alive(snake)
        ]
        if len(alive_snakes) < 2:
            logger.info(
                "Only one reported snake remains on turn %d; skipping Hisss observation",
                game_state.turn,
            )
            return self._supervised_move(game_state, None)

        occupied: set[tuple[int, int]] = set()
        for snake in current_snakes.values():
            if not self._snake_is_alive(snake):
                continue
            for point in snake.body:
                if self._valid_point(point, game_config.w, game_config.h):
                    occupied.add((int(point.x), int(point.y)))

        snakes_alive: list[bool] = []
        snake_pos: dict[int, list[tuple[int, int]]] = {}
        snake_health: list[int] = []
        snake_len: list[int] = []
        for slot in range(game_config.num_players):
            snake = snakes_by_slot.get(slot)
            if snake is None or not self._snake_is_alive(snake):
                snakes_alive.append(False)
                snake_pos[slot] = []
                snake_health.append(0)
                snake_len.append(0)
                continue

            has_hidden_body = self._has_hidden_body(
                snake,
                game_config.w,
                game_config.h,
            )
            if has_hidden_body:
                placeholder = self._find_hidden_placeholder(
                    game_config.w,
                    game_config.h,
                    game_state.you.head,
                    game_config.view_radius,
                    occupied,
                )
                # Reserve the placeholder before constructing the next
                # snake, so separate hidden bodies cannot overlap.
                occupied.add(placeholder)
            else:
                placeholder = (0, 0)
            body, body_length = self._body_for_hisss(
                snake, game_state, placeholder
            )
            snakes_alive.append(True)
            snake_pos[slot] = body
            reported_health = int(snake.health) if snake.health is not None else 100
            snake_health.append(max(1, reported_health))
            snake_len.append(body_length)

        food_pos = [
            [int(food.x), int(food.y)]
            for food in game_state.board.food
            if self._valid_point(food, game_config.w, game_config.h)
        ]
        state = BattleSnakeState(
            turn=int(game_state.turn),
            snakes_alive=snakes_alive,
            snake_pos=snake_pos,
            food_pos=food_pos,
            snake_health=snake_health,
            snake_len=snake_len,
        )

        env = hisss.BattleSnakeGame(game_config)
        try:
            env.set_state(state)
            current_food_set = {
                (int(food.x), int(food.y)) for food in game_state.board.food
            }
            try:
                self.obs_builder.clear_cache()
                obs, inverse_action_map = self.obs_builder.observe(
                    env=env,
                    player=0,
                    memory=memory,
                    announced_food=current_food_set,
                    symmetry=0,
                )
            except ValueError as error:
                logger.exception("Could not build PPO observation: %s", error)
                return self._supervised_move(game_state, None)

            action_idx, _ = self.model.predict(
                np.expand_dims(obs, axis=0), deterministic=True
            )
            transformed_action = int(np.asarray(action_idx).reshape(-1)[0])
            original_action = inverse_action_map.get(transformed_action)
            action_map = {
                int(hisss.UP): Direction.UP,
                int(hisss.RIGHT): Direction.RIGHT,
                int(hisss.DOWN): Direction.DOWN,
                int(hisss.LEFT): Direction.LEFT,
            }
            ppo_direction = action_map.get(original_action)
            if ppo_direction is None:
                logger.error("PPO returned invalid action index %s", transformed_action)
                return self._supervised_move(game_state, None)
            return self._supervised_move(game_state, ppo_direction)
        finally:
            self.obs_builder.clear_cache()
            env.close()


if __name__ == "__main__":
    from battlesnake_server import start_server

    model_path = os.path.join(os.path.dirname(__file__), "ppo_model.zip")
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    agent = PPOAgent(model_path=model_path, config_path=config_path)

    # Render supplies PORT through the environment.  A positional argument is
    # still supported for local execution.
    port_value = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PORT", "10000")
    start_server(agent=agent, port=int(port_value))
