from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hisss_compat import (
    apply_spatial_symmetry,
    get_observations_compat,
    symmetry_action_maps,
)


Coordinate = tuple[int, int]


@dataclass(slots=True)
class ExplicitMemory:
    food_last_seen: dict[Coordinate, int] = field(default_factory=dict)
    enemy_last_seen: dict[Coordinate, int] = field(default_factory=dict)
    last_updated_turn: int = -1

    def reset(self) -> None:
        self.food_last_seen.clear()
        self.enemy_last_seen.clear()
        self.last_updated_turn = -1


class ObservationBuilder:
    """Create the 19x29x29 numeric observation without hidden-information leaks."""

    def __init__(self, config):
        self.config = config
        self._cached_env_id: int | None = None
        self._cached_turn: int | None = None
        self._cached_alive: list[int] = []
        self._cached_raw: np.ndarray | None = None

    def clear_cache(self) -> None:
        self._cached_env_id = None
        self._cached_turn = None
        self._cached_alive = []
        self._cached_raw = None

    def _raw_observations(self, env) -> tuple[np.ndarray, list[int]]:
        cache_key = id(env)
        if (
            self._cached_raw is None
            or self._cached_env_id != cache_key
            or self._cached_turn != env.turns_played
        ):
            raw, _, _ = get_observations_compat(env, symmetry=0)
            self._cached_env_id = cache_key
            self._cached_turn = int(env.turns_played)
            self._cached_alive = list(env.players_at_turn())
            self._cached_raw = raw
        return self._cached_raw, self._cached_alive

    def _world_to_centered(self, head: Coordinate, coordinate: Coordinate) -> tuple[int, int]:
        center = self.config.board_width - 1
        return center + coordinate[0] - head[0], center + coordinate[1] - head[1]

    def _in_vision(self, head: Coordinate, coordinate: Coordinate) -> bool:
        return (
            abs(coordinate[0] - head[0]) + abs(coordinate[1] - head[1])
            <= self.config.view_radius
        )

    def _update_memory(
        self,
        env,
        player: int,
        memory: ExplicitMemory,
        announced_food: set[Coordinate],
    ) -> None:
        turn = int(env.turns_played)
        if memory.last_updated_turn == turn:
            return
        state = env.get_state()
        own_body = [tuple(map(int, point)) for point in state.snake_pos[player]]
        if not own_body:
            return
        head = own_body[0]
        actual_food = {tuple(map(int, point)) for point in state.food_pos}
        locally_visible_food = {point for point in actual_food if self._in_vision(head, point)}
        observed_food = locally_visible_food | set(announced_food)

        for coordinate in list(memory.food_last_seen):
            if self._in_vision(head, coordinate) and coordinate not in actual_food:
                del memory.food_last_seen[coordinate]
        for coordinate in observed_food:
            memory.food_last_seen[coordinate] = turn
        for coordinate, last_seen in list(memory.food_last_seen.items()):
            if turn - last_seen >= self.config.food_memory_horizon:
                del memory.food_last_seen[coordinate]

        visible_enemy: set[Coordinate] = set()
        for enemy in env.players_alive():
            if enemy == player:
                continue
            for point in state.snake_pos[enemy]:
                coordinate = tuple(map(int, point))
                if self._in_vision(head, coordinate):
                    visible_enemy.add(coordinate)
        for coordinate in list(memory.enemy_last_seen):
            if self._in_vision(head, coordinate) and coordinate not in visible_enemy:
                del memory.enemy_last_seen[coordinate]
        for coordinate in visible_enemy:
            memory.enemy_last_seen[coordinate] = turn
        for coordinate, last_seen in list(memory.enemy_last_seen.items()):
            if turn - last_seen >= self.config.enemy_memory_horizon:
                del memory.enemy_last_seen[coordinate]

        memory.last_updated_turn = turn

    def observe(
        self,
        env,
        player: int,
        memory: ExplicitMemory,
        announced_food: set[Coordinate],
        symmetry: int = 0,
    ) -> tuple[np.ndarray, dict[int, int]]:
        if not env.is_player_at_turn(player):
            raise ValueError(f"Cannot observe dead player {player}")
        raw_batch, alive = self._raw_observations(env)
        local_index = alive.index(player)
        raw = raw_batch[local_index].copy()
        if raw.shape != (
            self.config.observation_size,
            self.config.observation_size,
            self.config.hisss_channels,
        ):
            raise ValueError(f"Unexpected Hisss observation shape: {raw.shape}")

        state = env.get_state()
        own_body = [tuple(map(int, point)) for point in state.snake_pos[player]]
        if not own_body:
            raise ValueError("Alive player has no head")
        head = own_body[0]

        # Food is globally visible on its spawn turn. Hisss 1.2 masks current
        # food by radius, so inject only the wrapper-tracked new-food set.
        food_layer = env.layer_explanation["current_food"]
        for coordinate in announced_food:
            first, second = self._world_to_centered(head, coordinate)
            if 0 <= first < raw.shape[0] and 0 <= second < raw.shape[1]:
                raw[first, second, food_layer] = 1.0

        self._update_memory(env, player, memory, announced_food)
        food_channel = np.zeros(raw.shape[:2], dtype=np.float32)
        enemy_channel = np.zeros(raw.shape[:2], dtype=np.float32)
        turn = int(env.turns_played)

        for coordinate, last_seen in memory.food_last_seen.items():
            first, second = self._world_to_centered(head, coordinate)
            if 0 <= first < raw.shape[0] and 0 <= second < raw.shape[1]:
                age = turn - last_seen
                food_channel[first, second] = max(
                    0.0, 1.0 - age / self.config.food_memory_horizon
                )
        for coordinate, last_seen in memory.enemy_last_seen.items():
            first, second = self._world_to_centered(head, coordinate)
            if 0 <= first < raw.shape[0] and 0 <= second < raw.shape[1]:
                age = turn - last_seen
                enemy_channel[first, second] = max(
                    0.0, 1.0 - age / self.config.enemy_memory_horizon
                )

        combined = np.concatenate(
            (raw, food_channel[:, :, None], enemy_channel[:, :, None]), axis=-1
        )
        if symmetry:
            combined = apply_spatial_symmetry(combined, symmetry)
        _, inverse_action = symmetry_action_maps(symmetry)
        chw = np.transpose(combined, (2, 0, 1)).astype(np.float32, copy=False)
        if chw.shape != (
            self.config.input_channels,
            self.config.observation_size,
            self.config.observation_size,
        ):
            raise ValueError(f"Unexpected combined observation shape: {chw.shape}")
        if not np.isfinite(chw).all():
            raise ValueError("Observation contains NaN or Inf")
        return np.ascontiguousarray(chw), inverse_action

