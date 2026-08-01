from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import hisss

from .config import ExperimentConfig
from .curriculum import CurriculumState, current_phase, sample_requested_lineup
from .hisss_compat import build_game_state, make_restricted_config, set_hisss_seed
from .league import LeagueManager, SnapshotRecord
from .observation import ExplicitMemory, ObservationBuilder
from .opponents import BaselineController, FrozenPolicyCache, make_baseline_agent
from .rewards import ACTION_DELTAS, RewardTracker


class BlackoutSingleLearnerEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config_dict: dict[str, Any],
        league_manifest_path: str,
        worker_index: int = 0,
    ):
        super().__init__()
        self.config = ExperimentConfig.from_dict(config_dict)
        self.worker_index = int(worker_index)
        self.league_manifest_path = str(league_manifest_path)
        self.league = LeagueManager(self.config, self.league_manifest_path)
        self.observation_builder = ObservationBuilder(self.config)
        # Keep frozen League opponents on CPU. The trainable CNN may use CUDA,
        # but loading CUDA contexts inside every environment subprocess wastes
        # VRAM and can make Windows spawn-based vector environments unstable.
        self.policy_cache = FrozenPolicyCache(
            self.config.frozen_policy_cache_size, "cpu"
        )
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            # Hisss uses -1.0 for cells outside the centered game board.
            low=-1.0,
            high=1.0,
            shape=(
                self.config.input_channels,
                self.config.observation_size,
                self.config.observation_size,
            ),
            dtype=np.float32,
        )
        self.env = None
        self.rng = np.random.default_rng(self.config.base_seed + self.worker_index)
        self.global_step = 0
        self.curriculum_state = CurriculumState()
        self.game_id = ""
        self.episode_seed = 0
        self.learner_seat = 0
        self.symmetry = 0
        self.requested_lineup = "RRR"
        self.actual_lineup = "RRR"
        self.seat_codes: dict[int, str] = {}
        self.seat_snapshots: dict[int, SnapshotRecord] = {}
        self.baseline_controllers: dict[int, BaselineController] = {}
        self.memories: dict[int, ExplicitMemory] = {}
        self.food_spawn_turns: dict[tuple[int, int], int] = {}
        self.announced_food: set[tuple[int, int]] = set()
        self.reward_tracker = RewardTracker(self.config)
        self.learner_inverse_action = {index: index for index in range(4)}
        self.action_counts = [0, 0, 0, 0]
        self.started_at = 0.0
        self._episode_finished = False

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    def set_curriculum_state(self, state: dict[str, Any]) -> None:
        self.curriculum_state = CurriculumState.from_dict(state)

    def reload_league(self) -> None:
        self.league.reload()

    def _food_set(self) -> set[tuple[int, int]]:
        return {tuple(map(int, point)) for point in self.env.food_pos()}

    def _learner_observation(self) -> np.ndarray:
        observation, inverse = self.observation_builder.observe(
            self.env,
            self.learner_seat,
            self.memories[self.learner_seat],
            self.announced_food,
            self.symmetry,
        )
        self.learner_inverse_action = inverse
        return observation

    def _configure_opponents(self, requested: str, unique_ppo: bool = False) -> None:
        actual, assigned = self.league.resolve_training_lineup(
            requested, self.rng, unique_required=unique_ppo
        )
        pairs = list(zip(list(actual), assigned))
        self.rng.shuffle(pairs)
        opponent_seats = [seat for seat in range(self.config.num_players) if seat != self.learner_seat]
        self.seat_codes = {self.learner_seat: "L"}
        self.seat_snapshots = {}
        self.baseline_controllers = {}
        for seat, (code, snapshot) in zip(opponent_seats, pairs):
            self.seat_codes[seat] = code
            if code == "P" and snapshot is not None:
                self.seat_snapshots[seat] = snapshot
            else:
                agent = make_baseline_agent(code)
                self.baseline_controllers[seat] = BaselineController(code, seat, agent)
        self.actual_lineup = "".join(self.seat_codes[seat] for seat in opponent_seats)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if self.env is not None:
            self._finish_agents()
            self.env.close()
        if seed is None:
            seed = int(self.rng.integers(1, 2**31 - 1))
        self.episode_seed = int(seed)
        self.rng = np.random.default_rng(self.episode_seed)
        np.random.seed(self.episode_seed & 0xFFFFFFFF)
        set_hisss_seed(self.episode_seed)
        self.env = hisss.BattleSnakeGame(make_restricted_config(self.config))
        self.observation_builder.clear_cache()
        self.league.reload()
        # Deterministic for Gymnasium's same-seed contract while remaining
        # effectively unique across workers, training blocks, and episodes.
        self.game_id = (
            f"train-{self.worker_index}-{self.global_step}-{self.episode_seed}"
        )
        self.learner_seat = int(
            options.get("learner_seat", self.rng.integers(self.config.num_players))
        )
        self.symmetry = int(
            options.get(
                "symmetry",
                self.rng.integers(8) if self.config.random_symmetry else 0,
            )
        )
        self.requested_lineup = str(
            options.get(
                "lineup",
                sample_requested_lineup(
                    self.config, self.global_step, self.curriculum_state, self.rng
                ),
            )
        )
        if len(self.requested_lineup) != self.config.num_players - 1:
            raise ValueError(f"Lineup must contain three opponents: {self.requested_lineup}")
        self._configure_opponents(
            self.requested_lineup, unique_ppo=bool(options.get("unique_ppo", False))
        )
        self.memories = {
            seat: ExplicitMemory()
            for seat, code in self.seat_codes.items()
            if code in {"L", "P"}
        }
        self.announced_food = self._food_set()
        self.food_spawn_turns = {point: 0 for point in self.announced_food}
        for controller in self.baseline_controllers.values():
            controller.start(
                self.env, self.game_id, self.announced_food, self.food_spawn_turns
            )
        self.reward_tracker = RewardTracker(self.config)
        self.action_counts = [0, 0, 0, 0]
        self.started_at = time.perf_counter()
        self._episode_finished = False
        observation = self._learner_observation()
        info = {
            "game_id": self.game_id,
            "seed": self.episode_seed,
            "learner_seat": self.learner_seat,
            "requested_lineup": self.requested_lineup,
            "actual_lineup": self.actual_lineup,
            "phase": current_phase(
                self.config, self.global_step, self.curriculum_state
            )["name"],
        }
        return observation, info

    def _frozen_action(self, seat: int) -> int:
        observation, inverse = self.observation_builder.observe(
            self.env,
            seat,
            self.memories[seat],
            self.announced_food,
            self.symmetry,
        )
        transformed = self.policy_cache.predict(
            self.seat_snapshots[seat].model_path, observation
        )
        return int(inverse[transformed])

    def _infer_death_cause(
        self,
        before_state,
        learner_action: int,
        before_alive: set[int],
        after_alive: set[int],
    ) -> str:
        if self.learner_seat in after_alive:
            return ""
        body = [tuple(map(int, point)) for point in before_state.snake_pos[self.learner_seat]]
        if not body:
            return "unknown"
        delta = ACTION_DELTAS[learner_action]
        target = (body[0][0] + delta[0], body[0][1] + delta[1])
        if not (0 <= target[0] < self.config.board_width and 0 <= target[1] < self.config.board_height):
            return "wall-collision"
        if before_state.snake_health[self.learner_seat] <= 1 and target not in {
            tuple(map(int, point)) for point in before_state.food_pos
        }:
            return "out-of-health"
        if target in set(body[1:-1]):
            return "snake-self-collision"
        for enemy in before_alive - {self.learner_seat}:
            enemy_body = {
                tuple(map(int, point)) for point in before_state.snake_pos[enemy][:-1]
            }
            if target in enemy_body:
                return "snake-collision"
        return "head-collision-or-unknown"

    @staticmethod
    def _rank_after_death(before_count: int, after_count: int) -> float:
        simultaneous_deaths = max(1, before_count - after_count)
        return after_count + (simultaneous_deaths + 1) / 2.0

    def _finish_agents(self) -> None:
        if self._episode_finished:
            return
        for controller in self.baseline_controllers.values():
            try:
                controller.end(
                    self.env,
                    self.game_id,
                    self.announced_food,
                    self.food_spawn_turns,
                )
            except Exception:
                # Episode metrics and the simulator result must survive a baseline cleanup issue.
                pass
        self._episode_finished = True

    def step(self, action):
        if self.env is None or self.env.is_terminal():
            raise RuntimeError("step() called before reset() or after terminal state")
        transformed_learner_action = int(np.asarray(action).item())
        learner_action = int(self.learner_inverse_action[transformed_learner_action])
        self.action_counts[learner_action] += 1
        before_state = self.env.get_state()
        before_alive = set(self.env.players_alive())
        food_before = self._food_set()

        actions: list[int] = []
        for seat in self.env.players_at_turn():
            if seat == self.learner_seat:
                world_action = learner_action
            elif self.seat_codes[seat] == "P":
                world_action = self._frozen_action(seat)
            else:
                world_action = self.baseline_controllers[seat].act(
                    self.env,
                    self.game_id,
                    self.announced_food,
                    self.food_spawn_turns,
                )
            actions.append(int(world_action))

        self.env.step(actions=tuple(actions))
        self.observation_builder.clear_cache()
        after_alive = set(self.env.players_alive())
        food_after = self._food_set()
        self.announced_food = food_after - food_before
        for coordinate in self.announced_food:
            self.food_spawn_turns[coordinate] = int(self.env.turns_played)
        for coordinate in list(self.food_spawn_turns):
            if coordinate not in food_after:
                del self.food_spawn_turns[coordinate]

        learner_alive = self.learner_seat in after_alive
        win = learner_alive and len(after_alive) == 1
        death = not learner_alive
        terminated = bool(win or death or self.env.is_terminal())
        truncated = bool(
            not terminated and self.env.turns_played >= self.config.max_turns
        )
        breakdown = self.reward_tracker.step(
            before_state,
            before_alive,
            after_alive,
            self.learner_seat,
            learner_action,
            win,
            death,
        )

        if terminated or truncated:
            if win:
                rank = 1.0
                result = "win"
            elif death:
                rank = self._rank_after_death(len(before_alive), len(after_alive))
                result = "all_die" if not after_alive else "loss"
            else:
                rank = (len(after_alive) + 1) / 2.0
                result = "truncated"
            death_cause = self._infer_death_cause(
                before_state, learner_action, before_alive, after_alive
            )
            final_state = self.env.get_state()
            reward_summary = self.reward_tracker.summary()
            opponent_ids = [
                self.seat_snapshots[seat].snapshot_id
                for seat in sorted(self.seat_snapshots)
            ]
            episode_summary = {
                "game_id": self.game_id,
                "seed": self.episode_seed,
                "phase": current_phase(
                    self.config, self.global_step, self.curriculum_state
                )["name"],
                "requested_lineup": self.requested_lineup,
                "actual_lineup": self.actual_lineup,
                "learner_seat": self.learner_seat,
                "opponent_ids": "|".join(opponent_ids),
                "result": result,
                "win": int(win),
                "loss": int(death and not win),
                "all_die": int(not after_alive),
                "rank": rank,
                "turns": int(self.env.turns_played),
                **reward_summary,
                "final_length": int(final_state.snake_len[self.learner_seat]),
                "final_health": int(final_state.snake_health[self.learner_seat]),
                "death_cause": death_cause,
                "truncated": int(truncated),
                "actions_up": self.action_counts[0],
                "actions_right": self.action_counts[1],
                "actions_down": self.action_counts[2],
                "actions_left": self.action_counts[3],
                "duration_seconds": time.perf_counter() - self.started_at,
            }
            self._finish_agents()
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"episode_summary": episode_summary, **episode_summary}
        else:
            observation = self._learner_observation()
            info = {
                "game_id": self.game_id,
                "reward_components": breakdown.to_dict(),
                "actual_lineup": self.actual_lineup,
            }
        return observation, float(breakdown.total), terminated, truncated, info

    def render(self):
        return self.env.get_str_repr() if self.env is not None else ""

    def close(self):
        self._finish_agents() if self.env is not None else None
        if self.env is not None:
            self.env.close()
            self.env = None
        self.policy_cache.clear()


def make_env_factory(
    config_dict: dict[str, Any], league_manifest_path: str, worker_index: int
):
    def factory():
        return BlackoutSingleLearnerEnv(
            config_dict=config_dict,
            league_manifest_path=league_manifest_path,
            worker_index=worker_index,
        )

    return factory
