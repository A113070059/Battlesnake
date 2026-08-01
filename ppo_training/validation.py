from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any

import numpy as np

from .hisss_compat import get_observations_compat, make_restricted_config, set_hisss_seed
from .observation import ExplicitMemory, ObservationBuilder
from .rewards import RewardTracker


def run_hisss_regression(config) -> dict[str, Any]:
    """Validate Hisss encoding and the eliminated-player compatibility fix."""
    import hisss

    set_hisss_seed(config.base_seed)
    env = hisss.BattleSnakeGame(make_restricted_config(config))
    try:
        upstream, _, _ = env.get_obs(symmetry=0)
        compatible, action_map, inverse_map = get_observations_compat(env, symmetry=0)
        expected_shape = (
            config.num_players,
            config.observation_size,
            config.observation_size,
            config.hisss_channels,
        )
        if upstream.shape != expected_shape or compatible.shape != expected_shape:
            raise AssertionError(
                f"Unexpected Hisss observation shape: {upstream.shape}, {compatible.shape}"
            )
        if not np.array_equal(upstream, compatible):
            raise AssertionError("Compatibility encoder differs before any elimination")
        if any(inverse_map[action_map[action]] != action for action in range(4)):
            raise AssertionError("Hisss action transform is not invertible")
        for symmetry in range(8):
            upstream_symmetry, _, _ = env.get_obs(symmetry=symmetry)
            compatible_symmetry, transformed, inverse = get_observations_compat(
                env, symmetry=symmetry
            )
            if not np.array_equal(upstream_symmetry, compatible_symmetry):
                raise AssertionError(f"Symmetry {symmetry} differs from upstream Hisss")
            if any(inverse[transformed[action]] != action for action in range(4)):
                raise AssertionError(f"Symmetry {symmetry} action inverse is incorrect")

        builder = ObservationBuilder(config)
        observation, _ = builder.observe(
            env,
            player=0,
            memory=ExplicitMemory(),
            announced_food=set(),
            symmetry=0,
        )
        if observation.shape != (
            config.input_channels,
            config.observation_size,
            config.observation_size,
        ):
            raise AssertionError(f"Unexpected learner observation shape: {observation.shape}")
        if observation.dtype != np.float32 or not np.isfinite(observation).all():
            raise AssertionError("Observation must be finite float32")

        state = env.get_state()
        state.snakes_alive[0] = False
        state.snake_pos[0] = []
        env.set_state(state)
        alive = list(env.players_at_turn())
        after_elimination, _, _ = get_observations_compat(env, symmetry=0)
        if alive != [1, 2, 3] or after_elimination.shape[0] != 3:
            raise AssertionError(
                "Eliminated-player regression failed: compact observation indices are wrong"
            )
        for symmetry in range(8):
            compact, transformed, inverse = get_observations_compat(env, symmetry=symmetry)
            if compact.shape != (3, *expected_shape[1:]):
                raise AssertionError(f"Elimination symmetry {symmetry} has shape {compact.shape}")
            if any(inverse[transformed[action]] != action for action in range(4)):
                raise AssertionError(f"Elimination symmetry {symmetry} action inverse failed")
        return {
            "status": "ok",
            "upstream_shape_reported": tuple(env.get_obs_shape()),
            "actual_hisss_shape": expected_shape,
            "learner_shape": tuple(observation.shape),
            "alive_after_test_elimination": alive,
        }
    finally:
        env.close()


def _state(health: int = 100, food: set[tuple[int, int]] | None = None):
    return SimpleNamespace(
        snake_pos={0: [(5, 5)], 1: [(10, 10)], 2: [(11, 10)], 3: [(12, 10)]},
        food_pos=list(food or set()),
        snake_health=[health, 100, 100, 100],
    )


def run_reward_unit_tests(config) -> dict[str, Any]:
    alive = {0, 1, 2, 3}

    survival = RewardTracker(config)
    for _ in range(config.reward_survival_turn_cap + 10):
        survival.step(_state(), alive, alive, 0, 0, win=False, death=False)
    expected_survival = config.reward_survival_per_turn * config.reward_survival_turn_cap
    if not np.isclose(survival.cumulative.survival, expected_survival):
        raise AssertionError("Survival reward cap is incorrect")

    healthy = RewardTracker(config)
    healthy.step(_state(health=80, food={(5, 6)}), alive, alive, 0, 0, False, False)
    low = RewardTracker(config)
    low.step(_state(health=40, food={(5, 6)}), alive, alive, 0, 0, False, False)
    if not np.isclose(healthy.cumulative.food, config.reward_food_healthy):
        raise AssertionError("Healthy food reward is incorrect")
    if not np.isclose(low.cumulative.food, config.reward_food_low_health):
        raise AssertionError("Low-health food reward is incorrect")

    food_cap = RewardTracker(config)
    for _ in range(100):
        food_cap.step(_state(health=40, food={(5, 6)}), alive, alive, 0, 0, False, False)
    if not np.isclose(food_cap.cumulative.food, config.reward_food_cap):
        raise AssertionError("Food reward cap is incorrect")

    eliminations = RewardTracker(config)
    eliminations.step(_state(), alive, {0}, 0, 0, win=True, death=False)
    if not np.isclose(eliminations.cumulative.elimination, config.reward_elimination_cap):
        raise AssertionError("Elimination reward cap is incorrect")
    if not np.isclose(eliminations.cumulative.terminal, config.reward_win):
        raise AssertionError("Win reward is incorrect")

    death = RewardTracker(config)
    death.step(
        _state(food={(5, 6)}), alive, {1, 2, 3}, 0, 0, win=False, death=True
    )
    if not np.isclose(death.cumulative.terminal, config.reward_death):
        raise AssertionError("Death reward is incorrect")
    if death.cumulative.food != 0:
        raise AssertionError("A snake that dies on a food square must not receive food reward")

    return {
        "status": "ok",
        "survival_cap": survival.cumulative.survival,
        "food_healthy": healthy.cumulative.food,
        "food_low_health": low.cumulative.food,
        "food_cap": food_cap.cumulative.food,
        "elimination_cap": eliminations.cumulative.elimination,
        "win": eliminations.cumulative.terminal,
        "death": death.cumulative.terminal,
    }


def run_environment_validation(config, league_manifest_path: str | Path) -> dict[str, Any]:
    from gymnasium.utils.env_checker import check_env

    from .env import BlackoutSingleLearnerEnv

    env = BlackoutSingleLearnerEnv(
        config.to_dict(), str(league_manifest_path), worker_index=99_999
    )
    try:
        observation, _ = env.reset(seed=config.base_seed)
        if not env.observation_space.contains(observation):
            raise AssertionError(
                "reset() observation is outside observation_space: "
                f"shape={observation.shape}, dtype={observation.dtype}, "
                f"min={float(np.min(observation))}, max={float(np.max(observation))}, "
                f"expected_shape={env.observation_space.shape}, "
                f"expected_dtype={env.observation_space.dtype}, "
                f"low={float(np.min(env.observation_space.low))}, "
                f"high={float(np.max(env.observation_space.high))}"
            )
        check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=config.base_seed)
        episodes = 0
        steps = 0
        while steps < 128:
            observation, reward, terminated, truncated, _ = env.step(
                int(env.action_space.sample())
            )
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                raise AssertionError("Non-finite observation or reward")
            steps += 1
            if terminated or truncated:
                episodes += 1
                observation, _ = env.reset(seed=config.base_seed + episodes + 1)
        return {"status": "ok", "steps": steps, "episodes_completed": episodes}
    finally:
        env.close()


def run_model_smoke_test(
    config,
    league_manifest_path: str | Path,
    tensorboard_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the real CNN/PPO, learn two rollouts, and time deterministic inference."""
    from dataclasses import replace

    from .network import build_ppo, count_trainable_parameters, make_vector_env

    smoke_config = replace(
        config,
        resolved_n_envs=1,
        resolved_n_steps=config.rollout_buffer_target,
        tensorboard_enabled=False,
    )
    smoke_config.validate()
    vector_env = make_vector_env(smoke_config, str(league_manifest_path))
    try:
        model = build_ppo(smoke_config, vector_env, tensorboard_dir)
        parameter_count = count_trainable_parameters(model)
        if parameter_count != config.expected_parameter_count:
            raise AssertionError(
                f"CNN parameter count {parameter_count:,} != expected "
                f"{config.expected_parameter_count:,}"
            )
        model.learn(
            total_timesteps=2 * smoke_config.rollout_buffer_size,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        observation = vector_env.reset()
        samples: list[float] = []
        for _ in range(100):
            started = time.perf_counter()
            model.predict(observation, deterministic=True)
            samples.append((time.perf_counter() - started) * 1_000)
        p95_ms = float(np.percentile(samples, 95))
        if p95_ms >= 500:
            raise AssertionError(f"Deterministic inference p95 {p95_ms:.2f} ms exceeds 500 ms")
        return {
            "status": "ok",
            "learned_steps": int(model.num_timesteps),
            "parameter_count": parameter_count,
            "inference_p50_ms": float(np.percentile(samples, 50)),
            "inference_p95_ms": p95_ms,
            "inference_p99_ms": float(np.percentile(samples, 99)),
        }
    finally:
        vector_env.close()


def run_all_preflight(config, league_manifest_path: str | Path) -> dict[str, Any]:
    config.validate()
    return {
        "config": {"status": "ok", "semantic_hash": config.semantic_hash()},
        "hisss": run_hisss_regression(config),
        "rewards": run_reward_unit_tests(config),
        "environment": run_environment_validation(config, league_manifest_path),
    }
