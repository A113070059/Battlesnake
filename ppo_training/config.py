from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _default_curriculum() -> list[dict[str, Any]]:
    return [
        {"name": "basic_random", "start": 0, "mix": {"RRR": 1.00}},
        {
            "name": "random_hungry",
            "start": 1_000_000,
            "mix": {"RRR": 0.60, "HRR": 0.30, "HHR": 0.10},
        },
        {
            "name": "hungry_main",
            "start": 5_000_000,
            "mix": {"RRR": 0.10, "HRR": 0.30, "HHR": 0.40, "HHH": 0.20},
        },
        {
            "name": "initial_league",
            "start": 15_000_000,
            "mix": {
                "RRR": 0.10,
                "HRR": 0.15,
                "HHR": 0.20,
                "HHH": 0.15,
                "PHR": 0.25,
                "PPH": 0.15,
            },
        },
        {
            "name": "mature_league",
            "start": 30_000_000,
            "mix": {
                "RRR": 0.05,
                "HHR": 0.10,
                "HHH": 0.10,
                "PHR": 0.30,
                "PPH": 0.25,
                "PPP": 0.20,
            },
        },
    ]


@dataclass(slots=True)
class ExperimentConfig:
    """Single source of truth for every user-adjustable training setting."""

    schema_version: int = SCHEMA_VERSION
    run_id: str = "blackout_ppo_seed42"
    base_seed: int = 42
    resume_mode: str = "auto"  # new | auto | required
    resume_source: str | None = None
    output_root: str = "/kaggle/working/blackout_ppo"

    # Total and per-session budgets
    total_target_steps: int = 50_000_000
    session_target_steps: int | None = 5_000_000
    session_step_rounding: str = "floor"  # floor | ceil
    session_max_hours: float = 8.5
    save_reserve_minutes: int = 20
    smoke_test_only: bool = True
    smoke_test_steps: int = 100_000

    # CPU and vector environments
    device: str = "cpu"
    auto_select_n_envs: bool = True
    n_envs: int = 2
    n_envs_candidates: tuple[int, ...] = (1, 2, 4, 8, 16)
    n_envs_fallback: int = 2
    allow_benchmark_fallback: bool = False
    subprocess_start_method: str = "forkserver"
    omp_threads: int = 1
    mkl_threads: int = 1
    openblas_threads: int = 1
    torch_threads: int = 1

    # Benchmark
    run_benchmark: bool = True
    benchmark_warmup_steps: int = 2_048
    benchmark_measure_steps: int = 8_192
    benchmark_repeats: int = 1
    benchmark_max_ram_fraction: float = 0.85
    benchmark_max_cv: float = 0.15
    benchmark_tie_tolerance: float = 0.02

    # PPO rollout/update
    rollout_buffer_target: int = 2_048
    auto_scale_n_steps: bool = True
    n_steps: int = 1_024
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 0.00025
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    entropy_start: float = 0.0100
    entropy_end: float = 0.0010
    entropy_decay_steps: int = 30_000_000
    vf_coef: float = 0.50
    max_grad_norm: float = 0.50
    target_kl: float = 0.02
    normalize_advantage: bool = True
    train_updates_per_block: int = 64

    # Hisss / game rules
    board_width: int = 15
    board_height: int = 15
    num_players: int = 4
    view_radius: int = 5
    minimum_food: int = 1
    food_spawn_chance: int = 15
    max_turns: int = 1_000
    truncation_stop_fraction: float = 0.001
    random_symmetry: bool = True

    # Observation and explicit memory
    hisss_channels: int = 17
    food_memory_horizon: int = 50
    enemy_memory_horizon: int = 6
    input_channels: int = 19
    observation_size: int = 29

    # CNN
    conv_channels: tuple[int, int, int] = (32, 64, 64)
    kernel_size: int = 3
    padding: int = 1
    pool_size: int = 2
    hidden_size: int = 256
    activation: str = "relu"
    expected_parameter_count: int = 865_285

    # Reward
    reward_win: float = 1.00
    reward_death: float = -1.00
    reward_survival_per_turn: float = 0.0005
    reward_survival_turn_cap: int = 200
    reward_elimination: float = 0.10
    reward_elimination_cap: float = 0.30
    reward_food_healthy: float = 0.005
    reward_food_low_health: float = 0.01
    reward_food_health_threshold: int = 40
    reward_food_cap: float = 0.15

    # Curriculum gates and League
    curriculum: list[dict[str, Any]] = field(default_factory=_default_curriculum)
    phase2_gate_step: int = 5_000_000
    phase2_gate_suite: str = "RRR"
    phase2_gate_games: int = 500
    phase2_gate_win_rate: float = 0.90
    phase3_gate_step: int = 15_000_000
    phase3_gate_suite: str = "HHR"
    phase3_gate_games: int = 500
    phase3_gate_win_rate: float = 0.40
    snapshot_interval_steps: int = 2_000_000
    league_max_models: int = 10
    league_best_slots: int = 2
    league_recent_slots: int = 4
    league_diverse_slots: int = 4
    league_category_weights: dict[str, float] = field(
        default_factory=lambda: {"best": 0.40, "recent": 0.40, "diverse": 0.20}
    )
    catastrophic_rrr_floor: float = 0.50
    catastrophic_score_gap: float = 0.30
    diverse_score_gap: float = 0.10
    frozen_policy_cache_size: int = 4

    # Evaluation
    quick_eval_interval_steps: int = 500_000
    quick_eval_games_per_suite: int = 100
    full_eval_interval_steps: int = 2_000_000
    full_eval_games_per_suite: int = 500
    session_end_eval_games_per_suite: int = 100
    final_eval_games_per_suite: int = 1_000
    sanity_eval_games: int = 20
    final_fixed_seed_fraction: float = 0.50
    promotion_overall_delta: float = 0.02
    promotion_rrr_max_regression: float = 0.02
    promotion_hhh_max_regression: float = 0.03
    final_rrr_win_rate_target: float = 0.99
    demo_rrr_games: int = 10

    # Logging and reports
    progress_update_seconds: float = 5.0
    progress_print_every_steps: int = 50_000
    metric_flush_seconds: float = 30.0
    system_metric_seconds: float = 60.0
    save_every_steps: int = 131_072
    tensorboard_enabled: bool = True
    keep_per_game_training_metrics: bool = True
    csv_compression: str = "gzip"
    representative_replay_seeds: int = 20

    # Runtime-resolved fields
    resolved_n_envs: int | None = None
    resolved_n_steps: int | None = None
    session_id: int = 1

    def validate(self) -> None:
        if self.resume_mode not in {"new", "auto", "required"}:
            raise ValueError("resume_mode must be new, auto, or required")
        if self.session_step_rounding not in {"floor", "ceil"}:
            raise ValueError("session_step_rounding must be floor or ceil")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        if self.num_players != 4:
            raise ValueError("The approved curriculum assumes exactly four snakes")
        if self.input_channels != self.hisss_channels + 2:
            raise ValueError("input_channels must be Hisss channels + food/enemy memory")
        if self.board_width != self.board_height or self.board_width % 2 == 0:
            raise ValueError("Centered Hisss observations need an odd square board")
        if self.observation_size != 2 * self.board_width - 1:
            raise ValueError("observation_size must equal 2*board_width-1")
        if abs(sum(self.league_category_weights.values()) - 1.0) > 1e-9:
            raise ValueError("league category weights must sum to one")
        for phase in self.curriculum:
            if len(phase["mix"]) == 0 or abs(sum(phase["mix"].values()) - 1.0) > 1e-9:
                raise ValueError(f"Curriculum mix does not sum to one: {phase}")
        n_envs = self.effective_n_envs
        n_steps = self.effective_n_steps
        buffer_size = n_envs * n_steps
        if buffer_size % self.batch_size != 0:
            raise ValueError("n_steps*n_envs must be divisible by batch_size")
        if self.rollout_buffer_target <= 0 or self.batch_size <= 1:
            raise ValueError("Invalid PPO buffer/batch size")
        if self.session_target_steps is not None and self.session_target_steps <= 0:
            raise ValueError("session_target_steps must be positive or None")

    @property
    def effective_n_envs(self) -> int:
        return int(self.resolved_n_envs or self.n_envs)

    @property
    def effective_n_steps(self) -> int:
        if self.resolved_n_steps is not None:
            return int(self.resolved_n_steps)
        if self.auto_scale_n_steps:
            if self.rollout_buffer_target % self.effective_n_envs != 0:
                raise ValueError("rollout_buffer_target must be divisible by n_envs")
            return self.rollout_buffer_target // self.effective_n_envs
        return int(self.n_steps)

    @property
    def rollout_buffer_size(self) -> int:
        return self.effective_n_envs * self.effective_n_steps

    @property
    def train_block_steps(self) -> int:
        return self.train_updates_per_block * self.rollout_buffer_size

    def with_resolved_parallelism(self, n_envs: int) -> "ExperimentConfig":
        if n_envs <= 0:
            raise ValueError("n_envs must be positive")
        if self.auto_scale_n_steps:
            if self.rollout_buffer_target % n_envs != 0:
                raise ValueError("rollout_buffer_target is not divisible by selected n_envs")
            n_steps = self.rollout_buffer_target // n_envs
        else:
            n_steps = self.n_steps
        result = replace(self, resolved_n_envs=n_envs, resolved_n_steps=n_steps)
        result.validate()
        return result

    def rounded_session_steps(self) -> int | None:
        if self.session_target_steps is None:
            return None
        buffer_size = self.rollout_buffer_size
        ratio = self.session_target_steps / buffer_size
        updates = math.floor(ratio) if self.session_step_rounding == "floor" else math.ceil(ratio)
        return max(1, updates) * buffer_size

    def entropy_coefficient(self, global_step: int) -> float:
        fraction = max(0.0, 1.0 - global_step / self.entropy_decay_steps)
        return self.entropy_end + (self.entropy_start - self.entropy_end) * fraction

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        data = dict(value)
        for key in ("n_envs_candidates", "conv_channels"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        result = cls(**data)
        result.validate()
        return result

    def semantic_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        for key in (
            "resume_mode",
            "resume_source",
            "session_target_steps",
            "session_step_rounding",
            "session_max_hours",
            "save_reserve_minutes",
            "smoke_test_only",
            "smoke_test_steps",
            "progress_update_seconds",
            "progress_print_every_steps",
            "metric_flush_seconds",
            "system_metric_seconds",
            "session_id",
        ):
            data.pop(key, None)
        return data

    def semantic_hash(self) -> str:
        payload = json.dumps(self.semantic_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".tmp")
        temp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
