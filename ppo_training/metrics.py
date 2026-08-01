from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import time
from typing import Any, Iterable


TRAIN_UPDATE_FIELDS = [
    "run_id", "session_id", "global_step", "phase", "n_envs", "n_steps",
    "buffer_size", "wall_seconds", "steps_per_second", "learning_rate",
    "entropy_coefficient", "policy_entropy", "approx_kl", "clip_fraction",
    "policy_gradient_loss", "value_loss", "total_loss", "explained_variance",
    "gradient_norm", "n_updates", "episode_reward_mean", "episode_reward_std",
    "episode_reward_min", "episode_reward_max", "update_epochs",
    "win_rate_100", "win_rate_1000", "average_rank_100", "average_turns_100",
    "median_turns_100", "wins_100", "losses_100", "all_die_100",
    "reward_terminal_mean", "reward_survival_mean", "reward_elimination_mean",
    "reward_food_mean", "reward_terminal_sum", "reward_survival_sum",
    "reward_elimination_sum", "reward_food_sum", "food_mean", "final_length_mean",
    "final_health_mean", "wall_death_rate", "self_death_rate", "body_death_rate",
    "head_or_unknown_death_rate", "actions_up", "actions_right", "actions_down",
    "actions_left",
    "truncation_rate", "rollout_seconds", "update_seconds", "cpu_percent",
    "ram_bytes", "disk_free_bytes",
]

PROGRESS_FIELDS = [
    "run_id", "session_id", "timestamp", "global_step", "session_steps",
    "session_target", "session_percent", "global_target", "global_percent",
    "phase", "steps_per_second", "elapsed_seconds", "session_eta_seconds",
    "global_eta_seconds", "wall_budget_remaining_seconds", "episodes",
    "win_rate_100", "win_rate_1000",
    "average_rank_100", "reward_mean_100", "entropy_coefficient", "approx_kl",
    "cpu_percent", "ram_bytes", "recent_lineups",
]

EPISODE_FIELDS = [
    "run_id", "session_id", "global_step", "worker_index", "episode_id",
    "game_id", "seed", "phase", "requested_lineup", "actual_lineup",
    "learner_seat", "opponent_ids", "result", "win", "loss", "all_die",
    "rank", "turns", "reward_total", "reward_terminal", "reward_survival",
    "reward_elimination", "reward_food", "food_count", "elimination_count",
    "survived_turns", "final_length", "final_health", "death_cause", "truncated",
    "actions_up", "actions_right", "actions_down", "actions_left", "duration_seconds",
]

EVALUATION_GAME_FIELDS = [
    "run_id", "session_id", "evaluation_id", "evaluation_type", "global_step",
    "checkpoint_id", "checkpoint_hash", "suite", "suite_version", "game_index",
    "seed", "learner_seat", "opponent_ids", "result", "win", "loss", "all_die",
    "rank", "turns", "reward_total", "reward_terminal", "reward_survival",
    "reward_elimination", "reward_food", "food_count", "elimination_count",
    "final_length", "final_health", "death_cause", "truncated", "duration_seconds",
    "inference_p50_ms", "inference_p95_ms", "inference_p99_ms",
]

EVALUATION_SUMMARY_FIELDS = [
    "run_id", "session_id", "evaluation_id", "evaluation_type", "global_step",
    "checkpoint_id", "checkpoint_hash", "suite", "suite_version", "status",
    "na_reason", "games", "wins", "losses", "all_die", "ties", "win_rate",
    "win_ci_low", "win_ci_high", "average_rank", "rank_std", "average_turns",
    "median_turns", "p95_turns", "food_mean", "final_length_mean",
    "final_health_mean", "wall_collision_rate", "self_collision_rate",
    "body_collision_rate", "head_collision_or_unknown_rate", "inference_p50_ms",
    "inference_p95_ms", "inference_p99_ms", "duration_seconds",
]

BY_SEAT_FIELDS = EVALUATION_SUMMARY_FIELDS + ["learner_seat"]

BENCHMARK_FIELDS = [
    "run_id", "candidate_n_envs", "n_steps", "buffer_size", "repeat",
    "warmup_steps", "measure_steps", "duration_seconds", "steps_per_second",
    "games_per_second", "cpu_percent", "peak_ram_bytes", "ram_fraction",
    "throughput_cv", "valid", "invalid_reason", "selected",
]

SYSTEM_FIELDS = [
    "run_id", "session_id", "timestamp", "global_step", "cpu_percent",
    "ram_bytes", "ram_percent", "disk_free_bytes", "worker_count",
]

ACTION_DISTRIBUTION_FIELDS = [
    "run_id", "session_id", "global_step", "window_games", "actions_total",
    "up_count", "right_count", "down_count", "left_count", "up_fraction",
    "right_fraction", "down_fraction", "left_fraction",
]

LEAGUE_FIELDS = [
    "run_id", "session_id", "global_step", "event", "snapshot_id", "category",
    "valid", "reason", "overall_score", "hard_score", "model_path",
]

CHECKPOINT_FIELDS = [
    "run_id", "session_id", "global_step", "checkpoint_path", "status",
    "reason", "model_sha256", "duration_seconds",
]

SESSION_FIELDS = [
    "run_id", "session_id", "start_step", "end_step", "requested_steps",
    "rounded_steps", "completed_steps", "shortfall_steps", "wall_seconds",
    "steps_per_second", "phase", "stop_reason", "best_overall_id", "best_hard_id",
    "wr_rrr", "wr_hrr", "wr_hhr", "wr_hhh", "wr_phr", "wr_pph", "wr_ppp",
    "report_path", "checkpoint_path",
]


def _serialize(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


class CSVAppender:
    def __init__(self, path: Path, fieldnames: list[str], compressed: bool = False):
        self.path = path
        self.fieldnames = fieldnames
        self.compressed = compressed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        self._open()
        if not exists:
            self.writer.writeheader()
            self.flush()

    def _open(self) -> None:
        if self.compressed:
            self.handle = gzip.open(self.path, mode="at", encoding="utf-8", newline="")
        else:
            self.handle = self.path.open(mode="a", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.handle, fieldnames=self.fieldnames, extrasaction="ignore"
        )

    def append(self, row: dict[str, Any]) -> None:
        normalized = {key: _serialize(row.get(key, "")) for key in self.fieldnames}
        self.writer.writerow(normalized)

    def flush(self) -> None:
        self.handle.flush()
        if self.compressed:
            # Finalize a gzip member at each durability boundary. A hard Kaggle
            # stop can then lose only the current unflushed member, not the file.
            self.handle.close()
            self._open()

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


class MetricStore:
    def __init__(self, run_dir: str | Path, run_id: str, session_id: int):
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.run_id = run_id
        self.session_id = int(session_id)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        specs = {
            "train_updates": (self.metrics_dir / "train_updates.csv", TRAIN_UPDATE_FIELDS, False),
            "train_progress": (self.metrics_dir / "train_progress.csv", PROGRESS_FIELDS, False),
            "system_usage": (self.metrics_dir / "system_usage.csv", SYSTEM_FIELDS, False),
            "action_distribution": (
                self.metrics_dir / "action_distribution.csv", ACTION_DISTRIBUTION_FIELDS, False
            ),
            "episodes": (
                self.metrics_dir / "episode_summaries" / f"session_{session_id:04d}.csv.gz",
                EPISODE_FIELDS, True,
            ),
            "evaluation_games": (
                self.metrics_dir / "evaluation_games" / f"session_{session_id:04d}.csv.gz",
                EVALUATION_GAME_FIELDS, True,
            ),
            "evaluation_summary": (
                self.metrics_dir / "evaluation_summary.csv", EVALUATION_SUMMARY_FIELDS, False
            ),
            "evaluation_by_seat": (
                self.metrics_dir / "evaluation_by_seat.csv", BY_SEAT_FIELDS, False
            ),
            "benchmark": (self.metrics_dir / "benchmark_n_envs.csv", BENCHMARK_FIELDS, False),
            "league": (self.metrics_dir / "league_history.csv", LEAGUE_FIELDS, False),
            "checkpoint": (
                self.metrics_dir / "checkpoint_history.csv", CHECKPOINT_FIELDS, False
            ),
            "session": (self.metrics_dir / "session_summary.csv", SESSION_FIELDS, False),
        }
        self._dedupe_fields = {
            "train_updates": ("session_id", "global_step", "n_updates"),
            "train_progress": ("session_id", "global_step"),
            "action_distribution": ("session_id", "global_step"),
            "evaluation_games": ("evaluation_id", "suite", "game_index"),
            "evaluation_summary": ("evaluation_id", "suite"),
            "evaluation_by_seat": ("evaluation_id", "suite", "learner_seat"),
            "benchmark": ("candidate_n_envs", "repeat"),
            "league": ("snapshot_id", "event"),
            "checkpoint": ("checkpoint_path",),
            "session": ("session_id", "end_step"),
        }
        self._seen: dict[str, set[tuple[str, ...]]] = {}
        for name, fields in self._dedupe_fields.items():
            path = specs[name][0]
            try:
                rows = read_csv_rows(path)
            except (OSError, EOFError, csv.Error):
                rows = []
            self._seen[name] = {
                tuple(str(row.get(field, "")) for field in fields) for row in rows
            }
        self.tables = {
            name: CSVAppender(path, fields, compressed)
            for name, (path, fields, compressed) in specs.items()
        }
        self._last_flush = time.monotonic()

    def append(self, table: str, row: dict[str, Any]) -> bool:
        enriched = {"run_id": self.run_id, "session_id": self.session_id, **row}
        fields = self._dedupe_fields.get(table)
        if fields is not None:
            key = tuple(str(enriched.get(field, "")) for field in fields)
            if key in self._seen[table]:
                return False
            self._seen[table].add(key)
        self.tables[table].append(enriched)
        return True

    def maybe_flush(self, interval_seconds: float) -> None:
        if time.monotonic() - self._last_flush >= interval_seconds:
            self.flush()

    def flush(self) -> None:
        for table in self.tables.values():
            table.flush()
        self._last_flush = time.monotonic()

    def update_latest(self, values: dict[str, Any]) -> None:
        destination = self.metrics_dir / "latest_metrics.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(values, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        temporary.replace(destination)

    def manifest(self) -> dict[str, Any]:
        self.flush()
        files = []
        for path in sorted(self.metrics_dir.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(self.run_dir)),
                        "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                    }
                )
        return {"files": files, "generated_at": time.time()}

    def close(self) -> None:
        for table in self.tables.values():
            table.close()


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, mode="rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
