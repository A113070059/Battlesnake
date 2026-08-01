from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import psutil
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .checkpointing import CheckpointManager, TrainerState
from .curriculum import (
    current_phase,
    full_suites,
    quick_suites,
    session_end_suites,
    update_gates,
)
from .evaluation import evaluate_suites, file_sha256, print_evaluation, update_dashboard
from .league import LeagueManager, SnapshotRecord


def policy_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _process_usage() -> tuple[float, int, int]:
    process = psutil.Process()
    ram = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            ram += child.memory_info().rss
        except psutil.Error:
            pass
    disk = psutil.disk_usage(str(Path.cwd())).free
    return psutil.cpu_percent(interval=None), ram, disk


class TrainingMetricsCallback(BaseCallback):
    def __init__(
        self,
        config,
        metric_store,
        trainer_state: TrainerState,
        session_start_step: int,
        session_stop_step: int,
        session_started_at: float,
    ):
        super().__init__(verbose=0)
        self.config = config
        self.metric_store = metric_store
        self.trainer_state = trainer_state
        self.session_start_step = session_start_step
        self.session_stop_step = session_stop_step
        self.session_started_at = session_started_at
        self.episodes: deque[dict[str, Any]] = deque(maxlen=1_000)
        self.lineups: deque[str] = deque(maxlen=1_000)
        self.last_progress_time = 0.0
        self.last_system_time = 0.0
        self.last_permanent_step = session_start_step
        self.last_logged_update = -1
        self._last_step_sample = session_start_step
        self._last_time_sample = session_started_at
        self._rolling_speed = 0.0
        try:
            from tqdm.auto import tqdm

            self.progress = tqdm(
                total=max(0, session_stop_step - session_start_step),
                initial=max(0, int(trainer_state.global_step) - session_start_step),
                desc=f"Session {config.session_id}",
                unit="step",
                dynamic_ncols=True,
            )
        except Exception:
            self.progress = None

    def _on_step(self) -> bool:
        current = int(self.model.num_timesteps)
        infos = self.locals.get("infos", [])
        for worker_index, info in enumerate(infos):
            summary = info.get("episode_summary")
            if summary is None:
                continue
            self.trainer_state.episode_count += 1
            row = {
                "global_step": current,
                "worker_index": worker_index,
                "episode_id": self.trainer_state.episode_count,
                **summary,
            }
            self.episodes.append(row)
            self.lineups.append(str(summary.get("actual_lineup", "")))
            if self.config.keep_per_game_training_metrics:
                self.metric_store.append("episodes", row)

        now = time.monotonic()
        if now - self.last_progress_time >= self.config.progress_update_seconds:
            self._update_progress(current, now, permanent=False)
            self.last_progress_time = now
        if current - self.last_permanent_step >= self.config.progress_print_every_steps:
            self._update_progress(current, now, permanent=True)
            self.last_permanent_step = current
        if now - self.last_system_time >= self.config.system_metric_seconds:
            cpu, ram, disk = _process_usage()
            self.metric_store.append(
                "system_usage",
                {
                    "timestamp": time.time(),
                    "global_step": current,
                    "cpu_percent": cpu,
                    "ram_bytes": ram,
                    "ram_percent": psutil.virtual_memory().percent,
                    "disk_free_bytes": disk,
                    "worker_count": self.config.effective_n_envs,
                },
            )
            self.last_system_time = now
        self.metric_store.maybe_flush(self.config.metric_flush_seconds)
        return True

    def _rolling_values(self) -> dict[str, float]:
        last100 = list(self.episodes)[-100:]
        last1000 = list(self.episodes)[-1000:]

        def mean(rows, key, default=math.nan):
            values = [float(row[key]) for row in rows if key in row]
            return statistics.fmean(values) if values else default

        def total(rows, key):
            return sum(float(row.get(key, 0.0)) for row in rows)

        def cause_rate(pattern: str) -> float:
            if not last100:
                return math.nan
            return sum(pattern in str(row.get("death_cause", "")) for row in last100) / len(last100)

        turns = [float(row["turns"]) for row in last100 if "turns" in row]
        return {
            "win_rate_100": mean(last100, "win"),
            "win_rate_1000": mean(last1000, "win"),
            "average_rank_100": mean(last100, "rank"),
            "average_turns_100": mean(last100, "turns"),
            "median_turns_100": statistics.median(turns) if turns else math.nan,
            "wins_100": total(last100, "win"),
            "losses_100": total(last100, "loss"),
            "all_die_100": total(last100, "all_die"),
            "reward_mean_100": mean(last100, "reward_total"),
            "reward_terminal_mean": mean(last100, "reward_terminal"),
            "reward_survival_mean": mean(last100, "reward_survival"),
            "reward_elimination_mean": mean(last100, "reward_elimination"),
            "reward_food_mean": mean(last100, "reward_food"),
            "reward_terminal_sum": total(last100, "reward_terminal"),
            "reward_survival_sum": total(last100, "reward_survival"),
            "reward_elimination_sum": total(last100, "reward_elimination"),
            "reward_food_sum": total(last100, "reward_food"),
            "food_mean": mean(last100, "food_count"),
            "final_length_mean": mean(last100, "final_length"),
            "final_health_mean": mean(last100, "final_health"),
            "truncation_rate": mean(last100, "truncated", 0.0),
            "wall_death_rate": cause_rate("wall"),
            "self_death_rate": cause_rate("self"),
            "body_death_rate": cause_rate("snake-collision"),
            "head_or_unknown_death_rate": cause_rate("head"),
            "actions_up": total(last100, "actions_up"),
            "actions_right": total(last100, "actions_right"),
            "actions_down": total(last100, "actions_down"),
            "actions_left": total(last100, "actions_left"),
        }

    def _update_progress(self, current: int, now: float, permanent: bool) -> None:
        delta_steps = current - self._last_step_sample
        delta_time = max(1e-9, now - self._last_time_sample)
        instant_speed = delta_steps / delta_time
        self._rolling_speed = (
            instant_speed if self._rolling_speed <= 0 else 0.8 * self._rolling_speed + 0.2 * instant_speed
        )
        self._last_step_sample = current
        self._last_time_sample = now
        rolling = self._rolling_values()
        remaining_session = max(0, self.session_stop_step - current)
        remaining_global = max(0, self.config.total_target_steps - current)
        session_eta = remaining_session / self._rolling_speed if self._rolling_speed > 0 else math.nan
        global_eta = remaining_global / self._rolling_speed if self._rolling_speed > 0 else math.nan
        phase = current_phase(self.config, current, self.trainer_state.curriculum)["name"]
        lineups = dict(Counter(self.lineups).most_common())
        logger_values = getattr(self.model.logger, "name_to_value", {})
        approx_kl = float(logger_values.get("train/approx_kl", math.nan))
        cpu, ram, _ = _process_usage()
        row = {
            "timestamp": time.time(),
            "global_step": current,
            "session_steps": current - self.session_start_step,
            "session_target": self.session_stop_step - self.session_start_step,
            "session_percent": (current - self.session_start_step)
            / max(1, self.session_stop_step - self.session_start_step),
            "global_target": self.config.total_target_steps,
            "global_percent": current / max(1, self.config.total_target_steps),
            "phase": phase,
            "steps_per_second": self._rolling_speed,
            "elapsed_seconds": now - self.session_started_at,
            "session_eta_seconds": session_eta,
            "global_eta_seconds": global_eta,
            "wall_budget_remaining_seconds": max(
                0.0,
                self.config.session_max_hours * 3600
                - (now - self.session_started_at),
            ),
            "episodes": self.trainer_state.episode_count,
            **{key: rolling[key] for key in ("win_rate_100", "win_rate_1000", "average_rank_100", "reward_mean_100")},
            "entropy_coefficient": self.config.entropy_coefficient(current),
            "approx_kl": approx_kl,
            "cpu_percent": cpu,
            "ram_bytes": ram,
            "recent_lineups": lineups,
        }
        if self.progress is not None:
            expected = max(0, current - self.session_start_step)
            self.progress.update(max(0, expected - self.progress.n))
            self.progress.set_postfix(
                speed=f"{self._rolling_speed:.0f}/s",
                wr100=f"{rolling['win_rate_100']:.3f}" if math.isfinite(rolling["win_rate_100"]) else "n/a",
                global_step=f"{current/self.config.total_target_steps:.1%}",
                phase=phase,
                cpu=f"{cpu:.0f}%",
                refresh=False,
            )
        self.metric_store.update_latest(row)
        if permanent:
            self.metric_store.append("train_progress", row)
            recent = list(self.episodes)[-100:]
            counts = {
                direction: sum(int(item.get(f"actions_{direction}", 0)) for item in recent)
                for direction in ("up", "right", "down", "left")
            }
            action_total = sum(counts.values())
            self.metric_store.append(
                "action_distribution",
                {
                    "global_step": current,
                    "window_games": len(recent),
                    "actions_total": action_total,
                    **{f"{key}_count": value for key, value in counts.items()},
                    **{
                        f"{key}_fraction": value / action_total if action_total else math.nan
                        for key, value in counts.items()
                    },
                },
            )
            print(
                f"[progress] step={current:,} session={current-self.session_start_step:,}/"
                f"{self.session_stop_step-self.session_start_step:,} speed={self._rolling_speed:.1f}/s "
                f"WR100={rolling['win_rate_100']:.3f} rank100={rolling['average_rank_100']:.3f} "
                f"phase={phase}"
            )

    def _log_update(self) -> None:
        update_index = int(getattr(self.model, "_n_updates", 0))
        if update_index <= self.last_logged_update:
            return
        self.last_logged_update = update_index
        values = getattr(self.model.logger, "name_to_value", {})
        rolling = self._rolling_values()
        cpu, ram, disk = _process_usage()
        gradient_sq = 0.0
        for parameter in self.model.policy.parameters():
            if parameter.grad is not None:
                gradient_sq += float(parameter.grad.detach().norm(2).item()) ** 2
        current = int(self.model.num_timesteps)
        row = {
            "global_step": current,
            "phase": current_phase(self.config, current, self.trainer_state.curriculum)["name"],
            "n_envs": self.config.effective_n_envs,
            "n_steps": self.config.effective_n_steps,
            "buffer_size": self.config.rollout_buffer_size,
            "wall_seconds": time.monotonic() - self.session_started_at,
            "steps_per_second": self._rolling_speed,
            "learning_rate": values.get("train/learning_rate", self.config.learning_rate),
            "entropy_coefficient": self.config.entropy_coefficient(current),
            "policy_entropy": -float(values.get("train/entropy_loss", math.nan)),
            "approx_kl": values.get("train/approx_kl", math.nan),
            "clip_fraction": values.get("train/clip_fraction", math.nan),
            "policy_gradient_loss": values.get("train/policy_gradient_loss", math.nan),
            "value_loss": values.get("train/value_loss", math.nan),
            "total_loss": values.get("train/loss", math.nan),
            "explained_variance": values.get("train/explained_variance", math.nan),
            "gradient_norm": math.sqrt(gradient_sq),
            "n_updates": update_index,
            "episode_reward_mean": values.get("rollout/ep_rew_mean", math.nan),
            "episode_reward_std": statistics.pstdev(
                [float(row["reward_total"]) for row in self.episodes]
            ) if len(self.episodes) > 1 else 0.0,
            "episode_reward_min": min(
                (float(item["reward_total"]) for item in self.episodes), default=math.nan
            ),
            "episode_reward_max": max(
                (float(item["reward_total"]) for item in self.episodes), default=math.nan
            ),
            "update_epochs": self.config.n_epochs,
            **rolling,
            "rollout_seconds": math.nan,
            "update_seconds": math.nan,
            "cpu_percent": cpu,
            "ram_bytes": ram,
            "disk_free_bytes": disk,
        }
        self.metric_store.append("train_updates", row)

    def _on_rollout_start(self) -> None:
        if int(getattr(self.model, "_n_updates", 0)) > 0:
            self._log_update()

    def _on_training_end(self) -> None:
        self._log_update()
        self.metric_store.flush()

    def close(self) -> None:
        if self.progress is not None:
            self.progress.close()


def _round_up(value: int, divisor: int) -> int:
    return max(divisor, int(math.ceil(value / divisor)) * divisor)


def _record_snapshot_metrics(result) -> tuple[dict[str, float], list[float | None]]:
    metrics: dict[str, float] = {}
    rank_values: list[float] = []
    turn_values: list[float] = []
    for row in result.summaries:
        if row.get("status") != "ok":
            continue
        suite = row["suite"]
        metrics[f"WR_{suite}"] = float(row["win_rate"])
        rank_values.append(float(row["average_rank"]))
        turn_values.append(float(row["average_turns"]))
    metrics["average_rank"] = statistics.fmean(rank_values) if rank_values else math.nan
    metrics["average_survival_turns"] = statistics.fmean(turn_values) if turn_values else math.nan
    fingerprint = [
        metrics.get("WR_RRR"),
        metrics.get("WR_HHH"),
        metrics.get("WR_PHR"),
        metrics.get("WR_PPH"),
        metrics.get("WR_PPP"),
        metrics.get("average_rank"),
        metrics.get("average_survival_turns"),
    ]
    return metrics, fingerprint


def _write_session_report(
    run_dir: Path,
    config,
    state: TrainerState,
    start_step: int,
    stop_step: int,
    elapsed: float,
    stop_reason: str,
    session_result,
    checkpoint_path: Path,
) -> Path:
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"session_{config.session_id:04d}_report.md"
    lines = [
        f"# Session {config.session_id} Report",
        "",
        f"- Run: `{config.run_id}`",
        f"- Steps: `{start_step:,}` → `{state.global_step:,}`",
        f"- Requested session steps: `{config.session_target_steps}`",
        f"- Rounded session stop: `{stop_step:,}`",
        f"- Completed session steps: `{state.global_step-start_step:,}`",
        f"- Wall time: `{elapsed/3600:.3f}` hours",
        f"- Stop reason: `{stop_reason}`",
        f"- n_envs/n_steps/buffer: `{config.effective_n_envs}/{config.effective_n_steps}/{config.rollout_buffer_size}`",
        f"- Checkpoint: `{checkpoint_path}`",
        "",
        "## Session-end evaluation",
        "",
        "| Suite | Status | Games | Win rate | Average rank | Median turns |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in session_result.summaries:
        if row.get("status") == "ok":
            lines.append(
                f"| {row['suite']} | ok | {int(row['games'])} | {float(row['win_rate']):.4f} | "
                f"{float(row['average_rank']):.3f} | {float(row['median_turns']):.1f} |"
            )
        else:
            lines.append(f"| {row['suite']} | N/A: {row.get('na_reason','')} | 0 | | | |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_training_session(
    config,
    run_dir: str | Path,
    model,
    vector_env,
    league: LeagueManager,
    metric_store,
    checkpoint_manager: CheckpointManager,
    trainer_state: TrainerState,
):
    run_dir = Path(run_dir)
    session_started = time.monotonic()
    start_step = int(model.num_timesteps)
    trainer_state.global_step = start_step
    trainer_state.session_id = config.session_id
    trainer_state.session_start_step = start_step
    trainer_state.session_requested_steps = config.session_target_steps
    rounded = config.rounded_session_steps()
    if config.smoke_test_only:
        smoke_rounded = _round_up(config.smoke_test_steps, config.rollout_buffer_size)
        rounded = smoke_rounded if rounded is None else min(rounded, smoke_rounded)
    trainer_state.session_rounded_steps = rounded
    session_stop = config.total_target_steps if rounded is None else min(
        config.total_target_steps, start_step + rounded
    )
    deadline = session_started + config.session_max_hours * 3600 - config.save_reserve_minutes * 60
    callback = TrainingMetricsCallback(
        config, metric_store, trainer_state, start_step, session_stop, session_started
    )
    last_result = None
    last_checkpoint = None
    stop_reason = "session_step_target"

    while int(model.num_timesteps) < session_stop:
        current = int(model.num_timesteps)
        if time.monotonic() >= deadline:
            stop_reason = "wall_time_safety_deadline"
            break
        vector_env.env_method("set_global_step", current)
        vector_env.env_method("set_curriculum_state", trainer_state.curriculum.to_dict())
        vector_env.env_method("reload_league")
        model.ent_coef = config.entropy_coefficient(current)

        checkpoint_block = _round_up(config.save_every_steps, config.rollout_buffer_size)
        boundaries = [
            current + min(config.train_block_steps, checkpoint_block),
            session_stop,
        ]
        if trainer_state.next_quick_step > current:
            boundaries.append(trainer_state.next_quick_step)
        if trainer_state.next_full_step > current:
            boundaries.append(trainer_state.next_full_step)
        target = min(boundaries)
        learn_steps = _round_up(target - current, config.rollout_buffer_size)
        learn_steps = min(learn_steps, session_stop - current)
        if learn_steps <= 0:
            break
        try:
            model.learn(
                total_timesteps=learn_steps,
                callback=callback,
                reset_num_timesteps=False,
                tb_log_name=config.run_id,
            )
        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt_restored_previous_good"
            print(
                "KeyboardInterrupt received. Discarding the possibly partial rollout and "
                "restoring the previous complete checkpoint."
            )
            model, restored_state, restored_manifest, last_checkpoint = (
                checkpoint_manager.load_latest(vector_env)
            )
            trainer_state = restored_state
            trainer_state.session_id = config.session_id
            trainer_state.session_start_step = start_step
            league.manifest = restored_manifest
            league.save()
            break
        current = int(model.num_timesteps)
        trainer_state.global_step = current
        event_reasons: list[str] = []

        if current >= trainer_state.next_quick_step:
            suites = quick_suites(config, current, trainer_state.curriculum)
            quick_result = evaluate_suites(
                model,
                config,
                league,
                metric_store,
                suites,
                config.quick_eval_games_per_suite,
                "quick",
                current,
                f"current-{current:012d}",
                policy_hash(model),
                trainer_state.curriculum.to_dict(),
                f"quick-{trainer_state.next_quick_step}",
            )
            print_evaluation(quick_result)
            trainer_state.latest_win_rates.update(quick_result.win_rates())
            trainer_state.latest_evaluation_id = quick_result.evaluation_id

            gate_suite = None
            gate_games = 0
            if not trainer_state.curriculum.phase2_gate_passed and current >= config.phase2_gate_step:
                gate_suite, gate_games = config.phase2_gate_suite, config.phase2_gate_games
            elif not trainer_state.curriculum.phase3_gate_passed and current >= config.phase3_gate_step:
                gate_suite, gate_games = config.phase3_gate_suite, config.phase3_gate_games
            gate_result = None
            if gate_suite is not None:
                gate_result = evaluate_suites(
                    model,
                    config,
                    league,
                    metric_store,
                    [gate_suite],
                    gate_games,
                    "gate",
                    current,
                    f"current-{current:012d}",
                    policy_hash(model),
                    trainer_state.curriculum.to_dict(),
                    f"gate-{gate_suite}-{current}",
                )
                print_evaluation(gate_result)
                update_gates(
                    config,
                    current,
                    trainer_state.curriculum,
                    gate_result.win_rates(),
                    gate_result.game_counts(),
                )
            while trainer_state.next_quick_step <= current:
                trainer_state.next_quick_step += config.quick_eval_interval_steps
            event_reasons.append("quick_eval")
            last_result = gate_result or quick_result

        if current >= trainer_state.next_full_step:
            snapshot_id = f"ppo_{current:012d}"
            snapshot_path = run_dir / "snapshots" / f"{snapshot_id}.zip"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(snapshot_path.with_suffix("")))
            snapshot_hash = file_sha256(snapshot_path)
            full_result = evaluate_suites(
                model,
                config,
                league,
                metric_store,
                full_suites(),
                config.full_eval_games_per_suite,
                "full",
                current,
                snapshot_id,
                snapshot_hash,
                trainer_state.curriculum.to_dict(),
                f"full-{trainer_state.next_full_step}",
            )
            print_evaluation(full_result)
            metrics, fingerprint = _record_snapshot_metrics(full_result)
            record = SnapshotRecord(
                snapshot_id=snapshot_id,
                step=current,
                model_path=str(snapshot_path),
                config_hash=config.semantic_hash(),
                metrics=metrics,
                fingerprint=fingerprint,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            existing = league.valid_records()
            if (
                trainer_state.curriculum.phase2_gate_passed
                and metrics.get("WR_RRR", 1.0) < config.catastrophic_rrr_floor
            ):
                record.valid = False
                record.invalid_reason = "catastrophic WR_RRR regression"
            elif existing:
                best_score = max(item.overall_score() for item in existing)
                if record.overall_score() < best_score - config.catastrophic_score_gap:
                    record.valid = False
                    record.invalid_reason = "catastrophic overall-score regression"
            league.add_snapshot(record)
            trainer_state.best_overall_id = league.manifest.best_overall_id
            trainer_state.best_hard_id = league.manifest.best_hard_id
            trainer_state.latest_win_rates.update(full_result.win_rates())
            trainer_state.latest_evaluation_id = full_result.evaluation_id
            metric_store.append(
                "league",
                {
                    "global_step": current,
                    "event": "snapshot_added",
                    "snapshot_id": snapshot_id,
                    "category": record.category,
                    "valid": int(record.valid),
                    "reason": record.invalid_reason,
                    "overall_score": record.overall_score(),
                    "hard_score": record.hard_score(),
                    "model_path": str(snapshot_path),
                },
            )
            while trainer_state.next_full_step <= current:
                trainer_state.next_full_step += config.full_eval_interval_steps
            while trainer_state.next_snapshot_step <= current:
                trainer_state.next_snapshot_step += config.snapshot_interval_steps
            event_reasons.extend(["full_eval", "snapshot"])
            last_result = full_result
            update_dashboard(run_dir)

        trainer_state.global_step = current
        last_checkpoint, _, _ = checkpoint_manager.save(
            model,
            trainer_state,
            league,
            metric_store,
            reason="+".join(event_reasons) if event_reasons else "regular_block",
        )

    trainer_state.global_step = int(model.num_timesteps)
    trainer_state.session_id = config.session_id
    if trainer_state.global_step >= config.total_target_steps:
        stop_reason = "total_training_target"
    trainer_state.stop_reason = stop_reason
    session_games = min(
        config.session_end_eval_games_per_suite,
        8 if config.smoke_test_only else config.session_end_eval_games_per_suite,
    )
    session_result = evaluate_suites(
        model,
        config,
        league,
        metric_store,
        session_end_suites(),
        session_games,
        "session_end",
        trainer_state.global_step,
        f"current-{trainer_state.global_step:012d}",
        policy_hash(model),
        trainer_state.curriculum.to_dict(),
        f"session-{config.session_id}",
    )
    print_evaluation(session_result)
    trainer_state.latest_win_rates.update(session_result.win_rates())
    trainer_state.latest_evaluation_id = session_result.evaluation_id
    last_checkpoint, _, _ = checkpoint_manager.save(
        model, trainer_state, league, metric_store, reason="session_end"
    )
    elapsed = time.monotonic() - session_started
    report_path = _write_session_report(
        run_dir,
        config,
        trainer_state,
        start_step,
        session_stop,
        elapsed,
        stop_reason,
        session_result,
        last_checkpoint,
    )
    summary = {
        "start_step": start_step,
        "end_step": trainer_state.global_step,
        "requested_steps": config.session_target_steps,
        "rounded_steps": rounded,
        "completed_steps": trainer_state.global_step - start_step,
        "shortfall_steps": max(0, session_stop - trainer_state.global_step),
        "wall_seconds": elapsed,
        "steps_per_second": (trainer_state.global_step - start_step) / max(elapsed, 1e-9),
        "phase": current_phase(
            config, trainer_state.global_step, trainer_state.curriculum
        )["name"],
        "stop_reason": stop_reason,
        "best_overall_id": trainer_state.best_overall_id,
        "best_hard_id": trainer_state.best_hard_id,
        "wr_rrr": trainer_state.latest_win_rates.get("RRR", math.nan),
        "wr_hrr": trainer_state.latest_win_rates.get("HRR", math.nan),
        "wr_hhr": trainer_state.latest_win_rates.get("HHR", math.nan),
        "wr_hhh": trainer_state.latest_win_rates.get("HHH", math.nan),
        "wr_phr": trainer_state.latest_win_rates.get("PHR", math.nan),
        "wr_pph": trainer_state.latest_win_rates.get("PPH", math.nan),
        "wr_ppp": trainer_state.latest_win_rates.get("PPP", math.nan),
        "report_path": str(report_path),
        "checkpoint_path": str(last_checkpoint),
    }
    metric_store.append("session", summary)
    metric_store.flush()
    update_dashboard(run_dir)
    callback.close()
    return trainer_state, session_result, summary
