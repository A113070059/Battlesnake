from __future__ import annotations

from collections import defaultdict
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from stable_baselines3.common.callbacks import BaseCallback

from .network import build_ppo, make_vector_env


class _BenchmarkCallback(BaseCallback):
    def __init__(self, label: str, total_steps: int):
        super().__init__(verbose=0)
        self.label = str(label)
        self.total_steps = int(total_steps)
        self.episodes = 0
        self.peak_ram_bytes = 0
        self.started_at = 0.0
        self.start_step = 0
        self.last_printed_percent = -1
        self.progress = None

    def _on_training_start(self) -> None:
        self.started_at = time.perf_counter()
        self.start_step = int(self.model.num_timesteps)
        try:
            from tqdm.auto import tqdm

            self.progress = tqdm(
                total=self.total_steps,
                desc=self.label,
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )
        except Exception:
            self.progress = None
            print(f"[{self.label}] 0/{self.total_steps:,} steps (0%)", flush=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_summary" in info:
                self.episodes += 1
        if self.n_calls % 100 == 0:
            process = psutil.Process()
            rss = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
            self.peak_ram_bytes = max(self.peak_ram_bytes, rss)
        completed = min(
            self.total_steps,
            max(0, int(self.model.num_timesteps) - self.start_step),
        )
        if self.progress is not None:
            self.progress.update(max(0, completed - self.progress.n))
        else:
            percent = int(completed * 100 / max(1, self.total_steps))
            bucket = percent // 10
            if bucket > self.last_printed_percent:
                elapsed = time.perf_counter() - self.started_at
                speed = completed / elapsed if elapsed > 0 else 0.0
                remaining = (self.total_steps - completed) / speed if speed > 0 else math.inf
                eta = f"{remaining:.1f}s" if math.isfinite(remaining) else "unknown"
                print(
                    f"[{self.label}] {completed:,}/{self.total_steps:,} steps "
                    f"({percent}%) speed={speed:.1f}/s ETA={eta}",
                    flush=True,
                )
                self.last_printed_percent = bucket
        return True

    def _on_training_end(self) -> None:
        if self.progress is not None:
            self.progress.update(max(0, self.total_steps - self.progress.n))
            self.progress.close()


def _round_to_buffer(value: int, buffer_size: int) -> int:
    return max(buffer_size, int(np.ceil(value / buffer_size)) * buffer_size)


def benchmark_parallelism(config, league_manifest_path: str, metric_store):
    if not config.auto_select_n_envs:
        resolved = config.with_resolved_parallelism(config.n_envs)
        print(
            f"AUTO_SELECT_N_ENVS=False: using n_envs={resolved.effective_n_envs}, "
            f"n_steps={resolved.effective_n_steps}"
        )
        return resolved, []

    rows: list[dict[str, Any]] = []
    available_ram = psutil.virtual_memory().total
    candidates = tuple(config.n_envs_candidates)
    total_runs = len(candidates) * config.benchmark_repeats
    completed_runs = 0
    benchmark_started = time.perf_counter()
    print(
        f"Benchmark starting: {len(candidates)} n_envs candidates × "
        f"{config.benchmark_repeats} repeat(s) = {total_runs} run(s). "
        f"Each run: warmup={config.benchmark_warmup_steps:,}, "
        f"measure={config.benchmark_measure_steps:,} learner steps.",
        flush=True,
    )
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_rows: list[dict[str, Any]] = []
        for repeat in range(1, config.benchmark_repeats + 1):
            row: dict[str, Any] = {
                "candidate_n_envs": candidate,
                "repeat": repeat,
                "valid": 0,
                "selected": 0,
            }
            vector_env = None
            try:
                candidate_config = config.with_resolved_parallelism(candidate)
                row.update(
                    {
                        "n_steps": candidate_config.effective_n_steps,
                        "buffer_size": candidate_config.rollout_buffer_size,
                        "warmup_steps": candidate_config.benchmark_warmup_steps,
                        "measure_steps": candidate_config.benchmark_measure_steps,
                    }
                )
                vector_env = make_vector_env(candidate_config, league_manifest_path)
                model = build_ppo(candidate_config, vector_env, tensorboard_dir=None)
                warmup = _round_to_buffer(
                    candidate_config.benchmark_warmup_steps,
                    candidate_config.rollout_buffer_size,
                )
                measure = _round_to_buffer(
                    candidate_config.benchmark_measure_steps,
                    candidate_config.rollout_buffer_size,
                )
                run_label = (
                    f"n_envs={candidate} ({candidate_index}/{len(candidates)}) "
                    f"repeat={repeat}/{config.benchmark_repeats}"
                )
                print(
                    f"\n[{run_label}] n_steps={candidate_config.effective_n_steps}, "
                    f"buffer={candidate_config.rollout_buffer_size:,}",
                    flush=True,
                )
                warmup_callback = _BenchmarkCallback(
                    f"{run_label} warmup", warmup
                )
                model.learn(total_timesteps=warmup, callback=warmup_callback)
                callback = _BenchmarkCallback(f"{run_label} measure", measure)
                cpu_before = psutil.cpu_percent(interval=None)
                start_step = int(model.num_timesteps)
                started = time.perf_counter()
                model.learn(
                    total_timesteps=measure,
                    callback=callback,
                    reset_num_timesteps=False,
                )
                duration = time.perf_counter() - started
                measured_steps = int(model.num_timesteps) - start_step
                cpu_percent = psutil.cpu_percent(interval=None)
                row.update(
                    {
                        "duration_seconds": duration,
                        "steps_per_second": measured_steps / duration,
                        "games_per_second": callback.episodes / duration,
                        "cpu_percent": max(cpu_before, cpu_percent),
                        "peak_ram_bytes": callback.peak_ram_bytes,
                        "ram_fraction": callback.peak_ram_bytes / available_ram,
                        "valid": 1,
                        "invalid_reason": "",
                    }
                )
                completed_runs += 1
                total_elapsed = time.perf_counter() - benchmark_started
                average_run_seconds = total_elapsed / completed_runs
                overall_eta = average_run_seconds * (total_runs - completed_runs)
                print(
                    f"[{run_label}] complete: {row['steps_per_second']:.1f} steps/s, "
                    f"{row['games_per_second']:.3f} games/s, "
                    f"peak_RAM={row['peak_ram_bytes'] / (1024**3):.2f} GiB. "
                    f"Overall {completed_runs}/{total_runs}; ETA≈{overall_eta / 60:.1f} min.",
                    flush=True,
                )
            except Exception as error:
                row["invalid_reason"] = f"{type(error).__name__}: {error}"
                completed_runs += 1
                print(
                    f"[n_envs={candidate} repeat={repeat}] FAILED: "
                    f"{row['invalid_reason']}",
                    flush=True,
                )
            finally:
                if vector_env is not None:
                    vector_env.close()
            candidate_rows.append(row)
            rows.append(row)

        valid_speeds = [
            float(item["steps_per_second"])
            for item in candidate_rows
            if int(item.get("valid", 0)) == 1
        ]
        cv = (
            statistics.pstdev(valid_speeds) / statistics.fmean(valid_speeds)
            if len(valid_speeds) > 1 and statistics.fmean(valid_speeds) > 0
            else 0.0
        )
        for item in candidate_rows:
            item["throughput_cv"] = cv
            if int(item.get("valid", 0)) and (
                float(item.get("ram_fraction", 1.0)) > config.benchmark_max_ram_fraction
                or cv > config.benchmark_max_cv
            ):
                item["valid"] = 0
                item["invalid_reason"] = (
                    f"stability guard failed: ram={float(item.get('ram_fraction', 0)):.3f}, "
                    f"cv={cv:.3f}"
                )
        valid_after_guards = [
            float(item["steps_per_second"])
            for item in candidate_rows
            if int(item.get("valid", 0)) == 1
        ]
        candidate_speed = (
            statistics.median(valid_after_guards) if valid_after_guards else math.nan
        )
        print(
            f"Candidate n_envs={candidate} finished: median={candidate_speed:.1f} steps/s, "
            f"valid={len(valid_after_guards)}/{config.benchmark_repeats}, CV={cv:.3f}",
            flush=True,
        )

    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if int(row.get("valid", 0)):
            grouped[int(row["candidate_n_envs"])].append(float(row["steps_per_second"]))
    medians = {candidate: statistics.median(values) for candidate, values in grouped.items()}
    if not medians:
        for row in rows:
            metric_store.append("benchmark", row)
        metric_store.flush()
        if not config.allow_benchmark_fallback:
            raise RuntimeError("All n_envs benchmark candidates failed; fallback is disabled")
        selected = config.n_envs_fallback
    else:
        fastest = max(medians.values())
        near_fastest = [
            candidate
            for candidate, speed in medians.items()
            if speed >= fastest * (1.0 - config.benchmark_tie_tolerance)
        ]
        selected = min(near_fastest)

    for row in rows:
        if int(row["candidate_n_envs"]) == selected and int(row.get("valid", 0)):
            row["selected"] = 1
        metric_store.append("benchmark", row)
    metric_store.flush()

    resolved = config.with_resolved_parallelism(selected)
    table_rows = []
    for candidate in config.n_envs_candidates:
        candidate_values = grouped.get(candidate, [])
        table_rows.append(
            {
                "n_envs": candidate,
                "n_steps": config.rollout_buffer_target // candidate,
                "median_steps_s": statistics.median(candidate_values) if candidate_values else np.nan,
                "valid_repeats": len(candidate_values),
                "selected": candidate == selected,
            }
        )
    try:
        import pandas as pd
        from IPython.display import display

        display(pd.DataFrame(table_rows).sort_values("median_steps_s", ascending=False))
    except Exception:
        for row in table_rows:
            print(row)
    selected_speed = medians.get(selected, math.nan) if medians else math.nan
    print(
        f"Selected n_envs={selected}, n_steps={resolved.effective_n_steps}, "
        f"buffer={resolved.rollout_buffer_size}, median={selected_speed:.1f} learner steps/s"
    )
    return resolved, rows
