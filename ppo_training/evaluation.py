from __future__ import annotations

import concurrent.futures
from concurrent.futures.process import BrokenProcessPool

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

import numpy as np

from .env import BlackoutSingleLearnerEnv
from .league import LeagueManager
from .metrics import MetricStore


@dataclass(slots=True)
class EvaluationResult:
    evaluation_id: str
    summaries: list[dict[str, Any]]
    games: list[dict[str, Any]]

    def win_rates(self) -> dict[str, float]:
        return {
            row["suite"]: float(row["win_rate"])
            for row in self.summaries
            if row.get("status") == "ok"
        }

    def game_counts(self) -> dict[str, int]:
        return {
            row["suite"]: int(row["games"])
            for row in self.summaries
            if row.get("status") == "ok"
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        return math.nan, math.nan
    p = wins / games
    denominator = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / games + z * z / (4.0 * games * games))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else math.nan


def _summary_row(
    base: dict[str, Any], games: list[dict[str, Any]], inference_ms: list[float]
) -> dict[str, Any]:
    count = len(games)
    wins = sum(int(row["win"]) for row in games)
    losses = sum(int(row["loss"]) for row in games)
    all_die = sum(int(row["all_die"]) for row in games)
    ranks = [float(row["rank"]) for row in games]
    turns = [float(row["turns"]) for row in games]
    low, high = wilson_interval(wins, count)

    def cause_rate(fragment: str) -> float:
        if not count:
            return math.nan
        return sum(fragment in str(row["death_cause"]) for row in games) / count

    return {
        **base,
        "status": "ok",
        "na_reason": "",
        "games": count,
        "wins": wins,
        "losses": losses,
        "all_die": all_die,
        "ties": sum(abs(rank - round(rank)) > 1e-9 for rank in ranks),
        "win_rate": wins / count if count else math.nan,
        "win_ci_low": low,
        "win_ci_high": high,
        "average_rank": statistics.fmean(ranks) if ranks else math.nan,
        "rank_std": statistics.pstdev(ranks) if len(ranks) > 1 else 0.0,
        "average_turns": statistics.fmean(turns) if turns else math.nan,
        "median_turns": statistics.median(turns) if turns else math.nan,
        "p95_turns": _percentile(turns, 95),
        "food_mean": statistics.fmean(float(row["food_count"]) for row in games),
        "final_length_mean": statistics.fmean(float(row["final_length"]) for row in games),
        "final_health_mean": statistics.fmean(float(row["final_health"]) for row in games),
        "wall_collision_rate": cause_rate("wall"),
        "self_collision_rate": cause_rate("self"),
        "body_collision_rate": cause_rate("snake-collision"),
        "head_collision_or_unknown_rate": cause_rate("head"),
        "inference_p50_ms": _percentile(inference_ms, 50),
        "inference_p95_ms": _percentile(inference_ms, 95),
        "inference_p99_ms": _percentile(inference_ms, 99),
        "duration_seconds": sum(float(row["duration_seconds"]) for row in games),
    }


_worker_env = None
_worker_model = None

def _init_worker(config_dict, manifest_path, curriculum_state, global_step, model_path):
    global _worker_env, _worker_model
    import multiprocessing
    import torch
    from stable_baselines3 import PPO
    from .env import BlackoutSingleLearnerEnv
    
    torch.set_num_threads(1)
    
    _worker_env = BlackoutSingleLearnerEnv(
        config_dict, manifest_path, worker_index=90_000 + multiprocessing.current_process().pid
    )
    _worker_env.set_global_step(global_step)
    _worker_env.set_curriculum_state(curriculum_state)
    
    _worker_model = PPO.load(model_path, custom_objects={"device": "cpu"})

def _run_game_in_worker(args):
    global _worker_env, _worker_model
    import time
    game_seed = args["game_seed"]
    learner_seat = args["learner_seat"]
    suite = args["suite"]
    symmetry = args["symmetry"]
    
    observation, _ = _worker_env.reset(
        seed=game_seed,
        options={
            "learner_seat": learner_seat,
            "lineup": suite,
            "unique_ppo": True,
            "symmetry": symmetry,
        },
    )
    terminated = truncated = False
    final_info = {}
    game_inference_ms = []
    started = time.perf_counter()
    while not (terminated or truncated):
        infer_started = time.perf_counter()
        action, _ = _worker_model.predict(observation, deterministic=True)
        game_inference_ms.append((time.perf_counter() - infer_started) * 1000.0)
        observation, _, terminated, truncated, final_info = _worker_env.step(action)
        
    duration_seconds = time.perf_counter() - started
    summary = dict(final_info["episode_summary"])
    return summary, game_inference_ms, duration_seconds

def evaluate_suites(
    model,
    config,
    league: LeagueManager,
    metric_store: MetricStore,
    suites: Iterable[str],
    games_per_suite: int,
    evaluation_type: str,
    global_step: int,
    checkpoint_id: str,
    checkpoint_hash: str,
    curriculum_state: dict[str, Any],
    seed_namespace: str,
) -> EvaluationResult:
    evaluation_id = f"{evaluation_type}-{global_step:012d}-{seed_namespace}"
    all_summaries: list[dict[str, Any]] = []
    all_games: list[dict[str, Any]] = []
    suite_version = league.manifest.eval_suite_version

    temp_model_path = Path(config.output_root) / config.run_id / f"tmp_eval_model_{evaluation_type}.zip"
    temp_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(temp_model_path))

    max_workers = getattr(config, "worker_count", 8)
    if max_workers <= 0:
        max_workers = None

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(config.to_dict(), str(league.manifest_path), curriculum_state, global_step, str(temp_model_path)),
    ) as executor:
        try:
            for suite in suites:
                base = {
                    "evaluation_id": evaluation_id,
                    "evaluation_type": evaluation_type,
                    "global_step": global_step,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_hash": checkpoint_hash,
                    "suite": suite,
                    "suite_version": suite_version,
                }
                available, reason = league.can_evaluate(suite)
                if not available:
                    row = {
                        **base,
                        "status": "na",
                        "na_reason": reason,
                        "games": 0,
                    }
                    metric_store.append("evaluation_summary", row)
                    all_summaries.append(row)
                    continue

                suite_games: list[dict[str, Any]] = []
                inference_times: list[float] = []
                digest = hashlib.sha256(
                    f"{config.base_seed}|{seed_namespace}|{suite}".encode("utf-8")
                ).digest()
                suite_seed = int.from_bytes(digest[:4], "little")
                fixed_digest = hashlib.sha256(
                    f"{config.base_seed}|{suite_version}|fixed|{suite}".encode("utf-8")
                ).digest()
                fixed_suite_seed = int.from_bytes(fixed_digest[:4], "little")
                fixed_games = (
                    int(games_per_suite * config.final_fixed_seed_fraction)
                    if evaluation_type == "final"
                    else 0
                )
                
                game_tasks = []
                for game_index in range(games_per_suite):
                    seed_base = fixed_suite_seed if game_index < fixed_games else suite_seed
                    game_seed = (seed_base + game_index * 104_729) & 0x7FFFFFFF
                    learner_seat = game_index % config.num_players
                    symmetry = game_index % 8 if config.random_symmetry else 0
                    
                    game_args = {
                        "game_seed": game_seed,
                        "learner_seat": learner_seat,
                        "suite": suite,
                        "symmetry": symmetry,
                    }
                    future = executor.submit(_run_game_in_worker, game_args)
                    game_tasks.append((game_index, game_seed, learner_seat, future))

                for game_index, game_seed, learner_seat, future in game_tasks:
                    summary, game_inference_ms, duration_seconds = future.result()
                    summary.update(
                        {
                            **base,
                            "game_index": game_index,
                            "seed": game_seed,
                            "learner_seat": learner_seat,
                            "duration_seconds": duration_seconds,
                            "inference_p50_ms": _percentile(game_inference_ms, 50),
                            "inference_p95_ms": _percentile(game_inference_ms, 95),
                            "inference_p99_ms": _percentile(game_inference_ms, 99),
                        }
                    )
                    suite_games.append(summary)
                    all_games.append(summary)
                    inference_times.extend(game_inference_ms)
                    metric_store.append("evaluation_games", summary)

                summary_row = _summary_row(base, suite_games, inference_times)
                metric_store.append("evaluation_summary", summary_row)
                all_summaries.append(summary_row)
                for seat in range(config.num_players):
                    seat_games = [row for row in suite_games if int(row["learner_seat"]) == seat]
                    seat_inference = [
                        float(row["inference_p50_ms"]) for row in seat_games
                    ]
                    seat_row = _summary_row(
                        {**base, "learner_seat": seat}, seat_games, seat_inference
                    )
                    metric_store.append("evaluation_by_seat", seat_row)
        except BrokenProcessPool as e:
            print(f"Warning: Evaluation process pool broke: {e}")
        finally:
            metric_store.flush()
            try:
                temp_model_path.unlink()
            except OSError:
                pass
            
    return EvaluationResult(evaluation_id, all_summaries, all_games)


def print_evaluation(result: EvaluationResult) -> None:
    rows = []
    for item in result.summaries:
        if item.get("status") != "ok":
            rows.append(
                {
                    "Suite": item["suite"],
                    "Games": 0,
                    "WinRate": "N/A",
                    "95% CI": item.get("na_reason", "N/A"),
                    "AvgRank": "",
                    "MedianTurns": "",
                }
            )
        else:
            rows.append(
                {
                    "Suite": item["suite"],
                    "Games": int(item["games"]),
                    "Wins": int(item["wins"]),
                    "Losses": int(item["losses"]),
                    "AllDie": int(item["all_die"]),
                    "WinRate": f"{100*float(item['win_rate']):.1f}%",
                    "95% CI": f"[{100*float(item['win_ci_low']):.1f}, {100*float(item['win_ci_high']):.1f}]",
                    "AvgRank": f"{float(item['average_rank']):.3f}",
                    "MedianTurns": f"{float(item['median_turns']):.1f}",
                }
            )
    print(f"\nEvaluation: {result.evaluation_id}")
    try:
        import pandas as pd
        from IPython.display import display

        display(pd.DataFrame(rows).fillna(""))
    except Exception:
        for row in rows:
            print(row)


def update_dashboard(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    source = run_dir / "metrics" / "evaluation_summary.csv"
    if not source.exists():
        return
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        reports = run_dir / "reports"
        figures = reports / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(source)
        valid = frame[frame["status"] == "ok"].copy()
        if valid.empty:
            return
        pivot = valid.pivot_table(
            index=["global_step", "checkpoint_id"],
            columns="suite",
            values="win_rate",
            aggfunc="last",
        ).sort_index()
        axis = valid.pivot_table(
            index="global_step", columns="suite", values="win_rate", aggfunc="last"
        ).plot(marker="o", figsize=(10, 5), title="Win rate by evaluation suite")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Win rate")
        axis.figure.tight_layout()
        axis.figure.savefig(figures / "win_rate_by_suite.png", dpi=140)
        plt.close(axis.figure)

        updates_path = run_dir / "metrics" / "train_updates.csv"
        if updates_path.exists():
            updates = pd.read_csv(updates_path)
            if not updates.empty:
                loss_columns = [
                    column
                    for column in ("policy_gradient_loss", "value_loss", "total_loss")
                    if column in updates
                ]
                if loss_columns:
                    loss_axis = updates.plot(
                        x="global_step", y=loss_columns, figsize=(10, 5), title="PPO losses"
                    )
                    loss_axis.figure.tight_layout()
                    loss_axis.figure.savefig(figures / "ppo_losses.png", dpi=140)
                    plt.close(loss_axis.figure)
                reward_columns = [
                    column
                    for column in (
                        "episode_reward_mean", "reward_terminal_mean",
                        "reward_survival_mean", "reward_elimination_mean", "reward_food_mean",
                    )
                    if column in updates
                ]
                if reward_columns:
                    reward_axis = updates.plot(
                        x="global_step", y=reward_columns, figsize=(10, 5),
                        title="Reward components",
                    )
                    reward_axis.figure.tight_layout()
                    reward_axis.figure.savefig(figures / "reward_components.png", dpi=140)
                    plt.close(reward_axis.figure)

        progress_path = run_dir / "metrics" / "train_progress.csv"
        if progress_path.exists():
            progress = pd.read_csv(progress_path)
            if not progress.empty and "steps_per_second" in progress:
                speed_axis = progress.plot(
                    x="global_step", y=["steps_per_second", "cpu_percent"],
                    secondary_y="cpu_percent", figsize=(10, 5),
                    title="Training throughput and CPU usage",
                )
                speed_axis.figure.tight_layout()
                speed_axis.figure.savefig(figures / "throughput_and_cpu.png", dpi=140)
                plt.close(speed_axis.figure)

        html = "<h1>Battlesnake PPO latest results</h1>" + pivot.tail(20).to_html(
            float_format=lambda value: f"{value:.3f}"
        )
        html += (
            '<h2>Figures</h2><ul>'
            '<li><a href="figures/win_rate_by_suite.png">Win rates</a></li>'
            '<li><a href="figures/ppo_losses.png">PPO losses</a></li>'
            '<li><a href="figures/reward_components.png">Reward components</a></li>'
            '<li><a href="figures/throughput_and_cpu.png">Throughput and CPU</a></li>'
            '</ul>'
        )
        (reports / "latest_dashboard.html").write_text(html, encoding="utf-8")
    except Exception as error:
        (run_dir / "reports").mkdir(parents=True, exist_ok=True)
        (run_dir / "reports" / "dashboard_error.txt").write_text(
            str(error), encoding="utf-8"
        )
