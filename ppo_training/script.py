import sys
import re

with open('evaluation.py', 'r') as f:
    content = f.read()

imports_addition = '''from __future__ import annotations

import concurrent.futures
'''

content = content.replace('from __future__ import annotations\n', imports_addition)

workers_addition = '''
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
'''

content = content.replace('\ndef evaluate_suites(\n', workers_addition)

evaluate_suites_old = '''def evaluate_suites(
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
    environment = BlackoutSingleLearnerEnv(
        config.to_dict(), str(league.manifest_path), worker_index=90_000
    )
    environment.set_global_step(global_step)
    environment.set_curriculum_state(curriculum_state)
    suite_version = league.manifest.eval_suite_version

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
            for game_index in range(games_per_suite):
                seed_base = fixed_suite_seed if game_index < fixed_games else suite_seed
                game_seed = (seed_base + game_index * 104_729) & 0x7FFFFFFF
                learner_seat = game_index % config.num_players
                observation, _ = environment.reset(
                    seed=game_seed,
                    options={
                        "learner_seat": learner_seat,
                        "lineup": suite,
                        "unique_ppo": True,
                        "symmetry": game_index % 8 if config.random_symmetry else 0,
                    },
                )
                terminated = truncated = False
                final_info: dict[str, Any] = {}
                game_inference_ms: list[float] = []
                started = time.perf_counter()
                while not (terminated or truncated):
                    infer_started = time.perf_counter()
                    action, _ = model.predict(observation, deterministic=True)
                    game_inference_ms.append((time.perf_counter() - infer_started) * 1000.0)
                    observation, _, terminated, truncated, final_info = environment.step(action)
                summary = dict(final_info["episode_summary"])
                summary.update(
                    {
                        **base,
                        "game_index": game_index,
                        "seed": game_seed,
                        "learner_seat": learner_seat,
                        "duration_seconds": time.perf_counter() - started,
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
    finally:
        environment.close()
        metric_store.flush()
    return EvaluationResult(evaluation_id, all_summaries, all_games)'''

evaluate_suites_new = '''def evaluate_suites(
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
        finally:
            metric_store.flush()
            try:
                temp_model_path.unlink()
            except OSError:
                pass
            
    return EvaluationResult(evaluation_id, all_summaries, all_games)'''

if evaluate_suites_old in content:
    content = content.replace(evaluate_suites_old, evaluate_suites_new)
    with open('evaluation.py', 'w') as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAILED TO FIND OLD CONTENT")
