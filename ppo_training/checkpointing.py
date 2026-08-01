from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time
import uuid
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig
from .curriculum import CurriculumState
from .league import LeagueManager, LeagueManifest


@dataclass(slots=True)
class TrainerState:
    global_step: int = 0
    session_id: int = 1
    session_start_step: int = 0
    session_requested_steps: int | None = None
    session_rounded_steps: int | None = None
    episode_count: int = 0
    next_quick_step: int = 500_000
    next_full_step: int = 2_000_000
    next_snapshot_step: int = 2_000_000
    curriculum: CurriculumState = field(default_factory=CurriculumState)
    latest_win_rates: dict[str, float] = field(default_factory=dict)
    latest_evaluation_id: str = ""
    best_overall_id: str | None = None
    best_hard_id: str | None = None
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["curriculum"] = self.curriculum.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainerState":
        data = dict(value)
        data["curriculum"] = CurriculumState.from_dict(data.get("curriculum"))
        return cls(**data)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def runtime_manifest(device: str = "cpu") -> dict[str, Any]:
    packages = {}
    for name in (
        "hisss", "numpy", "pydantic", "gymnasium", "stable-baselines3",
        "torch", "psutil", "pandas", "matplotlib", "tensorboard",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": str(device),
    }


class CheckpointManager:
    def __init__(self, run_dir: str | Path, config: ExperimentConfig):
        self.run_dir = Path(run_dir)
        self.config = config
        self.checkpoint_root = self.run_dir / "checkpoints"
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

    @property
    def latest_pointer(self) -> Path:
        return self.checkpoint_root / "latest.json"

    def _destination_for_step(self, step: int) -> Path:
        base = self.checkpoint_root / f"step_{step:012d}"
        if not base.exists():
            return base
        index = 1
        while (self.checkpoint_root / f"step_{step:012d}_{index:03d}").exists():
            index += 1
        return self.checkpoint_root / f"step_{step:012d}_{index:03d}"

    def save(
        self,
        model,
        trainer_state: TrainerState,
        league: LeagueManager,
        metric_store,
        reason: str,
    ) -> tuple[Path, str, float]:
        started = time.perf_counter()
        metric_store.flush()
        destination = self._destination_for_step(int(model.num_timesteps))
        temporary = self.checkpoint_root / f".tmp-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)

        model.save(str(temporary / "ppo_model"))
        _atomic_json(temporary / "trainer_state.json", trainer_state.to_dict())
        torch.save(capture_rng_state(), temporary / "rng_state.pt")
        self.config.save(temporary / "config.json")
        _atomic_json(temporary / "runtime.json", runtime_manifest(self.config.device))
        _atomic_json(temporary / "league_manifest.json", league.manifest.to_dict())
        _atomic_json(temporary / "metrics_manifest.json", metric_store.manifest())
        _atomic_json(
            temporary / "checkpoint_metadata.json",
            {
                "reason": reason,
                "global_step": int(model.num_timesteps),
                "semantic_hash": self.config.semantic_hash(),
                "created_at": time.time(),
            },
        )

        checksums: dict[str, str] = {}
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name not in {"checksums.json", "complete.marker"}:
                checksums[str(path.relative_to(temporary))] = sha256_file(path)
        _atomic_json(temporary / "checksums.json", checksums)
        (temporary / "complete.marker").write_text("complete\n", encoding="utf-8")
        os.replace(temporary, destination)
        model_hash = checksums["ppo_model.zip"]
        _atomic_json(
            self.latest_pointer,
            {
                "path": str(destination),
                "global_step": int(model.num_timesteps),
                "model_sha256": model_hash,
                "semantic_hash": self.config.semantic_hash(),
            },
        )
        duration = time.perf_counter() - started
        metric_store.append(
            "checkpoint",
            {
                "global_step": int(model.num_timesteps),
                "checkpoint_path": str(destination),
                "status": "complete",
                "reason": reason,
                "model_sha256": model_hash,
                "duration_seconds": duration,
            },
        )
        metric_store.flush()
        self._cleanup_ordinary_checkpoints()
        return destination, model_hash, duration

    def _cleanup_ordinary_checkpoints(self) -> None:
        complete = [
            path
            for path in self.checkpoint_root.glob("step_*")
            if path.is_dir() and (path / "complete.marker").exists()
        ]
        complete.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        keep = set(complete[:2])
        for path in complete:
            try:
                metadata = json.loads(
                    (path / "checkpoint_metadata.json").read_text(encoding="utf-8")
                )
            except Exception:
                keep.add(path)
                continue
            step = int(metadata.get("global_step", -1))
            reason = str(metadata.get("reason", ""))
            if step % self.config.snapshot_interval_steps == 0 or "snapshot" in reason or "best" in reason:
                keep.add(path)
        for path in complete:
            if path not in keep:
                resolved = path.resolve()
                if resolved.parent == self.checkpoint_root.resolve() and resolved.name.startswith("step_"):
                    shutil.rmtree(resolved)

    def verify(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        directory = Path(checkpoint_dir)
        if not (directory / "complete.marker").exists():
            raise RuntimeError(f"Incomplete checkpoint: {directory}")
        checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
        for relative, expected in checksums.items():
            path = directory / relative
            if not path.exists() or sha256_file(path) != expected:
                raise RuntimeError(f"Checkpoint checksum mismatch: {path}")
        saved_config = ExperimentConfig.load(directory / "config.json")
        if saved_config.semantic_hash() != self.config.semantic_hash():
            raise RuntimeError(
                "Checkpoint semantic config does not match current resolved config"
            )
        return {
            "checksums": checksums,
            "config": saved_config,
            "trainer_state": TrainerState.from_dict(
                json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
            ),
            "league": LeagueManifest.from_dict(
                json.loads((directory / "league_manifest.json").read_text(encoding="utf-8"))
            ),
        }

    def latest(self) -> Path | None:
        if not self.latest_pointer.exists():
            return None
        try:
            value = json.loads(self.latest_pointer.read_text(encoding="utf-8"))
            path = Path(value["path"])
        except (OSError, json.JSONDecodeError, KeyError):
            return None
        if path.exists():
            return path
        relocated = self.checkpoint_root / path.name
        return relocated if relocated.exists() else None

    def checkpoint_candidates(self) -> list[Path]:
        """Return newest-first complete checkpoints, preferring latest.json."""
        candidates: list[Path] = []
        preferred = self.latest()
        if preferred is not None:
            candidates.append(preferred)
        discovered = sorted(
            (
                path
                for path in self.checkpoint_root.glob("step_*")
                if path.is_dir() and (path / "complete.marker").exists()
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        candidates.extend(path for path in discovered if path not in candidates)
        return candidates

    def load_latest(self, vector_env):
        from stable_baselines3 import PPO

        candidates = self.checkpoint_candidates()
        if not candidates:
            raise FileNotFoundError("No complete latest checkpoint")
        failures: list[str] = []
        for checkpoint in candidates:
            try:
                verified = self.verify(checkpoint)
                model = PPO.load(
                    str(checkpoint / "ppo_model.zip"), env=vector_env, device=self.config.device
                )
                if int(model.num_timesteps) != int(verified["trainer_state"].global_step):
                    raise RuntimeError("model.num_timesteps does not match trainer_state")
                rng_state = torch.load(
                    checkpoint / "rng_state.pt", map_location="cpu", weights_only=False
                )
                restore_rng_state(rng_state)
                return model, verified["trainer_state"], verified["league"], checkpoint
            except Exception as exc:
                failures.append(f"{checkpoint.name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("No valid checkpoint could be restored; " + " | ".join(failures))
