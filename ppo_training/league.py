from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class SnapshotRecord:
    snapshot_id: str
    step: int
    model_path: str
    config_hash: str
    metrics: dict[str, float] = field(default_factory=dict)
    fingerprint: list[float | None] = field(default_factory=list)
    valid: bool = True
    invalid_reason: str = ""
    category: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SnapshotRecord":
        return cls(**value)

    def overall_score(self) -> float:
        keys = ("WR_RRR", "WR_HHH", "WR_PHR", "WR_PPH", "WR_PPP")
        if all(key in self.metrics for key in keys):
            return (
                0.10 * self.metrics["WR_RRR"]
                + 0.20 * self.metrics["WR_HHH"]
                + 0.20 * self.metrics["WR_PHR"]
                + 0.25 * self.metrics["WR_PPH"]
                + 0.25 * self.metrics["WR_PPP"]
            )
        return 0.40 * self.metrics.get("WR_RRR", 0.0) + 0.60 * self.metrics.get(
            "WR_HHH", 0.0
        )

    def hard_score(self) -> float:
        if "WR_PPH" in self.metrics and "WR_PPP" in self.metrics:
            return 0.50 * self.metrics["WR_PPH"] + 0.50 * self.metrics["WR_PPP"]
        return -math.inf


@dataclass(slots=True)
class LeagueManifest:
    schema_version: int = 1
    snapshots: list[SnapshotRecord] = field(default_factory=list)
    best_overall_id: str | None = None
    best_hard_id: str | None = None
    eval_suite_version: str = "bootstrap-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshots": [record.to_dict() for record in self.snapshots],
            "best_overall_id": self.best_overall_id,
            "best_hard_id": self.best_hard_id,
            "eval_suite_version": self.eval_suite_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LeagueManifest":
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            snapshots=[SnapshotRecord.from_dict(item) for item in value.get("snapshots", [])],
            best_overall_id=value.get("best_overall_id"),
            best_hard_id=value.get("best_hard_id"),
            eval_suite_version=value.get("eval_suite_version", "bootstrap-v1"),
        )


class LeagueManager:
    def __init__(self, config, manifest_path: str | Path):
        self.config = config
        self.manifest_path = Path(manifest_path)
        if self.manifest_path.exists():
            self.manifest = LeagueManifest.from_dict(
                json.loads(self.manifest_path.read_text(encoding="utf-8"))
            )
        else:
            self.manifest = LeagueManifest()

    def save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.manifest_path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(self.manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        temp.replace(self.manifest_path)

    def reload(self) -> None:
        if self.manifest_path.exists():
            self.manifest = LeagueManifest.from_dict(
                json.loads(self.manifest_path.read_text(encoding="utf-8"))
            )

    def valid_records(self) -> list[SnapshotRecord]:
        return [
            record
            for record in self.manifest.snapshots
            if record.valid
            and record.config_hash == self.config.semantic_hash()
            and Path(record.model_path).exists()
        ]

    def can_evaluate(self, lineup: str) -> tuple[bool, str]:
        required = lineup.count("P")
        available = len(self.valid_records())
        if available < required:
            return False, f"needs {required} unique PPO snapshots, only {available} available"
        return True, ""

    @staticmethod
    def _fingerprint_distance(left: SnapshotRecord, right: SnapshotRecord) -> float:
        pairs = [
            (a, b)
            for a, b in zip(left.fingerprint, right.fingerprint)
            if a is not None and b is not None
        ]
        if not pairs:
            return 0.0
        return float(math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in pairs)))

    def refresh_categories(self) -> None:
        valid = self.valid_records()
        previous_best_id = self.manifest.best_overall_id
        for record in self.manifest.snapshots:
            record.category = ""
        if not valid:
            self.manifest.best_overall_id = None
            self.manifest.best_hard_id = None
            self.save()
            return

        candidate = max(valid, key=lambda record: (record.overall_score(), record.step))
        previous_best = next(
            (record for record in valid if record.snapshot_id == previous_best_id), None
        )
        best_overall = candidate
        if previous_best is not None and candidate.snapshot_id != previous_best.snapshot_id:
            enough_gain = (
                candidate.overall_score()
                >= previous_best.overall_score() + self.config.promotion_overall_delta
            )
            rrr_safe = (
                "WR_RRR" not in previous_best.metrics
                or "WR_RRR" not in candidate.metrics
                or candidate.metrics["WR_RRR"]
                >= previous_best.metrics["WR_RRR"]
                - self.config.promotion_rrr_max_regression
            )
            hhh_safe = (
                "WR_HHH" not in previous_best.metrics
                or "WR_HHH" not in candidate.metrics
                or candidate.metrics["WR_HHH"]
                >= previous_best.metrics["WR_HHH"]
                - self.config.promotion_hhh_max_regression
            )
            if not (enough_gain and rrr_safe and hhh_safe):
                best_overall = previous_best
        best_overall.category = "best"
        selected = [best_overall]
        hard_candidates = [record for record in valid if record.snapshot_id != best_overall.snapshot_id]
        best_hard = max(hard_candidates, key=lambda record: (record.hard_score(), record.step), default=None)
        if best_hard is not None and math.isfinite(best_hard.hard_score()):
            best_hard.category = "best"
            selected.append(best_hard)
        else:
            best_hard = None
        self.manifest.best_overall_id = best_overall.snapshot_id
        self.manifest.best_hard_id = best_hard.snapshot_id if best_hard else None

        remaining = [record for record in valid if record not in selected]
        recent = sorted(remaining, key=lambda record: record.step, reverse=True)[
            : self.config.league_recent_slots
        ]
        for record in recent:
            record.category = "recent"
        selected.extend(recent)

        remaining = [
            record
            for record in valid
            if record not in selected
            and record.overall_score()
            >= best_overall.overall_score() - self.config.diverse_score_gap
        ]
        diverse: list[SnapshotRecord] = []
        while remaining and len(diverse) < self.config.league_diverse_slots:
            candidate = max(
                remaining,
                key=lambda record: min(
                    (self._fingerprint_distance(record, chosen) for chosen in selected + diverse),
                    default=0.0,
                ),
            )
            candidate.category = "diverse"
            diverse.append(candidate)
            remaining.remove(candidate)

        self.save()

    def add_snapshot(self, record: SnapshotRecord) -> None:
        self.manifest.snapshots = [
            existing
            for existing in self.manifest.snapshots
            if existing.snapshot_id != record.snapshot_id
        ]
        self.manifest.snapshots.append(record)
        self.refresh_categories()

    def _records_by_category(self) -> dict[str, list[SnapshotRecord]]:
        result = {"best": [], "recent": [], "diverse": []}
        for record in self.valid_records():
            if record.category in result:
                result[record.category].append(record)
        # During bootstrap, every valid unclassified model is at least recent.
        if not any(result.values()):
            result["recent"] = sorted(self.valid_records(), key=lambda item: item.step, reverse=True)
        return result

    def sample_records(
        self,
        count: int,
        rng: np.random.Generator,
        unique_required: bool = False,
    ) -> list[SnapshotRecord]:
        if count <= 0:
            return []
        records_by_category = self._records_by_category()
        selected: list[SnapshotRecord] = []
        for _ in range(count):
            available_categories = [
                category
                for category, records in records_by_category.items()
                if any(record not in selected for record in records)
            ]
            if not available_categories:
                if unique_required:
                    break
                available_categories = [key for key, values in records_by_category.items() if values]
            if not available_categories:
                break
            weights = np.asarray(
                [self.config.league_category_weights[key] for key in available_categories],
                dtype=float,
            )
            weights /= weights.sum()
            category = str(rng.choice(available_categories, p=weights))
            candidates = [
                record for record in records_by_category[category] if record not in selected
            ]
            if not candidates:
                candidates = records_by_category[category]
            selected.append(candidates[int(rng.integers(len(candidates)))])
        return selected

    def resolve_training_lineup(
        self,
        requested: str,
        rng: np.random.Generator,
        unique_required: bool = False,
    ) -> tuple[str, list[SnapshotRecord | None]]:
        p_count = requested.count("P")
        sampled = self.sample_records(p_count, rng, unique_required=unique_required)
        sampled_iter = iter(sampled)
        actual_chars: list[str] = []
        assigned: list[SnapshotRecord | None] = []
        for char in requested:
            if char != "P":
                actual_chars.append(char)
                assigned.append(None)
                continue
            record = next(sampled_iter, None)
            if record is None:
                actual_chars.append("H")
                assigned.append(None)
            else:
                actual_chars.append("P")
                assigned.append(record)
        return "".join(actual_chars), assigned
