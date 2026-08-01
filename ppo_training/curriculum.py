from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class CurriculumState:
    phase2_gate_passed: bool = False
    phase3_gate_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "CurriculumState":
        return cls(**(value or {}))


def phase_index(config, global_step: int, state: CurriculumState) -> int:
    eligible = [
        index
        for index, phase in enumerate(config.curriculum)
        if global_step >= int(phase["start"])
    ]
    index = max(eligible, default=0)
    if index >= 2 and not state.phase2_gate_passed:
        return 1
    if index >= 3 and not state.phase3_gate_passed:
        return 2
    return index


def current_phase(config, global_step: int, state: CurriculumState) -> dict[str, Any]:
    return config.curriculum[phase_index(config, global_step, state)]


def sample_requested_lineup(
    config,
    global_step: int,
    state: CurriculumState,
    rng: np.random.Generator,
) -> str:
    phase = current_phase(config, global_step, state)
    lineups = list(phase["mix"])
    probabilities = np.asarray([phase["mix"][key] for key in lineups], dtype=float)
    probabilities /= probabilities.sum()
    return str(rng.choice(lineups, p=probabilities))


def quick_suites(config, global_step: int, state: CurriculumState) -> list[str]:
    index = phase_index(config, global_step, state)
    if index == 0:
        return ["RRR"]
    if index == 1:
        return ["RRR", "HRR", "HHR"]
    if index == 2:
        return ["RRR", "HRR", "HHR", "HHH"]
    return ["RRR", "HHH", "PHR", "PPH", "PPP"]


def full_suites() -> list[str]:
    return ["RRR", "HHH", "PHR", "PPH", "PPP"]


def session_end_suites() -> list[str]:
    return ["RRR", "HRR", "HHR", "HHH", "PHR", "PPH", "PPP"]


def update_gates(
    config,
    global_step: int,
    state: CurriculumState,
    suite_win_rates: dict[str, float],
    suite_game_counts: dict[str, int],
) -> list[str]:
    messages: list[str] = []
    if (
        not state.phase2_gate_passed
        and global_step >= config.phase2_gate_step
        and suite_game_counts.get(config.phase2_gate_suite, 0) >= config.phase2_gate_games
        and suite_win_rates.get(config.phase2_gate_suite, -1.0)
        >= config.phase2_gate_win_rate
    ):
        state.phase2_gate_passed = True
        messages.append("phase2_gate_passed")
    if (
        not state.phase3_gate_passed
        and global_step >= config.phase3_gate_step
        and suite_game_counts.get(config.phase3_gate_suite, 0) >= config.phase3_gate_games
        and suite_win_rates.get(config.phase3_gate_suite, -1.0)
        >= config.phase3_gate_win_rate
    ):
        state.phase3_gate_passed = True
        messages.append("phase3_gate_passed")
    return messages
