from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ACTION_DELTAS = {
    0: (0, 1),   # up
    1: (1, 0),   # right
    2: (0, -1),  # down
    3: (-1, 0),  # left
}


@dataclass(slots=True)
class RewardBreakdown:
    terminal: float = 0.0
    survival: float = 0.0
    elimination: float = 0.0
    food: float = 0.0

    @property
    def total(self) -> float:
        return self.terminal + self.survival + self.elimination + self.food

    def to_dict(self) -> dict[str, float]:
        result = asdict(self)
        result["total"] = self.total
        return result


class RewardTracker:
    def __init__(self, config):
        self.config = config
        self.survived_turns = 0
        self.elimination_reward_total = 0.0
        self.food_reward_total = 0.0
        self.food_count = 0
        self.elimination_count = 0
        self.cumulative = RewardBreakdown()

    def step(
        self,
        before_state,
        before_alive: set[int],
        after_alive: set[int],
        learner: int,
        learner_action: int,
        win: bool,
        death: bool,
    ) -> RewardBreakdown:
        result = RewardBreakdown()
        learner_survived = learner in after_alive

        if learner_survived:
            self.survived_turns += 1
            if self.survived_turns <= self.config.reward_survival_turn_cap:
                result.survival = self.config.reward_survival_per_turn

        eliminated_opponents = len(
            (before_alive - after_alive) - {learner}
        )
        if learner_survived and eliminated_opponents:
            remaining_cap = max(
                0.0,
                self.config.reward_elimination_cap - self.elimination_reward_total,
            )
            result.elimination = min(
                remaining_cap,
                eliminated_opponents * self.config.reward_elimination,
            )
            self.elimination_reward_total += result.elimination
            self.elimination_count += eliminated_opponents

        body = [tuple(map(int, point)) for point in before_state.snake_pos[learner]]
        if body:
            delta = ACTION_DELTAS[int(learner_action)]
            target = (body[0][0] + delta[0], body[0][1] + delta[1])
            food_before = {tuple(map(int, point)) for point in before_state.food_pos}
            if learner_survived and target in food_before:
                health = int(before_state.snake_health[learner])
                food_value = (
                    self.config.reward_food_healthy
                    if health > self.config.reward_food_health_threshold
                    else self.config.reward_food_low_health
                )
                remaining_cap = max(
                    0.0, self.config.reward_food_cap - self.food_reward_total
                )
                result.food = min(remaining_cap, food_value)
                if result.food > 0:
                    self.food_count += 1
                    self.food_reward_total += result.food

        if win:
            result.terminal = self.config.reward_win
        elif death:
            result.terminal = self.config.reward_death

        self.cumulative.terminal += result.terminal
        self.cumulative.survival += result.survival
        self.cumulative.elimination += result.elimination
        self.cumulative.food += result.food
        return result

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "reward_total": self.cumulative.total,
            "reward_terminal": self.cumulative.terminal,
            "reward_survival": self.cumulative.survival,
            "reward_elimination": self.cumulative.elimination,
            "reward_food": self.cumulative.food,
            "food_count": self.food_count,
            "elimination_count": self.elimination_count,
            "survived_turns": self.survived_turns,
        }
        return result
