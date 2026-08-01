from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from battlesnake_types import BaseAgent
from hungry_agent import HungryAgent
from random_agent import RandomAgent

from .hisss_compat import DIRECTION_TO_HISSS, build_game_state


def make_baseline_agent(code: str) -> BaseAgent:
    if code == "R":
        return RandomAgent()
    if code == "H":
        return HungryAgent()
    raise ValueError(f"Not a baseline opponent code: {code}")


class BaselineController:
    def __init__(self, code: str, seat: int, agent: BaseAgent):
        self.code = code
        self.seat = seat
        self.agent = agent

    def start(self, env, game_id, announced_food, food_spawn_turns) -> None:
        state = build_game_state(
            env, self.seat, game_id, announced_food, food_spawn_turns
        )
        self.agent.start(state)

    def act(self, env, game_id, announced_food, food_spawn_turns) -> int:
        state = build_game_state(
            env,
            self.seat,
            game_id,
            announced_food,
            food_spawn_turns,
            include_eliminated=True,
        )
        move = self.agent.move(state)
        return int(DIRECTION_TO_HISSS[move.move])

    def end(self, env, game_id, announced_food, food_spawn_turns) -> None:
        state = build_game_state(
            env,
            self.seat,
            game_id,
            announced_food,
            food_spawn_turns,
            include_eliminated=True,
        )
        self.agent.end(state)


class FrozenPolicyCache:
    """Small LRU cache so workers do not retain all League optimizers forever."""

    def __init__(self, max_size: int, device: str = "cpu"):
        self.max_size = max(1, int(max_size))
        self.device = device
        self._models: OrderedDict[str, Any] = OrderedDict()

    def _load(self, model_path: str):
        from stable_baselines3 import PPO

        path = str(Path(model_path))
        if path in self._models:
            model = self._models.pop(path)
            self._models[path] = model
            return model
        model = PPO.load(path, device=self.device)
        self._models[path] = model
        while len(self._models) > self.max_size:
            self._models.popitem(last=False)
        return model

    def predict(self, model_path: str, observation: np.ndarray) -> int:
        model = self._load(model_path)
        action, _ = model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())

    def clear(self) -> None:
        self._models.clear()

