from __future__ import annotations

import math
from typing import Iterable

import numpy as np

import hisss
from battlesnake_types import Direction, GameState


HISSS_TO_DIRECTION = {
    int(hisss.UP): Direction.UP,
    int(hisss.RIGHT): Direction.RIGHT,
    int(hisss.DOWN): Direction.DOWN,
    int(hisss.LEFT): Direction.LEFT,
}
DIRECTION_TO_HISSS = {value: key for key, value in HISSS_TO_DIRECTION.items()}


def set_hisss_seed(seed: int) -> None:
    """Seed the process-global Hisss C++ RNG."""
    from hisss.cpp.lib import CPP_LIB

    CPP_LIB.lib.set_seed(int(seed) & 0x7FFFFFFF)


def make_restricted_config(config):
    game_config = hisss.restricted_standard_config()
    game_config.w = int(config.board_width)
    game_config.h = int(config.board_height)
    game_config.num_players = int(config.num_players)
    game_config.view_radius = int(config.view_radius)
    game_config.min_food = int(config.minimum_food)
    game_config.food_spawn_chance = int(config.food_spawn_chance)
    game_config.all_actions_legal = True
    return game_config


def symmetry_action_maps(symmetry: int, num_actions: int = 4) -> tuple[dict[int, int], dict[int, int]]:
    sym_rot = symmetry % 8
    flip = sym_rot % 2 == 1
    num_rot = math.floor(sym_rot / 2)
    action_offset = -num_rot
    perm: dict[int, int] = {}
    inverse: dict[int, int] = {}
    for action in range(num_actions):
        transformed = (action + action_offset) % num_actions
        if flip:
            if transformed == 2:
                transformed = 0
            elif transformed == 0:
                transformed = 2
        perm[action] = transformed
        inverse[transformed] = action
    return perm, inverse


def apply_spatial_symmetry(array: np.ndarray, symmetry: int) -> np.ndarray:
    """Apply the same spatial transformation used by Hisss to HWC or HW arrays."""
    sym_rot = symmetry % 8
    flip = sym_rot % 2 == 1
    num_rot = math.floor(sym_rot / 2)
    result = np.rot90(array, k=num_rot, axes=(0, 1))
    if flip:
        result = np.flip(result, axis=1)
    return np.ascontiguousarray(result)


def get_observations_compat(env, symmetry: int = 0) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """Hisss 1.2 get_obs with the eliminated-player indexing bug fixed.

    The upstream 1.2 implementation indexes the compact alive-player result by
    permanent player id while applying restricted-view masks.  This function
    uses the compact local index and otherwise follows the upstream logic.
    """
    if env.is_closed:
        raise ValueError("Cannot get observations from a closed Hisss game")
    if env.is_terminal():
        raise ValueError("Cannot get observations from a terminal Hisss game")

    from hisss.game.utils import int_to_perm

    alive = list(env.players_at_turn())
    sym_rot = symmetry % 8
    flip = sym_rot % 2 == 1
    num_rot = math.floor(sym_rot / 2)
    sym_player = math.floor(symmetry / 8)
    enemy_perm = int_to_perm(sym_player, env.num_players - 1)

    observations = [
        env._get_custom_state_encoding(  # Hisss 1.2 has no public single-player equivalent.
            player=player,
            perm=enemy_perm,
            temperatures=None,
            single_temperature=None,
        )
        for player in alive
    ]
    result = np.stack(observations)
    result = np.rot90(result, k=num_rot, axes=(-3, -2))
    if flip:
        result = np.flip(result, axis=-2)
    result = result.copy()

    if env.cfg.view_radius is not None:
        masks: list[np.ndarray] = []
        layers = env.layer_explanation
        for local_index, _player_id in enumerate(alive):
            scaled_distance = result[local_index, :, :, layers["distance_map"]]
            distance = scaled_distance * (env.cfg.w + env.cfg.h - 2)
            current_mask = (distance <= env.cfg.view_radius).astype(np.float32)
            masks.append(current_mask)

            if "current_food" in layers:
                idx = layers["current_food"]
                result[local_index, :, :, idx] *= current_mask

            for enemy_channel in range(1, env.num_players):
                for suffix in ("snake_health", "snake_length", "snake_tail_distance"):
                    key = f"{enemy_channel}_{suffix}"
                    if key in layers:
                        result[local_index, :, :, layers[key]] = 0
                for suffix in (
                    "snake_body",
                    "snake_body_as_one_hot",
                    "snake_head",
                    "snake_tail",
                ):
                    key = f"{enemy_channel}_{suffix}"
                    if key in layers:
                        result[local_index, :, :, layers[key]] *= current_mask

        if env.cfg.ec.include_view_mask:
            result = np.concatenate(
                (result, np.asarray(masks, dtype=np.float32)[:, :, :, None]), axis=-1
            )

    action_perm, inverse_perm = symmetry_action_maps(symmetry, env.cfg.num_actions)
    if env.cfg.ec.flatten:
        result = result.reshape(len(alive), -1)
    return np.ascontiguousarray(result, dtype=np.float32), action_perm, inverse_perm


def _collapse_hidden_body(
    body: Iterable[tuple[int, int]],
    visible,
) -> tuple[list[dict[str, int] | None], bool]:
    collapsed: list[dict[str, int] | None] = []
    hidden_run = False
    has_hidden = False
    for x, y in body:
        if visible(x, y):
            collapsed.append({"x": int(x), "y": int(y)})
            hidden_run = False
        else:
            has_hidden = True
            if not hidden_run:
                collapsed.append(None)
                hidden_run = True
    return collapsed or [None], has_hidden


def build_game_state(
    env,
    player: int,
    game_id: str,
    announced_food: set[tuple[int, int]],
    food_spawn_turns: dict[tuple[int, int], int],
    include_eliminated: bool = False,
) -> GameState:
    """Create the partial GameState used by the existing Random/Hungry agents."""
    cfg = env.cfg
    own_body = [tuple(map(int, point)) for point in env.player_pos(player)]
    own_head = own_body[0] if own_body else None

    def visible(x: int, y: int) -> bool:
        if cfg.view_radius is None or own_head is None:
            return True
        return abs(x - own_head[0]) + abs(y - own_head[1]) <= cfg.view_radius

    food: list[dict[str, int]] = []
    for point in env.food_pos():
        coordinate = tuple(map(int, point))
        if visible(*coordinate) or coordinate in announced_food:
            food.append(
                {
                    "x": coordinate[0],
                    "y": coordinate[1],
                    "spawn_turn": int(food_spawn_turns.get(coordinate, env.turns_played)),
                }
            )

    hazards_array = np.asarray(env.get_hazards())
    hazards = [
        {"x": int(x), "y": int(y)}
        for y, x in np.argwhere(hazards_array)
    ]
    lengths = env.player_lengths()
    healths = env.player_healths()

    def snake_dict(index: int) -> dict | None:
        alive = env.is_player_alive(index)
        if not alive and not include_eliminated:
            return None
        full_body = [tuple(map(int, point)) for point in env.player_pos(index)]
        if index == player:
            body: list[dict[str, int] | None] = [
                {"x": x, "y": y} for x, y in full_body
            ]
            has_hidden = False
        else:
            body, has_hidden = _collapse_hidden_body(full_body, visible)
        head = body[0] if body and body[0] is not None else None
        reported_length = int(lengths[index]) if not has_hidden else len(body)
        return {
            "id": f"snake-{index}",
            "name": f"Snake {index}",
            "length": reported_length,
            "latency": "0",
            "squad": None,
            "health": int(healths[index]) if alive else 0,
            "head": head,
            "body": body,
            "customizations": {
                "color": [255, 0, 0],
                "head": "default",
                "tail": "default",
            },
            "elimination_event": None,
        }

    snakes = [
        value
        for index in range(env.num_players)
        if env.is_player_alive(index)
        if (value := snake_dict(index)) is not None
    ]
    data = {
        "turn": int(env.turns_played),
        "game": {
            "id": game_id,
            "source": "hisss-training",
            "timeout": 500,
            "ruleset": {
                "name": "standard",
                "version": "v1",
                "settings": {
                    "foodSpawnChance": int(cfg.food_spawn_chance),
                    "hazardDamagePerTurn": int(cfg.hazard_damage),
                    "minimumFood": int(cfg.min_food),
                    "viewRadius": cfg.view_radius,
                    "royale": {"shrinkEveryNTurns": int(cfg.shrink_n_turns)},
                    "squad": {
                        "allowBodyCollisions": False,
                        "sharedElimination": False,
                        "sharedHealth": False,
                        "sharedLength": False,
                    },
                },
            },
        },
        "board": {
            "height": int(cfg.h),
            "width": int(cfg.w),
            "food": food,
            "hazards": hazards,
            "snakes": snakes,
        },
        "you": snake_dict(player),
    }
    return GameState.model_validate(data)
