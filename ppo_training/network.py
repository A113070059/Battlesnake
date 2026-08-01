from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from .env import make_env_factory


class BlackoutCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, config_dict: dict[str, Any]):
        config_channels = tuple(config_dict["conv_channels"])
        hidden_size = int(config_dict["hidden_size"])
        kernel_size = int(config_dict["kernel_size"])
        padding = int(config_dict["padding"])
        pool_size = int(config_dict["pool_size"])
        super().__init__(observation_space, features_dim=hidden_size)
        input_channels = int(observation_space.shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, config_channels[0], kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(config_channels[0], config_channels[1], kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(config_channels[1], config_channels[2], kernel_size, padding=padding),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros((1, *observation_space.shape), dtype=torch.float32)
            flattened = int(self.cnn(sample).shape[1])
        self.projection = nn.Sequential(nn.Linear(flattened, hidden_size), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.projection(self.cnn(observations.float()))


def make_vector_env(config, league_manifest_path: str):
    factories = [
        make_env_factory(config.to_dict(), league_manifest_path, worker_index)
        for worker_index in range(config.effective_n_envs)
    ]
    if config.effective_n_envs == 1:
        vector_env = DummyVecEnv(factories)
    else:
        start_method = config.subprocess_start_method
        if os.name == "nt" and start_method in {"fork", "forkserver"}:
            start_method = "spawn"
        vector_env = SubprocVecEnv(factories, start_method=start_method)
    return VecMonitor(vector_env)


def build_ppo(config, vector_env, tensorboard_dir: str | Path | None = None) -> PPO:
    extractor_config = {
        "conv_channels": list(config.conv_channels),
        "hidden_size": config.hidden_size,
        "kernel_size": config.kernel_size,
        "padding": config.padding,
        "pool_size": config.pool_size,
    }
    policy_kwargs = {
        "features_extractor_class": BlackoutCNN,
        "features_extractor_kwargs": {"config_dict": extractor_config},
        "net_arch": {"pi": [], "vf": []},
        "activation_fn": nn.ReLU,
        "normalize_images": False,
        "share_features_extractor": True,
    }
    model = PPO(
        "CnnPolicy",
        vector_env,
        learning_rate=config.learning_rate,
        n_steps=config.effective_n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.entropy_coefficient(0),
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        normalize_advantage=config.normalize_advantage,
        target_kl=config.target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_dir) if tensorboard_dir else None,
        device=config.device,
        seed=config.base_seed,
        verbose=0,
    )
    parameter_count = sum(parameter.numel() for parameter in model.policy.parameters())
    if parameter_count != config.expected_parameter_count:
        raise RuntimeError(
            f"Policy parameter count {parameter_count:,} != expected "
            f"{config.expected_parameter_count:,}.\n{model.policy}"
        )
    return model


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.policy.parameters() if parameter.requires_grad)
