"""Shared configuration for the RL balance experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

OBSERVATION_SCALE = np.array([0.5, 0.5, 0.2, 0.25, 12.0, 12.0, 12.0, 12.0], dtype=np.float64)
ACTION_SCALE = 6000.0
ACTION_MODES = ("direct", "residual_lqr")
DEFAULT_ACTION_MODE = "direct"
DEFAULT_RESIDUAL_SCALE = 0.15
RESET_PROFILES = {
    "default": np.array([0.05, 0.05, 0.12, 0.12, 0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    "narrow": np.array([0.03, 0.03, 0.06, 0.06, 0.3, 0.3, 0.3, 0.3], dtype=np.float64),
    "posture_focus": np.array([0.05, 0.05, 0.06, 0.06, 0.5, 0.5, 0.3, 0.3], dtype=np.float64),
}
DEFAULT_RESET_PROFILE = "default"
REWARD_PROFILES = {
    "default": np.array([0.05, 0.05, 0.35, 0.45, 0.05, 0.05, 0.10, 0.10], dtype=np.float64),
    "posture_focus": np.array([0.03, 0.03, 0.42, 0.52, 0.03, 0.03, 0.07, 0.07], dtype=np.float64),
}
DEFAULT_REWARD_PROFILE = "default"
REWARD_WEIGHTS = REWARD_PROFILES[DEFAULT_REWARD_PROFILE]
SAC_PROFILES = {
    "default": {
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "batch_size": 256,
        "buffer_size": 200000,
        "learning_starts": 5000,
        "tau": 0.005,
        "train_steps": 300000,
    },
    "batch128": {
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "batch_size": 128,
        "buffer_size": 200000,
        "learning_starts": 5000,
        "tau": 0.005,
        "train_steps": 300000,
    },
    "batch128_lr1e4": {
        "learning_rate": 1e-4,
        "gamma": 0.995,
        "batch_size": 128,
        "buffer_size": 200000,
        "learning_starts": 5000,
        "tau": 0.005,
        "train_steps": 300000,
    },
    "batch128_gamma998": {
        "learning_rate": 3e-4,
        "gamma": 0.998,
        "batch_size": 128,
        "buffer_size": 200000,
        "learning_starts": 5000,
        "tau": 0.005,
        "train_steps": 300000,
    },
}
DEFAULT_SAC_PROFILE = "default"
CONTROL_FREQUENCY_HZ = 100.0
DEFAULT_MAX_STEPS = 1000
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass(frozen=True)
class TrainConfig:
    algo: str
    seed: int
    timesteps: int
    eval_freq: int
    eval_episodes: int
    output_dir: Path
    device: str = "cpu"
    train_reset_profile: str = DEFAULT_RESET_PROFILE
    train_reward_profile: str = DEFAULT_REWARD_PROFILE
    sac_profile: str = DEFAULT_SAC_PROFILE
    action_mode: str = DEFAULT_ACTION_MODE
    residual_scale: float = DEFAULT_RESIDUAL_SCALE


ALGO_DEFAULTS = {
    "sac": SAC_PROFILES[DEFAULT_SAC_PROFILE],
    "td3": {
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "batch_size": 256,
        "buffer_size": 200000,
        "learning_starts": 5000,
        "tau": 0.005,
        "train_steps": 300000,
    },
    "ppo": {
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "n_steps": 2048,
        "batch_size": 256,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "train_steps": 500000,
    },
}
