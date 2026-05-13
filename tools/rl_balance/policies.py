"""Controllers used in simulation and live deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

import lqr_from_matlab

from .config import ACTION_SCALE, OBSERVATION_SCALE


class Policy(Protocol):
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return normalized action in [-1, 1]."""


@dataclass
class LqrPolicy:
    """LQR baseline expressed in the normalized action interface."""

    action_scale: float = ACTION_SCALE

    def __post_init__(self) -> None:
        self.K = lqr_from_matlab.solve_lqr_from_matlab().K.astype(np.float64)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        raw_obs = np.asarray(obs, dtype=np.float64) * OBSERVATION_SCALE
        u = -(self.K @ raw_obs)
        action = np.clip(u / self.action_scale, -1.0, 1.0)
        return action.astype(np.float32)


MODEL_CLASSES = {"sac": SAC, "td3": TD3, "ppo": PPO}


@dataclass
class SB3Policy:
    model: object

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    @classmethod
    def load(cls, algo: str, model_path: Path | str) -> "SB3Policy":
        algo = algo.lower()
        if algo not in MODEL_CLASSES:
            raise ValueError(f"Unsupported algorithm: {algo}")
        model = MODEL_CLASSES[algo].load(str(model_path), device="cpu")
        return cls(model=model)
