"""Gymnasium environment for standing-balance experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import lqr_from_matlab

from .config import (
    ACTION_MODES,
    ACTION_SCALE,
    DEFAULT_ACTION_MODE,
    DEFAULT_MAX_STEPS,
    DEFAULT_RESET_PROFILE,
    DEFAULT_REWARD_PROFILE,
    DEFAULT_RESIDUAL_SCALE,
    OBSERVATION_SCALE,
    RESET_PROFILES,
    REWARD_PROFILES,
)
from .policies import LqrPolicy


@dataclass(frozen=True)
class TerminationThresholds:
    body_angle: float = 0.35
    pendulum_angle: float = 0.45
    wheel_speed: float = 20.0


class BalanceStandEnv(gym.Env[np.ndarray, np.ndarray]):
    """Linear standing-balance environment aligned with the current LQR state."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        reset_profile: str = DEFAULT_RESET_PROFILE,
        reward_profile: str = DEFAULT_REWARD_PROFILE,
    ) -> None:
        super().__init__()
        if reset_profile not in RESET_PROFILES:
            raise ValueError(f"Unsupported reset profile: {reset_profile}")
        if reward_profile not in REWARD_PROFILES:
            raise ValueError(f"Unsupported reward profile: {reward_profile}")
        result = lqr_from_matlab.solve_lqr_from_matlab()
        self.G = result.G.astype(np.float64)
        self.H = result.H.astype(np.float64)
        self.obs_scale = OBSERVATION_SCALE.copy()
        self.reward_profile = reward_profile
        self.reward_weights = REWARD_PROFILES[reward_profile].copy()
        self.reset_profile = reset_profile
        self.reset_scale = RESET_PROFILES[reset_profile].copy()
        self.thresholds = TerminationThresholds()
        self.action_scale = ACTION_SCALE
        self.max_steps = int(max_steps)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.state = np.zeros(8, dtype=np.float64)
        self.steps = 0
        self.failed = False
        self.termination_reason = "running"
        if seed is not None:
            self.reset(seed=seed)

    def normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        raw_obs = np.asarray(raw_obs, dtype=np.float64)
        return (raw_obs / self.obs_scale).astype(np.float32)

    def denormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        return obs * self.obs_scale

    def _sample_initial_state(self) -> np.ndarray:
        return self.np_random.uniform(low=-self.reset_scale, high=self.reset_scale).astype(np.float64)

    def _termination_reason(self, state: np.ndarray) -> str | None:
        if not np.all(np.isfinite(state)):
            return "invalid_state"
        if abs(state[2]) > self.thresholds.body_angle:
            return "body_angle_limit"
        if abs(state[3]) > self.thresholds.pendulum_angle:
            return "pendulum_angle_limit"
        if abs(state[4]) > self.thresholds.wheel_speed or abs(state[5]) > self.thresholds.wheel_speed:
            return "wheel_speed_limit"
        return None

    def _reward(self, state: np.ndarray, action: np.ndarray, failed: bool) -> float:
        state_norm = state / self.obs_scale
        reward = 1.0 - float(np.dot(self.reward_weights, state_norm * state_norm)) - 0.02 * float(np.sum(action**2))
        if failed:
            reward -= 25.0
        return reward

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.state = self._sample_initial_state()
        self.steps = 0
        self.failed = False
        self.termination_reason = "running"
        raw_obs = self.state.copy()
        obs = self.normalize_obs(raw_obs)
        return obs, {"raw_obs": raw_obs.astype(np.float32)}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(2)
        clipped_action = np.clip(action, -1.0, 1.0)
        physical_action = self.action_scale * clipped_action
        next_state = self.G @ self.state + self.H @ physical_action
        self.state = next_state.astype(np.float64)
        self.steps += 1

        termination_reason = self._termination_reason(self.state)
        terminated = termination_reason is not None
        truncated = self.steps >= self.max_steps and not terminated
        self.failed = terminated
        self.termination_reason = termination_reason or ("time_limit" if truncated else "running")
        reward = self._reward(self.state, clipped_action, terminated)
        raw_obs = self.state.astype(np.float32)
        obs = self.normalize_obs(raw_obs)
        info = {
            "raw_obs": raw_obs,
            "physical_action": physical_action.astype(np.float32),
            "control_energy": float(np.sum(physical_action**2)),
            "termination_reason": self.termination_reason,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


class ResidualLQREnv(gym.Wrapper):
    """Environment wrapper that adds a bounded RL residual on top of LQR."""

    def __init__(self, env: BalanceStandEnv, residual_scale: float = DEFAULT_RESIDUAL_SCALE) -> None:
        super().__init__(env)
        if residual_scale <= 0.0:
            raise ValueError(f"Residual scale must be positive, got {residual_scale}")
        self.residual_scale = float(residual_scale)
        self.teacher = LqrPolicy()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = env.observation_space

    def _current_obs(self) -> np.ndarray:
        return self.env.normalize_obs(self.env.state)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        residual_action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        teacher_action = np.asarray(self.teacher.predict(self._current_obs()), dtype=np.float64)
        combined_action = np.clip(teacher_action + self.residual_scale * residual_action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(combined_action.astype(np.float32))
        info["teacher_action"] = teacher_action.astype(np.float32)
        info["residual_action"] = residual_action.astype(np.float32)
        info["combined_action"] = combined_action.astype(np.float32)
        info["action_mode"] = "residual_lqr"
        info["residual_scale"] = self.residual_scale
        info["teacher_action_l2"] = float(np.linalg.norm(teacher_action))
        info["residual_action_l2"] = float(np.linalg.norm(residual_action))
        info["combined_action_saturated"] = bool(np.any(np.isclose(np.abs(combined_action), 1.0, atol=1e-6)))
        return obs, reward, terminated, truncated, info


def make_action_env(
    *,
    seed: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    reset_profile: str = DEFAULT_RESET_PROFILE,
    reward_profile: str = DEFAULT_REWARD_PROFILE,
    action_mode: str = DEFAULT_ACTION_MODE,
    residual_scale: float = DEFAULT_RESIDUAL_SCALE,
) -> gym.Env:
    if action_mode not in ACTION_MODES:
        raise ValueError(f"Unsupported action mode: {action_mode}")
    env = BalanceStandEnv(
        seed=seed,
        max_steps=max_steps,
        reset_profile=reset_profile,
        reward_profile=reward_profile,
    )
    if action_mode == "residual_lqr":
        return ResidualLQREnv(env, residual_scale=residual_scale)
    return env
