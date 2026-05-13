"""Training and evaluation utilities for the RL balance experiments."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .config import (
    ACTION_SCALE,
    ALGO_DEFAULTS,
    DEFAULT_ACTION_MODE,
    DEFAULT_RESET_PROFILE,
    DEFAULT_REWARD_PROFILE,
    DEFAULT_RESIDUAL_SCALE,
    DEFAULT_SAC_PROFILE,
    SAC_PROFILES,
    TrainConfig,
)
from .env import BalanceStandEnv, make_action_env
from .policies import LqrPolicy, Policy, SB3Policy


@dataclass
class EpisodeMetrics:
    success_rate: float
    mean_episode_length: float
    mean_return: float
    rms_theta_1: float
    rms_theta_2: float
    mean_control_energy: float
    wall_clock_train_time: float


@dataclass
class EpisodeTrace:
    algorithm: str
    seed: int
    step: int
    theta_1: float
    theta_2: float
    u_l: float
    u_r: float


ALGO_CLASSES = {"sac": SAC, "td3": TD3, "ppo": PPO}


def make_env(
    seed: int | None = None,
    max_steps: int | None = None,
    reset_profile: str = DEFAULT_RESET_PROFILE,
    reward_profile: str = DEFAULT_REWARD_PROFILE,
    action_mode: str = DEFAULT_ACTION_MODE,
    residual_scale: float = DEFAULT_RESIDUAL_SCALE,
) -> Any:
    kwargs: dict[str, Any] = {}
    if seed is not None:
        kwargs["seed"] = seed
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    kwargs["reset_profile"] = reset_profile
    kwargs["reward_profile"] = reward_profile
    kwargs["action_mode"] = action_mode
    kwargs["residual_scale"] = residual_scale
    return make_action_env(**kwargs)


def build_model(
    algo: str,
    env: BalanceStandEnv,
    seed: int,
    device: str = "cpu",
    sac_profile: str = DEFAULT_SAC_PROFILE,
):
    algo = algo.lower()
    if algo not in ALGO_CLASSES:
        raise ValueError(f"Unsupported algorithm: {algo}")
    if algo == "sac":
        if sac_profile not in SAC_PROFILES:
            raise ValueError(f"Unsupported SAC profile: {sac_profile}")
        kwargs = SAC_PROFILES[sac_profile].copy()
    else:
        kwargs = ALGO_DEFAULTS[algo].copy()
    kwargs.pop("train_steps", None)
    kwargs["policy_kwargs"] = {"net_arch": [128, 128], "activation_fn": torch.nn.ReLU}
    kwargs["device"] = device
    kwargs["seed"] = seed
    return ALGO_CLASSES[algo]("MlpPolicy", Monitor(env), verbose=0, **kwargs)


def evaluate_policy(
    env: BalanceStandEnv,
    policy: Policy,
    episodes: int,
    *,
    seed_offset: int = 0,
    algorithm: str = "policy",
    seed: int = -1,
    wall_clock_train_time: float = 0.0,
    capture_trace: bool = False,
) -> tuple[EpisodeMetrics, list[EpisodeTrace]]:
    returns: list[float] = []
    lengths: list[int] = []
    theta_1_values: list[float] = []
    theta_2_values: list[float] = []
    control_energies: list[float] = []
    successes = 0
    trace_rows: list[EpisodeTrace] = []
    captured = False

    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed_offset + episode_idx)
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        done = False
        episode_return = 0.0
        episode_length = 0
        episode_trace: list[EpisodeTrace] = []
        while not done:
            action = np.asarray(policy.predict(obs), dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            raw = info["raw_obs"]
            theta_1_values.append(float(raw[2]))
            theta_2_values.append(float(raw[3]))
            control_energies.append(float(np.sum(info["physical_action"] ** 2)))
            episode_return += reward
            episode_length += 1
            if capture_trace and not captured:
                episode_trace.append(
                    EpisodeTrace(
                        algorithm=algorithm,
                        seed=seed,
                        step=episode_length,
                        theta_1=float(raw[2]),
                        theta_2=float(raw[3]),
                        u_l=float(info["physical_action"][0]),
                        u_r=float(info["physical_action"][1]),
                    )
                )
            obs = next_obs
            done = terminated or truncated
        if not base_env.failed and base_env.steps >= base_env.max_steps:
            successes += 1
        returns.append(float(episode_return))
        lengths.append(episode_length)
        if capture_trace and not captured:
            trace_rows = episode_trace
            captured = True

    metrics = EpisodeMetrics(
        success_rate=float(successes / episodes),
        mean_episode_length=float(np.mean(lengths)),
        mean_return=float(np.mean(returns)),
        rms_theta_1=float(np.sqrt(np.mean(np.square(theta_1_values)))),
        rms_theta_2=float(np.sqrt(np.mean(np.square(theta_2_values)))),
        mean_control_energy=float(np.mean(control_energies)),
        wall_clock_train_time=float(wall_clock_train_time),
    )
    return metrics, trace_rows


class EvalAndSaveCallback(BaseCallback):
    """Periodic evaluation that logs metrics and saves the best model."""

    def __init__(
        self,
        algo: str,
        seed: int,
        eval_freq: int,
        eval_episodes: int,
        output_dir: Path,
        start_time: float,
        *,
        action_mode: str = DEFAULT_ACTION_MODE,
        residual_scale: float = DEFAULT_RESIDUAL_SCALE,
    ) -> None:
        super().__init__(verbose=0)
        self.algo = algo
        self.seed = seed
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.output_dir = output_dir
        self.action_mode = action_mode
        self.residual_scale = residual_scale
        self.metrics_path = output_dir / "metrics.csv"
        self.best_path = output_dir / "best_model.zip"
        self.start_time = start_time
        self.best_mean_return = float("-inf")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timesteps",
                    "mean_return",
                    "success_rate",
                    "mean_episode_length",
                    "rms_theta_1",
                    "rms_theta_2",
                    "mean_control_energy",
                    "wall_clock_train_time",
                ],
            )
            writer.writeheader()

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        eval_env = make_env(seed=self.seed, action_mode=self.action_mode, residual_scale=self.residual_scale)
        metrics, _ = evaluate_policy(
            eval_env,
            SB3Policy(self.model),
            self.eval_episodes,
            seed_offset=1000 + self.seed * 100,
            wall_clock_train_time=time.time() - self.start_time,
        )
        row = {
            "timesteps": self.num_timesteps,
            "mean_return": metrics.mean_return,
            "success_rate": metrics.success_rate,
            "mean_episode_length": metrics.mean_episode_length,
            "rms_theta_1": metrics.rms_theta_1,
            "rms_theta_2": metrics.rms_theta_2,
            "mean_control_energy": metrics.mean_control_energy,
            "wall_clock_train_time": metrics.wall_clock_train_time,
        }
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writerow(row)
        if metrics.mean_return > self.best_mean_return:
            self.best_mean_return = metrics.mean_return
            self.model.save(str(self.best_path))
        eval_env.close()
        return True


def train_single_run(config: TrainConfig) -> dict[str, Any]:
    output_dir = config.output_dir / config.algo / f"seed_{config.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(
        seed=config.seed,
        reset_profile=config.train_reset_profile,
        reward_profile=config.train_reward_profile,
        action_mode=config.action_mode,
        residual_scale=config.residual_scale,
    )
    model = build_model(
        config.algo,
        env,
        seed=config.seed,
        device=config.device,
        sac_profile=config.sac_profile,
    )
    start_time = time.time()
    callback = EvalAndSaveCallback(
        algo=config.algo,
        seed=config.seed,
        eval_freq=config.eval_freq,
        eval_episodes=config.eval_episodes,
        output_dir=output_dir,
        start_time=start_time,
        action_mode=config.action_mode,
        residual_scale=config.residual_scale,
    )
    model.learn(total_timesteps=config.timesteps, callback=callback, progress_bar=False)
    elapsed = time.time() - start_time
    final_model_path = output_dir / "final_model.zip"
    model.save(str(final_model_path))
    if not callback.best_path.exists():
        model.save(str(callback.best_path))
    metadata = {
        "algo": config.algo,
        "seed": config.seed,
        "timesteps": config.timesteps,
        "eval_freq": config.eval_freq,
        "eval_episodes": config.eval_episodes,
        "wall_clock_train_time": elapsed,
        "best_mean_return": callback.best_mean_return,
        "action_scale": ACTION_SCALE,
        "train_reset_profile": config.train_reset_profile,
        "train_reward_profile": config.train_reward_profile,
        "sac_profile": config.sac_profile,
        "action_mode": config.action_mode,
        "residual_scale": config.residual_scale,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    env.close()
    return metadata


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_training_curves(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in input_dir.glob("*/*/metrics.csv"):
        algo = metrics_path.parent.parent.name
        seed = int(metrics_path.parent.name.split("_")[-1])
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "algorithm": algo,
                        "seed": seed,
                        "timesteps": int(row["timesteps"]),
                        "mean_return": float(row["mean_return"]),
                        "success_rate": float(row["success_rate"]),
                    }
                )
    return rows


def evaluate_saved_runs(input_dir: Path, episodes: int) -> tuple[list[dict[str, Any]], list[EpisodeTrace]]:
    rows: list[dict[str, Any]] = []
    best_trace: list[EpisodeTrace] = []
    best_success = float("-inf")

    baseline_env = make_env(seed=0)
    baseline_metrics, baseline_trace = evaluate_policy(
        baseline_env,
        LqrPolicy(),
        episodes,
        seed_offset=0,
        algorithm="lqr",
        seed=-1,
        wall_clock_train_time=0.0,
        capture_trace=True,
    )
    rows.append({"algorithm": "lqr", "seed": -1, **asdict(baseline_metrics)})
    best_trace = baseline_trace
    best_success = baseline_metrics.success_rate
    baseline_env.close()

    for best_model_path in input_dir.glob("*/*/best_model.zip"):
        algo = best_model_path.parent.parent.name
        seed = int(best_model_path.parent.name.split("_")[-1])
        metadata_path = best_model_path.parent / "run_metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        action_mode = str(metadata.get("action_mode", DEFAULT_ACTION_MODE))
        residual_scale = float(metadata.get("residual_scale", DEFAULT_RESIDUAL_SCALE))
        env = make_env(seed=seed, action_mode=action_mode, residual_scale=residual_scale)
        metrics, trace = evaluate_policy(
            env,
            SB3Policy.load(algo, best_model_path),
            episodes,
            seed_offset=5000 + seed * 100,
            algorithm=algo,
            seed=seed,
            wall_clock_train_time=float(metadata.get("wall_clock_train_time", 0.0)),
            capture_trace=True,
        )
        row = {"algorithm": algo, "seed": seed, **asdict(metrics)}
        rows.append(row)
        if metrics.success_rate >= best_success:
            best_success = metrics.success_rate
            best_trace = trace
        env.close()
    return rows, best_trace


def aggregate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["algorithm"]), []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for algo, algo_rows in grouped.items():
        aggregate_rows.append(
            {
                "algorithm": algo,
                "runs": len(algo_rows),
                "success_rate": float(np.mean([float(r["success_rate"]) for r in algo_rows])),
                "mean_episode_length": float(np.mean([float(r["mean_episode_length"]) for r in algo_rows])),
                "mean_return": float(np.mean([float(r["mean_return"]) for r in algo_rows])),
                "rms_theta_1": float(np.mean([float(r["rms_theta_1"]) for r in algo_rows])),
                "rms_theta_2": float(np.mean([float(r["rms_theta_2"]) for r in algo_rows])),
                "mean_control_energy": float(np.mean([float(r["mean_control_energy"]) for r in algo_rows])),
                "wall_clock_train_time": float(np.mean([float(r["wall_clock_train_time"]) for r in algo_rows])),
            }
        )
    return aggregate_rows
