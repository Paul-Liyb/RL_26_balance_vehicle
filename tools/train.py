#!/usr/bin/env python3
"""Train SAC, TD3, or PPO for the standing-balance task."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_balance.config import (
    ACTION_MODES,
    ALGO_DEFAULTS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MODEL_PROFILE,
    DEFAULT_RESIDUAL_SCALE,
    DEFAULT_SEEDS,
    MODEL_PROFILES,
    RESET_PROFILES,
    REWARD_PROFILES,
    SAC_PROFILES,
    TrainConfig,
)
from rl_balance.experiments import train_single_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL policy for the balance-stand task.")
    parser.add_argument("--algo", choices=["sac", "td3", "ppo"], required=True)
    parser.add_argument("--timesteps", type=int, help="Override total timesteps.")
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--train-reset-profile", choices=sorted(RESET_PROFILES.keys()), default="default")
    parser.add_argument("--train-reward-profile", choices=sorted(REWARD_PROFILES.keys()), default="default")
    parser.add_argument("--sac-profile", choices=sorted(SAC_PROFILES.keys()), default="default")
    parser.add_argument("--action-mode", choices=ACTION_MODES, default="direct")
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE)
    parser.add_argument("--model-profile", choices=MODEL_PROFILES, default=DEFAULT_MODEL_PROFILE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timesteps = args.timesteps or int(ALGO_DEFAULTS[args.algo]["train_steps"])
    for seed in args.seeds:
        config = TrainConfig(
            algo=args.algo,
            seed=seed,
            timesteps=timesteps,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            output_dir=args.output_dir,
            device=args.device,
            train_reset_profile=args.train_reset_profile,
            train_reward_profile=args.train_reward_profile,
            sac_profile=args.sac_profile,
            action_mode=args.action_mode,
            residual_scale=args.residual_scale,
            model_profile=args.model_profile,
        )
        train_single_run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
