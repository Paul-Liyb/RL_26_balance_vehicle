#!/usr/bin/env python3
"""Run SAC/TD3/DQN comparison artifacts for the team presentation pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rl_balance.config import DEFAULT_MODEL_PROFILE, SIM_MODEL_PROFILES


TEAM_ALGOS = ("sac", "td3", "dqn")


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.check_call(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, plot, and render SAC/TD3/DQN using the shared team-style outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "team_pipeline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-profile", choices=SIM_MODEL_PROFILES, default=DEFAULT_MODEL_PROFILE)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--render-steps", type=int, default=160)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing checkpoints in --output-dir.")
    parser.add_argument("--skip-render", action="store_true", help="Only train/evaluate/plot, without GIF rollout rendering.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tools_dir = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        for algo in TEAM_ALGOS:
            run_command(
                [
                    sys.executable,
                    str(tools_dir / "train.py"),
                    "--algo",
                    algo,
                    "--model-profile",
                    args.model_profile,
                    "--device",
                    args.device,
                    "--output-dir",
                    str(args.output_dir),
                    "--timesteps",
                    str(args.timesteps),
                    "--eval-freq",
                    str(args.eval_freq),
                    "--eval-episodes",
                    str(args.eval_episodes),
                    "--seeds",
                    *[str(seed) for seed in args.seeds],
                ]
            )

    summary_dir = args.output_dir / "summary"
    plot_dir = args.output_dir / "plots"
    run_command(
        [
            sys.executable,
            str(tools_dir / "evaluate.py"),
            "--input-dir",
            str(args.output_dir),
            "--output-dir",
            str(summary_dir),
            "--episodes",
            str(args.eval_episodes),
            "--model-profile",
            args.model_profile,
        ]
    )
    run_command(
        [
            sys.executable,
            str(tools_dir / "plot_results.py"),
            "--input-dir",
            str(summary_dir),
            "--output-dir",
            str(plot_dir),
        ]
    )

    if not args.skip_render:
        video_dir = args.output_dir / "videos"
        run_command(
            [
                sys.executable,
                str(tools_dir / "render_rollout_video.py"),
                "--policy",
                "lqr",
                "--view",
                "3d",
                "--style",
                "ppt",
                "--model-profile",
                args.model_profile,
                "--steps",
                str(args.render_steps),
                "--fps",
                str(args.fps),
                "--output",
                str(video_dir / "lqr_3d.gif"),
            ]
        )
        for algo in TEAM_ALGOS:
            model_path = args.output_dir / algo / f"seed_{args.seeds[0]}" / "best_model.zip"
            if not model_path.exists():
                print(f"Skip render for {algo}: missing {model_path}", file=sys.stderr)
                continue
            run_command(
                [
                    sys.executable,
                    str(tools_dir / "render_rollout_video.py"),
                    "--policy",
                    "rl",
                    "--algo",
                    algo,
                    "--view",
                    "3d",
                    "--style",
                    "ppt",
                    "--model-path",
                    str(model_path),
                    "--model-profile",
                    args.model_profile,
                    "--steps",
                    str(args.render_steps),
                    "--fps",
                    str(args.fps),
                    "--output",
                    str(video_dir / f"{algo}_3d.gif"),
                ]
            )

    print(f"Team comparison artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
