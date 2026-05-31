#!/usr/bin/env python3
"""Render a 2D rollout animation for the balance-car simulator.

This is a visualizer for the current Gymnasium rollout, not a separate physics
engine. The plotted geometry uses measured robot dimensions where available.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.patches import Circle, Rectangle

import lqr_from_matlab
from rl_balance.config import DEFAULT_MODEL_PROFILE, SIM_MODEL_PROFILES
from rl_balance.experiments import make_env
from rl_balance.policies import LqrPolicy, SB3Policy


@dataclass(frozen=True)
class RolloutFrame:
    state: np.ndarray
    action: np.ndarray
    reward: float
    termination_reason: str


def collect_rollout(
    *,
    policy_type: str,
    algo: str | None,
    model_path: Path | None,
    model_profile: str,
    seed: int,
    steps: int,
) -> list[RolloutFrame]:
    env = make_env(seed=seed, max_steps=steps, model_profile=model_profile)
    if policy_type == "lqr":
        lqr_profile = model_profile if model_profile in lqr_from_matlab.available_model_profiles() else "measured_estimate"
        policy = LqrPolicy(model_profile=lqr_profile)
    else:
        if algo is None or model_path is None:
            raise ValueError("--algo and --model-path are required for --policy rl")
        policy = SB3Policy.load(algo, model_path)

    obs, info = env.reset(seed=seed)
    frames = [RolloutFrame(state=np.asarray(info["raw_obs"], dtype=np.float64), action=np.zeros(2), reward=0.0, termination_reason="reset")]

    for _ in range(steps):
        action = np.asarray(policy.predict(obs), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(
            RolloutFrame(
                state=np.asarray(info["raw_obs"], dtype=np.float64),
                action=np.asarray(info["physical_action"], dtype=np.float64),
                reward=float(reward),
                termination_reason=str(info["termination_reason"]),
            )
        )
        if terminated or truncated:
            break
    env.close()
    return frames


def body_geometry(state: np.ndarray) -> tuple[float, float, float, float, float, float]:
    wheel_x = 0.0325 * float(np.mean([state[0], state[1]]))
    wheel_y = 0.0325
    body_angle = float(state[2])
    # The linear model stores theta_2 as the upper-link angle relative to the body.
    pendulum_angle = float(state[2] + state[3])
    body_length = 0.119
    pendulum_length = 0.390
    pivot_x = wheel_x + body_length * np.sin(body_angle)
    pivot_y = wheel_y + body_length * np.cos(body_angle)
    tip_x = pivot_x + pendulum_length * np.sin(pendulum_angle)
    tip_y = pivot_y + pendulum_length * np.cos(pendulum_angle)
    return wheel_x, wheel_y, pivot_x, pivot_y, tip_x, tip_y


def render_rollout(frames: list[RolloutFrame], output_path: Path, fps: int, title: str) -> None:
    if len(frames) < 2:
        raise ValueError("Need at least two frames to render an animation")

    positions = np.array([body_geometry(frame.state) for frame in frames], dtype=np.float64)
    x_values = positions[:, [0, 2, 4]].reshape(-1)
    y_values = positions[:, [1, 3, 5]].reshape(-1)
    x_min, x_max = float(np.min(x_values) - 0.35), float(np.max(x_values) + 0.35)
    y_max = max(0.65, float(np.max(y_values) + 0.15))

    fig, (ax, info_ax) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [4, 1]},
        constrained_layout=True,
    )
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.02, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m, visualized from wheel angle)")
    ax.set_ylabel("height (m)")
    ax.axhline(0.0, color="#444444", linewidth=2.0)

    wheel = Circle((0.0, 0.0325), radius=0.0325, color="#2f5f73", alpha=0.95)
    body_line, = ax.plot([], [], color="#db7c26", linewidth=5, solid_capstyle="round", label="body")
    pendulum_line, = ax.plot([], [], color="#224b8d", linewidth=3, solid_capstyle="round", label="upper rod")
    tip_dot, = ax.plot([], [], "o", color="#224b8d", markersize=5)
    trail, = ax.plot([], [], color="#224b8d", linewidth=1, alpha=0.25)
    ax.add_patch(wheel)
    ax.legend(loc="upper right")

    info_ax.set_xlim(-1.0, 1.0)
    info_ax.set_ylim(-1.0, 1.0)
    info_ax.axis("off")
    left_bar = Rectangle((-0.85, -0.2), 0.0, 0.25, color="#457b9d")
    right_bar = Rectangle((0.0, -0.6), 0.0, 0.25, color="#e76f51")
    info_ax.add_patch(left_bar)
    info_ax.add_patch(right_bar)
    text = info_ax.text(-0.95, 0.45, "", fontsize=10, family="monospace", va="top")

    def update(frame_idx: int):
        frame = frames[frame_idx]
        wheel_x, wheel_y, pivot_x, pivot_y, tip_x, tip_y = body_geometry(frame.state)
        wheel.center = (wheel_x, wheel_y)
        body_line.set_data([wheel_x, pivot_x], [wheel_y, pivot_y])
        pendulum_line.set_data([pivot_x, tip_x], [pivot_y, tip_y])
        tip_dot.set_data([tip_x], [tip_y])
        trail_start = max(0, frame_idx - 80)
        trail.set_data(positions[trail_start : frame_idx + 1, 4], positions[trail_start : frame_idx + 1, 5])

        normalized_action = np.clip(frame.action / 6000.0, -1.0, 1.0)
        left_width = 0.75 * float(normalized_action[0])
        right_width = 0.75 * float(normalized_action[1])
        left_bar.set_x(-0.85 if left_width >= 0 else -0.85 + left_width)
        left_bar.set_width(abs(left_width))
        right_bar.set_x(0.0 if right_width >= 0 else right_width)
        right_bar.set_width(abs(right_width))
        text.set_text(
            f"step={frame_idx:04d}  reason={frame.termination_reason}\n"
            f"theta_1={frame.state[2]: .4f} rad  theta_2={frame.state[3]: .4f} rad\n"
            f"u_L={frame.action[0]: .1f}  u_R={frame.action[1]: .1f}  reward={frame.reward: .3f}"
        )
        return wheel, body_line, pendulum_line, tip_dot, trail, left_bar, right_bar, text

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            animation.save(output_path, writer=PillowWriter(fps=fps))
        elif suffix == ".mp4":
            if not writers.is_available("ffmpeg"):
                raise RuntimeError("MP4 output requires ffmpeg. Use a .gif output path on this machine.")
            animation.save(output_path, writer=FFMpegWriter(fps=fps))
        else:
            raise ValueError("Output path must end with .gif or .mp4")
    finally:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 2D GIF rollout from the current balance simulator.")
    parser.add_argument("--policy", choices=["lqr", "rl"], default="lqr")
    parser.add_argument("--algo", choices=["sac", "td3", "ppo"], help="Algorithm for --policy rl.")
    parser.add_argument("--model-path", type=Path, help="SB3 checkpoint for --policy rl.")
    parser.add_argument("--model-profile", choices=SIM_MODEL_PROFILES, default=DEFAULT_MODEL_PROFILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/videos/lqr_rollout.gif"),
        help="Animation output. GIF always works with Pillow; MP4 requires ffmpeg.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = collect_rollout(
        policy_type=args.policy,
        algo=args.algo,
        model_path=args.model_path,
        model_profile=args.model_profile,
        seed=args.seed,
        steps=args.steps,
    )
    title = f"{args.policy.upper()} rollout | profile={args.model_profile} | seed={args.seed}"
    render_rollout(frames, args.output, fps=args.fps, title=title)
    print(f"Rendered {len(frames)} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
