#!/usr/bin/env python3
"""Render rollout animations for the balance-car simulator.

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


WHEEL_RADIUS = 0.0325
TRACK_WIDTH = 0.184
BODY_LENGTH = 0.119
BODY_DEPTH = 0.065
UPPER_ROD_LENGTH = 0.390
BOX_SIGNS = np.array(
    [(length, width, depth) for length in (-1.0, 1.0) for width in (-1.0, 1.0) for depth in (-1.0, 1.0)],
    dtype=np.float64,
)
BOX_EDGE_INDICES = tuple(
    (i, j)
    for i in range(len(BOX_SIGNS))
    for j in range(i + 1, len(BOX_SIGNS))
    if np.count_nonzero(BOX_SIGNS[i] != BOX_SIGNS[j]) == 1
)


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
    action_mode = "discrete_direct" if policy_type == "rl" and algo == "dqn" else "direct"
    env = make_env(seed=seed, max_steps=steps, model_profile=model_profile, action_mode=action_mode)
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
    wheel_x = WHEEL_RADIUS * float(np.mean([state[0], state[1]]))
    wheel_y = WHEEL_RADIUS
    body_angle = float(state[2])
    # The linear model stores theta_2 as the upper-link angle relative to the body.
    pendulum_angle = float(state[2] + state[3])
    pivot_x = wheel_x + BODY_LENGTH * np.sin(body_angle)
    pivot_y = wheel_y + BODY_LENGTH * np.cos(body_angle)
    tip_x = pivot_x + UPPER_ROD_LENGTH * np.sin(pendulum_angle)
    tip_y = pivot_y + UPPER_ROD_LENGTH * np.cos(pendulum_angle)
    return wheel_x, wheel_y, pivot_x, pivot_y, tip_x, tip_y


def save_animation(animation: FuncAnimation, output_path: Path, fps: int, fig: plt.Figure) -> None:
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


def set_line_3d(line, points: np.ndarray) -> None:
    line.set_data_3d(points[:, 0], points[:, 1], points[:, 2])


def wheel_circle_points(center: np.ndarray) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 64)
    return np.column_stack(
        [
            center[0] + WHEEL_RADIUS * np.sin(angles),
            np.full_like(angles, center[1]),
            center[2] + WHEEL_RADIUS * np.cos(angles),
        ]
    )


def wheel_spoke_points(center: np.ndarray, wheel_angle: float, offset: float) -> np.ndarray:
    direction = np.array([np.sin(wheel_angle + offset), 0.0, np.cos(wheel_angle + offset)], dtype=np.float64)
    return np.vstack([center - WHEEL_RADIUS * direction, center + WHEEL_RADIUS * direction])


def robot_geometry_3d(state: np.ndarray) -> dict[str, np.ndarray]:
    wheel_x = WHEEL_RADIUS * float(np.mean([state[0], state[1]]))
    body_angle = float(state[2])
    upper_angle = float(state[2] + state[3])

    axle_center = np.array([wheel_x, 0.0, WHEEL_RADIUS], dtype=np.float64)
    left_wheel_center = axle_center + np.array([0.0, TRACK_WIDTH / 2.0, 0.0], dtype=np.float64)
    right_wheel_center = axle_center + np.array([0.0, -TRACK_WIDTH / 2.0, 0.0], dtype=np.float64)

    body_axis = np.array([np.sin(body_angle), 0.0, np.cos(body_angle)], dtype=np.float64)
    body_depth_axis = np.array([np.cos(body_angle), 0.0, -np.sin(body_angle)], dtype=np.float64)
    lateral_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    pivot = axle_center + BODY_LENGTH * body_axis
    tip = pivot + UPPER_ROD_LENGTH * np.array([np.sin(upper_angle), 0.0, np.cos(upper_angle)], dtype=np.float64)

    body_center = axle_center + 0.5 * BODY_LENGTH * body_axis
    body_half_width = 0.5 * TRACK_WIDTH * 0.72
    body_vertices = (
        body_center
        + BOX_SIGNS[:, 0:1] * (0.5 * BODY_LENGTH * body_axis)
        + BOX_SIGNS[:, 1:2] * (body_half_width * lateral_axis)
        + BOX_SIGNS[:, 2:3] * (0.5 * BODY_DEPTH * body_depth_axis)
    )

    return {
        "axle_center": axle_center,
        "left_wheel_center": left_wheel_center,
        "right_wheel_center": right_wheel_center,
        "pivot": pivot,
        "tip": tip,
        "body_vertices": body_vertices,
        "left_wheel_circle": wheel_circle_points(left_wheel_center),
        "right_wheel_circle": wheel_circle_points(right_wheel_center),
        "left_spoke_a": wheel_spoke_points(left_wheel_center, float(state[0]), 0.0),
        "left_spoke_b": wheel_spoke_points(left_wheel_center, float(state[0]), np.pi / 2.0),
        "right_spoke_a": wheel_spoke_points(right_wheel_center, float(state[1]), 0.0),
        "right_spoke_b": wheel_spoke_points(right_wheel_center, float(state[1]), np.pi / 2.0),
    }


def render_rollout_2d(frames: list[RolloutFrame], output_path: Path, fps: int, title: str) -> None:
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

    wheel = Circle((0.0, WHEEL_RADIUS), radius=WHEEL_RADIUS, color="#2f5f73", alpha=0.95)
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
    save_animation(animation, output_path, fps, fig)


def render_rollout_3d(frames: list[RolloutFrame], output_path: Path, fps: int, title: str) -> None:
    if len(frames) < 2:
        raise ValueError("Need at least two frames to render an animation")

    poses = [robot_geometry_3d(frame.state) for frame in frames]
    points = np.vstack(
        [
            np.vstack([pose["left_wheel_center"], pose["right_wheel_center"], pose["pivot"], pose["tip"], pose["body_vertices"]])
            for pose in poses
        ]
    )
    x_min, x_max = float(np.min(points[:, 0]) - 0.25), float(np.max(points[:, 0]) + 0.25)
    z_max = max(0.65, float(np.max(points[:, 2]) + 0.12))
    y_limit = 0.15

    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_zlim(0.0, z_max)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("left/right (m)")
    ax.set_zlabel("height (m)")
    ax.set_box_aspect((max(x_max - x_min, 0.3), 2.0 * y_limit, z_max))
    ax.view_init(elev=18, azim=-58)
    ax.grid(True, alpha=0.25)

    for y in (-TRACK_WIDTH / 2.0, TRACK_WIDTH / 2.0):
        ax.plot([x_min, x_max], [y, y], [0.0, 0.0], color="#b7b7b7", linewidth=1.0, alpha=0.7)
    ax.plot([x_min, x_max], [0.0, 0.0], [0.0, 0.0], color="#777777", linewidth=1.4, alpha=0.8)

    left_wheel, = ax.plot([], [], [], color="#345c72", linewidth=3.0, label="left wheel")
    right_wheel, = ax.plot([], [], [], color="#4f7f92", linewidth=3.0, label="right wheel")
    spokes = [ax.plot([], [], [], color="#1f3948", linewidth=1.6)[0] for _ in range(4)]
    axle_line, = ax.plot([], [], [], color="#222222", linewidth=2.4)
    body_edges = [ax.plot([], [], [], color="#db7c26", linewidth=2.0)[0] for _ in BOX_EDGE_INDICES]
    body_axis_line, = ax.plot([], [], [], color="#a84c13", linewidth=4.0, solid_capstyle="round")
    upper_rod, = ax.plot([], [], [], color="#224b8d", linewidth=3.2, solid_capstyle="round", label="upper rod")
    tip_dot, = ax.plot([], [], [], "o", color="#224b8d", markersize=5)
    trail, = ax.plot([], [], [], color="#224b8d", linewidth=1.2, alpha=0.25)
    action_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=9, family="monospace", va="top")
    ax.legend(loc="upper right")

    def update(frame_idx: int):
        frame = frames[frame_idx]
        pose = poses[frame_idx]
        set_line_3d(left_wheel, pose["left_wheel_circle"])
        set_line_3d(right_wheel, pose["right_wheel_circle"])
        for line, key in zip(spokes, ["left_spoke_a", "left_spoke_b", "right_spoke_a", "right_spoke_b"]):
            set_line_3d(line, pose[key])
        set_line_3d(axle_line, np.vstack([pose["left_wheel_center"], pose["right_wheel_center"]]))
        for line, edge in zip(body_edges, BOX_EDGE_INDICES):
            set_line_3d(line, pose["body_vertices"][list(edge)])
        set_line_3d(body_axis_line, np.vstack([pose["axle_center"], pose["pivot"]]))
        set_line_3d(upper_rod, np.vstack([pose["pivot"], pose["tip"]]))
        set_line_3d(tip_dot, pose["tip"].reshape(1, 3))
        trail_start = max(0, frame_idx - 80)
        trail_points = np.vstack([poses[idx]["tip"] for idx in range(trail_start, frame_idx + 1)])
        set_line_3d(trail, trail_points)

        action = np.clip(frame.action / 6000.0, -1.0, 1.0)
        action_text.set_text(
            f"step={frame_idx:04d}  reason={frame.termination_reason}\n"
            f"theta_body={frame.state[2]: .4f} rad  theta_upper={frame.state[3]: .4f} rad\n"
            f"u_L={frame.action[0]: .1f} ({action[0]:+.2f})  u_R={frame.action[1]: .1f} ({action[1]:+.2f})"
        )
        return (
            left_wheel,
            right_wheel,
            *spokes,
            axle_line,
            *body_edges,
            body_axis_line,
            upper_rod,
            tip_dot,
            trail,
            action_text,
        )

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    save_animation(animation, output_path, fps, fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a GIF/MP4 rollout from the current balance simulator.")
    parser.add_argument("--policy", choices=["lqr", "rl"], default="lqr")
    parser.add_argument("--algo", choices=["sac", "td3", "ppo", "dqn"], help="Algorithm for --policy rl.")
    parser.add_argument("--model-path", type=Path, help="SB3 checkpoint for --policy rl.")
    parser.add_argument("--model-profile", choices=SIM_MODEL_PROFILES, default=DEFAULT_MODEL_PROFILE)
    parser.add_argument("--view", choices=["3d", "2d"], default="3d", help="3d shows the double-wheel robot; 2d keeps the old side-view diagnostic.")
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
    title = f"{args.policy.upper()} rollout | view={args.view} | profile={args.model_profile} | seed={args.seed}"
    if args.view == "3d":
        render_rollout_3d(frames, args.output, fps=args.fps, title=title)
    else:
        render_rollout_2d(frames, args.output, fps=args.fps, title=title)
    print(f"Rendered {len(frames)} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
