#!/usr/bin/env python3
"""Render current policy rollouts with a MuJoCo visual model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

import lqr_from_matlab
from render_rollout_video import collect_rollout
from rl_balance.config import DEFAULT_MODEL_PROFILE, SIM_MODEL_PROFILES


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "wheeltec_balance_vehicle.xml"


def configure_mujoco_state(model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray) -> None:
    left_wheel = float(state[0])
    right_wheel = float(state[1])
    body_pitch = float(state[2])
    upper_pitch = float(state[3])
    slide_x = 0.0325 * float(np.mean([left_wheel, right_wheel]))

    joint_values = {
        "slide_x": slide_x,
        "left_wheel_spin": left_wheel,
        "right_wheel_spin": right_wheel,
        "body_pitch": -body_pitch,
        "upper_pitch": -upper_pitch,
    }
    for joint_name, value in joint_values.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def render_frames(
    *,
    frames,
    mjcf_path: Path,
    width: int,
    height: int,
    camera: str,
) -> list[np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    rendered: list[np.ndarray] = []
    try:
        for frame in frames:
            configure_mujoco_state(model, data, frame.state)
            renderer.update_scene(data, camera=camera)
            image = renderer.render()
            rendered.append(np.asarray(image, dtype=np.uint8).copy())
    finally:
        renderer.close()
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an LQR/RL rollout using MuJoCo.")
    parser.add_argument("--policy", choices=["lqr", "rl"], default="lqr")
    parser.add_argument("--algo", choices=["sac", "td3", "ppo", "dqn"], help="Algorithm for --policy rl.")
    parser.add_argument("--model-path", type=Path, help="SB3 checkpoint for --policy rl.")
    parser.add_argument("--model-profile", choices=SIM_MODEL_PROFILES, default=DEFAULT_MODEL_PROFILE)
    parser.add_argument("--mjcf-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera", default="presentation")
    parser.add_argument("--output", type=Path, default=Path("artifacts/videos/mujoco_lqr.gif"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mjcf_path.exists():
        raise FileNotFoundError(args.mjcf_path)
    if args.policy == "lqr" and args.model_profile not in lqr_from_matlab.available_model_profiles():
        print(f"Use measured_estimate LQR policy for empirical profile {args.model_profile}")

    rollout_frames = collect_rollout(
        policy_type=args.policy,
        algo=args.algo,
        model_path=args.model_path,
        model_profile=args.model_profile,
        seed=args.seed,
        steps=args.steps,
    )
    rendered = render_frames(
        frames=rollout_frames,
        mjcf_path=args.mjcf_path,
        width=args.width,
        height=args.height,
        camera=args.camera,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, rendered, fps=args.fps)
    print(f"Rendered {len(rendered)} MuJoCo frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
