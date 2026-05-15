#!/usr/bin/env python3
"""Reproduce the MATLAB LQR script in Python and compare with current gains."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete


CONTROL_RELATIVE_PATH = Path(
    "平衡车补充资料"
    "/WHEELTEC B585 二阶平衡机器人_HAL函数版本_修改版"
    "/MiniBalance/control.c"
)
WIFI_RELATIVE_PATH = Path("平衡车补充资料/wifi3.0.py")


@dataclass(frozen=True)
class PhysicalParams:
    """Physical parameters for the linearized balance-car model."""

    m_1: float
    m_2: float
    r: float
    L_1: float
    L_2: float
    l_1: float
    l_2: float
    I_1: float
    I_2: float
    g: float = 9.8


@dataclass(frozen=True)
class LqrResult:
    A: np.ndarray
    B: np.ndarray
    G: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    K: np.ndarray


def rod_inertia_about_center(mass: float, length: float) -> float:
    return (1.0 / 12.0) * mass * length**2


def rectangular_pitch_inertia_about_center(mass: float, height: float, depth: float) -> float:
    return (1.0 / 12.0) * mass * (height**2 + depth**2)


VENDOR_MATLAB_PARAMS = PhysicalParams(
    m_1=0.9,
    m_2=0.1,
    r=0.0335,
    L_1=0.126,
    L_2=0.390,
    l_1=0.126 / 2.0,
    l_2=0.390 / 2.0,
    I_1=rod_inertia_about_center(0.9, 0.126),
    I_2=rod_inertia_about_center(0.1, 0.390),
)

MEASURED_TOTAL_HEIGHT = 0.5415
MEASURED_BODY_DEPTH = 0.065
MEASURED_ROD_LENGTH = 0.390
MEASURED_WHEEL_RADIUS = 0.0325
MEASURED_BODY_HEIGHT = MEASURED_TOTAL_HEIGHT - MEASURED_ROD_LENGTH - MEASURED_WHEEL_RADIUS

MEASURED_ESTIMATE_PARAMS = PhysicalParams(
    m_1=1.0,
    m_2=0.1,
    r=MEASURED_WHEEL_RADIUS,
    L_1=MEASURED_BODY_HEIGHT,
    L_2=MEASURED_ROD_LENGTH,
    l_1=0.055,
    l_2=0.195,
    I_1=rectangular_pitch_inertia_about_center(1.0, MEASURED_BODY_HEIGHT, MEASURED_BODY_DEPTH),
    I_2=rod_inertia_about_center(0.1, MEASURED_ROD_LENGTH),
)

DEFAULT_MODEL_PROFILE = "vendor_matlab"
MODEL_PROFILES: dict[str, PhysicalParams] = {
    DEFAULT_MODEL_PROFILE: VENDOR_MATLAB_PARAMS,
    "measured_estimate": MEASURED_ESTIMATE_PARAMS,
}


def available_model_profiles() -> tuple[str, ...]:
    return tuple(MODEL_PROFILES.keys())


def get_physical_params(model_profile: str = DEFAULT_MODEL_PROFILE) -> PhysicalParams:
    try:
        return MODEL_PROFILES[model_profile]
    except KeyError as exc:
        profiles = ", ".join(available_model_profiles())
        raise ValueError(f"Unsupported model profile: {model_profile}. Available profiles: {profiles}") from exc


def build_continuous_model(model_profile: str = DEFAULT_MODEL_PROFILE) -> tuple[np.ndarray, np.ndarray]:
    """Build continuous-time A/B matrices for a selected physical profile."""
    params = get_physical_params(model_profile)
    m_1 = params.m_1
    m_2 = params.m_2
    r = params.r
    L_1 = params.L_1
    l_1 = params.l_1
    l_2 = params.l_2
    g = params.g
    I_1 = params.I_1
    I_2 = params.I_2

    p = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [
                (r / 2.0) * (m_1 * l_1 + m_2 * L_1),
                (r / 2.0) * (m_1 * l_1 + m_2 * L_1),
                m_1 * l_1**2 + m_2 * L_1**2 + I_1,
                m_2 * L_1 * l_2,
            ],
            [
                (r / 2.0) * m_2 * l_2,
                (r / 2.0) * m_2 * l_2,
                m_2 * L_1 * l_2,
                m_2 * l_2**2 + I_2,
            ],
        ],
        dtype=float,
    )

    q = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [
                0.0,
                0.0,
                (m_1 * l_1 + m_2 * L_1) * g,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, m_2 * g * l_2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    temp = np.linalg.solve(p, q)
    A = np.block([[np.zeros((4, 4)), np.eye(4)], [temp[:, :8]]])
    B = np.vstack([np.zeros((4, 2)), temp[:, 8:10]])
    return A, B


def solve_lqr_from_matlab(ts: float = 0.01, model_profile: str = DEFAULT_MODEL_PROFILE) -> LqrResult:
    """Replicate c2d + dlqr from the MATLAB script."""
    A, B = build_continuous_model(model_profile=model_profile)
    G, H, _, _, _ = cont2discrete((A, B, np.eye(8), np.zeros((8, 2))), ts, method="zoh")
    Q = np.diag([51.2938, 51.2938, 32.8281, 131.3123, 51.2938, 51.2938, 131.3123, 131.3123])
    R = 0.0005 * np.eye(2)
    P = solve_discrete_are(G, H, Q, R)
    K = np.linalg.solve(H.T @ P @ H + R, H.T @ P @ G)
    return LqrResult(A=A, B=B, G=G, H=H, Q=Q, R=R, K=K)


def controllability_matrix(G: np.ndarray, H: np.ndarray) -> np.ndarray:
    columns = [H]
    current = H
    for _ in range(1, G.shape[0]):
        current = G @ current
        columns.append(current)
    return np.hstack(columns)


def firmware_gain_reference() -> np.ndarray:
    return np.array(
        [
            [81.2695, -10.0616, -5492.4061, 18921.7098, 100.3633, 8.0376, 447.3084, 2962.7738],
            [-10.0616, 81.2695, -5492.4061, 18921.7098, 8.0376, 100.3633, 447.3084, 2962.7738],
        ],
        dtype=float,
    )


def parse_gain_matrix(text: str) -> np.ndarray:
    matches = {
        name: float(value)
        for name, value in re.findall(r"\b(K[12][1-8])\s*=\s*([-+]?\d+(?:\.\d+)?)", text)
    }
    required = [f"K{row}{col}" for row in (1, 2) for col in range(1, 9)]
    missing = [name for name in required if name not in matches]
    if missing:
        raise ValueError(f"Missing coefficients: {', '.join(missing)}")
    return np.array(
        [[matches[f"K1{i}"] for i in range(1, 9)], [matches[f"K2{i}"] for i in range(1, 9)]],
        dtype=float,
    )


def parse_gain_matrix_from_file(path: Path) -> np.ndarray:
    return parse_gain_matrix(path.read_text(encoding="utf-8", errors="ignore"))


def format_row(row: np.ndarray) -> str:
    return ", ".join(f"{value:.4f}" for value in row)


def describe_difference(name: str, computed: np.ndarray, reference: np.ndarray, atol: float) -> str:
    max_abs_diff = float(np.max(np.abs(computed - reference)))
    status = "MATCH" if np.allclose(computed, reference, atol=atol, rtol=0.0) else "DIFF"
    return f"{name}: {status} (max_abs_diff={max_abs_diff:.8f}, atol={atol:.8f})"


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_control_path(project_root: Path) -> Path:
    return project_root / CONTROL_RELATIVE_PATH


def default_wifi_path(project_root: Path) -> Path:
    return project_root / WIFI_RELATIVE_PATH


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the MATLAB LQR script and compare with current firmware gains."
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--atol", type=float, default=5e-4, help="Absolute tolerance for gain comparison.")
    parser.add_argument("--show-matrices", action="store_true", help="Print A/B/G/H/Q/R matrices.")
    parser.add_argument(
        "--model-profile",
        choices=available_model_profiles(),
        default=DEFAULT_MODEL_PROFILE,
        help="Physical model profile used to build A/B and LQR gains.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    control_path = default_control_path(project_root)
    wifi_path = default_wifi_path(project_root)

    params = get_physical_params(args.model_profile)
    result = solve_lqr_from_matlab(model_profile=args.model_profile)
    tc = controllability_matrix(result.G, result.H)
    control_rank = int(np.linalg.matrix_rank(tc))

    np.set_printoptions(precision=4, suppress=True)

    print(f"Project root: {project_root}")
    print(f"Control file: {control_path}")
    print(f"WiFi script: {wifi_path}")
    print(f"Model profile: {args.model_profile}")
    print(f"Discrete sample time Ts: 0.01 s")
    print(f"Controllability rank: {control_rank}")
    print("Physical parameters:")
    print(
        "  "
        f"m_1={params.m_1:.4f}, m_2={params.m_2:.4f}, r={params.r:.4f}, "
        f"L_1={params.L_1:.4f}, L_2={params.L_2:.4f}, "
        f"l_1={params.l_1:.4f}, l_2={params.l_2:.4f}, "
        f"I_1={params.I_1:.6f}, I_2={params.I_2:.6f}, g={params.g:.4f}"
    )
    print()

    print("K1 =")
    print(format_row(result.K[0]))
    print()
    print("K2 =")
    print(format_row(result.K[1]))
    print()

    if args.show_matrices:
        print("A =")
        print(result.A)
        print()
        print("B =")
        print(result.B)
        print()
        print("G =")
        print(result.G)
        print()
        print("H =")
        print(result.H)
        print()
        print("Q =")
        print(result.Q)
        print()
        print("R =")
        print(result.R)
        print()

    if args.model_profile == DEFAULT_MODEL_PROFILE:
        builtin_reference = firmware_gain_reference()
        print(describe_difference("Built-in reference", result.K, builtin_reference, args.atol))

        if control_path.exists():
            control_gain = parse_gain_matrix_from_file(control_path)
            print(describe_difference("control.c", result.K, control_gain, args.atol))
        else:
            print(f"control.c: SKIP ({control_path} not found)")

        if wifi_path.exists():
            wifi_gain = parse_gain_matrix_from_file(wifi_path)
            print(describe_difference("wifi3.0.py", result.K, wifi_gain, args.atol))
        else:
            print(f"wifi3.0.py: SKIP ({wifi_path} not found)")
    else:
        print("Built-in reference: SKIP (reference gains are for vendor_matlab)")
        print("control.c: SKIP (firmware gains are for vendor_matlab)")
        print("wifi3.0.py: SKIP (WiFi script gains are for vendor_matlab)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
