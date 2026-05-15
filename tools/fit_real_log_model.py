#!/usr/bin/env python3
"""Fit empirical one-step models from the real balance log."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import lqr_from_matlab


STATE_COLUMNS = [
    "theta_L",
    "theta_R",
    "theta_1",
    "theta_2",
    "theta_L_dot",
    "theta_R_dot",
    "theta_dot_1",
    "theta_dot_2",
]
ACTION_COLUMNS = ["u_L", "u_R"]
DEFAULT_LOG_HZ = 9.5


@dataclass(frozen=True)
class ModelMetrics:
    name: str
    overall_rmse: float
    state_rmse: dict[str, float]
    state_mae: dict[str, float]


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_csv_path() -> Path:
    return default_project_root() / "小组资料" / "balance_data.csv"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "model_fit"


def load_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    columns = set(rows[0])
    missing = [name for name in [*STATE_COLUMNS, *ACTION_COLUMNS] if name not in columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")

    states = np.array([[float(row[name]) for name in STATE_COLUMNS] for row in rows], dtype=np.float64)
    actions = np.array([[float(row[name]) for name in ACTION_COLUMNS] for row in rows], dtype=np.float64)
    return states, actions


def split_pairs(
    states: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return states[:-1], actions[:-1], states[1:]


def train_test_split(total: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    split = int(total * train_fraction)
    if split <= 0 or split >= total:
        raise ValueError(f"Not enough samples for train_fraction={train_fraction}")
    train_idx = np.arange(split)
    test_idx = np.arange(split, total)
    return train_idx, test_idx


def linear_features(states: np.ndarray, actions: np.ndarray | None = None) -> np.ndarray:
    if actions is None:
        return np.c_[states, np.ones(states.shape[0])]
    return np.c_[states, actions, np.ones(states.shape[0])]


def fit_linear(features: np.ndarray, targets: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(features[train_idx], targets[train_idx], rcond=None)
    return coefficients


def predict_physics(profile: str, dt: float, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    result = lqr_from_matlab.solve_lqr_from_matlab(ts=dt, model_profile=profile)
    return (result.G @ states.T).T + (result.H @ actions.T).T


def metric(name: str, pred: np.ndarray, target: np.ndarray) -> ModelMetrics:
    err = pred - target
    rmse = np.sqrt(np.mean(err * err, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    return ModelMetrics(
        name=name,
        overall_rmse=float(np.sqrt(np.mean(err * err))),
        state_rmse={column: float(value) for column, value in zip(STATE_COLUMNS, rmse)},
        state_mae={column: float(value) for column, value in zip(STATE_COLUMNS, mae)},
    )


def metrics_to_rows(metrics: Iterable[ModelMetrics]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in metrics:
        for column in STATE_COLUMNS:
            rows.append(
                {
                    "model": item.name,
                    "state": column,
                    "rmse": item.state_rmse[column],
                    "mae": item.state_mae[column],
                    "overall_rmse": item.overall_rmse,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def estimate_dt_from_wheels(states: np.ndarray) -> dict[str, float]:
    next_states = states[1:]
    current = states[:-1]
    estimates: list[np.ndarray] = []
    for angle_idx, velocity_idx in ((0, 4), (1, 5)):
        delta = next_states[:, angle_idx] - current[:, angle_idx]
        avg_velocity = 0.5 * (current[:, velocity_idx] + next_states[:, velocity_idx])
        mask = np.abs(avg_velocity) > 0.05
        dt_values = delta[mask] / avg_velocity[mask]
        dt_values = dt_values[np.isfinite(dt_values)]
        dt_values = dt_values[(dt_values > 0.0) & (dt_values < 0.5)]
        estimates.append(dt_values)
    combined = np.concatenate(estimates) if estimates else np.array([], dtype=np.float64)
    if combined.size == 0:
        return {}
    return {
        "wheel_dt_median": float(np.median(combined)),
        "wheel_dt_mean": float(np.mean(combined)),
        "wheel_dt_p10": float(np.percentile(combined, 10)),
        "wheel_dt_p90": float(np.percentile(combined, 90)),
        "wheel_dt_sample_count": int(combined.size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit empirical models from 小组资料/balance_data.csv.")
    parser.add_argument("--csv-path", type=Path, default=default_csv_path())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--log-hz", type=float, default=DEFAULT_LOG_HZ)
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=None,
        help="Discrete timestep for physics one-step comparisons. Defaults to 1/log_hz.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.csv_path.resolve()
    output_dir = args.output_dir.resolve()
    physics_dt = float(args.physics_dt if args.physics_dt is not None else 1.0 / args.log_hz)

    states, actions = load_log(csv_path)
    x0, u0, x1 = split_pairs(states, actions)
    train_idx, test_idx = train_test_split(len(x0), args.train_fraction)

    metrics: list[ModelMetrics] = []

    persistence_pred = x0[test_idx]
    metrics.append(metric("persistence", persistence_pred, x1[test_idx]))

    for profile in ("vendor_matlab", "measured_estimate"):
        pred = predict_physics(profile, physics_dt, x0[test_idx], u0[test_idx])
        metrics.append(metric(f"{profile}_physics_dt_{physics_dt:.6f}", pred, x1[test_idx]))

    x_features = linear_features(x0)
    x_coefficients = fit_linear(x_features, x1, train_idx)
    x_pred = x_features[test_idx] @ x_coefficients
    metrics.append(metric("fitted_closed_loop_x_to_x_next", x_pred, x1[test_idx]))

    xu_features = linear_features(x0, u0)
    xu_coefficients = fit_linear(xu_features, x1, train_idx)
    xu_pred = xu_features[test_idx] @ xu_coefficients
    metrics.append(metric("fitted_linear_xu_to_x_next", xu_pred, x1[test_idx]))

    base_train = predict_physics("measured_estimate", physics_dt, x0[train_idx], u0[train_idx])
    base_test = predict_physics("measured_estimate", physics_dt, x0[test_idx], u0[test_idx])
    delta_train = x1[train_idx] - base_train
    delta_features = linear_features(x0, u0)
    delta_coefficients = fit_linear(delta_features, delta_train, train_idx)
    delta_pred = base_test + delta_features[test_idx] @ delta_coefficients
    metrics.append(metric("measured_estimate_plus_delta_fit", delta_pred, x1[test_idx]))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "model_fit_metrics.csv", metrics_to_rows(metrics))

    np.savez(
        output_dir / "real_log_fit_parameters.npz",
        state_columns=np.array(STATE_COLUMNS),
        action_columns=np.array(ACTION_COLUMNS),
        x_to_x_next_coefficients=x_coefficients,
        xu_to_x_next_coefficients=xu_coefficients,
        measured_delta_coefficients=delta_coefficients,
        physics_dt=np.array([physics_dt], dtype=np.float64),
    )

    summary = {
        "csv_path": str(csv_path),
        "row_count": int(states.shape[0]),
        "transition_count": int(x0.shape[0]),
        "train_transition_count": int(train_idx.size),
        "test_transition_count": int(test_idx.size),
        "state_columns": STATE_COLUMNS,
        "action_columns": ACTION_COLUMNS,
        "log_hz": float(args.log_hz),
        "physics_dt": physics_dt,
        "dt_estimate_from_wheels": estimate_dt_from_wheels(states),
        "metrics": [asdict(item) for item in metrics],
        "notes": [
            "CSV is low-rate closed-loop LQR log data, so physics_dt comparisons assume a held action between rows.",
            "measured_estimate_plus_delta_fit is a data correction model, not a physical parameter identification result.",
        ],
    }
    (output_dir / "model_fit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"CSV: {csv_path}")
    print(f"Output: {output_dir}")
    print(f"Transitions: {x0.shape[0]} (train={train_idx.size}, test={test_idx.size})")
    print(f"Physics dt: {physics_dt:.6f} s")
    print()
    print("Overall test RMSE:")
    for item in metrics:
        print(f"  {item.name}: {item.overall_rmse:.6f}")
    print()
    print("Wrote:")
    print(f"  {output_dir / 'model_fit_summary.json'}")
    print(f"  {output_dir / 'model_fit_metrics.csv'}")
    print(f"  {output_dir / 'real_log_fit_parameters.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
