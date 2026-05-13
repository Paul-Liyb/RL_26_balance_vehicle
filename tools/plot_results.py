#!/usr/bin/env python3
"""Plot experiment summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from evaluation artifacts.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    training_curves = pd.read_csv(args.input_dir / "training_curves.csv")
    algorithm_summary_path = args.input_dir / "algorithm_summary.csv"
    summary = pd.read_csv(algorithm_summary_path if algorithm_summary_path.exists() else args.input_dir / "summary.csv")
    rollout_trace = pd.read_csv(args.input_dir / "rollout_trace.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    if not training_curves.empty:
        sns.lineplot(data=training_curves, x="timesteps", y="mean_return", hue="algorithm", ax=ax)
    ax.set_title("Training Return Curve")
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_return_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary, x="algorithm", y="success_rate", ax=ax)
    ax.set_title("Success Rate")
    fig.tight_layout()
    fig.savefig(args.output_dir / "success_rate_bar.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary, x="algorithm", y="mean_control_energy", ax=ax)
    ax.set_title("Mean Control Energy")
    fig.tight_layout()
    fig.savefig(args.output_dir / "control_energy_bar.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(rollout_trace["step"], rollout_trace["theta_1"])
    axes[0].set_ylabel("theta_1")
    axes[1].plot(rollout_trace["step"], rollout_trace["theta_2"])
    axes[1].set_ylabel("theta_2")
    axes[2].plot(rollout_trace["step"], rollout_trace["u_l"])
    axes[2].set_ylabel("u_L")
    axes[3].plot(rollout_trace["step"], rollout_trace["u_r"])
    axes[3].set_ylabel("u_R")
    axes[3].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(args.output_dir / "rollout_timeseries.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
