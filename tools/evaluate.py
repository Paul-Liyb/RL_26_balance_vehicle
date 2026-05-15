#!/usr/bin/env python3
"""Evaluate saved policies and produce summary artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from rl_balance.config import SIM_MODEL_PROFILES
from rl_balance.experiments import aggregate_summary, collect_training_curves, evaluate_saved_runs, write_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained policies and baseline.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--model-profile",
        choices=SIM_MODEL_PROFILES,
        help="Override the model profile used for baseline and saved-policy evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, best_trace = evaluate_saved_runs(args.input_dir, args.episodes, model_profile=args.model_profile)
    write_summary_csv(args.output_dir / "summary.csv", summary_rows)
    write_summary_csv(args.output_dir / "algorithm_summary.csv", aggregate_summary(summary_rows))
    write_summary_csv(args.output_dir / "training_curves.csv", collect_training_curves(args.input_dir))
    trace_rows = [asdict(row) for row in best_trace]
    write_summary_csv(args.output_dir / "rollout_trace.csv", trace_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
