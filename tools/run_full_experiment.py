#!/usr/bin/env python3
"""Run the default full experiment pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training, evaluation, and plotting for all algorithms.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tools_dir = Path(__file__).resolve().parent
    for algo in ("sac", "td3", "ppo"):
        subprocess.check_call(
            [sys.executable, str(tools_dir / "train.py"), "--algo", algo, "--output-dir", str(args.output_dir), "--device", args.device]
        )
    summary_dir = args.output_dir / "summary"
    plot_dir = args.output_dir / "plots"
    subprocess.check_call([sys.executable, str(tools_dir / "evaluate.py"), "--input-dir", str(args.output_dir), "--output-dir", str(summary_dir)])
    subprocess.check_call([sys.executable, str(tools_dir / "plot_results.py"), "--input-dir", str(summary_dir), "--output-dir", str(plot_dir)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
