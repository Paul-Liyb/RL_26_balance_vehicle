#!/usr/bin/env python3
"""Wrapper for SAC training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).resolve().with_name("train.py")
    return subprocess.call([sys.executable, str(script), "--algo", "sac", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
