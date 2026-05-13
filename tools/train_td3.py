#!/usr/bin/env python3
"""Wrapper for TD3 training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).resolve().with_name("train.py")
    return subprocess.call([sys.executable, str(script), "--algo", "td3", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
