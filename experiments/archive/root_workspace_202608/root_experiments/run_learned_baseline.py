#!/usr/bin/env python3
"""Forwarder for the P100 learned-baseline runner.

The runnable P100 package lives under ``p100_transfer/shadowkv_p100``. Keeping
this wrapper at the workspace root prevents accidentally running the stale
prototype script from an older experiment pass.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
TARGET = WORKSPACE / "p100_transfer" / "shadowkv_p100" / "experiments" / "run_learned_baseline.py"


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Missing learned-baseline runner: {TARGET}")
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
