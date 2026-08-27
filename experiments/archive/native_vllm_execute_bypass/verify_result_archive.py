#!/usr/bin/env python3
"""Fail unless a downloaded native-bypass archive contains a valid run."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile


def main() -> int:
    archive = Path(sys.argv[1])
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        if "RUN_STATUS.json" not in names:
            print("Missing RUN_STATUS.json", file=sys.stderr)
            return 2
        status = json.loads(handle.read("RUN_STATUS.json"))
        print(json.dumps(status, indent=2))
        if not status.get("benchmark_completed"):
            return 2
        if status.get("benchmark_return_code") != 0:
            return 2
        if "summary.json" not in names or "RESULTS.md" not in names:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
