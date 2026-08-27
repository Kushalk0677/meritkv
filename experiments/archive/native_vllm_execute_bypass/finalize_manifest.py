#!/usr/bin/env python3
"""Regenerate a result-directory SHA-256 manifest after the server stops."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "MANIFEST_SHA256.txt":
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST_SHA256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
