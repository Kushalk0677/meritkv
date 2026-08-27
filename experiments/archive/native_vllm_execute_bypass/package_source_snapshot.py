#!/usr/bin/env python3
"""Create the exact minimal MeritKV source snapshot used by the GPU run."""

from __future__ import annotations

from pathlib import Path
import sys
import zipfile


def main() -> None:
    project_root = Path(sys.argv[1]).resolve()
    archive = Path(sys.argv[2]).resolve()
    source = project_root / "v10" / "src" / "proactive_kv_cache"
    config = project_root / "v10" / "config" / "config.yaml"
    if not source.is_dir() or not config.is_file():
        raise RuntimeError(f"Missing source/config under {project_root}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            handle.write(path, path.relative_to(project_root).as_posix())
        handle.write(config, config.relative_to(project_root).as_posix())


if __name__ == "__main__":
    main()
