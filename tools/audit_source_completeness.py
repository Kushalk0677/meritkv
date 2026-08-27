"""Compare external evidence trees with a release by content hash."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path


SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def files_under(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--show", type=int, default=100)
    parser.add_argument(
        "--copy-missing-to",
        type=Path,
        help="Copy unique missing files below a source-named directory.",
    )
    args = parser.parse_args()

    release = args.release.resolve()
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in files_under(release):
        by_size[path.stat().st_size].append(path)

    release_hashes: dict[int, set[str]] = {}
    total_missing = 0
    for source in args.source:
        source = source.resolve()
        missing: list[Path] = []
        seen_source: set[str] = set()
        for path in files_under(source):
            size = path.stat().st_size
            candidates = by_size.get(size, [])
            source_hash = digest(path)
            if source_hash in seen_source:
                continue
            seen_source.add(source_hash)
            if size not in release_hashes:
                release_hashes[size] = {digest(candidate) for candidate in candidates}
            if source_hash not in release_hashes[size]:
                missing.append(path)
        total_missing += len(missing)
        print(f"{source}: {len(missing)} unique file contents not in release")
        for path in missing[: args.show]:
            print(f"  {path}")
        if args.copy_missing_to and missing:
            destination_root = args.copy_missing_to.resolve() / source.name
            for path in missing:
                destination = destination_root / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            print(f"  copied to {destination_root}")
    return 1 if total_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
