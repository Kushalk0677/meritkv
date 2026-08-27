"""Replace duplicate snapshot files with a path-level provenance map."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    snapshot = args.snapshot.resolve()
    snapshot.relative_to(repo)
    by_size: dict[int, list[Path]] = {}
    for path in repo.rglob("*"):
        if not path.is_file() or snapshot in path.parents or ".git" in path.parts:
            continue
        by_size.setdefault(path.stat().st_size, []).append(path)

    hash_cache: dict[Path, str] = {}
    rows: list[tuple[str, str, str, int]] = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file() or path.name == "DEDUPLICATION_MAP.csv":
            continue
        own_hash = digest(path)
        replacement = None
        for candidate in by_size.get(path.stat().st_size, []):
            if candidate not in hash_cache:
                hash_cache[candidate] = digest(candidate)
            if hash_cache[candidate] == own_hash:
                replacement = candidate
                break
        if replacement is not None:
            rows.append(
                (
                    path.relative_to(snapshot).as_posix(),
                    replacement.relative_to(repo).as_posix(),
                    own_hash,
                    path.stat().st_size,
                )
            )
            if args.apply:
                path.unlink()

    if args.apply:
        with (snapshot / "DEDUPLICATION_MAP.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["original_snapshot_path", "preserved_at", "sha256", "bytes"])
            writer.writerows(rows)
        for directory in sorted((path for path in snapshot.rglob("*") if path.is_dir()), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
    print(f"{'Removed' if args.apply else 'Found'} {len(rows)} duplicate snapshot files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
