"""Generate deterministic release inventories without modifying evidence."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
def release_files() -> list[Path]:
    """Return the exact tracked and unignored release paths."""
    output = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    paths: list[Path] = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        rel = Path(raw.decode("utf-8", errors="surrogateescape"))
        if rel.as_posix() in {
            "RELEASE_INVENTORY.csv",
            "SHA256SUMS.txt",
            "reproduction_packages/ARCHIVE_INVENTORY.csv",
            "reproduction_packages/archives/SHA256SUMS.txt",
        }:
            continue
        path = ROOT / rel
        if path.is_file():
            paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix())


def digest(path: Path) -> str:
    """Stream a file into SHA-256 so large raw artifacts stay memory-safe."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    files = release_files()
    with (ROOT / "RELEASE_INVENTORY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            writer.writerow([rel, path.stat().st_size, digest(path)])

    archives = [
        path
        for path in files
        if path.name.lower().endswith((".zip", ".tgz", ".tar.gz"))
    ]
    archive_manifest = ROOT / "reproduction_packages" / "archives" / "SHA256SUMS.txt"
    archive_manifest.parent.mkdir(parents=True, exist_ok=True)
    with archive_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for path in archives:
            handle.write(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n")

    with (ROOT / "reproduction_packages" / "ARCHIVE_INVENTORY.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["archive_path", "bytes", "sha256"])
        for path in archives:
            writer.writerow(
                [path.relative_to(ROOT).as_posix(), path.stat().st_size, digest(path)]
            )

    print(f"Inventoried {len(files)} files; recorded {len(archives)} archives.")


if __name__ == "__main__":
    main()
