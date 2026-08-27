"""Check structural completeness and integrity of the research release."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "ARTIFACT_MANIFEST.csv",
    "CLAIMS_TO_ARTIFACTS.md",
    "docs/EXPERIMENT_CATALOG.md",
    "docs/HARDWARE_AND_ENVIRONMENTS.md",
    "docs/REPOSITORY_STRUCTURE.md",
    "experiments/README.md",
    "reproduction_packages/README.md",
    "results/RESULTS.md",
    "runtime_experiments/README.md",
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    errors: list[str] = []

    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            errors.append(f"missing required path: {rel}")

    manifest = ROOT / "ARTIFACT_MANIFEST.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                rel = row.get("artifact_path", "").rstrip("/")
                if rel and not (ROOT / rel).exists():
                    errors.append(f"artifact manifest path does not exist: {rel}")

    ignored = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    for rel in ignored:
        normalized = rel.replace("\\", "/")
        if normalized.startswith(("results/", "runtime_experiments/")) and not any(
            marker in normalized for marker in ("/__pycache__/", "/.pytest_cache/", ".egg-info/")
        ) and not normalized.endswith((".pyc", ".pyo")):
            errors.append(f"ignored evidence file: {normalized}")

    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    for rel in tracked:
        path = ROOT / rel
        normalized = rel.replace("\\", "/")
        if "__pycache__" in normalized or normalized.endswith((".pyc", ".pyo")):
            errors.append(f"generated cache in release: {normalized}")
        if path.is_file() and path.stat().st_size > 100 * 1024 * 1024:
            errors.append(f"file exceeds GitHub 100 MiB limit: {normalized}")

    archive_manifest = ROOT / "reproduction_packages" / "archives" / "SHA256SUMS.txt"
    if archive_manifest.exists():
        for line in archive_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, rel = line.split(maxsplit=1)
            path = ROOT / rel.strip()
            if not path.exists() or sha256(path) != expected:
                errors.append(f"archive checksum mismatch: {rel.strip()}")

    if errors:
        print("Release audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Release audit passed ({len(tracked)} visible files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
