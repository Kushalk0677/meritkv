"""Check repository-local Markdown links."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote


LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FROZEN_PREFIXES = (
    "experiments/archive/",
    "reproduction_packages/",
    "runtime_experiments/qwen2.5/raw_campaigns/",
    "runtime_experiments/native_enforcement_blackwell/evidence_packages/",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    checked = 0

    for document in sorted(root.rglob("*.md")):
        relative_document = document.relative_to(root)
        normalized_document = relative_document.as_posix()
        if ".git" in relative_document.parts or normalized_document.startswith(FROZEN_PREFIXES):
            continue
        text = document.read_text(encoding="utf-8", errors="replace")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            file_target = unquote(target.split("#", 1)[0])
            resolved = (document.parent / file_target).resolve()
            checked += 1
            try:
                resolved.relative_to(root)
            except ValueError:
                errors.append(f"{relative_document}: link escapes repository: {target}")
                continue
            if not resolved.exists():
                errors.append(f"{relative_document}: missing local target: {target}")

    if errors:
        print("Markdown link check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Markdown link check passed: {checked} local links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
