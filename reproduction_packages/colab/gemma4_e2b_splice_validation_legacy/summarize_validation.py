#!/usr/bin/env python3
"""Summarize one or more exact-splice validator JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def numeric_max(values: list[float]) -> float | None:
    return max(values) if values else None


def summarize_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    cache_max: list[float] = []
    suffix_native_full: list[float] = []
    suffix_splice_full: list[float] = []
    suffix_splice_native: list[float] = []
    generation_native_full: list[float] = []
    generation_splice_full: list[float] = []
    generation_splice_native: list[float] = []
    first_native_mismatch = None
    first_splice_mismatch = None

    for case_index, case in enumerate(report["cases"]):
        for layer in case["prefix_cache"]["layers"]:
            cache_max.extend((layer["key"].get("max_abs", 0.0), layer["value"].get("max_abs", 0.0)))
        suffix_native_full.append(case["suffix_logits"]["native_vs_full"].get("max_abs", 0.0))
        suffix_splice_full.append(case["suffix_logits"]["splice_vs_full"].get("max_abs", 0.0))
        suffix_splice_native.append(case["suffix_logits"]["splice_vs_native"].get("max_abs", 0.0))
        for step in case["generation"]["steps"]:
            generation_native_full.append(step["native_vs_full"].get("max_abs", 0.0))
            generation_splice_full.append(step["splice_vs_full"].get("max_abs", 0.0))
            generation_splice_native.append(step["splice_vs_native"].get("max_abs", 0.0))
            if first_native_mismatch is None and not step["native_token_match"]:
                first_native_mismatch = {"case": case_index, "step": step["step"]}
            if first_splice_mismatch is None and not step["splice_token_match"]:
                first_splice_mismatch = {"case": case_index, "step": step["step"]}

    summary = dict(report["summary"])
    return {
        "file": path.name,
        "environment": report["environment"],
        "configuration": report["configuration"],
        "summary": summary,
        "max_abs": {
            "prefix_cache": numeric_max(cache_max),
            "suffix_native_vs_full": numeric_max(suffix_native_full),
            "suffix_splice_vs_full": numeric_max(suffix_splice_full),
            "suffix_splice_vs_native": numeric_max(suffix_splice_native),
            "generation_native_vs_full": numeric_max(generation_native_full),
            "generation_splice_vs_full": numeric_max(generation_splice_full),
            "generation_splice_vs_native": numeric_max(generation_splice_native),
        },
        "first_native_token_mismatch": first_native_mismatch,
        "first_splice_token_mismatch": first_splice_mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.json"))
    paths = [path for path in paths if path.name != args.output_json.name]
    summaries = [summarize_report(path) for path in paths]
    combined = {
        "reports": summaries,
        "all_reports_strict_pass": bool(summaries)
        and all(row["summary"]["all_strict_pass"] for row in summaries),
        "all_reports_token_pass": bool(summaries)
        and all(
            row["summary"]["native_token_passes"] == row["summary"]["cases"]
            and row["summary"]["splice_token_passes"] == row["summary"]["cases"]
            for row in summaries
        ),
    }
    args.output_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    lines = [
        "# Gemma-4-E2B Exact-Splice Validation Summary",
        "",
        f"- Reports: {len(summaries)}",
        f"- All strict pass: `{combined['all_reports_strict_pass']}`",
        f"- All token pass: `{combined['all_reports_token_pass']}`",
        "",
        "| Report | Attention | Cases | Strict | Cache | Native tokens | Splice tokens |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        config = row["configuration"]
        summary = row["summary"]
        lines.append(
            f"| `{row['file']}` | {config['attention_implementation']} | "
            f"{summary['cases']} | {summary['strict_passes']} | "
            f"{summary['cache_passes']} | {summary['native_token_passes']} | "
            f"{summary['splice_token_passes']} |"
        )
    lines.extend(("", "## Maximum Absolute Differences", ""))
    for row in summaries:
        lines.append(f"### `{row['file']}`")
        lines.append("")
        for key, value in row["max_abs"].items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(combined, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
