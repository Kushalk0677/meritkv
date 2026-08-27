#!/usr/bin/env python3
"""Measure frozen trace budgets with the pinned Gemma tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, default=HERE / "token_budget_manifest.json")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )
    report = {
        "schema_version": 1,
        "tokenizer_path": args.tokenizer,
        "tokenizer_class": type(tokenizer).__name__,
        "minimum_useful_hit_rule": "cached_tokens >= shared_prefix_tokens",
        "max_total_tokens": 16384,
        "page_size": 1,
        "traces": {},
    }
    for trace in sorted((HERE / "traces").glob("*.jsonl")):
        rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        hot_prefix_tokens: dict[str, int] = {}
        minimum_useful_cached_tokens: dict[str, int] = {}
        pressure_tokens = 0
        total_prompt_tokens = 0
        for row in rows:
            metadata = row.get("metadata") or {}
            prompt_tokens = len(tokenizer.encode(row["prompt"]))
            total_prompt_tokens += prompt_tokens
            if metadata.get("shadowkv_trace_class") == "long_one_off_prefix":
                pressure_tokens += prompt_tokens
            if metadata.get("hot_family") is not None:
                family = str(metadata["hot_family"])
                prefix = metadata["shared_prefix_text"]
                prefix_tokens = len(tokenizer.encode(prefix))
                previous = hot_prefix_tokens.setdefault(family, prefix_tokens)
                if previous != prefix_tokens:
                    raise SystemExit(f"hot family {family} prefix token count drifted")
                minimum_useful_cached_tokens[family] = prefix_tokens
        hot_set_tokens = sum(hot_prefix_tokens.values())
        report["traces"][str(trace.relative_to(HERE))] = {
            "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "requests": len(rows),
            "total_prompt_tokens": total_prompt_tokens,
            "one_off_pressure_prompt_tokens": pressure_tokens,
            "one_off_pressure_to_cap_ratio": pressure_tokens / 16384,
            "hot_prefix_tokens_by_family": hot_prefix_tokens,
            "minimum_useful_cached_tokens_by_family": minimum_useful_cached_tokens,
            "hot_set_tokens": hot_set_tokens,
            "hot_set_to_cap_ratio": hot_set_tokens / 16384,
            "requirements": {
                "one_off_pressure_gt_four_caps": pressure_tokens > 4 * 16384,
                "hot_set_le_half_cap": hot_set_tokens <= 16384 // 2,
            },
        }
    if not all(
        all(row["requirements"].values())
        for name, row in report["traces"].items()
        if "evaluation_seed" in name
    ):
        raise SystemExit("an evaluation trace does not satisfy the frozen capacity budget")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
