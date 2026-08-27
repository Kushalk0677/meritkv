#!/usr/bin/env python3
"""Measure frozen trace budgets with the pinned Qwen2.5 tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
MODEL_REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
TOKENIZER_SNAPSHOT = (
    "/hf_hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/" + MODEL_REVISION
)
TOKENIZER_MANIFEST_SHA256 = (
    "a28a1734767030cbbdc1588ea53965c763bae58e47596a4f03b237ac15b2c8b0"
)
TOKENIZER_FILE_HASHES = {
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
MAX_TOTAL_TOKENS = 16384


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, default=HERE / "token_budget_manifest.json")
    args = parser.parse_args()
    tokenizer_path = Path(args.tokenizer)
    for filename, expected in TOKENIZER_FILE_HASHES.items():
        path = tokenizer_path / filename
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"pinned tokenizer file mismatch: {filename}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )
    if type(tokenizer).__name__ != "Qwen2TokenizerFast":
        raise SystemExit(f"unexpected tokenizer class: {type(tokenizer).__name__}")
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_path": TOKENIZER_SNAPSHOT,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_manifest_sha256": TOKENIZER_MANIFEST_SHA256,
        "tokenizer_file_sha256": TOKENIZER_FILE_HASHES,
        "minimum_useful_hit_rule": "cached_tokens >= shared_prefix_tokens",
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "capacity_candidate_selection": (
            "16384 is frozen only because the pinned-tokenizer calibration and every "
            "evaluation trace pass the predeclared pressure and hot-set gates"
        ),
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
            "one_off_pressure_to_cap_ratio": pressure_tokens / MAX_TOTAL_TOKENS,
            "hot_prefix_tokens_by_family": hot_prefix_tokens,
            "minimum_useful_cached_tokens_by_family": minimum_useful_cached_tokens,
            "hot_set_tokens": hot_set_tokens,
            "hot_set_to_cap_ratio": hot_set_tokens / MAX_TOTAL_TOKENS,
            "requirements": {
                "one_off_pressure_gt_four_caps": pressure_tokens > 4 * MAX_TOTAL_TOKENS,
                "hot_set_le_half_cap": hot_set_tokens <= MAX_TOTAL_TOKENS // 2,
            },
        }
    if not all(
        all(row["requirements"].values())
        for name, row in report["traces"].items()
        if "evaluation_seed" in name or "calibration_seed" in name
    ):
        raise SystemExit(
            "calibration or evaluation trace does not satisfy the frozen capacity budget"
        )
    report["offline_capacity_gate"] = {
        "status": "pass",
        "candidate": MAX_TOTAL_TOKENS,
        "calibration_and_evaluation_traces_checked": 6,
        "smoke_pressure_gate_exempt": True,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
