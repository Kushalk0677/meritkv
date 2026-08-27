#!/usr/bin/env python3
"""Generate the corrected, frozen MeritKV execution plans."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODELS = [
    ("qwen25_32b", "Qwen/Qwen2.5-32B-Instruct"),
    ("gemma4_31b", "google/gemma-4-31B-it"),
]
ARMS = [
    "native_baseline",
    "meritkv_write_through",
    "meritkv_enforced",
    "forced_recompute",
]
DATASETS = ["daily_dialog", "samsum", "ag_news", "dolly", "xsum"]
PROMPT_MODES = ["templated", "rag"]
SEEDS = [1, 2, 3, 4, 5]
ARM_ORDERS = [
    [0, 1, 2, 3],
    [1, 3, 0, 2],
    [2, 0, 3, 1],
    [3, 2, 1, 0],
    [0, 2, 1, 3],
]


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    blocks: list[dict[str, object]] = []
    block_id = 0
    for seed_index, seed in enumerate(SEEDS):
        model_order = MODELS if seed_index % 2 == 0 else list(reversed(MODELS))
        for model_slug, model_id in model_order:
            for arm_index in ARM_ORDERS[seed_index]:
                block_id += 1
                blocks.append(
                    {
                        "block_id": block_id,
                        "model_slug": model_slug,
                        "model_id": model_id,
                        "arm": ARMS[arm_index],
                        "seed": seed,
                        "cells": 10,
                    }
                )

    cells: list[dict[str, object]] = []
    for block in blocks:
        cell_id = 0
        seed = int(block["seed"])
        datasets = DATASETS if seed % 2 else list(reversed(DATASETS))
        modes = PROMPT_MODES if seed % 2 else list(reversed(PROMPT_MODES))
        for dataset in datasets:
            for prompt_mode in modes:
                cell_id += 1
                cells.append(
                    {
                        "block_id": block["block_id"],
                        "cell_id": cell_id,
                        "model_slug": block["model_slug"],
                        "model_id": block["model_id"],
                        "arm": block["arm"],
                        "seed": seed,
                        "dataset": dataset,
                        "prompt_mode": prompt_mode,
                        "requests": 256,
                        "reset_before": "flush_cache+action_counters",
                    }
                )

    smoke: list[dict[str, object]] = []
    smoke_id = 0
    for model_slug, model_id in MODELS:
        for arm in ARMS:
            smoke_id += 1
            smoke.append(
                {
                    "smoke_id": smoke_id,
                    "model_slug": model_slug,
                    "model_id": model_id,
                    "arm": arm,
                    "seed": 1,
                    "dataset": "samsum",
                    "prompt_mode": "templated",
                    "requests": 16,
                }
            )

    proof: list[dict[str, object]] = []
    proof_id = 0
    for model_slug, model_id in MODELS:
        for arm in ["native_baseline", "meritkv_write_through", "meritkv_enforced"]:
            proof_id += 1
            proof.append(
                {
                    "proof_id": proof_id,
                    "model_slug": model_slug,
                    "model_id": model_id,
                    "arm": arm,
                    "seed": 20260820,
                    "dataset": "samsum",
                    "prompt_mode": "templated",
                    "requests": 64,
                    "trace": "admission_sensitive_trace.jsonl",
                }
            )

    write_tsv(HERE / "block_plan_40.tsv", blocks)
    write_tsv(HERE / "cell_plan_400.tsv", cells)
    write_tsv(HERE / "smoke_plan_8.tsv", smoke)
    write_tsv(HERE / "proof_plan_6.tsv", proof)


if __name__ == "__main__":
    main()
