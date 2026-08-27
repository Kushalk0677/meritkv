#!/usr/bin/env python3
"""Fail-closed offline probe for every frozen full-matrix workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

from proactive_kv_cache.datasets import DATASET_REGISTRY, load_public_text_rows


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepared_revision(dataset: Any) -> str:
    cache_files = getattr(dataset, "cache_files", None) or []
    revisions = {
        Path(item["filename"]).resolve().parent.name
        for item in cache_files
        if isinstance(item, dict) and item.get("filename")
    }
    if len(revisions) != 1:
        raise RuntimeError(f"expected one prepared cache revision, got {sorted(revisions)}")
    return revisions.pop()


def build_probe(config_path: Path) -> dict[str, Any]:
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required")
    if os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("HF_DATASETS_OFFLINE=1 is required")

    raw_config = config_path.read_bytes()
    config = json.loads(raw_config)
    workload = config["workload"]
    datasets = workload["datasets"]
    seeds = workload["seeds"]
    prompt_modes = workload["prompt_modes"]
    requests = int(workload["full_requests_per_cell"])

    dataset_receipts: list[dict[str, Any]] = []
    workload_receipts: list[dict[str, Any]] = []
    for dataset_name in datasets:
        registry = DATASET_REGISTRY[dataset_name]
        hf_name = registry["hf_name"]
        split = registry["default_split"]
        kwargs = dict(registry.get("hf_kwargs", {}))
        loaded = load_dataset(hf_name, split=split, **kwargs)
        dataset_receipts.append(
            {
                "dataset": dataset_name,
                "hf_name": hf_name,
                "split": split,
                "rows": len(loaded),
                "fingerprint": loaded._fingerprint,
                "prepared_revision": prepared_revision(loaded),
            }
        )
        for seed in seeds:
            for prompt_mode in prompt_modes:
                rows = load_public_text_rows(
                    dataset_name=dataset_name,
                    split=split,
                    limit=requests,
                    seed=int(seed),
                    prompt_mode=prompt_mode,
                )
                if len(rows) != requests:
                    raise RuntimeError(
                        f"{dataset_name}/{seed}/{prompt_mode}: "
                        f"expected {requests} rows, got {len(rows)}"
                    )
                workload_receipts.append(
                    {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "prompt_mode": prompt_mode,
                        "rows": len(rows),
                        "rows_sha256": canonical_sha256(rows),
                    }
                )

    core = {
        "datasets": dataset_receipts,
        "workloads": workload_receipts,
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "offline": True,
        "frozen_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "dataset_count": len(dataset_receipts),
        "workload_count": len(workload_receipts),
        "rows_probed": sum(row["rows"] for row in workload_receipts),
        "core_sha256": canonical_sha256(core),
        **core,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()

    receipt = build_probe(args.config)
    if args.expected:
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        if receipt["frozen_config_sha256"] != expected["frozen_config_sha256"]:
            raise RuntimeError("dataset manifest frozen-config hash mismatch")
        if receipt["datasets"] != expected["datasets"]:
            raise RuntimeError("dataset identity/fingerprint mismatch")
        if receipt["core_sha256"] != expected["core_sha256"]:
            raise RuntimeError("frozen workload row manifest mismatch")
        receipt["expected_manifest_sha256"] = hashlib.sha256(
            args.expected.read_bytes()
        ).hexdigest()
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
