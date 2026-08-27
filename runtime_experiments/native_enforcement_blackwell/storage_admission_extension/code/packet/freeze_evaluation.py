#!/usr/bin/env python3
"""Create the one-time evaluation freeze after calibration passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import verify_extension


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--packet", type=Path, default=HERE)
    args = parser.parse_args()
    packet = args.packet.resolve()
    results = args.results.resolve()
    control = results / "control"
    freeze_path = control / "evaluation_freeze.json"
    calibration_receipt_path = control / "calibration_receipt.json"
    if freeze_path.exists():
        existing = read_json(freeze_path)
        lfu = existing.get("lfu_optional_baseline") or {}
        if (
            existing.get("status") != "frozen"
            or existing.get("evaluation_plan") != verify_extension.EVALUATION_PLAN
            or existing.get("smoke_plan") != verify_extension.SMOKE_PLAN
            or existing.get("expected_cells") != 25
            or existing.get("expected_requests") != 2688
            or lfu.get("status") != "omitted_as_infeasible"
            or lfu.get("no_lfu_cells") is not True
            or lfu.get("receipt_sha256")
            != sha256(packet / verify_extension.LFU_RECEIPT)
            or lfu.get("crash_log_sha256")
            != sha256(packet / verify_extension.LFU_CRASH_LOG)
        ):
            raise SystemExit("existing evaluation freeze is invalid")
        print(
            json.dumps(
                {
                    "status": "already_frozen",
                    "evaluation_freeze": str(freeze_path),
                    "evaluation_freeze_sha256": sha256(freeze_path),
                },
                indent=2,
            )
        )
        return

    input_report = verify_extension.verify_inputs(packet)
    calibration_report = verify_extension.verify_phase(results, "calibration")
    calibration_cell = calibration_report["cells_detail"][0]
    hot = calibration_cell["phase_metrics"]
    established_hits = int(hot["establish_hot_hits"]["hot_cached_hits"])
    if calibration_cell["evicted_tokens"] <= 0:
        raise SystemExit("calibration did not produce physical evictions")
    if established_hits <= 0:
        raise SystemExit("calibration did not establish useful hot hits")

    calibration_cell_receipt = read_json(
        results
        / "calibration"
        / "seed_32000"
        / "native_admit_all_lru"
        / "cell_receipt.json"
    )
    computed_calibration_receipt = {
        "schema_version": 1,
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": 32000,
        "trace_sha256": calibration_cell_receipt["trace_sha256"],
        "disjoint_from_evaluation_seeds": True,
        "selected_max_total_tokens": 16384,
        "selection_rule": "single predeclared candidate; abort rather than retune",
        "hardware_specific_policy_threshold_adjustment_required": False,
        "policy_threshold": "preserved balanced policy",
        "evicted_tokens": calibration_cell["evicted_tokens"],
        "eviction_calls": calibration_cell["eviction_calls"],
        "established_hot_cached_hits": established_hits,
        "calibration_cell_receipt_sha256": sha256(
            results
            / "calibration"
            / "seed_32000"
            / "native_admit_all_lru"
            / "cell_receipt.json"
        ),
        "calibration_summary_sha256": calibration_cell["summary_sha256"],
        "calibration_trace_output_sha256": calibration_cell["trace_sha256"],
        "calibration_verification": calibration_report,
    }
    if calibration_receipt_path.exists():
        calibration_receipt = read_json(calibration_receipt_path)
        for key in (
            "status",
            "seed",
            "trace_sha256",
            "selected_max_total_tokens",
            "calibration_cell_receipt_sha256",
            "calibration_summary_sha256",
            "calibration_trace_output_sha256",
        ):
            if calibration_receipt.get(key) != computed_calibration_receipt.get(key):
                raise SystemExit(f"existing calibration receipt does not match live evidence: {key}")
    else:
        calibration_receipt = computed_calibration_receipt
        write_exclusive(calibration_receipt_path, calibration_receipt)

    bound_files = [
        "predeclared_protocol.json",
        "input_manifest.json",
        verify_extension.EVALUATION_PLAN,
        verify_extension.SMOKE_PLAN,
        verify_extension.LFU_RECEIPT,
        verify_extension.LFU_CRASH_LOG,
        verify_extension.LFU_SERVER_COMMANDS,
        "run_capacity_cell.py",
        "run_capacity_extension.py",
        "tests/test_lazy_eviction_metrics.py",
        "meritkv-capacity-pressure-20260822.service",
        "verify_extension.py",
        "freeze_evaluation.py",
        "generate_frozen_inputs.py",
        "measure_token_budget.py",
        "token_budget_manifest.json",
        *(f"traces/evaluation_seed_{seed}.jsonl" for seed in verify_extension.EVAL_SEEDS),
        "traces/smoke_seed_32010.jsonl",
    ]
    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_receipt_sha256": sha256(calibration_receipt_path),
        "calibration_seed": 32000,
        "evaluation_seeds": list(verify_extension.EVAL_SEEDS),
        "seeds_disjoint": True,
        "arms": list(verify_extension.ARMS),
        "model_id": "google/gemma-4-31B-it",
        "model_revision": "b9ea41a2887d8607f594846523f94c6cc75ac8a4",
        "runtime_image_id": "sha256:f4593e56ec7f7858f465a62f36dfc42ba8a2d9e91b7700cba50e8b21809e4d3f",
        "max_total_tokens": 16384,
        "policy_preset": "balanced",
        "policy_threshold_adjustment": None,
        "retuning_from_evaluation": False,
        "evaluation_plan": verify_extension.EVALUATION_PLAN,
        "smoke_plan": verify_extension.SMOKE_PLAN,
        "expected_cells": 25,
        "expected_requests": 2688,
        "lfu_optional_baseline": {
            "status": "omitted_as_infeasible",
            "receipt": verify_extension.LFU_RECEIPT,
            "receipt_sha256": sha256(packet / verify_extension.LFU_RECEIPT),
            "crash_log": verify_extension.LFU_CRASH_LOG,
            "crash_log_sha256": sha256(packet / verify_extension.LFU_CRASH_LOG),
            "no_lfu_cells": True,
        },
        "input_verification": input_report,
        "bound_files": {
            relative: sha256(packet / relative) for relative in bound_files
        },
        "bound_dependencies": {
            "meritkv_blackwell_20260820/run_meritkv_cell.py": sha256(
                packet.parent / "meritkv_blackwell_20260820" / "run_meritkv_cell.py"
            ),
            "meritkv_blackwell_20260820/trace_metrics.py": sha256(
                packet.parent / "meritkv_blackwell_20260820" / "trace_metrics.py"
            ),
            "meritkv_blackwell_20260817/run_meritkv_cell.py": sha256(
                packet.parent / "meritkv_blackwell_20260817" / "run_meritkv_cell.py"
            ),
            "FILE_MANIFEST_SHA256.txt": sha256(
                packet.parent / "FILE_MANIFEST_SHA256.txt"
            ),
        },
        "source_snapshot_files": {
            str(path.relative_to(packet.parent)): sha256(path)
            for path in sorted((packet.parent / "source_snapshot").rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        },
    }
    write_exclusive(freeze_path, freeze)
    print(
        json.dumps(
            {
                "status": "frozen",
                "calibration_receipt": str(calibration_receipt_path),
                "calibration_receipt_sha256": sha256(calibration_receipt_path),
                "evaluation_freeze": str(freeze_path),
                "evaluation_freeze_sha256": sha256(freeze_path),
            },
            indent=2,
        )
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
