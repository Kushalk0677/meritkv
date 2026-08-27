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
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
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
        lfu = existing.get("lfu_frequency_aware_baseline") or {}
        lfu_report = verify_extension.verify_lfu_feasibility(results)
        if (
            existing.get("status") != "frozen"
            or existing.get("evaluation_plan") != verify_extension.EVALUATION_PLAN
            or existing.get("smoke_plan") != verify_extension.SMOKE_PLAN
            or existing.get("expected_cells") != verify_extension.EXPECTED_CELLS
            or existing.get("expected_requests") != verify_extension.EXPECTED_REQUESTS
            or lfu.get("status") != "feasible_included"
            or lfu.get("runtime_receipt") != verify_extension.LFU_RUNTIME_RECEIPT
            or lfu.get("runtime_receipt_sha256") != lfu_report["receipt_sha256"]
            or lfu.get("evaluated_cells") != 6
        ):
            raise SystemExit("existing evaluation freeze is invalid")
        if (
            not calibration_receipt_path.is_file()
            or existing.get("calibration_receipt_sha256")
            != sha256(calibration_receipt_path)
        ):
            raise SystemExit("existing evaluation freeze lost its calibration binding")
        for relative, expected_hash in (existing.get("bound_files") or {}).items():
            path = packet / relative
            if not path.is_file() or sha256(path) != expected_hash:
                raise SystemExit(f"existing evaluation freeze packet drift: {relative}")
        for relative, expected_hash in (existing.get("bound_dependencies") or {}).items():
            path = packet.parent / relative
            if not path.is_file() or sha256(path) != expected_hash:
                raise SystemExit(f"existing evaluation freeze dependency drift: {relative}")
        source_files = existing.get("source_snapshot_files") or {}
        if not source_files:
            raise SystemExit("existing evaluation freeze lacks source import closure")
        for relative, expected_hash in source_files.items():
            path = packet.parent / relative
            if not path.is_file() or sha256(path) != expected_hash:
                raise SystemExit(f"existing evaluation freeze source drift: {relative}")
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
    lfu_report = verify_extension.verify_lfu_feasibility(results)
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
    lfu_completed = verify_extension.parse_timestamp(
        lfu_report["completed_at"], "LFU feasibility completed_at"
    )
    calibration_started = verify_extension.parse_timestamp(
        calibration_cell_receipt.get("started_at"), "calibration started_at"
    )
    if lfu_completed > calibration_started:
        raise SystemExit("calibration began before the excluded LFU feasibility gate passed")
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
        verify_extension.LFU_GATE,
        "run_capacity_cell.py",
        "run_capacity_extension.py",
        "tests/test_lazy_eviction_metrics.py",
        "verify_extension.py",
        "freeze_evaluation.py",
        "build_deliverables.py",
        "generate_frozen_inputs.py",
        "measure_token_budget.py",
        "token_budget_manifest.json",
        "traces/calibration_seed_32000.jsonl",
        *(f"traces/evaluation_seed_{seed}.jsonl" for seed in verify_extension.EVAL_SEEDS),
        "traces/smoke_seed_32010.jsonl",
    ]
    for unit in (
        "meritkv-qwen25-capacity-pressure-20260823.service",
        "meritkv-qwen25-capacity-pressure-20260823.timer",
    ):
        if not (packet / unit).is_file():
            raise SystemExit(f"missing frozen systemd unit: {unit}")
        bound_files.append(unit)
    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_receipt_sha256": sha256(calibration_receipt_path),
        "calibration_seed": 32000,
        "evaluation_seeds": list(verify_extension.EVAL_SEEDS),
        "seeds_disjoint": True,
        "arms": list(verify_extension.ARMS),
        "experiment": "meritkv_qwen25_capacity_pressure_20260824",
        "model_id": verify_extension.MODEL_ID,
        "model_revision": verify_extension.MODEL_REVISION,
        "runtime_image_id": verify_extension.RUNTIME_IMAGE_ID,
        "cache_implementation": "RadixCache",
        "hybrid_swa": False,
        "max_total_tokens": 16384,
        "policy_preset": "balanced",
        "policy_threshold_adjustment": None,
        "retuning_from_evaluation": False,
        "evaluation_plan": verify_extension.EVALUATION_PLAN,
        "smoke_plan": verify_extension.SMOKE_PLAN,
        "expected_cells": verify_extension.EXPECTED_CELLS,
        "expected_requests": verify_extension.EXPECTED_REQUESTS,
        "execution_order_claim": {
            "paired_seed_coverage": True,
            "chronological_plan_order_required": True,
            "evaluation_cells_non_overlapping": True,
            "smoke_plan_order_required": True,
            "smoke_cells_non_overlapping": True,
            "all_five_smoke_cells_before_evaluation": True,
            "note": "Cell timestamps must match the exact evaluation_plan_25.tsv row order.",
        },
        "lfu_frequency_aware_baseline": {
            "status": "feasible_included",
            "arm": "native_admit_all_lfu",
            "static_gate": verify_extension.LFU_GATE,
            "static_gate_sha256": sha256(packet / verify_extension.LFU_GATE),
            "runtime_receipt": verify_extension.LFU_RUNTIME_RECEIPT,
            "runtime_receipt_sha256": lfu_report["receipt_sha256"],
            "excluded_preflight_completed_at": lfu_report["completed_at"],
            "evaluated_cells": 6,
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
