#!/usr/bin/env python3
"""Strict, pinned instrumentation wrapper for the corrected MeritKV packet."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from trace_metrics import synchronize_measured_timings


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LEGACY_PATH = PROJECT / "meritkv_blackwell_20260817" / "run_meritkv_cell.py"
spec = importlib.util.spec_from_file_location("meritkv_legacy_wrapper", LEGACY_PATH)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load preserved wrapper {LEGACY_PATH}")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

adapter = legacy.adapter
base = legacy.base
ARM = legacy.ARM
DECISIONS = legacy.DECISIONS
CALLS = legacy.CALLS
TOKENIZER_PROVENANCE: dict[str, Any] = {}


def strict_controller_init(self, model: str, runtime: str, min_match_length: int = 8) -> None:
    from proactive_kv_cache.cache import TieredStateBank
    from transformers import AutoTokenizer

    expected_model = os.environ.get("MERITKV_MODEL_ID")
    snapshot = os.environ.get("MERITKV_TOKENIZER_SNAPSHOT")
    manifest_hash = os.environ.get("MERITKV_TOKENIZER_MANIFEST_SHA256")
    if not expected_model or model != expected_model:
        raise RuntimeError(f"policy model mismatch: {model!r} != {expected_model!r}")
    if not snapshot or not manifest_hash:
        raise RuntimeError("pinned tokenizer snapshot and manifest hash are required")
    self.runtime = runtime
    self.bank = TieredStateBank(
        max_memory_bytes=1024 * 1024, min_match_length=min_match_length
    )
    self.controller = adapter.AdaptiveReuseController()
    self.full_ms_per_token = 0.35
    self.reuse_overhead_ms = 1.0
    self._calibration_alpha = 0.20
    self._token_to_id = {}
    self._tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    TOKENIZER_PROVENANCE.clear()
    TOKENIZER_PROVENANCE.update(
        {
            "model_id": model,
            "snapshot": snapshot,
            "manifest_sha256": manifest_hash,
            "class": type(self._tokenizer).__name__,
            "fallback_used": False,
            "local_files_only": True,
        }
    )


def streaming_invoke(*args, **kwargs):
    result = legacy.streaming_invoke(*args, **kwargs)
    if CALLS:
        CALLS[-1]["output_text"] = str(getattr(result, "output_text", ""))
    return result


def traced_plan(self, prompt: str, metadata=None):
    import time

    started = time.perf_counter()
    plan, tokens = legacy.original_plan(self, prompt, metadata=metadata)
    DECISIONS.append(
        {
            "request_index": len(CALLS),
            "strategy": str(plan.strategy),
            "reason": str(plan.reason),
            "utility_score": float(plan.score),
            "expected_benefit_ms": float(plan.expected_benefit_ms),
            "expected_cost_ms": float(plan.expected_cost_ms),
            "expected_waste_ms": float(plan.expected_waste_ms),
            "confidence": float(plan.confidence),
            "layer_reuse_ratio": float(plan.layer_reuse_ratio),
            "reusable_prefix_tokens": int(plan.reusable_prefix_tokens or 0),
            "token_count": len(tokens),
            "shared_prefix_hint_tokens": (metadata or {}).get("shared_prefix_hint_tokens"),
            "shadowkv_trace_class": (metadata or {}).get("shadowkv_trace_class"),
            "reuse_family": (metadata or {}).get("reuse_family"),
            "skip_lookup": plan.strategy == "bypass" and ARM == "meritkv_enforced",
            "skip_write": plan.strategy == "bypass" and ARM == "meritkv_enforced",
            "planning_latency_ms": (time.perf_counter() - started) * 1000.0,
        }
    )
    return plan, tokens


def enriched_summarize(results, *, total_wall_time_s=None, extra_metrics=None):
    synchronize_measured_timings(
        CALLS,
        DECISIONS,
        results,
        require_decisions=ARM.startswith("meritkv_"),
    )
    summary = legacy.enriched_summarize(
        results,
        total_wall_time_s=total_wall_time_s,
        extra_metrics=extra_metrics,
    )
    decision_rows = [
        {
            key: row[key]
            for key in (
                "strategy",
                "reason",
                "utility_score",
                "expected_benefit_ms",
                "expected_cost_ms",
                "expected_waste_ms",
                "confidence",
                "layer_reuse_ratio",
                "reusable_prefix_tokens",
            )
        }
        for row in DECISIONS
    ]
    summary["policy_decision_digest"] = hashlib.sha256(
        json.dumps(decision_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary["policy_utility_breakdown_fields"] = [
        "utility_score",
        "expected_benefit_ms",
        "expected_cost_ms",
        "expected_waste_ms",
        "confidence",
        "layer_reuse_ratio",
    ]
    summary["tokenizer_provenance"] = dict(TOKENIZER_PROVENANCE)
    return summary


def enriched_save(output_dir: str, filename: str, payload: dict[str, Any]) -> Path:
    metric_objects = [
        value
        for key, value in payload.items()
        if key.startswith("sglang_") and isinstance(value, dict)
    ]
    if ARM.startswith("meritkv_") and len(metric_objects) == 1:
        metrics = metric_objects[0]
        logical_stores = int(metrics.get("admission_bypass_store_successes") or 0)
        metrics["admission_bypass_store_successes_scope"] = (
            "policy_controller_bookkeeping_only"
        )
        metrics["admission_policy_bookkeeping_bypass_store_successes"] = logical_stores
        if ARM == "meritkv_enforced":
            metrics["admission_physical_bypass_store_permitted_total"] = int(
                metrics.get("admission_native_bypass_store_allowed_total") or 0
            )
    payload["meritkv_stage"] = {
        "arm": ARM,
        "config_frozen": True,
        "retuning_from_results": False,
        "raw_command": " ".join(sys.argv),
        "frozen_config_sha256": os.environ.get("MERITKV_FROZEN_CONFIG_SHA256"),
        "runtime_image_id": os.environ.get("MERITKV_RUNTIME_IMAGE_ID"),
        "server_config_sha256": os.environ.get("MERITKV_SERVER_CONFIG_SHA256"),
        "tokenizer_provenance": dict(TOKENIZER_PROVENANCE),
    }
    return legacy.original_save(output_dir, filename, payload)


adapter.ExternalAdmissionController.__init__ = strict_controller_init
adapter.OpenAICompatClient.invoke = streaming_invoke
base.OpenAICompatClient.invoke = streaming_invoke
adapter.ExternalAdmissionController.plan = traced_plan
base.ExternalAdmissionController.plan = traced_plan
base.summarize_external_results = enriched_summarize
base.save_summary = enriched_save


def argument_int(flag: str, default: int) -> int:
    value = legacy._argument_value(flag, str(default))
    return int(value)


if __name__ == "__main__":
    base.main()
    output_dir = Path(legacy._argument_value("--output_dir", "results_external"))
    measured_requests = argument_int("--n_requests", 64)
    measured_calls = CALLS[-measured_requests:]
    if len(measured_calls) != measured_requests:
        raise SystemExit(
            f"measured call count {len(measured_calls)} != {measured_requests}"
        )
    if ARM.startswith("meritkv_") and len(DECISIONS) != measured_requests:
        raise SystemExit(
            f"decision count {len(DECISIONS)} != {measured_requests}"
        )
    rows = []
    for index, call in enumerate(measured_calls):
        row = dict(call)
        row["request_index"] = index
        if index < len(DECISIONS):
            row.update(DECISIONS[index])
            row["request_index"] = index
        if ARM.startswith("meritkv_"):
            row["policy_bookkeeping_stored_after_request"] = bool(
                row.get("stored_after_request")
            )
            row["stored_after_request_scope"] = "policy_controller_bookkeeping_only"
            if ARM == "meritkv_enforced":
                row["physical_store_permitted"] = not (
                    row.get("strategy") == "bypass"
                )
        row["arm"] = ARM
        rows.append(row)
    trace_path = output_dir / "meritkv_request_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
