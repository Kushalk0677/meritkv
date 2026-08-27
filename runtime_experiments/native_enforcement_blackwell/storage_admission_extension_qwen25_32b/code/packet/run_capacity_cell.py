#!/usr/bin/env python3
"""Qwen2.5 capacity-pressure client with independent skip-write/lookup actions.

The preserved August wrapper provides pinned tokenization, decision tracing,
timing, and energy measurement. This extension changes the actual native
request action for the skip-write-only arm and captures SGLang's physical
eviction metrics. It does not rewrite native counters after execution.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
STRICT_WRAPPER = PROJECT / "meritkv_blackwell_20260820" / "run_meritkv_cell.py"
EFFECTIVE_ARM = os.environ.get("MERITKV_EFFECTIVE_ARM", "")
RETENTION = os.environ.get("MERITKV_RETENTION_POLICY", "")
VALID_ARMS = {
    "native_admit_all_lru",
    "meritkv_write_through_lru",
    "meritkv_skip_write_only_lru",
    "meritkv_joint_skip_write_skip_lookup_lru",
    "native_admit_all_lfu",
}
UNDERLYING_ARMS = {
    "native_admit_all_lru": "native_baseline",
    "meritkv_write_through_lru": "meritkv_write_through",
    "meritkv_skip_write_only_lru": "meritkv_enforced",
    "meritkv_joint_skip_write_skip_lookup_lru": "meritkv_enforced",
    "native_admit_all_lfu": "native_baseline",
}
if EFFECTIVE_ARM not in VALID_ARMS:
    raise SystemExit(f"MERITKV_EFFECTIVE_ARM must be one of {sorted(VALID_ARMS)}")
EXPECTED_RETENTION = "lfu" if EFFECTIVE_ARM == "native_admit_all_lfu" else "lru"
if RETENTION != EXPECTED_RETENTION:
    raise SystemExit(
        f"MERITKV_RETENTION_POLICY={RETENTION!r} does not match "
        f"{EFFECTIVE_ARM} ({EXPECTED_RETENTION})"
    )

# The preserved wrapper validates this environment variable at import time.
os.environ["MERITKV_ARM"] = UNDERLYING_ARMS[EFFECTIVE_ARM]
sys.path.insert(0, str(STRICT_WRAPPER.parent))
spec = importlib.util.spec_from_file_location("meritkv_strict_wrapper", STRICT_WRAPPER)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load preserved strict wrapper {STRICT_WRAPPER}")
strict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strict)

adapter = strict.adapter
base = strict.base
legacy = strict.legacy
CALLS = strict.CALLS
DECISIONS = strict.DECISIONS
ORIGINAL_COLLECT = base.collect_shadowkv_admission_metrics
ORIGINAL_EXTRA_KEY = adapter.make_sglang_native_admission_extra_key


def traced_plan(self, prompt: str, metadata=None):
    import time

    started = time.perf_counter()
    plan, tokens = legacy.original_plan(self, prompt, metadata=metadata)
    bypass = str(plan.strategy) == "bypass"
    skip_write = bypass and EFFECTIVE_ARM in {
        "meritkv_skip_write_only_lru",
        "meritkv_joint_skip_write_skip_lookup_lru",
    }
    skip_lookup = bypass and EFFECTIVE_ARM == "meritkv_joint_skip_write_skip_lookup_lru"
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
            "phase": (metadata or {}).get("phase"),
            "reuse_family": (metadata or {}).get("reuse_family"),
            "hot_family": (metadata or {}).get("hot_family"),
            "hot_probe": bool((metadata or {}).get("hot_probe")),
            "skip_lookup": skip_lookup,
            "skip_write": skip_write,
            "planning_latency_ms": (time.perf_counter() - started) * 1000.0,
        }
    )
    return plan, tokens


def capacity_extra_key(
    *,
    skip_lookup: bool,
    skip_write: bool,
    original_extra_key: str | None = None,
    tag: str | None = None,
) -> str:
    if EFFECTIVE_ARM == "meritkv_skip_write_only_lru":
        skip_lookup, skip_write = False, True
    elif EFFECTIVE_ARM == "meritkv_joint_skip_write_skip_lookup_lru":
        skip_lookup, skip_write = True, True
    return ORIGINAL_EXTRA_KEY(
        skip_lookup=skip_lookup,
        skip_write=skip_write,
        original_extra_key=original_extra_key,
        tag=tag,
    )


def capacity_invoke(
    self,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    extra_body: dict[str, Any] | None = None,
):
    result = strict.streaming_invoke(
        self,
        prompt,
        max_tokens,
        temperature,
        timeout_s,
        extra_body=extra_body,
    )
    extra_key = str((extra_body or {}).get("extra_key") or "")
    if CALLS:
        CALLS[-1]["request_extra_key"] = extra_key or None
        CALLS[-1]["requested_skip_lookup"] = "lookup=0" in extra_key
        CALLS[-1]["requested_skip_write"] = "write=0" in extra_key
        CALLS[-1]["sent_prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return result


PROM_SAMPLE = re.compile(r"^(?P<sample>[^ ]+)\s+(?P<value>[-+0-9.eE]+)$")


def collect_capacity_metrics(client):
    report = ORIGINAL_COLLECT(client)
    values = dict(report.get("values") or {})
    errors: list[str] = []
    definitions: set[str] = set()
    observed_samples: set[str] = set()
    zero_initialized: list[str] = []
    required_definitions = {
        "sglang:evicted_tokens_total",
        "sglang:eviction_duration_seconds",
    }
    required_samples = {
        "sglang:evicted_tokens_total",
        "sglang:eviction_duration_seconds_count",
        "sglang:eviction_duration_seconds_sum",
    }
    missing_definitions = sorted(required_definitions)
    try:
        text = client.get_text("/metrics", timeout_s=10.0)
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("# HELP "):
                parts = line.split(None, 3)
                if len(parts) >= 3:
                    definitions.add(parts[2])
                continue
            if not line or line.startswith("#"):
                continue
            match = PROM_SAMPLE.match(line)
            if not match:
                continue
            sample = match.group("sample")
            metric_name = sample.split("{", 1)[0]
            if metric_name not in required_samples:
                continue
            values[f"prometheus::{sample}"] = float(match.group("value"))
            observed_samples.add(sample)
        missing_definitions = sorted(required_definitions - definitions)
        for metric_name in sorted(required_samples):
            if any(
                key.split("{", 1)[0] == f"prometheus::{metric_name}"
                for key in values
            ):
                continue
            definition = metric_name.removesuffix("_count").removesuffix("_sum")
            if definition in definitions:
                key = f'prometheus::{metric_name}{{cache_type="RadixCache"}}'
                values[key] = 0.0
                zero_initialized.append(metric_name)
    except Exception as exc:  # retained in evidence and rejected by the verifier
        errors.append(str(exc))
    report["values"] = values
    report["prometheus_capacity_metric_definitions"] = sorted(definitions)
    report["prometheus_capacity_metric_definitions_missing"] = missing_definitions
    report["prometheus_observed_samples"] = sorted(observed_samples)
    report["prometheus_lazy_registration_observed"] = bool(missing_definitions)
    report["prometheus_zero_initialized_samples"] = zero_initialized
    report["prometheus_zero_rule"] = (
        "A registered labeled Prometheus counter/histogram with no emitted child "
        "sample has not been incremented; record its value as zero. Before the "
        "first physical eviction only, an absent lazily registered family is zero "
        "only when the paired after snapshot proves that family and its samples."
    )
    report["prometheus_capacity_metrics_available"] = not errors
    report["prometheus_capacity_metrics_errors"] = errors
    report["available"] = bool(report.get("available")) and not errors
    return report


def capacity_save(output_dir: str, filename: str, payload: dict[str, Any]) -> Path:
    requested_lookup = sum(bool(row.get("skip_lookup")) for row in DECISIONS)
    requested_write = sum(bool(row.get("skip_write")) for row in DECISIONS)
    for key, metrics in payload.items():
        if not key.startswith("sglang_") or not isinstance(metrics, dict):
            continue
        metrics["admission_legacy_runner_assumed_skip_lookup_total"] = metrics.pop(
            "admission_native_skip_lookup_total", 0
        )
        metrics["admission_legacy_runner_assumed_skip_write_total"] = metrics.pop(
            "admission_native_skip_write_total", 0
        )
        metrics["meritkv_arm"] = EFFECTIVE_ARM
        metrics["admission_effective_skip_lookup_requested_total"] = requested_lookup
        metrics["admission_effective_skip_write_requested_total"] = requested_write
        metrics["capacity_pressure_retention_policy"] = RETENTION
        metrics["capacity_pressure_cache_implementation"] = "RadixCache"
        metrics["capacity_pressure_hybrid_swa_memory"] = False
        metrics["capacity_pressure_max_total_tokens"] = int(
            os.environ["MERITKV_MAX_TOTAL_TOKENS"]
        )
        metrics["evaluation_retuning"] = False
    payload["meritkv_capacity_extension"] = {
        "effective_arm": EFFECTIVE_ARM,
        "underlying_preserved_arm": UNDERLYING_ARMS[EFFECTIVE_ARM],
        "native_action_mapping": {
            "bypass_skip_lookup": EFFECTIVE_ARM
            == "meritkv_joint_skip_write_skip_lookup_lru",
            "bypass_skip_write": EFFECTIVE_ARM
            in {
                "meritkv_skip_write_only_lru",
                "meritkv_joint_skip_write_skip_lookup_lru",
            },
        },
        "retention_policy": RETENTION,
        "cache_implementation": "RadixCache",
        "hybrid_swa_memory": False,
        "max_total_tokens": int(os.environ["MERITKV_MAX_TOTAL_TOKENS"]),
        "page_size": int(os.environ.get("MERITKV_PAGE_SIZE", "1")),
        "runtime_image_id": os.environ.get("MERITKV_RUNTIME_IMAGE_ID"),
        "server_config_sha256": os.environ.get("MERITKV_SERVER_CONFIG_SHA256"),
        "predeclared_protocol_sha256": os.environ.get("MERITKV_PROTOCOL_SHA256"),
        "evaluation_freeze_sha256": os.environ.get("MERITKV_EVALUATION_FREEZE_SHA256"),
        "input_trace_sha256": os.environ.get("MERITKV_INPUT_TRACE_SHA256"),
        "config_frozen": True,
        "retuning_from_results": False,
        "raw_command": " ".join(sys.argv),
    }
    return legacy.original_save(output_dir, filename, payload)


adapter.ExternalAdmissionController.plan = traced_plan
base.ExternalAdmissionController.plan = traced_plan
adapter.OpenAICompatClient.invoke = capacity_invoke
base.OpenAICompatClient.invoke = capacity_invoke
base.make_sglang_native_admission_extra_key = capacity_extra_key
base.collect_shadowkv_admission_metrics = collect_capacity_metrics
base.save_summary = capacity_save


def argument_value(flag: str, default: str) -> str:
    return legacy._argument_value(flag, default)


def load_input_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    base.main()
    output_dir = Path(argument_value("--output_dir", "results_external"))
    trace_path = Path(argument_value("--trace_path", ""))
    measured_requests = int(argument_value("--n_requests", "0"))
    input_rows = load_input_rows(trace_path)
    if len(input_rows) != measured_requests:
        raise SystemExit(
            f"input trace length {len(input_rows)} != measured requests {measured_requests}"
        )
    measured_calls = CALLS[-measured_requests:]
    if len(measured_calls) != measured_requests:
        raise SystemExit(
            f"measured call count {len(measured_calls)} != {measured_requests}"
        )
    if EFFECTIVE_ARM.startswith("meritkv_") and len(DECISIONS) != measured_requests:
        raise SystemExit(
            f"decision count {len(DECISIONS)} != {measured_requests}"
        )

    rows: list[dict[str, Any]] = []
    for index, (call, source) in enumerate(zip(measured_calls, input_rows, strict=True)):
        row = dict(call)
        row["request_index"] = index
        metadata = dict(source.get("metadata") or {})
        row.update(
            {
                "request_id": int(source.get("request_id", index)),
                "phase": metadata.get("phase"),
                "shadowkv_trace_class": metadata.get("shadowkv_trace_class"),
                "reuse_family": metadata.get("reuse_family"),
                "hot_family": metadata.get("hot_family"),
                "hot_probe": bool(metadata.get("hot_probe")),
                "input_prompt_sha256": hashlib.sha256(
                    str(source["prompt"]).encode("utf-8")
                ).hexdigest(),
            }
        )
        if index < len(DECISIONS):
            row.update(DECISIONS[index])
            row["request_index"] = index
        row["arm"] = EFFECTIVE_ARM
        row["physical_store_permitted"] = not bool(row.get("requested_skip_write"))
        rows.append(row)

    trace_output = output_dir / "meritkv_capacity_request_trace.jsonl"
    trace_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
