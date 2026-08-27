#!/usr/bin/env python3
"""Relocatable verifier for the Gemma storage-admission extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ARMS = (
    "native_admit_all_lru",
    "meritkv_write_through_lru",
    "meritkv_skip_write_only_lru",
    "meritkv_joint_skip_write_skip_lookup_lru",
)
MERIT_ARMS = ARMS[1:4]
EVAL_SEEDS = (32001, 32002, 32003, 32004, 32005)
CALIBRATION_SEED = 32000
SMOKE_SEED = 32010
EVALUATION_PLAN = "evaluation_plan_20.tsv"
SMOKE_PLAN = "smoke_plan_4.tsv"
LFU_RECEIPT = "feasibility/lfu_infeasibility_receipt.json"
LFU_CRASH_LOG = "feasibility/lfu_launch_crash.log"
LFU_SERVER_COMMANDS = "feasibility/server_commands.jsonl"
RUNTIME_IMAGE_ID = "sha256:f4593e56ec7f7858f465a62f36dfc42ba8a2d9e91b7700cba50e8b21809e4d3f"
MODEL_REVISION = "b9ea41a2887d8607f594846523f94c6cc75ac8a4"


class VerificationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot read JSON {path}: {exc}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    fail(f"{path}:{number} is not an object")
                rows.append(row)
    except VerificationError:
        raise
    except Exception as exc:
        fail(f"cannot read JSONL {path}: {exc}")
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except Exception as exc:
        fail(f"cannot read TSV {path}: {exc}")


def verify_lfu_infeasibility(packet: Path) -> dict[str, Any]:
    """Verify the bound evidence for omitting Kushal's optional LFU baseline.

    The pinned Gemma image implements the working default cache as
    ``SWARadixCache``.  Its frequency selector is not an effective LFU baseline.
    The recorded attempt to force the ordinary Radix topology needed for an
    honest LFU comparison booted, then crashed on the first request.  This
    evidence is an infeasibility boundary, not an experimental cell.
    """

    receipt_path = packet / LFU_RECEIPT
    crash_path = packet / LFU_CRASH_LOG
    commands_path = packet / LFU_SERVER_COMMANDS
    for path in (receipt_path, crash_path, commands_path):
        if not path.is_file():
            fail(f"LFU infeasibility evidence is missing: {path.relative_to(packet)}")
    receipt = read_json(receipt_path)
    if receipt.get("schema_version") != 1:
        fail("LFU infeasibility receipt schema mismatch")
    if receipt.get("status") != "infeasible_on_pinned_image":
        fail("LFU infeasibility receipt does not record pinned-image infeasibility")
    if receipt.get("optional_baseline") != "LFU":
        fail("LFU infeasibility receipt names the wrong optional baseline")
    if receipt.get("runtime_image_id") != RUNTIME_IMAGE_ID:
        fail("LFU infeasibility receipt is not bound to the pinned runtime image")
    if receipt.get("model") != "google/gemma-4-31B-it":
        fail("LFU infeasibility receipt model mismatch")
    if receipt.get("revision") != MODEL_REVISION:
        fail("LFU infeasibility receipt model revision mismatch")
    if receipt.get("failure_stage") != "server_startup":
        fail("LFU infeasibility receipt has the wrong failed-launch stage")
    if receipt.get("no_successful_lfu_cells") is not True:
        fail("LFU infeasibility receipt does not exclude LFU result cells")
    if receipt.get("scientific_disposition") != "omitted_as_infeasible":
        fail("LFU infeasibility receipt has the wrong scientific disposition")
    if (
        receipt.get("crash_log") != LFU_CRASH_LOG
        or receipt.get("crash_log_sha256") != sha256(crash_path)
        or int(receipt.get("crash_log_bytes", -1)) != crash_path.stat().st_size
    ):
        fail("LFU receipt crash-log binding mismatch")
    if (
        receipt.get("server_commands_log") != LFU_SERVER_COMMANDS
        or receipt.get("server_commands_log_sha256") != sha256(commands_path)
        or int(receipt.get("server_commands_log_bytes", -1))
        != commands_path.stat().st_size
    ):
        fail("LFU receipt server-command binding mismatch")
    required_path = receipt.get("required_arm_path") or {}
    if (
        required_path.get("cache_implementation") != "SWARadixCache"
        or required_path.get("hybrid_swa") is not True
        or required_path.get("effective_retention") != "lru"
        or tuple(required_path.get("included_arms") or []) != ARMS
    ):
        fail("LFU receipt does not bind the four-arm SWARadixCache path")
    protocol_effect = receipt.get("protocol_effect") or {}
    if protocol_effect != {
        "calibration_cells": 1,
        "smoke_cells": 4,
        "evaluation_seeds": 5,
        "evaluation_cells": 20,
        "retuning_from_evaluation": False,
    }:
        fail("LFU receipt protocol effect mismatch")

    crash_text = crash_path.read_text(encoding="utf-8", errors="replace")
    for required in (
        f"models--google--gemma-4-31B-it/snapshots/{MODEL_REVISION}",
        "disable_hybrid_swa_memory=True",
        "impl=RadixCache",
        "hybrid_swa=False",
        "Scheduler hit an exception: Traceback",
        "RuntimeError: view size is not compatible with input tensor's size and stride",
    ):
        if required not in crash_text:
            fail(f"LFU infeasibility crash log lacks {required!r}")
    command_rows = read_jsonl(commands_path)
    if len(command_rows) != 1:
        fail("LFU infeasibility command evidence must contain exactly one launch")
    argv = command_rows[0].get("argv")
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        fail("LFU infeasibility command evidence lacks canonical argv")
    command = " ".join(argv)
    attempted = receipt.get("attempted_server_config")
    image_name = "shadowkv-sglang-native-admission:2026-08-20-swa-counters"
    try:
        container_image_index = argv.index(image_name)
    except ValueError:
        fail("LFU launch command does not contain the pinned image name")
    canonical_attempt = ["python3", *argv[container_image_index + 1 :]]
    if not isinstance(attempted, list) or attempted != canonical_attempt:
        fail("LFU receipt attempted server config is not bound to the launch log")
    for required in (
        "shadowkv-sglang-native-admission:2026-08-20-swa-counters",
        f"models--google--gemma-4-31B-it/snapshots/{MODEL_REVISION}",
        "--max-total-tokens 16384",
        "--page-size 1",
        "--disable-hybrid-swa-memory",
    ):
        if required not in command:
            fail(f"LFU infeasibility launch evidence lacks {required!r}")
    return {
        "status": "pass",
        "disposition": "omitted_as_infeasible",
        "receipt_sha256": sha256(receipt_path),
        "crash_log_sha256": sha256(crash_path),
        "server_commands_sha256": sha256(commands_path),
    }


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def verify_inputs(packet: Path = HERE) -> dict[str, Any]:
    protocol = read_json(packet / "predeclared_protocol.json")
    if protocol.get("preserve_prior_result_unchanged") is not True:
        fail("protocol does not preserve the prior result")
    arms = tuple(row["name"] for row in protocol.get("arms") or [])
    if arms != ARMS:
        fail(f"protocol arm mismatch: {arms}")
    candidates = protocol["runtime"].get("max_total_tokens_candidates")
    if candidates != [16384]:
        fail(f"capacity candidate set is not frozen: {candidates}")
    if protocol["evaluation"].get("retuning_from_evaluation") is not False:
        fail("evaluation retuning is not disabled")
    if set(protocol["evaluation"].get("seeds") or []) != set(EVAL_SEEDS):
        fail("evaluation seeds do not match the five predeclared seeds")
    if protocol["calibration"].get("seed") in EVAL_SEEDS:
        fail("calibration seed overlaps evaluation seeds")
    runtime = protocol.get("runtime") or {}
    if runtime.get("disable_hybrid_swa_memory") is not False:
        fail("protocol does not preserve Gemma's default hybrid SWA topology")
    if runtime.get("cache_implementation") != "SWARadixCache for every included arm":
        fail("protocol does not bind every included arm to SWARadixCache")
    if runtime.get("effective_retention") != "lru":
        fail("protocol does not bind the supported effective retention")
    registration = runtime.get("physical_eviction_metric_registration") or {}
    if (
        registration.get("mode") != "lazy_on_first_physical_eviction"
        or registration.get("primer_required_before_every_measured_cell") is not True
        or registration.get("calibration_before_may_be_absent") is not False
        or registration.get("calibration_after_required") is not True
        or registration.get("all_post_calibration_snapshots_required") is not True
    ):
        fail("protocol does not bind the lazy physical-eviction metric rule")
    lfu_infeasibility = verify_lfu_infeasibility(packet)

    input_manifest = read_json(packet / "input_manifest.json")
    if input_manifest.get("generator_sha256") != sha256(packet / "generate_frozen_inputs.py"):
        fail("input manifest is not bound to the trace generator")
    token_budget = read_json(packet / "token_budget_manifest.json")
    if token_budget.get("page_size") != 1:
        fail("token-budget manifest does not bind page_size=1")
    if token_budget.get("minimum_useful_hit_rule") != "cached_tokens >= shared_prefix_tokens":
        fail("token-budget manifest has the wrong useful-hit rule")
    manifest_traces = input_manifest.get("traces") or {}
    expected_trace_names = {
        f"traces/calibration_seed_{CALIBRATION_SEED}.jsonl",
        f"traces/smoke_seed_{SMOKE_SEED}.jsonl",
        *(f"traces/evaluation_seed_{seed}.jsonl" for seed in EVAL_SEEDS),
    }
    if set(manifest_traces) != expected_trace_names:
        fail("trace manifest inventory mismatch")

    trace_reports: dict[str, Any] = {}
    for name in sorted(expected_trace_names):
        path = packet / name
        rows = read_jsonl(path)
        record = manifest_traces[name]
        if sha256(path) != record.get("sha256"):
            fail(f"trace hash mismatch: {name}")
        budget = (token_budget.get("traces") or {}).get(name)
        if not isinstance(budget, dict) or budget.get("sha256") != record.get("sha256"):
            fail(f"token-budget receipt is not bound to trace: {name}")
        if len(rows) != int(record.get("requests", -1)):
            fail(f"trace request count mismatch: {name}")
        if [int(row.get("request_id", -1)) for row in rows] != list(range(len(rows))):
            fail(f"trace request IDs are not contiguous: {name}")
        phases = [str((row.get("metadata") or {}).get("phase")) for row in rows]
        required = {
            "bootstrap_hot_set",
            "establish_hot_hits",
            "capacity_pressure_with_hot_probes",
            "post_pressure_probe",
        }
        if not required.issubset(phases):
            fail(f"trace lacks required phases: {name}")
        if name.startswith("traces/evaluation_") and "recovery" not in phases:
            fail(f"evaluation trace lacks recovery phase: {name}")
        classes = Counter(
            str((row.get("metadata") or {}).get("shadowkv_trace_class")) for row in rows
        )
        if classes["long_one_off_prefix"] < 16:
            fail(f"trace lacks one-off pressure: {name}")
        hot_families = {
            int((row.get("metadata") or {}).get("hot_family"))
            for row in rows
            if (row.get("metadata") or {}).get("hot_family") is not None
        }
        if hot_families != {0, 1, 2, 3}:
            fail(f"trace hot-family inventory mismatch: {name}")
        if name.startswith("traces/evaluation_"):
            if budget.get("hot_set_tokens", 10**9) > 8192:
                fail(f"hot set exceeds half the frozen capacity: {name}")
            if budget.get("one_off_pressure_prompt_tokens", 0) <= 4 * 16384:
                fail(f"one-off pressure does not exceed four cache budgets: {name}")
            thresholds = budget.get("minimum_useful_cached_tokens_by_family") or {}
            if thresholds != {str(family): 1001 for family in range(4)}:
                fail(f"unexpected pinned hot-hit thresholds: {name}: {thresholds}")
        trace_reports[name] = {
            "sha256": sha256(path),
            "requests": len(rows),
            "classes": dict(classes),
            "phases": dict(Counter(phases)),
        }

    eval_plan = read_tsv(packet / EVALUATION_PLAN)
    if len(eval_plan) != 20:
        fail(f"evaluation plan must contain 20 cells, got {len(eval_plan)}")
    pairs = Counter((int(row["seed"]), row["arm"]) for row in eval_plan)
    if set(pairs.values()) != {1} or set(pairs) != {
        (seed, arm) for seed in EVAL_SEEDS for arm in ARMS
    }:
        fail("evaluation plan does not contain one cell per seed and arm")
    for row in eval_plan:
        trace_path = packet / row["trace"]
        expected = f"evaluation_seed_{row['seed']}.jsonl"
        if trace_path.name != expected or int(row["requests"]) != 124:
            fail(f"evaluation plan row is not paired to the frozen trace: {row}")
        if row["retention"] != "lru":
            fail(f"retention mismatch in plan row: {row}")
    for arm in ARMS:
        positions = sorted(int(row["order"]) for row in eval_plan if row["arm"] == arm)
        if len(positions) != 5 or set(positions) != {1, 2, 3, 4}:
            fail(f"arm is not position-balanced across seeds: {arm} -> {positions}")
    for seed in EVAL_SEEDS:
        positions = sorted(int(row["order"]) for row in eval_plan if int(row["seed"]) == seed)
        if positions != [1, 2, 3, 4]:
            fail(f"evaluation seed does not contain four unique positions: {seed}")

    smoke_plan = read_tsv(packet / SMOKE_PLAN)
    if len(smoke_plan) != 4 or {row["arm"] for row in smoke_plan} != set(ARMS):
        fail("smoke plan does not cover all four required arms")
    if any(row["retention"] != "lru" for row in smoke_plan):
        fail("smoke plan includes unsupported non-LRU retention")
    return {
        "status": "pass",
        "protocol_sha256": sha256(packet / "predeclared_protocol.json"),
        "input_manifest_sha256": sha256(packet / "input_manifest.json"),
        "token_budget_manifest_sha256": sha256(packet / "token_budget_manifest.json"),
        "evaluation_cells": len(eval_plan),
        "evaluation_seeds": list(EVAL_SEEDS),
        "arms": list(ARMS),
        "lfu_optional_baseline": lfu_infeasibility,
        "traces": trace_reports,
    }


def summary_and_metrics(cell: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    candidates = sorted(cell.glob("benchmark_*.json"))
    if len(candidates) != 1:
        fail(f"expected one benchmark summary in {cell}, found {len(candidates)}")
    summary = read_json(candidates[0])
    metric_keys = [key for key in summary if key.startswith("sglang_")]
    if len(metric_keys) != 1 or not isinstance(summary[metric_keys[0]], dict):
        fail(f"expected one SGLang metrics object in {candidates[0]}")
    return candidates[0], summary, summary[metric_keys[0]]


def counter_delta(metrics: dict[str, Any]) -> dict[str, float]:
    report = metrics.get("shadowkv_server_counters") or {}
    if report.get("available") is not True:
        fail("native action/capacity counters are unavailable")
    result: dict[str, float] = {}
    for key, value in (report.get("delta") or {}).items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[str(key)] = float(value)
    return result


def matching_counters(counters: dict[str, float], fragment: str) -> dict[str, float]:
    return {key: value for key, value in counters.items() if fragment in key}


def require_all_zero(values: dict[str, float], label: str) -> None:
    bad = {key: value for key, value in values.items() if abs(value) > 1e-9}
    if bad:
        fail(f"{label} leaked nonzero counters: {bad}")


def require_counter_value(
    counters: dict[str, float], fragment: str, expected: int, label: str
) -> dict[str, float]:
    values = matching_counters(counters, fragment)
    if not values:
        fail(f"{label} counter family is absent: {fragment}")
    bad = {key: value for key, value in values.items() if int(value) != expected}
    if bad:
        fail(f"{label} counter mismatch, expected {expected}: {bad}")
    return values


def useful_hot_hit(row: dict[str, Any]) -> bool:
    family = row.get("hot_family")
    if family is None:
        return False
    return int(row.get("cached_tokens") or 0) >= 1001


def phase_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_phase[str(row.get("phase"))].append(row)
    for phase, phase_rows in sorted(by_phase.items()):
        hot = [row for row in phase_rows if row.get("hot_probe")]
        cached_hot = [row for row in hot if useful_hot_hit(row)]
        result[phase] = {
            "requests": len(phase_rows),
            "hot_probes": len(hot),
            "hot_cached_hits": len(cached_hot),
            "hot_cached_hit_rate": len(cached_hot) / max(len(hot), 1),
            "ttft_mean_ms": sum(float(row["ttft_ms"]) for row in phase_rows)
            / max(len(phase_rows), 1),
            "ttft_p95_ms": percentile(
                (float(row["ttft_ms"]) for row in phase_rows), 0.95
            ),
            "end_to_end_mean_ms": sum(
                float(row["end_to_end_latency_ms"]) for row in phase_rows
            )
            / max(len(phase_rows), 1),
            "end_to_end_p95_ms": percentile(
                (float(row["end_to_end_latency_ms"]) for row in phase_rows), 0.95
            ),
        }

    post = [
        row
        for row in rows
        if row.get("phase") == "post_pressure_probe" and row.get("hot_family") is not None
    ]
    surviving = sorted(
        {
            int(row["hot_family"])
            for row in post
            if useful_hot_hit(row)
        }
    )
    recovery_rows = [row for row in rows if row.get("phase") == "recovery"]
    requests_to_recover: dict[str, int | None] = {}
    for family in range(4):
        post_hit = any(
            int(row.get("hot_family", -1)) == family and useful_hot_hit(row)
            for row in post
        )
        ordinal = 0 if post_hit else None
        family_recovery = [
            row for row in recovery_rows if int(row.get("hot_family", -1)) == family
        ]
        for family_probe, row in enumerate(family_recovery, 1):
            if useful_hot_hit(row):
                ordinal = family_probe
                break
        requests_to_recover[str(family)] = ordinal
    result["hot_set"] = {
        "post_pressure_surviving_families": surviving,
        "post_pressure_survival_count": len(surviving),
        "requests_to_recover_by_family": requests_to_recover,
        "complete_recovery": all(value is not None for value in requests_to_recover.values()),
    }
    return result


def verify_physical_eviction_primer(
    cell: Path, receipt: dict[str, Any]
) -> dict[str, Any]:
    relative = Path(str(receipt.get("physical_eviction_primer") or ""))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != ("control", "eviction_metric_primers")
        or relative.suffix != ".json"
    ):
        fail(f"cell does not bind a safe physical-eviction primer path: {cell}")
    results = cell.parents[2]
    path = results / relative
    if (
        not path.is_file()
        or sha256(path) != receipt.get("physical_eviction_primer_sha256")
    ):
        fail(f"physical-eviction primer receipt hash mismatch: {cell}")
    primer = read_json(path)
    if (
        primer.get("status") != "pass"
        or primer.get("purpose")
        != "non_measured_lazy_physical_eviction_metric_registration"
        or primer.get("runtime_image_id") != RUNTIME_IMAGE_ID
        or primer.get("server_config_sha256")
        != receipt.get("server_config_sha256")
        or primer.get("predeclared_protocol_sha256")
        != sha256(HERE / "predeclared_protocol.json")
        or primer.get("source_trace") != "traces/calibration_seed_32000.jsonl"
        or primer.get("source_trace_sha256")
        != sha256(HERE / "traces/calibration_seed_32000.jsonl")
        or primer.get("cache_flushed_after") is not True
        or primer.get("native_action_counters_reset_after") is not True
        or primer.get("excluded_from_calibration_smoke_and_evaluation") is not True
    ):
        fail(f"physical-eviction primer receipt invariants failed: {cell}")

    requests = primer.get("requests") or []
    if (
        not isinstance(requests, list)
        or int(primer.get("requests_executed", -1)) != len(requests)
        or not requests
    ):
        fail(f"physical-eviction primer request inventory is invalid: {cell}")
    source_rows = [
        row
        for row in read_jsonl(HERE / "traces/calibration_seed_32000.jsonl")
        if (row.get("metadata") or {}).get("shadowkv_trace_class")
        == "long_one_off_prefix"
    ]
    expected_requests = [
        {
            "request_id": row.get("request_id"),
            "input_prompt_sha256": hashlib.sha256(
                str(row["prompt"]).encode("utf-8")
            ).hexdigest(),
        }
        for row in source_rows[: len(requests)]
    ]
    actual_requests = [
        {
            "request_id": row.get("request_id"),
            "input_prompt_sha256": row.get("input_prompt_sha256"),
        }
        for row in requests
    ]
    if actual_requests != expected_requests or any(
        not isinstance(row.get("output_sha256"), str)
        or len(row["output_sha256"]) != 64
        for row in requests
    ):
        fail(f"physical-eviction primer is not bound to the frozen prompts: {cell}")

    required_definitions = {
        "sglang:evicted_tokens_total",
        "sglang:eviction_duration_seconds",
    }
    required_samples = {
        "sglang:evicted_tokens_total",
        "sglang:eviction_duration_seconds_count",
    }
    aggregates: dict[str, tuple[float, float]] = {}
    for name in ("registered", "post_flush"):
        snapshot = primer.get(name) or {}
        definitions = set(snapshot.get("definitions") or [])
        raw_samples = snapshot.get("samples") or {}
        if not isinstance(raw_samples, dict) or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in raw_samples.values()
        ):
            fail(f"physical-eviction primer has malformed {name} samples: {cell}")
        sample_names = {
            str(sample).split("{", 1)[0] for sample in raw_samples
        }
        evicted = sum(
            float(value)
            for sample, value in raw_samples.items()
            if str(sample).split("{", 1)[0] == "sglang:evicted_tokens_total"
        )
        calls = sum(
            float(value)
            for sample, value in raw_samples.items()
            if str(sample).split("{", 1)[0]
            == "sglang:eviction_duration_seconds_count"
        )
        if (
            definitions != required_definitions
            or not required_samples.issubset(sample_names)
            or not finite_positive(snapshot.get("evicted_tokens_total"))
            or not finite_positive(snapshot.get("eviction_duration_count"))
            or abs(float(snapshot["evicted_tokens_total"]) - evicted) > 1e-9
            or abs(float(snapshot["eviction_duration_count"]) - calls) > 1e-9
        ):
            fail(f"physical-eviction primer lacks real {name} telemetry: {cell}")
        aggregates[name] = (evicted, calls)
    before = primer.get("before") or {}
    before_evicted = before.get("evicted_tokens_total")
    before_calls = before.get("eviction_duration_count")
    before_samples = before.get("samples") or {}
    if (
        not isinstance(before_evicted, (int, float))
        or not math.isfinite(float(before_evicted))
        or float(before_evicted) < 0
        or not isinstance(before_calls, (int, float))
        or not math.isfinite(float(before_calls))
        or float(before_calls) < 0
        or not isinstance(before_samples, dict)
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in before_samples.values()
        )
    ):
        fail(f"physical-eviction primer has malformed before totals: {cell}")
    before_evicted_from_samples = sum(
        float(value)
        for sample, value in before_samples.items()
        if str(sample).split("{", 1)[0] == "sglang:evicted_tokens_total"
    )
    before_calls_from_samples = sum(
        float(value)
        for sample, value in before_samples.items()
        if str(sample).split("{", 1)[0]
        == "sglang:eviction_duration_seconds_count"
    )
    if (
        abs(float(before_evicted) - before_evicted_from_samples) > 1e-9
        or abs(float(before_calls) - before_calls_from_samples) > 1e-9
    ):
        fail(f"physical-eviction primer before sample totals are inconsistent: {cell}")
    recomputed_evicted = aggregates["registered"][0] - before_evicted_from_samples
    recomputed_calls = aggregates["registered"][1] - before_calls_from_samples
    delta = primer.get("physical_eviction_delta") or {}
    if (
        not finite_positive(delta.get("evicted_tokens"))
        or not finite_positive(delta.get("eviction_calls"))
        or abs(float(delta["evicted_tokens"]) - recomputed_evicted) > 1e-9
        or abs(float(delta["eviction_calls"]) - recomputed_calls) > 1e-9
        or aggregates["post_flush"][0] < aggregates["registered"][0]
        or aggregates["post_flush"][1] < aggregates["registered"][1]
    ):
        fail(f"physical-eviction primer delta/post-flush proof is inconsistent: {cell}")
    return primer


def verify_cell(cell: Path, phase: str, expected_seed: int, expected_arm: str) -> dict[str, Any]:
    receipt = read_json(cell / "cell_receipt.json")
    if receipt.get("status") != "pass":
        fail(f"cell receipt is not pass: {cell}")
    if receipt.get("phase") != phase or int(receipt.get("seed", -1)) != expected_seed:
        fail(f"cell phase/seed mismatch: {cell}")
    if receipt.get("arm") != expected_arm:
        fail(f"cell arm mismatch: {cell}")
    retention = "lru"
    if receipt.get("retention") != retention:
        fail(f"cell retention mismatch: {cell}")
    if int(receipt.get("max_total_tokens", -1)) != 16384:
        fail(f"cell capacity cap mismatch: {cell}")
    command = str(receipt.get("server_command") or "")
    server_argv = receipt.get("server_argv")
    if not isinstance(server_argv, list) or not all(isinstance(value, str) for value in server_argv):
        fail(f"cell lacks canonical server argv: {cell}")
    recomputed_server_hash = hashlib.sha256(
        json.dumps(server_argv, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if receipt.get("server_config_sha256") != recomputed_server_hash:
        fail(f"cell server-config hash does not match canonical argv: {cell}")
    for required in (
        "--model-path /hf_hub/models--google--gemma-4-31B-it/snapshots/b9ea41a2887d8607f594846523f94c6cc75ac8a4",
        "--served-model-name google/gemma-4-31B-it",
        "--context-length 4096",
        "--mem-fraction-static 0.88",
        "--max-total-tokens 16384",
        "--page-size 1",
        "--dtype float16",
        "--attention-backend triton",
        "--sampling-backend pytorch",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        f"--radix-eviction-policy {retention}",
        "--enable-cache-report",
        "--enable-metrics",
    ):
        if required not in command:
            fail(f"cell server command lacks {required!r}: {cell}")
    if "--disable-hybrid-swa-memory" in command:
        fail(f"cell disables the required Gemma hybrid SWA topology: {cell}")
    if receipt.get("cache_type") != "SWARadixCache" or receipt.get("hybrid_swa_memory") is not True:
        fail(f"cell does not prove the common SWARadixCache topology: {cell}")
    verify_physical_eviction_primer(cell, receipt)
    server_log = cell / "server_snapshot.log"
    if not server_log.is_file() or sha256(server_log) != receipt.get("server_log_sha256"):
        fail(f"cell server-log hash mismatch: {cell}")
    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    if "impl=SWARadixCache" not in log_text or "hybrid_swa=True" not in log_text:
        fail(f"cell startup log does not prove SWARadixCache/hybrid_swa=True: {cell}")
    lowered_log = log_text.lower()
    if any(term in lowered_log for term in ("out of memory", "oom-kill", "fatal", "traceback")):
        fail(f"cell server log contains fatal/OOM evidence: {cell}")
    if re.search(r"retracted[_ ]reqs?\s*[=:]\s*[1-9]", lowered_log):
        fail(f"cell server log contains request retractions: {cell}")
    expected_trace_relative = {
        "calibration": f"traces/calibration_seed_{CALIBRATION_SEED}.jsonl",
        "smoke": f"traces/smoke_seed_{SMOKE_SEED}.jsonl",
        "evaluation": f"traces/evaluation_seed_{expected_seed}.jsonl",
    }[phase]
    if receipt.get("trace_source") != expected_trace_relative:
        fail(f"cell points at the wrong frozen trace: {cell}")
    trace_source = HERE / expected_trace_relative
    input_manifest = read_json(HERE / "input_manifest.json")
    expected_trace_hash = input_manifest["traces"][expected_trace_relative]["sha256"]
    if (
        not trace_source.is_file()
        or sha256(trace_source) != expected_trace_hash
        or receipt.get("trace_sha256") != expected_trace_hash
    ):
        fail(f"cell input trace hash mismatch: {cell}")

    _, summary, metrics = summary_and_metrics(cell)
    extension = summary.get("meritkv_capacity_extension") or {}
    if extension.get("effective_arm") != expected_arm:
        fail(f"summary effective arm mismatch: {cell}")
    if extension.get("retuning_from_results") is not False:
        fail(f"cell permits result-driven retuning: {cell}")
    protocol_hash = sha256(HERE / "predeclared_protocol.json")
    if extension.get("predeclared_protocol_sha256") != protocol_hash:
        fail(f"cell protocol hash mismatch: {cell}")
    if extension.get("input_trace_sha256") != expected_trace_hash:
        fail(f"summary input-trace hash mismatch: {cell}")
    if int(extension.get("max_total_tokens", -1)) != 16384:
        fail(f"summary cache-cap mismatch: {cell}")
    if int(extension.get("page_size", -1)) != 1:
        fail(f"summary page-size mismatch: {cell}")
    if extension.get("retention_policy") != retention:
        fail(f"summary retention mismatch: {cell}")
    if extension.get("config_frozen") is not True:
        fail(f"summary is not marked config-frozen: {cell}")
    expected_image_id = RUNTIME_IMAGE_ID
    if extension.get("runtime_image_id") != expected_image_id:
        fail(f"summary runtime image mismatch: {cell}")
    if receipt.get("runtime_image_id") != expected_image_id:
        fail(f"cell receipt runtime image mismatch: {cell}")
    if (
        not extension.get("server_config_sha256")
        or extension.get("server_config_sha256") != receipt.get("server_config_sha256")
    ):
        fail(f"cell server-config hash mismatch: {cell}")
    cell_freeze_hash = receipt.get("evaluation_freeze_sha256")
    summary_freeze_hash = extension.get("evaluation_freeze_sha256")
    if phase == "calibration":
        if cell_freeze_hash not in {None, ""} or summary_freeze_hash not in {None, ""}:
            fail(f"calibration is improperly bound to an evaluation freeze: {cell}")
    else:
        freeze_path = results_root_from_cell(cell) / "control" / "evaluation_freeze.json"
        if not freeze_path.is_file():
            fail(f"evaluation freeze is missing for {phase} cell: {cell}")
        actual_freeze_hash = sha256(freeze_path)
        if cell_freeze_hash != actual_freeze_hash or summary_freeze_hash != actual_freeze_hash:
            fail(f"cell evaluation-freeze hash mismatch: {cell}")

    tokenizer = metrics.get("tokenizer_provenance") or {}
    if expected_arm.startswith("meritkv_"):
        if tokenizer.get("model_id") != "google/gemma-4-31B-it":
            fail(f"cell tokenizer model mismatch: {cell}")
        if tokenizer.get("manifest_sha256") != "202cc4cc84e68aa6ec005e5b35a0627b4de805cfe1df61d5c5fc5b1f25fcca0e":
            fail(f"cell tokenizer manifest mismatch: {cell}")
        if tokenizer.get("fallback_used") is not False or tokenizer.get("local_files_only") is not True:
            fail(f"cell tokenizer was not pinned offline: {cell}")

    rows = read_jsonl(cell / "meritkv_capacity_request_trace.jsonl")
    source_rows = read_jsonl(trace_source)
    if len(rows) != int(receipt.get("requests", -1)):
        fail(f"request-trace count mismatch: {cell}")
    if len(rows) != len(source_rows):
        fail(f"output and frozen source trace lengths differ: {cell}")
    if any(row.get("arm") != expected_arm for row in rows):
        fail(f"request-trace arm mismatch: {cell}")
    for index, (row, source) in enumerate(zip(rows, source_rows, strict=True)):
        metadata = dict(source.get("metadata") or {})
        expected_prompt_hash = hashlib.sha256(
            str(source.get("prompt") or "").encode("utf-8")
        ).hexdigest()
        expected_fields = {
            "request_id": int(source.get("request_id", index)),
            "phase": metadata.get("phase"),
            "shadowkv_trace_class": metadata.get("shadowkv_trace_class"),
            "reuse_family": metadata.get("reuse_family"),
            "hot_family": metadata.get("hot_family"),
            "hot_probe": bool(metadata.get("hot_probe")),
            "input_prompt_sha256": expected_prompt_hash,
            "sent_prompt_sha256": expected_prompt_hash,
        }
        actual_fields = {key: row.get(key) for key in expected_fields}
        if actual_fields != expected_fields:
            fail(
                f"request row {index} is not bound to the frozen source trace: "
                f"expected={expected_fields}, actual={actual_fields}"
            )
        if int(row.get("request_index", -1)) != index:
            fail(f"request trace index is not contiguous at {index}: {cell}")
        output_text = str(row.get("output_text") or "")
        if row.get("output_sha256") != hashlib.sha256(output_text.encode("utf-8")).hexdigest():
            fail(f"request output hash mismatch at {index}: {cell}")
        prompt_tokens = int(row.get("prompt_tokens") or 0)
        cached_tokens = int(row.get("cached_tokens") or 0)
        completion_tokens = int(row.get("completion_tokens") or 0)
        if prompt_tokens <= 0 or not 0 <= cached_tokens <= prompt_tokens:
            fail(f"request token accounting is invalid at {index}: {cell}")
        if completion_tokens != 1:
            fail(f"request did not return exactly one completion token at {index}: {cell}")
    capacity_report = metrics.get("shadowkv_server_counters") or {}
    required_capacity_definitions = {
        "sglang:evicted_tokens_total",
        "sglang:eviction_duration_seconds",
    }
    required_observed_samples = {
        "sglang:evicted_tokens_total",
        "sglang:eviction_duration_seconds_count",
    }
    for snapshot_name in ("before", "after"):
        snapshot = capacity_report.get(snapshot_name) or {}
        definitions = set(
            snapshot.get("prometheus_capacity_metric_definitions") or []
        )
        missing = required_capacity_definitions - definitions
        if missing:
            fail(
                f"{snapshot_name} snapshot lacks registered physical-eviction "
                f"metric definitions: {cell}"
            )
        observed = {
            str(sample).split("{", 1)[0]
            for sample in snapshot.get("prometheus_observed_samples") or []
        }
        if not required_observed_samples.issubset(observed):
            fail(
                f"{snapshot_name} snapshot lacks actually emitted physical-"
                f"eviction samples: {cell}"
            )
        if snapshot.get("prometheus_capacity_metrics_available") is not True:
            fail(f"{snapshot_name} physical-eviction metric capture failed: {cell}")
    counters = counter_delta(metrics)
    lookup_requested = matching_counters(counters, "skip_lookup_requested_total")
    lookup_executed = matching_counters(counters, "radix_skip_lookup_total")
    write_requested = matching_counters(counters, "skip_write_requested_total")
    write_finished = matching_counters(counters, "radix_skip_write_finished_total")
    write_unfinished = matching_counters(counters, "radix_skip_write_unfinished_total")
    bypass_rows = [row for row in rows if row.get("strategy") == "bypass"]
    bypasses = len(bypass_rows)

    if expected_arm == "native_admit_all_lru":
        if any("strategy" in row for row in rows):
            fail(f"native arm unexpectedly contains MeritKV decisions: {cell}")
        require_all_zero(lookup_requested | lookup_executed, "native skip lookup")
        require_all_zero(write_requested | write_finished | write_unfinished, "native skip write")
    elif expected_arm == "meritkv_write_through_lru":
        if bypasses <= 0:
            fail(f"write-through arm has no bypass decisions: {cell}")
        if any(row.get("requested_skip_lookup") or row.get("requested_skip_write") for row in rows):
            fail(f"write-through emitted native skip action: {cell}")
        require_all_zero(lookup_requested | lookup_executed, "write-through skip lookup")
        require_all_zero(write_requested | write_finished | write_unfinished, "write-through skip write")
    elif expected_arm == "meritkv_skip_write_only_lru":
        if bypasses <= 0:
            fail(f"skip-write-only arm has no bypass decisions: {cell}")
        if any(row.get("requested_skip_lookup") for row in rows):
            fail(f"skip-write-only requested skip lookup: {cell}")
        if sum(bool(row.get("requested_skip_write")) for row in rows) != bypasses:
            fail(f"skip-write-only request payload count mismatch: {cell}")
        require_all_zero(lookup_requested | lookup_executed, "skip-write-only skip lookup")
        require_counter_value(counters, "skip_write_requested_total", bypasses, "requested skip write")
        require_counter_value(counters, "radix_skip_write_finished_total", bypasses, "executed skip write")
        require_all_zero(write_unfinished, "unfinished skip write")
        if not any(row.get("hot_probe") and useful_hot_hit(row) for row in rows):
            fail(f"skip-write-only consumed no useful resident hot hit: {cell}")
    elif expected_arm == "meritkv_joint_skip_write_skip_lookup_lru":
        if bypasses <= 0:
            fail(f"joint arm has no bypass decisions: {cell}")
        if sum(bool(row.get("requested_skip_lookup")) for row in rows) != bypasses:
            fail(f"joint request skip-lookup count mismatch: {cell}")
        if sum(bool(row.get("requested_skip_write")) for row in rows) != bypasses:
            fail(f"joint request skip-write count mismatch: {cell}")
        require_counter_value(counters, "skip_lookup_requested_total", bypasses, "requested skip lookup")
        require_counter_value(counters, "radix_skip_lookup_total", bypasses, "executed skip lookup")
        require_counter_value(counters, "skip_write_requested_total", bypasses, "requested skip write")
        require_counter_value(counters, "radix_skip_write_finished_total", bypasses, "executed skip write")
        require_all_zero(write_unfinished, "unfinished joint skip write")
        if any(int(row.get("cached_tokens") or 0) != 0 for row in bypass_rows):
            fail(f"joint bypass consumed cached tokens: {cell}")

    evicted_series = {
        key: value
        for key, value in counters.items()
        if "prometheus::sglang:evicted_tokens_total" in key
    }
    eviction_call_series = {
        key: value
        for key, value in counters.items()
        if "prometheus::sglang:eviction_duration_seconds_count" in key
    }
    if not evicted_series or not eviction_call_series:
        fail(f"physical eviction telemetry series are absent: {cell}")
    evicted = sum(
        value
        for value in evicted_series.values()
    )
    eviction_calls = sum(
        value
        for value in eviction_call_series.values()
    )
    if phase in {"calibration", "evaluation"} and expected_arm == "native_admit_all_lru":
        if evicted <= 0 or eviction_calls <= 0:
            fail(f"native capacity-pressure cell has no physical eviction evidence: {cell}")
    for required in (
        "ttft_mean_ms",
        "ttft_p95_ms",
        "end_to_end_latency_mean_ms",
        "end_to_end_latency_p95_ms",
        "gpu_energy_j",
    ):
        value = metrics.get(required)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            fail(f"cell lacks required metric {required}: {cell}")

    phases = phase_metrics(rows)
    if phase == "evaluation":
        recovery = phases["hot_set"].get("requests_to_recover_by_family") or {}
        if set(recovery) != {"0", "1", "2", "3"}:
            fail(f"evaluation cell lacks per-family hot-set recovery evidence: {cell}")

    return {
        "status": "pass",
        "phase": phase,
        "seed": expected_seed,
        "arm": expected_arm,
        "requests": len(rows),
        "bypasses": bypasses,
        "policy_decision_digest": metrics.get("policy_decision_digest"),
        "evicted_tokens": evicted,
        "eviction_calls": eviction_calls,
        "ttft_mean_ms": metrics["ttft_mean_ms"],
        "ttft_p95_ms": metrics["ttft_p95_ms"],
        "end_to_end_latency_mean_ms": metrics["end_to_end_latency_mean_ms"],
        "end_to_end_latency_p95_ms": metrics["end_to_end_latency_p95_ms"],
        "gpu_energy_j": metrics["gpu_energy_j"],
        "cached_tokens_total": metrics.get("cached_tokens_total"),
        "phase_metrics": phases,
        "summary_sha256": sha256(summary_and_metrics(cell)[0]),
        "trace_sha256": sha256(cell / "meritkv_capacity_request_trace.jsonl"),
    }


def cell_path(results: Path, phase: str, seed: int, arm: str) -> Path:
    return results / phase / f"seed_{seed}" / arm


def results_root_from_cell(cell: Path) -> Path:
    # <results>/<phase>/seed_<n>/<arm>
    return cell.parents[2]


def verify_phase(results: Path, phase: str) -> dict[str, Any]:
    phase_root = results / phase
    lfu_cells = [
        str(path.relative_to(results))
        for path in phase_root.rglob("*_lfu")
        if path.is_dir()
    ]
    if lfu_cells:
        fail(f"unsupported LFU result cells are present: {sorted(lfu_cells)}")
    if phase == "smoke":
        expected = [(SMOKE_SEED, arm) for arm in ARMS]
    elif phase == "evaluation":
        expected = [(seed, arm) for seed in EVAL_SEEDS for arm in ARMS]
    elif phase == "calibration":
        expected = [(CALIBRATION_SEED, "native_admit_all_lru")]
    else:
        fail(f"unsupported phase: {phase}")
    cells = [
        verify_cell(cell_path(results, phase, seed, arm), phase, seed, arm)
        for seed, arm in expected
    ]
    if phase == "evaluation":
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for cell in cells:
            by_seed[int(cell["seed"])].append(cell)
        for seed, seed_cells in by_seed.items():
            digests = {
                cell["policy_decision_digest"]
                for cell in seed_cells
                if cell["arm"] in MERIT_ARMS
            }
            if len(digests) != 1 or None in digests:
                fail(f"MeritKV decision digests differ across arms for seed {seed}: {digests}")
        output_mismatches: list[dict[str, Any]] = []
        for seed in EVAL_SEEDS:
            arm_rows = {
                arm: read_jsonl(
                    cell_path(results, phase, seed, arm)
                    / "meritkv_capacity_request_trace.jsonl"
                )
                for arm in ARMS
            }
            for index in range(124):
                hashes = {arm: arm_rows[arm][index]["output_sha256"] for arm in ARMS}
                if len(set(hashes.values())) > 1:
                    output_mismatches.append(
                        {"seed": seed, "request_index": index, "output_sha256_by_arm": hashes}
                    )
        if output_mismatches:
            fail(
                f"exact output mismatch across arms at {len(output_mismatches)} "
                "paired request positions"
            )
        freeze = read_json(results / "control" / "evaluation_freeze.json")
        if freeze.get("status") != "frozen" or freeze.get("evaluation_seeds") != list(
            EVAL_SEEDS
        ):
            fail("evaluation freeze receipt is missing or invalid")
        if (
            tuple(freeze.get("arms") or []) != ARMS
            or freeze.get("evaluation_plan") != EVALUATION_PLAN
            or freeze.get("smoke_plan") != SMOKE_PLAN
            or freeze.get("expected_cells") != 25
            or freeze.get("expected_requests") != 2688
        ):
            fail("evaluation freeze does not bind the four-arm plan and exact totals")
        lfu = freeze.get("lfu_optional_baseline") or {}
        if (
            lfu.get("status") != "omitted_as_infeasible"
            or lfu.get("receipt") != LFU_RECEIPT
            or lfu.get("receipt_sha256") != sha256(HERE / LFU_RECEIPT)
            or lfu.get("crash_log") != LFU_CRASH_LOG
            or lfu.get("crash_log_sha256") != sha256(HERE / LFU_CRASH_LOG)
            or lfu.get("no_lfu_cells") is not True
        ):
            fail("evaluation freeze does not bind LFU infeasibility evidence")
        freeze_hash = sha256(results / "control" / "evaluation_freeze.json")
        for relative, expected_hash in (freeze.get("bound_files") or {}).items():
            bound = HERE / relative
            if not bound.is_file() or sha256(bound) != expected_hash:
                fail(f"frozen packet dependency drifted: {relative}")
        for relative, expected_hash in (freeze.get("bound_dependencies") or {}).items():
            bound = HERE.parent / relative
            if not bound.is_file() or sha256(bound) != expected_hash:
                fail(f"frozen imported dependency drifted: {relative}")
        source_files = freeze.get("source_snapshot_files") or {}
        if not source_files:
            fail("evaluation freeze does not bind the source_snapshot import closure")
        for relative, expected_hash in source_files.items():
            bound = HERE.parent / relative
            if not bound.is_file() or sha256(bound) != expected_hash:
                fail(f"frozen source_snapshot dependency drifted: {relative}")
        if any(
            read_json(
                cell_path(results, phase, int(cell["seed"]), str(cell["arm"]))
                / "cell_receipt.json"
            ).get("evaluation_freeze_sha256")
            != freeze_hash
            for cell in cells
        ):
            fail("an evaluation cell is not bound to the frozen evaluation receipt")
    report = {
        "status": "pass",
        "phase": phase,
        "cells": len(cells),
        "requests": sum(int(cell["requests"]) for cell in cells),
        "cells_detail": cells,
    }
    if phase == "evaluation":
        report["exact_output_mismatch_positions"] = len(output_mismatches)
        report["exact_output_mismatches"] = output_mismatches
    return report


def verify_all(packet: Path, results: Path) -> dict[str, Any]:
    inputs = verify_inputs(packet)
    calibration = verify_phase(results, "calibration")
    smoke = verify_phase(results, "smoke")
    evaluation = verify_phase(results, "evaluation")
    cells = calibration["cells"] + smoke["cells"] + evaluation["cells"]
    requests = calibration["requests"] + smoke["requests"] + evaluation["requests"]
    if cells != 25 or requests != 2688:
        fail(f"full experiment totals mismatch: cells={cells}, requests={requests}")
    return {
        "status": "pass",
        "experiment": "meritkv_gemma4_capacity_pressure_20260822",
        "prior_result_preservation_declared": True,
        "inputs": inputs,
        "phases": [calibration, smoke, evaluation],
        "cells": cells,
        "requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inputs_parser = sub.add_parser("inputs")
    inputs_parser.add_argument("--packet", type=Path, default=HERE)
    phase_parser = sub.add_parser("phase")
    phase_parser.add_argument("phase", choices=["calibration", "smoke", "evaluation"])
    phase_parser.add_argument("results", type=Path)
    cell_parser = sub.add_parser("cell")
    cell_parser.add_argument("phase", choices=["calibration", "smoke", "evaluation"])
    cell_parser.add_argument("results", type=Path)
    cell_parser.add_argument("seed", type=int)
    cell_parser.add_argument("arm", choices=ARMS)
    all_parser = sub.add_parser("all")
    all_parser.add_argument("results", type=Path)
    all_parser.add_argument("--packet", type=Path, default=HERE)
    args = parser.parse_args()
    try:
        if args.command == "inputs":
            report = verify_inputs(args.packet.resolve())
        elif args.command == "phase":
            report = verify_phase(args.results.resolve(), args.phase)
        elif args.command == "cell":
            results = args.results.resolve()
            report = verify_cell(
                cell_path(results, args.phase, args.seed, args.arm),
                args.phase,
                args.seed,
                args.arm,
            )
        else:
            report = verify_all(args.packet.resolve(), args.results.resolve())
    except VerificationError as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2))
        raise SystemExit(1)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
