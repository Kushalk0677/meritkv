#!/usr/bin/env python3
"""Aggregate and audit the corrected Gemma 4 native/ShadowKV++ matrix."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_full"
OUT = ROOT / "analysis"

MODELS = [
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
]
DATASETS = ["daily_dialog", "samsum", "ag_news", "dolly", "xsum"]
MODES = ["templated", "rag"]
ARMS = [
    "vllm_apc",
    "vllm_apc_shadowkv_plus",
    "sglang_radix_attention",
    "sglang_radix_attention_shadowkv_plus",
    "lmcache",
    "lmcache_shadowkv_plus",
]
PAIRS = {
    "vllm_apc": "vllm_apc_shadowkv_plus",
    "sglang_radix_attention": "sglang_radix_attention_shadowkv_plus",
    "lmcache": "lmcache_shadowkv_plus",
}


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def fmt(value: float, digits: int = 2) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def metric_block(payload: dict) -> tuple[str, dict]:
    keys = [key for key in payload if key not in {"config", "runtime", "idle_energy_baseline", "idle_stabilization"}]
    if len(keys) != 1:
        raise ValueError(f"Expected one metric block, found {keys}")
    return keys[0], payload[keys[0]]


def cache_evidence(arm: str, metrics: dict) -> float:
    counters = metrics.get("vllm_cache_metrics", {})
    if arm.startswith("vllm_apc"):
        return float(counters.get("local_cache_hit_tokens_delta", metrics.get("vllm_local_cache_hit_tokens_delta", 0)) or 0)
    if arm.startswith("lmcache"):
        return float(counters.get("external_prefix_cache_hits_delta", metrics.get("vllm_external_prefix_cache_hits_delta", 0)) or 0)
    return float(metrics.get("cached_tokens_total", 0) or 0)


OUT.mkdir(exist_ok=True)
rows: list[dict] = []
for path in sorted(RESULTS.rglob("benchmark_*.json")):
    payload = json.loads(path.read_text())
    arm, metrics = metric_block(payload)
    cfg = payload["config"]
    rows.append(
        {
            "model": cfg["model"],
            "dataset": cfg["dataset"],
            "mode": cfg["resolved_prompt_mode"],
            "arm": arm,
            "requests": metrics.get("requests_seen"),
            "prompt_tokens": metrics.get("prompt_tokens_total"),
            "cache_evidence_tokens": cache_evidence(arm, metrics),
            "mean_latency_ms": metrics.get("end_to_end_latency_mean_ms"),
            "p95_latency_ms": metrics.get("end_to_end_latency_p95_ms"),
            "throughput_rps": metrics.get("end_to_end_throughput_rps"),
            "gpu_energy_j": metrics.get("gpu_energy_j"),
            "idle_adjusted_gpu_energy_j": metrics.get("idle_adjusted_gpu_energy_j"),
            "gpu_joules_per_request": metrics.get("gpu_joules_per_request"),
            "energy_source": metrics.get("energy_source"),
            "energy_error": metrics.get("energy_error"),
            "reuse_failures": metrics.get("reuse_failures"),
            "idle_stable": payload.get("idle_stabilization", {}).get("stable"),
            "admission_enabled": metrics.get("admission_controller_enabled"),
            "admission_plans": metrics.get("admission_plans_total"),
            "admission_allows": metrics.get("admission_allow_total"),
            "admission_bypasses": metrics.get("admission_bypass_total"),
            "admission_reset_failures": metrics.get("admission_runtime_cache_reset_failures"),
            "admission_mode": metrics.get("admission_enforcement_mode"),
            "source_file": str(path.relative_to(ROOT)),
        }
    )

with (OUT / "all_cells.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

expected = {
    (model, dataset, mode, arm)
    for model in MODELS
    for dataset in DATASETS
    for mode in MODES
    for arm in ARMS
}
actual = {(row["model"], row["dataset"], row["mode"], row["arm"]) for row in rows}
duplicates = Counter((row["model"], row["dataset"], row["mode"], row["arm"]) for row in rows)

grouped: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
for row in rows:
    grouped[(row["model"], row["dataset"], row["mode"])][row["arm"]] = row

pair_rows: list[dict] = []
for cell, arms in sorted(grouped.items()):
    for native, overlay in PAIRS.items():
        if native not in arms or overlay not in arms:
            continue
        base = arms[native]
        candidate = arms[overlay]
        pair_rows.append(
            {
                "model": cell[0],
                "dataset": cell[1],
                "mode": cell[2],
                "runtime": native,
                "native_arm": native,
                "overlay_arm": overlay,
                "native_mean_latency_ms": base["mean_latency_ms"],
                "overlay_mean_latency_ms": candidate["mean_latency_ms"],
                "latency_ratio_overlay_vs_native": candidate["mean_latency_ms"] / base["mean_latency_ms"],
                "native_p95_latency_ms": base["p95_latency_ms"],
                "overlay_p95_latency_ms": candidate["p95_latency_ms"],
                "p95_ratio_overlay_vs_native": candidate["p95_latency_ms"] / base["p95_latency_ms"],
                "throughput_ratio_overlay_vs_native": candidate["throughput_rps"] / base["throughput_rps"],
                "energy_ratio_overlay_vs_native": candidate["gpu_energy_j"] / base["gpu_energy_j"],
                "overlay_allows": candidate["admission_allows"],
                "overlay_bypasses": candidate["admission_bypasses"],
            }
        )

with (OUT / "paired_native_vs_shadowkv_plus.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
    writer.writeheader()
    writer.writerows(pair_rows)

summary_rows: list[dict] = []
for model in MODELS:
    for arm in ARMS:
        subset = [row for row in rows if row["model"] == model and row["arm"] == arm]
        summary_rows.append(
            {
                "model": model,
                "arm": arm,
                "cells": len(subset),
                "mean_latency_ms": mean([row["mean_latency_ms"] for row in subset]),
                "mean_p95_latency_ms": mean([row["p95_latency_ms"] for row in subset]),
                "mean_throughput_rps": mean([row["throughput_rps"] for row in subset]),
                "mean_gpu_energy_j": mean([row["gpu_energy_j"] for row in subset]),
                "total_cache_evidence_tokens": sum(row["cache_evidence_tokens"] for row in subset),
                "total_allows": sum(row["admission_allows"] or 0 for row in subset),
                "total_bypasses": sum(row["admission_bypasses"] or 0 for row in subset),
            }
        )

with (OUT / "summary_by_model_arm.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
    writer.writeheader()
    writer.writerows(summary_rows)

pair_summary: list[dict] = []
for native in PAIRS:
    subset = [row for row in pair_rows if row["runtime"] == native]
    pair_summary.append(
        {
            "runtime": native,
            "pairs": len(subset),
            "mean_latency_delta_pct": 100 * (mean([row["latency_ratio_overlay_vs_native"] for row in subset]) - 1),
            "mean_p95_delta_pct": 100 * (mean([row["p95_ratio_overlay_vs_native"] for row in subset]) - 1),
            "mean_throughput_delta_pct": 100 * (mean([row["throughput_ratio_overlay_vs_native"] for row in subset]) - 1),
            "mean_energy_delta_pct": 100 * (mean([row["energy_ratio_overlay_vs_native"] for row in subset]) - 1),
            "latency_wins": sum(row["latency_ratio_overlay_vs_native"] < 1 for row in subset),
            "p95_wins": sum(row["p95_ratio_overlay_vs_native"] < 1 for row in subset),
            "total_allows": sum(row["overlay_allows"] or 0 for row in subset),
            "total_bypasses": sum(row["overlay_bypasses"] or 0 for row in subset),
        }
    )

with (OUT / "paired_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(pair_summary[0]))
    writer.writeheader()
    writer.writerows(pair_summary)

pair_model_summary: list[dict] = []
for model in MODELS:
    for native in PAIRS:
        subset = [row for row in pair_rows if row["model"] == model and row["runtime"] == native]
        pair_model_summary.append(
            {
                "model": model,
                "runtime": native,
                "pairs": len(subset),
                "mean_latency_delta_pct": 100 * (mean([row["latency_ratio_overlay_vs_native"] for row in subset]) - 1),
                "mean_p95_delta_pct": 100 * (mean([row["p95_ratio_overlay_vs_native"] for row in subset]) - 1),
                "mean_throughput_delta_pct": 100 * (mean([row["throughput_ratio_overlay_vs_native"] for row in subset]) - 1),
                "mean_energy_delta_pct": 100 * (mean([row["energy_ratio_overlay_vs_native"] for row in subset]) - 1),
                "latency_wins": sum(row["latency_ratio_overlay_vs_native"] < 1 for row in subset),
                "p95_wins": sum(row["p95_ratio_overlay_vs_native"] < 1 for row in subset),
            }
        )

with (OUT / "paired_summary_by_model.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(pair_model_summary[0]))
    writer.writeheader()
    writer.writerows(pair_model_summary)

plan_rows = []
with (ROOT / "block_plan_full.tsv").open() as handle:
    for line in handle:
        index, slug, model, arm = line.rstrip("\n").split("\t")
        if index == "block_index":
            continue
        plan_rows.append({"index": int(index), "arm": arm, "model_slug": slug, "model": model})

plan_position = {(row["arm"], row["model"]): row["index"] for row in plan_rows}
order_rows = []
for model in MODELS:
    for native, overlay in PAIRS.items():
        native_index = plan_position[(native, model)]
        overlay_index = plan_position[(overlay, model)]
        order_rows.append(
            {
                "model": model,
                "runtime": native,
                "native_block_index": native_index,
                "overlay_block_index": overlay_index,
                "first_arm": native if native_index < overlay_index else overlay,
            }
        )

with (OUT / "execution_order_audit.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(order_rows[0]))
    writer.writeheader()
    writer.writerows(order_rows)

lmcache_logs = sorted((RESULTS / "server_logs").glob("*_lmcache*.lmcache.log"))
lmcache_log_evidence = []
for path in lmcache_logs:
    text = path.read_text(errors="replace")
    stores = re.findall(r"Stored (\d+) tokens", text)
    retrieves = re.findall(r"Retrieved (\d+) tokens", text)
    lmcache_log_evidence.append(
        {
            "log": path.name,
            "store_events": len(stores),
            "stored_tokens": sum(map(int, stores)),
            "retrieve_events": len(retrieves),
            "retrieved_tokens": sum(map(int, retrieves)),
        }
    )

with (OUT / "lmcache_log_evidence.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(lmcache_log_evidence[0]))
    writer.writeheader()
    writer.writerows(lmcache_log_evidence)

issues: list[str] = []
if len(rows) != 300:
    issues.append(f"Expected 300 result files, found {len(rows)}")
if expected - actual:
    issues.append(f"Missing {len(expected - actual)} expected cells")
if actual - expected:
    issues.append(f"Found {len(actual - expected)} unexpected cells")
if any(count != 1 for count in duplicates.values()):
    issues.append("Found duplicate/non-unique cells")
if len(pair_rows) != 150:
    issues.append(f"Expected 150 native/overlay pairs, found {len(pair_rows)}")
if len(plan_rows) != 30 or len({(row['arm'], row['model']) for row in plan_rows}) != 30:
    issues.append("Full block plan is incomplete or non-unique")

for row in rows:
    label = f"{row['model']} {row['dataset']} {row['mode']} {row['arm']}"
    if row["requests"] != 256:
        issues.append(f"Request count is {row['requests']} for {label}")
    if row["energy_source"] != "nvml" or row["gpu_energy_j"] is None or row["energy_error"]:
        issues.append(f"Invalid energy evidence for {label}")
    if row["reuse_failures"] != 0:
        issues.append(f"Reuse failures={row['reuse_failures']} for {label}")
    if row["arm"].endswith("shadowkv_plus"):
        if not row["admission_enabled"]:
            issues.append(f"Admission controller disabled for {label}")
        if row["admission_plans"] != 256:
            issues.append(f"Admission plans={row['admission_plans']} for {label}")
        if (row["admission_allows"] or 0) + (row["admission_bypasses"] or 0) != 256:
            issues.append(f"Admission decisions incomplete for {label}")
        if row["admission_reset_failures"] != 0:
            issues.append(f"Admission reset failures for {label}")
        if row["admission_mode"] != "write_through_admission":
            issues.append(f"Unexpected admission mode for {label}")

for model in MODELS:
    for arm in ARMS:
        evidence = sum(row["cache_evidence_tokens"] for row in rows if row["model"] == model and row["arm"] == arm)
        if evidence <= 0:
            issues.append(f"No positive aggregate cache evidence for {model} on {arm}")

if len(lmcache_logs) != 10:
    issues.append(f"Expected 10 LMCache transfer logs, found {len(lmcache_logs)}")
for item in lmcache_log_evidence:
    if not item["store_events"] or not item["retrieve_events"]:
        issues.append(f"Missing LMCache store/retrieve evidence in {item['log']}")

zero_cache = [row for row in rows if row["cache_evidence_tokens"] <= 0]
unstable = [row for row in rows if row["idle_stable"] is False]
zero_groups = Counter((row["arm"], row["dataset"]) for row in zero_cache)

lines = [
    "# Corrected Gemma 4 Runtime + ShadowKV++ Matrix Audit",
    "",
    "## Verification",
    "",
    f"- Result cells: {len(rows)}/300",
    f"- Unique expected cells: {len(actual & expected)}/300",
    f"- Native/overlay pairs: {len(pair_rows)}/150",
    f"- Requests: {sum(row['requests'] for row in rows):,}; all cells at 256: {all(row['requests'] == 256 for row in rows)}",
    f"- NVML energy complete and error-free: {all(row['energy_source'] == 'nvml' and row['gpu_energy_j'] is not None and not row['energy_error'] for row in rows)}",
    f"- Overlay plans and decisions complete: {all(not row['arm'].endswith('shadowkv_plus') or (row['admission_plans'] == 256 and (row['admission_allows'] or 0) + (row['admission_bypasses'] or 0) == 256) for row in rows)}",
    f"- Overlay runtime-cache reset failures: {sum(row['admission_reset_failures'] or 0 for row in rows)}",
    f"- Reuse failures: {sum(row['reuse_failures'] for row in rows)}",
    f"- Idle stabilization timeouts: {len(unstable)}",
    f"- LMCache logs with stores and retrieves: {sum(item['store_events'] > 0 and item['retrieve_events'] > 0 for item in lmcache_log_evidence)}/{len(lmcache_log_evidence)}",
    f"- Randomized block plan: {len(plan_rows)}/30 unique model/arm blocks",
    "",
    "## Native Versus ShadowKV++",
    "",
    "| Runtime | Pairs | Mean latency delta | Mean P95 delta | Throughput delta | Energy delta | Latency wins | P95 wins | Allows | Bypasses |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for item in pair_summary:
    lines.append(
        f"| {item['runtime']} | {item['pairs']} | {item['mean_latency_delta_pct']:+.2f}% | "
        f"{item['mean_p95_delta_pct']:+.2f}% | {item['mean_throughput_delta_pct']:+.2f}% | "
        f"{item['mean_energy_delta_pct']:+.2f}% | {item['latency_wins']}/{item['pairs']} | "
        f"{item['p95_wins']}/{item['pairs']} | {item['total_allows']:,} | {item['total_bypasses']:,} |"
    )

lines += [
    "",
    "## Caveats And Anomalies",
    "",
    "- The ShadowKV++ arms are portable write-through policy overlays. Planning, server request, and feedback are included in end-to-end timing, but the external runtimes retain cache ownership.",
    "- A bypass records the policy decision but does not enforce native per-request skip-lookup/skip-write behavior. These arms are policy-observer measurements, not native runtime-hook implementations.",
    "- One randomized block schedule and one run per cell were used. Small deltas are not replicated performance estimates.",
    "- Runtime builds and memory settings differ by system; exact metadata and image inspection records are included.",
    f"- Individual zero-cache-evidence cells: {len(zero_cache)}.",
]
if zero_groups:
    summary = ", ".join(f"{arm}/{dataset}: {count}" for (arm, dataset), count in sorted(zero_groups.items()))
    lines.append(f"- Zero-cache evidence is concentrated in {summary}. These requests completed normally; for LMCache, AG News reusable prefixes were below the configured 256-token external-cache chunk boundary.")
if unstable:
    lines.append(f"- {len(unstable)} cells hit the idle-stabilization timeout. NVML energy is present, but those cells should be interpreted cautiously.")
else:
    lines.append("- All cells met the configured idle-power stabilization criterion.")

lines += ["", "## Audit Result", ""]
if issues:
    lines.append(f"**FAIL: {len(issues)} issue(s) found.**")
    lines.extend(f"- {issue}" for issue in issues)
else:
    lines.append("**PASS: coverage, request counts, energy, cache evidence, admission counters, transfer logs, and schedule checks passed.**")

(OUT / "ANOMALY_AUDIT.md").write_text("\n".join(lines) + "\n")
(OUT / "audit.json").write_text(
    json.dumps(
        {
            "pass": not issues,
            "issues": issues,
            "result_cells": len(rows),
            "expected_cells": 300,
            "native_overlay_pairs": len(pair_rows),
            "total_requests": sum(row["requests"] for row in rows),
            "idle_stabilization_timeouts": len(unstable),
            "zero_cache_cells": len(zero_cache),
            "zero_cache_groups": {f"{arm}/{dataset}": count for (arm, dataset), count in zero_groups.items()},
            "pair_summary": pair_summary,
            "lmcache_log_evidence": lmcache_log_evidence,
            "block_plan": plan_rows,
        },
        indent=2,
    )
    + "\n"
)
print(f"audit_pass={not issues} cells={len(rows)} pairs={len(pair_rows)} issues={len(issues)}")
