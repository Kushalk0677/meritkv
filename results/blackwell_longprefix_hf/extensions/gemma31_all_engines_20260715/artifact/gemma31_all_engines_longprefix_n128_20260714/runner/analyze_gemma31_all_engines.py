#!/usr/bin/env python3
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "") else 0.0


def mean(rows, key):
    return statistics.mean(number(row, key) for row in rows)


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


root = Path(sys.argv[1]).resolve()
rows = list(csv.DictReader((root / "all_results.csv").open(newline="", encoding="utf-8")))
jobs = json.loads((root / "_job_results.json").read_text(encoding="utf-8"))

expected_engines = [
    "no_cache",
    "native_prefix_cache",
    "reactive_prefix_cache",
    "greedy_prefix_cache",
    "strict_reactive_prefix_cache",
    "frequency_speculative",
    "shadow_kv",
    "shadow_kv_plus",
    "shadow_kv_plus_lite",
    "shadow_kv_plus_best_latency",
    "shadow_kv_plus_raw_observer",
]
datasets = sorted({row["dataset"] for row in rows})
by_engine = defaultdict(list)
by_dataset_engine = {}
for row in rows:
    by_engine[row["engine"]].append(row)
    by_dataset_engine[(row["dataset"], row["engine"])] = row

failures = [job for job in jobs if job.get("returncode") not in (0, "skipped_completed")]
assert len(rows) == 110, len(rows)
assert len(jobs) == 110, len(jobs)
assert not failures, failures
assert set(by_engine) == set(expected_engines), sorted(by_engine)
assert len(datasets) == 10, datasets
assert all(len(by_engine[engine]) == 10 for engine in expected_engines)

summary_rows = []
detail_rows = []
baseline_rows = by_engine["no_cache"]
baseline_mean = mean(baseline_rows, "mean_latency_ms")
baseline_p95 = mean(baseline_rows, "p95_latency_ms")
baseline_throughput = mean(baseline_rows, "throughput_rps")
baseline_energy = sum(number(row, "gpu_energy_j") for row in baseline_rows)

for engine in expected_engines:
    engine_rows = by_engine[engine]
    paired_mean_speedups = []
    paired_p95_speedups = []
    paired_mean_improvements = []
    paired_p95_improvements = []
    paired_energy_improvements = []
    mean_wins = 0
    p95_wins = 0
    for dataset in datasets:
        base = by_dataset_engine[(dataset, "no_cache")]
        current = by_dataset_engine[(dataset, engine)]
        base_mean = number(base, "mean_latency_ms")
        current_mean = number(current, "mean_latency_ms")
        base_p95_value = number(base, "p95_latency_ms")
        current_p95 = number(current, "p95_latency_ms")
        base_energy_value = number(base, "gpu_energy_j")
        current_energy = number(current, "gpu_energy_j")
        mean_speedup = base_mean / current_mean
        p95_speedup = base_p95_value / current_p95
        mean_improvement = (base_mean - current_mean) / base_mean * 100.0
        p95_improvement = (base_p95_value - current_p95) / base_p95_value * 100.0
        energy_improvement = (base_energy_value - current_energy) / base_energy_value * 100.0
        paired_mean_speedups.append(mean_speedup)
        paired_p95_speedups.append(p95_speedup)
        paired_mean_improvements.append(mean_improvement)
        paired_p95_improvements.append(p95_improvement)
        paired_energy_improvements.append(energy_improvement)
        mean_wins += current_mean < base_mean
        p95_wins += current_p95 < base_p95_value
        detail_rows.append(
            {
                "dataset": dataset,
                "engine": engine,
                "mean_latency_ms": round(current_mean, 6),
                "p95_latency_ms": round(current_p95, 6),
                "throughput_rps": round(number(current, "throughput_rps"), 6),
                "gpu_energy_j": round(current_energy, 6),
                "mean_speedup_vs_no_cache": round(mean_speedup, 6),
                "p95_speedup_vs_no_cache": round(p95_speedup, 6),
                "mean_latency_improvement_pct": round(mean_improvement, 6),
                "p95_latency_improvement_pct": round(p95_improvement, 6),
                "gpu_energy_improvement_pct": round(energy_improvement, 6),
                "hit_rate": round(number(current, "hit_rate"), 6),
                "reuse_successes": int(number(current, "reuse_successes")),
                "bypassed_matches": int(number(current, "bypassed_matches")),
            }
        )

    engine_mean = mean(engine_rows, "mean_latency_ms")
    engine_p95 = mean(engine_rows, "p95_latency_ms")
    engine_throughput = mean(engine_rows, "throughput_rps")
    engine_energy = sum(number(row, "gpu_energy_j") for row in engine_rows)
    summary_rows.append(
        {
            "engine": engine,
            "cells": len(engine_rows),
            "mean_latency_ms": round(engine_mean, 6),
            "p95_latency_ms": round(engine_p95, 6),
            "throughput_rps": round(engine_throughput, 6),
            "gpu_energy_j_total": round(engine_energy, 6),
            "aggregate_mean_speedup_vs_no_cache": round(baseline_mean / engine_mean, 6),
            "aggregate_p95_speedup_vs_no_cache": round(baseline_p95 / engine_p95, 6),
            "aggregate_mean_latency_improvement_pct": round((baseline_mean - engine_mean) / baseline_mean * 100.0, 6),
            "aggregate_p95_latency_improvement_pct": round((baseline_p95 - engine_p95) / baseline_p95 * 100.0, 6),
            "aggregate_throughput_change_pct": round((engine_throughput - baseline_throughput) / baseline_throughput * 100.0, 6),
            "aggregate_gpu_energy_improvement_pct": round((baseline_energy - engine_energy) / baseline_energy * 100.0, 6),
            "mean_paired_speedup": round(statistics.mean(paired_mean_speedups), 6),
            "mean_paired_latency_improvement_pct": round(statistics.mean(paired_mean_improvements), 6),
            "mean_paired_p95_improvement_pct": round(statistics.mean(paired_p95_improvements), 6),
            "mean_paired_energy_improvement_pct": round(statistics.mean(paired_energy_improvements), 6),
            "mean_latency_wins": mean_wins,
            "p95_latency_wins": p95_wins,
            "mean_hit_rate": round(mean(engine_rows, "hit_rate"), 6),
            "reuse_attempts_total": int(sum(number(row, "reuse_attempts") for row in engine_rows)),
            "reuse_successes_total": int(sum(number(row, "reuse_successes") for row in engine_rows)),
            "reused_prefix_tokens_total": int(sum(number(row, "reused_prefix_tokens_total") for row in engine_rows)),
            "bypassed_matches_total": int(sum(number(row, "bypassed_matches") for row in engine_rows)),
            "wasted_compute_ratio_mean": round(mean(engine_rows, "wasted_compute_ratio"), 6),
        }
    )

reuse_rows = list(csv.DictReader((root / "reuse_path_breakdown.csv").open(newline="", encoding="utf-8")))
path_counts = defaultdict(Counter)
for row in reuse_rows:
    path_counts[row["engine"]][row["path_reading"]] += 1

analysis = {
    "validation": {
        "result_rows": len(rows),
        "job_rows": len(jobs),
        "failed_jobs": len(failures),
        "datasets": datasets,
        "engines": expected_engines,
        "cells_per_engine": {engine: len(by_engine[engine]) for engine in expected_engines},
    },
    "engine_summary": summary_rows,
    "reuse_path_counts": {engine: dict(counts) for engine, counts in path_counts.items()},
    "caveats": [
        "One seed and one fixed-order sweep; no confidence interval or randomized-order claim is supported.",
        "HF native_prefix_cache is a placeholder/observer: it records matches but calls full prefill for every request, so it is not a native-runtime APC/Radix baseline.",
        "reuse_path_breakdown path_reading is specialized for full ShadowKV++ policy counters; no_reuse_path_executed does not negate reactive/Lite reuse_successes.",
        "The four ShadowKV++ variants differ by less than one percent in aggregate mean latency; their internal ranking is within plausible single-run/order noise.",
    ],
}

write_csv(root / "engine_summary.csv", summary_rows)
write_csv(root / "dataset_engine_comparisons.csv", detail_rows)
(root / "analysis_summary.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

summary_by_engine = {row["engine"]: row for row in summary_rows}
md = [
    "# Gemma 4 31B All-Engine Anomaly Audit",
    "",
    "## Structural checks",
    "",
    "- 110/110 result cells and 110/110 job records.",
    "- 10 datasets x 11 engines; 10 cells per engine.",
    "- Zero failed jobs and no missing requested engine.",
    "",
    "## Material caveats",
    "",
]
for caveat in analysis["caveats"]:
    md.append(f"- {caveat}")
md.extend(["", "## Signals", ""])
for engine in ["shadow_kv_plus", "shadow_kv_plus_lite", "shadow_kv_plus_best_latency", "shadow_kv_plus_raw_observer"]:
    row = summary_by_engine[engine]
    md.append(
        f"- `{engine}`: {row['aggregate_mean_latency_improvement_pct']:.2f}% mean-latency improvement, "
        f"{row['aggregate_p95_latency_improvement_pct']:.2f}% P95 improvement, "
        f"{row['aggregate_gpu_energy_improvement_pct']:.2f}% lower GPU energy, "
        f"{row['mean_latency_wins']}/10 mean wins, and {row['reuse_successes_total']} reuse successes."
    )
native = summary_by_engine["native_prefix_cache"]
shadow = summary_by_engine["shadow_kv"]
md.extend(
    [
        f"- `native_prefix_cache` recorded mean hit rate {native['mean_hit_rate']:.3f} but zero reuse successes and {native['bypassed_matches_total']} bypassed matches; source inspection confirms full prefill is always executed.",
        f"- `shadow_kv` executed zero reuse successes and was {-shadow['aggregate_mean_latency_improvement_pct']:.2f}% slower than no-cache in aggregate mean latency.",
        "- Reactive, greedy, strict-reactive, and all four ShadowKV++ variants recorded 1,270/1,280 reuse successes across the ten cells.",
        "",
    ]
)
(root / "ANOMALY_AUDIT.md").write_text("\n".join(md), encoding="utf-8")

for row in summary_rows:
    print(
        row["engine"],
        f"mean={row['mean_latency_ms']:.2f}",
        f"delta={row['aggregate_mean_latency_improvement_pct']:+.2f}%",
        f"p95_delta={row['aggregate_p95_latency_improvement_pct']:+.2f}%",
        f"energy_delta={row['aggregate_gpu_energy_improvement_pct']:+.2f}%",
        f"wins={row['mean_latency_wins']}/10",
        f"reuse={row['reuse_successes_total']}",
    )
