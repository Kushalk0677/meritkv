#!/usr/bin/env python3
"""Generate deterministic phased traces and paired arm plans.

This generator contains no result-dependent inputs. Running it twice must
produce byte-identical traces and plans.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRACE_DIR = HERE / "traces"
HOT_FAMILIES = 4
EVALUATION_SEEDS = [32001, 32002, 32003, 32004, 32005]
CALIBRATION_SEED = 32000
SMOKE_SEED = 32010
ARMS = [
    "native_admit_all_lru",
    "meritkv_write_through_lru",
    "meritkv_skip_write_only_lru",
    "meritkv_joint_skip_write_skip_lookup_lru",
    "native_admit_all_lfu",
]
ARM_ORDERS = [
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 0],
    [2, 3, 4, 0, 1],
    [3, 4, 0, 1, 2],
    [4, 0, 1, 2, 3],
]
EVALUATION_PLAN = "evaluation_plan_25.tsv"
SMOKE_PLAN = "smoke_plan_5.tsv"
LFU_FEASIBILITY_GATE = "feasibility/lfu_feasibility_gate.json"


def stable_nonce(seed: int, ordinal: int) -> str:
    return hashlib.sha256(f"{seed}:{ordinal}:meritkv-capacity".encode()).hexdigest()[:32]


def hot_prefix(family: int) -> str:
    anchor = (
        f"Recurring hot family {family}. Preserve this stable operating context, "
        "apply the same bounded policy, and use the shared evidence record. "
    )
    return anchor * 40


def hot_row(seed: int, request_class: str, phase: str, family: int, ordinal: int) -> dict:
    prefix = hot_prefix(family)
    return {
        "prompt": (
            f"{prefix}\nSeed {seed}; phase {phase}; family {family}; ordinal {ordinal}. "
            f"Return the integer {family}."
        ),
        "metadata": {
            "phase": phase,
            "shadowkv_trace_class": request_class,
            "reuse_family": f"hot_{family}",
            "hot_family": family,
            "hot_probe": request_class != "hot_bootstrap",
            "shared_prefix_text": prefix,
            "expected_behavior": "resident_exact_or_shared_prefix_hit_after_admission",
        },
    }


def one_off_row(seed: int, ordinal: int, phase: str) -> dict:
    nonce = stable_nonce(seed, ordinal)
    sentence = (
        f"Unique low-frequency prefix {ordinal} nonce {nonce}; this context is never reused. "
    )
    prefix = sentence * 48
    return {
        "prompt": f"{prefix}\nReturn the integer {ordinal % 10}.",
        "metadata": {
            "phase": phase,
            "shadowkv_trace_class": "long_one_off_prefix",
            "reuse_family": f"one_off_{ordinal}",
            "hot_family": None,
            "hot_probe": False,
            "expected_behavior": "meritkv_bypass_candidate",
        },
    }


def build_trace(seed: int, pressure_windows: int, recovery_cycles: int) -> list[dict]:
    rows: list[dict] = []
    hot_ordinals = [0] * HOT_FAMILIES

    for family in range(HOT_FAMILIES):
        rows.append(hot_row(seed, "hot_bootstrap", "bootstrap_hot_set", family, 0))
        hot_ordinals[family] += 1

    establish_cycles = 2 if pressure_windows >= 4 else 1
    for _ in range(establish_cycles):
        for family in range(HOT_FAMILIES):
            rows.append(
                hot_row(
                    seed,
                    "hot_established_probe",
                    "establish_hot_hits",
                    family,
                    hot_ordinals[family],
                )
            )
            hot_ordinals[family] += 1

    one_off_ordinal = 0
    for window in range(pressure_windows):
        for group in range(4):
            for _ in range(4):
                rows.append(
                    one_off_row(
                        seed,
                        one_off_ordinal,
                        "capacity_pressure_with_hot_probes",
                    )
                )
                one_off_ordinal += 1
            first_family = (window + group) % HOT_FAMILIES
            second_family = (first_family + 2) % HOT_FAMILIES
            for family in (first_family, second_family):
                rows.append(
                    hot_row(
                        seed,
                        "hot_pressure_probe",
                        "capacity_pressure_with_hot_probes",
                        family,
                        hot_ordinals[family],
                    )
                )
                hot_ordinals[family] += 1

    for family in range(HOT_FAMILIES):
        rows.append(
            hot_row(
                seed,
                "hot_post_pressure_probe",
                "post_pressure_probe",
                family,
                hot_ordinals[family],
            )
        )
        hot_ordinals[family] += 1

    for _ in range(recovery_cycles):
        for family in range(HOT_FAMILIES):
            rows.append(
                hot_row(
                    seed,
                    "hot_recovery_probe",
                    "recovery",
                    family,
                    hot_ordinals[family],
                )
            )
            hot_ordinals[family] += 1

    for request_id, row in enumerate(rows):
        row["request_id"] = request_id
        row["arrival_time"] = 0.0
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    calibration = build_trace(CALIBRATION_SEED, pressure_windows=2, recovery_cycles=1)
    smoke = build_trace(SMOKE_SEED, pressure_windows=1, recovery_cycles=0)
    if len(calibration) != 64 or len(smoke) != 36:
        raise SystemExit(f"unexpected control trace sizes: {len(calibration)=}, {len(smoke)=}")
    write_jsonl(TRACE_DIR / f"calibration_seed_{CALIBRATION_SEED}.jsonl", calibration)
    write_jsonl(TRACE_DIR / f"smoke_seed_{SMOKE_SEED}.jsonl", smoke)

    full_rows: list[dict] = []
    smoke_rows: list[dict] = []
    for seed_index, seed in enumerate(EVALUATION_SEEDS):
        trace = build_trace(seed, pressure_windows=4, recovery_cycles=3)
        if len(trace) != 124:
            raise SystemExit(f"unexpected evaluation trace size for {seed}: {len(trace)}")
        trace_name = f"evaluation_seed_{seed}.jsonl"
        write_jsonl(TRACE_DIR / trace_name, trace)
        for order_index, arm_index in enumerate(ARM_ORDERS[seed_index], start=1):
            arm = ARMS[arm_index]
            full_rows.append(
                {
                    "seed": seed,
                    "order": order_index,
                    "arm": arm,
                    "retention": "lfu" if arm.endswith("_lfu") else "lru",
                    "trace": f"traces/{trace_name}",
                    "requests": len(trace),
                    "reset_before": "flush_cache+action_counters",
                }
            )

    for order, arm in enumerate(ARMS, start=1):
        smoke_rows.append(
            {
                "order": order,
                "arm": arm,
                "retention": "lfu" if arm.endswith("_lfu") else "lru",
                "seed": SMOKE_SEED,
                "trace": f"traces/smoke_seed_{SMOKE_SEED}.jsonl",
                "requests": len(smoke),
            }
        )

    evaluation_plan_path = HERE / EVALUATION_PLAN
    smoke_plan_path = HERE / SMOKE_PLAN
    write_tsv(evaluation_plan_path, full_rows)
    write_tsv(smoke_plan_path, smoke_rows)
    if len(full_rows) != 25 or len(smoke_rows) != 5:
        raise SystemExit(
            f"unexpected frozen plan sizes: evaluation={len(full_rows)}, smoke={len(smoke_rows)}"
        )
    for seed in EVALUATION_SEEDS:
        seed_arms = [row["arm"] for row in full_rows if row["seed"] == seed]
        if len(seed_arms) != 5 or set(seed_arms) != set(ARMS):
            raise SystemExit(f"seed {seed} does not contain each required arm exactly once")

    position_counts = {
        arm: {
            str(position): sum(
                1
                for row in full_rows
                if row["arm"] == arm and row["order"] == position
            )
            for position in range(1, 6)
        }
        for arm in ARMS
    }
    if any(
        set(counts.values()) != {1}
        for counts in position_counts.values()
    ):
        raise SystemExit(f"evaluation is not a position-balanced 5x5 Latin square: {position_counts}")

    feasibility_gate_path = HERE / LFU_FEASIBILITY_GATE
    if not feasibility_gate_path.is_file():
        raise SystemExit(f"missing frozen fail-closed LFU gate: {feasibility_gate_path}")
    manifest = {
        "schema_version": 1,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calibration_seed": CALIBRATION_SEED,
        "smoke_seed": SMOKE_SEED,
        "evaluation_seeds": EVALUATION_SEEDS,
        "arms": ARMS,
        "cache_topology": "Qwen RadixCache for every arm; hybrid_swa=False",
        "evaluation_design": {
            "cells": len(full_rows),
            "plan": EVALUATION_PLAN,
            "plan_sha256": hashlib.sha256(evaluation_plan_path.read_bytes()).hexdigest(),
            "position_counts_by_arm": position_counts,
            "chronological_execution_required": True,
            "balance_rule": (
                "five cyclic Latin rows; over five seeds every arm appears once per seed and "
                "exactly once in each chronological position"
            ),
        },
        "smoke_design": {
            "cells": len(smoke_rows),
            "plan": SMOKE_PLAN,
            "plan_sha256": hashlib.sha256(smoke_plan_path.read_bytes()).hexdigest(),
        },
        "lfu_frequency_aware_baseline": {
            "status": "included_fail_closed_pending_gpu_preflight",
            "scientific_disposition": "abort_extension_if_gate_fails",
            "gate": LFU_FEASIBILITY_GATE,
            "gate_sha256": hashlib.sha256(feasibility_gate_path.read_bytes()).hexdigest(),
        },
        "traces": {},
    }
    for path in sorted(TRACE_DIR.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        manifest["traces"][str(path.relative_to(HERE))] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "requests": len(rows),
            "phase_counts": {
                phase: sum(1 for row in rows if row["metadata"]["phase"] == phase)
                for phase in sorted({row["metadata"]["phase"] for row in rows})
            },
            "class_counts": {
                request_class: sum(
                    1
                    for row in rows
                    if row["metadata"]["shadowkv_trace_class"] == request_class
                )
                for request_class in sorted(
                    {row["metadata"]["shadowkv_trace_class"] for row in rows}
                )
            },
        }
    (HERE / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
