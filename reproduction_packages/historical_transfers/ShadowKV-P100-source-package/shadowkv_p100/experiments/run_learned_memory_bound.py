#!/usr/bin/env python3
"""Run a memory-bound learned-baseline recovery trace.

This is the missing mechanism check for the learned admission baseline.  The
standard public-dataset matrix tests mean speedup, but it does not ask whether
a learned per-request policy preserves useful prefixes when cache capacity is
tight.  This script creates a small, deterministic pressure trace:

  1. warm a set of reusable "victim" prefixes,
  2. insert many one-off distractor prefixes under a clamped KV budget,
  3. probe the victims again and measure survival / recovery.

The learned raw and learned utility-component policies are evaluated beside
NoCache, naive ShadowKV, and MeritKV on the same trace.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proactive_kv_cache.config_loader import CONFIG
from proactive_kv_cache.engines import maybe_shutdown, summarize_engine
from proactive_kv_cache.utils import set_seed
from proactive_kv_cache.workload import Request

from run_benchmark import (  # noqa: E402
    build_engine,
    load_backend_from_args,
    prepare_request_metadata,
    resolve_model,
    _profile_shadowkv_costs,
    _warmup_backend,
)


DEFAULT_SEEDS = [42, 123, 456, 789, 999]
DEFAULT_ENGINES = [
    "no_cache",
    "shadow_kv",
    "shadow_kv_plus",
    "learned_raw",
    "learned_utility",
]


def _prefix_words(label: str, count: int) -> str:
    words = [
        f"policy-{label}",
        "regulated",
        "assistant",
        "cache",
        "reuse",
        "memory",
        "tenant",
        "workflow",
        "risk",
        "audit",
        "latency",
        "prefix",
        "decision",
        "utility",
        "evidence",
        "guard",
    ]
    return " ".join(words[i % len(words)] for i in range(count))


def make_memory_bound_workload(
    *,
    seed: int,
    n_victims: int,
    n_distractors: int,
    victim_warmup_repeats: int,
    pressure_repeats: int,
    recovery_repeats: int,
    prefix_words: int,
    mean_inter_arrival_ms: float,
) -> list[Request]:
    del seed  # deterministic structure; seed is kept in metadata/config.
    requests: list[Request] = []
    arrival = 0.0
    request_id = 0

    def add_request(phase: str, family: str, shared_prefix: str, suffix: str, is_victim: bool, repeat: int = 0) -> None:
        nonlocal arrival, request_id
        prompt = f"{shared_prefix}{suffix}"
        requests.append(
            Request(
                request_id=request_id,
                prompt=prompt,
                arrival_time=arrival,
                metadata={
                    "source_workload": "memory_bound_learned_trace",
                    "variant": "victim_distractor_recovery",
                    "prompt_mode": "templated",
                    "shared_prefix_text": shared_prefix,
                    "memory_phase": phase,
                    "memory_family": family,
                    "memory_is_victim": is_victim,
                    "memory_recovery_repeat": repeat,
                },
            )
        )
        request_id += 1
        arrival += mean_inter_arrival_ms / 1000.0

    victim_prefixes: list[tuple[str, str]] = []
    for idx in range(n_victims):
        family = f"victim_{idx:02d}"
        prefix = (
            f"System: Memory-bound KV reuse trace for {family}. "
            f"{_prefix_words(f'victim{idx}', prefix_words)} "
            "Instruction: preserve this operational context and answer the case note. Case: "
        )
        victim_prefixes.append((family, prefix))
        for repeat in range(1, victim_warmup_repeats + 1):
            add_request(
                "victim_warmup",
                family,
                prefix,
                f"Warmup request {repeat}-{idx}. Summarize account-review requirements.",
                True,
                repeat,
            )

    for idx in range(n_distractors):
        family = f"distractor_{idx:03d}"
        prefix = (
            f"System: One-off pressure prefix for {family}. "
            f"{_prefix_words(f'distractor{idx}', prefix_words)} "
            "Instruction: answer once, then this prefix should not matter again. Case: "
        )
        for repeat in range(1, pressure_repeats + 1):
            add_request(
                "distractor_pressure",
                family,
                prefix,
                f"Pressure request {repeat}-{idx}. Produce a short compliance triage note.",
                False,
                repeat,
            )

    for repeat in range(1, recovery_repeats + 1):
        phase = "victim_recovery_first" if repeat == 1 else "victim_recovery_later"
        for idx, (family, prefix) in enumerate(victim_prefixes):
            add_request(
                phase,
                family,
                prefix,
                f"Recovery request {repeat}-{idx}. Reuse the earlier context if it survived.",
                True,
                repeat,
            )

    return requests


def make_benchmark_args(args: argparse.Namespace, *, policy_path: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        backend=args.backend,
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        tensor_parallel_size=1,
        trust_remote_code=False,
        disable_native_prefix_caching=True,
        include_experimental=True,
        include_runtime_baselines=False,
        include_semantic_ablations=False,
        engines=None,
        share_backend=True,
        policy_preset=args.policy_preset,
        enable_policy_tuning=False,
        policy_tuning_requests=0,
        policy_tuning_presets=[],
        policy_tuning_metric="utility_adjusted_latency",
        learned_policy_path=str(policy_path) if policy_path is not None else None,
        learned_policy_threshold=args.learned_policy_threshold,
        outcome_exploration_rate=0.0,
        outcome_exploration_seed=0,
        enable_policy_trace=args.enable_policy_trace,
        enable_decision_log=False,
        semantic_index_diagnostics=False,
        allow_unsafe_semantic_kv_reuse=False,
        early_layer_reuse_ratio=0.35,
        logit_guard_threshold=0.08,
        min_reuse_prefix_tokens=None,
        disable_utility_admission=False,
        utility_min_net_saved_ms=None,
        measure_energy=False,
        gpu_index=0,
        idle_baseline_seconds=0.0,
        energy_output_csv=None,
        workload="memory_bound",
        variant="victim_distractor_recovery",
        dataset=None,
        prompt_mode="templated",
        dataset_split=None,
        n_requests=args.n_requests,
        simulate_arrivals=False,
        mean_inter_arrival_ms=args.mean_inter_arrival_ms,
        max_arrival_sleep_ms=0.0,
        max_memory_mb=args.memory_budget_mb,
        speculative_k=args.speculative_k,
        idle_threshold_ms=args.idle_threshold_ms,
        seed=0,
        output_dir=str(args.out_root),
        resolved_prompt_mode="templated",
        shadowkv_policy_calibration=None,
        shadowkv_policy_kwargs=None,
        policy_tuning_report=None,
        _shadowkv_policy_calibration_base=None,
    )


def clamp_runtime_memory(memory_budget_mb: int) -> None:
    CONFIG.update(
        {
            "hardware.max_gpu_memory_mb": int(memory_budget_mb),
            "hardware.max_cpu_memory_mb": int(memory_budget_mb),
        }
    )


def clamp_engine_bank(engine, memory_budget_mb: int) -> None:
    budget_bytes = max(int(memory_budget_mb), 1) * 1024 ** 2
    engine.bank.max_gpu_memory_bytes = budget_bytes
    engine.bank.max_cpu_memory_bytes = 0
    engine.bank.max_disk_memory_bytes = 0


def cleanup_after_backend(device: str) -> None:
    gc.collect()
    if device.startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def auto_memory_budget_mb(args: argparse.Namespace, backend, requests: list[Request]) -> int:
    if args.memory_budget_mb > 0:
        return int(args.memory_budget_mb)
    first = next((req for req in requests if (req.metadata or {}).get("memory_is_victim")), requests[0])
    prefix_text = str((first.metadata or {}).get("shared_prefix_text") or first.prompt)
    prefix_tokens = backend.tokenize(prefix_text)
    entry_bytes = max(int(backend.estimate_kv_cache_bytes(len(prefix_tokens))), 1)
    budget_bytes = entry_bytes * max(int(args.capacity_entries), 1)
    return max(1, int(math.ceil(budget_bytes / (1024 * 1024))))


def phase_stats(rows: Iterable[dict]) -> dict[str, object]:
    rows = list(rows)
    latencies = [float(row["latency_ms"]) for row in rows]
    hits = [bool(row["was_cache_hit"]) for row in rows]
    families = {str(row.get("memory_family")) for row in rows}
    hit_families = {str(row.get("memory_family")) for row in rows if row.get("was_cache_hit")}
    return {
        "requests": len(rows),
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": float(sorted(latencies)[int(0.95 * (len(latencies) - 1))]) if latencies else 0.0,
        "hit_rate": sum(1 for hit in hits if hit) / max(len(hits), 1),
        "families": len(families),
        "families_hit": len(hit_families),
        "family_hit_rate": len(hit_families) / max(len(families), 1),
    }


def summarize_trace(engine_summary: dict, rows: list[dict]) -> dict:
    phases = sorted({str(row["memory_phase"]) for row in rows})
    phase_summary = {phase: phase_stats(row for row in rows if row["memory_phase"] == phase) for phase in phases}
    recovery_rows = [row for row in rows if str(row["memory_phase"]).startswith("victim_recovery")]
    first_rows = [row for row in rows if row["memory_phase"] == "victim_recovery_first"]
    later_rows = [row for row in rows if row["memory_phase"] == "victim_recovery_later"]
    merged = dict(engine_summary)
    merged.update(
        {
            "memory_trace": {
                "phase_summary": phase_summary,
                "victim_recovery": phase_stats(recovery_rows),
                "victim_recovery_first": phase_stats(first_rows),
                "victim_recovery_later": phase_stats(later_rows),
                "peak_memory_used_mb": max((float(row["memory_used_mb"]) for row in rows), default=0.0),
                "peak_entries_stored": max((int(row["entries_stored"]) for row in rows), default=0),
                "evictions": max((int(row.get("evictions_cumulative", 0)) for row in rows), default=0),
            }
        }
    )
    return merged


def add_baseline_comparisons(summary: dict[str, dict]) -> None:
    baseline = summary.get("no_cache")
    if not baseline:
        return
    baseline_overall = float(baseline.get("mean_latency_ms") or 0.0)
    baseline_recovery = float(
        baseline.get("memory_trace", {})
        .get("victim_recovery", {})
        .get("mean_latency_ms")
        or 0.0
    )
    baseline_first = float(
        baseline.get("memory_trace", {})
        .get("victim_recovery_first", {})
        .get("mean_latency_ms")
        or 0.0
    )
    for metrics in summary.values():
        mean_latency = float(metrics.get("mean_latency_ms") or 0.0)
        if baseline_overall and mean_latency:
            metrics["speedup_vs_no_cache_mean"] = baseline_overall / mean_latency
        trace = metrics.get("memory_trace", {})
        recovery = float(trace.get("victim_recovery", {}).get("mean_latency_ms") or 0.0)
        first = float(trace.get("victim_recovery_first", {}).get("mean_latency_ms") or 0.0)
        if baseline_recovery and recovery:
            trace["victim_recovery_speedup_vs_no_cache"] = baseline_recovery / recovery
            trace["victim_recovery_latency_delta_vs_no_cache_ms"] = recovery - baseline_recovery
        if baseline_first and first:
            trace["first_recovery_speedup_vs_no_cache"] = baseline_first / first
            trace["first_recovery_latency_delta_vs_no_cache_ms"] = first - baseline_first


def run_engine_trace(
    args: argparse.Namespace,
    bench_args: SimpleNamespace,
    backend,
    requests: list[Request],
    *,
    engine_label: str,
    engine_name: str,
    policy_path: Path | None,
    calibration: dict,
) -> dict:
    bench_args.learned_policy_path = str(policy_path) if policy_path is not None else None
    bench_args._shadowkv_policy_calibration_base = calibration
    engine = build_engine(bench_args, backend, engine_name)
    engine.name = engine_label
    engine.trace_enabled = bool(args.enable_policy_trace)
    clamp_engine_bank(engine, bench_args.max_memory_mb)
    finalized = False
    rows: list[dict] = []
    shared_prefix_token_cache: dict[str, int] = {}
    try:
        _warmup_backend(backend, requests[:1])
        for req in requests:
            tokens = backend.tokenize(req.prompt)
            metadata = prepare_request_metadata(backend, req, tokens, shared_prefix_token_cache)
            result = engine.serve_tokens(req.request_id, tokens, metadata=metadata)
            bank = engine.bank.snapshot_metrics()
            rows.append(
                {
                    "request_id": req.request_id,
                    "memory_phase": metadata.get("memory_phase"),
                    "memory_family": metadata.get("memory_family"),
                    "memory_is_victim": metadata.get("memory_is_victim"),
                    "memory_recovery_repeat": metadata.get("memory_recovery_repeat"),
                    "latency_ms": float(result.latency_ms),
                    "was_cache_hit": bool(result.was_cache_hit),
                    "matched_prefix_length": int(result.matched_prefix_length),
                    "tokens_recomputed": int(result.tokens_recomputed),
                    "cache_tier": result.cache_tier,
                    "memory_used_mb": float(bank.get("memory_used_mb", 0.0)),
                    "entries_stored": int(bank.get("entries_stored", 0)),
                    "evictions_cumulative": int(bank.get("evictions", 0)),
                }
            )
        maybe_shutdown(engine)
        finalized = True
        metrics = summarize_trace(summarize_engine(engine), rows)
    finally:
        if not finalized and hasattr(engine, "finalize"):
            try:
                maybe_shutdown(engine)
            except Exception:
                pass
    return {"summary": metrics, "rows": rows}


def aggregate_seed_summaries(seed_summaries: dict[str, dict]) -> dict:
    engines = sorted({engine for seed in seed_summaries.values() for engine in seed.get("engines", {})})
    aggregate: dict[str, dict] = {}
    for engine in engines:
        engine_rows = [seed["engines"][engine] for seed in seed_summaries.values() if engine in seed.get("engines", {})]
        aggregate[engine] = {
            "seeds": len(engine_rows),
            "speedup_vs_no_cache_mean": statistics.mean(
                float(row.get("speedup_vs_no_cache_mean", 0.0)) for row in engine_rows
            ),
            "hit_rate": statistics.mean(float(row.get("hit_rate", 0.0)) for row in engine_rows),
            "wasted_compute_ratio": statistics.mean(float(row.get("wasted_compute_ratio", 0.0)) for row in engine_rows),
            "evictions": statistics.mean(float(row.get("memory_trace", {}).get("evictions", 0.0)) for row in engine_rows),
            "victim_recovery_hit_rate": statistics.mean(
                float(row.get("memory_trace", {}).get("victim_recovery", {}).get("hit_rate", 0.0)) for row in engine_rows
            ),
            "first_recovery_hit_rate": statistics.mean(
                float(row.get("memory_trace", {}).get("victim_recovery_first", {}).get("hit_rate", 0.0)) for row in engine_rows
            ),
            "victim_recovery_speedup_vs_no_cache": statistics.mean(
                float(row.get("memory_trace", {}).get("victim_recovery_speedup_vs_no_cache", 0.0)) for row in engine_rows
            ),
            "peak_memory_used_mb": statistics.mean(
                float(row.get("memory_trace", {}).get("peak_memory_used_mb", 0.0)) for row in engine_rows
            ),
            "learned_policy_flip_to_bypass_total": sum(
                int(row.get("learned_policy_flip_to_bypass_total", 0)) for row in engine_rows
            ),
            "learned_policy_flip_to_admit_total": sum(
                int(row.get("learned_policy_flip_to_admit_total", 0)) for row in engine_rows
            ),
        }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learned baselines on a memory-bound victim/recovery trace.")
    parser.add_argument("--backend", default="hf", choices=["hf", "fake"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out_root", default=str(ROOT / "results" / "learned_baseline_memory_bound"))
    parser.add_argument("--policy_raw", default=None)
    parser.add_argument("--policy_utility", default=None)
    parser.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES, choices=DEFAULT_ENGINES)
    parser.add_argument("--n_victims", type=int, default=6)
    parser.add_argument("--n_distractors", type=int, default=32)
    parser.add_argument("--victim_warmup_repeats", type=int, default=2)
    parser.add_argument("--pressure_repeats", type=int, default=2)
    parser.add_argument("--recovery_repeats", type=int, default=2)
    parser.add_argument("--prefix_words", type=int, default=56)
    parser.add_argument("--capacity_entries", type=int, default=3)
    parser.add_argument("--memory_budget_mb", type=int, default=0, help="0 means auto-size to capacity_entries * victim-prefix KV bytes.")
    parser.add_argument("--mean_inter_arrival_ms", type=float, default=50.0)
    parser.add_argument("--policy_preset", default="balanced")
    parser.add_argument("--learned_policy_threshold", type=float, default=None)
    parser.add_argument("--speculative_k", type=int, default=2)
    parser.add_argument("--idle_threshold_ms", type=float, default=30.0)
    parser.add_argument("--enable_policy_trace", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_policy = Path(args.policy_raw) if args.policy_raw else out_root.parent / "learned_policy_raw.json"
    utility_policy = Path(args.policy_utility) if args.policy_utility else out_root.parent / "learned_policy_utility.json"

    for label, path in (("learned_raw", raw_policy), ("learned_utility", utility_policy)):
        if label in args.engines and not path.exists():
            raise RuntimeError(f"{label} requires policy JSON at {path}")

    print("=" * 72)
    print("Memory-bound learned-baseline trace")
    print("=" * 72)
    print(f"model={args.model} backend={args.backend} device={args.device} dtype={args.dtype}")
    print(f"seeds={args.seeds}")
    print(f"out_root={out_root}")
    print(f"engines={args.engines}")

    seed_summaries: dict[str, dict] = {}
    for seed in args.seeds:
        seed_dir = out_root / f"seed_{seed}"
        summary_path = seed_dir / "memory_bound_summary.json"
        if args.resume and summary_path.exists():
            print(f"[resume] {summary_path}", flush=True)
            seed_summaries[str(seed)] = json.loads(summary_path.read_text(encoding="utf-8"))
            continue
        if args.dry_run:
            print(f"[dry-run] would run seed={seed} -> {seed_dir}", flush=True)
            continue

        seed_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        requests = make_memory_bound_workload(
            seed=seed,
            n_victims=args.n_victims,
            n_distractors=args.n_distractors,
            victim_warmup_repeats=args.victim_warmup_repeats,
            pressure_repeats=args.pressure_repeats,
            recovery_repeats=args.recovery_repeats,
            prefix_words=args.prefix_words,
            mean_inter_arrival_ms=args.mean_inter_arrival_ms,
        )
        args.n_requests = len(requests)
        bench_args = make_benchmark_args(args)
        backend = load_backend_from_args(bench_args)
        try:
            auto_budget = auto_memory_budget_mb(args, backend, requests)
            args.memory_budget_mb = auto_budget
            bench_args.max_memory_mb = auto_budget
            clamp_runtime_memory(auto_budget)
            calibration = _profile_shadowkv_costs(backend)
            engines: dict[str, dict] = {}
            traces: dict[str, list[dict]] = {}
            engine_specs = {
                "no_cache": ("no_cache", None),
                "shadow_kv": ("shadow_kv", None),
                "shadow_kv_plus": ("shadow_kv_plus", None),
                "learned_raw": ("shadow_kv_plus_learned", raw_policy),
                "learned_utility": ("shadow_kv_plus_learned", utility_policy),
            }
            for label in args.engines:
                engine_name, policy_path = engine_specs[label]
                print(f"[seed {seed}] running {label} (budget={auto_budget} MB)", flush=True)
                result = run_engine_trace(
                    args,
                    bench_args,
                    backend,
                    requests,
                    engine_label=label,
                    engine_name=engine_name,
                    policy_path=policy_path,
                    calibration=calibration,
                )
                engines[label] = result["summary"]
                traces[label] = result["rows"]
            add_baseline_comparisons(engines)
            seed_summary = {
                "config": {
                    "model": resolve_model(args.model),
                    "backend": args.backend,
                    "device": args.device,
                    "dtype": args.dtype,
                    "seed": seed,
                    "memory_budget_mb": auto_budget,
                    "capacity_entries": args.capacity_entries,
                    "n_victims": args.n_victims,
                    "n_distractors": args.n_distractors,
                    "victim_warmup_repeats": args.victim_warmup_repeats,
                    "pressure_repeats": args.pressure_repeats,
                    "recovery_repeats": args.recovery_repeats,
                    "prefix_words": args.prefix_words,
                    "policy_raw": str(raw_policy),
                    "policy_utility": str(utility_policy),
                },
                "engines": engines,
            }
            summary_path.write_text(json.dumps(seed_summary, indent=2), encoding="utf-8")
            (seed_dir / "memory_bound_trace.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
            seed_summaries[str(seed)] = seed_summary
            print(f"Saved {summary_path}", flush=True)
        finally:
            backend_device = str(getattr(backend, "device", ""))
            del backend
            cleanup_after_backend(backend_device)

    if not args.dry_run:
        aggregate = {
            "config": {
                "model": resolve_model(args.model),
                "backend": args.backend,
                "device": args.device,
                "dtype": args.dtype,
                "seeds": args.seeds,
                "engines": args.engines,
            },
            "seeds": seed_summaries,
            "aggregate": aggregate_seed_summaries(seed_summaries),
        }
        aggregate_path = out_root / "memory_bound_aggregate.json"
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"Saved aggregate to {aggregate_path}", flush=True)


if __name__ == "__main__":
    main()
