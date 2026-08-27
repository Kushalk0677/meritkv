#!/usr/bin/env python3
"""Learned admission baseline for the P100 package.

This driver trains a small logistic-regression admission policy from
ShadowKV++ policy traces, saves it as a portable JSON file, then runs a real
held-out engine named ``shadow_kv_plus_learned``. The learned engine uses the
same cache mechanics as ShadowKV++ but replaces the hand-written admit/bypass
decision with the learned classifier.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


REPO = Path(__file__).resolve().parent.parent

TRAIN_SEEDS = [42, 123]
TEST_SEEDS = [456]
TRAIN_DATASETS = ["ag_news", "banking77", "dolly", "samsum", "ultrachat"]
TEST_DATASETS = ["alpaca_eval", "cnn_dailymail", "daily_dialog", "oasst1", "xsum"]
MODES = ["templated", "semantic"]
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_N_REQUESTS = 128

BASE_ENGINES = ["no_cache", "shadow_kv", "shadow_kv_plus"]
PHASE3_ENGINES = ["no_cache", "shadow_kv", "shadow_kv_plus", "shadow_kv_plus_learned"]
DEFAULT_OUTCOME_EXPLORATION_RATE = 0.10

RAW_FEATURE_NAMES = [
    "matched_prefix_len",
    "exact_match_len",
    "semantic_prefix_len",
    "semantic_lcp_len",
    "semantic_similarity",
    "semantic_match_available",
    "token_count",
    "prefix_ratio",
    "shared_prefix_hint_tokens",
    "ewma_hit_rate",
    "ewma_waste_ratio",
    "is_templated",
    "is_semantic",
    "is_rag",
    "is_raw",
]

UTILITY_FEATURE_NAMES = [
    *RAW_FEATURE_NAMES,
    "policy_expected_benefit_ms",
    "policy_expected_cost_ms",
    "policy_expected_waste_ms",
    "policy_confidence",
    "policy_health",
]

POLICY_VARIANTS = {
    "raw": RAW_FEATURE_NAMES,
    "utility": UTILITY_FEATURE_NAMES,
}


def model_slug(model: str) -> str:
    return model.replace("/", "_")


def prompt_mode_flags(row: dict) -> dict[str, float]:
    mode = str(row.get("prompt_mode") or (row.get("metadata") or {}).get("prompt_mode") or "").lower()
    return {
        "is_templated": 1.0 if mode == "templated" else 0.0,
        "is_semantic": 1.0 if mode == "semantic" else 0.0,
        "is_rag": 1.0 if mode == "rag" else 0.0,
        "is_raw": 1.0 if mode == "raw" else 0.0,
    }


def _outcome_label(row: dict) -> int | None:
    if row.get("outcome_observed_admission") is True and "outcome_label_admit_positive" in row:
        return int(row.get("outcome_label_admit_positive") or 0)
    strategy = row.get("policy_strategy")
    if strategy in {"exact", "semantic_partial"}:
        if row.get("was_cache_hit"):
            full_mpt = float(row.get("ewma_full_ms_per_token_before") or 0.0)
            matched = float(row.get("matched_prefix_length") or 0.0)
            recomputed = float(row.get("tokens_recomputed") or 0.0)
            latency = float(row.get("latency_ms") or 0.0)
            if full_mpt <= 0.0 or matched <= 0.0:
                return 1
            suffix_estimate = full_mpt * recomputed
            observed_overhead = max(latency - suffix_estimate, 0.0)
            return 1 if (full_mpt * matched - observed_overhead) > 0.0 else 0
        return 0
    return None


def row_to_features(row: dict) -> dict[str, float] | None:
    if row.get("engine") != "shadow_kv_plus":
        return None
    strategy = row.get("policy_strategy")
    if strategy not in {"exact", "semantic_partial", "bypass"}:
        return None
    label = _outcome_label(row)
    if label is None:
        return None

    token_count = float(row.get("token_count") or 0.0)
    exact_len = float(row.get("matched_prefix_length") or 0.0)
    semantic_prefix_len = float(row.get("semantic_prefix_len") or 0.0)
    reusable = float(row.get("policy_reusable_prefix_tokens") or 0.0)
    matched = max(exact_len, semantic_prefix_len, reusable)
    token_count = max(token_count, matched, 1.0)
    flags = prompt_mode_flags(row)

    feat = {
        "matched_prefix_len": matched,
        "exact_match_len": exact_len,
        "semantic_prefix_len": semantic_prefix_len,
        "semantic_lcp_len": float(row.get("semantic_lcp_len") or 0.0),
        "semantic_similarity": float(row.get("semantic_similarity") or 0.0),
        "semantic_match_available": 1.0 if row.get("semantic_match_available") else 0.0,
        "token_count": token_count,
        "prefix_ratio": matched / max(token_count, 1.0),
        "shared_prefix_hint_tokens": float(row.get("shared_prefix_hint_tokens") or 0.0),
        "policy_expected_benefit_ms": float(row.get("policy_expected_benefit_ms") or 0.0),
        "policy_expected_cost_ms": float(row.get("policy_expected_cost_ms") or 0.0),
        "policy_expected_waste_ms": float(row.get("policy_expected_waste_ms") or 0.0),
        "policy_confidence": float(row.get("policy_confidence") or 0.0),
        "policy_health": float(row.get("policy_health") or 0.0),
        "ewma_hit_rate": float(row.get("ewma_hit_rate") or 0.0),
        "ewma_waste_ratio": float(row.get("ewma_waste_ratio") or 0.0),
        **flags,
        "label": float(label),
    }
    return feat


def load_trace_features(paths: Iterable[Path]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            count = 0
            for line in fh:
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                feat = row_to_features(parsed)
                if feat is not None:
                    rows.append(feat)
                    count += 1
        print(f"  {path}: {count} usable observed ShadowKV++ outcomes")
    return rows


def matrix(features: list[dict[str, float]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row.get(name, 0.0)) for name in feature_names] for row in features], dtype=np.float64)
    y = np.asarray([int(row["label"]) for row in features], dtype=np.int64)
    return x, y


def train_policy(features: list[dict[str, float]], output_path: Path, *, variant: str, feature_names: list[str]) -> dict:
    if len(features) < 10:
        raise RuntimeError(f"Need at least 10 observed outcome rows to train {variant}; found {len(features)}")
    x, y = matrix(features, feature_names)
    labels = sorted(set(int(v) for v in y))
    if len(labels) < 2:
        constant = float(labels[0])
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0.0] = 1.0
        payload = {
            "model_type": "constant_logistic",
            "label_mode": "realized_admission_outcome",
            "feature_variant": variant,
            "feature_names": feature_names,
            "coef": [0.0 for _ in feature_names],
            "intercept": 20.0 if constant >= 0.5 else -20.0,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "train_accuracy": 1.0,
            "train_samples": int(len(features)),
            "train_positive": int(y.sum()),
            "train_negative": int(len(y) - y.sum()),
            "train_admit": int(y.sum()),
            "train_bypass": int(len(y) - y.sum()),
            "note": "Only one realized outcome class appeared in the training traces.",
        }
    else:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required: pip install scikit-learn") from exc

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
        clf.fit(x_scaled, y)
        payload = {
            "model_type": "logistic_regression",
            "label_mode": "realized_admission_outcome",
            "feature_variant": variant,
            "feature_names": feature_names,
            "coef": clf.coef_[0].astype(float).tolist(),
            "intercept": float(clf.intercept_[0]),
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
            "train_accuracy": float(clf.score(x_scaled, y)),
            "train_samples": int(len(features)),
            "train_positive": int(y.sum()),
            "train_negative": int(len(y) - y.sum()),
            "train_admit": int(y.sum()),
            "train_bypass": int(len(y) - y.sum()),
            "note": (
                "Trained from realized admitted-reuse outcomes. Ordinary bypasses remain missing "
                "counterfactuals unless collected through outcome exploration."
            ),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Saved learned policy to {output_path}")
    print(f"  Train samples={payload['train_samples']} positive={payload['train_positive']} negative={payload['train_negative']} acc={payload['train_accuracy']:.3f}")
    return payload


def run_benchmark_cell(
    *,
    backend: str,
    model: str,
    device: str,
    dtype: str,
    dataset: str,
    mode: str,
    seed: int,
    n_requests: int,
    engines: list[str],
    output_dir: Path,
    learned_policy_path: Path | None = None,
    learned_policy_threshold: float = 0.50,
    outcome_exploration_rate: float = 0.0,
    outcome_exploration_seed: int = 0,
    share_backend: bool = True,
    dry_run: bool = False,
) -> bool:
    cmd = [
        sys.executable,
        str(REPO / "experiments" / "run_benchmark.py"),
        "--backend", backend,
        "--model", model,
        "--device", device,
        "--dtype", dtype,
        "--workload", "public_dataset",
        "--dataset", dataset,
        "--prompt_mode", mode,
        "--n_requests", str(n_requests),
        "--seed", str(seed),
        "--include_experimental",
        "--enable_policy_trace",
        "--disable_arrival_simulation",
        "--engines",
        *engines,
        "--output_dir", str(output_dir),
    ]
    if share_backend:
        cmd.append("--share_backend")
    if "shadow_kv_plus_learned" in engines:
        if learned_policy_path is None:
            raise ValueError("learned_policy_path is required for shadow_kv_plus_learned")
        cmd.extend(["--learned_policy_path", str(learned_policy_path), "--learned_policy_threshold", str(learned_policy_threshold)])
    if outcome_exploration_rate > 0.0:
        cmd.extend(["--outcome_exploration_rate", str(outcome_exploration_rate), "--outcome_exploration_seed", str(outcome_exploration_seed)])

    print("  " + " ".join(cmd))
    if dry_run:
        return True
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, text=True, timeout=7200)
    return result.returncode == 0


def benchmark_jsons(root: Path) -> list[Path]:
    return [Path(p) for p in glob.glob(str(root / "**" / "benchmark_*.json"), recursive=True)]


def summarize_results(test_dir: Path, summary_path: Path) -> dict:
    per_engine: dict[str, list[dict]] = {engine: [] for engine in PHASE3_ENGINES}
    for path in benchmark_jsons(test_dir):
        data = json.loads(path.read_text(encoding="utf-8"))
        no_cache = data.get("no_cache") or {}
        nc_lat = float(no_cache.get("mean_latency_ms") or 0.0)
        for engine in PHASE3_ENGINES:
            row = data.get(engine)
            if not isinstance(row, dict) or "mean_latency_ms" not in row:
                continue
            lat = float(row.get("mean_latency_ms") or 0.0)
            speedup = float(row.get("speedup_vs_no_cache_mean") or (nc_lat / lat if nc_lat > 0 and lat > 0 else 1.0))
            per_engine[engine].append({
                "mean_latency_ms": lat,
                "speedup": speedup,
                "waste_ratio": float(row.get("wasted_compute_ratio") or 0.0),
                "hit_rate": float(row.get("hit_rate") or 0.0),
                "requests_seen": int(row.get("requests_seen") or 0),
                "learned_policy_admit_total": int(row.get("learned_policy_admit_total") or 0),
                "learned_policy_bypass_total": int(row.get("learned_policy_bypass_total") or 0),
                "learned_policy_flip_to_admit_total": int(row.get("learned_policy_flip_to_admit_total") or 0),
                "learned_policy_flip_to_bypass_total": int(row.get("learned_policy_flip_to_bypass_total") or 0),
            })

    summary = {}
    print("\nHeld-out Phase 3 summary")
    print(f"{'engine':28s} {'lat(ms)':>9s} {'speedup':>9s} {'waste':>8s} {'hit':>8s} {'N':>7s}")
    print("-" * 76)
    for engine, rows in per_engine.items():
        if not rows:
            continue
        summary[engine] = {
            "cells": len(rows),
            "mean_latency_ms": float(np.mean([r["mean_latency_ms"] for r in rows])),
            "mean_speedup": float(np.mean([r["speedup"] for r in rows])),
            "mean_waste": float(np.mean([r["waste_ratio"] for r in rows])),
            "mean_hit_rate": float(np.mean([r["hit_rate"] for r in rows])),
            "requests_seen": int(sum(r["requests_seen"] for r in rows)),
            "learned_policy_admit_total": int(sum(r["learned_policy_admit_total"] for r in rows)),
            "learned_policy_bypass_total": int(sum(r["learned_policy_bypass_total"] for r in rows)),
            "learned_policy_flip_to_admit_total": int(sum(r["learned_policy_flip_to_admit_total"] for r in rows)),
            "learned_policy_flip_to_bypass_total": int(sum(r["learned_policy_flip_to_bypass_total"] for r in rows)),
        }
        s = summary[engine]
        print(f"{engine:28s} {s['mean_latency_ms']:9.2f} {s['mean_speedup']:9.3f} {s['mean_waste']:8.3f} {s['mean_hit_rate']:8.3f} {s['requests_seen']:7d}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run a learned ShadowKV++ admission baseline.")
    parser.add_argument("--phase", choices=["all", "train", "phase3", "evaluate"], default="all")
    parser.add_argument("--backend", default="hf", choices=["hf", "fake"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--n_requests", type=int, default=DEFAULT_N_REQUESTS)
    parser.add_argument("--out_root", default=str(REPO / "results" / "learned_baseline"))
    parser.add_argument("--policy_path", default=None)
    parser.add_argument("--learned_policy_threshold", type=float, default=0.50)
    parser.add_argument("--outcome_exploration_rate", type=float, default=DEFAULT_OUTCOME_EXPLORATION_RATE)
    parser.add_argument("--outcome_exploration_seed", type=int, default=20260708)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-train-benchmarks", action="store_true", help="Use existing train/policy_trace.jsonl files.")
    parser.add_argument("--skip-test-benchmarks", action="store_true", help="Use existing held-out test benchmark JSONs.")
    parser.add_argument(
        "--skip-policy-training",
        action="store_true",
        help="Use an existing learned_policy.json without retraining from traces. This is implicit for --phase phase3.",
    )
    parser.add_argument("--no-share-backend", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    train_dir = out_root / "train"
    test_dir = out_root / "test"
    policy_paths = {
        "raw": out_root / "learned_policy_raw.json",
        "utility": Path(args.policy_path) if args.policy_path else out_root / "learned_policy_utility.json",
    }
    legacy_policy_path = out_root / "learned_policy.json"

    print("=" * 72)
    print("Learned admission baseline")
    print("=" * 72)
    print(f"model={args.model} backend={args.backend} device={args.device} dtype={args.dtype}")
    print(f"train datasets={TRAIN_DATASETS} seeds={TRAIN_SEEDS}")
    print(f"test datasets={TEST_DATASETS} seeds={TEST_SEEDS}")
    print(f"out_root={out_root}")
    print(f"policy_raw={policy_paths['raw']}")
    print(f"policy_utility={policy_paths['utility']}")
    print(f"outcome_exploration_rate={args.outcome_exploration_rate}")

    if args.phase in {"all", "train"} and not args.skip_train_benchmarks:
        print("\n[Phase 1] Running train traces")
        for seed in TRAIN_SEEDS:
            for dataset in TRAIN_DATASETS:
                for mode in MODES:
                    cell = train_dir / model_slug(args.model) / mode / f"seed_{seed}" / dataset
                    ok = run_benchmark_cell(
                        backend=args.backend,
                        model=args.model,
                        device=args.device,
                        dtype=args.dtype,
                        dataset=dataset,
                        mode=mode,
                        seed=seed,
                        n_requests=args.n_requests,
                        engines=BASE_ENGINES,
                        output_dir=cell,
                        outcome_exploration_rate=args.outcome_exploration_rate,
                        outcome_exploration_seed=args.outcome_exploration_seed + seed,
                        share_backend=not args.no_share_backend,
                        dry_run=args.dry_run,
                    )
                    if not ok:
                        raise RuntimeError(f"Train cell failed: {dataset}/{mode}/seed_{seed}")

    if args.phase in {"all", "train"} and not args.skip_policy_training:
        print("\n[Phase 2] Training learned policy from traces")
        trace_paths = sorted(Path(p) for p in glob.glob(str(train_dir / "**" / "policy_trace.jsonl"), recursive=True))
        if not trace_paths:
            if args.dry_run:
                print(f"Dry run: no training traces found under {train_dir}; skipping training.")
                trace_paths = []
                features = []
            else:
                raise RuntimeError(f"No training policy_trace.jsonl files found under {train_dir}")
        else:
            features = load_trace_features(trace_paths)
        if args.dry_run:
            print("Dry run: not writing learned policies.")
        else:
            for variant, names in POLICY_VARIANTS.items():
                train_policy(features, policy_paths[variant], variant=variant, feature_names=names)
            if policy_paths["utility"] != legacy_policy_path:
                shutil.copyfile(policy_paths["utility"], legacy_policy_path)
    elif args.skip_policy_training or args.phase == "phase3":
        missing = [variant for variant, path in policy_paths.items() if not path.exists()]
        if not missing:
            print("\n[Phase 2] Using existing learned policies:")
            for variant, path in policy_paths.items():
                print(f"  {variant}: {path}")
        elif args.phase == "phase3" and not args.skip_policy_training:
            trace_paths = sorted(Path(p) for p in glob.glob(str(train_dir / "**" / "policy_trace.jsonl"), recursive=True))
            if trace_paths:
                print(f"\n[Phase 2] Rebuilding missing learned policies from {len(trace_paths)} train trace file(s)")
                features = load_trace_features(trace_paths)
                if args.dry_run:
                    print("Dry run: not writing learned policies.")
                else:
                    for variant in missing:
                        train_policy(features, policy_paths[variant], variant=variant, feature_names=POLICY_VARIANTS[variant])
                    if policy_paths["utility"].exists() and policy_paths["utility"] != legacy_policy_path:
                        shutil.copyfile(policy_paths["utility"], legacy_policy_path)
            elif args.dry_run:
                print(f"\n[Phase 2] Dry run: would use existing learned policies under {out_root}")
            else:
                raise RuntimeError(
                    f"Expected learned policies at {policy_paths}, but missing {missing}. "
                    f"No train policy_trace.jsonl files were found under {train_dir} either. "
                    "Run --phase train first, or pass --policy_path for the utility JSON and keep the raw JSON beside it."
                )
        elif args.dry_run:
            print(f"\n[Phase 2] Dry run: would use existing learned policies under {out_root}")
        else:
            raise RuntimeError(
                f"Expected learned policies at {policy_paths}, but missing {missing}. "
                "Run --phase train first, or pass --policy_path to the JSON produced by Phase 2."
            )

    if args.phase in {"all", "phase3"} and not args.skip_test_benchmarks:
        print("\n[Phase 3] Running held-out learned baseline")
        for variant, policy_path in policy_paths.items():
            print(f"\n[Phase 3] Variant={variant} policy={policy_path}")
            for seed in TEST_SEEDS:
                for dataset in TEST_DATASETS:
                    for mode in MODES:
                        cell = test_dir / variant / model_slug(args.model) / mode / f"seed_{seed}" / dataset
                        ok = run_benchmark_cell(
                            backend=args.backend,
                            model=args.model,
                            device=args.device,
                            dtype=args.dtype,
                            dataset=dataset,
                            mode=mode,
                            seed=seed,
                            n_requests=args.n_requests,
                            engines=PHASE3_ENGINES,
                            output_dir=cell,
                            learned_policy_path=policy_path,
                            learned_policy_threshold=args.learned_policy_threshold,
                            share_backend=not args.no_share_backend,
                            dry_run=args.dry_run,
                        )
                        if not ok:
                            raise RuntimeError(f"Test cell failed: {variant}/{dataset}/{mode}/seed_{seed}")

    if args.phase in {"all", "phase3", "evaluate"} and not args.dry_run:
        combined = {}
        for variant in POLICY_VARIANTS:
            combined[variant] = summarize_results(test_dir / variant, out_root / f"phase3_summary_{variant}.json")
        (out_root / "phase3_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"\nSaved combined summary to {out_root / 'phase3_summary.json'}")


if __name__ == "__main__":
    main()
