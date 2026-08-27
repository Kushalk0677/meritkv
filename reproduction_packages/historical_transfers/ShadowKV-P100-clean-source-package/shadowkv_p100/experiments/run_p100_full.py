#!/usr/bin/env python3
"""
run_p100_full.py -- full-fidelity P100 re-run (single start, ~full day).

Faithful reproduction of the committed final_p100 methodology, with the
corrected code:
  - memory-breakeven guard DISABLED (Path C; see engines.py)
  - semantic mode EXECUTES approximate reuse (--allow_unsafe_semantic_kv_reuse)

Config:
    n_requests = 256
    seeds      = 42, 123, 456
    engines    = ALL experimental (--include_experimental, 9 engines)
                 + semantic ablations in semantic mode
    energy     = measured (NVML) with a 10s idle baseline
    models     = 5   x   datasets = 10   x   modes = 3
    -> 5 x 10 x 3 x 3 = 450 isolated runs  (~20-24 h)

To close the paper's 5-seed claim, set SEEDS = [42, 123, 456, 789, 999]
(≈1.5 days). Isolated subprocess per cell, resumable, cooldown between
each full model. Writes an aggregate CSV and a zip at the end.

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export CUDA_VISIBLE_DEVICES=0
    python experiments/run_p100_full.py
    python experiments/run_p100_full.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------- config ----------------
N_REQUESTS = 256
SEEDS = [42, 123, 456]           # set [42,123,456,789,999] for the full 5-seed set
MODELS = [
    "gpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
]
DATASETS = [
    "ag_news", "alpaca_eval", "banking77", "cnn_dailymail", "daily_dialog",
    "dolly", "oasst1", "samsum", "ultrachat", "xsum",
]
MODES = ["raw", "templated", "semantic"]
DEVICE = "cuda"
DTYPE = "float16"
MEAN_INTER_ARRIVAL_MS = "50"
MAX_ARRIVAL_SLEEP_MS = "500"
IDLE_BASELINE_SECONDS = "10"
COOLDOWN_BETWEEN_MODELS_SEC = 120
RESULTS_DIRNAME = "results_p100_n256_full_3seed"
KEY_ENGINE = "shadow_kv_plus"
# ----------------------------------------


def model_tag(m: str) -> str:
    return m.replace("/", "_").replace(":", "_").replace(".", "_")


def cell_dir(root: Path, model: str, mode: str, seed: int, dataset: str) -> Path:
    return root / model_tag(model) / mode / f"seed_{seed}" / dataset


def latest_json(d: Path) -> Path | None:
    if not d.exists():
        return None
    fs = sorted(d.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return fs[0] if fs else None


def is_done(d: Path) -> bool:
    p = latest_json(d)
    if p is None:
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data.get(KEY_ENGINE), dict) and "mean_latency_ms" in data[KEY_ENGINE]


def fmt(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


def build_cmd(out: Path, model: str, mode: str, seed: int, dataset: str, args) -> list[str]:
    cmd = [
        sys.executable, "experiments/run_benchmark.py",
        "--backend", "hf",
        "--model", model,
        "--device", args.device,
        "--dtype", DTYPE,
        "--workload", "public_dataset",
        "--dataset", dataset,
        "--prompt_mode", mode,
        "--n_requests", str(args.n_requests),
        "--simulate_arrivals",
        "--mean_inter_arrival_ms", MEAN_INTER_ARRIVAL_MS,
        "--max_arrival_sleep_ms", MAX_ARRIVAL_SLEEP_MS,
        "--seed", str(seed),
        "--include_experimental",
        "--measure_energy",
        "--idle_baseline_seconds", IDLE_BASELINE_SECONDS,
        "--output_dir", str(out),
    ]
    if mode == "semantic":
        cmd.append("--include_semantic_ablations")
        cmd.append("--allow_unsafe_semantic_kv_reuse")
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    return cmd


def aggregate(root: Path, log: logging.Logger) -> None:
    rows = []
    fields = [
        "mean_latency_ms", "p95_latency_ms", "speedup_vs_no_cache_mean",
        "hit_rate", "wasted_compute_ratio", "reuse_successes",
        "semantic_partial_hits", "semantic_opportunity_plans_total",
        "semantic_blocked_by_backend_total", "semantic_quality_divergence_sum",
        "speculation_cooldown_events", "gpu_energy_j",
    ]
    for path in sorted(root.rglob("benchmark_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = data.get("config", {})
        for engine, m in data.items():
            if engine in ("config", "capabilities") or not isinstance(m, dict):
                continue
            row = {
                "model": cfg.get("model"), "dataset": cfg.get("dataset"),
                "prompt_mode": cfg.get("prompt_mode"), "seed": cfg.get("seed"),
                "engine": engine,
            }
            row.update({f: m.get(f) for f in fields})
            rows.append(row)
    if not rows:
        log.warning("aggregate: no rows found")
        return
    import csv
    cols = list(rows[0].keys())
    out_csv = root / "aggregate_all_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log.info("aggregate: %d rows -> %s", len(rows), out_csv)


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-fidelity P100 re-run (n=256, 9 engines, 3 seeds).")
    ap.add_argument("--output_dir", default=str(ROOT / RESULTS_DIRNAME))
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--n_requests", type=int, default=N_REQUESTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--cooldown", type=int, default=COOLDOWN_BETWEEN_MODELS_SEC)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(root / "_sweep.log", mode="a", encoding="utf-8")],
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    log = logging.getLogger("full")

    env = dict(os.environ)
    env.pop("SHADOWKV_BREAKEVEN_GUARD", None)  # Path C: guard stays disabled

    if not args.dry_run:
        try:
            from experiments.hw_detect import apply_detected_config
            apply_detected_config(log=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("hw_detect skipped: %s", exc)

    total = len(args.models) * len(args.datasets) * len(args.modes) * len(args.seeds)
    log.info("=" * 80)
    log.info("FULL RUN  n=%d  engines=ALL_EXPERIMENTAL  models=%d datasets=%d modes=%d seeds=%d  total=%d",
             args.n_requests, len(args.models), len(args.datasets), len(args.modes), len(args.seeds), total)
    log.info("  guard=DISABLED  semantic_reuse=EXECUTED(unsafe)  energy=ON  out=%s", root)
    log.info("=" * 80)

    start = time.time()
    jid = done = failed = 0
    for mi, model in enumerate(args.models):
        log.info("#" * 80)
        log.info("MODEL %d/%d  %s", mi + 1, len(args.models), model)
        log.info("#" * 80)
        for seed in args.seeds:
            for mode in args.modes:
                for dataset in args.datasets:
                    jid += 1
                    out = cell_dir(root, model, mode, seed, dataset)
                    elapsed = time.time() - start
                    eta = (total - jid) / max(jid / max(elapsed, 1), 1e-9)
                    if is_done(out):
                        log.info("[%d/%d] SKIP %s %s seed=%d %s (done)  ETA~%s",
                                 jid, total, model_tag(model), mode, seed, dataset, fmt(eta))
                        done += 1
                        continue
                    out.mkdir(parents=True, exist_ok=True)
                    cmd = build_cmd(out, model, mode, seed, dataset, args)
                    log.info("[%d/%d] RUN %s %s seed=%d %s  ETA~%s",
                             jid, total, model_tag(model), mode, seed, dataset, fmt(eta))
                    log.info("  CMD %s", " ".join(cmd))
                    if args.dry_run:
                        continue
                    t0 = time.time()
                    with open(out / "run.log", "w", encoding="utf-8") as fh:
                        rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
                    if rc == 0 and is_done(out):
                        log.info("  OK (%s)", fmt(time.time() - t0))
                        done += 1
                    else:
                        log.error("  FAIL rc=%s (%s) -- see %s", rc, fmt(time.time() - t0), out / "run.log")
                        failed += 1
        if args.cooldown and mi < len(args.models) - 1 and not args.dry_run:
            log.info("-- cooldown %ds before next model --", args.cooldown)
            time.sleep(args.cooldown)

    log.info("=" * 80)
    log.info("DONE  done=%d failed=%d total=%d  wall=%s", done, failed, total, fmt(time.time() - start))
    log.info("=" * 80)

    if not args.dry_run:
        aggregate(root, log)
        zip_path = shutil.make_archive(str(root), "zip", str(root))
        log.info("ZIP  %s", zip_path)
        log.info("scp %s@HOST:%s C:\\Users\\kusha\\Downloads\\", os.environ.get("USER", "kushalkhemani"), zip_path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
