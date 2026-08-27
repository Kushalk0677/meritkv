#!/usr/bin/env python3
"""
run_p100_phi3.py -- Path C breakeven re-run for Phi-3 on the P100.

Reproduces the committed final_p100 Phi-3 sweep EXACTLY (same flags,
n_requests, arrival simulation, all experimental engines in one process),
with the only change being the memory-breakeven guard, which is now
DISABLED by default in engines.py (see SHADOWKV_BREAKEVEN_GUARD).

Purpose: test whether the waste-adaptive speculation machinery prevents
the 0.895x Phi-3 raw failure once the bogus k*~442 guard is gone.

Matrix (matches committed P100 Phi-3):
    1 model x 3 seeds x 3 prompt modes x 10 datasets = 90 benchmark files
    (each file contains all 9 experimental engines)

Usage (from the transfer root, after `pip install -e .`):
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export CUDA_VISIBLE_DEVICES=0            # pick a free P100
    python experiments/run_p100_phi3.py

    # Sanity A/B: reproduce the ORIGINAL guarded behaviour instead
    python experiments/run_p100_phi3.py --restore-guard --output_dir results_p100_phi3_guarded

    # Preview commands without running
    python experiments/run_p100_phi3.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "cuda"
DTYPE = "float16"
N_REQUESTS = 256
MEAN_INTER_ARRIVAL_MS = "50"
MAX_ARRIVAL_SLEEP_MS = "500"
SEEDS = [42, 123, 456]
MODES = ["raw", "templated", "semantic"]
DATASETS = [
    "ag_news", "alpaca_eval", "banking77", "cnn_dailymail", "daily_dialog",
    "dolly", "oasst1", "samsum", "ultrachat", "xsum",
]
# Comparison engine whose completion marks a cell done.
KEY_ENGINE = "shadow_kv_plus"


def model_tag(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


def cell_dir(root: Path, mode: str, seed: int, dataset: str) -> Path:
    return root / model_tag(MODEL) / mode / f"seed_{seed}" / dataset


def latest_json(d: Path) -> Path | None:
    if not d.exists():
        return None
    files = sorted(d.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def is_done(d: Path) -> bool:
    p = latest_json(d)
    if p is None:
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data.get(KEY_ENGINE), dict) and "mean_latency_ms" in data[KEY_ENGINE]


def build_cmd(out: Path, dataset: str, mode: str, seed: int, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "experiments/run_benchmark.py",
        "--backend", "hf",
        "--model", MODEL,
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
        "--output_dir", str(out),
    ]
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="Phi-3 P100 breakeven re-run (Path C).")
    ap.add_argument("--output_dir", default=str(ROOT / "results_p100_phi3_rerun"))
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--n_requests", type=int, default=N_REQUESTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--restore-guard", dest="restore_guard", action="store_true",
                    help="Set SHADOWKV_BREAKEVEN_GUARD=1 to reproduce the original guarded behaviour.")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(root / "_sweep.log", mode="a", encoding="utf-8")],
    )
    log = logging.getLogger("phi3_rerun")

    env = dict(os.environ)
    if args.restore_guard:
        env["SHADOWKV_BREAKEVEN_GUARD"] = "1"
        log.info("GUARD: restored (SHADOWKV_BREAKEVEN_GUARD=1) -- reproducing committed behaviour")
    else:
        env.pop("SHADOWKV_BREAKEVEN_GUARD", None)
        log.info("GUARD: DISABLED (Path C) -- testing waste-adaptive machinery")

    # Detect P100 hardware (memory/PCIe bandwidth) into config.yaml, like the paper runs.
    if not args.dry_run:
        try:
            from experiments.hw_detect import apply_detected_config
            apply_detected_config(log=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("hw_detect skipped: %s", exc)

    jobs = [(s, m, d) for s in args.seeds for m in args.modes for d in args.datasets]
    log.info("MODEL=%s jobs=%d out=%s", MODEL, len(jobs), root)

    done = failed = 0
    for i, (seed, mode, dataset) in enumerate(jobs, 1):
        out = cell_dir(root, mode, seed, dataset)
        if is_done(out):
            log.info("[%d/%d] SKIP seed=%d mode=%s ds=%s (done)", i, len(jobs), seed, mode, dataset)
            done += 1
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(out, dataset, mode, seed, args)
        log.info("[%d/%d] RUN seed=%d mode=%s ds=%s", i, len(jobs), seed, mode, dataset)
        log.info("  CMD %s", " ".join(cmd))
        if args.dry_run:
            continue
        t0 = time.time()
        logf = out / "run.log"
        with open(logf, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        if rc == 0 and is_done(out):
            log.info("  OK (%.1fs)", dt)
            done += 1
        else:
            log.error("  FAIL rc=%s (%.1fs) -- see %s", rc, dt, logf)
            failed += 1

    log.info("DONE done=%d failed=%d total=%d", done, failed, len(jobs))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
