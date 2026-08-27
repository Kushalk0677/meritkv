#!/usr/bin/env python3
"""
run_p100_semantic.py -- Executed semantic-reuse re-run on the P100.

The committed runs detected 691 semantic opportunities but executed ZERO
approximate semantic substitutions (all blocked, because
allow_approximate_semantic_reuse defaults True only for the fake backend).
This script turns the flag ON with --allow_unsafe_semantic_kv_reuse so the
engine actually performs approximate semantic KV reuse on real HF backends,
producing real executed-reuse speedup AND quality-divergence numbers.

Only no_cache (baseline) and shadow_kv_plus (with unsafe reuse) are run --
the other experimental engines are irrelevant to the semantic claim and
only cost time. Policy trace + semantic index diagnostics are enabled so
you can audit which matches fired.

Matrix (default = minimum defensible):
    5 models x 10 datasets x 1 seed (42) x semantic mode = 50 benchmark files
Expand with --seeds 42 123 456 for CIs (-> 150 files).

Usage (from the transfer root, after `pip install -e .`):
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export CUDA_VISIBLE_DEVICES=0
    python experiments/run_p100_semantic.py

    python experiments/run_p100_semantic.py --dry-run
    python experiments/run_p100_semantic.py --models microsoft/Phi-3-mini-4k-instruct --seeds 42 123 456

IMPORTANT -- output quality: this run reports EXECUTED-reuse latency plus the
engine's divergence proxy (semantic_quality_divergence_*). Reviewers will
also want output fidelity (ROUGE ref-vs-reuse) of approximate reuse, which
the latency benchmark does not compute. After this completes, run a targeted
fidelity pass with experiments/run_fidelity_equiv.py on the same models.
Expect Qwen2.5 to degrade badly in float16 (cf. the Table VI 0.20 ROUGE).
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

DEVICE = "cuda"
DTYPE = "float16"
N_REQUESTS = 256
MEAN_INTER_ARRIVAL_MS = "50"
MAX_ARRIVAL_SLEEP_MS = "500"
SEEDS = [42]
MODE = "semantic"
ENGINES = ["no_cache", "shadow_kv_plus"]
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
KEY_ENGINE = "shadow_kv_plus"


def model_tag(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


def cell_dir(root: Path, model: str, seed: int, dataset: str) -> Path:
    return root / model_tag(model) / MODE / f"seed_{seed}" / dataset


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


def build_cmd(out: Path, model: str, dataset: str, seed: int, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "experiments/run_benchmark.py",
        "--backend", "hf",
        "--model", model,
        "--device", args.device,
        "--dtype", DTYPE,
        "--workload", "public_dataset",
        "--dataset", dataset,
        "--prompt_mode", MODE,
        "--n_requests", str(args.n_requests),
        "--simulate_arrivals",
        "--mean_inter_arrival_ms", MEAN_INTER_ARRIVAL_MS,
        "--max_arrival_sleep_ms", MAX_ARRIVAL_SLEEP_MS,
        "--seed", str(seed),
        "--engines", *ENGINES,
        "--allow_unsafe_semantic_kv_reuse",
        "--enable_policy_trace",
        "--semantic_index_diagnostics",
        "--output_dir", str(out),
    ]
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="Executed semantic-reuse P100 re-run.")
    ap.add_argument("--output_dir", default=str(ROOT / "results_p100_semantic_unsafe"))
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--n_requests", type=int, default=N_REQUESTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(root / "_sweep.log", mode="a", encoding="utf-8")],
    )
    log = logging.getLogger("semantic_rerun")

    if not args.dry_run:
        try:
            from experiments.hw_detect import apply_detected_config
            apply_detected_config(log=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("hw_detect skipped: %s", exc)

    jobs = [(m, s, d) for m in args.models for s in args.seeds for d in args.datasets]
    log.info("EXECUTED semantic reuse ON (--allow_unsafe_semantic_kv_reuse); jobs=%d out=%s", len(jobs), root)

    done = failed = 0
    for i, (model, seed, dataset) in enumerate(jobs, 1):
        out = cell_dir(root, model, seed, dataset)
        if is_done(out):
            log.info("[%d/%d] SKIP %s seed=%d ds=%s (done)", i, len(jobs), model, seed, dataset)
            done += 1
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(out, model, dataset, seed, args)
        log.info("[%d/%d] RUN %s seed=%d ds=%s", i, len(jobs), model, seed, dataset)
        log.info("  CMD %s", " ".join(cmd))
        if args.dry_run:
            continue
        t0 = time.time()
        logf = out / "run.log"
        with open(logf, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=str(ROOT), env=dict(os.environ), stdout=fh, stderr=subprocess.STDOUT).returncode
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
