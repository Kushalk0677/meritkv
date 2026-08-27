"""
Paper-quality P100 sweep — ONE subprocess per (model, dataset, mode, seed).
Loads model ONCE, runs all 7 engines, then exits.
OOM-safe: only 1 load per job instead of 7.

Copy to: C:\shadowkv\p100_transfer\shadowkv_p100\run_paper_sweep_p100.py
Run: python run_paper_sweep_p100.py
"""

import subprocess, sys, os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "experiments" / "run_benchmark.py"
OUT = ROOT / "results_p100_n64_paper_5seed"

MODELS = [
    "gpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
]
DATASETS = [
    "ag_news","alpaca_eval","banking77","cnn_dailymail",
    "daily_dialog","dolly","oasst1","samsum","ultrachat","xsum",
]
MODES = ["raw","templated","semantic"]
SEEDS = [42,123,456,789,999]

# All 7 engines + semantic ablations in ONE call
# Model loads ONCE and is shared across all engines (requires patched run_benchmark.py)
ENGINES = [
    "no_cache","reactive_prefix_cache","greedy_prefix_cache",
    "strict_reactive_prefix_cache","frequency_speculative",
    "shadow_kv","shadow_kv_plus",
]

LOG = OUT / "_sweep.log"
OUT.mkdir(parents=True, exist_ok=True)

total = len(MODELS) * len(DATASETS) * len(MODES) * len(SEEDS)
done = 0

def log(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{t} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("="*80)
log("SWEEP: P100  n=64  seeds=5  OOM-safe (load once per mode)")
log(f"  models={len(MODELS)} datasets={len(DATASETS)} modes={len(MODES)} seeds={len(SEEDS)}")
log(f"  engines={','.join(ENGINES)}")
log(f"  total={total} jobs")
log(f"  IMPORTANT: run_benchmark.py must have share_backend patch applied")
log("="*80)

for model in MODELS:
    for ds in DATASETS:
        for mode in MODES:
            for seed in SEEDS:
                done += 1
                slug = model.split("/")[-1].replace(".","_")
                od = OUT / slug / mode / f"seed_{seed}" / ds
                log(f"[{done}/{total}] {slug[:20]:20s} {mode:10s} seed={seed} {ds:15s}")
                
                cmd = [
                    sys.executable, str(SCRIPT),
                    "--backend","hf","--model",model,
                    "--device","cuda","--dtype","float16",
                    "--include_experimental",
                    "--include_semantic_ablations",
                    "--allow_unsafe_semantic_kv_reuse",
                    "--share_backend",  # ← uses shared backend patch
                    "--engines"] + ENGINES + [
                    "--workload","public_dataset",
                    "--dataset",ds,"--prompt_mode",mode,
                    "--n_requests","64","--simulate_arrivals",
                    "--mean_inter_arrival_ms","50",
                    "--seed",str(seed),"--output_dir",str(od)]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    log(f"  {'OK' if r.returncode==0 else f'FAIL({r.returncode})'}")
                except Exception as e:
                    log(f"  ERROR: {e}")

log("="*80)
log(f"DONE {done}/{total}")
log("="*80)
