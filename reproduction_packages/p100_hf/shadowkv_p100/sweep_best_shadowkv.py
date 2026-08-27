"""
Three-phase sweep with SEPARATE output directories for each engine.
Phase 1: shadow_kv_plus -> results_p100_phase1_skp
Phase 2: no_cache       -> results_p100_phase2_nc
Phase 3: remaining 5 engines with --share_backend -> results_p100_phase3_baselines
"""

import subprocess, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "experiments" / "run_benchmark.py"

MODELS = [
    "gpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]
DATASETS = [
    "ag_news","alpaca_eval","banking77","cnn_dailymail",
    "daily_dialog","dolly","oasst1","samsum","ultrachat","xsum",
]
MODES = ["raw","templated","semantic"]
SEEDS = [42,123,456,789,999]

def run_phase(engine, extra_flags, out_dir):
    """Run one engine across all models/datasets/modes/seeds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG = out_dir / "_sweep.log"
    done = 0
    total = len(MODELS) * len(DATASETS) * len(MODES) * len(SEEDS)
    
    for model in MODELS:
        for ds in DATASETS:
            for mode in MODES:
                for seed in SEEDS:
                    done += 1
                    slug = model.split("/")[-1].replace(".","_")
                    od = out_dir / slug / mode / f"seed_{seed}" / ds
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    line = f"{timestamp} [{done}/{total}] {slug[:18]:18s} {mode:10s} seed={seed} {ds:12s} {engine}"
                    print(line, flush=True)
                    with open(LOG, "a") as f:
                        f.write(line + "\n")
                    
                    cmd = [
                        sys.executable, str(SCRIPT),
                        "--backend","hf","--model",model,
                        "--device","cuda","--dtype","float16",
                        "--engines", engine,
                    ] + extra_flags + [
                        "--workload","public_dataset",
                        "--dataset",ds,"--prompt_mode",mode,
                        "--n_requests","64","--simulate_arrivals",
                        "--mean_inter_arrival_ms","50",
                        "--seed",str(seed),"--output_dir",str(od)]
                    try:
                        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                        status = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
                        print(f"  {status}", flush=True)
                    except Exception as e:
                        print(f"  ERROR: {e}", flush=True)

# Phase 1: shadow_kv_plus alone
run_phase("shadow_kv_plus",
          ["--include_experimental", "--include_semantic_ablations", "--allow_unsafe_semantic_kv_reuse"],
          ROOT / "results_p100_phase1_skp")

print("PHASE 1 COMPLETE", flush=True)

# Phase 2: no_cache alone (DIFFERENT DIRECTORY — no overwrites)
run_phase("no_cache", [],
          ROOT / "results_p100_phase2_nc")

print("PHASE 2 COMPLETE", flush=True)
print("Speedup = phase2_nc / phase1_skp (matched by model/dataset/mode/seed)", flush=True)
