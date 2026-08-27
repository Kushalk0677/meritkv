# Reproducing Results

Raw artifacts keep stable engine IDs: `shadow_kv_plus` displays as MeritKV, `shadow_kv` displays as MeritKV-Sem, and `shadow_kv_plus_lite` displays as MeritKV-Lite.


This document describes how to reproduce the repository's HuggingFace benchmark checks and how to read the runtime experiment tables.

## Canonical HF Results

The main controlled results use the HuggingFace backend on T4 and P100 GPUs. The public result bundle is stored under `results/controlled_results/`.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
pip install pytest
```

### Run the Test Suite

```bash
python -m pytest -q
```

The exact count can change as tests are added. A slow HF KV-correctness test may be skipped when the required model/GPU environment is not available.

### Run a Small Reproduction

```bash
python experiments/run_benchmark.py \
  --backend hf \
  --model distilgpt2 \
  --device cpu \
  --workload public_dataset \
  --dataset ag_news \
  --prompt_mode templated \
  --n_requests 32 \
  --include_experimental \
  --disable_arrival_simulation \
  --output_dir results/hf_cpu_agnews_templated
```

### Full Paper-Style Sweep

The public aggregate results summarize 5 models, 10 datasets, 3 prompt modes, and 3 seeds (`42`, `123`, `456`) for T4 and P100 controlled runs. See the main `README.md` for a smaller example loop. A full sweep requires CUDA GPU access and enough time to run all model/dataset/mode combinations.

## Result Layout

```text
results/
  paper_tables/           # Canonical machine-readable paper tables
  controlled_results/     # T4/P100 controlled JSONs and aggregate CSVs
  isolated_baseline_comparison/ # Four-model P100 baseline matrix
  realistic_results/      # Process-isolated no_cache and MeritKV (`shadow_kv_plus`) JSONs
  blackwell_longprefix_hf/ # Twelve-instance custom-splice scale study
  fidelity_examples/      # Per-sample fidelity examples
  mixed_traffic/          # Admission and mixed-workload summaries
  memory_bound_trace/     # Three-phase capacity-pressure traces
  memory_bound_trace_multiround/ # Four-arm enforced traces
```

Primary aggregate files:

```text
results/paper_tables/
results/controlled_results/summary_by_engine.csv
results/controlled_results/summary_by_mode_engine.csv
results/controlled_results/manifest.json
```

## Runtime Results

The runtime experiments in `runtime_experiments/` use SGLang, LMCache, and vLLM
on an RTX PRO 6000 Blackwell GPU with Qwen2.5 and Gemma-4 models. The broad
integrations are write-through and therefore measure compatibility and observed
overlay cost rather than enforced acceleration.

The top-level runtime CSVs are paper-facing measured aggregates. Their schemas
differ by campaign: Qwen includes absolute SGLang values, scale ratios, and the
32B five-replicate aggregate; Gemma records per-model overlay latency deltas.
Selected raw records are retained below `raw/`, but there is no single checked-in
script that regenerates every cross-runtime aggregate from a uniform raw tree.
Use each family README and preserve the paper's aggregation conventions,
especially the mean-of-paired-ratios convention for vLLM-32B.
