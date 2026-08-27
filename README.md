# MeritKV: Utility-Gated Admission Control for KV-Cache Reuse in LLM Serving

![Manuscript under review](https://img.shields.io/badge/manuscript-under_review-lightgrey.svg)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MeritKV is a per-request policy controller for KV cache reuse in LLM serving. Instead of treating reuse as a yes/no prefix-cache mechanism, it asks whether reuse helps this particular request. Each candidate is scored with a net-utility objective, `U = benefit - cost - waste`, and the controller can admit reuse, bypass it, or record a semantic opportunity without blindly executing approximate KV substitution.

> **Authors:** Kushal Khemani, Evan Leri, Dr. Sparsh Mittal

---

## Engine Name Aliases

Raw artifacts keep the stable engine IDs used during execution. Public-facing text maps them as follows:

| Engine ID | Display name |
|---|---|
| `shadow_kv_plus` | MeritKV |
| `shadow_kv` | MeritKV-Sem |
| `shadow_kv_plus_lite` | MeritKV-Lite |

---

## What Is Novel

The central novelty is the per-request utility decision. A conventional prefix cache reuses whenever a matching prefix exists. MeritKV adds a request-level admission step that estimates whether the matched reuse is actually worth executing under the current workload, model, and hardware conditions.

This gives the system three useful properties:

- **Explicit bypass:** low-value reuse can be skipped instead of adding overhead.
- **Waste accounting:** speculative work is measured as waste when it is not reused.
- **Semantic opportunity tracking:** semantic matches can be detected and scored without assuming approximate KV reuse is always correctness-preserving.

---

## How It Works

MeritKV combines three cooperating components:

- **TieredStateBank** - stores KV entries keyed by prefix token sequences with longest-prefix lookup via a radix trie. Per-prefix statistics track frequency, observation count, branching factor, and memory footprint.
- **AdaptiveReuseController** - evaluates each request and returns a reuse plan: bypass, exact reuse, semantic opportunity, or guarded semantic reuse when the backend and safety checks allow it.
- **SemanticKVIndex** - detects paraphrase-equivalent request families with a lightweight token sketch, so semantic opportunities can be recorded without adding a sentence encoder to the hot path.

---

## Key Results

### Controlled and Process-Isolated HF Evaluation

The controlled Hugging Face study covers five models, ten datasets, three
prompt modes, and three seeds (`42`, `123`, `456`) on T4 and P100 GPUs. The
aggregate CSVs live under `results/controlled_results/`.

Baselines include no-cache, reactive prefix caching, greedy prefix caching, strict reactive prefix caching, frequency speculation, MeritKV-Sem (`shadow_kv`), and MeritKV (`shadow_kv_plus`).

| Engine | Mean Speedup | Waste | Hit Rate |
|--------|:-----------:|:-----:|:--------:|
| No cache | 1.000x | 0.000 | 0.000 |
| Reactive | 1.214x | 0.000 | 0.317 |
| Greedy | 1.221x | 0.000 | 0.320 |
| Strict reactive | 1.254x | 0.000 | 0.310 |
| Frequency spec. | 1.208x | 0.284 | 0.617 |
| MeritKV-Sem | 1.287x | 0.264 | 0.606 |
| **MeritKV** | **1.365x** | **0.156** | **0.402** |

The result is a latency-waste tradeoff rather than a hit-rate claim: MeritKV
improves controlled latency while reducing speculative waste relative to
MeritKV-Sem. In the separate four-model process-isolated baseline matrix,
MeritKV is `1.089x` faster than strict reactive overall, with per-dataset ratios
from `1.028x` to `1.113x`.

### Production Runtime Compatibility

Runtime-system experiments under `runtime_experiments/` cover SGLang, LMCache,
and vLLM on Blackwell with Qwen2.5 from 1.5B to 32B and five Gemma-4 variants.
These integrations are write-through: they measure compatibility and observed
decision-layer overhead, not production acceleration caused by MeritKV.

| Metric | Value |
|--------|-------|
| Qwen2.5 SGLang overlay/native latency ratio | 1.013 to 0.953 |
| Qwen2.5 vLLM overlay/native latency ratio | 0.975 to 0.993 |
| vLLM-32B APC+MeritKV vs APC mean latency | +0.44% |
| vLLM-32B APC+MeritKV vs APC P95 latency | +0.59% |
| vLLM-32B APC+MeritKV vs APC idle-adjusted energy/request | -0.22% |
| Mean Gemma-4 overlay overhead | +1.3% SGLang, +0.5% vLLM, +0.4% LMCache |

The separate Qwen2.5-1.5B balanced-admission experiment uses a native SGLang
per-request admission hook rather than the broad write-through integration. On
the pathological short-prefix semantic AG News workload, MeritKV skipped 254 of
256 cache lookups and reduced attempted cache-query traffic from 957.8 MB to
6.8 MB (99.3%). Mean latency increased from 10.9 ms to 20.2 ms, so this result
demonstrates enforced waste avoidance, not a latency improvement. The complete
bundle is under `runtime_experiments/qwen2.5/sglang/balanced_admission/`.

The broad runtime study must not be read as an acceleration result. Enforced
execute-or-bypass evidence appears separately in the capacity-pressure traces:
at 56 GB and 28 GB, enforcement lowers mean latency by `11.6%` and `16.7%`
versus the write-through overlay and improves final recovery by 41 and 61
percentage points.

A separate frozen native-enforcement matrix on Qwen2.5-32B and Gemma-4-31B
verifies that requested bypasses reached the SGLang cache: all 42 proof-cell
bypasses per model matched executed skip-lookup and skip-write counters. The
full workload bypassed only 50 of 12,800 requests per model and therefore
showed near-parity, not acceleration, versus write-through (+0.197% and
+0.144% mean latency). The complete 414-cell evidence package is under
`runtime_experiments/native_enforcement_blackwell/`.

### KV Cache Reuse Fidelity

Fidelity examples live in `results/fidelity_examples/`. They evaluate a custom
Hugging Face `DynamicCache` splice, not native vLLM/SGLang/LMCache cache
correctness. These files are diagnostic examples, not a claim that approximate
semantic KV substitution is universally safe.

| Model | ROUGE-L | Interpretation |
|-------|:-------:|----------------|
| TinyLlama 1.1B | 0.966 | high observed agreement |
| Gemma 2B | 0.974 | high observed agreement |
| Phi-3 Mini | 0.931 | intermediate observed agreement |
| GPT-2 | 0.876 | marginally above the intermediate-band boundary |
| Qwen2.5 1.5B | 0.200 | lower observed agreement; timing excluded |

The descriptive agreement bands are not universal correctness thresholds. All
Qwen float16 custom-splice speedups are excluded from validated performance,
and the observed pattern is not attributed to architecture.

The separate explicit-state continuation check under float16 SDPA compares the
splice directly with equivalent native cached decoding. It agrees on 956/960
decoded tokens across five T4 models, including 192/192 for Qwen2.5-1.5B and
Gemma-2B. This direct-equivalence diagnostic answers a different question from
the crop-and-replay ROUGE-L results above; its raw JSONs and summary are under
`results/exact_splice_validation/`.

---

## Repository Layout

```text
src/proactive_kv_cache/        Core engines, cache bank, controller, models
  engines.py                   BaseEngine, NoCacheEngine, ShadowKVPlusEngine (internal MeritKV class)
  cache.py                     TieredStateBank with radix trie
  controller.py                AdaptiveReuseController, utility scoring
  policy.py                    CostAwareSlackPolicy
  semantic.py                  SemanticKVIndex, token sketching
  models.py                    Backend abstraction (FakeBackend, HFBackend)
  datasets.py                  Dataset loading and prompt templates
  metrics.py                   Engine metrics and aggregation
  policy_learning.py           Offline grid-search learner
  backend_adapters.py          Experimental runtime adapter layer
  telemetry.py                 JSON decision logger
  energy.py                    GPU energy metering

experiments/
  run_benchmark.py             Main benchmark entry point
  run_fidelity_equiv.py        KV cache reuse fidelity pipeline
  eval_comprehensive.py        ROUGE-L and exact-match evaluator
  profile_plan.py              Controller Plan() latency profiler
  analyze_shadowkv_results.py  Result parser and policy-summary generator
  archive/                     Superseded experiment scripts and notebooks

reproduction_packages/        Expanded frozen packages, Colab notebooks, archives

results/
  controlled_results/          T4/P100 controlled benchmark JSONs and CSV summaries
  paper_tables/                Canonical machine-readable paper tables
  isolated_baseline_comparison/ Four-model process-isolated baseline comparison
  realistic_results/           Process-isolated no-cache and MeritKV JSON outputs
  blackwell_longprefix_hf/      Twelve-instance long-prefix aggregates and provenance
  fidelity_examples/           Per-sample KV reuse fidelity examples
  exact_splice_validation/     Explicit-state continuation raw JSONs and summary
  mixed_traffic/                Admission and mixed-workload summaries
  memory_bound_trace/           Three-phase capacity-pressure traces
  memory_bound_trace_multiround/ Four-arm enforced multiround traces
  sweep_timing/                Small timing/smoke outputs
  RESULTS.md                   Public result-bundle guide
  architectural_robustness.md  Controlled versus realistic validation notes

runtime_experiments/           SGLang, LMCache, and vLLM result tables
literature_accurate_baselines/ Runtime-baseline adapters and source notes
docs/                          Design, methodology, hardware, and experiment catalog
tests/                         Unit and regression tests
tools/                         Release inventory and integrity checks
```

For a uniform map from every experiment family to its code, raw records,
aggregates, hardware captures, and reproduction package, see
[`docs/EXPERIMENT_CATALOG.md`](docs/EXPERIMENT_CATALOG.md). The repository
contains all project-retained raw evidence; model weights, downloaded datasets,
framework caches, and virtual environments are intentionally not redistributed.
Method and dependency entry points are
[`docs/METHODOLOGY_INDEX.md`](docs/METHODOLOGY_INDEX.md) and
[`docs/DEPENDENCIES_AND_PACKAGES.md`](docs/DEPENDENCIES_AND_PACKAGES.md).

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .\.venv\Scripts\Activate.ps1       # Windows PowerShell
```

### 2. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
pip install -e .
pip install pytest
```

### 3. Verify with a smoke test

```bash
python experiments/run_benchmark.py \
  --backend fake \
  --workload synthetic \
  --variant high_skew \
  --n_requests 40 \
  --include_experimental \
  --disable_arrival_simulation \
  --output_dir results/smoke_fake
```

Expected output: a benchmark JSON and summary under `results/smoke_fake/`.

### 4. Run the test suite

```bash
python -m pytest -q
```

The exact count can change as tests are added. The optional slow HF KV-correctness test may be skipped when the required environment is not available.

---

## Using the Controller

```python
from proactive_kv_cache.engines import ShadowKVPlusEngine
from proactive_kv_cache.models import FakeBackend

backend = FakeBackend()
engine = ShadowKVPlusEngine(backend=backend, max_memory_mb=256)

tokens = tuple(range(50, 100))
metadata = {'prompt_mode': 'raw', 'arrival_time': 0.0}

result = engine.serve_tokens(request_id=0, tokens=tokens, metadata=metadata)
print(f"Reused {result.matched_prefix_length} tokens, "
      f"recomputed {result.tokens_recomputed}, "
      f"latency {result.latency_ms:.2f} ms")
```

---

## Experiments

| Experiment | Description |
|---|---|
| `run_benchmark.py` | HF/fake benchmark runner across models, datasets, modes, seeds, and engines |
| `run_fidelity_equiv.py` | KV reuse fidelity checks via `DynamicCache.crop()` |
| `profile_plan.py` | Controller `Plan()` latency profiling |
| `eval_comprehensive.py` | ROUGE-L and exact-match evaluation |
| `analyze_shadowkv_results.py` | Result parser and policy-summary generator |
| `run_blackwell_semantic_n128.py` | Auxiliary RTX PRO 6000 Blackwell semantic-execution sweep; not a distinct paper result |
| `run_p100_isolated_sweep.py` | Conservative isolated P100 rerun driver |

### Reproduce a Small Matrix

```bash
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="ag_news"
N=64
OUT="results/reproduction_qwen_agnews"

for SEED in 42 123 456; do
  for MODE in raw templated semantic; do
    python experiments/run_benchmark.py \
      --backend hf --model "$MODEL" --device cuda:0 --dtype float16 \
      --workload public_dataset --dataset "$DATASET" --prompt_mode "$MODE" \
      --n_requests "$N" --seed "$SEED" --include_experimental \
      --disable_arrival_simulation --output_dir "$OUT/$MODE/seed_$SEED"
  done
done
```

### Reproduce Fidelity Examples

```bash
python experiments/run_fidelity_equiv.py --device cpu \
  --models tinyllama \
  --datasets samsum alpaca_eval banking77 \
  --n_samples 8 --max_gen_tokens 64
```

For GPU checks, use `experiments/fidelity_equiv_colab.ipynb` from the archive/experiment folder if present in your working copy.

---

## Included Result Artifacts

| Artifact | Path |
|---|---|
| Controlled T4/P100 benchmark JSONs and summaries | `results/controlled_results/` |
| Aggregate controlled CSVs | `results/controlled_results/summary_by_engine.csv`, `results/controlled_results/summary_by_mode_engine.csv` |
| Canonical machine-readable paper tables | `results/paper_tables/` |
| Process-isolated baseline matrix | `results/isolated_baseline_comparison/` |
| Process-isolated no-cache and MeritKV outputs | `results/realistic_results/` |
| Blackwell long-prefix aggregates and provenance | `results/blackwell_longprefix_hf/` |
| Fidelity example JSONs | `results/fidelity_examples/` |
| Explicit-state continuation validation | `results/exact_splice_validation/` |
| Mixed/admission comparisons | `results/mixed_traffic/` |
| Capacity-pressure traces | `results/memory_bound_trace/`, `results/memory_bound_trace_multiround/` |
| Runtime-system experiments | `runtime_experiments/` |
| Native SGLang balanced admission | `runtime_experiments/qwen2.5/sglang/balanced_admission/` |
| Native SGLang enforcement matrix | `runtime_experiments/native_enforcement_blackwell/` |
| Native Gemma-4-31B storage admission | `runtime_experiments/native_enforcement_blackwell/storage_admission_extension/` |

---

## What Is Not Included

To keep the repository lightweight:

- **Paper source / LaTeX** - the manuscript is maintained separately.
- **Model weights** - all models are downloaded from Hugging Face at runtime.
- **Datasets** - all datasets are downloaded from Hugging Face at runtime.
- **Transient local outputs** outside the curated `results/` and `runtime_experiments/` snapshots.
- **Complete raw runtime archives** for every aggregate row. Selected raw
  campaigns are retained; the complete release is separate.

---

## Full Documentation

| Topic | Document |
|-------|----------|
| System design and rationale | [docs/design_overview.md](docs/design_overview.md) |
| Engine failure analysis | [docs/engine_failures.md](docs/engine_failures.md) |
| Architectural robustness | [results/architectural_robustness.md](results/architectural_robustness.md) |
| Fidelity experiment methodology | [docs/experimental_setup.md](docs/experimental_setup.md) |
| Fidelity results and coupled utility | [docs/results_table.md](docs/results_table.md) |
| Qwen float16 diagnostic analysis | [docs/fidelity_deep_analysis.md](docs/fidelity_deep_analysis.md) |
| Semantic correctness boundary | [docs/semantic_fidelity.md](docs/semantic_fidelity.md) |
| Runtime experiments | [runtime_experiments/README.md](runtime_experiments/README.md) |
| Blackwell reproduction | [docs/reproducing_blackwell.md](docs/reproducing_blackwell.md) |
| P100 reproduction | [docs/reproducing_p100.md](docs/reproducing_p100.md) |
| External artifacts | [docs/artifact_manifest.md](docs/artifact_manifest.md) |

---

## Citation

Citation metadata will be added when a public paper identifier is available.

---

## License

MIT License.
