# Reproducibility Diagnosis

## Criterion 1: Paper-Level — Seeds, Hardware, Versions, Settings, Statistics

| Item | Status | Detail |
|------|--------|--------|
| Seeds | ⚠️ Needs fix | README says 42/123/456 (3 seeds). Paper says 750 runs / 5 seeds. Need to align. |
| Hardware | ✅ Specified | T4 (16 GB, 320 GB/s), P100 (12 GB, 549 GB/s), Blackwell RTX PRO 6000 (96 GB) |
| Model versions | ⚠️ Partial | HuggingFace model IDs listed but no pinned commit hashes (models may update) |
| Dataset versions | ⚠️ Partial | Dataset IDs listed. cnn_dailymail pinned to 3.0.0. Others use default split. |
| Prompt modes | ✅ Clear | raw, templated, semantic documented in paper |
| Cache settings | ⚠️ Partial | config/config.yaml has defaults. Not all ~40 constants explained in paper. |
| Statistical procedures | ✅ Good | 95% CI, p-values reported for per-dataset comparisons |

## Criterion 2: Artifact-Level — Scripts Mapping to Paper Tables

| Paper Table | Corresponding Script/File | Status |
|-------------|--------------------------|--------|
| Table 2 (headline) | `results/controlled_results/summary_by_engine.csv` | ✅ Direct |
| Per-mode breakdown | `results/controlled_results/summary_by_mode_engine.csv` | ✅ Direct |
| Runtime tables | `runtime_experiments/{sglang,vllm,lmcache}/results.csv` | ✅ Direct |
| Fidelity tables | `results/fidelity_examples/f16/all_results.json` | ✅ Direct |
| Ablation tables | `experiments/run_admission_baselines.py` | ✅ Runnable |
| Mixed-traffic | `experiments/run_mixed_traffic.py` | ✅ Runnable |
| Memory-bound | `experiments/run_memory_bound_trace.py` | ✅ Runnable |
| Sensitivity | `experiments/run_sensitivity_sweep.py` (in v10/) | ⚠️ Needs copy to repo |

## Criterion 3: Log-Level Verification — Raw CSV/JSON Backing Tables

| Metric | Backing File | Status |
|--------|-------------|--------|
| Mean speedup 1.365× | `summary_by_engine.csv` → `shadow_kv_plus` row | ✅ |
| Waste 0.156 | Same CSV | ✅ |
| CI [1.342, 1.388] | Same CSV (`speedup_ci95_low`, `speedup_ci95_high`) | ✅ |
| Per-model speedup | `summary_by_mode_engine.csv` | ✅ |
| Runtime SGLang | `sglang/results.csv` | ✅ |
| Runtime vLLM | `vllm/results.csv` | ✅ |
| Runtime LMCache | `lmcache/results.csv` | ✅ |
| Fidelity per-sample | `fidelity_examples/f16/*.json` | ✅ |
| Memory-bound ledger | `results/mixed_traffic/mixed_results.json` | ✅ |
| Individual benchmark JSONs | `controlled_results/t4/`, `controlled_results/p100/` | ✅ |

## Criterion 4: Smoke-Test Execution

| Requirement | Status | Detail |
|-------------|--------|--------|
| Smoke test command | ✅ | `run_benchmark.py --backend fake` runs on CPU, no GPU needed |
| Test suite | ✅ | `pytest` → 88 passed, 1 skipped |
| Fidelity smoke | ✅ | `run_fidelity_equiv.py --device cpu --models gpt2 --n_samples 2` works on CPU |

The smoke test in the README works correctly on any machine with the repo
cloned and dependencies installed. No GPU required.

## Criterion 5: Selective Reruns — Auditability Without Blackwell

| Claim | Audit Path | Hardware Needed |
|-------|-----------|----------------|
| Headline HF speedup | `run_benchmark.py --backend hf` | Any GPU (T4+) |
| HF results reproducibility | Raw CSVs in `controlled_results/` | None (data provided) |
| Runtime numbers | Raw CSVs in `runtime_experiments/` | None (data provided) |
| SGLang/vLLM reproduction | needs Blackwell runtime | Blackwell RTX 6000 |
| Fidelity float16 | `run_fidelity_equiv.py --device cuda` | Any GPU (T4+) |
| Fidelity float32 | `run_fidelity_equiv.py --device cpu` | Any CPU |
| Admission baselines | `run_admission_baselines.py` | None (FakeBackend) |
| Mixed traffic | `run_mixed_traffic.py` | None (FakeBackend) |
| Memory-bound trace | `run_memory_bound_trace.py` | None (FakeBackend) |
| Blackwell runtime reproduction | needs Blackwell | Blackwell RTX 6000 |

**Verdict:** The controlled HF results are fully reproducible. The runtime
results (SGLang/vLLM) are auditable via provided CSVs but require a
Blackwell-class GPU to rerun — which is standard for systems papers with
expensive hardware requirements. The memory-bound, admission-baseline, and
mixed-traffic experiments all run on FakeBackend (CPU, no GPU).
