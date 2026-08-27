# Experiment Catalog

This catalog is the uniform entry point for the complete evidence tree. “Raw”
means all records retained from the campaign are included; it does not imply
that a campaign completed cells that failed before producing output.

| Family | Scope and status | Code / pipeline | Raw evidence and aggregates |
|---|---|---|---|
| Controlled T4/P100 | Five models, ten datasets, raw/semantic/templated modes, seeds 42/123/456; 898 measured cells. Two T4 Phi-3 templated cells were unavailable and are identified in the manifest. | `experiments/run_benchmark.py`, `experiments/analyze_shadowkv_results.py` | `results/controlled_results/` |
| Process-isolated baselines | Four-model P100 no-cache and cache-policy comparison. | `experiments/run_benchmark.py`; family README | `results/isolated_baseline_comparison/` |
| Process-isolated MeritKV | Five-model HF study with per-run manifests; Qwen custom-splice timing remains diagnostic. | `experiments/run_benchmark.py` | `results/realistic_results/` |
| Blackwell long prefix | Twelve model instances, ten datasets, five seeds, custom-HF long-prefix economic and agreement study; auxiliary Gemma all-engine extension is separately scoped. | expanded package and source snapshots in `results/blackwell_longprefix_hf/` and `reproduction_packages/blackwell_longprefix_semantic_n128/` | `results/blackwell_longprefix_hf/` |
| Crop-and-replay fidelity | Per-sample decoded-output agreement for the custom HF splice. | `experiments/run_fidelity_equiv.py`, `experiments/eval_comprehensive.py` | `results/fidelity_examples/` |
| Explicit-state continuation | Direct splice versus equivalent cached reference on five T4 models; notebooks, scripts, logs, and JSON reports retained. | `reproduction_packages/colab/exact_splice_validation/` | `results/exact_splice_validation/` |
| Mixed/admission policies | Controlled mixed-traffic and admission-policy comparisons. | maintained and archived runners under `experiments/` | `results/mixed_traffic/`, `results/paper_tables/` |
| Learned admission | Four completed models with disjoint learning/evaluation records; the Phi-3 attempt stopped at model load because of OOM and produced no fabricated cell. | package scripts and retained commands in the family tree | `results/learned_admission_baseline/` |
| Capacity pressure | Three-phase and four-arm multiround locality traces. The multiround family enforces skip-write/skip-lookup at two cache budgets. | `experiments/archive/v10_snapshot_202608/run_four_arm_trace.py` and analysis scripts | `results/memory_bound_trace/`, `results/memory_bound_trace_multiround/` |
| Qwen runtime | vLLM, SGLang, and LMCache compatibility matrices, balanced admission, Qwen-14B/32B historical campaigns, energy, policy overlay, and diagnostics. Broad overlays are write-through unless a family README says enforced. | `literature_accurate_baselines/`, `runtime_experiments/qwen2.5/**/code`, archived session scripts | `runtime_experiments/qwen2.5/` |
| Gemma runtime | Five Gemma-4 Blackwell models across vLLM, SGLang, and LMCache with complete retained raw trees. | family scripts, commands, and raw manifests | `runtime_experiments/gemma4/` |
| Native enforcement | Frozen Qwen2.5-32B/Gemma-4-31B action-proof and full matrix, plus model-specific storage-admission extensions, verifiers, logs, counters, and request records. | code snapshots inside each package | `runtime_experiments/native_enforcement_blackwell/` |
| Native vLLM auxiliary | Measured execute-or-bypass development experiment retained for provenance; not a paper-facing production claim. | `experiments/archive/native_vllm_execute_bypass/` | same directory, including downloaded result packages |
| Energy estimates | Modelled/scaled estimates only; excluded from measured evidence. | derivation scripts and notes in family tree | `results/energy_estimates/` |

## Reading Order

1. Start with `results/paper_tables/` and `CLAIMS_TO_ARTIFACTS.md` for the
   manuscript-facing boundary.
2. Open the relevant family README for its matrix, exclusions, and aggregation.
3. Inspect raw manifests, commands, environment captures, and request records.
4. Use `reproduction_packages/README.md` when a frozen transfer or Colab bundle
   is required.

Historical and auxiliary measurements are retained at full depth but never
silently promoted into the current paper claim set.
