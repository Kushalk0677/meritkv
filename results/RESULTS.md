# Result Bundle Guide

This directory contains measured result families, complete project-retained raw traces, compact
aggregates, and auxiliary diagnostics for the current MeritKV paper. Stable
engine IDs map as follows:

| Engine ID | Paper name |
|---|---|
| `shadow_kv_plus` | MeritKV |
| `shadow_kv` | MeritKV-Sem |
| `shadow_kv_plus_lite` | MeritKV-Lite |

## Evidence Families

| Folder | Evidence and scope |
|---|---|
| `paper_tables/` | Canonical machine-readable counterparts of the paper's quantitative tables. |
| `controlled_results/` | Five-model T4/P100 controlled study; 898 measured cells and aggregate policy tables. |
| `isolated_baseline_comparison/` | Four-model P100 process-isolated baseline comparison. |
| `realistic_results/` | Five-model process-isolated no-cache/MeritKV custom-splice results. Qwen float16 timing is diagnostic only. |
| `blackwell_longprefix_hf/` | Twelve-instance Blackwell long-prefix economic and output-agreement study using a custom HF splice. |
| `fidelity_examples/` | Retained per-sample custom-splice agreement records. |
| `exact_splice_validation/` | Direct cached-continuation JSONs and comparison summary. |
| `mixed_traffic/` | Admission-policy and mixed-workload comparisons. |
| `learned_admission_baseline/` | Four-model learned-policy comparison. |
| `memory_bound_trace/` | Three-phase capacity-pressure locality traces. |
| `memory_bound_trace_multiround/` | Four-arm enforced Gemma-4-31B multiround traces at two cache budgets. |
| `energy_estimates/` | Explicitly modelled estimates, excluded from measured paper evidence. |
| `archive/` | Complete retained V10 and root-workspace result snapshots; historical, not paper-facing. |

## Primary Controlled Aggregate

`controlled_results/summary_by_engine.csv` reports the main controlled
comparison. MeritKV reaches `1.365x` speedup and `0.156` speculative waste,
versus `1.287x/0.264` for MeritKV-Sem. These are controlled HF policy results,
not production-runtime acceleration.

The corresponding process-isolated four-model comparison is separate:
MeritKV/strict-reactive is `1.089x` overall, with per-dataset ratios from
`1.028x` to `1.113x`. Do not mix these two aggregations.

For direct table-by-table checking, use the compact CSVs in `paper_tables/`;
the larger family directories retain the underlying aggregates and complete
available raw provenance.

The experiment-to-code, environment, and raw-data map is maintained in
`../docs/EXPERIMENT_CATALOG.md`. “Complete” refers to all files retained from
the executed campaign; a cell that failed before producing a measurement is
documented rather than synthesized.

## Runtime and Enforcement Boundary

Production runtime tables live under `../runtime_experiments/`. MeritKV is
write-through in the broad vLLM, SGLang, and LMCache matrices; those results
measure compatibility and observed overlay cost. The four-arm multiround traces
under `memory_bound_trace_multiround/` separately enforce skip-write and
skip-lookup under capacity pressure.

## Fidelity Boundary

The long-prefix and fidelity folders use a custom Hugging Face
`DynamicCache` crop-and-splice diagnostic. Output agreement is
model-, precision-, and path-dependent. All Qwen float16 splice speedups are
excluded from validated performance. Runtime Qwen measurements use native
caching but are not correctness evidence.

## Interpretation Rules

- Read speedup together with waste, hit rate, and the baseline definition.
- Recovery is a cache-locality metric, not a latency metric.
- Raw-mode gains can reflect bypass and avoided overhead.
- Approximate semantic opportunities are not automatically safe substitutions.
- Historical provenance files and auxiliary one-off extensions do not override
  the current paper-facing summaries.
