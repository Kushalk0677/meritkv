# Repository Artifact Manifest

This is the complete public research repository, not the compact review
artifact snapshot. It contains source, measured aggregates, complete project-retained raw campaigns,
historical provenance, and explicitly modelled estimates. The current paper and
`CLAIMS_TO_ARTIFACTS.md` define the paper-facing evidence boundary.

## Current Paper Evidence

| Evidence family | Repository path |
|---|---|
| Canonical machine-readable paper tables | `results/paper_tables/` |
| Controlled T4/P100 study | `results/controlled_results/` |
| Process-isolated baseline comparison | `results/isolated_baseline_comparison/` |
| Five-model process-isolated study | `results/realistic_results/` |
| Blackwell long-prefix study | `results/blackwell_longprefix_hf/` |
| Fidelity diagnostics | `results/fidelity_examples/`, `docs/results_table.md` |
| Runtime compatibility | `runtime_experiments/qwen2.5/`, `runtime_experiments/gemma4/` |
| Native SGLang balanced admission | `runtime_experiments/qwen2.5/sglang/balanced_admission/` |
| Mixed/admission and learned baselines | `results/mixed_traffic/`, `results/learned_admission_baseline/` |
| Capacity-pressure evidence | `results/memory_bound_trace/`, `results/memory_bound_trace_multiround/` |
| Expanded reproduction and Colab packages | `reproduction_packages/` |
| Native enforcement and both storage-admission extensions | `runtime_experiments/native_enforcement_blackwell/` |

## Scope Classes

| Class | Examples | How to use |
|---|---|---|
| Paper-facing aggregate | `results/paper_tables/`, top-level runtime CSVs, and current summaries | Use for reported values and claims. |
| Complete paper-facing runtime bundle | `runtime_experiments/qwen2.5/sglang/balanced_admission/` | Audit the native balanced-admission table from aggregate, request-level, log, command, and environment records. |
| Raw campaign provenance | Runtime `raw/`, long-prefix `provenance/`, operator notes | Audit a specific campaign; do not override current aggregation. |
| Auxiliary experiment | Blackwell semantic n=128, Gemma all-engine extension | Method development or follow-up analysis; not a distinct paper result. |
| Historical development report | `docs/reports/`, `experiments/archive/` | Implementation history only. |
| Modelled estimate | `results/energy_estimates/` | Excluded from measured paper evidence. |

## Excluded Runtime Dependencies

All retained research transfer packages are included as named archives and/or
expanded trees. Model weights, downloaded datasets, virtual environments,
framework caches, and GPU-driver installations remain external dependencies.
They are recoverable from public identifiers and version records rather than
being redistributed.

## Consistency Rule

Campaign-specific notes may retain their original numbers, but they must carry
a scope notice. Current READMEs and the claim map use the paper's aggregation,
write-through/enforced distinction, and fidelity limitations. No missing seed
record should be reconstructed from a paper aggregate.

The uniform family-level index is `EXPERIMENT_CATALOG.md`; repository layout
and evidence-editing rules are in `REPOSITORY_STRUCTURE.md`.
