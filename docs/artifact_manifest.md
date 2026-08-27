# Repository Artifact Manifest

This is the complete working repository, not the compact anonymised reviewer
snapshot. It contains source, measured aggregates, selected raw campaigns,
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

## Scope Classes

| Class | Examples | How to use |
|---|---|---|
| Paper-facing aggregate | `results/paper_tables/`, top-level runtime CSVs, and current summaries | Use for reported values and claims. |
| Complete paper-facing runtime bundle | `runtime_experiments/qwen2.5/sglang/balanced_admission/` | Audit the native balanced-admission table from aggregate, request-level, log, command, and environment records. |
| Raw campaign provenance | Runtime `raw/`, long-prefix `provenance/`, operator notes | Audit a specific campaign; do not override current aggregation. |
| Auxiliary experiment | Blackwell semantic n=128, Gemma all-engine extension | Method development or follow-up analysis; not a distinct paper result. |
| Historical development report | `docs/reports/`, `experiments/archive/` | Implementation history only. |
| Modelled estimate | `results/energy_estimates/` | Excluded from measured paper evidence. |

## External-Only Artifacts

Large transfer packages, environment archives, model weights, downloaded
datasets, caches, and duplicate repository snapshots should remain outside git.
Attach release-quality archives separately and record checksums and provenance.

## Consistency Rule

Campaign-specific notes may retain their original numbers, but they must carry
a scope notice. Current READMEs and the claim map use the paper's aggregation,
write-through/enforced distinction, and fidelity limitations. No missing seed
record should be reconstructed from a paper aggregate.
