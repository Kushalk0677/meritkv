# Methodology Index

This page connects the maintained method description to campaign-specific
protocols. Frozen package instructions take precedence when reproducing an old
measurement because framework APIs and launch flags changed across campaigns.

| Topic | Maintained description | Campaign evidence |
|---|---|---|
| Utility and admission decision | `meritkv_research_design.md`, `design_overview.md`, `src/proactive_kv_cache/controller.py`, `utility.py` | Controlled and mixed/admission families in `docs/EXPERIMENT_CATALOG.md` |
| Cost calibration and breakeven | `algorithm_details.md`, `learned_admission_baseline_math.md`, configuration and run manifests | `results/controlled_results/`, `results/learned_admission_baseline/` |
| Workload construction | `src/proactive_kv_cache/datasets.py`, `workload.py`, runner arguments | Per-cell manifests and command records |
| Process isolation and aggregation | `reproducing_results.md`, family READMEs | `results/controlled_results/`, `isolated_baseline_comparison/`, `realistic_results/` |
| Custom-splice fidelity | `experimental_setup.md`, `semantic_fidelity.md`, `fidelity_deep_analysis.md` | `results/fidelity_examples/`, `results/exact_splice_validation/`, Colab packages |
| Runtime overlays | `runtime_experiments.md`, baseline adapter docs | `runtime_experiments/qwen2.5/`, `runtime_experiments/gemma4/` |
| Native enforcement | Package plans, commands, verifier rules, action counters | `runtime_experiments/native_enforcement_blackwell/` |
| Capacity pressure and locality | Family reports and trace runners | `results/memory_bound_trace/`, `memory_bound_trace_multiround/` |
| Hardware and software | `HARDWARE_AND_ENVIRONMENTS.md`, `DEPENDENCIES_AND_PACKAGES.md` | Package metadata, manifests, `pip_freeze`, `nvidia-smi`, server logs |
| Integrity and scope | `REPOSITORY_STRUCTURE.md`, `RAW_EVIDENCE_COMPLETENESS.md` | `RELEASE_INVENTORY.csv`, archive checksums, claim map |

## Uniform Reproduction Sequence

1. Select the family from `EXPERIMENT_CATALOG.md`.
2. Read its README and distinguish measured, modelled, write-through, enforced,
   diagnostic, and historical scope.
3. Install the family-pinned environment rather than assuming the root
   requirements reproduce every historical backend.
4. Fetch the recorded public model and dataset revisions.
5. Preserve seeds, request order, precision, prompt mode, cache budget, warm-up,
   and generation parameters.
6. Run the package verifier or aggregation script without result-driven tuning.
7. Compare generated summaries with raw cells and the paper table; never fill a
   failed or absent cell from an aggregate.
