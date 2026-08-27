# Hardware and Software Environments

This index separates measured machines from analytical hardware projections.
The linked manifests, `nvidia-smi` captures, package receipts, and environment
JSON files are the authoritative records; this page is a navigation layer.

## Measured Platforms

| Platform | Principal use | Recorded configuration | Primary captures |
|---|---|---|---|
| NVIDIA Tesla T4 | Controlled HF study; fidelity and exact-splice diagnostics | 16 GB class T4; campaign-specific Python, PyTorch, Transformers, CUDA, precision, and attention implementation are stored with each run | `results/controlled_results/`, `results/exact_splice_validation/`, `reproduction_packages/colab/` |
| NVIDIA Tesla P100-PCIE-12GB | Controlled HF study and process-isolated baselines | 12,288 MiB, driver `570.133.20` in retained manifests | `results/controlled_results/`, `results/isolated_baseline_comparison/`, `reproduction_packages/p100_hf/` |
| NVIDIA RTX PRO 6000 Blackwell | Long-prefix HF, production-runtime overlays, native enforcement, and storage admission | 96 GB class Blackwell; campaign snapshots record driver, CUDA, backend revision, model revision, launch command, and topology | `results/blackwell_longprefix_hf/`, `runtime_experiments/gemma4/`, `runtime_experiments/qwen2.5/`, `runtime_experiments/native_enforcement_blackwell/` |

The explicit-state continuation package records, for example, Tesla T4,
Python 3.12.13, PyTorch 2.11.0+cu128, Transformers 5.10.2, CUDA 12.8, float16,
and SDPA. Those versions apply to that validation package only; production
runtime packages carry their own pinned stack.

## Modelled Platforms

A100, H100, and H200 values used in breakeven or energy analyses are analytical
hardware substitutions, not measurements from those GPUs. They are labelled as
modelled in `results/energy_estimates/` and the corresponding derivation docs.
They must not be used as raw runtime evidence.

## Reproduction Requirements

- Use the exact model/checkpoint revision in the run manifest or package plan.
- Use the recorded precision and attention implementation; cached continuation
  can be numerically path-dependent.
- Preserve request order, warm-up policy, seed, generation parameters, cache
  budget, and admission thresholds.
- For native enforcement, verify requested actions against executed backend
  counters rather than relying only on controller logs.
- Model weights and datasets are fetched from their public identifiers and are
  not redistributed in this repository.
