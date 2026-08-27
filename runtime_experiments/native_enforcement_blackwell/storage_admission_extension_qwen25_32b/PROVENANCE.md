# Provenance and Reproduction Boundary

## Frozen evidence package

| Package | SHA-256 | Verification |
|---|---|---|
| `MeritKV-Blackwell-storage-admission-extension-Qwen2.5-32B-2026-08-24.zip` | `5b2b81fd335ab89266591a94582b43b15bfa07e0037ff1caec5752cd8f242f7c` | Bundled verifier passes 31 cells and 3,344 requests |

The delivered ZIP and its `.sha256` sidecar are retained byte-for-byte in the private evidence repository. Its complete extracted tree remains authoritative for request traces, native counters, server logs, launch receipts, freeze records, output hashes, and restoration evidence.

The verifier command is:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 packet/verify_extension.py delivery results --packet packet
```

It is intended for Linux or WSL, matching the experiment environment. The frozen verifier's source-snapshot key comparison uses POSIX path separators, so native Windows execution is not the reference verification path.

## Frozen configuration

- Model: `Qwen/Qwen2.5-32B-Instruct`
- Revision: `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`
- Hardware: NVIDIA RTX PRO 6000 Blackwell, 96 GB
- Runtime: SGLang 0.5.13
- Precision and kernels: float16, Triton attention, PyTorch sampling
- Native cache: ordinary `RadixCache`, `hybrid_swa=False`, 16,384-token capacity, page size 1
- Retention: LRU for four arms and native LFU for the frequency-aware baseline
- Evaluation seeds: 32001, 32002, 32003, 32004, 32005
- Requests: 124 per seed and arm
- Generation: temperature 0, one output token
- Execution order: frozen 5-by-5 Latin-square plan

## Curated-code inventory

The `code/` directory retains the executed components that materially support review:

- `packet/generate_frozen_inputs.py`: deterministic trace and paired-arm-plan generation.
- `packet/run_capacity_cell.py`: storage-admission action mapping and physical metric capture.
- `packet/verify_extension.py`: relocatable integrity, completion, counter, output, and scientific checks for the full private package.
- `packet/freeze_evaluation.py`: calibration-to-evaluation freeze checks.
- `packet/measure_token_budget.py`: pinned-tokenizer capacity validation.
- `packet/evaluation_plan_25.tsv` and `packet/smoke_plan_5.tsv`: frozen run order.
- `packet/feasibility/lfu_feasibility_gate.json`: predeclared LFU inclusion gate.
- `packet/tests/test_lazy_eviction_metrics.py`: physical-eviction metric regression test.
- `meritkv_blackwell_20260817/` and `meritkv_blackwell_20260820/`: preserved instrumentation wrappers used by the cell runner.
- `source_snapshot/`: compact Python import closure used by those wrappers.

Machine-specific service units, server restoration controls, raw commands, server logs, and request traces remain only in the complete private package.
Author/operator identity metadata is also omitted from the anonymous curated derivative.

## Aggregate construction

`paired_seed_results.csv` is extracted from each frozen evaluation cell's `evaluation_verification.json` record and benchmark JSON. Decision and physical-action counts are read from the benchmark metrics rather than inferred from arm names. `aggregate_results.csv` reports arithmetic means across the five paired seeds. Percentage changes and Student-t 95% confidence intervals use the five paired per-seed ratios, not ratios of rounded aggregate means.

No missing cell or unavailable metric was reconstructed. The exact-output field comes from the package-level cross-arm verification, which reports zero mismatch positions across the evaluated matrix.
