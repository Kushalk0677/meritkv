# Provenance and Reproduction Boundary

## Evidence packages

| Package | SHA-256 | Manifest verification |
|---|---|---|
| Blackwell storage-admission extension, 2026-08-23 | `8ea89aae9c21963a5983898333aebfef488edb45e83908f1755e021336d38797` | 258/258 entries matched |
| August 20 native-enforcement self-containment supplement, 2026-08-23 | `1e8a7e28901921e50ca89d24a1d00f9e14eb71c123594e696e8848f6e0acb2fa` | 64/64 entries matched |
| Original August 20 native-enforcement package | `e7943267252fe160afdbbe37c72884eccd029cc66f22419688657d962d5f749b` | Referenced unchanged by the supplement |

The bundled storage-extension verifier completed successfully with 25 cells and 2,688 requests. The original package and its 970-file manifest remain unchanged; the self-containment supplement adds the preserved runner, source import closure, and implementation lineage without rewriting the earlier evidence.

## Frozen configuration

- Model: `google/gemma-4-31B-it`
- Revision: `b9ea41a2887d8607f594846523f94c6cc75ac8a4`
- Hardware: NVIDIA RTX PRO 6000 Blackwell, 96 GB
- Runtime: SGLang 0.5.13
- Precision and kernels: float16, Triton attention, PyTorch sampling
- Native cache: Gemma `SWARadixCache`, LRU, 16,384-token capacity, page size 1
- Evaluation seeds: 32001, 32002, 32003, 32004, 32005
- Requests: 124 per seed and arm
- Generation: temperature 0, one output token
- Input order: frozen before the evaluation matrix

## Public code inventory

The `code/` directory preserves the parts of the executed implementation that materially support review:

- `packet/generate_frozen_inputs.py`: deterministic trace and paired-arm-plan generation.
- `packet/run_capacity_cell.py`: independent skip-write and skip-lookup action mapping plus physical eviction capture.
- `packet/verify_extension.py`: relocatable integrity, completion, counter, and scientific checks.
- `packet/freeze_evaluation.py`: calibration-to-evaluation freeze checks.
- `packet/measure_token_budget.py`: pinned-tokenizer capacity-budget validation.
- `packet/tests/test_lazy_eviction_metrics.py`: metric-collection regression test.
- `runtime_patch/apply_swa_native_admission.py`: fail-closed native hook for Gemma `SWARadixCache`.
- `meritkv_blackwell_20260817/` and `meritkv_blackwell_20260820/`: preserved instrumentation wrappers used by the cell runner.
- `source_snapshot/`: compact Python import closure used by those wrappers.

The server-orchestration controller is not in the public package because it contains machine-specific service paths and restoration controls. This omission does not remove the executed cell logic, frozen protocol, cache hook, or verifier. Full commands, server logs, request traces, receipts, and the complete evidence packages are preserved under `evidence_packages/`.

## Aggregate construction

`paired_seed_results.csv` contains direct scalar fields read from the 20 evaluation-cell benchmark JSON files and the corresponding native counters. `aggregate_results.csv` reports arithmetic means across the five paired seeds. Percentage changes and 95% confidence intervals use the mean and Student-t interval of the five paired per-seed ratios, not ratios of rounded aggregate means.

No missing cell, failed LFU run, or unavailable record was reconstructed. The optional native LFU attempt is excluded because it produced zero successful cells; its failure receipt and crash log remain in the private evidence package.
