# ShadowKV SGLang Identical-Blocks Rerun Deliverable

Created: 2026-06-17
Author: Keystone on IronGod

## What Changed From The Previous 2026-06-16 Package

The earlier three-model repeated run completed successfully, but models were processed in model-major blocks. This package contains the cleaner rerun with matched workload blocks:

- 30 matched blocks: `3 reps x 5 datasets x 2 prompt modes`.
- Every block contains all 3 models and all 3 baselines.
- Each model appears in positions 1/2/3 exactly 30 times.
- Each baseline appears in positions 1/2/3 exactly 30 times.

The workload size and runtime settings are otherwise unchanged.

## Scope

- Models: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`
- Datasets: `daily_dialog`, `samsum`, `ag_news`, `dolly`, `xsum`
- Modes: `templated`, `rag`
- Repetitions: `3`
- Requests per cell: `256`
- Baselines: `sglang_radix_attention`, `sglang_radix_attention_shadowkv_plus`, `lmcache_no_native_radix`
- Runtime image: `shadowkv-sglang-lmcache:2026-06-14-no-native-radix`
- Runtime settings: context length `4096`, `mem-fraction-static=0.80`, chunked prefill disabled, dtype `float16`, Triton attention backend, PyTorch sampling backend, CUDA graphs disabled.

## Completion

- Full matrix: `270/270`
- LMCache summaries: `90/90`
- Error scan hits: `0`
- Runtime config mismatches: `0`
- Darwin production service restored on first-light: `darwin28b-reason-vllm`, endpoint `127.0.0.1:8015/v1/models`

## Headline Result

ShadowKV++ remains slightly slower than native Radix on 1.5B and 3B, and faster on 7B. Cached-token totals are identical to native Radix in every paired ShadowKV++ cell, so this should still be framed as an admission/policy overlay on native Radix rather than a replacement cache mechanism.

Patched LMCache no-native-Radix remains functional, but slower than native Radix on these public workloads because it gives up fine-grained Radix prefix reuse and only benefits when reuse reaches LMCache's 256-token chunk granularity.

## Key Files

- `raw_results/results_sglang_lmcache_shadowkv_3models_identical_blocks_3reps_2026-06-16/SUMMARY_3MODELS_3BASELINES_3REPS.md`
- `raw_results/results_sglang_lmcache_shadowkv_3models_identical_blocks_3reps_2026-06-16/ANOMALY_AUDIT_IDENTICAL_BLOCKS_2026-06-17.md`
- `raw_results/results_sglang_lmcache_shadowkv_3models_identical_blocks_3reps_2026-06-16/aggregate_3models_3baselines_3reps_full.csv`
- `raw_results/results_sglang_lmcache_shadowkv_3models_identical_blocks_3reps_2026-06-16/paired_delta_summary_by_model_3models_3reps.csv`
- `run_logs/sglang_lmcache_shadowkv_3models_identical_blocks_3reps_2026-06-16.log`
- `session_files/run_shadowkv_sglang_lmcache_3models_identical_blocks_3reps_2026-06-16.sh`

