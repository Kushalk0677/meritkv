# ShadowKV SGLang Qwen14B Identical-Blocks Deliverable

Created: 2026-06-18
Author: Keystone on IronGod

## Scope

- Model: `Qwen/Qwen2.5-14B-Instruct`
- Datasets: `daily_dialog`, `samsum`, `ag_news`, `dolly`, `xsum`
- Modes: `templated`, `rag`
- Repetitions: `3`
- Requests per cell: `256`
- Baselines: `sglang_radix_attention`, `sglang_radix_attention_shadowkv_plus`, `lmcache_no_native_radix`
- Runtime image: `shadowkv-sglang-lmcache:2026-06-14-no-native-radix`
- Runtime settings: context length `4096`, `mem-fraction-static=0.80`, chunked prefill disabled, dtype `float16`, Triton attention backend, PyTorch sampling backend, CUDA graphs disabled.

## Schedule

The run uses the same matched-block design as the cleaner three-model rerun:

- 30 matched blocks: `3 reps x 5 datasets x 2 prompt modes`.
- Every block contains all 3 baselines.
- Each baseline appears in positions 1/2/3 exactly 10 times.

## Completion

- Full matrix: `90/90`
- LMCache summaries: `30/30`
- Error scan hits: `0`
- Runtime config mismatches: `0`
- Darwin production service restored on first-light: `darwin28b-reason-vllm`, endpoint `127.0.0.1:8015/v1/models`

## Headline Result

ShadowKV++ improved over native SGLang Radix on Qwen14B:

- Mean latency: `-1.73%`
- P95 latency: `-2.07%`
- Throughput: `+1.76%`
- Cached-token delta: `0`

LMCache no-native-Radix remained functional but slower than native Radix:

- Mean latency: `+12.70%`
- P95 latency: `+21.81%`
- Throughput: `-11.18%`

Interpretation: this remains evidence for ShadowKV++ as an admission/policy overlay on native Radix/APC behavior, not as a replacement cache mechanism, because cached-token totals match native Radix exactly.

## Key Files

- `raw_results/results_sglang_lmcache_shadowkv_qwen14b_identical_blocks_3reps_2026-06-17/SUMMARY_QWEN14B_3BASELINES_3REPS.md`
- `raw_results/results_sglang_lmcache_shadowkv_qwen14b_identical_blocks_3reps_2026-06-17/ANOMALY_AUDIT_QWEN14B_IDENTICAL_BLOCKS_2026-06-18.md`
- `raw_results/results_sglang_lmcache_shadowkv_qwen14b_identical_blocks_3reps_2026-06-17/aggregate_qwen14b_3baselines_3reps_full.csv`
- `raw_results/results_sglang_lmcache_shadowkv_qwen14b_identical_blocks_3reps_2026-06-17/paired_delta_summary_by_model_qwen14b_3reps.csv`
- `run_logs/sglang_lmcache_shadowkv_qwen14b_identical_blocks_3reps_2026-06-17.log`
- `session_files/run_shadowkv_sglang_lmcache_qwen14b_identical_blocks_3reps_2026-06-17.sh`

