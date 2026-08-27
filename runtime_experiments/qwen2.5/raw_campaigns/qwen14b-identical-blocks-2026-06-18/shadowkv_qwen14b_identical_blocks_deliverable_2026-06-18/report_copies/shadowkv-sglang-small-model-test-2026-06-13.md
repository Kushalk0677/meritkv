---
title: ShadowKV SGLang Small Model Test 2026-06-13
created: 2026-06-13
updated: 2026-06-18
author: Keystone (IronGod)
authors: [Keystone]
last_editor: Keystone (IronGod)
type: project
tags: [shadowkv, sglang, lmcache, qwen14b, first-light]
sources: []
---

# ShadowKV SGLang Small Model Test - 2026-06-13

## Result

The smaller-model ShadowKV++ test completed on `Qwen/Qwen2.5-14B-Instruct`.

The LMCache isolation request exposed a SGLang integration limitation: in this SGLang build, passing `--disable-radix-cache` prevents `LMCRadixCache` from being constructed. That means `--enable-lmcache --disable-radix-cache` does not measure LMCache retrieval; it bypasses the LMCache tree path entirely.

2026-06-14 update: this limitation was worked around with a patched SGLang/LMCache runtime image. See [[#True LMCache Without Native Radix Patch - 2026-06-14]].

## LMCache Isolation Check

Host: `first-light`

Image: `shadowkv-sglang-lmcache:2026-06-08-layerwise-patch`

Config: `/home/jade_hand/research/shadowkv/session_files/lmcache_sglang_layerwise_30gb_2026-06-13.yaml`

I ran two LMCache checks on Qwen14B:

1. Public dataset smoke: `daily_dialog`, `templated`, 64 requests, `--disable-radix-cache`.
2. Controlled long-prefix trace: 32 requests with a repeated prefix large enough to exceed the 256-token LMCache chunk threshold, also with `--disable-radix-cache`.

Both completed, but both produced:

- `cached_tokens_total=0`
- no LMCache retrieval log lines
- no LMCache store/retrieve counters indicating useful retrieval

Source check:

`sglang/srt/managers/scheduler.py` only constructs `LMCRadixCache` inside the non-disabled radix-cache branch. With `--disable-radix-cache`, SGLang chooses `ChunkCache` instead, so LMCache is not active even though `--enable-lmcache` is accepted in server args.

Conclusion: Kushal's requested `--enable-lmcache --disable-radix-cache` isolation method is not valid for this installed SGLang path. Disabling Radix disables the LMCache radix backend too.

## LMCache Version-Matched Smoke

I built and tested a disposable version-matched image:

`shadowkv-sglang-lmcache:2026-06-13-sglang0513-lmcache047`

Version pair:

- `sglang==0.5.13`
- `lmcache==0.4.7`

Build file:

`/home/jade_hand/research/shadowkv/session_files/Dockerfile.sglang_lmcache_2026-06-13-versionmatch-0513-047`

Smoke script:

`/home/jade_hand/research/shadowkv/session_files/run_lmcache_versionmatch_smoke_2026-06-13.sh`

Result root:

`/home/jade_hand/research/shadowkv/results_lmcache_versionmatch_qwen14b_smoke_2026-06-13`

Config:

`/home/jade_hand/research/shadowkv/session_files/lmcache_mp_qwen14b_versionmatch_2026-06-13.yaml`

Smoke design:

1. Start LMCache MP server with 20 GB CPU L1 cache.
2. Start SGLang Qwen2.5-14B with `--enable-lmcache --lmcache-config-file ...`.
3. Send one long-prefix request to populate Radix and LMCache.
4. Send the same request again to confirm native Radix hit.
5. `POST /flush_cache` to clear SGLang Radix.
6. Send the same request a third time to test LMCache retrieval after Radix is cleared.

Observed responses:

| Step | Elapsed | Cached tokens |
| --- | ---: | ---: |
| First store | 0.862 s | none reported |
| Second Radix hit | 0.215 s | 2,645 |
| Third after Radix flush | 0.070 s | 2,560 |

LMCache server log evidence:

- `Stored 2560 tokens in 0.009 seconds`
- `Retrieved 2560 tokens in 0.001 seconds`

SGLang log evidence:

- Before flush: `#cached-token: 2645`
- After flush: `#cached-token: 2560`

Interpretation:

Version matching works, but not as Radix-disabled isolation. In SGLang 0.5.13, LMCache is still constructed as `LMCRadixCache`, so it is best treated as a Radix-plus-host-cache path. It can retrieve KV from LMCache after the native Radix tree is flushed or missing the prefix. The valid LMCache comparison should therefore be framed as SGLang Radix vs SGLang Radix+LMCache spill/retrieve, not as SGLang no-Radix vs LMCache.

## ShadowKV++ Qwen14B Matrix

Scope:

- Model: `Qwen/Qwen2.5-14B-Instruct`
- Datasets: `daily_dialog`, `samsum`, `ag_news`, `dolly`, `xsum`
- Modes: `templated`, `rag`
- Baselines: `sglang_radix_attention`, `sglang_radix_attention_shadowkv_plus`
- Requests per cell: 64
- Total cells: 20
- Max generated tokens: 1
- Energy: NVML enabled

Result root:

`/home/jade_hand/research/shadowkv/results_sglang_shadowkv_qwen14b_small_2026-06-13`

Run script:

`/home/jade_hand/research/shadowkv/session_files/run_shadowkv_sglang_qwen14b_small_shadowkv_2026-06-13.sh`

Summary artifacts:

- `/home/jade_hand/research/shadowkv/results_sglang_shadowkv_qwen14b_small_2026-06-13/summary_sglang_shadowkv_qwen14b_small_2026-06-13.csv`
- `/home/jade_hand/research/shadowkv/results_sglang_shadowkv_qwen14b_small_2026-06-13/summary_sglang_shadowkv_qwen14b_small_2026-06-13.json`
- `/home/jade_hand/research/shadowkv/results_sglang_shadowkv_qwen14b_small_2026-06-13/summary_sglang_shadowkv_qwen14b_small_2026-06-13.md`

Aggregate averages:

| Baseline | Mean latency ms | P95 latency ms | Throughput rps | Cached tokens total | Idle-adjusted J/request | Admission plans | Allows | Bypasses | Stores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sglang_radix_attention` | 45.25 | 50.12 | 22.25 | 86542 | 18.69 | 0 | 0 | 0 | 0 |
| `sglang_radix_attention_shadowkv_plus` | 44.44 | 48.55 | 22.64 | 86542 | 18.70 | 640 | 630 | 10 | 640 |

Interpretation:

ShadowKV++ overlay was active and produced policy decisions on every measured request. It was slightly faster than native SGLang Radix on this 14B small test: mean latency improved by about 1.8%, P95 by about 3.1%, and throughput by about 1.8%. Energy was effectively tied.

The cached-token total did not change between native SGLang and ShadowKV++ because this overlay is acting as admission/policy over the native runtime cache, not as a separate direct KV injection path.

Per-cell ShadowKV++ vs SGLang:

| Dataset | Mode | Latency delta | P95 delta | Throughput delta | Energy delta | Cached token delta | Plans | Allows | Bypasses | Stores |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ag_news` | `templated` | -1.40% | -1.47% | 1.42% | 0.52% | 0 | 64 | 63 | 1 | 64 |
| `ag_news` | `rag` | -1.59% | -1.53% | 1.62% | -3.86% | 0 | 64 | 63 | 1 | 64 |
| `daily_dialog` | `templated` | -1.39% | -2.83% | 1.41% | 0.94% | 0 | 64 | 63 | 1 | 64 |
| `daily_dialog` | `rag` | -1.45% | -1.82% | 1.47% | 0.06% | 0 | 64 | 63 | 1 | 64 |
| `dolly` | `templated` | -1.58% | -3.67% | 1.61% | -1.54% | 0 | 64 | 63 | 1 | 64 |
| `dolly` | `rag` | -1.82% | -2.21% | 1.85% | 0.32% | 0 | 64 | 63 | 1 | 64 |
| `samsum` | `templated` | -1.82% | -1.75% | 1.85% | -3.10% | 0 | 64 | 63 | 1 | 64 |
| `samsum` | `rag` | -1.87% | -1.83% | 1.90% | 4.96% | 0 | 64 | 63 | 1 | 64 |
| `xsum` | `templated` | -2.09% | -3.15% | 2.13% | 0.00% | 0 | 64 | 63 | 1 | 64 |
| `xsum` | `rag` | -2.52% | -9.38% | 2.59% | 1.66% | 0 | 64 | 63 | 1 | 64 |

## First-Light State After 2026-06-13 Run

After the run:

- `qwen36-27b-fp8-vllm` was restored and health-checked on `127.0.0.1:8014`.
- No SGLang benchmark processes remained active.
- The working patched image remained: `shadowkv-sglang-lmcache:2026-06-08-layerwise-patch`.

## 2026-06-13 Next Step (Superseded)

For LMCache specifically, the next valid experiment is a version-matched `sglang==0.5.13` / `lmcache==0.4.7` matrix framed as Radix vs Radix+LMCache. The smoke proves LMCache store/retrieve works after Radix flush, but it does not support Kushal's requested Radix-disabled isolation.

For ShadowKV, the small-model test is valid as a policy-overlay comparison against native SGLang Radix.

Supersession: the 2026-06-14 runtime patch below enabled a true no-native-Radix LMCache baseline.

## True LMCache Without Native Radix Patch - 2026-06-14

I built a patched SGLang/LMCache image so `--enable-lmcache --disable-radix-cache` measures LMCache retrieval without native SGLang Radix satisfying the prefix hits first.

Image:

`shadowkv-sglang-lmcache:2026-06-14-no-native-radix`

Patch file:

`/home/jade_hand/research/shadowkv/session_files/Dockerfile.sglang_lmcache_2026-06-14-no-native-radix`

Patched runtime behavior:

- `registry.py` constructs `LMCRadixCache` when `--enable-lmcache` is set, even if `--disable-radix-cache` is also set.
- `lmc_radix_cache.py` treats `disable_radix_cache=True` as no-native-Radix mode: `match_prefix()` starts from an empty device-tree match, finished requests store to LMCache before their KV slots are freed, and temporary host-loaded Radix nodes are evicted so future hits must come from LMCache rather than resident native Radix state.

Smoke result:

- Script: `/home/jade_hand/research/shadowkv/session_files/run_lmcache_no_native_radix_smoke_2026-06-14.sh`
- Result root: `/home/jade_hand/research/shadowkv/results_lmcache_no_native_radix_qwen14b_smoke_2026-06-14`
- Server args included both `--enable-lmcache` and `--disable-radix-cache`.
- No manual `/flush_cache` was used between repeated requests.
- First long-prefix request stored KV; second and third requests reported `cached_tokens=2560`.
- LMCache server log showed `Stored 2560 tokens in 0.008 seconds` and two `Retrieved 2560 tokens in 0.001 seconds` events.

Matrix scope:

- Model: `Qwen/Qwen2.5-14B-Instruct`
- Datasets: `daily_dialog`, `samsum`, `ag_news`, `dolly`, `xsum`
- Modes: `templated`, `rag`
- Requests per cell: 64
- Baseline: `lmcache_no_native_radix`
- Energy: NVML enabled

Result root:

`/home/jade_hand/research/shadowkv/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14`

Summary artifacts:

- `/home/jade_hand/research/shadowkv/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14/summary_lmcache_no_native_radix_qwen14b_2026-06-14.csv`
- `/home/jade_hand/research/shadowkv/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14/summary_lmcache_no_native_radix_qwen14b_2026-06-14.json`
- `/home/jade_hand/research/shadowkv/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14/summary_lmcache_no_native_radix_qwen14b_2026-06-14.md`

Aggregate:

| Cells | Mean latency ms | P95 latency ms | Throughput rps | Idle-adjusted J/request | Prompt tokens | Cached tokens | LMCache retrieves | LMCache stores |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 51.56 | 62.97 | 19.61 | 23.74 | 167304 | 7168 | 27 | 242 |

Per-cell results:

| Dataset | Mode | Mean ms | P95 ms | RPS | Idle J/req | Prompt tokens | Cached tokens | Retrieves | Stores |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ag_news` | `templated` | 46.15 | 47.90 | 21.67 | 20.59 | 11153 | 0 | 0 | 0 |
| `ag_news` | `rag` | 46.44 | 49.77 | 21.53 | 21.24 | 11857 | 0 | 0 | 0 |
| `daily_dialog` | `templated` | 47.38 | 59.27 | 21.10 | 21.28 | 13043 | 256 | 1 | 7 |
| `daily_dialog` | `rag` | 48.51 | 60.57 | 20.61 | 22.55 | 14003 | 256 | 1 | 12 |
| `dolly` | `templated` | 48.33 | 64.27 | 20.69 | 22.58 | 12309 | 0 | 0 | 13 |
| `dolly` | `rag` | 48.77 | 64.20 | 20.50 | 22.22 | 13461 | 0 | 0 | 15 |
| `samsum` | `templated` | 52.59 | 68.59 | 19.02 | 23.78 | 17787 | 1024 | 4 | 37 |
| `samsum` | `rag` | 53.09 | 67.69 | 18.84 | 24.35 | 18747 | 1280 | 5 | 38 |
| `xsum` | `templated` | 61.38 | 70.61 | 16.29 | 28.72 | 26992 | 2048 | 8 | 60 |
| `xsum` | `rag` | 62.95 | 76.85 | 15.88 | 30.11 | 27952 | 2304 | 8 | 60 |

Interpretation:

This is the first valid no-native-Radix LMCache baseline in this SGLang setup. It proves LMCache can retrieve KV without Radix serving as the visible hit source, but the public-workload reuse is much smaller than native SGLang Radix because LMCache is chunk-aligned at 256 tokens and does not preserve Radix's fine-grained short-prefix matches. Compared with the 2026-06-13 Qwen14B native SGLang Radix aggregate, mean latency moved from `45.25 ms` to `51.56 ms`, throughput from `22.25 rps` to `19.61 rps`, cached tokens from `86542` to `7168`, and idle-adjusted energy from `18.69` to `23.74` J/request.

Production restore:

- `darwin28b-reason-vllm` was restarted after the run.
- `http://127.0.0.1:8015/v1/models` returned `darwin28b-reason`.
- No no-native-Radix benchmark container remained active.

## Three-Model Repeated SGLang Baseline Run - 2026-06-16

Kushal requested a more rigorous follow-up: all three baselines on smaller models, repeated three times, with randomized execution order, 256 requests per cell, identical runtime settings, and recorded hardware/package metadata.

Scope:

- Models: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`
- Datasets: `daily_dialog`, `samsum`, `ag_news`, `dolly`, `xsum`
- Modes: `templated`, `rag`
- Repetitions: 3
- Requests per cell: 256
- Baselines: `sglang_radix_attention`, `sglang_radix_attention_shadowkv_plus`, `lmcache_no_native_radix`
- Total jobs: 270
- Runtime image for all baselines: `shadowkv-sglang-lmcache:2026-06-14-no-native-radix`
- Shared runtime settings: context length `4096`, `mem-fraction-static=0.80`, chunked prefill disabled, dtype `float16`, Triton attention backend, PyTorch sampling backend, CUDA graphs disabled.

Result root:

`/home/jade_hand/research/shadowkv/results_sglang_lmcache_shadowkv_3models_3reps_2026-06-16`

Run script:

`/home/jade_hand/research/shadowkv/session_files/run_shadowkv_sglang_lmcache_3models_3reps_2026-06-16.sh`

Status:

- Smoke passed before the full run.
- Full matrix completed: `270/270`.
- LMCache log summaries: `90/90`.
- Aggregation artifacts were generated at the result root.
- `darwin28b-reason-vllm` was restored and health-checked on `127.0.0.1:8015`.

Primary summary:

| Model | Baseline | Cells | Mean latency ms | P95 latency ms | Throughput rps | Idle J/req | Cached tokens | Plans | Allows | Bypasses | LMCache retrieves | LMCache stores |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen25_15b` | `lmcache_no_native_radix` | 30 | 12.38 | 15.09 | 81.67 | 2.91 | 23552 | 0 | 0 | 0 | 87 | 2902 |
| `qwen25_15b` | `sglang_radix_attention` | 30 | 11.32 | 13.18 | 88.61 | 2.21 | 1055310 | 0 | 0 | 0 | 0 | 0 |
| `qwen25_15b` | `sglang_radix_attention_shadowkv_plus` | 30 | 11.41 | 13.28 | 87.97 | 2.24 | 1055310 | 7680 | 7650 | 30 | 0 | 0 |
| `qwen25_3b` | `lmcache_no_native_radix` | 30 | 16.93 | 20.58 | 59.98 | 5.81 | 24064 | 0 | 0 | 0 | 87 | 2897 |
| `qwen25_3b` | `sglang_radix_attention` | 30 | 15.30 | 17.55 | 65.73 | 4.42 | 1055378 | 0 | 0 | 0 | 0 | 0 |
| `qwen25_3b` | `sglang_radix_attention_shadowkv_plus` | 30 | 15.39 | 17.68 | 65.33 | 4.43 | 1055378 | 7680 | 7650 | 30 | 0 | 0 |
| `qwen25_7b` | `lmcache_no_native_radix` | 30 | 28.09 | 35.54 | 36.23 | 12.73 | 23040 | 0 | 0 | 0 | 85 | 2964 |
| `qwen25_7b` | `sglang_radix_attention` | 30 | 24.23 | 27.49 | 41.58 | 9.77 | 1055719 | 0 | 0 | 0 | 0 | 0 |
| `qwen25_7b` | `sglang_radix_attention_shadowkv_plus` | 30 | 23.52 | 26.65 | 42.81 | 9.74 | 1055719 | 7680 | 7650 | 30 | 0 | 0 |

Paired deltas vs native Radix:

| Model | Comparison | Paired cells | Mean latency delta | Mean P95 delta | Mean throughput delta | Cached token delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen25_15b` | `lmcache_no_native_radix` vs Radix | 30 | 9.09% | 14.47% | -8.02% | -1031758 |
| `qwen25_15b` | ShadowKV++ vs Radix | 30 | 0.74% | 0.81% | -0.71% | 0 |
| `qwen25_3b` | `lmcache_no_native_radix` vs Radix | 30 | 10.27% | 17.01% | -9.03% | -1031314 |
| `qwen25_3b` | ShadowKV++ vs Radix | 30 | 0.61% | 0.83% | -0.60% | 0 |
| `qwen25_7b` | `lmcache_no_native_radix` vs Radix | 30 | 15.45% | 29.18% | -13.18% | -1032679 |
| `qwen25_7b` | ShadowKV++ vs Radix | 30 | -2.90% | -3.04% | 2.99% | 0 |

Interpretation:

This run confirms Kushal's caution. The original single-pass Qwen14B ShadowKV++ result was promising but not conclusive. In the repeated small-model matrix, ShadowKV++ is approximately parity/slightly slower than native SGLang Radix on 1.5B and 3B, and faster on 7B. Cached-token totals remain identical to native Radix for ShadowKV++, so the overlay is still best described as admission/policy over native Radix rather than a separate KV injection path. The patched LMCache no-native-Radix baseline works across all three models, but it is slower than native Radix on this workload because it loses most fine-grained Radix prefix reuse.

## Identical-Blocks Rerun - 2026-06-17

This rerun addresses the remaining ordering caveat in the first three-model repeated run. The workload size and runtime settings are unchanged, but the schedule is now arranged as matched workload blocks instead of model-major blocks.

Schedule:

- 30 matched blocks: `3 reps x 5 datasets x 2 prompt modes`.
- Each block contains all three models and all three baselines.
- Model position and baseline position are perfectly balanced: each model appears in positions 1/2/3 exactly 30 times, and each baseline appears in positions 1/2/3 exactly 30 times.

Result root:

`/home/jade_hand/research/shadowkv/results_sglang_lmcache_shadowkv_3models_identical_blocks_3reps_2026-06-16`

Run script:

`/home/jade_hand/research/shadowkv/session_files/run_shadowkv_sglang_lmcache_3models_identical_blocks_3reps_2026-06-16.sh`

Status:

- Full matrix completed: `270/270`.
- LMCache log summaries: `90/90`.
- Aggregate CSV/JSON/Markdown artifacts generated.
- Audit artifact generated: `ANOMALY_AUDIT_IDENTICAL_BLOCKS_2026-06-17.md`.
- `darwin28b-reason-vllm` was restored and health-checked on `127.0.0.1:8015`.

Primary summary:

| Model | Baseline | Cells | Mean latency ms | P95 latency ms | Throughput rps | Idle J/req | Cached tokens | Plans | Allows | Bypasses | LMCache retrieves | LMCache stores |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen25_15b` | `lmcache_no_native_radix` | 30 | 12.36 | 15.11 | 81.83 | 2.93 | 23552 | 0 | 0 | 0 | 87 | 2902 |
| `qwen25_15b` | `sglang_radix_attention` | 30 | 11.28 | 13.05 | 88.99 | 2.23 | 1055310 | 0 | 0 | 0 | 0 | 0 |
| `qwen25_15b` | `sglang_radix_attention_shadowkv_plus` | 30 | 11.40 | 13.23 | 88.00 | 2.24 | 1055310 | 7680 | 7650 | 30 | 0 | 0 |
| `qwen25_3b` | `lmcache_no_native_radix` | 30 | 16.92 | 20.57 | 59.96 | 5.82 | 23552 | 0 | 0 | 0 | 87 | 2902 |
| `qwen25_3b` | `sglang_radix_attention` | 30 | 15.29 | 17.54 | 65.80 | 4.41 | 1055310 | 0 | 0 | 0 | 0 | 0 |
| `qwen25_3b` | `sglang_radix_attention_shadowkv_plus` | 30 | 15.41 | 17.69 | 65.29 | 4.42 | 1055310 | 7680 | 7650 | 30 | 0 | 0 |
| `qwen25_7b` | `lmcache_no_native_radix` | 30 | 28.02 | 35.67 | 36.28 | 12.71 | 23552 | 0 | 0 | 0 | 87 | 2902 |
| `qwen25_7b` | `sglang_radix_attention` | 30 | 24.18 | 27.39 | 41.67 | 9.68 | 1055310 | 0 | 0 | 0 | 0 | 0 |
| `qwen25_7b` | `sglang_radix_attention_shadowkv_plus` | 30 | 23.47 | 26.64 | 42.89 | 9.72 | 1055310 | 7680 | 7650 | 30 | 0 | 0 |

Paired deltas vs native Radix:

| Model | Comparison | Paired cells | Mean latency delta | Mean P95 delta | Mean throughput delta | Cached token delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen25_15b` | `lmcache_no_native_radix` vs Radix | 30 | 9.33% | 15.76% | -8.24% | -1031758 |
| `qwen25_15b` | ShadowKV++ vs Radix | 30 | 1.14% | 1.54% | -1.10% | 0 |
| `qwen25_3b` | `lmcache_no_native_radix` vs Radix | 30 | 10.35% | 17.06% | -9.13% | -1031758 |
| `qwen25_3b` | ShadowKV++ vs Radix | 30 | 0.80% | 0.82% | -0.78% | 0 |
| `qwen25_7b` | `lmcache_no_native_radix` vs Radix | 30 | 15.47% | 30.18% | -13.21% | -1031758 |
| `qwen25_7b` | ShadowKV++ vs Radix | 30 | -2.89% | -2.75% | 2.98% | 0 |

Audit notes:

- Error scan hits: `0`.
- Runtime config mismatches: `0`.
- ShadowKV++ vs native Radix cached-token mismatches: `0/90`.
- ShadowKV++ admission pattern was identical in every overlay cell: `(256 plans, 255 allows, 1 bypass, 256 stores)`.
- LMCache no-native-Radix had `27/90` cells with zero retrieve events, concentrated in short-prefix workloads: `ag_news`, `daily_dialog`, and `dolly`.

Interpretation:

The identical-block rerun removes the prior model-major ordering caveat. The headline pattern remains consistent: ShadowKV++ is slightly slower on 1.5B and 3B and faster on 7B, while cached-token totals remain identical to native Radix. This supports the current framing that ShadowKV++ is an admission/policy overlay on top of native Radix/APC behavior, not a replacement cache mechanism. The patched LMCache no-native-Radix path remains valid, but it is slower than native Radix on these public workloads because it gives up fine-grained Radix prefix reuse and only benefits when reuse reaches LMCache's 256-token chunk granularity.

## Qwen14B Identical-Blocks Rerun - 2026-06-18

This run applies the same matched-block method to `Qwen/Qwen2.5-14B-Instruct`, using the same patched runtime image and runtime settings as the three-model run.

Scope:

- Model: `Qwen/Qwen2.5-14B-Instruct`
- Datasets: `daily_dialog`, `samsum`, `ag_news`, `dolly`, `xsum`
- Modes: `templated`, `rag`
- Repetitions: 3
- Requests per cell: 256
- Baselines: `sglang_radix_attention`, `sglang_radix_attention_shadowkv_plus`, `lmcache_no_native_radix`
- Total jobs: 90
- Runtime image: `shadowkv-sglang-lmcache:2026-06-14-no-native-radix`
- Shared runtime settings: context length `4096`, `mem-fraction-static=0.80`, chunked prefill disabled, dtype `float16`, Triton attention backend, PyTorch sampling backend, CUDA graphs disabled.

Result root:

`/home/jade_hand/research/shadowkv/results_sglang_lmcache_shadowkv_qwen14b_identical_blocks_3reps_2026-06-17`

Run script:

`/home/jade_hand/research/shadowkv/session_files/run_shadowkv_sglang_lmcache_qwen14b_identical_blocks_3reps_2026-06-17.sh`

Status:

- Smoke passed before the full run.
- Full matrix completed: `90/90`.
- LMCache log summaries: `30/30`.
- Aggregate CSV/JSON/Markdown artifacts generated.
- Audit artifact generated: `ANOMALY_AUDIT_QWEN14B_IDENTICAL_BLOCKS_2026-06-18.md`.
- `darwin28b-reason-vllm` was restored and health-checked on `127.0.0.1:8015`.

Primary summary:

| Model | Baseline | Cells | Mean latency ms | P95 latency ms | Throughput rps | Idle J/req | Cached tokens | Plans | Allows | Bypasses | LMCache retrieves | LMCache stores |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen25_14b` | `lmcache_no_native_radix` | 30 | 52.31 | 64.11 | 19.36 | 24.75 | 23808 | 0 | 0 | 0 | 88 | 2902 |
| `qwen25_14b` | `sglang_radix_attention` | 30 | 46.30 | 52.57 | 21.75 | 19.39 | 1055310 | 0 | 0 | 0 | 0 | 0 |
| `qwen25_14b` | `sglang_radix_attention_shadowkv_plus` | 30 | 45.49 | 51.47 | 22.13 | 19.36 | 1055310 | 7680 | 7650 | 30 | 0 | 0 |

Paired deltas vs native Radix:

| Model | Comparison | Paired cells | Mean latency delta | Mean P95 delta | Mean throughput delta | Cached token delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen25_14b` | `lmcache_no_native_radix` vs Radix | 30 | 12.70% | 21.81% | -11.18% | -1031502 |
| `qwen25_14b` | ShadowKV++ vs Radix | 30 | -1.73% | -2.07% | 1.76% | 0 |

Audit notes:

- Error scan hits: `0`.
- Runtime config mismatches: `0`.
- Baseline positions were balanced exactly: each baseline appeared in positions 1/2/3 exactly 10 times.
- ShadowKV++ vs native Radix cached-token mismatches: `0/30`.
- ShadowKV++ admission pattern was identical in every overlay cell: `(256 plans, 255 allows, 1 bypass, 256 stores)`.
- LMCache no-native-Radix had `8/30` cells with zero retrieve events, concentrated in `ag_news` plus one `daily_dialog` and one `dolly` cell.

Interpretation:

The Qwen14B matched-block result is the cleanest direct follow-up to the original Qwen14B signal. ShadowKV++ shows a mean latency improvement of `1.73%`, P95 improvement of `2.07%`, and throughput improvement of `1.76%` versus native SGLang Radix under balanced baseline ordering. Cached-token totals still match native Radix exactly, so this remains evidence for ShadowKV++ as an admission/policy overlay rather than a replacement cache. Patched LMCache no-native-Radix remains functional, but it is slower than native Radix on these public workloads because the host-cache path captures far less fine-grained prefix reuse.
