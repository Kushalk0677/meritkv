# Gemma 4 31B HF Blackwell All-Engine Report

> **Auxiliary extension:** This fixed-order, single-seed campaign is not a
> paper-facing engine ranking. It is retained for diagnostic completeness.

Date: 2026-07-15  
Operator: Keystone (Codex on IronGod)  
Execution host: first-light  
GPU: NVIDIA RTX PRO 6000 Blackwell  
Model: `google/gemma-4-31B-it`

## Executive result

The requested all-engine HF sweep completed successfully: 11/11 smoke cells and 110/110 full cells, with zero failed jobs and no external GPU-workload abort. Production `qwen36-27b-fp8-vllm` was restored and verified through `/v1/models` after the sweep.

All four ShadowKV++ variants reduced aggregate mean latency by about 31%, aggregate P95 latency by about 37%, and total measured GPU energy by about 33% versus no-cache. Each variant won all 10 dataset cells and executed 1,270/1,280 prefix reuses, with one cold request per dataset.

This is a strong single-sweep signal, not a final significance claim. The run used one seed and a fixed engine order. The four ShadowKV++ variants are separated by less than 1% aggregate mean latency, so their internal ranking should be treated as noise until randomized repetitions are run.

## Configuration

- Datasets: AG News, AlpacaEval, Banking77, CNN/DailyMail, DailyDialog, Dolly, OASST1, SAMSum, UltraChat, and XSum.
- Semantic mode with a common 128-token scaffold repeated four times.
- 128 requests per cell, seed 42, CUDA float16.
- One isolated Python process per model/dataset/engine cell.
- Balanced ShadowKV policy preset.
- NVML energy measurement with a five-second idle baseline.
- Policy traces enabled.
- Experimental engines explicitly enabled with `--include_experimental`.

## Aggregate results

Positive latency and energy percentages mean improvement versus no-cache.

| Engine | Mean ms | P95 ms | Throughput rps | Mean latency | P95 latency | Throughput | GPU energy | Mean wins | Reuse successes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_cache` | 179.34 | 199.20 | 4.353 | baseline | baseline | baseline | baseline | 0/10 | 0 |
| `native_prefix_cache` | 180.22 | 199.74 | 4.326 | -0.50% | -0.27% | -0.61% | -0.50% | 3/10 | 0 |
| `reactive_prefix_cache` | 154.37 | 173.80 | 4.879 | +13.92% | +12.75% | +12.09% | +14.85% | 10/10 | 1,270 |
| `greedy_prefix_cache` | 156.26 | 175.23 | 4.810 | +12.87% | +12.03% | +10.51% | +14.46% | 10/10 | 1,270 |
| `strict_reactive_prefix_cache` | 155.61 | 174.87 | 4.850 | +13.23% | +12.21% | +11.42% | +14.42% | 10/10 | 1,270 |
| `frequency_speculative` | 180.59 | 200.32 | 4.328 | -0.70% | -0.56% | -0.56% | -0.76% | 4/10 | 0 |
| `shadow_kv` | 181.33 | 201.57 | 4.378 | -1.11% | -1.19% | +0.59% | -1.07% | 1/10 | 0 |
| `shadow_kv_plus` | 123.08 | 125.49 | 5.799 | +31.37% | +37.00% | +33.24% | +33.21% | 10/10 | 1,270 |
| `shadow_kv_plus_lite` | 123.26 | 125.03 | 5.823 | +31.27% | +37.23% | +33.78% | +33.30% | 10/10 | 1,270 |
| `shadow_kv_plus_best_latency` | 123.98 | 125.95 | 5.790 | +30.87% | +36.77% | +33.03% | +33.14% | 10/10 | 1,270 |
| `shadow_kv_plus_raw_observer` | 123.30 | 125.98 | 5.800 | +31.25% | +36.75% | +33.25% | +33.36% | 10/10 | 1,270 |

## Reuse audit

- Reactive, greedy, strict-reactive, and all four ShadowKV++ variants recorded 127/128 successful prefix reuses in every dataset cell.
- Each of those engines reused 162,560 prefix tokens across the ten datasets.
- Full `shadow_kv_plus`, `shadow_kv_plus_best_latency`, and `shadow_kv_plus_raw_observer` cells were classified as `exact_scaffold_only`.
- All aggregate rows reported zero wasted-compute ratio.
- `shadow_kv` recorded no reuse under the balanced policy and was slightly slower than no-cache.

## Important native-cache caveat

`native_prefix_cache` is not a real vLLM APC or SGLang Radix result in this HF harness. Source inspection shows that it stores placeholder match metadata, records apparent prefix hits, and then calls full prefill for every request. It recorded a mean hit rate of 0.700, zero reuse successes, and 787 bypassed matches. Its latency should therefore be read as an observational placeholder, not as native-runtime cache performance.

The reactive and ShadowKV++ engines use the HF backend's external KV path and do execute real prefix reuse. Also, `reuse_path_breakdown.csv` labels Lite and reactive variants as `no_reuse_path_executed` because that classifier only recognizes the full ShadowKV++ policy counters. Their 1,270 `reuse_successes`, 0.992 hit rate, and reused-token totals are the applicable execution evidence.

## Interpretation

Gemma 4 31B is comfortably above the HF external-KV break-even point. Standard reactive reuse saves about 13% mean latency, while the ShadowKV++ fast scaffold path saves about 31%. The additional gain is consistent across all ten datasets and is accompanied by a similar energy reduction.

The experiment does not establish that one ShadowKV++ variant is better than another. `shadow_kv_plus`, Lite, best-latency, and raw-observer are effectively tied in this one fixed-order pass. A publishable ranking needs at least three randomized repetitions with identical blocks.

For a true native-runtime comparison, Gemma 4 31B must be run separately under an actual runtime exposing APC/Radix cache behavior. The `native_prefix_cache` row in this package cannot substitute for that experiment.

## Validation and artifacts

- Outer Blackwell runner patched to propagate `--include_experimental` to each isolated cell.
- Focused and regression tests: 22 passed, 1 opt-in slow HF correctness test skipped.
- Full job ledger: 110 entries, zero nonzero return codes.
- Smoke job ledger: 11 entries, zero nonzero return codes.
- Source hashes, source snapshot, runtime package versions, Docker image metadata, GPU metadata, policy traces, raw benchmark JSON, and logs are included.
- Production model after run: `Qwen/Qwen3.6-27B-FP8` on `127.0.0.1:8014`.
