# Qwen2.5-1.5B Runtime Waste Reviewer-Gap Handoff

Generated from `reviewer_gap_qwen15b_waste_2026-07-03` on first-light.

## Bottom Line

This targeted Blackwell run does show the baseline pathology Kushal wanted to check: both vLLM APC and SGLang RadixAttention have non-zero cache-miss waste and many per-request slowdowns versus paired no-cache references on the AG News pathological workloads. That argues the paper is not only measuring an internal HF-baseline artifact.

However, this run does not support a claim that ShadowKV++ flattened the tail for this exact Qwen2.5-1.5B setup. The ShadowKV++ policy allowed almost every request and bypassed only 4/1/1 requests across semantic AG News, templated AG News, and SAMSum control. Because of that, miss-waste ratios stayed effectively the same as native APC/Radix, and P95 latency was slightly worse for the ShadowKV++ variants in all three cells.

## Scope

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Requests per cell: `256`
- Seed: `20260703`
- KV estimate: `28672` bytes/token (`28.00` KiB/token), fp16
- Workloads: AG News semantic, AG News templated, SAMSum templated control
- Requested engines: vLLM APC, vLLM APC + ShadowKV++, SGLang RadixAttention, SGLang RadixAttention + ShadowKV++
- Paired no-cache references are included only to compute per-request `speedup < 1.0` and failure-waste metrics.

## Instrumentation Definitions

- `p95_latency_ms`: P95 end-to-end request latency measured around planning/client request/feedback, not only server-side HTTP timing.
- `cache_miss_waste_bytes`: `(cache_query_tokens - cache_hit_tokens) * kv_bytes_per_token`; this is the byte proxy for cache entries queried/transferred but not satisfied by a hit.
- `cache_miss_waste_ratio`: `total_cache_miss_waste_bytes / total_cache_query_bytes`.
- `speedup_lt_1_count`: number of paired requests slower than the same request under that runtime's no-cache reference.
- `failure_waste_bytes`: cache-query bytes for requests where paired speedup was below 1.0; this is stricter than miss waste.
- vLLM cache counters came from `/metrics` deltas; SGLang cache counters came from `usage.prompt_tokens_details.cached_tokens`.
- vLLM ShadowKV++ is a write-through policy overlay in this harness; SGLang ShadowKV++ uses the native admission hook to request per-request skip-lookup/skip-write.

## Requested Engine Results

| Workload | Engine | Mean ms | P95 ms | Mean speedup | Speedup<1 | Miss waste | Failure waste | Hit tokens | Bypass | Native skip lookup | Native skip write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pathological_semantic_ag_news | vllm_apc | 14.294 | 16.303 | 1.035 | 100 | 51.5% | 39.1% | 16192 | 0 | 0 | 0 |
| pathological_semantic_ag_news | vllm_apc_shadowkv_plus | 14.979 | 16.755 | 0.988 | 148 | 51.5% | 58.1% | 16192 | 4 | 0 | 0 |
| pathological_semantic_ag_news | sglang_radix_attention | 11.086 | 13.385 | 1.011 | 117 | 46.7% | 45.5% | 17795 | 0 | 0 | 0 |
| pathological_semantic_ag_news | sglang_radix_attention_shadowkv_plus | 11.538 | 13.693 | 0.987 | 149 | 46.9% | 58.5% | 17723 | 4 | 4 | 0 |
| pathological_templated_ag_news | vllm_apc | 14.497 | 16.291 | 1.029 | 97 | 37.4% | 37.7% | 28576 | 0 | 0 | 0 |
| pathological_templated_ag_news | vllm_apc_shadowkv_plus | 14.890 | 16.638 | 1.001 | 131 | 37.4% | 51.3% | 28576 | 1 | 0 | 0 |
| pathological_templated_ag_news | sglang_radix_attention | 10.996 | 13.067 | 1.011 | 108 | 33.4% | 42.5% | 30396 | 0 | 0 | 0 |
| pathological_templated_ag_news | sglang_radix_attention_shadowkv_plus | 12.014 | 15.844 | 0.936 | 180 | 33.4% | 70.6% | 30396 | 1 | 1 | 0 |
| control_templated_samsum | vllm_apc | 15.139 | 17.144 | 1.038 | 87 | 53.6% | 31.0% | 33024 | 0 | 0 | 0 |
| control_templated_samsum | vllm_apc_shadowkv_plus | 15.398 | 17.675 | 1.020 | 107 | 53.6% | 40.5% | 33024 | 1 | 0 | 0 |
| control_templated_samsum | sglang_radix_attention | 11.502 | 13.528 | 1.044 | 92 | 49.1% | 31.0% | 36226 | 0 | 0 | 0 |
| control_templated_samsum | sglang_radix_attention_shadowkv_plus | 11.984 | 14.048 | 1.002 | 122 | 49.1% | 43.7% | 36226 | 1 | 1 | 0 |

## No-Cache Reference Rows

These are not requested engines. They are the paired references used for `speedup_lt_1_count` and failure-waste attribution.

| Workload | Reference | Mean ms | P95 ms | Query MB |
|---|---:|---:|---:|---:|
| pathological_semantic_ag_news | vllm_no_cache_reference | 14.774 | 16.204 | 957.79 |
| pathological_semantic_ag_news | sglang_no_cache_reference | 11.142 | 13.566 | 957.79 |
| pathological_templated_ag_news | vllm_no_cache_reference | 14.850 | 16.511 | 1308.27 |
| pathological_templated_ag_news | sglang_no_cache_reference | 11.023 | 12.861 | 1308.27 |
| control_templated_samsum | vllm_no_cache_reference | 15.667 | 18.287 | 2040.27 |
| control_templated_samsum | sglang_no_cache_reference | 11.951 | 14.236 | 2040.27 |

## Interpretation

- vLLM APC shows reviewer-relevant waste on both pathological workloads: 51.5% miss-waste / 100 slower requests in semantic AG News, and 37.4% miss-waste / 97 slower requests in templated AG News.
- SGLang RadixAttention shows the same general pathology: 46.7% miss-waste / 117 slower requests in semantic AG News, and 33.4% miss-waste / 108 slower requests in templated AG News.
- The SAMSum control still has substantial measured miss-waste in this short-output 1.5B setup, so it should be treated as a control for relative behavior, not as a zero-waste clean-room case.
- ShadowKV++ did not reduce miss-waste in this run because admission mostly allowed requests. vLLM has no native skip in this harness, and SGLang recorded native skip-lookup only on the bypassed requests: 4, 1, and 1.
- If Kushal wants a clean ShadowKV++ advantage claim on this experiment, the next step is not more repeats of this exact policy; it is to adjust or inspect the admission rule so it materially bypasses the short-prefix cache cases being measured.

## Verification

- `summary.csv` and `summary_compact.csv`: 18 data rows each.
- `per_request/`: 18 JSONL files, each with exactly 256 request records.
- first-light production `qwen36-27b-fp8-vllm` was restored after the run and `/v1/models` responded on `127.0.0.1:8014`.
- The run used one warmup request per server, then reset runtime cache before measured requests.

## Environment

- `captured_at=2026-07-03T13:53:02-04:00`
- `hostname=first-light`
- `kernel=Linux first-light 6.12.76 #1-NixOS SMP PREEMPT_DYNAMIC Thu Mar  5 15:04:32 UTC 2026 x86_64 GNU/Linux`
- `cwd=/home/jade_hand/research/shadowkv`
- `vllm_image=shadowkv-shadowkv:latest`
- `sglang_image=shadowkv-sglang-native-admission:2026-06-22-counters`
- `run_id=reviewer_gap_qwen15b_waste_2026-07-03`
- `| NVIDIA-SMI 580.119.02             Driver Version: 580.119.02     CUDA Version: 13.0     |`
- `|   0  NVIDIA RTX PRO 6000 Blac...    Off |   00000000:01:00.0  On |                  Off |`
- vLLM image runtime: `vllm==0.19.2rc1.dev107+g4eafc7292.cu130`; `sglang=unavailable (No package metadata was found for sglang)`; `torch==2.11.0+cu130`; `transformers==5.5.4`; `datasets==4.8.5`; `numpy==2.2.6`; `flashinfer-python==0.6.8.post1`
- SGLang image runtime: `vllm==0.19.2rc1.dev107+g4eafc7292.cu130`; `sglang==0.5.13`; `torch==2.11.0+cu130`; `transformers==5.8.1`; `datasets==4.8.5`; `numpy==2.2.6`; `flashinfer-python==0.6.12`

## Files To Review

- `summary_compact.csv`: reviewer-facing summary with P95, waste ratios, and slowdown counts.
- `summary.csv`: full run-level metrics.
- `requested_engine_results.csv`: only the four requested engines, excluding no-cache references.
- `per_request/<workload>/<engine>.jsonl`: request-level latency/cache/waste/speedup records.
- `metadata/`: host, GPU, Docker image, package-version, and git-status snapshots.
- `session_files/reviewer_gap_qwen15b_waste_2026_07_03.py`: benchmark harness used for this run, included in the zip bundle.
