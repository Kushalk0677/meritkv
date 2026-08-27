# ShadowKV++ Long-Prefix HF Results: Qwen32B and Gemma 4

This package combines three clean Blackwell model sweeps using the same patched Hugging Face runtime, workload matrix, and measurement protocol.

## Combined result

| Model | ShadowKV++ mean speedup | P95 speedup | GPU energy reduction | Mean speedup range | P95 speedup range |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-32B-Instruct | 1.109x | 1.094x | 11.1% | 1.057–1.135x | 1.002–1.160x |
| Gemma 4 31B IT | 1.561x | 1.720x | 30.4% | 1.139–2.703x | 1.145–2.958x |
| Gemma 4 26B-A4B IT | 1.204x | 1.233x | 20.6% | 1.043–1.595x | 1.049–1.591x |

All three models were above no-cache parity for both mean and P95 latency in every dataset. Every ShadowKV++ cell recorded 127 of 128 exact-scaffold reuse successes, a 99.21875% hit rate, 16,256 reused prefix tokens, zero wasted-compute ratio, and zero backend fallbacks.

## Shared protocol

- Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM
- Runtime: `shadowkv-hf-blackwell:20260706`
- PyTorch/Transformers: 2.11.0+cu130 / 5.10.2
- Datasets: AG News, AlpacaEval, Banking77, CNN/DailyMail, DailyDialog, Dolly, OASST1, SAMSum, UltraChat, XSum
- Engines: `no_cache`, `shadow_kv`, `shadow_kv_plus`
- Requests: 128 per cell
- Seed: 42
- Workload: semantic mode with common long scaffold repeated four times
- Prefix reused: 128 tokens
- Energy and per-request policy tracing: enabled

The Qwen32 sweep contains 30 successful cells. The two-model Gemma sweep contains 60 successful cells. Both clean result roots have zero failed jobs and no external GPU-workload abort marker.

## Runtime patch

Large FP16 models exposed an avoidable second model load in policy calibration. The harness previously loaded the measured backend and then loaded another full model for unmeasured calibration probes. The patch profiles costs on the already-loaded backend and retains only the resulting calibration dictionary. A regression test verifies that calibration does not invoke the model loader. The full patched test suite passed `93 passed, 1 skipped`.

## Interpretation

The cross-model trend is consistent with exact prefix reuse becoming more valuable as avoided prefill work becomes more expensive. Gemma 4 31B shows the largest effect, followed by Gemma 4 26B-A4B and Qwen2.5-32B. The dense Gemma model's advantage over A4B is consistent with the MoE model activating only a subset of experts per token.

The dataset-level range is wide, particularly for Gemma. A fixed 128-token prefix removes a different fraction of each workload's total prefill, so this is plausible. However, each cell has one seed and one execution, and engines ran in fixed order. Final claims require at least three repetitions with randomized engine order and paired confidence intervals.

These runs validate exact scaffold reuse, not approximate semantic-partial KV reuse. All successful reuse paths were `exact_scaffold_only`; no semantic-partial path executed.

Production `qwen36-27b-fp8-vllm` was restored after each sweep and was verified through `/v1/models`.

