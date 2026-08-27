# Qwen2.5 Cross-Runtime Summary

These are the measured Blackwell aggregates reported in the current paper. No
model-size point is interpolated. Runtime cache behavior is write-through, so
the overlay rows establish compatibility and observed decision-layer overhead,
not MeritKV-caused production acceleration.

## SGLang and LMCache

| Model | LMCache latency (ms) | Native Radix latency (ms) | Radix+MeritKV latency (ms) | Overlay/native |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 12.9 | 11.7 | 11.9 | 1.013 |
| Qwen2.5-3B | 17.3 | 15.6 | 15.7 | 1.006 |
| Qwen2.5-7B | 28.6 | 24.6 | 23.8 | 0.976 |
| Qwen2.5-14B | 53.4 | 47.0 | 46.2 | 0.981 |
| Qwen2.5-32B | 102.1 | 101.7 | 96.9 | 0.953 |

The like-for-like comparison is native RadixAttention versus
RadixAttention+MeritKV. LMCache without Radix caches only one to two tokens per
request here and is contextual rather than the principal baseline.

## vLLM APC Scale Ratios

| Model | APC+MeritKV/APC latency |
|---|---:|
| Qwen2.5-1.5B | 0.985 |
| Qwen2.5-3B | 0.990 |
| Qwen2.5-7B | 0.975 |
| Qwen2.5-14B | 0.980 |
| Qwen2.5-32B | 0.993 |

## vLLM Qwen2.5-32B Five-Replicate Campaign

| Engine | Mean latency (ms) | P95 latency (ms) | Idle-adjusted J/request |
|---|---:|---:|---:|
| No cache | 72.84 | 89.96 | 36.32 |
| APC | 59.37 | 73.62 | 26.63 |
| APC+MeritKV | 59.62 | 73.93 | 26.59 |

The paper's APC+MeritKV/APC changes, `+0.44%` mean latency, `+0.59%` P95,
and `-0.22%` energy, are means of paired per-replicate ratios. They therefore
need not equal ratios of the rounded arithmetic means.

Selected raw 32B campaign records and the separate k-star prefix profile remain
under the corresponding `raw/` and `kstar/` folders. Raw summaries preserve
campaign-level observations and do not override this paper-facing aggregation.
