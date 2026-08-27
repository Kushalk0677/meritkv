# Cross-Runtime Summary

## SGLang: Latency and Energy (vs LMCache baseline)

Mean across all 10 datasets, both modes:

| Model | Radix Latency | Radix+SKVPP Latency | SKVPP Energy vs LMCache |
|---|---:|---:|---:|
| Gemma 4 E2B (2.3B) | 55.6ms | **42.7ms (-24%)** | -34% |
| Gemma 4 12B | 67.8ms | **64.2ms (-6%)** | -32% |
| Gemma 4 26B-A4B | 96.0ms | **88.4ms (-8%)** | -28% |
| Gemma 4 31B | 77.6ms | **73.0ms (-6%)** | -43% |

## vLLM: Latency and Energy (vs No-Cache)

Mean across all 10 datasets, both modes:

| Model | APC Latency | APC+SKVPP Latency | SKVPP Energy vs No-Cache |
|---|---:|---:|---:|
| Gemma 4 E2B (2.3B) | 27.5ms | **23.5ms (-14%)** | -42% |
| Gemma 4 12B | 41.7ms | **37.1ms (-11%)** | -40% |
| Gemma 4 26B-A4B | 55.0ms | **49.4ms (-10%)** | -36% |
| Gemma 4 31B | 48.4ms | **42.4ms (-13%)** | -49% |

