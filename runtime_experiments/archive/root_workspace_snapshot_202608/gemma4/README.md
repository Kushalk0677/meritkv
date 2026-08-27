# Gemma 4 Runtime Experiments

Runtime benchmark results for Gemma 4 models on Blackwell.

## Hardware

NVIDIA RTX PRO 6000 Blackwell.

## Models

| Model | Params | Architecture |
|-------|:------:|:------------:|
| Gemma 4 E2B IT | 2.3B | Gemma-4 MoE |
| Gemma 4 12B IT | 12B | Gemma-4 Dense |
| Gemma 4 26B-A4B IT | 26B | Gemma-4 MoE |
| Gemma 4 31B IT | 31B | Gemma-4 Dense |

## Layout

| Path | Contents |
|---|---|
| `sglang/results.csv` | SGLang comparison: LMCache, RadixAttention, RadixAttention + MeritKV |
| `vllm/results.csv` | vLLM comparison: no-cache, APC, APC + MeritKV |
| `lmcache/results.csv` | LMCache subset for SGLang comparison |
| `summary.md` | Cross-runtime summary |

## Methodology

MeritKV reuse was applied with exact-scaffold matching. Each cell ran 256 requests
at seed 42 with NVML energy measurement, five-second idle baseline, and process-isolated
engine execution. Agreement trends across runtimes were cross-checked against Qwen2.5
runs on the same hardware.

## Notes

Gemma 4's attention mechanism is robust to KV cache perturbations
(ROUGE-L 0.977-0.988 on Blackwell fidelity measurements), enabling
aggressive admission.
