# Gemma 4 Blackwell Runtime Results

Measured runtime baselines on RTX PRO 6000 Blackwell.

## Hardware

NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM).

## Models

| Model | Params |
|-------|:-----:|
| Gemma 4 E2B | 2.3B |
| Gemma 4 E4B | 4B |
| Gemma 4 12B | 12B |
| Gemma 4 26B-A4B | 26B |
| Gemma 4 31B | 31B |

## Layout

| Path | Contents |
|------|----------|
| `vllm/results.csv` | vLLM APC, APC + MeritKV, and no-cache results (5 seeds). |
| `sglang/results.csv` | SGLang RadixAttention, +MeritKV, and no-cache results (5 seeds). |
| `lmcache/results.csv` | LMCache + vLLM, +MeritKV, and no-cache results (5 seeds). |
| `vllm/raw/` | Per-cell benchmark JSONs, aggregate, logs, and admission reports (all 5 seeds). |
| `sglang/raw/` | Per-cell benchmark JSONs, aggregate, logs, and admission reports (all 5 seeds). |
| `lmcache/raw/` | Per-cell benchmark JSONs, aggregate, logs, and admission reports (all 5 seeds). |

## Coverage

| Runtime | Models | Datasets | Modes | Engines | Seeds | Rows | Raw JSONs |
|---------|:-----:|:--------:|:----:|:-------:|:----:|:----:|:---------:|
| vLLM | 5 | 5 | 2 | 3 | 5 | 750 | 2250 |
| SGLang | 5 | 5 | 2 | 3 | 5 | 750 | 2250 |
| LMCache | 5 | 5 | 2 | 3 | 5 | 750 | 2250 |

## Notes

- 256 requests per cell, temperature 0, one output token.
- NVML energy measurements for all cells.
- Zero request failures.
- Speedup percentage is relative to the no-cache baseline for each (model, dataset, mode, seed) cell.
