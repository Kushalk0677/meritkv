# Gemma 4 Blackwell Runtime Experiments

This folder contains Gemma 4 Blackwell runtime benchmark results
with no-cache, native runtime, and MeritKV admission-policy arms across
5 seeds for each cell.

## Contents

| Path | Contents |
|------|----------|
| `vllm/results.csv` | vLLM APC, APC + MeritKV, and no-cache. |
| `sglang/results.csv` | SGLang RadixAttention, +MeritKV, and no-cache. |
| `lmcache/results.csv` | LMCache + vLLM, +MeritKV, and no-cache. |
| `vllm/raw/` | Per-cell benchmark JSONs, aggregate, logs, and admission reports (all 5 seeds). |
| `sglang/raw/` | Per-cell benchmark JSONs, aggregate, logs, and admission reports (all 5 seeds). |
| `lmcache/raw/` | Per-cell benchmark JSONs, aggregate, logs, and admission reports (all 5 seeds). |

## Models

E2B (2.3B), E4B (4B), 12B, 26B-A4B, 31B

## Datasets

ag_news, daily_dialog, dolly, samsum, xsum

## Modes

rag, templated

## Seeds

42, 123, 456, 789, 999
