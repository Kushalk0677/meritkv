# Gemma 4 Blackwell Runtime Experiments

This folder contains the measured Gemma-4 Blackwell write-through runtime
campaign and the paper-facing overlay aggregates across five seeds.

## Contents

| Path | Contents |
|------|----------|
| `vllm/results.csv` | MeritKV-vs-APC mean latency differences. |
| `sglang/results.csv` | MeritKV-vs-RadixAttention mean latency differences. |
| `lmcache/results.csv` | MeritKV-vs-LMCache mean latency differences. |
| `*/raw/legacy_detailed_results_precompact.csv` | Preserved former detailed top-level export. |
| `vllm/raw/full/rep_*` | Per-cell benchmark JSONs for all five seeds. |
| `sglang/raw/full/rep_*` | Per-cell benchmark JSONs for all five seeds. |
| `lmcache/raw/full/rep_*` | Per-cell benchmark JSONs for all five seeds. |
| `run_metadata.json` | Hardware, model, dataset, seed, and measurement protocol metadata. |
| `summary.md` | Cross-runtime summary and coverage notes. |

## Models

| Paper label | Checkpoint |
|---|---|
| E2B | `google/gemma-4-E2B-it` |
| E4B | `google/gemma-4-E4B-it` |
| 12B | `google/gemma-4-12B-it` |
| 26B-A4B | `google/gemma-4-26B-A4B-it` |
| 31B | `google/gemma-4-31B-it` |

E4B is the 4B checkpoint used only in this runtime-overlay study, not an
additional member of the twelve-model long-prefix scale study.

## Datasets

`ag_news`, `daily_dialog`, `dolly`, `samsum`, and `xsum`.

## Modes

`rag` and `templated`.

## Seeds

`42`, `123`, `456`, `789`, and `999`.

Each paper-facing row summarizes 50 matched cells. Raw no-cache, native-cache,
and overlay cell records remain under the runtime-specific `raw/` directories.
The overlay is write-through, so its positive deltas measure overhead rather
than enforced MeritKV acceleration.
