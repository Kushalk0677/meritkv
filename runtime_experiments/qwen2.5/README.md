# Qwen2.5 Runtime Experiments

This folder contains measured Qwen2.5 runtime aggregates and selected raw
Blackwell campaign records. The current paper-facing aggregation is in
`summary.md`.

## Hardware

NVIDIA RTX PRO 6000 Blackwell.

## Models

Qwen2.5 1.5B, 3B, 7B, 14B, and 32B appear in the curated tables. The included vLLM and SGLang run files are for Qwen2.5-32B. The k-star prefix profile includes Qwen2.5-1.5B, Qwen2.5-7B, and Qwen2.5-32B.

## Layout

| Path | Contents |
|---|---|
| `sglang/results.csv` | Paper-aligned SGLang/Radix and contextual LMCache aggregates. |
| `sglang/raw/q32b_full_20260608/` | Full SGLang/LMCache Qwen2.5-32B run files. |
| `sglang/balanced_admission/` | Complete Qwen2.5-1.5B native SGLang balanced-admission bundle, including request records, logs, commands, configuration, and environment metadata. |
| `vllm/results.csv` | Measured scale ratios and the 32B five-replicate aggregate. |
| `vllm/raw/q32b_5rep_20260701/` | 5-replicate vLLM Qwen2.5-32B aggregate and available benchmark JSONs. |
| `vllm/raw/q32b_20260603/` | Earlier full vLLM Qwen2.5-32B run files. |
| `lmcache/results.csv` | LMCache aggregate rows used in the SGLang comparison. |
| `kstar/raw/prefix_profile_20260701/` | Primary k-star prefix-length profile. |
| `kstar/raw/response_usage_probe_20260701/` | Response-usage probe for the k-star run. |
| `kstar/raw/run_logs_20260701/` | Logs for the k-star and vLLM runtime runs. |
| `summary.md` | Cross-runtime Qwen2.5 summary. |

## Notes

- The top-level CSVs are compact paper-aligned measured aggregates, not
  reconstructed per-dataset or per-seed tables.
- Every reported table row is a direct measurement. The checked-in raw bundles
  are selected audit artifacts, with the most complete coverage for Qwen2.5-32B.
- The production integration is write-through. Overlay differences are
  compatibility/overhead observations, not MeritKV-caused acceleration.
- The balanced-admission folder is a separately scoped native SGLang
  enforcement experiment. It demonstrates lookup/waste avoidance with an
  explicit latency cost and does not change the interpretation of the broad
  write-through scale study.
- The `raw/` subfolders keep the run files needed to audit or regenerate selected values without cluttering the top-level runtime folders.
- The July 1 vLLM aggregate contains all five replicates. The June 3 vLLM run is retained as an additional full run tree.

