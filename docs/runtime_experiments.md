# Runtime Experiment Methodology

This document describes the methodology used for the real-world runtime
experiments in `runtime_experiments/`.

## Hardware

- **GPU**: NVIDIA RTX PRO 6000 Blackwell (96 GB)
- **CUDA / driver**: CUDA 13.0 / 580.119.02

## Software

| System | Version |
|--------|---------|
| vLLM | `0.19.2rc1.dev107+cu130` |
| SGLang | `0.5.13` |
| PyTorch | `2.11.0+cu130` |
| transformers | `5.5.4` |
| datasets | `4.8.5` |
| NumPy | `2.2.6` |
| FlashInfer | `0.6.8` |

## Qwen2.5 Models

The Qwen runtime campaign uses:

| Paper label | Public checkpoint |
|---|---|
| 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| 3B | `Qwen/Qwen2.5-3B-Instruct` |
| 7B | `Qwen/Qwen2.5-7B-Instruct` |
| 14B | `Qwen/Qwen2.5-14B-Instruct` |
| 32B | `Qwen/Qwen2.5-32B-Instruct` |

## Datasets

Ten datasets from the HuggingFace hub, covering four task types:

| Task Type | Datasets |
|-----------|----------|
| Classification | AG News, Banking77 |
| Instruction | AlpacaEval, Dolly |
| Conversational | DailyDialog, OASST1, UltraChat |
| Summarisation | SAMSum, XSum, CNN/DailyMail |

Two prompt modes: `templated` (shared serving scaffold) and `rag`
(retrieval-augmented generation scaffold). Each with 256 requests per job.

## Engines

### SGLang experiments

Three engines, compared pairwise on identical request sequences:

1. **`sglang_radix_attention`**: Native SGLang RadixAttention prefix caching.
2. **`sglang_radix_attention_shadowkv_plus`**: MeritKV policy controller
   as an overlay on SGLang's RadixAttention.
3. **`lmcache_no_native_radix`**: LMCache without native RadixAttention
   integration (baseline without prefix-tree matching).

The 1.5B--14B SGLang rows use three measured replicates. The 32B
SGLang/Radix, LMCache, and SGLang+MeritKV rows are also measured; the 32B
SGLang+MeritKV point is not derived from the smaller-model trend.

### Native SGLang balanced-admission experiment

The separate Qwen2.5-1.5B balanced-admission experiment uses a native SGLang
per-request hook rather than the write-through overlay used in the runtime scale
study. It uses the `balanced` preset with `util_min_ms=2.0`,
`min_bootstrap_admissions=0`, 256 requests, seed 20260703, and the pathological
short-prefix semantic AG News workload.

Native RadixAttention attempted 957.788160 MB of cache-query traffic with mean
latency 10.895 ms. MeritKV enforced 254 skip-lookups, reducing attempted
cache-query traffic to 6.766592 MB (99.3%) while increasing mean latency to
20.233 ms. This cell supports enforced runtime waste avoidance, not latency
improvement. The complete records are in
`runtime_experiments/qwen2.5/sglang/balanced_admission/`.

The experiment used a separate native-admission SGLang image. Its exact
container, Python-package, GPU, driver, and CUDA metadata are retained inside
the bundle; those image-specific versions should be used instead of assuming
that every entry in the broad runtime software table applies to this cell.

### vLLM experiments

Three engines across the Qwen2.5 family:

1. **`vllm_no_cache`**: vLLM with prefix caching disabled.
2. **`vllm_apc`**: vLLM with Automatic Prefix Caching (hash-based).
3. **`vllm_apc_shadowkv_plus`**: MeritKV overlay on vLLM APC.

All reported vLLM table rows are direct timing measurements. The separate
7B/32B latency-and-energy campaign uses five replicates; the cross-runtime scale
campaign supplies measured points across 1.5B--32B. GPU energy is measured via
NVML where available.

### Gemma-4 overlay experiments

The separate Gemma-4 runtime-overlay study uses E2B, E4B, 12B, 26B-A4B, and
31B across five datasets (`ag_news`, `daily_dialog`, `dolly`, `samsum`, and
`xsum`), two modes (`rag` and `templated`), and five seeds. E4B is the
`google/gemma-4-E4B-it` checkpoint used only in this overlay study; it is not
an extra member of the twelve-model long-prefix study.

## Metrics

- **Mean latency**: Average request latency in milliseconds.
- **Throughput**: Requests per second.
- **Cached tokens**: Mean number of KV cache tokens reused per request.
- **GPU energy**: Total GPU energy consumption in Joules (NVML).
- **Speedup**: `(baseline_latency / engine_latency - 1) * 100`.

## Data Sources

The checked-in top-level runtime CSVs are compact, direct-measurement aggregate
tables. Selected raw benchmark JSON files and run records are included,
principally for the Qwen2.5-32B and Gemma-4 campaigns; they do not constitute a
uniform per-size raw release for every paper row. Depending on the campaign,
raw bundles may contain:

- Individual benchmark JSONs (one per engine/dataset/mode)
- Standard output and error logs
- Run shell scripts for reproducibility
- Hardware metadata (nvidia-smi, pip freeze, Docker image)

The balanced-admission bundle is a complete release rather than a selected raw
subset: it contains 18 engine/workload result cells, 18 per-request JSONL files
with 256 records each, three 256-request input traces, aggregate CSV/JSON
summaries, run and server logs, commands, experiment configuration, environment
metadata, and the captured source subset used by the harness.

## Limitations

- SGLang + MeritKV at 32B is an actual timed measurement, not a value derived from lower-size SGLang ratio trends.
- All vLLM rows (1.5B, 3B, 7B, 14B, and 32B) are direct timed measurements; none are scaled or interpolated from anchors.
- The Gemma-4 overlay family is reported separately. Runtime results do not
  establish behavior for untested model/backend combinations.
