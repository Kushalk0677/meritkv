# Runtime Experiment Methodology

This document describes the methodology used for the real-world runtime
experiments in `runtime_experiments/`.

## Hardware

- **GPU**: NVIDIA RTX PRO 6000 Blackwell
- **CPU**: Intel Xeon Gold 5418Y
- **RAM**: 128 GB
- **OS**: Linux (Ubuntu 22.04)

## Software

| System | Version |
|--------|---------|
| SGLang | v0.4.0.post2 |
| LMCache | v0.2.1 |
| vLLM | v0.6.0 |
| Python | 3.10 |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |

## Models

All experiments use the Qwen2.5 model family:

| Model | Parameters |
|-------|-----------|
| Qwen2.5-1.5B-Instruct | 1.54B |
| Qwen2.5-3B-Instruct | 3.09B |
| Qwen2.5-7B-Instruct | 7.61B |
| Qwen2.5-14B-Instruct | 14.7B |
| Qwen2.5-32B-Instruct | 32.5B |

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

All SGLang results use **3 independent replicates** per cell. Reported
values are means across replicates with 95% CIs.

### vLLM experiments

Three engines across the Qwen2.5 family:

1. **`vllm_no_cache`**: vLLM with prefix caching disabled.
2. **`vllm_apc`**: vLLM with Automatic Prefix Caching (hash-based).
3. **`vllm_apc_shadowkv_plus`**: MeritKV overlay on vLLM APC.

All reported vLLM table rows are direct timing measurements. Replicate counts are
recorded in the corresponding campaign artifacts; GPU energy is measured via NVML
where available.

## Metrics

- **Mean latency**: Average request latency in milliseconds.
- **Throughput**: Requests per second.
- **Cached tokens**: Mean number of KV cache tokens reused per request.
- **GPU energy**: Total GPU energy consumption in Joules (NVML).
- **Speedup**: `(baseline_latency / engine_latency - 1) * 100`.

## Data Sources

The checked-in runtime CSVs are the curated, direct-measurement tables. Selected
raw benchmark JSON files and run logs are included under `runtime_experiments/`,
principally for the Qwen2.5-32B campaigns; they do not constitute a per-size raw
release for every curated table row. Each included raw bundle contains:

- Individual benchmark JSONs (one per engine/dataset/mode)
- Standard output and error logs
- Run shell scripts for reproducibility
- Hardware metadata (nvidia-smi, pip freeze, Docker image)

## Limitations

- SGLang + MeritKV at 32B is an actual timed measurement, not a value derived from lower-size SGLang ratio trends.
- All vLLM rows (1.5B, 3B, 7B, 14B, and 32B) are direct timed measurements; none are scaled or interpolated from anchors.
- Results on other model families (GPT-2, TinyLlama, Gemma, Phi-3)
  and other GPU types are not yet available.
