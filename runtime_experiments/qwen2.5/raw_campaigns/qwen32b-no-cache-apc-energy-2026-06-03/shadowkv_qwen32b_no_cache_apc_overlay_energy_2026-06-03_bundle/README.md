# ShadowKV Qwen32B No-Cache APC Overlay Energy Run

Prepared for Kushal.

## Scope

- Host: first-light
- Model: Qwen/Qwen2.5-32B-Instruct
- Baselines: vllm_no_cache, vllm_apc, vllm_apc_shadowkv_plus
- Datasets: daily_dialog, samsum, ag_news, dolly, xsum
- Modes: templated, rag
- Requests: 256 measured requests per full job
- Warmup: 16 requests before measured metrics
- Energy: NVML cumulative GPU energy captured

## Key Files

- results_vllm_qwen32b_no_cache_apc_overlay_energy_2026-06-03/: raw smoke/full benchmark JSONs
- aggregate_qwen32b_no_cache_apc_overlay_energy_full.csv: full matrix aggregate
- comparison_vs_no_cache_qwen32b_full.csv: APC and overlay compared against no-cache
- summary_qwen32b_no_cache_apc_overlay_energy.json: headline summary
- qwen32b_no_cache_apc_overlay_energy_20260603.log: full run log
- qwen32b_no_cache_apc_overlay_energy_20260603.bad_output_dir_*.log: first launch attempt log; stopped early because output_dir was container-local
- run_shadowkv_vllm_qwen32b_no_cache_apc_overlay_energy_2026-06-03.sh: final run script

## Headline

No-cache recorded 0 vLLM cached prompt tokens. APC and APC+ShadowKV++ each recorded 294,208 cached prompt tokens across the full matrix. Mean latency averaged 72.79 ms no-cache, 59.43 ms APC, and 58.95 ms overlay. Total measured GPU energy was 111,318 J no-cache, 83,367 J APC, and 83,646 J overlay.
