# SGLang / ShadowKV++ / LMCache 3-Model 3-Rep Identical-Blocks Summary

Rows: 270

Schedule: 30 matched workload blocks. Each block uses one rep/dataset/mode prompt seed across all three models and all three baselines, with balanced baseline-order and model-order permutations.

## By Model And Baseline

| Model | Baseline | Cells | Mean ms | P95 ms | RPS | Idle J/req | Cached tokens | Plans | Allows | Bypasses | LMCache retrieves | LMCache stores |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen25_15b | lmcache_no_native_radix | 30 | 12.36 | 15.11 | 81.83 | 2.93 | 23552 | 0 | 0 | 0 | 87 | 2902 |
| qwen25_15b | sglang_radix_attention | 30 | 11.28 | 13.05 | 88.99 | 2.23 | 1055310 | 0 | 0 | 0 | 0 | 0 |
| qwen25_15b | sglang_radix_attention_shadowkv_plus | 30 | 11.40 | 13.23 | 88.00 | 2.24 | 1055310 | 7680 | 7650 | 30 | 0 | 0 |
| qwen25_3b | lmcache_no_native_radix | 30 | 16.92 | 20.57 | 59.96 | 5.82 | 23552 | 0 | 0 | 0 | 87 | 2902 |
| qwen25_3b | sglang_radix_attention | 30 | 15.29 | 17.54 | 65.80 | 4.41 | 1055310 | 0 | 0 | 0 | 0 | 0 |
| qwen25_3b | sglang_radix_attention_shadowkv_plus | 30 | 15.41 | 17.69 | 65.29 | 4.42 | 1055310 | 7680 | 7650 | 30 | 0 | 0 |
| qwen25_7b | lmcache_no_native_radix | 30 | 28.02 | 35.67 | 36.28 | 12.71 | 23552 | 0 | 0 | 0 | 87 | 2902 |
| qwen25_7b | sglang_radix_attention | 30 | 24.18 | 27.39 | 41.67 | 9.68 | 1055310 | 0 | 0 | 0 | 0 | 0 |
| qwen25_7b | sglang_radix_attention_shadowkv_plus | 30 | 23.47 | 26.64 | 42.89 | 9.72 | 1055310 | 7680 | 7650 | 30 | 0 | 0 |

## Paired Deltas Vs Native Radix

| Model | Comparison | Paired cells | Mean latency delta % | Mean P95 delta % | Mean throughput delta % | Cached token delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen25_15b | lmcache_no_native_radix_vs_sglang_radix_attention | 30 | 9.33 | 15.76 | -8.24 | -1031758 |
| qwen25_15b | sglang_radix_attention_shadowkv_plus_vs_sglang_radix_attention | 30 | 1.14 | 1.54 | -1.10 | 0 |
| qwen25_3b | lmcache_no_native_radix_vs_sglang_radix_attention | 30 | 10.35 | 17.06 | -9.13 | -1031758 |
| qwen25_3b | sglang_radix_attention_shadowkv_plus_vs_sglang_radix_attention | 30 | 0.80 | 0.82 | -0.78 | 0 |
| qwen25_7b | lmcache_no_native_radix_vs_sglang_radix_attention | 30 | 15.47 | 30.18 | -13.21 | -1031758 |
| qwen25_7b | sglang_radix_attention_shadowkv_plus_vs_sglang_radix_attention | 30 | -2.89 | -2.75 | 2.98 | 0 |
