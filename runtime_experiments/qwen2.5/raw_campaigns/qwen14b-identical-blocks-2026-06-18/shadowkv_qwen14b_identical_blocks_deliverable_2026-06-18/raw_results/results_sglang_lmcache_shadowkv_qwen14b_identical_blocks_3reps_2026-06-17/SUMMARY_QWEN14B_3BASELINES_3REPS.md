# SGLang / ShadowKV++ / LMCache Qwen14B 3-Rep Identical-Blocks Summary

Rows: 90

Schedule: 30 matched workload blocks. Each block uses one rep/dataset/mode prompt seed across all three baselines, with balanced baseline-order permutations.

## By Model And Baseline

| Model | Baseline | Cells | Mean ms | P95 ms | RPS | Idle J/req | Cached tokens | Plans | Allows | Bypasses | LMCache retrieves | LMCache stores |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen25_14b | lmcache_no_native_radix | 30 | 52.31 | 64.11 | 19.36 | 24.75 | 23808 | 0 | 0 | 0 | 88 | 2902 |
| qwen25_14b | sglang_radix_attention | 30 | 46.30 | 52.57 | 21.75 | 19.39 | 1055310 | 0 | 0 | 0 | 0 | 0 |
| qwen25_14b | sglang_radix_attention_shadowkv_plus | 30 | 45.49 | 51.47 | 22.13 | 19.36 | 1055310 | 7680 | 7650 | 30 | 0 | 0 |

## Paired Deltas Vs Native Radix

| Model | Comparison | Paired cells | Mean latency delta % | Mean P95 delta % | Mean throughput delta % | Cached token delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen25_14b | lmcache_no_native_radix_vs_sglang_radix_attention | 30 | 12.70 | 21.81 | -11.18 | -1031502 |
| qwen25_14b | sglang_radix_attention_shadowkv_plus_vs_sglang_radix_attention | 30 | -1.73 | -2.07 | 1.76 | 0 |
