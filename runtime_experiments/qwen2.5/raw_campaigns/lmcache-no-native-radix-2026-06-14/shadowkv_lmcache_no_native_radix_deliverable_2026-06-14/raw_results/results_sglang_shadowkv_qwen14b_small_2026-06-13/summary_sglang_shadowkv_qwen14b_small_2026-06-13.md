# SGLang ShadowKV Qwen2.5-14B Small Run Summary - 2026-06-13

Result root: `/home/jade_hand/research/shadowkv/results_sglang_shadowkv_qwen14b_small_2026-06-13`

## Aggregate Averages

| Baseline | Mean latency ms | P95 latency ms | Throughput rps | Cached tokens total | Idle-adjusted J/request | Admission plans | Allows | Bypasses | Stores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sglang_radix_attention | 45.25 | 50.12 | 22.25 | 86542 | 18.69 | 0 | 0 | 0 | 0 |
| sglang_radix_attention_shadowkv_plus | 44.44 | 48.55 | 22.64 | 86542 | 18.70 | 640 | 630 | 10 | 640 |

## Per-Cell ShadowKV++ vs SGLang

| Dataset | Mode | Latency delta | P95 delta | Throughput delta | Energy delta | Cached token delta | Plans | Allows | Bypasses | Stores |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ag_news | templated | -1.40% | -1.47% | 1.42% | 0.52% | 0 | 64 | 63 | 1 | 64 |
| ag_news | rag | -1.59% | -1.53% | 1.62% | -3.86% | 0 | 64 | 63 | 1 | 64 |
| daily_dialog | templated | -1.39% | -2.83% | 1.41% | 0.94% | 0 | 64 | 63 | 1 | 64 |
| daily_dialog | rag | -1.45% | -1.82% | 1.47% | 0.06% | 0 | 64 | 63 | 1 | 64 |
| dolly | templated | -1.58% | -3.67% | 1.61% | -1.54% | 0 | 64 | 63 | 1 | 64 |
| dolly | rag | -1.82% | -2.21% | 1.85% | 0.32% | 0 | 64 | 63 | 1 | 64 |
| samsum | templated | -1.82% | -1.75% | 1.85% | -3.10% | 0 | 64 | 63 | 1 | 64 |
| samsum | rag | -1.87% | -1.83% | 1.90% | 4.96% | 0 | 64 | 63 | 1 | 64 |
| xsum | templated | -2.09% | -3.15% | 2.13% | 0.00% | 0 | 64 | 63 | 1 | 64 |
| xsum | rag | -2.52% | -9.38% | 2.59% | 1.66% | 0 | 64 | 63 | 1 | 64 |
