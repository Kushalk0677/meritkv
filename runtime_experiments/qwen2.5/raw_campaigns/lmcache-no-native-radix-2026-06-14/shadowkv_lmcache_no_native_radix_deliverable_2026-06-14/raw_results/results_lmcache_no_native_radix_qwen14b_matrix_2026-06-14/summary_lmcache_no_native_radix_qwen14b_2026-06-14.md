# LMCache No Native Radix Qwen14B Summary - 2026-06-14

Cells: 10
Mean latency ms: 51.56
P95 latency ms: 62.97
Throughput rps: 19.61
Idle-adjusted J/request: 23.74
Prompt tokens total: 167304
Cached tokens total: 7168
LMCache retrieve events: 27
LMCache store events: 242

| Dataset | Mode | Mean ms | P95 ms | RPS | Idle J/req | Prompt tokens | Cached tokens | Retrieve events | Store events |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ag_news | rag | 46.44 | 49.77 | 21.53 | 21.24 | 11857 | 0 | 0 | 0 |
| ag_news | templated | 46.15 | 47.90 | 21.67 | 20.59 | 11153 | 0 | 0 | 0 |
| daily_dialog | rag | 48.51 | 60.57 | 20.61 | 22.55 | 14003 | 256 | 1 | 12 |
| daily_dialog | templated | 47.38 | 59.27 | 21.10 | 21.28 | 13043 | 256 | 1 | 7 |
| dolly | rag | 48.77 | 64.20 | 20.50 | 22.22 | 13461 | 0 | 0 | 15 |
| dolly | templated | 48.33 | 64.27 | 20.69 | 22.58 | 12309 | 0 | 0 | 13 |
| samsum | rag | 53.09 | 67.69 | 18.84 | 24.35 | 18747 | 1280 | 5 | 38 |
| samsum | templated | 52.59 | 68.59 | 19.02 | 23.78 | 17787 | 1024 | 4 | 37 |
| xsum | rag | 62.95 | 76.85 | 15.88 | 30.11 | 27952 | 2304 | 8 | 60 |
| xsum | templated | 61.38 | 70.61 | 16.29 | 28.72 | 26992 | 2048 | 8 | 60 |
