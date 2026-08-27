# SGLang Native ShadowKV++ Admission Qwen14B Gate Summary

Rows: 12
Hook tests: `passed`
Gate passed: `True`

Primary metric: end-to-end latency around ShadowKV planning, server request, and feedback. HTTP/server latency is diagnostic only.

## By Baseline

| Baseline | Cells | E2E mean ms | E2E P95 ms | E2E RPS | HTTP mean ms | HTTP P95 ms | Cached tokens | Plans | Allows | Bypasses | Skip lookup | Skip write | Server skip lookup | Server skip write |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sglang_radix_attention | 6 | 60.29 | 56.57 | 16.59 | 60.28 | 56.56 | 94137 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sglang_radix_attention_shadowkv_plus | 6 | 66.52 | 67.64 | 15.05 | 65.96 | 67.10 | 40591 | 1536 | 488 | 1048 | 1048 | 1028 | 1048 | 1028 |

## Paired ShadowKV++ vs Native Radix

- Paired cells: `6`
- Mean E2E latency delta: `10.38%`
- Mean E2E P95 delta: `19.56%`
- Mean E2E throughput delta: `-9.23%`
- Cached token delta total: `-53546`
- Native hook skip-lookups: `1048`
- Native hook skip-writes: `1028`

## Packaging Checks

- Source snapshot: `metadata/source_snapshot/`
- Runtime versions: `metadata/runtime_versions.txt`
- Docker image metadata filename uses only Windows-safe characters.
- Profiler traces are not included.
