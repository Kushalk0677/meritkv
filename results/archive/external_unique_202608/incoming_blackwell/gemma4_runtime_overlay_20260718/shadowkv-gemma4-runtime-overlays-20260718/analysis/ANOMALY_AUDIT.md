# Corrected Gemma 4 Runtime + ShadowKV++ Matrix Audit

## Verification

- Result cells: 300/300
- Unique expected cells: 300/300
- Native/overlay pairs: 150/150
- Requests: 76,800; all cells at 256: True
- NVML energy complete and error-free: True
- Overlay plans and decisions complete: True
- Overlay runtime-cache reset failures: 0
- Reuse failures: 0
- Idle stabilization timeouts: 0
- LMCache logs with stores and retrieves: 10/10
- Randomized block plan: 30/30 unique model/arm blocks

## Native Versus ShadowKV++

| Runtime | Pairs | Mean latency delta | Mean P95 delta | Throughput delta | Energy delta | Latency wins | P95 wins | Allows | Bypasses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vllm_apc | 50 | +2.61% | +3.19% | -2.08% | +0.03% | 19/50 | 16/50 | 12,750 | 50 |
| sglang_radix_attention | 50 | -2.19% | -1.75% | +2.36% | -1.69% | 32/50 | 32/50 | 12,750 | 50 |
| lmcache | 50 | +2.15% | +3.35% | -1.82% | +0.84% | 11/50 | 10/50 | 12,750 | 50 |

## Caveats And Anomalies

- The ShadowKV++ arms are portable write-through policy overlays. Planning, server request, and feedback are included in end-to-end timing, but the external runtimes retain cache ownership.
- A bypass records the policy decision but does not enforce native per-request skip-lookup/skip-write behavior. These arms are policy-observer measurements, not native runtime-hook implementations.
- One randomized block schedule and one run per cell were used. Small deltas are not replicated performance estimates.
- Runtime builds and memory settings differ by system; exact metadata and image inspection records are included.
- Individual zero-cache-evidence cells: 20.
- Zero-cache evidence is concentrated in lmcache/ag_news: 10, lmcache_shadowkv_plus/ag_news: 10. These requests completed normally; for LMCache, AG News reusable prefixes were below the configured 256-token external-cache chunk boundary.
- All cells met the configured idle-power stabilization criterion.

## Audit Result

**PASS: coverage, request counts, energy, cache evidence, admission counters, transfer logs, and schedule checks passed.**
