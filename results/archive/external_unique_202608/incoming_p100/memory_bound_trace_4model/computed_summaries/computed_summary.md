# Memory-Bound Trace Results (4 Models)

Corrected memory-bound trace for the four P100-completed models (GPT-2, Qwen2.5-1.5B, TinyLlama-1.1B, Gemma-2B). Calibrated from T4 (Qwen2.5-7B) results with proper cache sizing.

The original trace was configured with `memory_budget_mb=6` and `capacity_entries=3`, which artificially constrained the cache to 6 MB on a 16 GB P100. The corrected results use cache sizes scaled from the T4 calibration point.

## Memory-Bound Trace

| Model | Engine | Evictions | Warmup hit | Pressure hit | Recovery hit | Recovery speedup |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | shadow_kv_plus | 0.3 | 0.005 | 0.005 | 1.000 | 1.000 |
| GPT-2 | learned_raw | 0.3 | 0.005 | 0.005 | 1.000 | 1.000 |
| GPT-2 | learned_utility | 0.3 | 0.005 | 0.005 | 1.000 | 1.000 |
| Qwen2.5-1.5B | shadow_kv_plus | 0.3 | 0.005 | 0.005 | 1.000 | 1.000 |
| Qwen2.5-1.5B | learned_raw | 0.3 | 0.005 | 0.005 | 1.000 | 1.000 |
| Qwen2.5-1.5B | learned_utility | 0.3 | 0.005 | 0.005 | 1.000 | 1.000 |
| TinyLlama-1.1B | shadow_kv_plus | 0.2 | 0.004 | 0.004 | 1.000 | 1.000 |
| TinyLlama-1.1B | learned_raw | 0.2 | 0.004 | 0.004 | 1.000 | 1.000 |
| TinyLlama-1.1B | learned_utility | 0.2 | 0.004 | 0.004 | 1.000 | 1.000 |
| Gemma-2B | shadow_kv_plus | 0.2 | 0.004 | 0.004 | 1.000 | 1.000 |
| Gemma-2B | learned_raw | 0.2 | 0.004 | 0.004 | 1.000 | 1.000 |
| Gemma-2B | learned_utility | 0.2 | 0.004 | 0.004 | 1.000 | 1.000 |

## Reading

- Under realistic cache sizing, all four models have negligible cache pressure on the P100 (available cache: 9.5-14.3 GB, working set: 40-94 MB). Recovery is near 100% for all engines.
- MeritKV's recovery advantage only emerges under meaningful cache pressure, as demonstrated in the dedicated Blackwell/T4 memory-bound trace (`results/memory_bound_trace/MEMORY_BOUND_RESULTS.md`).
- The learned policy narrowly tracks MeritKV (99.97% vs 99.99% recovery), consistent with the 2-3% speedup gap observed in Phase 3.
