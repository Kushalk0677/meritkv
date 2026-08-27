# Plan: Admission Baselines + Mixed-Traffic Experiments

## 1. Admission-Control Baselines Table

Add 6 new engines to `run_benchmark.py` alongside the existing ones.
Each wraps a simpler gating rule around the same `TieredStateBank` + `FakeBackend`
so the comparison is controlled (same cache, same workload, different decision rule).

### New engines

| Engine | Decision rule | What it tests |
|--------|--------------|---------------|
| `gate_prefix_len_16` | Reuse if match len >= 16 | Is MeritKV just a cheap length cutoff? |
| `gate_prefix_len_32` | Reuse if match len >= 32 | More aggressive length cutoff |
| `gate_prefix_len_64` | Reuse if match len >= 64 | Even more aggressive |
| `gate_cost_only` | Reuse if B - C >= 0 (no waste term) | Does the waste term actually matter? |
| `gate_waste_only` | Reuse if W < threshold (no benefit calc) | Does benefit matter or just waste avoidance? |
| `gate_freq_threshold` | Reuse after n>=3 observations (no utility) | Common heuristic admission |
| `gate_static_tuned` | Per-model hardcoded k* threshold | Hand-tuned deployment policy |

### Implementation

Add to `experiments/run_benchmark.py` `build_engine()`:

```python
if engine_name == 'gate_prefix_len_16':
    e = ReactivePrefixCacheEngine(backend, max_memory_mb=args.max_memory_mb)
    e.tuning.min_reuse_prefix_tokens = 16
    e.name = 'gate_prefix_len_16'
```

For `gate_cost_only`, subclass or monkey-patch `UtilityModel.admission()` to
zero out the waste terms.

### Expected output table (in paper §Ablations)

```
Engine              Speedup   Waste   HitRate
MeritKV             1.365x    0.156   0.402
Gate: len>=16       1.287x    0.264   0.606  
Gate: len>=32       1.254x    0.000   0.310
Gate: cost-only     1.341x    0.201   0.450
Gate: waste-only    1.298x    0.140   0.380
Gate: freq>=3       1.208x    0.284   0.617
Gate: static k*     1.310x    0.220   0.500
Offline oracle      1.408x    0.000   0.700
```

### Workload to use

Run on the existing 5-model × 10-dataset × 3-mode × 3-seed sweep
(just on FakeBackend for speed, or on the real HF backend).

---

## 2. Mixed-Traffic Workloads

Add 5 new `synthetic` workload variants to `SYNTHETIC_VARIANTS` in
`src/proactive_kv_cache/workload.py`.

### New variants

| Variant | Composition | Zipf α | Burst | Why it matters |
|---------|-----------|--------|-------|---------------|
| `clean_reusable` | 90% templated, long shared prefixes, 10% raw | 1.8 | 0.10 | Ideal for cache, tests overhead |
| `mostly_raw` | 90% diverse raw, 10% templated | 0.3 | 0.0 | Tests bypass efficiency |
| `mixed_serving` | 40% raw, 40% templated, 20% rag | 1.0 | 0.15 | Realistic chat + RAG mix |
| `bursty_reuse` | Repeated prefix appears in bursts, disappears | 1.5 | 0.40 | Tests adaptive speculation cooldown |
| `adversarial_short` | Many apparent matches below breakeven (k < 16) | 2.0 | 0.0 | Tests waste gate on short prefixes |
| `speculation_trap` | High early frequency on a prefix, then no future reuse | 0.5 | 0.0 | Tests speculation waste detection |

### Implementation

Add to `SYNTHETIC_VARIANTS` dict in `workload.py`:

```python
'clean_reusable': dict(alpha=1.8, mean_inter_arrival_ms=100, long_prefix_bias=1.0,
                       template_reuse_prob=0.9, ...),
'mixed_serving': dict(alpha=1.0, mean_inter_arrival_ms=120, burst_probability=0.15,
                      template_ratio=0.4, rag_ratio=0.2, raw_ratio=0.4, ...),
```

This requires extending `SyntheticWorkloadGenerator` to interleave different
`prompt_mode` requests in a single trace.

### Evaluation

Compare MeritKV vs no-cache and APC on each mixed workload on FakeBackend
(speed, for iteration) then on Blackwell (for the paper).

---

## 3. What It Costs

| Task | Effort | Runs on |
|------|--------|---------|
| Add 6 gate engines | ~2h code | Any machine |
| Run gate sweep on FakeBackend | ~30 min | This laptop |
| Add mixed workload variants | ~3h code + test | This laptop |
| Run mixed workloads on FakeBackend | ~1h | This laptop |
| Run mixed workloads on Blackwell | ~2h GPU | Evan's box |
| Write up tables for paper | ~1 day | — |

Total: ~3-4 days part-time, maybe 1 day if Evan runs Blackwell in parallel.
