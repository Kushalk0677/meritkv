# Recency/Frequency-Aware Baseline on the Gemma-4-31B Capacity-Pressure Trace

## Measured Results

**Model**: google/gemma-4-31B-it on Blackwell RTX PRO 6000.
**Trace**: multi-round capacity-pressure trace, 140 requests per run
(20 fill + 4 × (15 churn + 15 recovery)), 5 seeds (42, 123, 456, 789, 999).
**Budgets**: 28 GB and 56 GB.
**Cache layer**: all policies enforced through the same controlled
`TieredStateBank` cache (fresh cache per seed; identical request ordering
per seed across arms — results are paired).
**Arms**: admit-all, FIFO occupancy cap, LRU, LFU, MeritKV.

The artifact key `native_apc` denotes the controlled-bank **admit-all** arm;
it is retained for filename compatibility but is not a direct measurement of
vLLM's internally managed APC eviction policy. Native vLLM APC remains in the
original multiround experiment.

The comparison answers: **"Is MeritKV's benefit just caching less?"** The
FIFO occupancy cap controls for simple capacity restriction. LRU and LFU isolate
eviction policy (admit everything, vary only eviction order). MeritKV
combines selective admission with utility-based selection.

---

## High Pressure (28 GB budget)

| Metric | Admit-all | FIFO cap | LRU | LFU | MeritKV |
|--------|:---------:|:--------:|:---:|:---:|:-------:|
| Final-round recovery* | 20.8% | 33.7% | 20.5% | 52.1% | **80.3%** |
| Total evictions | 47.4 | 38.4 | 45.4 | 29.0 | **11.0** |
| Mean latency | 78.2 ms | 72.4 ms | 76.9 ms | 69.6 ms | **65.0 ms** |
| Declined admissions (fill) | 0 | 12 | 0 | 0 | 15 |

\* Recovery is the controlled-cache ledger's final-round shared-prefix
reuse fraction. It is token-level and therefore is not restricted to
increments of $1/15$.

### Per-round recovery (high pressure)

| Round | Admit-all | FIFO cap | LRU | LFU | MeritKV |
|-------|:---------:|:--------:|:---:|:---:|:-------:|
| 1 | 41.7% | 58.2% | 42.9% | 69.8% | 90.9% |
| 2 | 32.3% | 48.6% | 33.8% | 62.2% | 86.9% |
| 3 | 26.0% | 39.8% | 26.6% | 56.6% | 84.4% |
| 4 | 20.8% | 33.7% | 20.5% | 52.1% | **80.3%** |

---

## Moderate Pressure (56 GB budget)

| Metric | Admit-all | FIFO cap | LRU | LFU | MeritKV |
|--------|:---------:|:--------:|:---:|:---:|:-------:|
| Final-round recovery* | 48.6% | 59.8% | 49.5% | 72.2% | **87.4%** |
| Total evictions | 31.2 | 24.4 | 30.0 | 17.2 | **5.8** |
| Mean latency | 71.1 ms | 68.0 ms | 70.8 ms | 65.8 ms | **62.3 ms** |
| Declined admissions (fill) | 0 | 12 | 0 | 0 | 15 |

\* Recovery uses the same token-level shared-prefix definition as at 28 GB.

### Per-round recovery (moderate pressure)

| Round | Admit-all | FIFO cap | LRU | LFU | MeritKV |
|-------|:---------:|:--------:|:---:|:---:|:-------:|
| 1 | 67.7% | 77.4% | 69.4% | 86.9% | 95.7% |
| 2 | 59.1% | 69.3% | 61.2% | 80.2% | 92.4% |
| 3 | 54.1% | 65.0% | 54.8% | 75.8% | 91.7% |
| 4 | 48.6% | 59.8% | 49.5% | 72.2% | **87.4%** |

---

## Interpretation

**LRU tracks the admit-all comparator.** At both budgets, LRU final-round
recovery (20.5% / 49.5%) is close to the controlled-bank admit-all arm
(20.8% / 48.6%). No separate equivalence claim is made here; the paired
per-seed values and confidence intervals are released with the aggregates.

**LFU is the informative eviction-policy comparison.** Frequency-aware
eviction (LFU) preserves the reusable shared prefix under churn far better
than recency-based eviction: 52.1% / 72.2% final-round recovery vs
20.5% / 49.5% for LRU, with 29.0 / 17.2 total evictions vs 45.4 / 30.0.

**FIFO cap controls for capacity restriction.** The FIFO occupancy cap
(33.7% / 59.8% recovery, 38.4 / 24.4 evictions) improves on admit-all by
admitting less and reducing pollution, but it evicts blindly in insertion
order and does not preserve the specific entries the recovery phase reuses.

**MeritKV remains the best.** MeritKV's utility-based selection preserves
the entries the final recovery round reuses: 80.3% / 87.4% recovery and
11.0 / 5.8 total evictions, giving higher recovery and fewer evictions than
both the FIFO cap and LFU.

| Comparison (final-round recovery) | 28 GB | 56 GB |
|-----------------------------------|:-----:|:-----:|
| MeritKV vs admit-all | +59.5 pp | +38.8 pp |
| MeritKV vs FIFO cap | +46.6 pp | +27.7 pp |
| MeritKV vs LRU | +59.8 pp | +37.9 pp |
| MeritKV vs LFU | +28.2 pp | +15.2 pp |
| LFU vs LRU | +31.6 pp | +22.7 pp |
| LRU vs admit-all | −0.3 pp | +0.9 pp |

**Conclusion (scoped to this trace).** On the Gemma-4-31B capacity-pressure
trace, a standard frequency-aware eviction policy (LFU) recovers much of
MeritKV's advantage over recency-based caching, confirming that part of the
benefit is explainable by eviction policy rather than utility scoring.
However, MeritKV still outperforms LFU by 28.2 pp (28 GB) and 15.2 pp
(56 GB) in final-round recovery, with 18.0 and 11.4 fewer evictions,
respectively.
The residual gap beyond what a frequency-aware policy achieves is evidence
for utility-guided selection: preserving the specific entries the recovery
phase reuses, not merely keeping frequently used ones. The benefit is
therefore **not simply "caching less"** (the FIFO cap controls for that and
underperforms LFU and MeritKV); it combines frequency-aware retention with
utility-guided selection.

---

## Released Aggregate Artifacts

- Per-seed compact ledgers: `v10/results/memory_bound_lru_lfu/{high,moderate}_pressure/{arm}/seed_{42,123,456,789,999}/trace.json`
- Per-seed summaries: same directory, `summary.json`
- Per-arm aggregates: `v10/results/memory_bound_lru_lfu/{budget}/aggregate_{arm}.json`
- Run-status logs: `.../run_logs/run_seed_*.log`
- Experiment metadata: `v10/results/memory_bound_lru_lfu/run_metadata.json`
- Checksums: `v10/results/memory_bound_lru_lfu/MANIFEST_SHA256.txt`

All values are 5-seed means; 95% CIs are in the aggregate JSONs. The released
artifact set contains compact per-seed ledgers, summaries, aggregates,
run-status logs, metadata, and checksums.
