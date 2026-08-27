# Memory-Bound Trace Results

This folder contains cache-pressure traces for MeritKV. The trace is a three-phase workload: fill reusable prefixes, churn the cache with distractors and victims, then measure recovery.

## Contents

| Path | Contents |
|---|---|
| `MEMORY_BOUND_RESULTS.md` | Combined paper-facing summary for Qwen/Phi and Gemma memory-bound traces. |
| `qwen2.5/` | Qwen2.5 Blackwell memory-bound raw artifacts and summary. |
| `gemma4/` | Gemma-4 Blackwell aggregate plus legacy raw provenance; `lru_lfu/` contains the measured five-policy capacity-pressure comparison. |

## Scope Notes

- The Gemma-4 LRU/LFU extension compares admit-all, FIFO cap, LRU, LFU,
  and MeritKV at two cache budgets over five paired seeds.


- Qwen2.5 Blackwell artifacts include aggregate and selected per-seed JSON
  traces.
- Gemma-4 per-seed JSONs are retained as legacy provenance; paper-facing claims
  use the aggregate because some derived seed fields do not reconcile.
- The T4/Phi rows are currently summary-only in this public folder.
- Recovery rate is a cache-hit/recovery metric, not a latency metric.

