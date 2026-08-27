# Gemma 4 Memory-Bound Trace

This folder contains the Gemma-4 Blackwell memory-bound trace. Source
provenance is recorded in `SOURCE.txt`.

## Coverage

- Models: `google/gemma-4-12B-it`, `google/gemma-4-31B-it`.
- Backend/GPU: vLLM on RTX PRO 6000 Blackwell.
- Seeds: `42`, `123`, `456`, `789`, `999`.
- Engines: `no_cache`, native vLLM APC, and MeritKV.
- Trace shape: 40 fill requests, 30 churn/victim requests, and 30 recovery requests.

## Included Files

- `MEMORY_BOUND_RESULTS.md`: paper-facing summary table.
- `gemma4_memory_bound_summary.csv`: compact aggregate table for quick inspection.
- `gemma_4_12b/` and `gemma_4_31b/`: aggregate JSON plus per-seed summary and trace JSON.
- `MEMORY_BOUND_LRU_LFU_BASELINE.md`: paper-facing five-policy comparison.
- `lru_lfu/`: complete controlled-bank LRU/LFU per-seed traces, summaries,
  aggregates, logs, metadata, and checksums.
- `MANIFEST_SHA256.txt`: checksums for copied files.
- `SOURCE.txt`: source provenance.

## Interpretation

Use Phase 3 recovery, victim misses/evictions, and declined Phase 1 admissions
from the aggregate as the evidence. The retained seed JSONs are legacy
provenance: some generic derived fields and latency records do not reconcile
with the aggregate and must not be used to reconstruct paper values.
