# Canonical Paper Tables

These CSVs are the machine-readable counterparts of the quantitative tables in
the current MeritKV paper. Values are measured aggregates unless a row is
explicitly marked as analytical or descriptive. The larger raw execution
archives remain in their original result-family directories.

Where a raw per-seed record could not be reconciled with the canonical
aggregate, it is excluded rather than reconstructed.

Rows carry their own scope where eligibility matters. In particular, the
Qwen2.5 templated custom-splice timing is retained for transparency but marked
diagnostic and excluded from validated performance; semantic-execution rows
are diagnostic rather than production-safety evidence.

| CSV | Paper table |
|---|---|
| `per_dataset_speedup.csv` | Per-dataset MeritKV/strict-reactive comparison |
| `calibration.csv` | Calibrated per-model cost model |
| `controlled_breakdowns.csv` | Controlled per-model and model-by-mode results |
| `realistic_by_dataset.csv` | Process-isolated templated results by dataset |
| `semantic_execution.csv` | Semantic detection/execution mechanism |
| `fidelity_summary.csv` | Five-model custom-splice agreement summary |
| `fidelity_by_dataset.csv` | Five-model agreement by dataset |
| `fidelity_precision.csv` | Float32/float16 comparison |
| `sensitivity_oat.csv` | One-at-a-time sensitivity |
| `admission_all.csv` | Admission-policy comparison |
| `mixed_traffic.csv` | Mixed-traffic results on T4 and Blackwell |
| `memory_bound_locality.csv` | Three-phase capacity-pressure locality |
| `memory_bound_multiround.csv` | Four-arm multiround enforcement trace |
| `runtime_qwen.csv` | Qwen2.5 native-cache/write-through runtime comparison |
| `runtime_gemma_overlay.csv` | Gemma-4 overlay overhead across three backends |
