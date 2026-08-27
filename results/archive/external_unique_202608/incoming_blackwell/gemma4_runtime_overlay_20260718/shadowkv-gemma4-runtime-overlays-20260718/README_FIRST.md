# Read Me First

This is the corrected Gemma 4 Blackwell runtime matrix. It includes native and
ShadowKV++ overlay arms for vLLM APC, SGLang RadixAttention, and LMCache.

Start with:

1. `REPORT_FOR_KUSHAL.md`
2. `analysis/ANOMALY_AUDIT.md`
3. `analysis/paired_summary.csv`
4. `OVERLAY_SEMANTICS.md`

Raw smoke and full results are under `results_smoke/` and `results_full/`.
Runtime/package/hardware evidence is under `metadata/`, and the exact benchmark
source is under `source_snapshot/`.

Do not describe the overlay arms as native per-request enforcement. They use
`write_through_admission`: decisions are observed and timed, while each runtime
retains cache ownership.
