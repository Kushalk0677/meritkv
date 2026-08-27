# SGLang and LMCache Runtime Results

`results.csv` is the paper-aligned measured aggregate table for SGLang
RadixAttention, RadixAttention plus the write-through MeritKV decision overlay,
and the contextual LMCache comparison. LMCache rows are also present in
`../lmcache/results.csv`.

The raw Qwen2.5-32B campaign under `raw/q32b_full_20260608/` is retained as
provenance. Its local summary records that specific campaign and does not
override the current paper aggregation.
The former detailed top-level export is preserved as
`raw/legacy_detailed_results_precompact.csv`.

Because the integration does not enforce admission inside SGLang, overlay
differences are compatibility/overhead measurements.

The exception is the separately scoped `balanced_admission/` experiment. It
uses a native SGLang per-request hook to enforce skip-lookup and skip-write
decisions for Qwen2.5-1.5B. Its complete result bundle is retained independently
from the write-through scale aggregates above.
