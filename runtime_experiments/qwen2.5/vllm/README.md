# vLLM Runtime Results

`results.csv` contains two measured paper-facing campaigns: APC+MeritKV/APC
latency ratios for Qwen2.5 1.5B through 32B, and the no-cache/APC/APC+MeritKV
aggregates for the separate five-replicate 32B campaign.

The primary 32B raw campaign is under `raw/q32b_5rep_20260701/`; earlier 32B
records remain under `raw/q32b_20260603/`. Paired percentage changes are means
of per-replicate ratios and need not equal ratios of rounded aggregate means.
The former detailed top-level export is preserved as
`raw/legacy_detailed_results_precompact.csv`.

The integration is write-through, so APC+MeritKV versus APC measures observed
overlay compatibility/overhead rather than enforced acceleration.
