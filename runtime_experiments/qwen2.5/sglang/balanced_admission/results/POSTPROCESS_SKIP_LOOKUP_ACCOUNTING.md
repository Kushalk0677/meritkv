# Skip-Lookup Waste Accounting Correction

SGLang native admission bypass requests set `skip_lookup=True`. These requests do not transfer or miss cache entries, so this postprocess sets cache-query, cache-hit, cache-miss-waste, and failure-waste bytes to zero for rows where `shadowkv_admission_mode=native_sglang_hook` and `shadowkv_plan_strategy=bypass`. Latency and speedup fields are unchanged.

Adjusted rows: `256` across `3` per-request files.
