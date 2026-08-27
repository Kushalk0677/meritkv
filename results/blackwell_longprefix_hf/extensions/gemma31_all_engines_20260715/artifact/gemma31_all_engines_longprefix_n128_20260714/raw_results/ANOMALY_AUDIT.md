# Gemma 4 31B All-Engine Anomaly Audit

> **Auxiliary audit:** This checks the single-seed extension only. It is not a
> distinct paper result or a multi-seed engine ranking.

## Structural checks

- 110/110 result cells and 110/110 job records.
- 10 datasets x 11 engines; 10 cells per engine.
- Zero failed jobs and no missing requested engine.

## Material caveats

- One seed and one fixed-order sweep; no confidence interval or randomized-order claim is supported.
- HF native_prefix_cache is a placeholder/observer: it records matches but calls full prefill for every request, so it is not a native-runtime APC/Radix baseline.
- reuse_path_breakdown path_reading is specialized for full ShadowKV++ policy counters; no_reuse_path_executed does not negate reactive/Lite reuse_successes.
- The four ShadowKV++ variants differ by less than one percent in aggregate mean latency; their internal ranking is within plausible single-run/order noise.

## Signals

- `shadow_kv_plus`: 31.37% mean-latency improvement, 37.00% P95 improvement, 33.21% lower GPU energy, 10/10 mean wins, and 1270 reuse successes.
- `shadow_kv_plus_lite`: 31.27% mean-latency improvement, 37.23% P95 improvement, 33.30% lower GPU energy, 10/10 mean wins, and 1270 reuse successes.
- `shadow_kv_plus_best_latency`: 30.87% mean-latency improvement, 36.77% P95 improvement, 33.14% lower GPU energy, 10/10 mean wins, and 1270 reuse successes.
- `shadow_kv_plus_raw_observer`: 31.25% mean-latency improvement, 36.75% P95 improvement, 33.36% lower GPU energy, 10/10 mean wins, and 1270 reuse successes.
- `native_prefix_cache` recorded mean hit rate 0.700 but zero reuse successes and 787 bypassed matches; source inspection confirms full prefill is always executed.
- `shadow_kv` executed zero reuse successes and was 1.11% slower than no-cache in aggregate mean latency.
- Reactive, greedy, strict-reactive, and all four ShadowKV++ variants recorded 1,270/1,280 reuse successes across the ten cells.
