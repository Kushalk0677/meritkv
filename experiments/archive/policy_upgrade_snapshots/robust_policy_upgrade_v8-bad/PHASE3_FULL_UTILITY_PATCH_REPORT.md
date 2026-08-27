# Phase 3 Full-Path Utility + Lite Candidate Discovery Patch

This patch applies the Phase 3 utility gate where the previous benchmark showed it was needed most: the full ShadowKV++ exact-prefix path.

## What changed

1. **Full ShadowKV++ exact reuse now uses utility admission**
   - `ShadowKVPlusEngine` accepts:
     - `min_reuse_prefix_tokens`
     - `enable_utility_admission`
     - `utility_min_net_saved_ms`
   - `run_benchmark.py` passes the existing CLI flags into full ShadowKV++ as well as Lite.
   - Exact cache candidates are admitted only when estimated net utility is positive enough.
   - Negative / too-small exact candidates are bypassed and counted.

2. **Lite now stores the same exact scaffold candidates as full ShadowKV++**
   - Lite no longer requires `hint >= min_reuse_prefix_tokens` before storing.
   - The minimum reuse threshold is now correctly applied at admission time, not candidate creation time.
   - This should prevent Lite from remaining at `hit_rate = 0` simply because scaffold hints are shorter than the current break-even threshold.

3. **Telemetry now becomes meaningful on both engines**
   - For full ShadowKV++ and Lite, expect these counters to move when candidates exist:
     - `utility_admission_checks_total`
     - `utility_admission_admit_total`
     - `negative_utility_bypass_total`
     - `small_prefix_bypass_total`
     - `net_latency_saved_estimate_ms_total`

## Validation

Full test suite:

```text
76 passed, 1 skipped
```

Smoke test with repeated 20-token scaffold:

```text
Lite: hit_rate=0.875, utility_checks=7, admits=7, store_successes=1
Full: hit_rate=0.875, utility_checks=7, admits=7, store_successes=1
```

## What to expect in the next Colab run

Compared with the previous result where:

```text
shadow_kv_plus_lite: hit_rate=0, utility_checks=0
shadow_kv_plus: hit_rate≈0.984, utility_enabled=False
```

The next run should show:

```text
shadow_kv_plus_lite: hit_rate > 0 if exact scaffold candidates exist
shadow_kv_plus: utility_admission_enabled=True
shadow_kv_plus: utility_admission_checks_total > 0
shadow_kv_plus: negative_utility_bypass_total may become > 0
```

A lower hit rate is acceptable if latency improves. The point of ShadowKV++ is not maximum hit rate; it is positive net utility.
