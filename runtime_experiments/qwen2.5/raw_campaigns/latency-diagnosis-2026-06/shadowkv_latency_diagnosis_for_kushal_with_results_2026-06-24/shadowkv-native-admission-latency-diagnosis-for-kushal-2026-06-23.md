---
title: ShadowKV Native Admission Latency Diagnosis for Kushal 2026-06-23
created: 2026-06-23
updated: 2026-06-23
author: Keystone (Codex on IronGod)
authors: [Keystone]
last_editor: Keystone (Codex on IronGod)
type: project
tags: [shadowkv, sglang, latency, native-admission, kushal, first-light]
sources:
  - shadowkv-sglang-small-model-test-2026-06-13
---

# ShadowKV Native Admission Latency Diagnosis for Kushal - 2026-06-23

## Message For Kushal

Kushal,

The latest Qwen2.5-14B native-admission gate proves that the SGLang hook is real, but it also explains why ShadowKV++ did not improve latency in that run. The slowdown is not mainly Python-side ShadowKV planning overhead. Planning averaged about `0.51 ms/request` and feedback averaged about `0.03 ms/request`. The larger issue is that ShadowKV++ caused SGLang to lose too many useful native Radix hits: cached tokens fell by `53546` total, about `34.9` cached tokens/request.

The diagnosis points to a cold-start admission problem and a calibration problem:

- The workload had `924` one-off cache-pressure requests and `612` hot reusable-prefix requests.
- ShadowKV++ produced `1048` `no_reuse_signal` bypasses, so about `124` reusable-prefix requests were also treated as no-reuse.
- Hot-family admission failed discretely. Some cells stored all `4/4` hot prefix families, some stored `3/4`, and one stored only `2/4`. If the first request in a hot family is not stored, every later request in that family misses.
- The policy produced `455` `exact_prefix_bootstrap` decisions but only `33` `exact_prefix_net_positive` decisions. That means most "allow" decisions were bootstrap exceptions, not the utility model confidently admitting reuse.
- `reuse_overhead_ms` drifted to about `35-38 ms`, which likely overestimates incremental cache-reuse overhead by folding fixed HTTP/server latency into the reuse cost.
- The synthetic cache-pressure workload may not actually pressure Qwen14B on this 80GB setup. SGLang logs showed token usage as `0.00`, so native Radix was not visibly harmed by one-off writes. In that regime, skip-write has little upside and any hot-prefix miss is pure loss.

Question: should we proceed with the patch series below? I recommend yes, but as a reviewable debug patch first, followed by a smoke test and one 256-request Qwen14B validation block before any larger sweep.

## Current Evidence

Result root:

`/home/jade_hand/research/shadowkv/results_sglang_native_admission_qwen14b_gate_2026-06-22`

Deliverable already packaged:

`/Users/evanleri/Desktop/shadowkv_qwen14b_native_admission_gate_2026-06-22.zip`

Aggregate:

| Baseline | Cells | E2E mean ms | E2E P95 ms | E2E RPS | Cached tokens | Plans | Allows | Bypasses | Skip lookup | Skip write |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native Radix | 6 | 60.29 | 56.57 | 16.59 | 94137 | 0 | 0 | 0 | 0 | 0 |
| ShadowKV++ native admission | 6 | 66.52 | 67.64 | 15.05 | 40591 | 1536 | 488 | 1048 | 1048 | 1028 |

Paired result:

| Metric | ShadowKV++ vs native Radix |
| --- | ---: |
| Mean end-to-end latency | `+10.38%` |
| P95 end-to-end latency | `+19.56%` |
| End-to-end throughput | `-9.23%` |
| Cached-token delta | `-53546` |

Per-cell symptom:

| Rep | Mode | E2E delta ms | Cached-token delta | Excess hot no-reuse | Bypass-store allowed | Final reuse overhead ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `templated` | 2.64 | -7160 | 4 | 4 | 37.87 |
| 1 | `rag` | 5.40 | -9949 | 29 | 3 | 35.38 |
| 2 | `templated` | 10.20 | -11487 | 54 | 2 | 35.67 |
| 2 | `rag` | 10.05 | -7782 | 4 | 4 | 36.93 |
| 3 | `templated` | 4.85 | -9386 | 29 | 3 | 38.38 |
| 3 | `rag` | 4.24 | -7782 | 4 | 4 | 36.49 |

## Proposed Patch Series

These changes are intended to be submitted as a reviewable patch before running the full validation again.

### Patch 1 - Per-request decision trace

Files:

- `/home/jade_hand/research/shadowkv/literature_accurate_baselines/run_runtime_cache_baseline.py`
- `/home/jade_hand/research/shadowkv/literature_accurate_baselines/adapter_lib.py`

Change:

- Add `admission_decision_trace.jsonl` to each ShadowKV++ output directory.
- Add `admission_decision_summary.json` with counts by trace class, reuse family, plan reason, skip-lookup, skip-write, and cached-token result.

Per-request fields:

- `request_id`
- request index
- `shadowkv_trace_class`
- `reuse_family`
- expected behavior
- prompt mode
- shared-prefix hint tokens
- exact match length
- plan strategy
- plan reason
- plan score
- reusable prefix tokens
- skip-lookup flag
- skip-write flag
- bypass-store allowed flag
- stored-after-request flag
- prompt tokens
- cached tokens
- HTTP latency
- end-to-end latency
- planning latency
- feedback latency

Reason:

The aggregate proves there is a cached-token deficit, but it does not show exactly which hot families were lost. This trace makes the failure inspectable without rerunning under a debugger.

### Patch 2 - Hinted-prefix cold-start store bootstrap

File:

`/home/jade_hand/research/shadowkv/literature_accurate_baselines/adapter_lib.py`

Change:

- Extend `ExternalAdmissionController.should_store_after_bypass`.
- If metadata includes `shared_prefix_hint_tokens` and `shared_prefix_text`, and the hinted prefix is not already in the external admission bank, allow the first store for that hinted prefix even when the request's lookup is bypassed.
- Track this as a separate reason/counter, for example `hinted_prefix_cold_store`.
- Keep one-off requests unaffected because they do not carry `shared_prefix_text`.

Intended behavior:

- Hot reusable prefix, first occurrence: skip lookup if no match exists, but allow write so later requests can hit.
- Hot reusable prefix, later occurrences: policy can use actual exact-prefix history.
- One-off cache pressure request: skip lookup and skip write as before.

Reason:

The current policy can decide "no reuse signal" before a hot family has been stored once. That creates a self-fulfilling miss: no initial store means no later exact match.

### Patch 3 - Fix external-runtime reuse overhead calibration

File:

`/home/jade_hand/research/shadowkv/literature_accurate_baselines/adapter_lib.py`

Change:

- Rework `_update_runtime_calibration`.
- Do not treat `result.latency_ms - estimated_uncached_ms` as uncapped reuse overhead.
- Add a conservative cap for external runtime reuse overhead, initially around `4-8 ms`, or a fraction of observed request latency.
- Record raw observed overhead and capped overhead separately in metrics.
- Keep `full_ms_per_token` calibration from uncached requests, but prevent cached-request fixed server latency from inflating `reuse_overhead_ms` to `35-38 ms`.

Reason:

The current calibration makes reusable prefixes look net-negative even when native Radix shows useful cached-token hits. It is probably estimating fixed request/server time, not incremental cache-reuse overhead.

### Patch 4 - Admission-sensitive workload with actual cache pressure

Files:

- `/home/jade_hand/research/shadowkv/session_files/make_admission_sensitive_trace_2026_06_22.py`
- new runner, probably `/home/jade_hand/research/shadowkv/session_files/run_shadowkv_sglang_native_admission_qwen14b_latency_debug_2026-06-23.sh`

Change:

- Fork the trace generator rather than overwrite the 2026-06-22 generator.
- Add knobs:
  - `--hot-family-count`
  - `--hot-repeats-per-family`
  - `--one-off-ratio`
  - `--one-off-token-count`
  - `--pressure-scale`
- Add a pressure mode that creates enough unique one-off KV to compete with hot prefixes.
- Keep native Radix and ShadowKV++ under identical runtime settings.
- Consider bounding SGLang KV capacity with a shared setting such as `--max-total-tokens` or lower shared `--mem-fraction-static`, but only if both baselines use the exact same setting and the run remains stable.

Reason:

The previous pressure trace caused policy divergence but may not have caused real cache pressure on the 80GB Qwen14B setup. Skip-write can only help latency if native Radix pays a cost for storing one-offs.

### Patch 5 - Aggregation by trace class and reuse family

Files:

- `/home/jade_hand/research/shadowkv/literature_accurate_baselines/run_runtime_cache_baseline.py`
- aggregation logic inside the Qwen14B gate runner

Change:

- Report latency/cached-token stats separately for:
  - `hot_reusable_prefix`
  - `one_off_cache_pressure`
  - each `reuse_family`
- Add explicit counters:
  - hot requests allowed
  - hot requests bypassed
  - hot first stores allowed
  - hot first stores denied
  - one-off writes skipped
  - one-off writes allowed

Reason:

Kushal will likely want to know whether ShadowKV improves the intended class even if aggregate latency is mixed. Class-level aggregation avoids hiding the policy behavior behind one total number.

## Validation Plan After Approval

1. Apply the patch series on first-light in a clearly named branch or backup set.
2. Build a new Docker image, likely `shadowkv-sglang-native-admission:2026-06-23-latency-debug`.
3. Run deterministic hook tests again.
4. Run a 16-request smoke to verify decision trace output and server counters.
5. Run one 256-request Qwen2.5-14B validation block, not the full matrix.
6. Inspect:
   - hot reusable-prefix cache hit retention
   - one-off skip-write rate
   - per-request decision trace
   - cached-token delta versus native Radix
   - end-to-end latency delta
7. Only if the one-block result is clean, rerun the full 3-repetition Qwen14B validation.

## Approval Request

Do you want us to proceed with this patch series?

Recommended answer: yes, but as a reviewable latency-debug patch first. I would not launch another full 14B or 32B sweep until the one-block validation shows:

- hot reusable prefixes are retained rather than accidentally bypassed;
- one-off writes are skipped;
- cached-token loss is eliminated or intentionally explained;
- end-to-end latency moves toward parity or improvement.

