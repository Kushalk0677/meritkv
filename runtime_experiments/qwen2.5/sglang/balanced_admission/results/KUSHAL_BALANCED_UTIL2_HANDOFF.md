# Qwen2.5-1.5B Balanced Admission Rerun Handoff

## Bottom Line

Kushal's requested balanced-admission rerun completed on Blackwell with `admission_preset=balanced`, `policy.utility.util_min_ms=2.0`, and `min_bootstrap_admissions=0` verified in the output CSVs. The strongest effect is SGLang semantic AG News: ShadowKV++ skipped 254/256 cache lookups, reducing attempted cache-query bytes from about 957.8 MB to 6.8 MB. That does show the intended waste-avoidance behavior for the pathological semantic case.

The caveat is latency: the same SGLang semantic ShadowKV++ cell got slower, with mean/P95 moving from native Radix 10.895/12.932 ms to ShadowKV++ 20.233/23.641 ms. So this run supports a waste-reduction claim for native SGLang admission, not a latency-improvement claim.

vLLM remains a write-through overlay in this harness, so its bypass decisions are observed but not enforceable as native skip-lookups. As expected, vLLM APC+ShadowKV++ query bytes and miss-waste ratios remain identical to native vLLM APC.

## Instrumentation Note

For SGLang native admission, `skip_lookup=True` means no cache lookup/transfer was attempted. I corrected the waste accounting so those bypass rows contribute zero cache-query and zero miss-waste bytes. Latency and speedup fields were not changed. See `POSTPROCESS_SKIP_LOOKUP_ACCOUNTING.md`.

## Balanced Rerun Results

| Workload | Engine | Mean ms | P95 ms | Speedup<1 | Query MB | Miss Waste | Failure Waste | Bypass | Skip Lookup | Skip Write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pathological_semantic_ag_news | vllm_apc | 14.647 | 17.022 | 145 | 957.79 | 0.5153 | 0.5590 | 0 | 0 | 0 |
| pathological_semantic_ag_news | vllm_apc_shadowkv_plus | 14.953 | 16.905 | 153 | 957.79 | 0.5153 | 0.5902 | 7 | 0 | 0 |
| pathological_semantic_ag_news | sglang_radix_attention | 10.895 | 12.932 | 106 | 957.79 | 0.4673 | 0.4125 | 0 | 0 | 0 |
| pathological_semantic_ag_news | sglang_radix_attention_shadowkv_plus | 20.233 | 23.641 | 255 | 6.77 | 0.4025 | 0.5169 | 254 | 254 | 250 |
| pathological_templated_ag_news | vllm_apc | 14.291 | 16.146 | 81 | 1308.27 | 0.3737 | 0.3130 | 0 | 0 | 0 |
| pathological_templated_ag_news | vllm_apc_shadowkv_plus | 15.092 | 17.146 | 158 | 1308.27 | 0.3737 | 0.6174 | 1 | 0 | 0 |
| pathological_templated_ag_news | sglang_radix_attention | 11.321 | 13.455 | 133 | 1308.27 | 0.3338 | 0.5173 | 0 | 0 | 0 |
| pathological_templated_ag_news | sglang_radix_attention_shadowkv_plus | 11.600 | 13.684 | 143 | 1303.11 | 0.3312 | 0.5534 | 1 | 1 | 0 |
| control_templated_samsum | vllm_apc | 15.153 | 17.048 | 88 | 2040.27 | 0.5359 | 0.3268 | 0 | 0 | 0 |
| control_templated_samsum | vllm_apc_shadowkv_plus | 15.634 | 18.611 | 122 | 2040.27 | 0.5359 | 0.4743 | 1 | 0 | 0 |
| control_templated_samsum | sglang_radix_attention | 11.601 | 14.095 | 95 | 2040.27 | 0.4909 | 0.3416 | 0 | 0 | 0 |
| control_templated_samsum | sglang_radix_attention_shadowkv_plus | 12.484 | 15.013 | 155 | 2026.97 | 0.4876 | 0.5694 | 1 | 1 | 0 |

## Comparison Against Previous Low-Latency Run

| Workload | Engine | Query MB Balanced | Query MB Low-Latency | Query Delta | Bypass Balanced | Bypass Low-Latency | Mean ms Balanced | Mean ms Low-Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pathological_semantic_ag_news | vllm_apc | 957.79 | 957.79 | 0.0% | 0 | 0 | 14.647 | 14.294 |
| pathological_semantic_ag_news | vllm_apc_shadowkv_plus | 957.79 | 957.79 | 0.0% | 7 | 4 | 14.953 | 14.979 |
| pathological_semantic_ag_news | sglang_radix_attention | 957.79 | 957.79 | 0.0% | 0 | 0 | 10.895 | 11.086 |
| pathological_semantic_ag_news | sglang_radix_attention_shadowkv_plus | 6.77 | 957.79 | -99.3% | 254 | 4 | 20.233 | 11.538 |
| pathological_templated_ag_news | vllm_apc | 1308.27 | 1308.27 | 0.0% | 0 | 0 | 14.291 | 14.497 |
| pathological_templated_ag_news | vllm_apc_shadowkv_plus | 1308.27 | 1308.27 | 0.0% | 1 | 1 | 15.092 | 14.890 |
| pathological_templated_ag_news | sglang_radix_attention | 1308.27 | 1308.27 | 0.0% | 0 | 0 | 11.321 | 10.996 |
| pathological_templated_ag_news | sglang_radix_attention_shadowkv_plus | 1303.11 | 1308.27 | -0.4% | 1 | 1 | 11.600 | 12.014 |
| control_templated_samsum | vllm_apc | 2040.27 | 2040.27 | 0.0% | 0 | 0 | 15.153 | 15.139 |
| control_templated_samsum | vllm_apc_shadowkv_plus | 2040.27 | 2040.27 | 0.0% | 1 | 1 | 15.634 | 15.398 |
| control_templated_samsum | sglang_radix_attention | 2040.27 | 2040.27 | 0.0% | 0 | 0 | 11.601 | 11.502 |
| control_templated_samsum | sglang_radix_attention_shadowkv_plus | 2026.97 | 2040.27 | -0.7% | 1 | 1 | 12.484 | 11.984 |

## Readout

- Pathological semantic AG News: the balanced policy materially changes SGLang behavior and nearly eliminates attempted cache lookups, but at clear latency cost.
- Pathological templated AG News: balanced policy only bypassed 1/256 requests, so it did not materially change waste or latency.
- SAMSum control: balanced policy only bypassed 1/256 requests, so it converges with native Radix/APC as expected.
- vLLM APC+ShadowKV++ still cannot prove runtime waste reduction until vLLM has a native per-request skip-lookup/write hook; the overlay can identify bypass decisions but cannot enforce them inside APC.

## Verification

- `summary.csv` and `summary_compact.csv`: 18 data rows each.
- `per_request/`: 18 JSONL files, each with exactly 256 request records.
- first-light production `qwen36-27b-fp8-vllm` was restored and `/v1/models` responded on `127.0.0.1:8014`.
- Smoke test before the full run confirmed `balanced/util=2/bootstrap=0` produced native SGLang skip-lookups.

## Files

- `summary_compact.csv`: reviewer-facing summary.
- `comparison_vs_low_latency_2026-07-03.csv`: balanced rerun compared with the previous low-latency run.
- `per_request/<workload>/<engine>.jsonl`: request-level records with corrected skip-lookup waste accounting.
- `POSTPROCESS_SKIP_LOOKUP_ACCOUNTING.md`: accounting correction note.
- `metadata/`: host/GPU/runtime/source snapshots.
