# Qwen2.5-1.5B Balanced Admission Rerun

This rerun repeats the reviewer-gap three-workload experiment with `admission_preset=balanced`, `policy.utility.util_min_ms=2.0`, and `min_bootstrap_admissions=0`.

SGLang native `skip_lookup` bypass rows are accounted as zero cache-query/waste bytes because no cache lookup or transfer was attempted. Latency and speedup fields are unchanged by this accounting correction.

| Workload | Engine | Mean ms | P95 ms | Mean speedup | Speedup<1 | Miss waste | Failure waste | Allow | Bypass | Skip lookup | Skip write | Adjusted rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_templated_samsum | sglang_radix_attention | 11.601 | 14.095 | 1.030 | 95 | 0.4909 | 0.3416 | 0 | 0 | 0 | 0 | 0 |
| control_templated_samsum | sglang_radix_attention_shadowkv_plus | 12.484 | 15.013 | 0.960 | 155 | 0.4876 | 0.5694 | 255 | 1 | 1 | 0 | 1 |
| control_templated_samsum | vllm_apc | 15.153 | 17.048 | 1.042 | 88 | 0.5359 | 0.3268 | 0 | 0 | 0 | 0 | 0 |
| control_templated_samsum | vllm_apc_shadowkv_plus | 15.634 | 18.611 | 1.013 | 122 | 0.5359 | 0.4743 | 255 | 1 | 0 | 0 | 0 |
| pathological_semantic_ag_news | sglang_radix_attention | 10.895 | 12.932 | 1.029 | 106 | 0.4673 | 0.4125 | 0 | 0 | 0 | 0 | 0 |
| pathological_semantic_ag_news | sglang_radix_attention_shadowkv_plus | 20.233 | 23.641 | 0.559 | 255 | 0.4025 | 0.5169 | 2 | 254 | 254 | 250 | 254 |
| pathological_semantic_ag_news | vllm_apc | 14.647 | 17.022 | 1.004 | 145 | 0.5153 | 0.5590 | 0 | 0 | 0 | 0 | 0 |
| pathological_semantic_ag_news | vllm_apc_shadowkv_plus | 14.953 | 16.905 | 0.983 | 153 | 0.5153 | 0.5902 | 249 | 7 | 0 | 0 | 0 |
| pathological_templated_ag_news | sglang_radix_attention | 11.321 | 13.455 | 1.008 | 133 | 0.3338 | 0.5173 | 0 | 0 | 0 | 0 | 0 |
| pathological_templated_ag_news | sglang_radix_attention_shadowkv_plus | 11.600 | 13.684 | 0.984 | 143 | 0.3312 | 0.5534 | 255 | 1 | 1 | 0 | 1 |
| pathological_templated_ag_news | vllm_apc | 14.291 | 16.146 | 1.036 | 81 | 0.3737 | 0.3130 | 0 | 0 | 0 | 0 | 0 |
| pathological_templated_ag_news | vllm_apc_shadowkv_plus | 15.092 | 17.146 | 0.982 | 158 | 0.3737 | 0.6174 | 255 | 1 | 0 | 0 | 0 |

## Postprocess

- Adjusted skip-lookup rows: `256` across `3` per-request files.
- See `POSTPROCESS_SKIP_LOOKUP_ACCOUNTING.md` for the accounting note.
