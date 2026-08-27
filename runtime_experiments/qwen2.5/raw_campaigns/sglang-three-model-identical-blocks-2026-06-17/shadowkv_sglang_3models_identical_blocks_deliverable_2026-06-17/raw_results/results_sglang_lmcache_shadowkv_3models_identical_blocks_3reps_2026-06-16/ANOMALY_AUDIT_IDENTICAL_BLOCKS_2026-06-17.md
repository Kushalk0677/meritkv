# Identical-Blocks Run Audit - 2026-06-17

## Completion

- Benchmark JSONs: `270/270`.
- LMCache summaries: `90/90`.
- Aggregate summary exists: `True`.
- Error scan hits in stderr/log files: `0`.

## Schedule Balance

- Workload blocks: `30`.
- Blocks with non-9 job count: `0`.
- Blocks with uneven model representation: `0`.
- Blocks with uneven baseline representation: `0`.

### Model Position Counts

| Model | Position 1 | Position 2 | Position 3 |
| --- | ---: | ---: | ---: |
| `qwen25_15b` | 30 | 30 | 30 |
| `qwen25_3b` | 30 | 30 | 30 |
| `qwen25_7b` | 30 | 30 | 30 |

### Baseline Position Counts

| Baseline | Position 1 | Position 2 | Position 3 |
| --- | ---: | ---: | ---: |
| `lmcache_no_native_radix` | 30 | 30 | 30 |
| `sglang_radix_attention` | 30 | 30 | 30 |
| `sglang_radix_attention_shadowkv_plus` | 30 | 30 | 30 |

## Runtime Consistency

- Native Radix / ShadowKV++ server-arg mismatches: `0`.
- LMCache cells use the generic no-native-Radix launcher and `lmcache_engine=sglang`.

## Cache And Admission Checks

- ShadowKV++ vs native Radix cached-token mismatches: `0/90`.
- ShadowKV++ admission tuple counts `(plans, allows, bypasses, stores)`:
  - `(256, 255, 1, 256)`: `90` cells
- LMCache no-native-Radix cells with zero retrieve events: `27/90`.
- Zero-retrieve LMCache cells by dataset:
  - `ag_news`: `18`
  - `daily_dialog`: `6`
  - `dolly`: `3`

## Interpretation

The rerun completed the same 270-cell matrix under matched workload blocks. The prior model-major ordering caveat is removed: each rep/dataset/mode block contains all three models and all three baselines with balanced position permutations. ShadowKV++ cached-token totals still match native Radix exactly, so it remains an admission/policy overlay measurement rather than a distinct KV-hit mechanism. LMCache no-native-Radix remains functional but lower-reuse because the public workloads often do not cross the 256-token chunk threshold for useful host-cache retrieval.
