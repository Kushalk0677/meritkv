# Qwen14B Identical-Blocks Run Audit - 2026-06-18

## Completion

- Benchmark JSONs: `90/90`.
- LMCache summaries: `30/30`.
- Aggregate summary exists: `True`.
- Error scan hits in stderr/log files: `0`.

## Schedule Balance

- Workload blocks: `30`.
- Blocks with non-3 job count: `0`.
- Blocks with uneven baseline representation: `0`.

### Baseline Position Counts

| Baseline | Position 1 | Position 2 | Position 3 |
| --- | ---: | ---: | ---: |
| `lmcache_no_native_radix` | 10 | 10 | 10 |
| `sglang_radix_attention` | 10 | 10 | 10 |
| `sglang_radix_attention_shadowkv_plus` | 10 | 10 | 10 |

## Runtime Consistency

- Native Radix / ShadowKV++ server-arg mismatches: `0`.
- LMCache cells use the generic no-native-Radix launcher and `lmcache_engine=sglang`.

## Cache And Admission Checks

- ShadowKV++ vs native Radix cached-token mismatches: `0/30`.
- ShadowKV++ admission tuple counts `(plans, allows, bypasses, stores)`:
  - `(256, 255, 1, 256)`: `30` cells
- LMCache no-native-Radix cells with zero retrieve events: `8/30`.
- Zero-retrieve LMCache cells by dataset:
  - `ag_news`: `6`
  - `daily_dialog`: `1`
  - `dolly`: `1`

## Interpretation

The Qwen14B matched-block rerun completed the 90-cell matrix cleanly. The schedule removes baseline-order imbalance: every rep/dataset/mode block includes all three baselines, and baseline positions are balanced. ShadowKV++ cached-token totals still match native Radix exactly, so the result remains an admission/policy overlay measurement. LMCache no-native-Radix is functional but lower-reuse because host-cache retrieval depends on 256-token chunks.
