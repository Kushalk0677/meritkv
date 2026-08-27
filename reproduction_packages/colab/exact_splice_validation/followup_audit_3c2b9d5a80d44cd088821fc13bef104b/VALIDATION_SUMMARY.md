# MeritKV Exact-Splice Validation Summary

- Reports: 6
- All strict pass: `False`
- All token pass: `False`
- All token-semantic pass: `False`

| Report | Attention | Cases | Semantic | Strict | Cache | Native tokens | Splice tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| `gemma2b_t4_f16_sdpa_natural.json` | sdpa | 8 | 8 | 0 | 1 | 8 | 8 |
| `gpt2_t4_f16_sdpa_natural.json` | sdpa | 8 | 7 | 0 | 1 | 7 | 7 |
| `gpt2_t4_f32_sdpa_random.json` | sdpa | 8 | 8 | 4 | 4 | 8 | 8 |
| `phi3mini_t4_f16_sdpa_natural.json` | sdpa | 8 | 7 | 0 | 1 | 7 | 8 |
| `qwen25_15b_t4_f16_sdpa_natural.json` | sdpa | 8 | 8 | 0 | 1 | 8 | 8 |
| `tinyllama_t4_f16_sdpa_natural.json` | sdpa | 8 | 7 | 0 | 2 | 8 | 7 |

## Maximum Absolute Differences

### `gemma2b_t4_f16_sdpa_natural.json`

- prefix_cache: `0.015625`
- suffix_native_vs_full: `0.109375`
- suffix_splice_vs_full: `0.125`
- suffix_splice_vs_native: `0.09375`
- generation_native_vs_full: `0.109375`
- generation_splice_vs_full: `0.125`
- generation_splice_vs_native: `0.125`

### `gpt2_t4_f16_sdpa_natural.json`

- prefix_cache: `0.046875`
- suffix_native_vs_full: `0.625`
- suffix_splice_vs_full: `0.3125`
- suffix_splice_vs_native: `0.375`
- generation_native_vs_full: `1.0625`
- generation_splice_vs_full: `2.25`
- generation_splice_vs_native: `1.1875`

### `gpt2_t4_f32_sdpa_random.json`

- prefix_cache: `3.4749507904052734e-05`
- suffix_native_vs_full: `0.0002288818359375`
- suffix_splice_vs_full: `0.000244140625`
- suffix_splice_vs_native: `0.00011444091796875`
- generation_native_vs_full: `0.00020599365234375`
- generation_splice_vs_full: `0.000213623046875`
- generation_splice_vs_native: `6.866455078125e-05`

### `phi3mini_t4_f16_sdpa_natural.json`

- prefix_cache: `0.07470703125`
- suffix_native_vs_full: `0.19140625`
- suffix_splice_vs_full: `0.06640625`
- suffix_splice_vs_native: `0.16796875`
- generation_native_vs_full: `0.140625`
- generation_splice_vs_full: `0.14453125`
- generation_splice_vs_native: `0.0986328125`

### `qwen25_15b_t4_f16_sdpa_natural.json`

- prefix_cache: `0.15625`
- suffix_native_vs_full: `0.04248046875`
- suffix_splice_vs_full: `0.0361328125`
- suffix_splice_vs_native: `0.046875`
- generation_native_vs_full: `0.0703125`
- generation_splice_vs_full: `0.078125`
- generation_splice_vs_native: `0.126953125`

### `tinyllama_t4_f16_sdpa_natural.json`

- prefix_cache: `0.03125`
- suffix_native_vs_full: `0.0302734375`
- suffix_splice_vs_full: `0.02734375`
- suffix_splice_vs_native: `0.03125`
- generation_native_vs_full: `0.0625`
- generation_splice_vs_full: `0.0625`
- generation_splice_vs_native: `0.048828125`

