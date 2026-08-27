# MeritKV Exact-Splice Validation Summary

- Reports: 5
- All strict pass: `False`
- All token pass: `False`
- All token-semantic pass: `False`

| Report | Attention | Cases | Semantic | Strict | Cache | Native tokens | Splice tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| `gemma2b_t4_f16_sdpa.json` | sdpa | 8 | 8 | 0 | 4 | 8 | 8 |
| `gpt2_t4_f16_sdpa.json` | sdpa | 8 | 5 | 0 | 0 | 6 | 6 |
| `phi3mini_t4_f16_sdpa.json` | sdpa | 8 | 7 | 0 | 4 | 7 | 7 |
| `qwen25_15b_t4_f16_sdpa.json` | sdpa | 8 | 8 | 0 | 4 | 8 | 8 |
| `tinyllama_t4_f16_sdpa.json` | sdpa | 8 | 8 | 0 | 4 | 8 | 8 |

## Maximum Absolute Differences

### `gemma2b_t4_f16_sdpa.json`

- prefix_cache: `0.03948974609375`
- suffix_native_vs_full: `0.2109375`
- suffix_splice_vs_full: `0.203125`
- suffix_splice_vs_native: `0.1875`
- generation_native_vs_full: `0.140625`
- generation_splice_vs_full: `0.125`
- generation_splice_vs_native: `0.15625`

### `gpt2_t4_f16_sdpa.json`

- prefix_cache: `0.0546875`
- suffix_native_vs_full: `0.25`
- suffix_splice_vs_full: `0.1875`
- suffix_splice_vs_native: `0.25`
- generation_native_vs_full: `0.125`
- generation_splice_vs_full: `0.125`
- generation_splice_vs_native: `0.125`

### `phi3mini_t4_f16_sdpa.json`

- prefix_cache: `0.046875`
- suffix_native_vs_full: `0.125`
- suffix_splice_vs_full: `0.09375`
- suffix_splice_vs_native: `0.1171875`
- generation_native_vs_full: `0.0703125`
- generation_splice_vs_full: `0.0703125`
- generation_splice_vs_native: `0.05078125`

### `qwen25_15b_t4_f16_sdpa.json`

- prefix_cache: `0.125`
- suffix_native_vs_full: `0.109375`
- suffix_splice_vs_full: `0.109375`
- suffix_splice_vs_native: `0.060546875`
- generation_native_vs_full: `0.35546875`
- generation_splice_vs_full: `0.353515625`
- generation_splice_vs_native: `0.028228759765625`

### `tinyllama_t4_f16_sdpa.json`

- prefix_cache: `0.03125`
- suffix_native_vs_full: `0.0478515625`
- suffix_splice_vs_full: `0.037109375`
- suffix_splice_vs_native: `0.03515625`
- generation_native_vs_full: `0.03125`
- generation_splice_vs_full: `0.03125`
- generation_splice_vs_native: `0.01953125`

