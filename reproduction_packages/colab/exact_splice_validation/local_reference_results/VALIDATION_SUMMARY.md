# Tiny-Model Harness Reference Validation Summary

- Reports: 3
- All strict pass: `True`
- All token pass: `True`

| Report | Attention | Cases | Strict | Cache | Native tokens | Splice tokens |
|---|---|---:|---:|---:|---:|---:|
| `tiny_llama_cpu_f32.json` | eager | 9 | 9 | 9 | 9 | 9 |
| `tiny_qwen2_cpu_bf16.json` | eager | 9 | 9 | 9 | 9 | 9 |
| `tiny_qwen2_cpu_f32.json` | eager | 4 | 4 | 4 | 4 | 4 |

## Maximum Absolute Differences

### `tiny_llama_cpu_f32.json`

- prefix_cache: `0.0`
- suffix_native_vs_full: `0.0`
- suffix_splice_vs_full: `0.0`
- suffix_splice_vs_native: `0.0`
- generation_native_vs_full: `2.086162567138672e-07`
- generation_splice_vs_full: `2.086162567138672e-07`
- generation_splice_vs_native: `0.0`

### `tiny_qwen2_cpu_bf16.json`

- prefix_cache: `0.0`
- suffix_native_vs_full: `0.0`
- suffix_splice_vs_full: `0.0`
- suffix_splice_vs_native: `0.0`
- generation_native_vs_full: `0.001953125`
- generation_splice_vs_full: `0.001953125`
- generation_splice_vs_native: `0.0`

### `tiny_qwen2_cpu_f32.json`

- prefix_cache: `0.0`
- suffix_native_vs_full: `0.0`
- suffix_splice_vs_full: `0.0`
- suffix_splice_vs_native: `0.0`
- generation_native_vs_full: `2.384185791015625e-07`
- generation_splice_vs_full: `2.384185791015625e-07`
- generation_splice_vs_native: `0.0`

