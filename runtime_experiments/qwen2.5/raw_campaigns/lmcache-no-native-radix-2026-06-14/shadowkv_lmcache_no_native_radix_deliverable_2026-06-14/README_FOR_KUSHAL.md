# ShadowKV / SGLang / LMCache Deliverable - 2026-06-14

Prepared by Keystone on IronGod for Kushal.

## Headline

We patched the SGLang + LMCache runtime so `--enable-lmcache --disable-radix-cache` becomes a real LMCache-without-native-Radix baseline instead of falling back to SGLang `ChunkCache`.

The patched smoke test passed: with no manual `/flush_cache`, repeated long-prefix requests reported `cached_tokens=2560`, and LMCache logged one `Stored 2560 tokens` event plus two `Retrieved 2560 tokens` events.

The full Qwen2.5-14B matrix completed across 5 datasets x 2 prompt modes.

## Main Result

| Baseline | Cells | Mean latency ms | P95 latency ms | Throughput rps | Idle-adjusted J/request | Prompt tokens | Cached tokens | LMCache retrieves | LMCache stores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LMCache without native Radix | 10 | 51.56 | 62.97 | 19.61 | 23.74 | 167304 | 7168 | 27 | 242 |

For comparison, the 2026-06-13 native SGLang Radix aggregate on the same Qwen14B workload was:

| Baseline | Mean latency ms | P95 latency ms | Throughput rps | Cached tokens | Idle-adjusted J/request |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native SGLang Radix | 45.25 | 50.12 | 22.25 | 86542 | 18.69 |
| SGLang Radix + ShadowKV++ overlay | 44.44 | 48.55 | 22.64 | 86542 | 18.70 |
| LMCache without native Radix | 51.56 | 62.97 | 19.61 | 7168 | 23.74 |

Interpretation: LMCache works once isolated from native Radix, but this workload loses most of Radix's fine-grained short-prefix reuse because LMCache reuse is chunk-aligned at 256 tokens. The result is useful as a true baseline, but it is not faster than native SGLang Radix on this public-dataset Qwen14B setup.

## Runtime Patch

Patched image on first-light:

`shadowkv-sglang-lmcache:2026-06-14-no-native-radix`

Version pair:

- `sglang==0.5.13`
- `lmcache==0.4.7`

Patch behavior:

- `registry.py` constructs `LMCRadixCache` when LMCache is enabled, even if `disable_radix_cache=True`.
- `lmc_radix_cache.py` treats `disable_radix_cache=True` as no-native-Radix mode:
  - starts prefix matching from an empty device-tree match,
  - stores finished-request KV into LMCache before freeing request KV slots,
  - evicts temporary host-loaded device nodes so later hits must come from LMCache rather than resident native Radix state.

## Included Files

- `report/ShadowKV_SGLang_LMCache_Report_2026-06-14.md` - Obsidian project report with full narrative and tables.
- `raw_results/results_lmcache_no_native_radix_qwen14b_smoke_2026-06-14/` - controlled smoke logs and response JSON.
- `raw_results/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14/` - raw 5x2 no-native-Radix LMCache matrix results.
- `raw_results/results_sglang_shadowkv_qwen14b_small_2026-06-13/` - comparison raw results for native SGLang Radix and ShadowKV++ overlay.
- `session_files/` - Dockerfile, run scripts, launcher, and LMCache YAML configs.
- `run_logs/` - first-light top-level run logs and status files.
- `verification/` - service restore check, relevant container state, patched image inspect output, and runtime static check.
- `checksums/SHA256SUMS.txt` - SHA-256 checksums for package files.
- `FILE_MANIFEST.txt` - full file list.

## Raw Result Entry Points

No-native-Radix LMCache summary:

`raw_results/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14/summary_lmcache_no_native_radix_qwen14b_2026-06-14.md`

CSV:

`raw_results/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14/summary_lmcache_no_native_radix_qwen14b_2026-06-14.csv`

JSON:

`raw_results/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14/summary_lmcache_no_native_radix_qwen14b_2026-06-14.json`

Native Radix / ShadowKV++ comparison summary:

`raw_results/results_sglang_shadowkv_qwen14b_small_2026-06-13/summary_sglang_shadowkv_qwen14b_small_2026-06-13.md`

## Host Restore

After the experiment, first-light production service `darwin28b-reason-vllm` was restored.

Verification in this package:

- `verification/first_light_darwin28b_models_response.json`
- `verification/first_light_relevant_containers_after_run.txt`

At packaging time, `qwen36-27b-fp8-vllm` was stopped-retained and `darwin28b-reason-vllm` was serving `darwin28b-reason` on `127.0.0.1:8015`.
