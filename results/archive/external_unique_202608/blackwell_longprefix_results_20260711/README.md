# Blackwell Long-Prefix HF Results, 2026-07-11

This folder contains a public combined view plus four Blackwell long-prefix result bundles:

- `public_combined_all_models/`: public-facing combined folder with all model runs placed together as if they came from one package
- `qwen2_5_14b_longprefix/`: `ShadowKV_Qwen2.5-14B_LongPrefix_Blackwell_20260711.zip`
- `qwen2_5_7b_gemma4_12b_longprefix/`: `ShadowKV_Qwen7B_Gemma4-12B_LongPrefix_Blackwell_20260711.zip`
- `qwen2_5_32b_gemma4_31b_26b_a4b_longprefix/`: `ShadowKV_Qwen32B_Gemma4-31B_Gemma4-26B-A4B_Blackwell_20260711.zip`
- `hf_blackwell_7model_longprefix_gemma4_e2b_20260713/`: `ShadowKV_HF_Blackwell_7Model_LongPrefix_Gemma4-E2B_20260713.zip`

These are result bundles, not just summaries. Each bundle contains raw benchmark JSONs, policy traces, logs, metadata, source snapshots, aggregation scripts, and machine-readable audits.

For public reading or sharing, start with `public_combined_all_models/README.md`. The other folders preserve the original imported bundle boundaries.

## What Was Run

- Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Backend: Hugging Face
- Mode: semantic workload generation with a common long scaffold
- Engines: `no_cache`, `shadow_kv`, `shadow_kv_plus`
- Requests per cell: 128
- Seed: 42
- Datasets: AG News, AlpacaEval, Banking77, CNN/DailyMail, DailyDialog, Dolly, OASST1, SAMSum, UltraChat, XSum
- Shared-prefix repeats: 4
- Cached prefix cap: 128 tokens
- Energy measurement: NVML total and idle-adjusted energy
- Isolation: one Python subprocess per model/dataset/mode/engine cell

## Key Interpretation

These runs validate reliable exact long-scaffold KV reuse. They do not validate approximate semantic-partial reuse.

In every checked `shadow_kv_plus` cell:

```text
reuse_successes = 127 / 128
reused_prefix_tokens_total = 16,256
hit_rate = 0.9921875
policy_exact_total = 127
policy_semantic_partial_total = 0
semantic_partial_hits = 0
wasted_compute_ratio = 0
```

So the correct wording is:

> The Blackwell long-prefix runs validate reliable exact-scaffold KV reuse under HF for Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B, Phi-3 Mini 4K, TinyLlama 1.1B, GPT-2, Gemma 4 E2B, Gemma 4 12B, Gemma 4 31B, and Gemma 4 26B-A4B. They do not validate approximate semantic-partial reuse, because no semantic-partial reuse path executed.

## Aggregate Results

| Model | Engine | Mean Speedup vs No Cache | Worst Dataset Mean Speedup | Best Dataset Mean Speedup | P95 Speedup Mean | GPU Energy Reduction |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | `shadow_kv` | 1.000x | 0.998x | 1.002x | 1.000x | -0.5% |
| Qwen2.5-7B-Instruct | `shadow_kv_plus` | 1.019x | 0.992x | 1.040x | 1.011x | 6.9% |
| Qwen2.5-14B-Instruct | `shadow_kv` | 1.000x | 0.997x | 1.002x | 1.000x | -0.1% |
| Qwen2.5-14B-Instruct | `shadow_kv_plus` | 1.046x | 0.987x | 1.087x | 1.009x | 8.6% |
| Qwen2.5-32B-Instruct | `shadow_kv` | 0.998x | 0.994x | 1.001x | 0.998x | -0.2% |
| Qwen2.5-32B-Instruct | `shadow_kv_plus` | 1.109x | 1.057x | 1.135x | 1.094x | 11.1% |
| Gemma 4 12B IT | `shadow_kv` | 1.000x | 0.998x | 1.001x | 1.000x | -0.4% |
| Gemma 4 12B IT | `shadow_kv_plus` | 1.433x | 1.073x | 2.331x | 1.539x | 28.6% |
| Gemma 4 26B-A4B IT | `shadow_kv` | 1.000x | 0.996x | 1.003x | 1.001x | -0.1% |
| Gemma 4 26B-A4B IT | `shadow_kv_plus` | 1.204x | 1.043x | 1.595x | 1.233x | 20.6% |
| Gemma 4 31B IT | `shadow_kv` | 0.998x | 0.994x | 1.000x | 0.998x | -0.5% |
| Gemma 4 31B IT | `shadow_kv_plus` | 1.561x | 1.139x | 2.703x | 1.720x | 30.4% |

## Seven-Model Break-Even Sweep

The 2026-07-13 bundle is a separate seven-model execution block. It uses the same long-prefix HF setup and includes small models plus Gemma 4 E2B. It should be read as a break-even calibration result rather than merged with the earlier large-model table above.

| Model | Engine | Mean Speedup vs No Cache | Worst Dataset Mean Speedup | Best Dataset Mean Speedup | P95 Speedup Mean | GPU Energy Reduction |
|---|---|---:|---:|---:|---:|---:|
| GPT-2 | `shadow_kv` | 1.002x | 0.897x | 1.093x | 1.002x | -1.3% |
| GPT-2 | `shadow_kv_plus` | 0.762x | 0.704x | 0.855x | 0.818x | 0.1% |
| TinyLlama 1.1B Chat | `shadow_kv` | 1.001x | 0.989x | 1.018x | 1.001x | -1.9% |
| TinyLlama 1.1B Chat | `shadow_kv_plus` | 0.836x | 0.792x | 0.876x | 0.852x | -6.2% |
| Qwen2.5-1.5B-Instruct | `shadow_kv` | 1.007x | 0.991x | 1.015x | 1.001x | -1.2% |
| Qwen2.5-1.5B-Instruct | `shadow_kv_plus` | 0.855x | 0.814x | 0.882x | 0.866x | -0.2% |
| Qwen2.5-3B-Instruct | `shadow_kv` | 1.001x | 0.997x | 1.006x | 1.000x | 0.4% |
| Qwen2.5-3B-Instruct | `shadow_kv_plus` | 0.959x | 0.838x | 0.998x | 0.938x | 4.9% |
| Phi-3 Mini 4K Instruct | `shadow_kv` | 1.004x | 0.997x | 1.043x | 1.004x | -0.6% |
| Phi-3 Mini 4K Instruct | `shadow_kv_plus` | 0.913x | 0.863x | 0.950x | 0.919x | 0.2% |
| Gemma 4 E2B IT | `shadow_kv` | 1.000x | 0.995x | 1.003x | 0.998x | -0.9% |
| Gemma 4 E2B IT | `shadow_kv_plus` | 1.277x | 1.091x | 1.684x | 1.289x | 24.7% |
| Gemma 4 12B IT | `shadow_kv` | 0.999x | 0.998x | 1.003x | 0.998x | -0.5% |
| Gemma 4 12B IT | `shadow_kv_plus` | 1.430x | 1.069x | 2.333x | 1.532x | 28.4% |

## Result Takeaways

`shadow_kv` stayed at parity because it did not reuse in this exact-scaffold setup.

`shadow_kv_plus` successfully reused the exact scaffold in every dataset cell. The benefit is small for Qwen2.5-7B, moderate for Qwen2.5-14B, stronger and more uniform for Qwen2.5-32B, and very strong for the Gemma 4 large models.

The seven-model 2026-07-13 sweep adds an important break-even point: exact reuse is correct in all checked cells, but it is not automatically faster. `shadow_kv_plus` is slower for GPT-2, TinyLlama, Qwen2.5-1.5B, Qwen2.5-3B, and Phi-3 Mini because the fixed reuse overhead dominates avoided prefill compute. Gemma 4 E2B and Gemma 4 12B are positive in that sweep.

Qwen2.5-14B has a positive mean result, but not a uniform tail result. CNN/DailyMail is below parity on mean latency, and several Qwen2.5-14B P95 cells are below parity.

Qwen2.5-32B is above parity on mean and P95 in every dataset, though the weakest P95 cells are close enough to parity that they should not be overclaimed from one run.

Gemma 4 31B is the strongest result. Gemma 4 26B-A4B is also strong, but smaller than dense 31B, consistent with the A4B model activating fewer experts per token.

## Included Files

Each result subfolder includes:

- `REPORT_FOR_KUSHAL.md`: human-readable result report and interpretation
- `aggregate_summary.csv`: model/engine aggregate metrics
- `dataset_results.csv`: dataset-level paired comparisons
- `anomaly_audit.json`: machine-readable completeness and reuse-path checks
- `analyze_results.py`: script used to regenerate summaries
- `MANIFEST_SHA256.txt`: file hash manifest
- `raw_results/`: raw benchmark outputs, policy traces, per-cell summaries, logs, and metadata
- `run_logs/`: top-level sweep logs
- `source_snapshot/`: source snapshot used for the run

The Qwen2.5-14B bundle also includes:

- `smoke_results/`: 16-request compatibility and reuse-path validation

The Qwen2.5-32B/Gemma 4 31B/Gemma 4 26B-A4B bundle includes:

- `COMBINED_SUMMARY_FOR_KUSHAL.md`: cross-model summary for the three larger models
- `Qwen2.5-32B/`: complete Qwen32B report, raw results, smoke, logs, source, audit, and checksums
- `Gemma4-31B-and-26B-A4B/`: complete two-model Gemma report, raw results, smoke, logs, source, audit, and checksums

The Qwen2.5-7B/Gemma 4 12B bundle does not include `smoke_results/`; its README does not claim that it does.

The seven-model bundle includes:

- `REPORT_FOR_KUSHAL.md`: seven-model break-even interpretation and caveats
- `model_summary.csv`, `engine_summary.csv`, `aggregate_summary.csv`, and `dataset_results.csv`
- `anomaly_audit.json`: machine-readable completeness and exact-reuse audit
- `raw_results/`: all 210 raw benchmark cells, aggregate CSVs, policy traces, and metadata
- `smoke_results/`: six-model and Gemma 4 E2B smoke outputs
- `run_logs/`: execution logs and status files
- `source_snapshot/`: benchmark source and launch wrappers

## Files To Read First

Start with:

```text
qwen2_5_14b_longprefix/REPORT_FOR_KUSHAL.md
qwen2_5_7b_gemma4_12b_longprefix/REPORT_FOR_KUSHAL.md
qwen2_5_32b_gemma4_31b_26b_a4b_longprefix/COMBINED_SUMMARY_FOR_KUSHAL.md
qwen2_5_32b_gemma4_31b_26b_a4b_longprefix/Qwen2.5-32B/REPORT_FOR_KUSHAL.md
qwen2_5_32b_gemma4_31b_26b_a4b_longprefix/Gemma4-31B-and-26B-A4B/REPORT_FOR_KUSHAL.md
hf_blackwell_7model_longprefix_gemma4_e2b_20260713/REPORT_FOR_KUSHAL.md
```

Then inspect:

```text
qwen2_5_14b_longprefix/anomaly_audit.json
qwen2_5_7b_gemma4_12b_longprefix/anomaly_audit.json
qwen2_5_32b_gemma4_31b_26b_a4b_longprefix/Qwen2.5-32B/anomaly_audit.json
qwen2_5_32b_gemma4_31b_26b_a4b_longprefix/Gemma4-31B-and-26B-A4B/anomaly_audit.json
hf_blackwell_7model_longprefix_gemma4_e2b_20260713/anomaly_audit.json
```

For raw metrics:

```text
qwen2_5_14b_longprefix/raw_results/all_results.csv
qwen2_5_14b_longprefix/raw_results/comparisons_vs_no_cache.csv
qwen2_5_7b_gemma4_12b_longprefix/raw_results/all_results.csv
qwen2_5_7b_gemma4_12b_longprefix/raw_results/comparisons_vs_no_cache.csv
qwen2_5_32b_gemma4_31b_26b_a4b_longprefix/Qwen2.5-32B/raw_results/all_results.csv
qwen2_5_32b_gemma4_31b_26b_a4b_longprefix/Qwen2.5-32B/raw_results/comparisons_vs_no_cache.csv
qwen2_5_32b_gemma4_31b_26b_a4b_longprefix/Gemma4-31B-and-26B-A4B/raw_results/all_results.csv
qwen2_5_32b_gemma4_31b_26b_a4b_longprefix/Gemma4-31B-and-26B-A4B/raw_results/comparisons_vs_no_cache.csv
hf_blackwell_7model_longprefix_gemma4_e2b_20260713/raw_results/all_results.csv
hf_blackwell_7model_longprefix_gemma4_e2b_20260713/raw_results/comparisons_vs_no_cache.csv
```

## Caveats

These are one-seed, one-run-per-cell results. Engine order was fixed as:

```text
no_cache -> shadow_kv -> shadow_kv_plus
```

Before making a final paper-level performance claim, repeat with randomized engine order and paired confidence intervals.

Do not describe these runs as proof of semantic partial reuse. They are evidence for exact long-scaffold reuse.

Do not average the 2026-07-13 seven-model sweep with the earlier 2026-07-11 bundles without explicitly saying that they are separate execution blocks. The seven-model sweep deliberately includes smaller models to expose the break-even behavior, so its negative small-model rows are part of the result rather than anomalies.
