---
title: Nemotron + ShadowKV Task Queue
created: 2026-05-13
updated: 2026-06-17
author: Codex (IronGod)
authors: [Codex, Keystone]
last_editor: Keystone (Codex on IronGod)
type: project
tags: [task-queue, nemotron, shadowkv, first-light]
sources:
  - nemotron-reasoning-challenge
  - shadowkv-plus
---

# Nemotron + ShadowKV Task Queue

Purpose: durable queue for heartbeat follow-up work while first-light GPU jobs run. Treat this as the current task state file for Nemotron training and ShadowKV continuation.

## Current State

- **Host:** first-light
- **Nemotron project:** `/home/jade_hand/projects/nemotron_pipeline`
- **ShadowKV project:** `/home/jade_hand/research/shadowkv`
- **GPU reservation:** `darwin28b-reason-vllm` is the current production service and was restored on `127.0.0.1:8015` after the 2026-06-14 SGLang LMCache no-native-Radix run. `qwen36-27b-fp8-vllm` is stopped-retained unless explicitly requested.
- **Active Nemotron run:** none as of 2026-05-14 12:22 EDT.
- **Latest queue log:** `/home/jade_hand/projects/nemotron_pipeline/outputs/logs/CDF_unsloth_qlora_b32_ga1_queue_2026-05-14.log`
- **Latest completion ledger:** `/home/jade_hand/projects/nemotron_pipeline/outputs/benchmarks/unsloth_qlora_full_runs_2026-05-14.csv`
- **Last ShadowKV check:** 2026-06-17 02:00 EDT - identical-block three-model SGLang/Radix/ShadowKV++/patched-LMCache matrix completed, 270/270 cells persisted, aggregate/audit artifacts generated, and production `darwin28b-reason-vllm` was restored.
- **Tokenized training path:** `data/tokenized/A_real_heavy_len768` created with 72,750 rows. Smoke adapter saved separately at `outputs/adapters/A_real_heavy_bf16_tokenized_smoke`.

## Queue

| ID | Status | Priority | Task | Success Criteria |
|---|---|---:|---|---|
| NEM-001 | done | 1 | Monitor `A_real_heavy_bf16` full BF16 LoRA run. | Run finishes or fails; final metrics, process status, GPU status, and output path recorded. |
| NEM-002 | done | 1 | Add token length audit for prepared Nemotron recipes. | Length stats after chat template/tokenization reported by recipe and task type: p50/p90/p95/p99/max and percent over 768/1024/1280/1536. |
| NEM-003 | done | 1 | Decide max sequence length from audit instead of assuming 1536. | Recommendation recorded for 768/1024/1280/1536, including truncation risk for hard categories. |
| NEM-004 | done | 1 | Implement CPU-parallel pretokenized dataset path on first-light Ryzen 9 9950X. | Fast tokenizer + batched `datasets.map` + about `num_proc=24`; tokenized Arrow datasets saved under `data/tokenized/<recipe>_len<cap>` with `input_ids`, `attention_mask`, `labels`. |
| NEM-005 | done | 2 | Patch/stage trainer to load tokenized datasets. | Future BF16 runs can use `load_from_disk`, skip `dataset_text_field`, and use a padding collator for token IDs/labels. |
| NEM-006 | done | 2 | Smoke-test tokenized trainer path. | Step time/GPU utilization compared to batch-4 BF16 baseline (~27 sec/step, ~87GB VRAM). |
| SHKV-001 | done | 2 | Continue ShadowKV vLLM APC experiment after Nemotron status and pretokenization work are handled or clearly queued. | Targeted vLLM APC matrix runs or fails with clear logs. |
| SHKV-002 | done | 2 | Run literature-accurate vLLM runtime baseline, not deprecated inline `VLLMBackend`. | Use `literature_accurate_baselines/run_runtime_cache_baseline.py` with `vllm_apc` and `vllm_apc_shadowkv_plus`. |
| SHKV-003 | done | 2 | Write ShadowKV results to wiki. | Dated section in `shadowkv-plus.md` with commands, paths, headline table, and interpretation; brief `log.md` entry appended. |
| SHKV-004 | done | 1 | Patch SGLang/LMCache to run true LMCache without native Radix. | `--enable-lmcache --disable-radix-cache` constructs the LMCache cache path, smoke shows LMCache retrieve without manual Radix flush, and the 5x2 Qwen14B matrix is aggregated. |
| SHKV-005 | done | 1 | Rerun all three SGLang baselines on three smaller Qwen2.5 models with 3 reps and randomized order. | 270/270 jobs completed, aggregates generated, production `darwin28b-reason-vllm` restored, and Kushal deliverable packaged. |
| SHKV-006 | done | 1 | Rerun the three-model SGLang matrix with identical matched blocks. | 30 matched workload blocks completed; 270/270 jobs, 90/90 LMCache summaries, aggregate/audit artifacts, balanced model/baseline positions, production restore, and Kushal deliverable packaged. |
| NEM-007 | done | 1 | Run full `B_synthetic_heavy` Unsloth QLoRA at 32 × 1. | Adapter saved, status 0, final metrics and peak VRAM recorded. |
| NEM-008 | done | 1 | Run full `C_hard_task_specialist`, `D_numeric_specialist`, and `F_format_hardening` Unsloth QLoRA at 32 × 1. | Sequential queue completes all three without failure; final metrics, peak VRAM, and adapter paths recorded. |
| NEM-009 | next | 1 | Evaluate completed adapters and prepare merge candidates. | Adapter scores recorded; M1-M4 or revised merge plan selected from evidence. |
| NEM-010 | conditional | 1 | Rerun `A_real_heavy` under Unsloth QLoRA at 32 × 1 if quick-signal mixed-backend merges look promising or inconsistent. | Same-backend Unsloth A adapter exists and can replace BF16 A in M1-M4 for cleaner final merge/evaluation. |

## Execution Notes

- Do not interrupt or corrupt completed Nemotron adapter outputs.
- Do not stop `darwin28b-reason-vllm` unless GPU work requires it; restore and verify `127.0.0.1:8015/v1/models` afterward. Do not restart `qwen36-27b` unless explicitly requested or required for a specific vLLM benchmark path.
- For Nemotron sequence length, treat `1536` as the current conservative ceiling, not a proven optimum.
- Prefer empirical length audit before lowering sequence length.
- Candidate sequence caps: `768`, `1024`, `1280`, `1536`.
- 2026-05-13 audit result: all prepared recipes are short after chat templating. Worst recipe max is 342 tokens; all candidate caps 768/1024/1280/1536 have 0.0% truncation. Recommendation for current prepared data: use `max_seq_length=768`; consider adding a future `512` candidate if new data stays below this range.

## 2026-05-13 Nemotron Optimization Results

- Final BF16 adapter: `/home/jade_hand/projects/nemotron_pipeline/outputs/adapters/A_real_heavy_bf16`.
- Audit output: `/home/jade_hand/projects/nemotron_pipeline/outputs/audits/token_lengths_2026-05-13/token_length_audit.csv`.
- Trainer files staged: `src/nemotron_winning_pipeline/token_length_audit.py`, `src/nemotron_winning_pipeline/pretokenize_dataset.py`, `src/nemotron_winning_pipeline/train_qlora.py`.
- Pretokenized dataset: `/home/jade_hand/projects/nemotron_pipeline/data/tokenized/A_real_heavy_len768`.
- Smoke output: `/home/jade_hand/projects/nemotron_pipeline/outputs/adapters/A_real_heavy_bf16_tokenized_smoke`.
- Smoke metrics: 2 steps, `train_runtime=60.1408`, `train_samples_per_second=1.064`, `train_steps_per_second=0.033`, `train_loss=25.95376968383789`, `mean_token_accuracy=0.426035113632679`. This validates the loader path but is too short for quality conclusions.

## 2026-05-14 Nemotron Unsloth QLoRA Full Runs

- Backend/config: `configs/blackwell_rtx6000_unsloth_qlora.yaml`, `run_with_cuda_unsloth.sh`, tokenized `*_len768` datasets.
- Batch setting: per-device batch 32, gradient accumulation 1, effective batch 32.
- B ledger: `/home/jade_hand/projects/nemotron_pipeline/outputs/benchmarks/unsloth_qlora_full_runs_2026-05-13.csv`.
- C/D/F ledger: `/home/jade_hand/projects/nemotron_pipeline/outputs/benchmarks/unsloth_qlora_full_runs_2026-05-14.csv`.

| Adapter | Status | Runtime | Samples/sec | Steps/sec | Train loss | Peak VRAM | Output |
|---|---:|---:|---:|---:|---:|---:|---|
| B synthetic heavy | 0 | 9,539s | 8.083 | 0.253 | 0.7414 | 92,146 MiB | `/home/jade_hand/projects/nemotron_pipeline/outputs/adapters/B_synthetic_heavy_unsloth_qlora_b32_ga1` |
| C hard task specialist | 0 | 6,395s | 8.021 | 0.251 | 0.9939 | 90,158 MiB | `/home/jade_hand/projects/nemotron_pipeline/outputs/adapters/C_hard_task_specialist_unsloth_qlora_b32_ga1` |
| D numeric specialist | 0 | 3,834s | 8.962 | 0.280 | 0.4397 | 84,266 MiB | `/home/jade_hand/projects/nemotron_pipeline/outputs/adapters/D_numeric_specialist_unsloth_qlora_b32_ga1` |
| F format hardening | 0 | 9,509s | 7.650 | 0.239 | 0.6420 | 92,444 MiB | `/home/jade_hand/projects/nemotron_pipeline/outputs/adapters/F_format_hardening_unsloth_qlora_b32_ga1` |

Queue result: `QUEUE_END 2026-05-14T12:22:03-04:00 C/D/F complete`. The monitor automation `monitor-nemotron-cdf-queue` was deleted after completion.

## 2026-05-14 Quick-Signal Merge Note

For fastest signal, the first M1-M4 merge attempt may use BF16 `A_real_heavy_bf16` plus Unsloth B/C/D/F via a separate symlink adapter root, leaving canonical adapter directories untouched. This is structurally valid because BF16 A and Unsloth B/C/D/F have identical LoRA tensor keysets and shapes, but it is a mixed-backend merge. If the merge scores are promising, unstable, or hard to interpret, rerun full `A_real_heavy` under Unsloth QLoRA at 32 × 1 and rebuild M1-M4 from a same-backend Unsloth family.

## 2026-05-13 ShadowKV vLLM APC Results

- Runner script: `/home/jade_hand/research/shadowkv/session_files/run_shadowkv_vllm_apc_qwen14b_2026-05-13.sh`.
- Results root: `/home/jade_hand/research/shadowkv/results_vllm_apc_qwen14b_2026-05-13`.
- Scope: Qwen2.5-14B-Instruct, `vllm_apc` and `vllm_apc_shadowkv_plus`, datasets `daily_dialog`, `samsum`, `ag_news`, modes `templated` and `rag`, `n_requests=64`.
- Execution path: one-off `shadowkv-shadowkv` Docker container with `/datapool/cache/huggingface` mounted to `/cache/huggingface`; the long-running `shadowkv` container was left running; `qwen36-27b` remained stopped/restart disabled.
- Result: 12/12 matrix jobs completed. vLLM calls were ~27-30 ms mean/p95 for most jobs, but every JSON reported `cached_tokens_total=0`; the current completion-endpoint metric path does not prove APC prefix hits. ShadowKV++ admission did plan on all 64 requests per job, usually `allow=63`, `bypass=1`, and `store_successes=64`.

## 2026-05-26 vLLM APC Observability Check

- Report: [[shadowkv-vllm-apc-observability-report-2026-05-26]].
- Host: first-light, live `qwen36-27b-fp8-vllm` on `127.0.0.1:8014`.
- Finding: vLLM prefix caching is functioning for exact shared prefixes. A controlled two-request probe produced `15,680` cached prompt tokens on the second request in `/metrics`.
- Important caveat: the OpenAI-compatible response returned `prompt_tokens_details: null`, so the ShadowKV runner's `cached_tokens_total=0` reflects response-level observability failure, not confirmed APC failure.
- Patch status: done. `literature_accurate_baselines/run_runtime_cache_baseline.py` and `adapter_lib.py` now scrape vLLM `/metrics` deltas before and after measured runs and store top-level `vllm_*_delta` metrics.
- Smoke status: passed. Positive long-prefix smoke recorded `vllm_prefix_cache_hits_delta=21,952` for `vllm_apc` and `43,904` for `vllm_apc_shadowkv_plus`, while response `cached_tokens_total` remained `0`.
- Extended run: complete on full-attention `Qwen/Qwen2.5-14B-Instruct` after Evan clarified the Qwen3.6/GDN path is not the right ShadowKV target.
- Result root: `/home/jade_hand/research/shadowkv/results_vllm_apc_qwen14b_metrics_2026-05-26`.
- Matrix result: 12/12 jobs complete; total vLLM prefix-cache hit tokens `82,816`; total response-level cached tokens still `0`.
- Production restore: `qwen36-27b-fp8-vllm` restarted and health-checked on `127.0.0.1:8014`.
- Main-variable gate: checked. Current vLLM path is valid for exact-prefix APC, but not for real injected semantic/partial KV reuse. The code raises on external `past_key_values` in the vLLM backend; the real external-KV path is HuggingFace. Deterministic traces were added and a fake-backend semantic smoke produced `semantic_partial_hits=4`, proving the variable is observable only in the simulation/opportunity path today.
- Gate artifacts: `/home/jade_hand/research/shadowkv/session_files/main_variable_traces_2026-05-26/` and `/home/jade_hand/research/shadowkv/results_main_variable_smoke_2026-05-26/fake_semantic_no_scaffold_long/`.
- Kushal clarification: for Blackwell large-model work, treat ShadowKV++ as a vLLM admission/policy overlay, not direct `past_key_values` injection.
- Qwen2.5-32B Blackwell policy-overlay run: complete. Result root `/home/jade_hand/research/shadowkv/results_vllm_apc_qwen32b_policy_overlay_warmup_parity_2026-05-27`; 12/12 full jobs, `607,084` prompt tokens, `352,352` vLLM prefix-cache hit tokens, response-level cached tokens still `0`. Production `qwen36-27b-fp8-vllm` restored and health-checked.
- Next action: package/share the report with Kushal, or run a higher-request-count pass if he wants tighter confidence intervals.

## 2026-06-03 ShadowKV Qwen32B No-Cache/APC/Overlay Run

- Report: [[shadowkv-vllm-apc-observability-report-2026-05-26]].
- Kushal request: add explicit `vllm_no_cache` to Qwen32B Blackwell runtime-baseline path, keep `vllm_apc` and `vllm_apc_shadowkv_plus`, use 256 requests, datasets `daily_dialog`, `samsum`, `ag_news`, `dolly`, `xsum`, modes `templated` and `rag`, and collect energy if NVML supports it.
- Runner patch: `run_runtime_cache_baseline.py` now exposes `--measure_energy`; `adapter_lib.py` now exposes `build_vllm_no_cache_command()` with explicit `--no-enable-prefix-caching`.
- Run script: `/home/jade_hand/research/shadowkv/session_files/run_shadowkv_vllm_qwen32b_no_cache_apc_overlay_energy_2026-06-03.sh`.
- Result root: `/home/jade_hand/research/shadowkv/results_vllm_qwen32b_no_cache_apc_overlay_energy_2026-06-03`.
- Run log: `/home/jade_hand/research/shadowkv/run_logs/qwen32b_no_cache_apc_overlay_energy_20260603.log`.
- Status: complete; smoke `3/3`, full matrix `30/30`.
- Aggregates: `aggregate_qwen32b_no_cache_apc_overlay_energy_full.csv`, `comparison_vs_no_cache_qwen32b_full.csv`, and JSON equivalents in the result root.
- Headline: no-cache recorded `0` cached prompt tokens; APC and APC+ShadowKV++ each recorded `294,208` vLLM cached prompt tokens. Average mean latency improved from `72.79 ms` no-cache to `59.43 ms` APC and `58.95 ms` overlay. Total GPU energy dropped from `111,318 J` no-cache to about `83.4 kJ` for both APC arms.
- Production restore: `qwen36-27b-fp8-vllm` restarted and observed running after the matrix.

## 2026-06-08 ShadowKV SGLang + LMCache Qwen32B Run

- Report: [[shadowkv-sglang-lmcache-report-2026-06-05]].
- Scope: `Qwen/Qwen2.5-32B-Instruct`, datasets `daily_dialog`, `samsum`, `ag_news`, `dolly`, `xsum`, modes `templated` and `rag`, baselines `sglang_radix_attention` and `lmcache --lmcache_engine sglang`, `n_requests=256`, NVML energy enabled.
- Runtime patch: isolated image `shadowkv-sglang-lmcache:2026-06-08-layerwise-patch` keeps `sglang==0.5.12.post1` and `lmcache==0.4.3`, adds CUDA 12 runtime libraries for LMCache, and overrides `SGLangLayerwiseGPUConnector.initialize_kvcaches_ptr()` so SGLang's nested `[k_pool, v_pool]` KV layout is preserved.
- Result root: `/home/jade_hand/research/shadowkv/results_sglang_lmcache_qwen32b_full_2026-06-08`.
- Run script: `/home/jade_hand/research/shadowkv/session_files/run_shadowkv_sglang_lmcache_qwen32b_2026-06-08.sh`.
- Status: complete; full matrix `20/20`.
- Headline: LMCache reached functional parity with SGLang Radix. Aggregate mean latency was `99.09 ms` for SGLang Radix and `99.45 ms` for LMCache; throughput was `10.19 rps` vs `10.15 rps`; cached tokens total was `351,688` vs `359,986`; idle-adjusted energy was `44.98` vs `44.79` J/request.
- Production restore: `qwen36-27b-fp8-vllm` restarted and health-checked on `127.0.0.1:8014`.

## 2026-06-13 SGLang Small-Model ShadowKV Test

- Report: [[shadowkv-sglang-small-model-test-2026-06-13]].
- LMCache isolation check: `--enable-lmcache --disable-radix-cache` is not valid in the installed SGLang path because disabling Radix routes to `ChunkCache` and prevents `LMCRadixCache` construction. Public-dataset and controlled long-prefix Qwen14B LMCache checks both completed but showed `cached_tokens_total=0` and no LMCache retrieval activity.
- Version-matched LMCache smoke: complete with `shadowkv-sglang-lmcache:2026-06-13-sglang0513-lmcache047` (`sglang==0.5.13`, `lmcache==0.4.7`). Result root `/home/jade_hand/research/shadowkv/results_lmcache_versionmatch_qwen14b_smoke_2026-06-13`. After `POST /flush_cache` cleared SGLang Radix, the repeated long-prefix request still reported `2,560` cached tokens and LMCache logged `Retrieved 2560 tokens in 0.001 seconds`. This validates Radix+LMCache spill/retrieve, not Radix-disabled LMCache isolation.
- ShadowKV test scope: `Qwen/Qwen2.5-14B-Instruct`, datasets `daily_dialog`, `samsum`, `ag_news`, `dolly`, `xsum`, modes `templated` and `rag`, baselines `sglang_radix_attention` and `sglang_radix_attention_shadowkv_plus`, `n_requests=64`.
- Result root: `/home/jade_hand/research/shadowkv/results_sglang_shadowkv_qwen14b_small_2026-06-13`.
- Status: complete; matrix `20/20`.
- Headline: ShadowKV++ overlay was active on all measured requests: `640` admission plans, `630` allows, `10` bypasses, `640` stores. Aggregate mean latency improved from `45.25 ms` native SGLang to `44.44 ms` ShadowKV++; P95 improved from `50.12 ms` to `48.55 ms`; throughput improved from `22.25 rps` to `22.64 rps`; energy was effectively tied (`18.69` vs `18.70` idle-adjusted J/request).
- Production restore: `qwen36-27b-fp8-vllm` restarted and health-checked on `127.0.0.1:8014`.

## 2026-06-14 SGLang LMCache No-Native-Radix Baseline

- Report: [[shadowkv-sglang-small-model-test-2026-06-13]].
- Patch image: `shadowkv-sglang-lmcache:2026-06-14-no-native-radix`.
- Patch file: `/home/jade_hand/research/shadowkv/session_files/Dockerfile.sglang_lmcache_2026-06-14-no-native-radix`.
- Smoke root: `/home/jade_hand/research/shadowkv/results_lmcache_no_native_radix_qwen14b_smoke_2026-06-14`.
- Matrix root: `/home/jade_hand/research/shadowkv/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14`.
- Status: complete; smoke passed and matrix `10/10`.
- Smoke proof: with `--enable-lmcache --disable-radix-cache` and no manual `/flush_cache`, the repeated long-prefix requests returned `cached_tokens=2560`; LMCache logged one store and two retrieves of `2560` tokens.
- Matrix aggregate: mean latency `51.56 ms`, P95 `62.97 ms`, throughput `19.61 rps`, idle-adjusted energy `23.74` J/request, prompt tokens `167304`, cached tokens `7168`, LMCache retrieve events `27`, store events `242`.
- Interpretation: valid true LMCache-without-native-Radix baseline. It works, but loses most native Radix fine-grained short-prefix reuse because LMCache reuse is chunk-aligned at 256 tokens.
- Production restore: `darwin28b-reason-vllm` restarted and health-checked on `127.0.0.1:8015`.

## 2026-06-16 Three-Model Repeated SGLang Baseline Run

- Report: [[shadowkv-sglang-small-model-test-2026-06-13]].
- Scope: Qwen2.5 `1.5B`, `3B`, and `7B`; datasets `daily_dialog`, `samsum`, `ag_news`, `dolly`, `xsum`; modes `templated` and `rag`; baselines native SGLang Radix, SGLang Radix + ShadowKV++ overlay, and patched LMCache no-native-Radix.
- Runtime image: `shadowkv-sglang-lmcache:2026-06-14-no-native-radix`.
- Result root: `/home/jade_hand/research/shadowkv/results_sglang_lmcache_shadowkv_3models_3reps_2026-06-16`.
- Run script: `/home/jade_hand/research/shadowkv/session_files/run_shadowkv_sglang_lmcache_3models_3reps_2026-06-16.sh`.
- Status: complete; smoke passed, full matrix `270/270`, LMCache summaries `90/90`, aggregate CSV/JSON/Markdown artifacts generated.
- Headline: ShadowKV++ is parity/slightly slower than native Radix on 1.5B and 3B, faster on 7B; patched LMCache no-native-Radix works but is slower than native Radix on all three models because it loses most fine-grained Radix prefix reuse.
- Production restore: `darwin28b-reason-vllm` restarted and health-checked on `127.0.0.1:8015`.

## 2026-06-17 Identical-Blocks SGLang Rerun

- Report: [[shadowkv-sglang-small-model-test-2026-06-13]].
- Purpose: remove the previous model-major ordering caveat by using matched workload blocks while preserving the same models, datasets, prompt modes, baselines, request count, runtime image, and runtime settings.
- Result root: `/home/jade_hand/research/shadowkv/results_sglang_lmcache_shadowkv_3models_identical_blocks_3reps_2026-06-16`.
- Run script: `/home/jade_hand/research/shadowkv/session_files/run_shadowkv_sglang_lmcache_3models_identical_blocks_3reps_2026-06-16.sh`.
- Status: complete; full matrix `270/270`, LMCache summaries `90/90`, aggregates generated, and audit artifact `ANOMALY_AUDIT_IDENTICAL_BLOCKS_2026-06-17.md` generated.
- Schedule audit: 30 matched workload blocks; each model appears in positions 1/2/3 exactly 30 times; each baseline appears in positions 1/2/3 exactly 30 times; error scan hits `0`.
- Headline: ShadowKV++ remains slightly slower than native Radix on 1.5B and 3B, faster on 7B; cached-token totals match native Radix exactly in all paired cells. Patched LMCache no-native-Radix remains functional but slower because it gives up fine-grained Radix reuse.
- Production restore: `darwin28b-reason-vllm` restarted and health-checked on `127.0.0.1:8015`.

## Related

- [[nemotron-reasoning-challenge]]
- [[shadowkv-plus]]
