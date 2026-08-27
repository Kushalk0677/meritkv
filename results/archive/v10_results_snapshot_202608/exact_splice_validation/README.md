# Exact-Prefix Crop-and-Splice Validation

## Purpose

This directory records correctness tests for the standalone validator at
`v10/experiments/validate_exact_splice.py`. The validator compares:

1. full recompute of a modified token sequence;
2. clean prefix-only caching followed by its suffix; and
3. a cache produced from a longer source sequence, cropped to a token-identical
   prefix, then followed by the same suffix.

It passes the complete attention mask, logical `cache_position`, and
`position_ids` on every cached continuation. Correctness is checked using
per-layer K/V tensors, aligned suffix logits, and per-step greedy token IDs.
Decoded-text similarity is not used as a correctness criterion. The completed
real-checkpoint T4 validation and its interpretation are reported in
[`EXACT_SPLICE_VALIDATION.md`](EXACT_SPLICE_VALIDATION.md); the original local
tiny-model smoke tests remain below as harness checks.

## Local Smoke-Test Environment

- Platform: Windows, CPU only
- Python: 3.14.3
- PyTorch: 2.11.0+cpu
- Transformers: 5.7.0
- Attention implementation: eager
- Inputs: deterministic synthetic token sequences with token-identical prefixes

The local machine has no visible NVIDIA runtime or complete cached paper-model
weights. These runs validate the harness and cache protocol; they are not paper
evidence for the real Qwen2.5 or Gemma checkpoints.

## Results

| Model/configuration | Cases | Prefix-KV passes | Native-token passes | Splice-token passes | Strict passes |
|---|---:|---:|---:|---:|---:|
| Tiny Qwen2, float32 | 4 | 4 | 4 | 4 | 4 |
| Tiny Llama, float32 | 9 | 9 | 9 | 9 | 9 |
| Tiny Qwen2, bfloat16 | 9 | 9 | 9 | 9 | 9 |
| **Total** | **22** | **22** | **22** | **22** | **22** |

Across all 22 cases:

- cropped-prefix K/V tensors equal the independently computed clean-prefix K/V
  tensors within the declared tolerances;
- aligned suffix logits are identical between clean native caching and the
  corrected cropped-splice path;
- native and splice greedy token IDs match full recompute at every tested step;
- splice and native-cache generation logits are identical;
- the largest full-recompute versus cached-generation logit difference is
  `2.3842e-7` in float32 and `0.001953125` in bfloat16.

Result files:

- `tiny_qwen2_cpu_f32.json`
- `tiny_llama_cpu_f32.json`
- `tiny_qwen2_cpu_bf16.json`

## What This Establishes

The corrected protocol is internally sound on two rotary-cache model classes.
It also shows the expected floating-point distinction: cached and full
recompute paths can have tiny logit differences while preserving every greedy
token, whereas the clean-cache and cropped-splice paths remain identical under
the same continuation protocol.

This does **not** yet validate the paper's real checkpoints, GPU precision,
attention kernels, or old timing results. Do not cite these tiny random-model
runs as manuscript evidence.

## GPU Smoke Commands

Run the smallest known-problem case first:

```bash
python experiments/validate_exact_splice.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda:0 \
  --dtype float16 \
  --attn-implementation eager \
  --sequence-length 64 \
  --shared-ratios 0.5 0.75 \
  --samples 4 \
  --max-new-tokens 8 \
  --atol 0.001 \
  --rtol 0.001 \
  --output results/exact_splice_validation/qwen25_15b_gpu_f16_smoke.json
```

Then run the headline Gemma model:

```bash
python experiments/validate_exact_splice.py \
  --model-id google/gemma-4-31B-it \
  --device cuda:0 \
  --dtype float16 \
  --attn-implementation eager \
  --sequence-length 128 \
  --shared-ratios 0.25 0.5 0.75 0.9 \
  --samples 8 \
  --max-new-tokens 16 \
  --atol 0.001 \
  --rtol 0.001 \
  --output results/exact_splice_validation/gemma4_31b_gpu_f16_smoke.json
```

After the eager-attention smoke test, repeat using the exact attention
implementation, model revision, PyTorch/Transformers versions, precision, and
generation length used by the timed paper configuration. The JSON report
records the software environment and all comparison metrics.

## Paper-Evidence Reporting Criteria

Before using the result to answer the reviewer:

- validate the actual checkpoint and precision used for every model family
  whose custom-splice timing remains a performance claim;
- report observed greedy token-ID and next-token-argmax agreement exactly,
  including isolated float16 argmax changes rather than relaxing tolerances;
- report logit-difference distributions and declared numerical tolerances;
- retain the per-case JSON reports, exact commands, model revisions, and
  environment manifest;
- if correcting the splice changes the evaluated path, rerun the timing cells
  attached to that path rather than relabelling old timings as validated.
