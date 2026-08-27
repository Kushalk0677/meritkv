# MeritKV Gemma-4-E2B Exact-Splice Validation: Colab Package

## Goal

Validate the corrected token-identical Hugging Face KV crop-and-splice path on
the smallest Gemma-4 checkpoint already evaluated in the paper:

`google/gemma-4-E2B-it`

The validator compares full recompute, clean prefix caching, and cache
crop-and-splice. It checks per-layer K/V tensors, aligned suffix logits, and
greedy token IDs. It does not use ROUGE or decoded-text similarity as a
correctness criterion.

## Package Contents

- `MeritKV_Gemma4_E2B_Splice_Validation_Colab.ipynb`: guided Colab workflow.
- `validate_exact_splice.py`: standalone validator.
- `summarize_validation.py`: combines eager and SDPA JSON reports.
- `requirements-colab.txt`: the Transformers stack used by the original E2B
  experiment. Colab's CUDA-enabled PyTorch is intentionally not replaced.
- `local_reference_results/`: successful CPU harness checks on tiny Qwen2 and
  Llama configurations. These validate the harness only and are not paper
  evidence.
- `MANIFEST_SHA256.txt`: package integrity ledger.

## Before Opening Colab

1. Accept the Gemma model terms on Hugging Face for the account you will use.
2. Create a read token at Hugging Face.
3. In Colab, select **Runtime > Change runtime type > T4 GPU**.
4. Add the token as a Colab secret named `HF_TOKEN`, or enter it interactively
   when prompted.

## Recommended Workflow

1. Open the included notebook in Colab.
2. Upload this ZIP when the notebook asks for it.
3. Run every cell in order.
4. The notebook first runs a tiny local harness self-test.
5. It resolves and records the exact Hugging Face revision of
   `google/gemma-4-E2B-it`.
6. It runs a float16 eager-attention validation.
7. It runs a float16 SDPA validation, which is the more relevant practical HF
   execution path.
8. It creates and downloads `gemma4_e2b_splice_outputs.zip`.

The smoke configuration uses 128-token sequences, 50% and 75% shared-prefix
ratios, four deterministic token pairs, and eight greedy continuation tokens.
This is intentionally small enough for a T4 while testing cache construction,
suffix continuation, and repeated decoding.

## Pass Criteria

For every case:

- the cropped cache has the expected logical prefix length;
- clean-prefix and cropped-prefix K/V tensors pass the declared float16
  tolerance;
- suffix next-token argmax agrees between full recompute, clean caching, and
  crop-and-splice;
- all eight greedy token IDs agree with full recompute;
- actual logit and K/V differences are retained in the JSON, even when the
  binary pass succeeds.

The initial float16 tolerance is `atol=0.001, rtol=0.001`. Do not loosen it to
force a pass. If strict numerical comparison fails while token identity passes,
retain the failure and inspect the recorded differences.

## What to Return

Return the notebook-generated ZIP without editing its JSON files. It contains:

- exact model revision and software environment;
- eager and SDPA validator reports;
- combined JSON and Markdown summaries;
- the exact command log; and
- SHA-256 checksums.

## Scope

A successful E2B run validates the corrected Gemma-4-E2B custom-splice path in
the tested float16 configuration. It does not automatically validate every
Gemma-4 size, Qwen, or the previous timing results. If the corrected path is
used to support a performance claim, the associated timing configuration must
also be rerun through that corrected implementation.
