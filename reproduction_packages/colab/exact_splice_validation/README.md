# MeritKV Gemma-4-E2B and Qwen2.5-1.5B Exact-Splice Validation: Colab Package

## Goal

Validate the corrected token-identical Hugging Face KV crop-and-splice path on
two checkpoints already evaluated in the paper:

`google/gemma-4-E2B-it`

`Qwen/Qwen2.5-1.5B-Instruct`

The validator compares full recompute, clean prefix caching, and cache
crop-and-splice. It checks per-layer K/V tensors, aligned suffix logits, and
greedy token IDs. It does not use ROUGE or decoded-text similarity as a
correctness criterion.

## Package Contents

- `run_with_colab_cli.sh`: preferred one-command Linux/macOS/WSL workflow.
- `run_primary5_certification.sh`: supported-path certification across the
  five primary-study models using float16 SDPA on T4.
- `run_remote_validation.py`: remote T4 orchestrator used by the CLI workflow.
- `COLAB_CLI_README.md`: CLI setup, security, execution, and output details.
- `MeritKV_Gemma4_E2B_Qwen25_Splice_Validation_Colab.ipynb`: guided Colab workflow.
- `validate_exact_splice.py`: standalone validator.
- `summarize_validation.py`: combines both models' eager and SDPA JSON reports.
- `requirements-colab.txt`: the recorded Transformers stack used for this
  validation. Colab's CUDA-enabled PyTorch is intentionally not replaced.
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

The preferred workflow is the official Colab CLI. On Windows, open WSL, enter
this package directory, and run:

```bash
bash run_with_colab_cli.sh
```

It provisions the T4, runs all four model/attention combinations, downloads a
selection-labelled archive under `downloaded_results/`, and stops the runtime.
See `COLAB_CLI_README.md` for details.

The notebook remains available as a browser-based fallback:

1. Open the included notebook in Colab.
2. Upload this ZIP when the notebook asks for it.
3. Run every cell in order.
4. The notebook first runs a tiny local harness self-test.
5. It resolves and records the exact Hugging Face revision of both checkpoints.
6. It runs float16 eager-attention validation on Gemma and Qwen.
7. It runs float16 SDPA validation on Gemma and Qwen.
8. It automatically creates and downloads
   `meritkv_exact_splice_validation_outputs.zip`.

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

- exact model revisions and software environment;
- separate eager and SDPA validator reports for both models;
- combined JSON and Markdown summaries;
- the exact command log; and
- SHA-256 checksums.

## Scope

Successful runs validate the corrected custom-splice path for Gemma-4-E2B and
Qwen2.5-1.5B in the tested float16 configurations. They do not automatically
validate other model sizes or the previous timing results. If the corrected
path is used to support a performance claim, the associated timing
configuration must also be rerun through that corrected implementation.
