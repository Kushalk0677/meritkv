# Colab CLI Workflow

This is the preferred headless workflow for the MeritKV exact-splice
validation. It provisions a T4, runs Gemma-4-E2B and Qwen2.5-1.5B under eager
and SDPA attention, downloads one evidence archive, and releases the runtime.

## Platform

The official Google Colab CLI currently supports Linux and macOS. On Windows,
run the workflow inside WSL2. Do not run the shell script from PowerShell or
Command Prompt.

## One-time preparation on Windows

Open PowerShell as Administrator and install WSL if it is not already
available:

```powershell
wsl --install
```

Restart Windows if requested, open the installed Ubuntu terminal, and enter
the package directory through the mounted Windows drive:

```bash
cd /mnt/c/shadowkv/v10/colab_packages/gemma4_e2b_qwen25_15b_splice_validation
```

Accept the `google/gemma-4-E2B-it` access terms on Hugging Face before running
the validation. Have a Hugging Face read token ready.

## Run

```bash
bash run_with_colab_cli.sh
```

For the five-model primary-study certification matrix, run instead:

```bash
bash run_primary5_certification.sh
```

This selects GPT-2, TinyLlama-1.1B-Chat, Qwen2.5-1.5B-Instruct,
Gemma-2B-IT, and Phi-3-mini-4k-instruct under float16 SDPA on T4. It reports
finite-logit, generated-token, active-cache, and logit-distance comparisons.

On the first run, the Colab CLI opens its Google OAuth flow. The script then
prompts for the Hugging Face token without echoing it. The token is written to
a permission-restricted temporary file, uploaded to the ephemeral VM, deleted
locally, consumed by the remote driver, and deleted remotely. It is never
placed in a command-line argument.

If the official `colab` command is not installed, the script installs
`google-colab-cli` automatically. It prefers `uv`; otherwise it creates a
user-local isolated Python virtual environment.

The script also verifies the CLI's `jupyter_kernel_client.KernelClient` API
before requesting a GPU. `google-colab-cli 0.6.0` may resolve an incompatible
PyPI dependency; when detected, the script repairs only the isolated CLI tool
environment using the exact Google fork commit pinned by the official Colab
CLI source (`f18e982c3265df5e923aa9def101ab3fd737e139`).

## Output

The completed default two-model archive is downloaded automatically to:

```text
downloaded_results/meritkv_exact_splice_validation_outputs_gemma4_e2b__qwen25_15b.zip
```

The archive contains:

- immutable Hugging Face model revisions;
- environment and `nvidia-smi` records;
- separate eager and SDPA JSON reports and raw logs for both models;
- token-, logit-, and per-layer KV comparisons;
- a combined Markdown and JSON summary;
- `RUN_STATUS.json`; and
- SHA-256 checksums.

A strict validation failure is retained as evidence and does not prevent the
archive from being downloaded. An execution failure is also recorded when
possible; the script returns a nonzero status after downloading the retained
archive.

The Colab session is stopped automatically, including after failure. Set
`KEEP_SESSION=1` only when deliberate remote debugging is required:

```bash
KEEP_SESSION=1 bash run_with_colab_cli.sh
```

To choose a different local output directory:

```bash
MERITKV_RESULTS_DIR=/path/to/results bash run_with_colab_cli.sh
```

To rerun only one checkpoint after retaining the other checkpoint's completed
archive, select its key explicitly:

```bash
MERITKV_MODELS=gemma4_e2b bash run_with_colab_cli.sh
```

The other valid key is `qwen25_15b`. The default remains both models. The
selection is included in the downloaded filename, so a one-model recovery run
does not overwrite an earlier archive.

## Authentication distinction

The Google OAuth prompt authorizes the CLI to control the Colab runtime. The
Hugging Face token separately authorizes checkpoint download inside that
runtime. `colab auth` configures Google Cloud credentials and is not a
substitute for the Hugging Face token.
