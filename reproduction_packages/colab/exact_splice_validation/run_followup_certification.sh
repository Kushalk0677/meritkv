#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MERITKV_REMOTE_DRIVER="run_followup_validation.py"
export MERITKV_REMOTE_ARCHIVE="/content/meritkv_exact_splice_followup_outputs.zip"
export MERITKV_OUTPUT_TAG="followup_gpt2_f32__primary5_natural_f16_sdpa"
exec bash "${PACKAGE_DIR}/run_with_colab_cli.sh"
