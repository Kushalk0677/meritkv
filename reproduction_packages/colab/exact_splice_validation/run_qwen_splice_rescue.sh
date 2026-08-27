#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MERITKV_REMOTE_DRIVER="run_qwen_splice_rescue.py"
export MERITKV_REMOTE_ARCHIVE="/content/meritkv_qwen_splice_rescue_outputs.zip"
export MERITKV_OUTPUT_TAG="qwen25_15b_old_vs_corrected_rouge_speed"
exec bash "${PACKAGE_DIR}/run_with_colab_cli.sh"
