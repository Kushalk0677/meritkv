#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MERITKV_MODELS="gpt2,tinyllama,qwen25_15b,gemma2b,phi3mini"
export MERITKV_ATTENTION="sdpa"
exec bash "${PACKAGE_DIR}/run_with_colab_cli.sh"
