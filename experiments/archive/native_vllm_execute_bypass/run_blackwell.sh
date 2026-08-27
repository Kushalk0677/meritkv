#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MERITKV_MODEL="${MERITKV_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
export MERITKV_GPU_MEMORY_UTILIZATION="${MERITKV_GPU_MEMORY_UTILIZATION:-0.90}"
export MERITKV_OUTPUT_DIR="${MERITKV_OUTPUT_DIR:-${ROOT}/results/blackwell_qwen25_32b_$(date -u +%Y%m%dT%H%M%SZ)}"

set +e
bash "${ROOT}/run_experiment.sh"
RUN_STATUS=$?
set -e

ARCHIVE="${MERITKV_OUTPUT_DIR}.zip"
python3 "${ROOT}/finalize_manifest.py" "${MERITKV_OUTPUT_DIR}"
python3 -m zipfile -c "${ARCHIVE}" "${MERITKV_OUTPUT_DIR}"
sha256sum "${ARCHIVE}"
echo "RESULTS=${MERITKV_OUTPUT_DIR}"
echo "RESULT_ARCHIVE=${ARCHIVE}"

exit "${RUN_STATUS}"
