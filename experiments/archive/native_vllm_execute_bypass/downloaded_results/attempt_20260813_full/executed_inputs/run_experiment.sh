#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${ROOT}/../../.." && pwd)"
MERITKV_SOURCE_ROOT="${MERITKV_SOURCE_ROOT:-${PROJECT_ROOT}/v10/src}"
SHADOWKV_CONFIG="${SHADOWKV_CONFIG:-${PROJECT_ROOT}/v10/config/config.yaml}"
MODEL="${MERITKV_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${MERITKV_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${MERITKV_GPU_MEMORY_UTILIZATION:-0.90}"
OUTPUT_DIR="${MERITKV_OUTPUT_DIR:-${ROOT}/results/native_vllm_execute_bypass_$(date -u +%Y%m%dT%H%M%SZ)}"
SERVER_LOG_DIR="${OUTPUT_DIR}/server"
mkdir -p -- "${SERVER_LOG_DIR}"
mkdir -p -- "${OUTPUT_DIR}/executed_inputs"

if [[ ! -d "${MERITKV_SOURCE_ROOT}/proactive_kv_cache" ]]; then
    echo "Missing frozen MeritKV source at ${MERITKV_SOURCE_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${SHADOWKV_CONFIG}" ]]; then
    echo "Missing frozen MeritKV configuration at ${SHADOWKV_CONFIG}" >&2
    exit 1
fi
export MERITKV_SOURCE_ROOT SHADOWKV_CONFIG
cp -R -- "${MERITKV_SOURCE_ROOT}/proactive_kv_cache" "${OUTPUT_DIR}/executed_inputs/proactive_kv_cache"
cp -- "${SHADOWKV_CONFIG}" "${OUTPUT_DIR}/executed_inputs/config.yaml"
cp -- "${ROOT}/run_native_execute_bypass.py" "${ROOT}/run_experiment.sh" "${ROOT}/run_remote_experiment.py" "${ROOT}/finalize_manifest.py" "${ROOT}/requirements.txt" "${OUTPUT_DIR}/executed_inputs/"

SERVER_PID=""
cleanup() {
    local status=$?
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 -m pip freeze > "${OUTPUT_DIR}/pip_freeze.txt"
nvidia-smi > "${OUTPUT_DIR}/nvidia_smi.txt"
python3 --version > "${OUTPUT_DIR}/python_version.txt" 2>&1

echo "Starting one vLLM server with APC enabled..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype float16 \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --enforce-eager \
    > "${SERVER_LOG_DIR}/stdout.txt" \
    2> "${SERVER_LOG_DIR}/stderr.txt" &
SERVER_PID=$!

cd -- "${PROJECT_ROOT}"
python3 "${ROOT}/run_native_execute_bypass.py" \
    --api-base "http://127.0.0.1:${PORT}" \
    --model "${MODEL}" \
    --output-dir "${OUTPUT_DIR}"

kill "${SERVER_PID}" >/dev/null 2>&1 || true
wait "${SERVER_PID}" >/dev/null 2>&1 || true
SERVER_PID=""
python3 "${ROOT}/finalize_manifest.py" "${OUTPUT_DIR}"
echo "Completed: ${OUTPUT_DIR}"
