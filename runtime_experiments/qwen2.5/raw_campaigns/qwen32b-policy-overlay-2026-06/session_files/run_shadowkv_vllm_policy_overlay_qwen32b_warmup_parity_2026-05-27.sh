#!/usr/bin/env bash
set -euo pipefail

HOST_REPO="/home/jade_hand/research/shadowkv"
CONTAINER_REPO="/workspace/shadowkv"
IMAGE="shadowkv-shadowkv"
PROD_CONTAINER="qwen36-27b-fp8-vllm"
ROOT_OUT="results_vllm_apc_qwen32b_policy_overlay_warmup_parity_2026-05-27"
LOG_DIR="${HOST_REPO}/run_logs"
MODEL="Qwen/Qwen2.5-32B-Instruct"
FULL_REQUESTS="${FULL_REQUESTS:-256}"
SMOKE_REQUESTS="${SMOKE_REQUESTS:-16}"

mkdir -p "${LOG_DIR}"
cd "${HOST_REPO}"

restore_prod() {
  echo "RESTORE_PROD $(date -Is)"
  if docker ps -a --format '{{.Names}}' | grep -qx "${PROD_CONTAINER}"; then
    docker start "${PROD_CONTAINER}" >/dev/null || true
    for _ in $(seq 1 180); do
      if curl -fsS http://127.0.0.1:8014/health >/dev/null 2>&1; then
        echo "PROD_HEALTH_OK $(date -Is)"
        return 0
      fi
      sleep 2
    done
    echo "PROD_HEALTH_TIMEOUT $(date -Is)"
  else
    echo "PROD_CONTAINER_MISSING ${PROD_CONTAINER}"
  fi
}

trap restore_prod EXIT

echo "RUN_START $(date -Is)"
echo "MODEL ${MODEL}"
echo "ROOT_OUT ${ROOT_OUT}"
echo "FULL_REQUESTS ${FULL_REQUESTS}"

if docker ps --format '{{.Names}}' | grep -qx "${PROD_CONTAINER}"; then
  echo "STOP_PROD $(date -Is)"
  docker stop "${PROD_CONTAINER}" >/dev/null
fi

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

run_one() {
  local phase="$1"
  local dataset="$2"
  local mode="$3"
  local baseline="$4"
  local n_requests="$5"
  local out_dir="${ROOT_OUT}/${phase}/${dataset}/${mode}/${baseline}"
  if compgen -G "${out_dir}/benchmark_*.json" >/dev/null; then
    echo "SKIP existing ${phase} ${dataset} ${mode} ${baseline}"
    return 0
  fi
  echo "RUN ${phase} ${dataset} ${mode} ${baseline} n=${n_requests} $(date -Is)"
  docker run --rm \
    --network host \
    --device nvidia.com/gpu=all \
    --ipc host \
    -e HF_HOME=/cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    -e TRANSFORMERS_CACHE=/cache/huggingface \
    -v "${HOST_REPO}:${CONTAINER_REPO}" \
    -v /datapool/cache/huggingface:/cache/huggingface \
    -w "${CONTAINER_REPO}" \
    --entrypoint bash \
    "${IMAGE}" \
    -lc "python3 literature_accurate_baselines/run_runtime_cache_baseline.py \
      --baseline '${baseline}' \
      --model '${MODEL}' \
      --workload public_dataset \
      --dataset '${dataset}' \
      --prompt_mode '${mode}' \
      --n_requests '${n_requests}' \
      --disable_arrival_simulation \
      --output_dir '${out_dir}' \
      --max_tokens 1 \
      --dtype float16 \
      --launch_server \
      --server_ready_timeout_s 720 \
      --request_timeout_s 180 \
      --warmup_requests 16 \
      --enable_admission_tuning \
      --admission_tuning_requests 16 \
      --admission_tuning_metric cached_adjusted_latency \
      --server_extra_arg=--max-model-len \
      --server_extra_arg=4096 \
      --server_extra_arg=--gpu-memory-utilization \
      --server_extra_arg=0.88"
}

for baseline in vllm_apc vllm_apc_shadowkv_plus; do
  run_one smoke daily_dialog templated "${baseline}" "${SMOKE_REQUESTS}"
done

for dataset in daily_dialog samsum ag_news; do
  for mode in templated rag; do
    for baseline in vllm_apc vllm_apc_shadowkv_plus; do
      run_one full "${dataset}" "${mode}" "${baseline}" "${FULL_REQUESTS}"
    done
  done
done

echo "RUN_COMPLETE $(date -Is)"
