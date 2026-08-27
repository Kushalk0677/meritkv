#!/usr/bin/env bash
set -euo pipefail

REPO=/home/jade_hand/research/shadowkv
RESULT_ROOT="$REPO/results_vllm_qwen32b_no_cache_apc_overlay_energy_2026-06-03"
CONTAINER_RESULT_ROOT="/workspace/shadowkv/results_vllm_qwen32b_no_cache_apc_overlay_energy_2026-06-03"
LOG_DIR="$REPO/run_logs"
MODEL="Qwen/Qwen2.5-32B-Instruct"
IMAGE="shadowkv-shadowkv"
PROD_CONTAINER="qwen36-27b-fp8-vllm"
FULL_REQUESTS=256
SMOKE_REQUESTS=16
WARMUP_REQUESTS=16
GPU_MEM_UTIL="0.88"
MAX_MODEL_LEN="4096"

mkdir -p "$RESULT_ROOT" "$RESULT_ROOT/smoke" "$RESULT_ROOT/full" "$LOG_DIR"

restore_prod() {
  set +e
  if docker ps -a --format '{{.Names}}' | grep -qx "$PROD_CONTAINER"; then
    docker start "$PROD_CONTAINER" >/dev/null 2>&1 || true
  fi
}
trap restore_prod EXIT

if docker ps --format '{{.Names}}' | grep -qx "$PROD_CONTAINER"; then
  echo "[$(date -Is)] stopping production container $PROD_CONTAINER for exclusive GPU run"
  docker stop "$PROD_CONTAINER"
else
  echo "[$(date -Is)] production container $PROD_CONTAINER not running"
fi

run_one() {
  local phase="$1"
  local baseline="$2"
  local dataset="$3"
  local mode="$4"
  local requests="$5"
  local outdir="$RESULT_ROOT/$phase/$dataset/$mode/$baseline"
  local container_outdir="$CONTAINER_RESULT_ROOT/$phase/$dataset/$mode/$baseline"
  mkdir -p "$outdir"
  echo "[$(date -Is)] START phase=$phase baseline=$baseline dataset=$dataset mode=$mode requests=$requests"

  local admission_args=()
  if [[ "$baseline" == "vllm_apc_shadowkv_plus" ]]; then
    admission_args+=(--enable_admission_tuning --admission_tuning_requests 16 --admission_tuning_metric cached_adjusted_latency)
  fi

  docker run --rm \
    --network host \
    --device nvidia.com/gpu=all \
    --ipc host \
    -e HF_HOME=/cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    -e TRANSFORMERS_CACHE=/cache/huggingface \
    -v "$REPO:/workspace/shadowkv" \
    -v /datapool/cache/huggingface:/cache/huggingface \
    -w /workspace/shadowkv \
    --entrypoint python3 \
    "$IMAGE" \
      literature_accurate_baselines/run_runtime_cache_baseline.py \
      --baseline "$baseline" \
      --model "$MODEL" \
      --workload public_dataset \
      --dataset "$dataset" \
      --prompt_mode "$mode" \
      --n_requests "$requests" \
      --disable_arrival_simulation \
      --max_tokens 1 \
      --dtype float16 \
      --launch_server \
      --server_ready_timeout_s 720 \
      --request_timeout_s 180 \
      --warmup_requests "$WARMUP_REQUESTS" \
      --measure_energy \
      --idle_baseline_seconds 5 \
      --output_dir "$container_outdir" \
      --server_extra_arg=--max-model-len \
      --server_extra_arg="$MAX_MODEL_LEN" \
      --server_extra_arg=--gpu-memory-utilization \
      --server_extra_arg="$GPU_MEM_UTIL" \
      "${admission_args[@]}"

  echo "[$(date -Is)] DONE phase=$phase baseline=$baseline dataset=$dataset mode=$mode requests=$requests"
}

BASELINES=(vllm_no_cache vllm_apc vllm_apc_shadowkv_plus)
DATASETS=(daily_dialog samsum ag_news dolly xsum)
MODES=(templated rag)

for baseline in "${BASELINES[@]}"; do
  run_one smoke "$baseline" daily_dialog templated "$SMOKE_REQUESTS"
done

echo "[$(date -Is)] smoke phase complete; starting full matrix"
for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    for baseline in "${BASELINES[@]}"; do
      run_one full "$baseline" "$dataset" "$mode" "$FULL_REQUESTS"
    done
  done
done

echo "[$(date -Is)] full matrix complete"
