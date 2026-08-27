#!/usr/bin/env bash
set -euo pipefail

cd /home/jade_hand/research/shadowkv

IMAGE="shadowkv-sglang-lmcache:2026-06-08-layerwise-patch"
MODEL="Qwen/Qwen2.5-14B-Instruct"
RESULT_ROOT_HOST="/home/jade_hand/research/shadowkv/results_sglang_shadowkv_qwen14b_small_2026-06-13"
RESULT_ROOT_CONTAINER="/workspace/shadowkv/results_sglang_shadowkv_qwen14b_small_2026-06-13"
LOG="/home/jade_hand/research/shadowkv/run_logs/sglang_shadowkv_qwen14b_small_20260613.log"
STATUS="/home/jade_hand/research/shadowkv/run_logs/sglang_shadowkv_qwen14b_small_20260613.status"

DATASETS=(daily_dialog samsum ag_news dolly xsum)
MODES=(templated rag)
BASELINES=(sglang_radix_attention sglang_radix_attention_shadowkv_plus)

N_REQUESTS=64
WARMUP_REQUESTS=8
MAX_TOKENS=1

mkdir -p "$(dirname "$LOG")" "$RESULT_ROOT_HOST"

restore_production() {
  set +e
  docker start qwen36-27b-fp8-vllm >/dev/null 2>&1 || true
}
trap restore_production EXIT

run_one() {
  local baseline="$1"
  local dataset="$2"
  local mode="$3"
  local out_dir="${RESULT_ROOT_CONTAINER}/${dataset}/${mode}/${baseline}"

  echo "[$(date -Is)] START baseline=${baseline} dataset=${dataset} mode=${mode}" | tee -a "$LOG"
  printf 'baseline=%s\ndataset=%s\nmode=%s\nstarted_at=%s\n' "$baseline" "$dataset" "$mode" "$(date -Is)" > "$STATUS"

  docker run --rm --network host --device nvidia.com/gpu=all --ipc host \
    -e HF_HOME=/cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    -e TRANSFORMERS_CACHE=/cache/huggingface \
    -e USE_HUB_KERNELS=NO \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e PYTHONHASHSEED=0 \
    -v /home/jade_hand/research/shadowkv:/workspace/shadowkv \
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
      --n_requests "$N_REQUESTS" \
      --disable_arrival_simulation \
      --max_tokens "$MAX_TOKENS" \
      --launch_server \
      --python_executable python3 \
      --server_ready_timeout_s 900 \
      --request_timeout_s 300 \
      --warmup_requests "$WARMUP_REQUESTS" \
      --measure_energy \
      --idle_baseline_seconds 5 \
      --output_dir "$out_dir" \
      --server_extra_arg=--context-length \
      --server_extra_arg=4096 \
      --server_extra_arg=--mem-fraction-static \
      --server_extra_arg=0.80 \
      --server_extra_arg=--dtype \
      --server_extra_arg=float16 \
      --server_extra_arg=--attention-backend \
      --server_extra_arg=triton \
      --server_extra_arg=--sampling-backend \
      --server_extra_arg=pytorch \
      --server_extra_arg=--disable-cuda-graph \
      --server_extra_arg=--disable-piecewise-cuda-graph 2>&1 | tee -a "$LOG"

  echo "[$(date -Is)] DONE baseline=${baseline} dataset=${dataset} mode=${mode}" | tee -a "$LOG"
}

main() {
  echo "[$(date -Is)] SMALL SHADOWKV RUN START image=${IMAGE} model=${MODEL}" | tee -a "$LOG"
  docker image inspect "$IMAGE" >/dev/null
  if docker ps --format '{{.Names}}' | grep -qx qwen36-27b-fp8-vllm; then
    docker stop qwen36-27b-fp8-vllm | tee -a "$LOG"
  fi

  for dataset in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
      for baseline in "${BASELINES[@]}"; do
        run_one "$baseline" "$dataset" "$mode"
      done
    done
  done

  printf 'completed_at=%s\n' "$(date -Is)" > "$STATUS"
  echo "[$(date -Is)] SMALL SHADOWKV RUN DONE result_root=${RESULT_ROOT_HOST}" | tee -a "$LOG"
}

main "$@"
