#!/usr/bin/env bash
set -euo pipefail

cd /home/jade_hand/research/shadowkv

VLLM_IMAGE="shadowkv-shadowkv:latest"
SGLANG_IMAGE="shadowkv-sglang-native-admission:2026-06-22-counters"
RUN_ID="reviewer_gap_qwen15b_waste_balanced_util2_2026-07-04"
RESULT_ROOT_HOST="/home/jade_hand/research/shadowkv/results_${RUN_ID}"
RESULT_ROOT_CONTAINER="/workspace/shadowkv/results_${RUN_ID}"
LOG="/home/jade_hand/research/shadowkv/run_logs/${RUN_ID}.log"
STATUS="/home/jade_hand/research/shadowkv/run_logs/${RUN_ID}.status"
CONTAINER_NAME="shadowkv-reviewer-gap-qwen15b-balanced-util2"

mkdir -p "$RESULT_ROOT_HOST" "$(dirname "$LOG")"

write_status() {
  local state="$1"
  shift || true
  {
    echo "state=${state}"
    echo "run_id=${RUN_ID}"
    echo "updated_at=$(date -Is)"
    echo "result_root=${RESULT_ROOT_HOST}"
    echo "log=${LOG}"
    echo "vllm_image=${VLLM_IMAGE}"
    echo "sglang_image=${SGLANG_IMAGE}"
    for kv in "$@"; do echo "$kv"; done
  } > "$STATUS"
}

restore_production() {
  set +e
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  cd /home/jade_hand/active/services/qwen36-27b-fp8-vllm && docker compose up -d >/dev/null 2>&1 || docker start qwen36-27b-fp8-vllm >/dev/null 2>&1 || true
  for _ in $(seq 1 90); do
    if curl -fsS http://127.0.0.1:8014/v1/models >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
}
trap restore_production EXIT

stop_production_for_gpu() {
  if docker ps --format '{{.Names}}' | grep -qx qwen36-27b-fp8-vllm; then
    echo "[$(date -Is)] stopping active GPU service qwen36-27b-fp8-vllm" | tee -a "$LOG"
    docker stop qwen36-27b-fp8-vllm | tee -a "$LOG"
  fi
}

snapshot_metadata() {
  local meta="${RESULT_ROOT_HOST}/metadata"
  local snap="${meta}/source_snapshot"
  mkdir -p "$meta" "$snap/session_files" "$snap/literature_accurate_baselines" "$snap/src/proactive_kv_cache"
  {
    echo "captured_at=$(date -Is)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -a)"
    echo "cwd=$(pwd)"
    echo "vllm_image=${VLLM_IMAGE}"
    echo "sglang_image=${SGLANG_IMAGE}"
    echo "run_id=${RUN_ID}"
  } > "$meta/host.txt"
  nvidia-smi > "$meta/nvidia-smi.txt" 2>&1 || true
  nvidia-smi -q > "$meta/nvidia-smi-q.txt" 2>&1 || true
  docker image inspect "$VLLM_IMAGE" > "$meta/docker_image_inspect_vllm.json" 2>&1 || true
  docker image inspect "$SGLANG_IMAGE" > "$meta/docker_image_inspect_sglang.json" 2>&1 || true
  docker run --rm --entrypoint python3 "$VLLM_IMAGE" -c 'import importlib.metadata as md
for name in ["vllm", "sglang", "torch", "transformers", "datasets", "numpy", "flashinfer-python"]:
    try:
        print(f"{name}=={md.version(name)}")
    except Exception as exc:
        print(f"{name}=unavailable ({exc})")
' > "$meta/runtime_versions_vllm.txt" 2>&1 || true
  docker run --rm --entrypoint python3 "$SGLANG_IMAGE" -c 'import importlib.metadata as md
for name in ["vllm", "sglang", "torch", "transformers", "datasets", "numpy", "flashinfer-python"]:
    try:
        print(f"{name}=={md.version(name)}")
    except Exception as exc:
        print(f"{name}=unavailable ({exc})")
' > "$meta/runtime_versions_sglang.txt" 2>&1 || true
  docker run --rm --entrypoint python3 "$VLLM_IMAGE" -m pip freeze > "$meta/pip_freeze_vllm.txt" 2>&1 || true
  docker run --rm --entrypoint python3 "$SGLANG_IMAGE" -m pip freeze > "$meta/pip_freeze_sglang.txt" 2>&1 || true
  cp session_files/reviewer_gap_qwen15b_waste_balanced_util2_2026_07_04.py "$snap/session_files/" 2>/dev/null || true
  cp session_files/run_shadowkv_reviewer_gap_qwen15b_waste_balanced_util2_2026_07_04.sh "$snap/session_files/" 2>/dev/null || true
  cp literature_accurate_baselines/adapter_lib.py "$snap/literature_accurate_baselines/" 2>/dev/null || true
  cp literature_accurate_baselines/run_runtime_cache_baseline.py "$snap/literature_accurate_baselines/" 2>/dev/null || true
  cp src/proactive_kv_cache/workload.py "$snap/src/proactive_kv_cache/" 2>/dev/null || true
  cp src/proactive_kv_cache/datasets.py "$snap/src/proactive_kv_cache/" 2>/dev/null || true
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git status --short > "$snap/git-status.txt" 2>&1 || true
    git rev-parse HEAD > "$snap/git-head.txt" 2>&1 || true
  else
    echo "/home/jade_hand/research/shadowkv is not a git repository" > "$snap/git-status.txt"
  fi
}

run_pass() {
  local runtime_filter="$1"
  local image="$2"
  write_status "running_${runtime_filter}" "container=${CONTAINER_NAME}" "image=${image}"
  echo "[$(date -Is)] START ${RUN_ID} runtime=${runtime_filter} image=${image}" | tee -a "$LOG"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker run --rm --name "$CONTAINER_NAME" \
    --network host --device nvidia.com/gpu=all --ipc host \
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
    "$image" \
      session_files/reviewer_gap_qwen15b_waste_balanced_util2_2026_07_04.py \
      --output-root "$RESULT_ROOT_CONTAINER" \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --n-requests 256 \
      --seed 20260703 \
      --runtime-filter "$runtime_filter" \
      --context-length 4096 \
      --dtype float16 \
      --dtype-bytes 2 \
      --max-tokens 1 \
      --temperature 0.0 \
      --admission_preset balanced \
      --policy.utility.util_min_ms 2.0 \
      --min_bootstrap_admissions 0 2>&1 | tee -a "$LOG"
  echo "[$(date -Is)] DONE ${RUN_ID} runtime=${runtime_filter}" | tee -a "$LOG"
}

main() {
  write_status "starting"
  if ! docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    echo "missing Docker image: $VLLM_IMAGE" >&2
    exit 1
  fi
  if ! docker image inspect "$SGLANG_IMAGE" >/dev/null 2>&1; then
    echo "missing Docker image: $SGLANG_IMAGE" >&2
    exit 1
  fi
  snapshot_metadata
  stop_production_for_gpu
  run_pass "vllm" "$VLLM_IMAGE"
  run_pass "sglang" "$SGLANG_IMAGE"
  echo "[$(date -Is)] DONE ${RUN_ID}" | tee -a "$LOG"
  write_status "complete" "completed_at=$(date -Is)"
}

main "$@"
