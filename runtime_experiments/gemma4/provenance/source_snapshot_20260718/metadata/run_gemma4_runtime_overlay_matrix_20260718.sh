#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ONLY_ARM="${ONLY_ARM:-${ONLY_RUNTIME:-}}"
ONLY_MODEL_SLUG="${ONLY_MODEL_SLUG:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: $0 [smoke|full]" >&2
  exit 2
fi

PROJECT="/home/jade_hand/research/shadowkv/gemma4_runtime_overlay_matrix_20260718"
SOURCE_HOST="$PROJECT/source_snapshot"
SOURCE_CONTAINER="/workspace/project/source_snapshot"
RESULT_HOST="$PROJECT/results_${MODE}"
RESULT_CONTAINER="/workspace/project/results_${MODE}"
LOG="$PROJECT/run_logs/gemma4_runtime_overlay_matrix_${MODE}_20260718.log"
STATUS="$PROJECT/run_logs/gemma4_runtime_overlay_matrix_${MODE}_20260718.status"
SERVER_NAME="shadowkv-gemma4-runtime-server"
LMCACHE_SERVER_NAME="shadowkv-gemma4-lmcache-server"
LMCACHE_ZMQ_PORT=5557
LMCACHE_HTTP_PORT=18080
PROD_NAME="qwen36-27b-fp8-vllm"
PROD_DIR="/home/jade_hand/active/services/qwen36-27b-fp8-vllm"

VLLM_IMAGE="vllm/vllm-openai:nightly"
SGLANG_IMAGE="shadowkv-sglang-gemma4-unified:20260717"
LMCACHE_IMAGE="lmcache/vllm-openai:v0.5.1"
CLIENT_IMAGE="$SGLANG_IMAGE"
# Avoid FlashInfer's first-use 97-unit MoE JIT on Blackwell. It exhausted host
# RAM for Gemma 4 26B-A4B; Triton is a supported unquantized MoE backend in
# both vLLM images and keeps the APC/LMCache kernel choice aligned.
VLLM_MOE_ARGS=(--moe-backend triton)

MODELS=(
  "gemma4_e2b|google/gemma-4-E2B-it"
  "gemma4_e4b|google/gemma-4-E4B-it"
  "gemma4_12b|google/gemma-4-12B-it"
  "gemma4_26b_a4b|google/gemma-4-26B-A4B-it"
  "gemma4_31b|google/gemma-4-31B-it"
)
ARMS=(
  vllm_apc
  vllm_apc_shadowkv_plus
  sglang_radix_attention
  sglang_radix_attention_shadowkv_plus
  lmcache
  lmcache_shadowkv_plus
)
DATASETS=(daily_dialog samsum ag_news dolly xsum)
PROMPT_MODES=(templated rag)

N_REQUESTS=256
WARMUP_REQUESTS=16
IDLE_BASELINE_SECONDS=10
IDLE_STABILIZATION_SECONDS=30
if [[ "$MODE" == "smoke" ]]; then
  DATASETS=(samsum)
  PROMPT_MODES=(templated)
  N_REQUESTS=16
  WARMUP_REQUESTS=2
  IDLE_BASELINE_SECONDS=5
  IDLE_STABILIZATION_SECONDS=5
fi

mkdir -p "$RESULT_HOST/server_logs" "$PROJECT/run_logs" "$PROJECT/metadata"
touch "$LOG"

write_status() {
  local state="$1"
  shift || true
  {
    echo "state=$state"
    echo "mode=$MODE"
    echo "updated_at=$(date -Is)"
    echo "project=$PROJECT"
    echo "result_root=$RESULT_HOST"
    echo "log=$LOG"
    for item in "$@"; do echo "$item"; done
  } > "$STATUS"
}

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG"
}

stop_server() {
  set +e
  docker logs "$SERVER_NAME" > "$CURRENT_SERVER_LOG" 2>&1 || true
  docker logs "$LMCACHE_SERVER_NAME" > "${CURRENT_SERVER_LOG%.log}.lmcache.log" 2>&1 || true
  docker rm -f "$SERVER_NAME" >/dev/null 2>&1 || true
  docker rm -f "$LMCACHE_SERVER_NAME" >/dev/null 2>&1 || true
  set -e
}

restore_production() {
  set +e
  docker rm -f "$SERVER_NAME" >/dev/null 2>&1 || true
  docker rm -f "$LMCACHE_SERVER_NAME" >/dev/null 2>&1 || true
  cd "$PROD_DIR" && docker compose up -d >/dev/null 2>&1
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8014/v1/models >/dev/null 2>&1; then
      set -e
      return 0
    fi
    sleep 5
  done
  docker start "$PROD_NAME" >/dev/null 2>&1 || true
  set -e
  return 1
}

CURRENT_SERVER_LOG="$RESULT_HOST/server_logs/bootstrap.log"
trap restore_production EXIT

assert_gpu_exclusive_except_production() {
  python3 - <<'PY'
import csv
import subprocess
import sys

prod = set()
try:
    text = subprocess.check_output(
        ["docker", "top", "qwen36-27b-fp8-vllm", "-eo", "pid,ppid,comm"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    for line in text.splitlines()[1:]:
        first = line.split(maxsplit=1)[0] if line.strip() else ""
        if first.isdigit():
            prod.add(int(first))
except Exception:
    pass

query = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
unexpected = []
for row in csv.reader(query.splitlines()):
    if len(row) < 3:
        continue
    pid = int(row[0].strip())
    name = row[1].strip()
    try:
        memory_mb = float(row[2].strip())
    except ValueError:
        memory_mb = 0.0
    if pid not in prod and memory_mb > 128:
        unexpected.append((pid, name, memory_mb))
if unexpected:
    print(f"unexpected GPU compute workloads: {unexpected}", file=sys.stderr)
    raise SystemExit(1)
PY
}

capture_metadata() {
  local meta="$PROJECT/metadata"
  mkdir -p "$meta"
  date -Is > "$meta/captured_at.txt"
  uname -a > "$meta/uname.txt"
  nvidia-smi > "$meta/nvidia-smi.txt"
  nvidia-smi -q > "$meta/nvidia-smi-q.txt"
  docker image inspect "$VLLM_IMAGE" > "$meta/vllm_image_inspect.json"
  docker image inspect "$SGLANG_IMAGE" > "$meta/sglang_image_inspect.json"
  docker image inspect "$LMCACHE_IMAGE" > "$meta/lmcache_image_inspect.json"
  for spec in "vllm:$VLLM_IMAGE" "sglang:$SGLANG_IMAGE" "lmcache:$LMCACHE_IMAGE"; do
    local label="${spec%%:*}"
    local image="${spec#*:}"
    docker run --rm --entrypoint python3 "$image" - <<'PY' > "$meta/runtime_versions_${label}.txt" 2>&1 || true
import importlib.metadata as md
for name in ["vllm", "sglang", "lmcache", "torch", "transformers", "flashinfer-python", "pynvml"]:
    try:
        print(f"{name}=={md.version(name)}")
    except Exception as exc:
        print(f"{name}=unavailable ({exc})")
PY
  done
  cp "$PROJECT/docker/"Dockerfile* "$meta/"
  cp "$PROJECT/session_files/run_gemma4_runtime_overlay_matrix_20260718.sh" "$meta/"
}

wait_ready() {
  local port="$1"
  local deadline=$((SECONDS + 1200))
  while (( SECONDS < deadline )); do
    if ! docker ps --format '{{.Names}}' | grep -qx "$SERVER_NAME"; then
      docker logs "$SERVER_NAME" 2>&1 | tail -n 240 | tee -a "$LOG" >&2
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  docker logs "$SERVER_NAME" 2>&1 | tail -n 240 | tee -a "$LOG" >&2
  return 1
}

launch_server() {
  local arm="$1"
  local model="$2"
  local port
  docker rm -f "$SERVER_NAME" >/dev/null 2>&1 || true
  docker rm -f "$LMCACHE_SERVER_NAME" >/dev/null 2>&1 || true

  if [[ "$arm" == vllm_apc* ]]; then
    port=8000
    docker run -d --name "$SERVER_NAME" --network host \
      --device nvidia.com/gpu=all --ipc host \
      -e HF_HOME=/cache/huggingface \
      -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
      -e TRANSFORMERS_CACHE=/cache/huggingface \
      -v /datapool/cache/huggingface:/cache/huggingface \
      --entrypoint vllm "$VLLM_IMAGE" \
      serve "$model" --host 127.0.0.1 --port "$port" \
      --served-model-name "$model" --enable-prefix-caching \
      "${VLLM_MOE_ARGS[@]}" \
      --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.88 \
      >/dev/null
  elif [[ "$arm" == sglang_radix_attention* ]]; then
    port=30000
    docker run -d --name "$SERVER_NAME" --network host \
      --device nvidia.com/gpu=all --ipc host \
      -e HF_HOME=/cache/huggingface \
      -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
      -e TRANSFORMERS_CACHE=/cache/huggingface \
      -e USE_HUB_KERNELS=NO -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
      -v /datapool/cache/huggingface:/cache/huggingface \
      --entrypoint python3 "$SGLANG_IMAGE" \
      -m sglang.launch_server --model-path "$model" \
      --host 127.0.0.1 --port "$port" --enable-cache-report --enable-metrics \
      --context-length 4096 --mem-fraction-static 0.88 \
      --dtype float16 --attention-backend triton --sampling-backend pytorch \
      --disable-cuda-graph --disable-piecewise-cuda-graph \
      >/dev/null
  else
    port=8000
    docker run -d --name "$LMCACHE_SERVER_NAME" --network host --ipc host \
      --device nvidia.com/gpu=all \
      --entrypoint lmcache "$LMCACHE_IMAGE" \
      server --host 127.0.0.1 --port "$LMCACHE_ZMQ_PORT" \
      --http-host 127.0.0.1 --http-port "$LMCACHE_HTTP_PORT" \
      --l1-size-gb 20 --eviction-policy LRU --chunk-size 256 \
      >/dev/null
    for _ in $(seq 1 60); do
      if curl -fsS "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck" >/dev/null 2>&1; then
        break
      fi
      if ! docker inspect -f '{{.State.Running}}' "$LMCACHE_SERVER_NAME" 2>/dev/null | grep -qx true; then
        docker logs "$LMCACHE_SERVER_NAME" 2>&1 | tail -n 240 | tee -a "$LOG" >&2
        return 1
      fi
      sleep 2
    done
    docker run -d --name "$SERVER_NAME" --network host \
      --device nvidia.com/gpu=all --ipc host \
      -e HF_HOME=/cache/huggingface \
      -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
      -e TRANSFORMERS_CACHE=/cache/huggingface \
      -e USE_HUB_KERNELS=NO -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
      -v /datapool/cache/huggingface:/cache/huggingface \
      -v "$PROJECT:/workspace/project" \
      --entrypoint vllm "$LMCACHE_IMAGE" \
      serve "$model" --host 127.0.0.1 --port "$port" \
      --served-model-name "$model" --no-enable-prefix-caching \
      --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":${LMCACHE_ZMQ_PORT}}}" \
      "${VLLM_MOE_ARGS[@]}" \
      --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.82 \
      >/dev/null
  fi
  if ! wait_ready "$port"; then
    return 1
  fi
  echo "$port"
}

cell_complete() {
  local out_dir="$1"
  local baseline="$2"
  python3 - "$out_dir" "$baseline" "$N_REQUESTS" "$MODE" <<'PY'
import json
from pathlib import Path
import sys

out_dir = Path(sys.argv[1])
baseline = sys.argv[2]
expected = int(sys.argv[3])
mode = sys.argv[4]
files = list(out_dir.glob("benchmark_*.json"))
if len(files) != 1:
    raise SystemExit(1)
data = json.loads(files[0].read_text())
metrics = data.get(baseline) or {}
ok = (
    int(metrics.get("requests_seen", -1)) == expected
    and metrics.get("energy_source") not in (None, "unavailable")
    and metrics.get("gpu_energy_j") is not None
)
if baseline.endswith("_shadowkv_plus"):
    ok = ok and (
        metrics.get("admission_controller_enabled") is True
        and int(metrics.get("admission_plans_total", -1)) == expected
        and int(metrics.get("admission_allow_total", 0))
        + int(metrics.get("admission_bypass_total", 0)) == expected
        and int(metrics.get("admission_runtime_cache_reset_failures", 0)) == 0
        and metrics.get("admission_enforcement_mode") == "write_through_admission"
    )
if mode == "smoke":
    if baseline.startswith("vllm_apc"):
        ok = ok and float(metrics.get("vllm_local_cache_hit_tokens_delta", 0)) > 0
    elif baseline.startswith("sglang_radix_attention"):
        ok = ok and float(metrics.get("cached_tokens_total", 0)) > 0
    elif baseline.startswith("lmcache"):
        ok = ok and float(metrics.get("vllm_external_prefix_cache_hits_delta", 0)) > 0
raise SystemExit(0 if ok else 1)
PY
}

block_complete() {
  local arm="$1"
  local model_slug="$2"
  local baseline="$arm"
  local dataset prompt_mode out_host
  for dataset in "${DATASETS[@]}"; do
    for prompt_mode in "${PROMPT_MODES[@]}"; do
      out_host="$RESULT_HOST/$model_slug/$dataset/$prompt_mode/$arm"
      if ! cell_complete "$out_host" "$baseline"; then
        return 1
      fi
    done
  done
  return 0
}

run_cell() {
  local arm="$1"
  local model_slug="$2"
  local model="$3"
  local dataset="$4"
  local prompt_mode="$5"
  local port="$6"
  local baseline="$arm"
  local runner_baseline="$arm"
  local extra=()
  if [[ "$arm" == *_shadowkv_plus ]]; then
    # This is the harness's cross-runtime ShadowKV++ admission/policy overlay.
    # External runtimes retain ownership of their caches; write-through mode
    # records policy decisions without claiming native per-request enforcement.
    extra+=(--admission_preset balanced --admission_mode write_through_admission)
  fi
  local out_host="$RESULT_HOST/$model_slug/$dataset/$prompt_mode/$arm"
  local out_container="$RESULT_CONTAINER/$model_slug/$dataset/$prompt_mode/$arm"
  mkdir -p "$out_host"
  if cell_complete "$out_host" "$baseline"; then
    log "SKIP complete arm=$arm model=$model dataset=$dataset mode=$prompt_mode"
    return 0
  fi
  rm -f "$out_host"/benchmark_*.json
  write_status running \
    "arm=$arm" "model_slug=$model_slug" "model=$model" \
    "dataset=$dataset" "prompt_mode=$prompt_mode"
  log "CELL START arm=$arm model=$model dataset=$dataset mode=$prompt_mode requests=$N_REQUESTS"
  docker run --rm --network host --device nvidia.com/gpu=all --ipc host \
    -e HF_HOME=/cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    -e TRANSFORMERS_CACHE=/cache/huggingface \
    -e PYTHONPATH="$SOURCE_CONTAINER/src" \
    -v /datapool/cache/huggingface:/cache/huggingface \
    -v "$PROJECT:/workspace/project" \
    -w "$SOURCE_CONTAINER" \
    --entrypoint python3 "$CLIENT_IMAGE" \
    literature_accurate_baselines/run_runtime_cache_baseline.py \
      --baseline "$runner_baseline" "${extra[@]}" \
      --model "$model" --workload public_dataset \
      --dataset "$dataset" --prompt_mode "$prompt_mode" \
      --n_requests "$N_REQUESTS" --disable_arrival_simulation \
      --max_tokens 1 --temperature 0 --warmup_requests "$WARMUP_REQUESTS" \
      --api_base "http://127.0.0.1:${port}" \
      --server_ready_timeout_s 120 --request_timeout_s 300 \
      --measure_energy --idle_baseline_seconds "$IDLE_BASELINE_SECONDS" \
      --idle_stabilization_seconds "$IDLE_STABILIZATION_SECONDS" \
      --idle_stabilization_tolerance_w 3 \
      --output_dir "$out_container" 2>&1 | tee -a "$LOG"
  cell_complete "$out_host" "$baseline"
  log "CELL DONE arm=$arm model=$model dataset=$dataset mode=$prompt_mode"
}

make_block_plan() {
  python3 - "$PROJECT/block_plan_${MODE}.tsv" "$MODE" <<'PY'
import itertools
import random
import sys

out, mode = sys.argv[1:]
models = [
    ("gemma4_e2b", "google/gemma-4-E2B-it"),
    ("gemma4_e4b", "google/gemma-4-E4B-it"),
    ("gemma4_12b", "google/gemma-4-12B-it"),
    ("gemma4_26b_a4b", "google/gemma-4-26B-A4B-it"),
    ("gemma4_31b", "google/gemma-4-31B-it"),
]
arms = [
    "vllm_apc",
    "vllm_apc_shadowkv_plus",
    "sglang_radix_attention",
    "sglang_radix_attention_shadowkv_plus",
    "lmcache",
    "lmcache_shadowkv_plus",
]
rng = random.Random(20260718 if mode == "full" else 2026071801)
blocks = list(itertools.product(models, arms))
rng.shuffle(blocks)
with open(out, "w", encoding="utf-8") as f:
    f.write("block_index\tmodel_slug\tmodel\tarm\n")
    for idx, ((slug, model), arm) in enumerate(blocks, 1):
        f.write(f"{idx}\t{slug}\t{model}\t{arm}\n")
PY
}

main() {
  write_status preparing
  assert_gpu_exclusive_except_production
  capture_metadata
  make_block_plan
  log "RUN START mode=$MODE requests=$N_REQUESTS"
  docker stop "$PROD_NAME" >/dev/null

  local total_blocks=30
  while IFS=$'\t' read -r block_index model_slug model arm; do
    [[ "$block_index" == "block_index" ]] && continue
    [[ -n "$ONLY_ARM" && "$arm" != "$ONLY_ARM" ]] && continue
    [[ -n "$ONLY_MODEL_SLUG" && "$model_slug" != "$ONLY_MODEL_SLUG" ]] && continue
    if block_complete "$arm" "$model_slug"; then
      log "SKIP BLOCK complete index=$block_index/$total_blocks arm=$arm model=$model"
      continue
    fi
    CURRENT_SERVER_LOG="$RESULT_HOST/server_logs/${block_index}_${model_slug}_${arm}.log"
    write_status launching_server \
      "block_index=$block_index" "total_blocks=$total_blocks" \
      "arm=$arm" "model_slug=$model_slug" "model=$model"
    log "BLOCK START index=$block_index/$total_blocks arm=$arm model=$model"
    local port
    port="$(launch_server "$arm" "$model")"
    for dataset in "${DATASETS[@]}"; do
      for prompt_mode in "${PROMPT_MODES[@]}"; do
        run_cell "$arm" "$model_slug" "$model" "$dataset" "$prompt_mode" "$port"
      done
    done
    stop_server
    log "BLOCK DONE index=$block_index/$total_blocks arm=$arm model=$model"
    sleep 10
  done < "$PROJECT/block_plan_${MODE}.tsv"

  restore_production
  trap - EXIT
  write_status completed "production_restored=true"
  log "RUN COMPLETE mode=$MODE production_restored=true"
}

main "$@"
