#!/usr/bin/env bash
set -euo pipefail

cd /home/jade_hand/research/shadowkv

IMAGE="shadowkv-sglang-lmcache:2026-06-14-no-native-radix"
MODEL="/cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
MODEL_ARG="/cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
RESULT_ROOT_HOST="/home/jade_hand/research/shadowkv/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14"
RESULT_ROOT_CONTAINER="/workspace/shadowkv/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14"
LOG="/home/jade_hand/research/shadowkv/run_logs/lmcache_no_native_radix_qwen14b_matrix_20260614.log"
STATUS="/home/jade_hand/research/shadowkv/run_logs/lmcache_no_native_radix_qwen14b_matrix_20260614.status"
CONFIG="/home/jade_hand/research/shadowkv/session_files/lmcache_mp_qwen14b_no_native_radix_matrix_2026-06-14.yaml"

DATASETS=(daily_dialog samsum ag_news dolly xsum)
MODES=(templated rag)
BASELINE="lmcache_no_native_radix"
N_REQUESTS=64
WARMUP_REQUESTS=8
MAX_TOKENS=1

mkdir -p "$RESULT_ROOT_HOST" "$(dirname "$LOG")" "$(dirname "$CONFIG")"

cat > "$CONFIG" <<'YAML'
chunk_size: 256
local_cpu: true
max_local_cpu_size: 20.0
use_layerwise: false
save_decode_cache: false
mp_host: 127.0.0.1
mp_port: 5555
YAML

restore_production() {
  set +e
  docker rm -f lmcache-no-native-radix-cell >/dev/null 2>&1 || true
  cd /home/jade_hand/active/services/darwin28b-reason-vllm && docker compose up -d >/dev/null 2>&1 || docker start darwin28b-reason-vllm >/dev/null 2>&1 || true
}
trap restore_production EXIT

run_one() {
  local dataset="$1"
  local mode="$2"
  local out_dir_container="${RESULT_ROOT_CONTAINER}/${dataset}/${mode}/${BASELINE}"
  local out_dir_host="${RESULT_ROOT_HOST}/${dataset}/${mode}/${BASELINE}"

  echo "[$(date -Is)] START dataset=${dataset} mode=${mode}" | tee -a "$LOG"
  printf 'baseline=%s\ndataset=%s\nmode=%s\nstarted_at=%s\n' "$BASELINE" "$dataset" "$mode" "$(date -Is)" > "$STATUS"
  mkdir -p "$out_dir_host"

  docker rm -f lmcache-no-native-radix-cell >/dev/null 2>&1 || true
  docker run --rm --name lmcache-no-native-radix-cell \
    --network host --device nvidia.com/gpu=all --ipc host \
    -e HF_HOME=/cache/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    -e TRANSFORMERS_CACHE=/cache/huggingface \
    -e USE_HUB_KERNELS=NO \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e PYTHONHASHSEED=0 \
    -e LMCACHE_DISABLE_BANNER=1 \
    -e MODEL_PATH="$MODEL" \
    -e LMCACHE_CONFIG_FILE_PATH="/workspace/shadowkv/session_files/lmcache_mp_qwen14b_no_native_radix_matrix_2026-06-14.yaml" \
    -e LMCACHE_RESULT_DIR="$out_dir_container" \
    -v /home/jade_hand/research/shadowkv:/workspace/shadowkv \
    -v /datapool/cache/huggingface:/cache/huggingface \
    -w /workspace/shadowkv \
    --entrypoint python3 \
    "$IMAGE" \
      literature_accurate_baselines/run_runtime_cache_baseline.py \
      --baseline lmcache \
      --lmcache_engine sglang \
      --model "$MODEL_ARG" \
      --workload public_dataset \
      --dataset "$dataset" \
      --prompt_mode "$mode" \
      --n_requests "$N_REQUESTS" \
      --disable_arrival_simulation \
      --max_tokens "$MAX_TOKENS" \
      --launch_server \
      --server_command "bash /workspace/shadowkv/session_files/launch_lmcache_no_native_radix_server_2026-06-14.sh" \
      --server_ready_timeout_s 900 \
      --request_timeout_s 300 \
      --warmup_requests "$WARMUP_REQUESTS" \
      --measure_energy \
      --idle_baseline_seconds 5 \
      --output_dir "$out_dir_container" 2>&1 | tee -a "$LOG"

  python3 - <<'PY' "$out_dir_host"
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
lm = root / "lmcache_server.log"
text = lm.read_text(errors="replace") if lm.exists() else ""
summary = {
    "lmcache_log_bytes": len(text),
    "lmcache_retrieve_events": len(re.findall(r"Retrieved \d+ tokens", text)),
    "lmcache_store_events": len(re.findall(r"Stored \d+ tokens", text)),
    "lmcache_retrieved_token_lines": re.findall(r"Retrieved \d+ tokens[^\n]*", text),
    "lmcache_stored_token_lines": re.findall(r"Stored \d+ tokens[^\n]*", text),
}
(root / "lmcache_log_summary.json").write_text(json.dumps(summary, indent=2))
PY

  echo "[$(date -Is)] DONE dataset=${dataset} mode=${mode}" | tee -a "$LOG"
}

aggregate() {
  python3 - <<'PY' "$RESULT_ROOT_HOST"
import csv, json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
rows = []
for summary_path in sorted(root.glob("*/*/lmcache_no_native_radix/summary_*.json")):
    data = json.loads(summary_path.read_text())
    cfg = data.get("config", {})
    metrics = data.get("metrics", data)
    log_summary_path = summary_path.parent / "lmcache_log_summary.json"
    log_summary = json.loads(log_summary_path.read_text()) if log_summary_path.exists() else {}
    rows.append({
        "dataset": cfg.get("dataset") or summary_path.parents[2].name,
        "mode": cfg.get("prompt_mode") or summary_path.parents[1].name,
        "baseline": "lmcache_no_native_radix",
        "mean_latency_ms": metrics.get("mean_latency_ms"),
        "p95_latency_ms": metrics.get("p95_latency_ms"),
        "throughput_rps": metrics.get("throughput_rps"),
        "cached_tokens_total": metrics.get("cached_tokens_total"),
        "cached_tokens_mean": metrics.get("cached_tokens_mean"),
        "idle_adjusted_joules_per_request": metrics.get("idle_adjusted_joules_per_request"),
        "gpu_energy_joules": metrics.get("gpu_energy_joules"),
        "lmcache_retrieve_events": log_summary.get("lmcache_retrieve_events", 0),
        "lmcache_store_events": log_summary.get("lmcache_store_events", 0),
        "summary_path": str(summary_path),
    })
csv_path = root / "summary_lmcache_no_native_radix_qwen14b_2026-06-14.csv"
json_path = root / "summary_lmcache_no_native_radix_qwen14b_2026-06-14.json"
md_path = root / "summary_lmcache_no_native_radix_qwen14b_2026-06-14.md"
if rows:
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
json_path.write_text(json.dumps(rows, indent=2))
def mean(key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else None
total_cached = sum(int(r.get("cached_tokens_total") or 0) for r in rows)
total_retrieve = sum(int(r.get("lmcache_retrieve_events") or 0) for r in rows)
total_store = sum(int(r.get("lmcache_store_events") or 0) for r in rows)
lines = [
    "# LMCache No Native Radix Qwen14B Summary - 2026-06-14",
    "",
    f"Cells: {len(rows)}",
    f"Mean latency ms: {mean('mean_latency_ms'):.2f}" if mean("mean_latency_ms") is not None else "Mean latency ms: n/a",
    f"P95 latency ms: {mean('p95_latency_ms'):.2f}" if mean("p95_latency_ms") is not None else "P95 latency ms: n/a",
    f"Throughput rps: {mean('throughput_rps'):.2f}" if mean("throughput_rps") is not None else "Throughput rps: n/a",
    f"Cached tokens total: {total_cached}",
    f"LMCache retrieve events: {total_retrieve}",
    f"LMCache store events: {total_store}",
    "",
    "| Dataset | Mode | Mean ms | P95 ms | RPS | Cached tokens | Retrieve events | Store events |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for r in rows:
    lines.append(
        f"| {r['dataset']} | {r['mode']} | {float(r['mean_latency_ms']):.2f} | "
        f"{float(r['p95_latency_ms']):.2f} | {float(r['throughput_rps']):.2f} | "
        f"{int(r.get('cached_tokens_total') or 0)} | {int(r.get('lmcache_retrieve_events') or 0)} | "
        f"{int(r.get('lmcache_store_events') or 0)} |"
    )
md_path.write_text("\\n".join(lines) + "\\n")
print(md_path)
PY
}

main() {
  echo "[$(date -Is)] MATRIX START image=${IMAGE} model=${MODEL}" | tee "$LOG"
  docker image inspect "$IMAGE" >/dev/null
  if docker ps --format '{{.Names}}' | grep -qx darwin28b-reason-vllm; then
    echo "[$(date -Is)] stopping active GPU service darwin28b-reason-vllm" | tee -a "$LOG"
    docker stop darwin28b-reason-vllm | tee -a "$LOG"
  fi

  for dataset in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
      run_one "$dataset" "$mode"
    done
  done

  aggregate | tee -a "$LOG"
  printf 'completed_at=%s\n' "$(date -Is)" > "$STATUS"
  echo "[$(date -Is)] MATRIX DONE result_root=${RESULT_ROOT_HOST}" | tee -a "$LOG"
}

main "$@"
