#!/usr/bin/env bash
set -euo pipefail

cd /home/jade_hand/research/shadowkv

MODE="${1:-full}"
if [[ "$MODE" != "deterministic" && "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: $0 [deterministic|smoke|full]" >&2
  exit 2
fi

IMAGE="shadowkv-sglang-native-admission:2026-06-22-counters"
DOCKERFILE="/home/jade_hand/research/shadowkv/session_files/Dockerfile.sglang_2026-06-22-native-admission-counters"
RUN_ID="sglang_native_admission_qwen14b_gate_2026-06-22"
if [[ "$MODE" == "smoke" ]]; then
  RUN_ID="sglang_native_admission_qwen14b_gate_smoke_2026-06-22"
elif [[ "$MODE" == "deterministic" ]]; then
  RUN_ID="sglang_native_admission_hook_deterministic_2026-06-22"
fi

RESULT_ROOT_HOST="/home/jade_hand/research/shadowkv/results_${RUN_ID}"
RESULT_ROOT_CONTAINER="/workspace/shadowkv/results_${RUN_ID}"
LOG="/home/jade_hand/research/shadowkv/run_logs/${RUN_ID}.log"
STATUS="/home/jade_hand/research/shadowkv/run_logs/${RUN_ID}.status"
PLAN_HOST="${RESULT_ROOT_HOST}/job_plan.tsv"

MODEL_SLUG="qwen25_14b"
MODEL="Qwen/Qwen2.5-14B-Instruct"
BASELINES=(sglang_radix_attention sglang_radix_attention_shadowkv_plus)
MODES=(templated rag)
REPS=(1 2 3)
N_REQUESTS=256
MAX_TOKENS=1
RANDOM_SEED=20260622
IDLE_BASELINE_SECONDS=10
IDLE_STABILIZATION_SECONDS=30

if [[ "$MODE" == "smoke" ]]; then
  MODES=(templated)
  REPS=(1)
  N_REQUESTS=48
  IDLE_BASELINE_SECONDS=5
  IDLE_STABILIZATION_SECONDS=5
fi

mkdir -p "$RESULT_ROOT_HOST" "$(dirname "$LOG")"

write_status() {
  local state="$1"
  shift || true
  {
    echo "state=${state}"
    echo "mode=${MODE}"
    echo "run_id=${RUN_ID}"
    echo "updated_at=$(date -Is)"
    echo "result_root=${RESULT_ROOT_HOST}"
    echo "log=${LOG}"
    echo "image=${IMAGE}"
    for kv in "$@"; do echo "$kv"; done
  } > "$STATUS"
}

restore_production() {
  set +e
  docker rm -f shadowkv-native-admission-qwen14b-cell >/dev/null 2>&1 || true
  docker rm -f shadowkv-native-admission-qwen14b-deterministic >/dev/null 2>&1 || true
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

build_image() {
  write_status "building_image"
  echo "[$(date -Is)] BUILD IMAGE ${IMAGE}" | tee -a "$LOG"
  docker build -f "$DOCKERFILE" -t "$IMAGE" . 2>&1 | tee -a "$LOG"
}

snapshot_sources() {
  local snap="${RESULT_ROOT_HOST}/metadata/source_snapshot"
  mkdir -p "$snap"
  local files=(
    "literature_accurate_baselines/adapter_lib.py"
    "literature_accurate_baselines/run_runtime_cache_baseline.py"
    "src/proactive_kv_cache/cache.py"
    "src/proactive_kv_cache/controller.py"
    "src/proactive_kv_cache/metrics.py"
    "src/proactive_kv_cache/utility.py"
    "src/proactive_kv_cache/utility_policy.py"
    "src/proactive_kv_cache/workload.py"
    "session_files/Dockerfile.sglang_2026-06-22-native-admission-counters"
    "session_files/test_sglang_native_admission_hooks_2026_06_22.py"
    "session_files/make_admission_sensitive_trace_2026_06_22.py"
    "session_files/run_shadowkv_sglang_native_admission_qwen14b_gate_2026-06-22.sh"
  )
  for file in "${files[@]}"; do
    mkdir -p "$snap/$(dirname "$file")"
    cp "$file" "$snap/$file"
  done
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git status --short > "$snap/git-status.txt" 2>&1 || true
    git rev-parse HEAD > "$snap/git-head.txt" 2>&1 || true
  else
    echo "/home/jade_hand/research/shadowkv is not a git repository" > "$snap/git-status.txt"
  fi
}

capture_metadata() {
  local meta="${RESULT_ROOT_HOST}/metadata"
  local safe_image
  safe_image="$(printf "%s" "$IMAGE" | tr -c 'A-Za-z0-9._-' '_')"
  mkdir -p "$meta"
  {
    echo "captured_at=$(date -Is)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -a)"
    echo "cwd=$(pwd)"
    echo "image=${IMAGE}"
    echo "dockerfile=${DOCKERFILE}"
  } > "$meta/host.txt"
  nvidia-smi > "$meta/nvidia-smi.txt" 2>&1 || true
  nvidia-smi -q > "$meta/nvidia-smi-q.txt" 2>&1 || true
  docker image inspect "$IMAGE" > "$meta/docker_image_inspect_${safe_image}.json"
  docker run --rm --entrypoint python3 "$IMAGE" -c 'import importlib.metadata as md
for name in ["sglang", "lmcache", "torch", "transformers", "vllm", "flashinfer-python", "numpy"]:
    try:
        print(f"{name}=={md.version(name)}")
    except Exception as exc:
        print(f"{name}=unavailable ({exc})")
' > "$meta/runtime_versions.txt" 2>&1 || true
  docker run --rm --entrypoint python3 "$IMAGE" -m pip freeze > "$meta/pip_freeze.txt" 2>&1 || true
  snapshot_sources
}

run_deterministic_tests() {
  local out_dir="${RESULT_ROOT_HOST}/hook_tests"
  mkdir -p "$out_dir"
  write_status "running_deterministic_hook_tests"
  echo "[$(date -Is)] DETERMINISTIC HOOK TESTS" | tee -a "$LOG"
  docker rm -f shadowkv-native-admission-qwen14b-deterministic >/dev/null 2>&1 || true
  docker run -d --name shadowkv-native-admission-qwen14b-deterministic \
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
    "$IMAGE" \
      -m sglang.launch_server \
      --model-path "$MODEL" \
      --host 127.0.0.1 \
      --port 30000 \
      --enable-cache-report \
      --enable-metrics \
      --context-length 4096 \
      --chunked-prefill-size -1 \
      --mem-fraction-static 0.80 \
      --dtype float16 \
      --attention-backend triton \
      --sampling-backend pytorch \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph >/dev/null

  docker run --rm --network host \
    -v /home/jade_hand/research/shadowkv:/workspace/shadowkv \
    -w /workspace/shadowkv \
    --entrypoint python3 \
    "$IMAGE" \
      session_files/test_sglang_native_admission_hooks_2026_06_22.py \
      --api-base http://127.0.0.1:30000 \
      --model "$MODEL" \
      --output "${RESULT_ROOT_CONTAINER}/hook_tests/hook_test_report.json" 2>&1 | tee -a "$LOG"

  docker logs shadowkv-native-admission-qwen14b-deterministic > "$out_dir/server.log" 2>&1 || true
  docker rm -f shadowkv-native-admission-qwen14b-deterministic >/dev/null 2>&1 || true
}

make_plan() {
  python3 - <<'PY' "$PLAN_HOST" "$RANDOM_SEED" "${MODES[@]}" -- "${BASELINES[@]}" -- "${REPS[@]}"
import itertools
import random
import sys

out = sys.argv[1]
seed = int(sys.argv[2])
parts = sys.argv[3:]
sections = []
current = []
for item in parts:
    if item == "--":
        sections.append(current)
        current = []
    else:
        current.append(item)
sections.append(current)
modes, baselines, reps = sections
rng = random.Random(seed)
blocks = [(int(rep), mode) for rep in reps for mode in modes]
rng.shuffle(blocks)
orders = list(itertools.permutations(baselines))
jobs = []
for block_index, (rep, mode) in enumerate(blocks, 1):
    order = list(orders[(block_index + seed) % len(orders)])
    if rng.random() < 0.5:
        order.reverse()
    block_seed = seed + rep * 1000 + modes.index(mode) * 100
    block_key = f"block_{block_index:02d}:rep_{rep}:{mode}"
    for baseline_position, baseline in enumerate(order, 1):
        jobs.append((len(jobs) + 1, rep, mode, baseline, block_seed, baseline_position, block_index, block_key))
with open(out, "w", encoding="utf-8") as f:
    f.write("job_index\trep\tmode\tbaseline\tseed\tbaseline_position\tblock_index\tblock_key\n")
    for row in jobs:
        f.write("\t".join(str(item) for item in row) + "\n")
print(f"wrote {len(jobs)} jobs to {out}")
PY
}

make_traces() {
  local trace_dir="${RESULT_ROOT_HOST}/traces"
  mkdir -p "$trace_dir"
  for rep in "${REPS[@]}"; do
    for mode in "${MODES[@]}"; do
      local seed=$((RANDOM_SEED + rep * 1000))
      if [[ "$mode" == "rag" ]]; then
        seed=$((seed + 100))
      fi
      python3 session_files/make_admission_sensitive_trace_2026_06_22.py \
        --mode "$mode" \
        --n-requests "$N_REQUESTS" \
        --seed "$seed" \
        --output "${trace_dir}/rep_${rep}_${mode}.jsonl" 2>&1 | tee -a "$LOG"
    done
  done
}

run_one() {
  local job_index="$1"
  local total_jobs="$2"
  local rep="$3"
  local prompt_mode="$4"
  local baseline="$5"
  local seed="$6"
  local baseline_position="$7"
  local block_index="$8"
  local block_key="$9"

  local out_dir_container="${RESULT_ROOT_CONTAINER}/${MODEL_SLUG}/rep_${rep}/${prompt_mode}/${baseline}"
  local out_dir_host="${RESULT_ROOT_HOST}/${MODEL_SLUG}/rep_${rep}/${prompt_mode}/${baseline}"
  local trace_container="${RESULT_ROOT_CONTAINER}/traces/rep_${rep}_${prompt_mode}.jsonl"
  mkdir -p "$out_dir_host"

  write_status "running" \
    "job_index=${job_index}" \
    "total_jobs=${total_jobs}" \
    "model_slug=${MODEL_SLUG}" \
    "model=${MODEL}" \
    "rep=${rep}" \
    "prompt_mode=${prompt_mode}" \
    "baseline=${baseline}" \
    "seed=${seed}" \
    "baseline_position=${baseline_position}" \
    "block_index=${block_index}" \
    "block_key=${block_key}"
  echo "[$(date -Is)] START ${job_index}/${total_jobs} block=${block_index} baseline_pos=${baseline_position} rep=${rep} mode=${prompt_mode} baseline=${baseline}" | tee -a "$LOG"

  docker rm -f shadowkv-native-admission-qwen14b-cell >/dev/null 2>&1 || true
  local admission_args=()
  if [[ "$baseline" == "sglang_radix_attention_shadowkv_plus" ]]; then
    admission_args=(--admission_mode native_sglang_hook)
  fi

  docker run --rm --name shadowkv-native-admission-qwen14b-cell \
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
    "$IMAGE" \
      literature_accurate_baselines/run_runtime_cache_baseline.py \
      --baseline "$baseline" \
      --model "$MODEL" \
      --workload synthetic \
      --variant high_skew \
      --trace_path "$trace_container" \
      --n_requests "$N_REQUESTS" \
      --disable_arrival_simulation \
      --max_tokens "$MAX_TOKENS" \
      --seed "$seed" \
      --launch_server \
      --server_ready_timeout_s 900 \
      --request_timeout_s 300 \
      --warmup_requests 0 \
      --measure_energy \
      --idle_baseline_seconds "$IDLE_BASELINE_SECONDS" \
      --idle_stabilization_seconds "$IDLE_STABILIZATION_SECONDS" \
      --idle_stabilization_tolerance_w 3 \
      --output_dir "$out_dir_container" \
      --python_executable python3 \
      --server_extra_arg=--context-length \
      --server_extra_arg=4096 \
      --server_extra_arg=--chunked-prefill-size \
      --server_extra_arg=-1 \
      --server_extra_arg=--mem-fraction-static \
      --server_extra_arg=0.80 \
      --server_extra_arg=--dtype \
      --server_extra_arg=float16 \
      --server_extra_arg=--attention-backend \
      --server_extra_arg=triton \
      --server_extra_arg=--sampling-backend \
      --server_extra_arg=pytorch \
      --server_extra_arg=--disable-cuda-graph \
      --server_extra_arg=--disable-piecewise-cuda-graph \
      "${admission_args[@]}" 2>&1 | tee -a "$LOG"

  echo "[$(date -Is)] DONE ${job_index}/${total_jobs} block=${block_index} baseline_pos=${baseline_position} rep=${rep} mode=${prompt_mode} baseline=${baseline}" | tee -a "$LOG"
}

aggregate_results() {
  write_status "aggregating"
  python3 - <<'PY' "$RESULT_ROOT_HOST"
import csv
import json
import pathlib
import statistics
import sys
from collections import defaultdict

root = pathlib.Path(sys.argv[1])

def num(row, key):
    value = row.get(key)
    return None if value is None else float(value)

def mean(values):
    clean = [float(v) for v in values if v is not None]
    return statistics.mean(clean) if clean else None

def total(values):
    return sum(int(float(v or 0)) for v in values)

def fmt(value):
    return "n/a" if value is None else f"{float(value):.2f}"

rows = []
for bench_path in sorted(root.glob("qwen25_14b/rep_*/*/*/benchmark_*.json")):
    rel = bench_path.relative_to(root)
    model_slug, rep_s, mode, baseline = rel.parts[:4]
    rep = int(rep_s.replace("rep_", ""))
    data = json.loads(bench_path.read_text())
    cfg = data.get("config", {})
    metrics = data.get(cfg.get("baseline", "")) or data.get(baseline) or {}
    counters = metrics.get("shadowkv_server_counters", {}).get("delta", {})
    rows.append({
        "model_slug": model_slug,
        "model": cfg.get("model"),
        "rep": rep,
        "mode": mode,
        "baseline": baseline,
        "seed": cfg.get("seed"),
        "end_to_end_latency_mean_ms": metrics.get("end_to_end_latency_mean_ms", metrics.get("mean_latency_ms")),
        "end_to_end_latency_p50_ms": metrics.get("end_to_end_latency_p50_ms", metrics.get("p50_latency_ms")),
        "end_to_end_latency_p95_ms": metrics.get("end_to_end_latency_p95_ms", metrics.get("p95_latency_ms")),
        "end_to_end_latency_p99_ms": metrics.get("end_to_end_latency_p99_ms", metrics.get("p99_latency_ms")),
        "end_to_end_throughput_rps": metrics.get("end_to_end_throughput_rps", metrics.get("throughput_rps")),
        "measured_wall_time_s": metrics.get("measured_wall_time_s"),
        "http_latency_mean_ms": metrics.get("http_latency_mean_ms"),
        "http_latency_p95_ms": metrics.get("http_latency_p95_ms"),
        "http_service_throughput_rps": metrics.get("http_service_throughput_rps"),
        "idle_adjusted_joules_per_request": metrics.get("idle_adjusted_joules_per_request"),
        "gpu_energy_j": metrics.get("gpu_energy_j"),
        "prompt_tokens_total": metrics.get("prompt_tokens_total"),
        "completion_tokens_total": metrics.get("completion_tokens_total"),
        "cached_tokens_total": metrics.get("cached_tokens_total"),
        "cached_tokens_mean": metrics.get("cached_tokens_mean"),
        "admission_plans_total": metrics.get("admission_plans_total"),
        "admission_allow_total": metrics.get("admission_allow_total"),
        "admission_bypass_total": metrics.get("admission_bypass_total"),
        "admission_native_hook_bypass_total": metrics.get("admission_native_hook_bypass_total"),
        "admission_native_bypass_store_allowed_total": metrics.get("admission_native_bypass_store_allowed_total"),
        "admission_native_skip_lookup_total": metrics.get("admission_native_skip_lookup_total"),
        "admission_native_skip_write_total": metrics.get("admission_native_skip_write_total"),
        "admission_bypass_store_successes": metrics.get("admission_bypass_store_successes"),
        "admission_reason_counts": json.dumps(metrics.get("admission_reason_counts", {}), sort_keys=True),
        "workload_trace_class_counts": json.dumps(metrics.get("workload_trace_class_counts", {}), sort_keys=True),
        "server_extra_key_controls_total": counters.get("extra_key_controls_total", 0),
        "server_skip_lookup_requested_total": counters.get("skip_lookup_requested_total", 0),
        "server_skip_write_requested_total": counters.get("skip_write_requested_total", 0),
        "server_scheduler_skip_lookup_total": counters.get("scheduler_skip_lookup_total", 0),
        "server_radix_skip_lookup_total": counters.get("radix_skip_lookup_total", 0),
        "server_radix_skip_write_finished_total": counters.get("radix_skip_write_finished_total", 0),
        "server_radix_skip_write_unfinished_total": counters.get("radix_skip_write_unfinished_total", 0),
        "benchmark_path": str(bench_path),
    })

aggregate_csv = root / "aggregate_qwen14b_native_admission_gate.csv"
aggregate_json = root / "aggregate_qwen14b_native_admission_gate.json"
if rows:
    with aggregate_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
aggregate_json.write_text(json.dumps(rows, indent=2) + "\n")

summary_rows = []
by_baseline = defaultdict(list)
for row in rows:
    by_baseline[row["baseline"]].append(row)
for baseline, group in sorted(by_baseline.items()):
    summary_rows.append({
        "baseline": baseline,
        "cells": len(group),
        "mean_end_to_end_latency_ms": mean(num(r, "end_to_end_latency_mean_ms") for r in group),
        "mean_end_to_end_p95_ms": mean(num(r, "end_to_end_latency_p95_ms") for r in group),
        "mean_end_to_end_throughput_rps": mean(num(r, "end_to_end_throughput_rps") for r in group),
        "mean_http_latency_ms": mean(num(r, "http_latency_mean_ms") for r in group),
        "mean_http_p95_ms": mean(num(r, "http_latency_p95_ms") for r in group),
        "mean_idle_adjusted_joules_per_request": mean(num(r, "idle_adjusted_joules_per_request") for r in group),
        "prompt_tokens_total": total(r.get("prompt_tokens_total") for r in group),
        "cached_tokens_total": total(r.get("cached_tokens_total") for r in group),
        "admission_plans_total": total(r.get("admission_plans_total") for r in group),
        "admission_allow_total": total(r.get("admission_allow_total") for r in group),
        "admission_bypass_total": total(r.get("admission_bypass_total") for r in group),
        "admission_native_skip_lookup_total": total(r.get("admission_native_skip_lookup_total") for r in group),
        "admission_native_skip_write_total": total(r.get("admission_native_skip_write_total") for r in group),
        "server_skip_lookup_requested_total": total(r.get("server_skip_lookup_requested_total") for r in group),
        "server_skip_write_requested_total": total(r.get("server_skip_write_requested_total") for r in group),
        "server_radix_skip_lookup_total": total(r.get("server_radix_skip_lookup_total") for r in group),
        "server_radix_skip_write_total": total(r.get("server_radix_skip_write_finished_total") for r in group) + total(r.get("server_radix_skip_write_unfinished_total") for r in group),
    })

summary_csv = root / "summary_by_baseline_qwen14b_native_admission_gate.csv"
summary_json = root / "summary_by_baseline_qwen14b_native_admission_gate.json"
if summary_rows:
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
summary_json.write_text(json.dumps(summary_rows, indent=2) + "\n")

paired = []
index = {(r["rep"], r["mode"], r["baseline"]): r for r in rows}
for radix in rows:
    if radix["baseline"] != "sglang_radix_attention":
        continue
    other = index.get((radix["rep"], radix["mode"], "sglang_radix_attention_shadowkv_plus"))
    if not other:
        continue
    base_lat = num(radix, "end_to_end_latency_mean_ms")
    other_lat = num(other, "end_to_end_latency_mean_ms")
    base_p95 = num(radix, "end_to_end_latency_p95_ms")
    other_p95 = num(other, "end_to_end_latency_p95_ms")
    base_rps = num(radix, "end_to_end_throughput_rps")
    other_rps = num(other, "end_to_end_throughput_rps")
    paired.append({
        "rep": radix["rep"],
        "mode": radix["mode"],
        "comparison": "sglang_radix_attention_shadowkv_plus_vs_sglang_radix_attention",
        "end_to_end_latency_delta_pct": ((other_lat - base_lat) / base_lat * 100.0) if base_lat else None,
        "end_to_end_p95_delta_pct": ((other_p95 - base_p95) / base_p95 * 100.0) if base_p95 else None,
        "end_to_end_throughput_delta_pct": ((other_rps - base_rps) / base_rps * 100.0) if base_rps else None,
        "cached_token_delta": int(other.get("cached_tokens_total") or 0) - int(radix.get("cached_tokens_total") or 0),
        "skip_lookup_delta": int(other.get("admission_native_skip_lookup_total") or 0),
        "skip_write_delta": int(other.get("admission_native_skip_write_total") or 0),
    })

paired_csv = root / "paired_deltas_vs_native_radix_qwen14b_native_admission_gate.csv"
paired_json = root / "paired_deltas_vs_native_radix_qwen14b_native_admission_gate.json"
if paired:
    with paired_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        writer.writeheader()
        writer.writerows(paired)
paired_json.write_text(json.dumps(paired, indent=2) + "\n")

hook_report_path = root / "hook_tests" / "hook_test_report.json"
hook_report = json.loads(hook_report_path.read_text()) if hook_report_path.exists() else {"status": "missing"}
gate_passed = (
    hook_report.get("status") == "passed"
    and any(int(row.get("admission_native_skip_write_total") or 0) > 0 for row in rows)
    and any(int(row.get("server_skip_write_requested_total") or 0) > 0 for row in rows)
)

paired_summary = {
    "paired_cells": len(paired),
    "mean_end_to_end_latency_delta_pct": mean(r.get("end_to_end_latency_delta_pct") for r in paired),
    "mean_end_to_end_p95_delta_pct": mean(r.get("end_to_end_p95_delta_pct") for r in paired),
    "mean_end_to_end_throughput_delta_pct": mean(r.get("end_to_end_throughput_delta_pct") for r in paired),
    "cached_token_delta_total": total(r.get("cached_token_delta") for r in paired),
    "skip_lookup_total": total(r.get("skip_lookup_delta") for r in paired),
    "skip_write_total": total(r.get("skip_write_delta") for r in paired),
}
(root / "paired_delta_summary_qwen14b_native_admission_gate.json").write_text(json.dumps(paired_summary, indent=2) + "\n")

lines = [
    "# SGLang Native ShadowKV++ Admission Qwen14B Gate Summary",
    "",
    f"Rows: {len(rows)}",
    f"Hook tests: `{hook_report.get('status')}`",
    f"Gate passed: `{gate_passed}`",
    "",
    "Primary metric: end-to-end latency around ShadowKV planning, server request, and feedback. HTTP/server latency is diagnostic only.",
    "",
    "## By Baseline",
    "",
    "| Baseline | Cells | E2E mean ms | E2E P95 ms | E2E RPS | HTTP mean ms | HTTP P95 ms | Cached tokens | Plans | Allows | Bypasses | Skip lookup | Skip write | Server skip lookup | Server skip write |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in summary_rows:
    lines.append(
        f"| {row['baseline']} | {row['cells']} | {fmt(row['mean_end_to_end_latency_ms'])} | "
        f"{fmt(row['mean_end_to_end_p95_ms'])} | {fmt(row['mean_end_to_end_throughput_rps'])} | "
        f"{fmt(row['mean_http_latency_ms'])} | {fmt(row['mean_http_p95_ms'])} | {row['cached_tokens_total']} | "
        f"{row['admission_plans_total']} | {row['admission_allow_total']} | {row['admission_bypass_total']} | "
        f"{row['admission_native_skip_lookup_total']} | {row['admission_native_skip_write_total']} | "
        f"{row['server_skip_lookup_requested_total']} | {row['server_skip_write_requested_total']} |"
    )
lines.extend([
    "",
    "## Paired ShadowKV++ vs Native Radix",
    "",
    f"- Paired cells: `{paired_summary['paired_cells']}`",
    f"- Mean E2E latency delta: `{fmt(paired_summary['mean_end_to_end_latency_delta_pct'])}%`",
    f"- Mean E2E P95 delta: `{fmt(paired_summary['mean_end_to_end_p95_delta_pct'])}%`",
    f"- Mean E2E throughput delta: `{fmt(paired_summary['mean_end_to_end_throughput_delta_pct'])}%`",
    f"- Cached token delta total: `{paired_summary['cached_token_delta_total']}`",
    f"- Native hook skip-lookups: `{paired_summary['skip_lookup_total']}`",
    f"- Native hook skip-writes: `{paired_summary['skip_write_total']}`",
    "",
    "## Packaging Checks",
    "",
    "- Source snapshot: `metadata/source_snapshot/`",
    "- Runtime versions: `metadata/runtime_versions.txt`",
    "- Docker image metadata filename uses only Windows-safe characters.",
    "- Profiler traces are not included.",
])
(root / "SUMMARY_QWEN14B_NATIVE_ADMISSION_GATE.md").write_text("\n".join(lines) + "\n")
print(root / "SUMMARY_QWEN14B_NATIVE_ADMISSION_GATE.md")
print(json.dumps({"rows": len(rows), "paired": len(paired), "gate_passed": gate_passed}, indent=2))
PY
}

main() {
  : > "$LOG"
  write_status "starting"
  echo "[$(date -Is)] RUN START mode=${MODE} image=${IMAGE} result_root=${RESULT_ROOT_HOST}" | tee -a "$LOG"
  build_image
  capture_metadata | tee -a "$LOG"
  stop_production_for_gpu
  run_deterministic_tests
  if [[ "$MODE" == "deterministic" ]]; then
    write_status "completed" "completed_at=$(date -Is)"
    echo "[$(date -Is)] DETERMINISTIC DONE result_root=${RESULT_ROOT_HOST}" | tee -a "$LOG"
    return
  fi
  make_plan | tee -a "$LOG"
  make_traces
  local total_jobs
  total_jobs="$(($(wc -l < "$PLAN_HOST") - 1))"
  tail -n +2 "$PLAN_HOST" | while IFS=$'\t' read -r job_index rep prompt_mode baseline seed baseline_position block_index block_key; do
    run_one "$job_index" "$total_jobs" "$rep" "$prompt_mode" "$baseline" "$seed" "$baseline_position" "$block_index" "$block_key"
  done
  aggregate_results | tee -a "$LOG"
  write_status "completed" "total_jobs=${total_jobs}" "completed_at=$(date -Is)"
  echo "[$(date -Is)] RUN DONE mode=${MODE} result_root=${RESULT_ROOT_HOST}" | tee -a "$LOG"
}

main "$@"
