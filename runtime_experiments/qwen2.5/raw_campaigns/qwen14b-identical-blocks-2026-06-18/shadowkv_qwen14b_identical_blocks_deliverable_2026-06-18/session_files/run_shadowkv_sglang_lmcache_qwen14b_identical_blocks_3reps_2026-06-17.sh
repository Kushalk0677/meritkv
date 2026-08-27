#!/usr/bin/env bash
set -euo pipefail

cd /home/jade_hand/research/shadowkv

MODE="${1:-full}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: $0 [smoke|full]" >&2
  exit 2
fi

IMAGE="shadowkv-sglang-lmcache:2026-06-14-no-native-radix"
RUN_ID="sglang_lmcache_shadowkv_qwen14b_identical_blocks_3reps_2026-06-17"
if [[ "$MODE" == "smoke" ]]; then
  RUN_ID="sglang_lmcache_shadowkv_qwen14b_identical_blocks_smoke_2026-06-17"
fi

RESULT_ROOT_HOST="/home/jade_hand/research/shadowkv/results_${RUN_ID}"
RESULT_ROOT_CONTAINER="/workspace/shadowkv/results_${RUN_ID}"
LOG="/home/jade_hand/research/shadowkv/run_logs/${RUN_ID}.log"
STATUS="/home/jade_hand/research/shadowkv/run_logs/${RUN_ID}.status"
PLAN_HOST="${RESULT_ROOT_HOST}/job_plan.tsv"
PLAN_CONTAINER="${RESULT_ROOT_CONTAINER}/job_plan.tsv"
CONFIG_HOST="/home/jade_hand/research/shadowkv/session_files/lmcache_mp_qwen14b_no_native_radix_2026-06-17.yaml"
CONFIG_CONTAINER="/workspace/shadowkv/session_files/lmcache_mp_qwen14b_no_native_radix_2026-06-17.yaml"
LAUNCHER_CONTAINER="/workspace/shadowkv/session_files/launch_lmcache_no_native_radix_server_generic_2026-06-16.sh"

DATASETS=(daily_dialog samsum ag_news dolly xsum)
MODES=(templated rag)
BASELINES=(sglang_radix_attention sglang_radix_attention_shadowkv_plus lmcache_no_native_radix)
MODELS=(
  "qwen25_14b|Qwen/Qwen2.5-14B-Instruct"
)
REPS=(1 2 3)
N_REQUESTS=256
WARMUP_REQUESTS=8
MAX_TOKENS=1
RANDOM_SEED=20260616

if [[ "$MODE" == "smoke" ]]; then
  DATASETS=(daily_dialog)
  MODES=(templated)
  MODELS=("qwen25_14b|Qwen/Qwen2.5-14B-Instruct")
  REPS=(1)
  N_REQUESTS=16
  WARMUP_REQUESTS=2
fi

mkdir -p "$RESULT_ROOT_HOST" "$(dirname "$LOG")" "$(dirname "$CONFIG_HOST")"

cat > "$CONFIG_HOST" <<'YAML'
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
  docker rm -f shadowkv-qwen14b-cell >/dev/null 2>&1 || true
  cd /home/jade_hand/active/services/darwin28b-reason-vllm && docker compose up -d >/dev/null 2>&1 || docker start darwin28b-reason-vllm >/dev/null 2>&1 || true
}
trap restore_production EXIT

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

capture_metadata() {
  local meta="${RESULT_ROOT_HOST}/metadata"
  mkdir -p "$meta"
  {
    echo "captured_at=$(date -Is)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -a)"
    echo "cwd=$(pwd)"
  } > "$meta/host.txt"
  nvidia-smi > "$meta/nvidia-smi.txt" 2>&1 || true
  nvidia-smi -q > "$meta/nvidia-smi-q.txt" 2>&1 || true
  docker image inspect "$IMAGE" > "$meta/docker_image_inspect_${IMAGE//[:\\/]/_}.json"
  docker run --rm --entrypoint python3 "$IMAGE" - <<'PY' > "$meta/runtime_versions.txt" 2>&1 || true
import importlib.metadata as md
for name in ["sglang", "lmcache", "torch", "transformers", "vllm", "flashinfer-python"]:
    try:
        print(f"{name}=={md.version(name)}")
    except Exception as exc:
        print(f"{name}=unavailable ({exc})")
PY
  docker run --rm --entrypoint python3 "$IMAGE" -m pip freeze > "$meta/pip_freeze.txt" 2>&1 || true
  cp "$CONFIG_HOST" "$meta/lmcache_config.yaml"
}

make_plan() {
  python3 - <<'PY' "$PLAN_HOST" "$RANDOM_SEED" "${MODELS[@]}" -- "${DATASETS[@]}" -- "${MODES[@]}" -- "${BASELINES[@]}" -- "${REPS[@]}"
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
models, datasets, modes, baselines, reps = sections
rng = random.Random(seed)
jobs = []
workload_blocks = [
    (int(rep), dataset, mode)
    for rep in reps
    for dataset in datasets
    for mode in modes
]
rng.shuffle(workload_blocks)

baseline_permutations = list(itertools.permutations(baselines))
model_permutations = list(itertools.permutations(models))

def balanced_orders(permutations, count):
    orders = []
    while len(orders) < count:
        shuffled = permutations[:]
        rng.shuffle(shuffled)
        orders.extend(shuffled)
    return orders[:count]

baseline_orders = balanced_orders(baseline_permutations, len(workload_blocks))
model_orders = balanced_orders(model_permutations, len(workload_blocks))

for block_index, ((rep, dataset, mode), baseline_order, model_order) in enumerate(
    zip(workload_blocks, baseline_orders, model_orders),
    1,
):
    block_seed = 20260000 + rep * 1000 + datasets.index(dataset) * 20 + modes.index(mode)
    block_key = f"block_{block_index:02d}:{rep}:{dataset}:{mode}"
    for baseline_position, baseline in enumerate(baseline_order, 1):
        for model_position, model_spec in enumerate(model_order, 1):
            model_slug, model_name = model_spec.split("|", 1)
            jobs.append({
                "model_slug": model_slug,
                "model": model_name,
                "rep": rep,
                "dataset": dataset,
                "mode": mode,
                "baseline": baseline,
                "seed": block_seed,
                "baseline_position": baseline_position,
                "model_position": model_position,
                "block_index": block_index,
                "block_key": block_key,
            })
with open(out, "w", encoding="utf-8") as f:
    f.write("job_index\tmodel_slug\tmodel\trep\tdataset\tmode\tbaseline\tseed\tbaseline_position\tmodel_position\tblock_index\tblock_key\n")
    for idx, job in enumerate(jobs, 1):
        f.write(
            f"{idx}\t{job['model_slug']}\t{job['model']}\t{job['rep']}\t{job['dataset']}\t"
            f"{job['mode']}\t{job['baseline']}\t{job['seed']}\t{job['baseline_position']}\t"
            f"{job['model_position']}\t{job['block_index']}\t{job['block_key']}\n"
        )
print(f"wrote {len(jobs)} jobs to {out}")
PY
}

prefetch_models() {
  write_status "prefetching_models"
  echo "[$(date -Is)] PREFETCH MODELS" | tee -a "$LOG"
  for spec in "${MODELS[@]}"; do
    local model="${spec#*|}"
    echo "[$(date -Is)] PREFETCH ${model}" | tee -a "$LOG"
    docker run --rm --network host \
      -e HF_HOME=/cache/huggingface \
      -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
      -e TRANSFORMERS_CACHE=/cache/huggingface \
      -v /datapool/cache/huggingface:/cache/huggingface \
      --entrypoint python3 \
      "$IMAGE" \
      -c "from huggingface_hub import snapshot_download; print(snapshot_download('${model}', local_files_only=False))" 2>&1 | tee -a "$LOG"
  done
}

run_one() {
  local job_index="$1"
  local total_jobs="$2"
  local model_slug="$3"
  local model="$4"
  local rep="$5"
  local dataset="$6"
  local mode="$7"
  local baseline="$8"
  local seed="$9"
  local baseline_position="${10}"
  local model_position="${11}"
  local block_index="${12}"
  local block_key="${13}"

  local baseline_dir="$baseline"
  local runner_baseline="$baseline"
  if [[ "$baseline" == "lmcache_no_native_radix" ]]; then
    runner_baseline="lmcache"
  fi

  local out_dir_container="${RESULT_ROOT_CONTAINER}/${model_slug}/rep_${rep}/${dataset}/${mode}/${baseline_dir}"
  local out_dir_host="${RESULT_ROOT_HOST}/${model_slug}/rep_${rep}/${dataset}/${mode}/${baseline_dir}"
  mkdir -p "$out_dir_host"

  write_status "running" \
    "job_index=${job_index}" \
    "total_jobs=${total_jobs}" \
    "model_slug=${model_slug}" \
    "model=${model}" \
    "rep=${rep}" \
    "dataset=${dataset}" \
    "prompt_mode=${mode}" \
    "baseline=${baseline}" \
    "seed=${seed}" \
    "baseline_position=${baseline_position}" \
    "model_position=${model_position}" \
    "block_index=${block_index}" \
    "block_key=${block_key}"
  echo "[$(date -Is)] START ${job_index}/${total_jobs} block=${block_index} model_pos=${model_position} baseline_pos=${baseline_position} model=${model_slug} rep=${rep} dataset=${dataset} mode=${mode} baseline=${baseline} seed=${seed} block_key=${block_key}" | tee -a "$LOG"

  docker rm -f shadowkv-qwen14b-cell >/dev/null 2>&1 || true

  local common_docker=(
    docker run --rm --name shadowkv-qwen14b-cell
    --network host --device nvidia.com/gpu=all --ipc host
    -e HF_HOME=/cache/huggingface
    -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub
    -e TRANSFORMERS_CACHE=/cache/huggingface
    -e USE_HUB_KERNELS=NO
    -e FLASHINFER_DISABLE_VERSION_CHECK=1
    -e PYTHONHASHSEED=0
    -e LMCACHE_DISABLE_BANNER=1
    -v /home/jade_hand/research/shadowkv:/workspace/shadowkv
    -v /datapool/cache/huggingface:/cache/huggingface
    -w /workspace/shadowkv
    --entrypoint python3
    "$IMAGE"
    literature_accurate_baselines/run_runtime_cache_baseline.py
    --baseline "$runner_baseline"
    --model "$model"
    --workload public_dataset
    --dataset "$dataset"
    --prompt_mode "$mode"
    --n_requests "$N_REQUESTS"
    --disable_arrival_simulation
    --max_tokens "$MAX_TOKENS"
    --seed "$seed"
    --launch_server
    --server_ready_timeout_s 900
    --request_timeout_s 300
    --warmup_requests "$WARMUP_REQUESTS"
    --measure_energy
    --idle_baseline_seconds 5
    --output_dir "$out_dir_container"
  )

  if [[ "$baseline" == "lmcache_no_native_radix" ]]; then
    docker run --rm --name shadowkv-qwen14b-cell \
      --network host --device nvidia.com/gpu=all --ipc host \
      -e HF_HOME=/cache/huggingface \
      -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
      -e TRANSFORMERS_CACHE=/cache/huggingface \
      -e USE_HUB_KERNELS=NO \
      -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
      -e PYTHONHASHSEED=0 \
      -e LMCACHE_DISABLE_BANNER=1 \
      -e MODEL_PATH="$model" \
      -e LMCACHE_CONFIG_FILE_PATH="$CONFIG_CONTAINER" \
      -e LMCACHE_RESULT_DIR="$out_dir_container" \
      -e MEM_FRACTION_STATIC=0.80 \
      -e CONTEXT_LENGTH=4096 \
      -e SGLANG_DTYPE=float16 \
      -e SGLANG_ATTENTION_BACKEND=triton \
      -e SGLANG_SAMPLING_BACKEND=pytorch \
      -v /home/jade_hand/research/shadowkv:/workspace/shadowkv \
      -v /datapool/cache/huggingface:/cache/huggingface \
      -w /workspace/shadowkv \
      --entrypoint python3 \
      "$IMAGE" \
        literature_accurate_baselines/run_runtime_cache_baseline.py \
        --baseline lmcache \
        --lmcache_engine sglang \
        --model "$model" \
        --workload public_dataset \
        --dataset "$dataset" \
        --prompt_mode "$mode" \
        --n_requests "$N_REQUESTS" \
        --disable_arrival_simulation \
        --max_tokens "$MAX_TOKENS" \
        --seed "$seed" \
        --launch_server \
        --server_command "bash ${LAUNCHER_CONTAINER}" \
        --server_ready_timeout_s 900 \
        --request_timeout_s 300 \
        --warmup_requests "$WARMUP_REQUESTS" \
        --measure_energy \
        --idle_baseline_seconds 5 \
        --output_dir "$out_dir_container" 2>&1 | tee -a "$LOG"
    python3 - <<'PY' "$out_dir_host"
import json
import pathlib
import re
import sys

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
  else
    "${common_docker[@]}" \
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
      --server_extra_arg=--disable-piecewise-cuda-graph 2>&1 | tee -a "$LOG"
  fi

  echo "[$(date -Is)] DONE ${job_index}/${total_jobs} block=${block_index} model_pos=${model_position} baseline_pos=${baseline_position} model=${model_slug} rep=${rep} dataset=${dataset} mode=${mode} baseline=${baseline}" | tee -a "$LOG"
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
rows = []
for bench_path in sorted(root.glob("*/*/*/*/*/benchmark_*.json")):
    rel = bench_path.relative_to(root)
    model_slug, rep_s, dataset, mode, baseline = rel.parts[:5]
    rep = int(rep_s.replace("rep_", ""))
    data = json.loads(bench_path.read_text())
    cfg = data.get("config", {})
    metrics = (
        data.get(baseline)
        or data.get(cfg.get("baseline", ""))
        or data.get("lmcache")
        or data.get("sglang_radix_attention")
        or data.get("sglang_radix_attention_shadowkv_plus")
        or {}
    )
    log_summary_path = bench_path.parent / "lmcache_log_summary.json"
    log_summary = json.loads(log_summary_path.read_text()) if log_summary_path.exists() else {}
    rows.append({
        "model_slug": model_slug,
        "model": cfg.get("model"),
        "rep": rep,
        "dataset": dataset,
        "mode": mode,
        "baseline": baseline,
        "runner_baseline": cfg.get("baseline"),
        "seed": cfg.get("seed"),
        "mean_latency_ms": metrics.get("mean_latency_ms"),
        "p50_latency_ms": metrics.get("p50_latency_ms"),
        "p95_latency_ms": metrics.get("p95_latency_ms"),
        "p99_latency_ms": metrics.get("p99_latency_ms"),
        "throughput_rps": metrics.get("throughput_rps"),
        "idle_adjusted_joules_per_request": metrics.get("idle_adjusted_joules_per_request"),
        "gpu_energy_j": metrics.get("gpu_energy_j"),
        "prompt_tokens_total": metrics.get("prompt_tokens_total"),
        "completion_tokens_total": metrics.get("completion_tokens_total"),
        "cached_tokens_total": metrics.get("cached_tokens_total"),
        "cached_tokens_mean": metrics.get("cached_tokens_mean"),
        "admission_plans_total": metrics.get("admission_plans_total"),
        "admission_allow_total": metrics.get("admission_allow_total"),
        "admission_bypass_total": metrics.get("admission_bypass_total"),
        "admission_bypass_store_successes": metrics.get("admission_bypass_store_successes"),
        "store_successes": metrics.get("store_successes"),
        "lmcache_retrieve_events": log_summary.get("lmcache_retrieve_events", 0),
        "lmcache_store_events": log_summary.get("lmcache_store_events", 0),
        "benchmark_path": str(bench_path),
    })

def num(row, key):
    value = row.get(key)
    return None if value is None else float(value)

def mean(values):
    clean = [float(v) for v in values if v is not None]
    return statistics.mean(clean) if clean else None

def total(values):
    return sum(int(v or 0) for v in values)

full_csv = root / "aggregate_qwen14b_3baselines_3reps_full.csv"
full_json = root / "aggregate_qwen14b_3baselines_3reps_full.json"
if rows:
    with full_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
full_json.write_text(json.dumps(rows, indent=2) + "\n")

summary_rows = []
by_model_baseline = defaultdict(list)
for row in rows:
    by_model_baseline[(row["model_slug"], row["baseline"])].append(row)
for (model_slug, baseline), group in sorted(by_model_baseline.items()):
    summary_rows.append({
        "model_slug": model_slug,
        "baseline": baseline,
        "cells": len(group),
        "mean_latency_ms": mean(num(r, "mean_latency_ms") for r in group),
        "mean_p95_latency_ms": mean(num(r, "p95_latency_ms") for r in group),
        "mean_throughput_rps": mean(num(r, "throughput_rps") for r in group),
        "mean_idle_adjusted_joules_per_request": mean(num(r, "idle_adjusted_joules_per_request") for r in group),
        "prompt_tokens_total": total(r.get("prompt_tokens_total") for r in group),
        "cached_tokens_total": total(r.get("cached_tokens_total") for r in group),
        "admission_plans_total": total(r.get("admission_plans_total") for r in group),
        "admission_allow_total": total(r.get("admission_allow_total") for r in group),
        "admission_bypass_total": total(r.get("admission_bypass_total") for r in group),
        "lmcache_retrieve_events": total(r.get("lmcache_retrieve_events") for r in group),
        "lmcache_store_events": total(r.get("lmcache_store_events") for r in group),
    })

summary_csv = root / "summary_by_model_baseline_qwen14b_3reps.csv"
summary_json = root / "summary_by_model_baseline_qwen14b_3reps.json"
if summary_rows:
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
summary_json.write_text(json.dumps(summary_rows, indent=2) + "\n")

paired = []
index = {(r["model_slug"], r["rep"], r["dataset"], r["mode"], r["baseline"]): r for r in rows}
for r in rows:
    if r["baseline"] != "sglang_radix_attention":
        continue
    key_base = (r["model_slug"], r["rep"], r["dataset"], r["mode"])
    radix = r
    for other in ("sglang_radix_attention_shadowkv_plus", "lmcache_no_native_radix"):
        o = index.get((*key_base, other))
        if not o:
            continue
        base_lat = num(radix, "mean_latency_ms")
        other_lat = num(o, "mean_latency_ms")
        base_p95 = num(radix, "p95_latency_ms")
        other_p95 = num(o, "p95_latency_ms")
        base_rps = num(radix, "throughput_rps")
        other_rps = num(o, "throughput_rps")
        paired.append({
            "model_slug": r["model_slug"],
            "rep": r["rep"],
            "dataset": r["dataset"],
            "mode": r["mode"],
            "comparison": f"{other}_vs_sglang_radix_attention",
            "latency_delta_pct": ((other_lat - base_lat) / base_lat * 100.0) if base_lat else None,
            "p95_delta_pct": ((other_p95 - base_p95) / base_p95 * 100.0) if base_p95 else None,
            "throughput_delta_pct": ((other_rps - base_rps) / base_rps * 100.0) if base_rps else None,
            "cached_token_delta": int(o.get("cached_tokens_total") or 0) - int(radix.get("cached_tokens_total") or 0),
        })
paired_csv = root / "paired_deltas_vs_native_radix_qwen14b_3reps.csv"
paired_json = root / "paired_deltas_vs_native_radix_qwen14b_3reps.json"
if paired:
    with paired_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        writer.writeheader()
        writer.writerows(paired)
paired_json.write_text(json.dumps(paired, indent=2) + "\n")

paired_summary = []
by_model_comparison = defaultdict(list)
for row in paired:
    by_model_comparison[(row["model_slug"], row["comparison"])].append(row)
for (model_slug, comparison), group in sorted(by_model_comparison.items()):
    paired_summary.append({
        "model_slug": model_slug,
        "comparison": comparison,
        "paired_cells": len(group),
        "mean_latency_delta_pct": mean(r.get("latency_delta_pct") for r in group),
        "mean_p95_delta_pct": mean(r.get("p95_delta_pct") for r in group),
        "mean_throughput_delta_pct": mean(r.get("throughput_delta_pct") for r in group),
        "cached_token_delta_total": total(r.get("cached_token_delta") for r in group),
    })
paired_summary_csv = root / "paired_delta_summary_by_model_qwen14b_3reps.csv"
paired_summary_json = root / "paired_delta_summary_by_model_qwen14b_3reps.json"
if paired_summary:
    with paired_summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_summary[0].keys()))
        writer.writeheader()
        writer.writerows(paired_summary)
paired_summary_json.write_text(json.dumps(paired_summary, indent=2) + "\n")

lines = [
    "# SGLang / ShadowKV++ / LMCache Qwen14B 3-Rep Identical-Blocks Summary",
    "",
    f"Rows: {len(rows)}",
    "",
    "Schedule: 30 matched workload blocks. Each block uses one rep/dataset/mode prompt seed across all three baselines, with balanced baseline-order permutations.",
    "",
    "## By Model And Baseline",
    "",
    "| Model | Baseline | Cells | Mean ms | P95 ms | RPS | Idle J/req | Cached tokens | Plans | Allows | Bypasses | LMCache retrieves | LMCache stores |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in summary_rows:
    def fmt(v):
        return "n/a" if v is None else f"{float(v):.2f}"
    lines.append(
        f"| {row['model_slug']} | {row['baseline']} | {row['cells']} | {fmt(row['mean_latency_ms'])} | "
        f"{fmt(row['mean_p95_latency_ms'])} | {fmt(row['mean_throughput_rps'])} | "
        f"{fmt(row['mean_idle_adjusted_joules_per_request'])} | {row['cached_tokens_total']} | "
        f"{row['admission_plans_total']} | {row['admission_allow_total']} | {row['admission_bypass_total']} | "
        f"{row['lmcache_retrieve_events']} | {row['lmcache_store_events']} |"
    )
lines.extend([
    "",
    "## Paired Deltas Vs Native Radix",
    "",
    "| Model | Comparison | Paired cells | Mean latency delta % | Mean P95 delta % | Mean throughput delta % | Cached token delta |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
])
for row in paired_summary:
    def fmt(v):
        return "n/a" if v is None else f"{float(v):.2f}"
    lines.append(
        f"| {row['model_slug']} | {row['comparison']} | {row['paired_cells']} | "
        f"{fmt(row['mean_latency_delta_pct'])} | {fmt(row['mean_p95_delta_pct'])} | "
        f"{fmt(row['mean_throughput_delta_pct'])} | {row['cached_token_delta_total']} |"
    )
(root / "SUMMARY_QWEN14B_3BASELINES_3REPS.md").write_text("\n".join(lines) + "\n")
print(root / "SUMMARY_QWEN14B_3BASELINES_3REPS.md")
print(json.dumps({"rows": len(rows), "summary_rows": len(summary_rows), "paired_rows": len(paired)}, indent=2))
PY
}

main() {
  : > "$LOG"
  write_status "starting"
  echo "[$(date -Is)] RUN START mode=${MODE} image=${IMAGE} result_root=${RESULT_ROOT_HOST}" | tee -a "$LOG"
  docker image inspect "$IMAGE" >/dev/null
  chmod +x /home/jade_hand/research/shadowkv/session_files/launch_lmcache_no_native_radix_server_generic_2026-06-16.sh
  make_plan | tee -a "$LOG"
  capture_metadata | tee -a "$LOG"
  prefetch_models

  if docker ps --format '{{.Names}}' | grep -qx darwin28b-reason-vllm; then
    echo "[$(date -Is)] stopping active GPU service darwin28b-reason-vllm" | tee -a "$LOG"
    docker stop darwin28b-reason-vllm | tee -a "$LOG"
  fi

  local total_jobs
  total_jobs="$(($(wc -l < "$PLAN_HOST") - 1))"
  local line
  tail -n +2 "$PLAN_HOST" | while IFS=$'\t' read -r job_index model_slug model rep dataset prompt_mode baseline seed baseline_position model_position block_index block_key; do
    run_one "$job_index" "$total_jobs" "$model_slug" "$model" "$rep" "$dataset" "$prompt_mode" "$baseline" "$seed" "$baseline_position" "$model_position" "$block_index" "$block_key"
  done

  aggregate_results | tee -a "$LOG"
  write_status "completed" "total_jobs=${total_jobs}" "completed_at=$(date -Is)"
  echo "[$(date -Is)] RUN DONE mode=${MODE} result_root=${RESULT_ROOT_HOST}" | tee -a "$LOG"
}

main "$@"
