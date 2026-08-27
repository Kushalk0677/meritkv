#!/usr/bin/env bash
set -euo pipefail

cd /home/jade_hand/research/shadowkv

IMAGE="shadowkv-sglang-lmcache:2026-06-14-no-native-radix"
MODEL="/cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
HOST_RESULT_DIR="/home/jade_hand/research/shadowkv/results_lmcache_no_native_radix_qwen14b_smoke_2026-06-14"
LOG="/home/jade_hand/research/shadowkv/run_logs/lmcache_no_native_radix_qwen14b_smoke_20260614.log"
STATUS="/home/jade_hand/research/shadowkv/run_logs/lmcache_no_native_radix_qwen14b_smoke_20260614.status"
CONFIG="/home/jade_hand/research/shadowkv/session_files/lmcache_mp_qwen14b_no_native_radix_2026-06-14.yaml"

mkdir -p "$HOST_RESULT_DIR" "$(dirname "$LOG")" "$(dirname "$CONFIG")"

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
  docker rm -f lmcache-no-native-radix-smoke >/dev/null 2>&1 || true
  cd /home/jade_hand/active/services/darwin28b-reason-vllm && docker compose up -d >/dev/null 2>&1 || docker start darwin28b-reason-vllm >/dev/null 2>&1 || true
}
trap restore_production EXIT

echo "started_at=$(date -Is)" > "$STATUS"
echo "[$(date -Is)] LMCache no-native-radix smoke start image=${IMAGE}" | tee "$LOG"

docker rm -f lmcache-no-native-radix-smoke >/dev/null 2>&1 || true
if docker ps --format '{{.Names}}' | grep -qx darwin28b-reason-vllm; then
  echo "[$(date -Is)] stopping active GPU service darwin28b-reason-vllm" | tee -a "$LOG"
  docker stop darwin28b-reason-vllm | tee -a "$LOG"
fi

docker run -d --name lmcache-no-native-radix-smoke \
  --network host --device nvidia.com/gpu=all --ipc host \
  -e HF_HOME=/cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/cache/huggingface \
  -e USE_HUB_KERNELS=NO \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e PYTHONHASHSEED=0 \
  -e LMCACHE_DISABLE_BANNER=1 \
  -v /home/jade_hand/research/shadowkv:/workspace/shadowkv \
  -v /datapool/cache/huggingface:/cache/huggingface \
  -w /workspace/shadowkv \
  --entrypoint bash \
  "$IMAGE" \
  -lc '
set -euo pipefail
OUT=/workspace/shadowkv/results_lmcache_no_native_radix_qwen14b_smoke_2026-06-14
mkdir -p "$OUT"
lmcache server \
  --host 127.0.0.1 \
  --port 5555 \
  --http-host 127.0.0.1 \
  --http-port 8088 \
  --prometheus-port 9098 \
  --chunk-size 256 \
  --l1-size-gb 20 \
  --l1-init-size-gb 1 \
  --eviction-policy LRU \
  --lookup-hash-log-dir "$OUT/lmcache_lookup_hash" \
  > "$OUT/lmcache_server.log" 2>&1 &
LMCACHE_PID=$!
python3 - <<PY
import time, urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:8088/status", timeout=1).read()
        print("lmcache_ready")
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("LMCache HTTP status timeout")
PY
python3 -m sglang.launch_server \
  --model-path '"$MODEL"' \
  --host 127.0.0.1 \
  --port 30000 \
  --enable-lmcache \
  --lmcache-config-file /workspace/shadowkv/session_files/lmcache_mp_qwen14b_no_native_radix_2026-06-14.yaml \
  --disable-radix-cache \
  --enable-cache-report \
  --context-length 4096 \
  --chunked-prefill-size -1 \
  --mem-fraction-static 0.72 \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  > "$OUT/sglang_server.log" 2>&1 &
SGLANG_PID=$!
python3 - <<PY
import time, urllib.request
for _ in range(900):
    try:
        urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=1).read()
        print("sglang_ready")
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("SGLang health timeout")
PY
python3 - <<PY
import json, time, urllib.request
out_dir="$OUT"
prefix = "System: controlled true LMCache no-native-Radix smoke. " + ("Shared retrieval validation context. " * 520)
prompt = prefix + "\\nQuestion: produce one short deterministic answer."
payload = {
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1,
    "temperature": 0,
}
def post(path, obj=None):
    data = json.dumps(obj or {}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:30000{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=300).read().decode()
responses = []
for label in ["first_store", "second_lmcache_hit_no_flush", "third_lmcache_hit_no_flush"]:
    started = time.time()
    raw = post("/v1/chat/completions", payload)
    responses.append({"label": label, "elapsed_s": time.time() - started, "raw": json.loads(raw)})
with open(f"{out_dir}/responses.json", "w") as f:
    json.dump(responses, f, indent=2)
print(json.dumps([{"label": r["label"], "elapsed_s": r["elapsed_s"], "usage": r["raw"].get("usage")} for r in responses], indent=2))
PY
curl -s http://127.0.0.1:30000/get_server_info > "$OUT/sglang_server_info.json" || true
curl -s http://127.0.0.1:8088/status > "$OUT/lmcache_status.json" || true
kill "$SGLANG_PID" "$LMCACHE_PID" || true
wait "$SGLANG_PID" "$LMCACHE_PID" || true
'

echo "[$(date -Is)] waiting for smoke container" | tee -a "$LOG"
docker logs -f lmcache-no-native-radix-smoke 2>&1 | tee -a "$LOG" &
LOG_FOLLOW_PID=$!
set +e
CONTAINER_STATUS="$(docker wait lmcache-no-native-radix-smoke)"
echo "$CONTAINER_STATUS" | tee -a "$LOG"
kill "$LOG_FOLLOW_PID" >/dev/null 2>&1 || true
set -e

docker rm -f lmcache-no-native-radix-smoke >/dev/null 2>&1 || true

python3 - <<'PY' "$HOST_RESULT_DIR"
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
summary = {"result_root": str(root), "responses_file": str(root / "responses.json")}
patterns = {
    "lookup_hits": r"lookup|LOOKUP|Lookup",
    "retrieve_hits": r"retrieve|retrieved|Retrieved|RETRIEVE|num_retrieved",
    "store_hits": r"store|stored|Stored|STORE",
    "error_hits": r"error|exception|traceback",
}
for name in ["sglang_server.log", "lmcache_server.log"]:
    path = root / name
    text = path.read_text(errors="replace") if path.exists() else ""
    summary[name] = {"bytes": len(text)}
    for key, pat in patterns.items():
        flags = re.I if key == "error_hits" else 0
        summary[name][key] = len(re.findall(pat, text, flags))
    summary[name]["retrieved_token_lines"] = re.findall(r"Retrieved \\d+ tokens[^\\n]*", text)
    summary[name]["stored_token_lines"] = re.findall(r"Stored \\d+ tokens[^\\n]*", text)
    summary[name]["cached_token_lines"] = re.findall(r"#cached-token: \\d+", text)
try:
    responses = json.loads((root / "responses.json").read_text())
    summary["responses"] = [
        {"label": r.get("label"), "elapsed_s": r.get("elapsed_s"), "usage": (r.get("raw") or {}).get("usage")}
        for r in responses
    ]
except Exception as exc:
    summary["responses_error"] = repr(exc)
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "completed_at=$(date -Is)" >> "$STATUS"
echo "container_status=${CONTAINER_STATUS}" >> "$STATUS"
echo "[$(date -Is)] LMCache no-native-radix smoke done status=${CONTAINER_STATUS}" | tee -a "$LOG"
exit "$CONTAINER_STATUS"
