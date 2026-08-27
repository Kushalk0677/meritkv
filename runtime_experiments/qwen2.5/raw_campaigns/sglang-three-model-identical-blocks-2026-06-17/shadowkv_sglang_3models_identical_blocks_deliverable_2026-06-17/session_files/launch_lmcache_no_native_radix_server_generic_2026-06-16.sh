#!/usr/bin/env bash
set -euo pipefail

OUT="${LMCACHE_RESULT_DIR:?LMCACHE_RESULT_DIR is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
CONFIG="${LMCACHE_CONFIG_FILE_PATH:?LMCACHE_CONFIG_FILE_PATH is required}"

PORT="${SGLANG_PORT:-30000}"
HOST="${SGLANG_HOST:-127.0.0.1}"
LMCACHE_PORT="${LMCACHE_PORT:-5555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8088}"
LMCACHE_PROM_PORT="${LMCACHE_PROM_PORT:-9098}"
LMCACHE_L1_GB="${LMCACHE_L1_GB:-20}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
DTYPE="${SGLANG_DTYPE:-float16}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-pytorch}"

mkdir -p "$OUT/lmcache_lookup_hash"

cleanup() {
  set +e
  if [[ -n "${SGLANG_PID:-}" ]]; then kill "$SGLANG_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${LMCACHE_PID:-}" ]]; then kill "$LMCACHE_PID" >/dev/null 2>&1 || true; fi
  wait "${SGLANG_PID:-}" "${LMCACHE_PID:-}" >/dev/null 2>&1 || true
}
trap cleanup TERM INT EXIT

lmcache server \
  --host 127.0.0.1 \
  --port "$LMCACHE_PORT" \
  --http-host 127.0.0.1 \
  --http-port "$LMCACHE_HTTP_PORT" \
  --prometheus-port "$LMCACHE_PROM_PORT" \
  --chunk-size 256 \
  --l1-size-gb "$LMCACHE_L1_GB" \
  --l1-init-size-gb 1 \
  --eviction-policy LRU \
  --lookup-hash-log-dir "$OUT/lmcache_lookup_hash" \
  > "$OUT/lmcache_server.log" 2>&1 &
LMCACHE_PID=$!

python3 - <<PY
import time
import urllib.request

url = "http://127.0.0.1:${LMCACHE_HTTP_PORT}/status"
for _ in range(120):
    try:
        urllib.request.urlopen(url, timeout=1).read()
        raise SystemExit(0)
    except Exception:
        time.sleep(1)
raise SystemExit("LMCache HTTP status timeout")
PY

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --enable-lmcache \
  --lmcache-config-file "$CONFIG" \
  --disable-radix-cache \
  --enable-cache-report \
  --enable-metrics \
  --context-length "$CONTEXT_LENGTH" \
  --chunked-prefill-size -1 \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --dtype "$DTYPE" \
  --attention-backend "$ATTENTION_BACKEND" \
  --sampling-backend "$SAMPLING_BACKEND" \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph &
SGLANG_PID=$!

wait "$SGLANG_PID"
