#!/usr/bin/env bash
set -euo pipefail

OUT="${LMCACHE_RESULT_DIR:?LMCACHE_RESULT_DIR is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
CONFIG="${LMCACHE_CONFIG_FILE_PATH:?LMCACHE_CONFIG_FILE_PATH is required}"

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

python3 - <<'PY'
import time, urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:8088/status", timeout=1).read()
        raise SystemExit(0)
    except Exception:
        time.sleep(1)
raise SystemExit("LMCache HTTP status timeout")
PY

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 30000 \
  --enable-lmcache \
  --lmcache-config-file "$CONFIG" \
  --disable-radix-cache \
  --enable-cache-report \
  --context-length 4096 \
  --chunked-prefill-size -1 \
  --mem-fraction-static 0.72 \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph &
SGLANG_PID=$!

wait "$SGLANG_PID"
