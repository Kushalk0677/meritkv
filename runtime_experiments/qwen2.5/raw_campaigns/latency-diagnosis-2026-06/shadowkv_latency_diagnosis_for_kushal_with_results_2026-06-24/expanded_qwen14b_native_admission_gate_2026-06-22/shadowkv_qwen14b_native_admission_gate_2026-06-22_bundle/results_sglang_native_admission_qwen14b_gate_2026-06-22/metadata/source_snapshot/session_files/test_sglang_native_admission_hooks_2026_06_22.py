from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from literature_accurate_baselines.adapter_lib import (
    OpenAICompatClient,
    collect_shadowkv_admission_metrics,
    make_sglang_native_admission_extra_key,
    reset_runtime_cache,
    reset_shadowkv_admission_metrics,
)


def _long_prompt(label: str) -> str:
    repeated = " ".join(
        [
            "The native SGLang admission hook must prove per-request cache lookup and cache write control."
            for _ in range(72)
        ]
    )
    return (
        f"Case {label}.\n"
        "Instruction: answer with one short token after reading the whole context.\n"
        f"Context: {repeated}\n"
        "Question: what is the requested verification target?"
    )


def _request(client: OpenAICompatClient, prompt: str, *, extra_key: str | None = None) -> dict:
    extra_body = {"extra_key": extra_key} if extra_key else None
    start = time.perf_counter()
    result = client.invoke(
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,
        timeout_s=300.0,
        extra_body=extra_body,
    )
    return {
        "latency_ms": result.latency_ms,
        "wall_ms": (time.perf_counter() - start) * 1000.0,
        "prompt_tokens": result.prompt_tokens,
        "cached_tokens": result.cached_tokens,
        "extra_key": extra_key,
    }


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic SGLang native-admission hook proof tests.")
    parser.add_argument("--api-base", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAICompatClient(api_base=args.api_base, model=args.model, endpoint="chat")
    client.wait_until_ready(timeout_s=900.0)

    failures: list[str] = []
    tests: dict[str, object] = {}

    reset_runtime_cache(client, "sglang")
    reset_shadowkv_admission_metrics(client)
    prompt = _long_prompt("skip-lookup")
    warm = _request(client, prompt)
    repeat = _request(client, prompt)
    skip_lookup = _request(
        client,
        prompt,
        extra_key=make_sglang_native_admission_extra_key(
            skip_lookup=True,
            skip_write=False,
            tag="deterministic_skip_lookup",
        ),
    )
    metrics_after_skip_lookup = collect_shadowkv_admission_metrics(client)
    tests["skip_lookup"] = {
        "warm": warm,
        "repeat_control": repeat,
        "skip_lookup_request": skip_lookup,
        "server_metrics": metrics_after_skip_lookup,
    }
    _require(repeat["cached_tokens"] > 0, "control repeat did not report cached tokens before skip-lookup test", failures)
    _require(skip_lookup["cached_tokens"] == 0, "skip-lookup request reported cached tokens instead of zero", failures)
    skip_lookup_counters = metrics_after_skip_lookup.get("values") or {}
    _require(
        float(skip_lookup_counters.get("skip_lookup_requested_total", 0)) >= 1,
        "server counter skip_lookup_requested_total did not increment",
        failures,
    )
    _require(
        float(skip_lookup_counters.get("radix_skip_lookup_total", 0)) >= 1,
        "server counter radix_skip_lookup_total did not increment",
        failures,
    )

    reset_runtime_cache(client, "sglang")
    reset_shadowkv_admission_metrics(client)
    prompt = _long_prompt("skip-write")
    skip_write_first = _request(
        client,
        prompt,
        extra_key=make_sglang_native_admission_extra_key(
            skip_lookup=False,
            skip_write=True,
            tag="deterministic_skip_write",
        ),
    )
    repeat_after_skip_write = _request(client, prompt)
    repeat_after_normal_write = _request(client, prompt)
    metrics_after_skip_write = collect_shadowkv_admission_metrics(client)
    tests["skip_write"] = {
        "skip_write_first": skip_write_first,
        "repeat_after_skip_write": repeat_after_skip_write,
        "repeat_after_normal_write": repeat_after_normal_write,
        "server_metrics": metrics_after_skip_write,
    }
    _require(
        repeat_after_skip_write["cached_tokens"] == 0,
        "second request after skip-write hit the cache; skip-write did not suppress insertion",
        failures,
    )
    _require(
        repeat_after_normal_write["cached_tokens"] > 0,
        "third request after normal write did not hit the cache; control cache path is unhealthy",
        failures,
    )
    skip_write_counters = metrics_after_skip_write.get("values") or {}
    _require(
        float(skip_write_counters.get("skip_write_requested_total", 0)) >= 1,
        "server counter skip_write_requested_total did not increment",
        failures,
    )
    _require(
        float(skip_write_counters.get("radix_skip_write_finished_total", 0))
        + float(skip_write_counters.get("radix_skip_write_unfinished_total", 0))
        >= 1,
        "server Radix skip-write counters did not increment",
        failures,
    )

    report = {
        "status": "failed" if failures else "passed",
        "api_base": args.api_base,
        "model": args.model,
        "failures": failures,
        "tests": tests,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
