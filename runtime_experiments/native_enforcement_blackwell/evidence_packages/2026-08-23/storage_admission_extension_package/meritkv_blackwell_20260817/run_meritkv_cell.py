#!/usr/bin/env python3
"""Instrument the preserved external-runtime runner for the frozen MeritKV arms."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE = PROJECT / "source_snapshot"
sys.path.insert(0, str(SOURCE))

from literature_accurate_baselines import adapter_lib as adapter  # noqa: E402
from literature_accurate_baselines import run_runtime_cache_baseline as base  # noqa: E402


ARM = os.environ.get("MERITKV_ARM", "")
VALID_ARMS = {
    "native_baseline",
    "meritkv_write_through",
    "meritkv_enforced",
    "forced_recompute",
}
if ARM not in VALID_ARMS:
    raise SystemExit(f"MERITKV_ARM must be one of {sorted(VALID_ARMS)}")

DECISIONS: list[dict[str, Any]] = []
CALLS: list[dict[str, Any]] = []


def _percentile(values: list[float], q: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return values[low] * (1.0 - fraction) + values[high] * fraction


def _extract_output(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    delta = first.get("delta") or {}
    for key in ("content", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str):
            return value
    text = first.get("text")
    return text if isinstance(text, str) else ""


def streaming_invoke(
    self: adapter.OpenAICompatClient,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    extra_body: dict[str, Any] | None = None,
) -> adapter.ExternalCallResult:
    if self.endpoint == "chat":
        path = "/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    else:
        path = "/v1/completions"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    payload.update(extra_body or {})
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        self.api_base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = dict(chunk["usage"])
            piece = _extract_output(chunk)
            if piece:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(piece)
    end = time.perf_counter()
    output = "".join(pieces)
    prompt_details = usage.get("prompt_tokens_details") or {}
    result = adapter.ExternalCallResult(
        request_id=len(CALLS),
        latency_ms=(end - start) * 1000.0,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
        cached_tokens=int(prompt_details.get("cached_tokens") or 0),
    )
    result.ttft_ms = ((first_token_at or end) - start) * 1000.0
    result.output_sha256 = hashlib.sha256(output.encode("utf-8")).hexdigest()
    result.output_text = output
    CALLS.append(
        {
            "request_index": len(CALLS),
            "ttft_ms": result.ttft_ms,
            "end_to_end_latency_ms": result.latency_ms,
            "http_latency_ms": result.latency_ms,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cached_tokens": result.cached_tokens,
            "output_sha256": result.output_sha256,
        }
    )
    return result


original_plan = adapter.ExternalAdmissionController.plan
original_record = adapter.ExternalAdmissionController.record_after_request
original_should_store = adapter.ExternalAdmissionController.should_store_after_bypass
original_summarize = base.summarize_external_results
original_save = base.save_summary


def traced_plan(self: adapter.ExternalAdmissionController, prompt: str, metadata=None):
    started = time.perf_counter()
    plan, tokens = original_plan(self, prompt, metadata=metadata)
    DECISIONS.append(
        {
            "request_index": len(CALLS),
            "strategy": str(plan.strategy),
            "reason": str(plan.reason),
            "utility_score": float(plan.score),
            "reusable_prefix_tokens": int(getattr(plan, "reusable_prefix_tokens", 0) or 0),
            "token_count": len(tokens),
            "shared_prefix_hint_tokens": (metadata or {}).get("shared_prefix_hint_tokens"),
            "shadowkv_trace_class": (metadata or {}).get("shadowkv_trace_class"),
            "reuse_family": (metadata or {}).get("reuse_family"),
            "skip_lookup": plan.strategy == "bypass" and ARM == "meritkv_enforced",
            "skip_write": plan.strategy == "bypass" and ARM == "meritkv_enforced",
            "planning_latency_ms": (time.perf_counter() - started) * 1000.0,
        }
    )
    return plan, tokens


def traced_record(self, tokens, plan, metadata=None, *, result=None, allow_bypass_store=False):
    # Policy state is deliberately arm-independent. Runtime measurements do not
    # retune the controller, and both MeritKV arms receive the same logical
    # write-through bookkeeping. The enforced arm's physical SGLang request is
    # still skip-lookup+skip-write because the hook decision was made before
    # this feedback call.
    if ARM in {"meritkv_write_through", "meritkv_enforced"}:
        had_instance_override = "should_store_after_bypass" in self.__dict__
        previous_override = self.__dict__.get("should_store_after_bypass")
        self.should_store_after_bypass = original_should_store.__get__(self, type(self))
        try:
            stored = original_record(
                self,
                tokens,
                plan,
                metadata=metadata,
                result=None,
                allow_bypass_store=True,
            )
        finally:
            if had_instance_override:
                self.should_store_after_bypass = previous_override
            else:
                del self.should_store_after_bypass
    else:
        stored = original_record(
            self,
            tokens,
            plan,
            metadata=metadata,
            result=None,
            allow_bypass_store=allow_bypass_store,
        )
    if DECISIONS:
        DECISIONS[-1]["stored_after_request"] = bool(stored)
        if result is not None:
            DECISIONS[-1]["cached_tokens"] = int(result.cached_tokens or 0)
            DECISIONS[-1]["output_sha256"] = getattr(result, "output_sha256", None)
            DECISIONS[-1]["ttft_ms"] = getattr(result, "ttft_ms", None)
            DECISIONS[-1]["end_to_end_latency_ms"] = result.latency_ms
    return stored


def enriched_summarize(results, *, total_wall_time_s=None, extra_metrics=None):
    summary = original_summarize(
        results,
        total_wall_time_s=total_wall_time_s,
        extra_metrics=extra_metrics,
    )
    ttfts = [float(getattr(result, "ttft_ms", result.latency_ms)) for result in results]
    output_hashes = [str(getattr(result, "output_sha256", "")) for result in results]
    summary["ttft_mean_ms"] = sum(ttfts) / max(len(ttfts), 1)
    summary["ttft_p50_ms"] = _percentile(ttfts, 0.50)
    summary["ttft_p95_ms"] = _percentile(ttfts, 0.95)
    summary["output_hashes"] = output_hashes
    summary["output_agreement_digest"] = hashlib.sha256(
        "\n".join(output_hashes).encode("utf-8")
    ).hexdigest()
    decision_rows = [
        {
            "strategy": row["strategy"],
            "reason": row["reason"],
            "utility_score": row["utility_score"],
        }
        for row in DECISIONS
    ]
    summary["policy_decision_digest"] = hashlib.sha256(
        json.dumps(decision_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary["meritkv_arm"] = ARM
    summary["config_frozen"] = True
    summary["retuning_from_results"] = False
    summary["policy_state_mode"] = "frozen_arm_independent"
    return summary


def enriched_save(output_dir: str, filename: str, payload: dict[str, Any]) -> Path:
    payload["meritkv_stage"] = {
        "arm": ARM,
        "config_frozen": True,
        "retuning_from_results": False,
        "raw_command": " ".join(sys.argv),
        "frozen_config_sha256": os.environ.get("MERITKV_FROZEN_CONFIG_SHA256"),
    }
    return original_save(output_dir, filename, payload)


adapter.OpenAICompatClient.invoke = streaming_invoke
adapter.ExternalAdmissionController.plan = traced_plan
adapter.ExternalAdmissionController.record_after_request = traced_record
base.OpenAICompatClient.invoke = streaming_invoke
base.ExternalAdmissionController.plan = traced_plan
base.ExternalAdmissionController.record_after_request = traced_record
base.summarize_external_results = enriched_summarize
base.save_summary = enriched_save

if ARM == "meritkv_enforced":
    adapter.ExternalAdmissionController.should_store_after_bypass = lambda self, tokens, plan, metadata=None: False


def _argument_value(flag: str, default: str) -> str:
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return default


if __name__ == "__main__":
    base.main()
    output_dir = Path(_argument_value("--output_dir", "results_external"))
    trace_path = output_dir / "meritkv_request_trace.jsonl"
    rows = []
    for index, call in enumerate(CALLS):
        row = dict(call)
        if index < len(DECISIONS):
            row.update(DECISIONS[index])
        row["arm"] = ARM
        rows.append(row)
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
