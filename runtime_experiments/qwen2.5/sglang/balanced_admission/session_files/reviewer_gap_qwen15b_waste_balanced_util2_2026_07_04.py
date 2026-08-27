from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from literature_accurate_baselines.adapter_lib import (
    ExternalAdmissionController,
    ExternalCallResult,
    ManagedServer,
    OpenAICompatClient,
    build_requests,
    build_sglang_radix_attention_command,
    build_vllm_apc_command,
    build_vllm_no_cache_command,
    collect_shadowkv_admission_metrics,
    collect_vllm_cache_metrics,
    diff_counter_metrics,
    diff_vllm_cache_metrics,
    load_trace_requests,
    make_sglang_native_admission_extra_key,
    model_slug,
    reset_runtime_cache,
    reset_shadowkv_admission_metrics,
    resolve_model,
    summarize_external_results,
    vllm_compat_env_updates,
)


REQUESTED_ENGINES = (
    "vllm_apc",
    "vllm_apc_shadowkv_plus",
    "sglang_radix_attention",
    "sglang_radix_attention_shadowkv_plus",
)

REFERENCE_ENGINES = ("vllm_no_cache_reference", "sglang_no_cache_reference")

ADMISSION_PRESETS = ("conservative", "balanced", "aggressive_prefix", "low_latency")

WORKLOADS = (
    {
        "workload_id": "pathological_semantic_ag_news",
        "dataset": "ag_news",
        "prompt_mode": "semantic",
        "role": "pathological_semantic",
    },
    {
        "workload_id": "pathological_templated_ag_news",
        "dataset": "ag_news",
        "prompt_mode": "templated",
        "role": "pathological_templated",
    },
    {
        "workload_id": "control_templated_samsum",
        "dataset": "samsum",
        "prompt_mode": "templated",
        "role": "control",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def percentile(values: Iterable[float], pct: float) -> float:
    clean = sorted(float(v) for v in values)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * (float(pct) / 100.0)
    low = int(math.floor(rank))
    high = min(low + 1, len(clean) - 1)
    frac = rank - low
    return clean[low] * (1.0 - frac) + clean[high] * frac


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def kv_bytes_per_token(model: str, dtype_bytes: int) -> dict[str, Any]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(resolve_model(model) or model, trust_remote_code=True)
    hidden_size = int(getattr(cfg, "hidden_size"))
    attention_heads = int(getattr(cfg, "num_attention_heads"))
    kv_heads = int(getattr(cfg, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden_size // attention_heads))
    layers = int(getattr(cfg, "num_hidden_layers"))
    bytes_per_token = 2 * layers * kv_heads * head_dim * int(dtype_bytes)
    return {
        "model": resolve_model(model) or model,
        "num_hidden_layers": layers,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "dtype_bytes": int(dtype_bytes),
        "kv_bytes_per_token": int(bytes_per_token),
        "kv_kib_per_token": float(bytes_per_token / 1024.0),
    }


def workload_args(*, model: str, dataset: str, prompt_mode: str, n_requests: int, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        workload="public_dataset",
        variant="high_skew",
        dataset=dataset,
        prompt_mode=prompt_mode,
        resolved_prompt_mode=prompt_mode,
        dataset_split=None,
        n_requests=n_requests,
        simulate_arrivals=False,
        mean_inter_arrival_ms=50.0,
        max_arrival_sleep_ms=0.0,
        seed=seed,
        output_dir="unused",
        max_tokens=1,
        temperature=0.0,
        request_timeout_s=600.0,
    )


def request_to_trace_row(req: Any) -> dict[str, Any]:
    row = {
        "request_id": int(req.request_id),
        "prompt": str(req.prompt),
        "arrival_time": float(req.arrival_time),
        "prompt_sha256": sha256_text(str(req.prompt)),
    }
    if req.metadata:
        row["metadata"] = dict(req.metadata)
    return row


def make_traces(args: argparse.Namespace, output_root: Path) -> dict[str, Path]:
    trace_dir = output_root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for offset, spec in enumerate(WORKLOADS):
        seed = args.seed + offset * 1009
        ns = workload_args(
            model=args.model,
            dataset=spec["dataset"],
            prompt_mode=spec["prompt_mode"],
            n_requests=args.n_requests,
            seed=seed,
        )
        requests = build_requests(ns)
        rows = [request_to_trace_row(req) for req in requests]
        path = trace_dir / f"{spec['workload_id']}.jsonl"
        write_jsonl(path, rows)
        paths[spec["workload_id"]] = path
    return paths


def engine_runtime(engine: str) -> str:
    if engine.startswith("vllm_"):
        return "vllm"
    if engine.startswith("sglang_"):
        return "sglang"
    raise ValueError(f"Unknown engine runtime for {engine}")


def is_shadowkv_engine(engine: str) -> bool:
    return engine.endswith("_shadowkv_plus")


def apply_admission_policy(args: argparse.Namespace, admission: ExternalAdmissionController) -> ExternalAdmissionController:
    preset = str(args.admission_preset or "balanced")
    if preset not in ADMISSION_PRESETS:
        raise ValueError(f"Unknown admission preset: {preset}")
    admission.admission_preset = preset
    if preset == "conservative":
        admission.full_ms_per_token = 0.30
        admission.reuse_overhead_ms = 2.0
        admission.bank.min_match_length = max(int(admission.bank.min_match_length), 16)
    elif preset == "aggressive_prefix":
        admission.full_ms_per_token = 0.45
        admission.reuse_overhead_ms = 0.55
        admission.bank.min_match_length = min(int(admission.bank.min_match_length), 8)
    elif preset == "low_latency":
        admission.full_ms_per_token = 0.50
        admission.reuse_overhead_ms = 0.25
        admission.bank.min_match_length = min(int(admission.bank.min_match_length), 6)
    admission.controller.policy.min_utility_ms = float(args.admission_util_min_ms)
    admission.controller.policy.min_bootstrap_admissions = int(args.admission_min_bootstrap_admissions)
    return admission


def build_sglang_no_cache_command(ns: SimpleNamespace) -> list[str]:
    cmd = [
        ns.python_executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        resolve_model(ns.model) or ns.model,
        "--host",
        ns.host,
        "--port",
        str(ns.port),
        "--disable-radix-cache",
        "--enable-cache-report",
        "--enable-metrics",
    ]
    if ns.tp and ns.tp > 1:
        cmd.extend(["--tp", str(ns.tp)])
    if getattr(ns, "attention_backend", None):
        cmd.extend(["--attention-backend", str(ns.attention_backend)])
    for extra in ns.server_extra_arg:
        cmd.append(extra)
    return cmd


def server_args_for_engine(args: argparse.Namespace, engine: str, runtime: str) -> tuple[SimpleNamespace, list[str], dict[str, str]]:
    server_extra: list[str]
    port = args.vllm_port if runtime == "vllm" else args.sglang_port
    if runtime == "vllm":
        server_extra = [
            "--max-model-len",
            str(args.context_length),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        ns = SimpleNamespace(
            model=args.model,
            host=args.host,
            port=port,
            dtype=args.dtype,
            tp=args.tp,
            server_extra_arg=server_extra,
            python_executable=args.python_executable,
        )
        if engine == "vllm_no_cache_reference":
            return ns, build_vllm_no_cache_command(ns), vllm_compat_env_updates()
        return ns, build_vllm_apc_command(ns), vllm_compat_env_updates()

    server_extra = [
        "--context-length",
        str(args.context_length),
        "--chunked-prefill-size",
        "-1",
        "--mem-fraction-static",
        str(args.sglang_mem_fraction_static),
        "--dtype",
        args.dtype,
        "--sampling-backend",
        "pytorch",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
    ]
    ns = SimpleNamespace(
        model=args.model,
        host=args.host,
        port=port,
        tp=args.tp,
        attention_backend=args.sglang_attention_backend,
        server_extra_arg=server_extra,
        python_executable=args.python_executable,
    )
    if engine == "sglang_no_cache_reference":
        return ns, build_sglang_no_cache_command(ns), {}
    return ns, build_sglang_radix_attention_command(ns), {}


def load_reference_latencies(output_root: Path, workload_id: str, runtime: str) -> dict[int, float]:
    engine = "vllm_no_cache_reference" if runtime == "vllm" else "sglang_no_cache_reference"
    path = output_root / "per_request" / workload_id / f"{engine}.jsonl"
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {int(row["request_id"]): float(row["end_to_end_latency_ms"]) for row in rows}


def request_metrics_delta(runtime: str, client: OpenAICompatClient) -> dict[str, Any] | None:
    if runtime == "vllm":
        return collect_vllm_cache_metrics(client)
    return None


def diff_request_metrics(runtime: str, before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    if runtime == "vllm" and before is not None and after is not None:
        return diff_vllm_cache_metrics(before, after)
    return {"available": False, "delta": {}}


def cache_tokens_for_request(runtime: str, result: ExternalCallResult, diff: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    if runtime == "vllm":
        query_tokens = int(round(float(diff.get("prefix_cache_queries_delta") or 0.0)))
        cached_tokens = int(round(float(diff.get("prompt_tokens_cached_delta") or 0.0)))
        if cached_tokens <= 0:
            cached_tokens = int(round(float(diff.get("local_cache_hit_tokens_delta") or 0.0)))
        if query_tokens <= 0:
            query_tokens = int(result.prompt_tokens or 0)
        return query_tokens, cached_tokens, {
            "vllm_prefix_cache_queries_delta": diff.get("prefix_cache_queries_delta", 0.0),
            "vllm_prefix_cache_hits_delta": diff.get("prefix_cache_hits_delta", 0.0),
            "vllm_prompt_tokens_cached_delta": diff.get("prompt_tokens_cached_delta", 0.0),
            "vllm_local_cache_hit_tokens_delta": diff.get("local_cache_hit_tokens_delta", 0.0),
        }

    query_tokens = int(result.prompt_tokens or 0)
    cached_tokens = int(result.cached_tokens or 0)
    return query_tokens, cached_tokens, {}


def update_admission_after_request(
    *,
    admission: ExternalAdmissionController | None,
    admission_mode: str | None,
    plan: Any,
    tokens: tuple[int, ...],
    metadata: dict[str, Any],
    result: ExternalCallResult,
    native_bypass_store_allowed: bool,
) -> tuple[float, bool]:
    if admission is None or plan is None:
        return 0.0, False
    start = time.perf_counter()
    stored = admission.record_after_request(
        tokens,
        plan,
        metadata=metadata,
        result=result,
        allow_bypass_store=(
            admission_mode == "write_through_admission"
            or (admission_mode == "native_sglang_hook" and native_bypass_store_allowed)
        ),
    )
    return (time.perf_counter() - start) * 1000.0, bool(stored)


def run_engine(
    *,
    args: argparse.Namespace,
    output_root: Path,
    workload_spec: dict[str, str],
    trace_path: Path,
    engine: str,
    kv_info: dict[str, Any],
) -> dict[str, Any]:
    runtime = engine_runtime(engine)
    work_id = workload_spec["workload_id"]
    cell_dir = output_root / "cells" / work_id / engine
    log_dir = output_root / "server_logs" / work_id / engine
    per_request_path = output_root / "per_request" / work_id / f"{engine}.jsonl"
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    server_ns, command, env_updates = server_args_for_engine(args, engine, runtime)
    env_updates = dict(env_updates)
    env_updates.setdefault("HF_HOME", os.environ.get("HF_HOME", "/cache/huggingface"))
    env_updates.setdefault("HUGGINGFACE_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", "/cache/huggingface/hub"))
    env_updates.setdefault("TRANSFORMERS_CACHE", os.environ.get("TRANSFORMERS_CACHE", "/cache/huggingface"))
    env_updates.setdefault("USE_HUB_KERNELS", "NO")
    env_updates.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env_updates.setdefault("PYTHONHASHSEED", "0")

    admission = None
    admission_mode = None
    admission_metrics = {
        "admission_selected_preset": None,
        "admission_policy_util_min_ms": None,
        "admission_policy_min_bootstrap_admissions": None,
        "admission_full_ms_per_token": None,
        "admission_reuse_overhead_ms": None,
        "admission_bank_min_match_length": None,
        "admission_plans_total": 0,
        "admission_allow_total": 0,
        "admission_bypass_total": 0,
        "admission_write_through_bypass_total": 0,
        "admission_native_hook_bypass_total": 0,
        "admission_native_skip_lookup_total": 0,
        "admission_native_skip_write_total": 0,
        "admission_native_bypass_store_allowed_total": 0,
        "admission_bypass_store_successes": 0,
        "shadowkv_planning_latency_total_ms": 0.0,
        "shadowkv_feedback_latency_total_ms": 0.0,
    }
    admission_reason_counts: dict[str, int] = {}
    admission_strategy_counts: dict[str, int] = {}
    shared_prefix_token_cache: dict[str, int] = {}
    if is_shadowkv_engine(engine):
        admission = ExternalAdmissionController(resolve_model(args.model) or args.model, runtime=runtime)
        admission = apply_admission_policy(args, admission)
        admission_mode = "native_sglang_hook" if runtime == "sglang" else "write_through_admission"
        admission_metrics["admission_selected_preset"] = str(args.admission_preset)
        admission_metrics["admission_policy_util_min_ms"] = float(admission.controller.policy.min_utility_ms)
        admission_metrics["admission_policy_min_bootstrap_admissions"] = int(admission.controller.policy.min_bootstrap_admissions)
        admission_metrics["admission_full_ms_per_token"] = float(admission.full_ms_per_token)
        admission_metrics["admission_reuse_overhead_ms"] = float(admission.reuse_overhead_ms)
        admission_metrics["admission_bank_min_match_length"] = int(admission.bank.min_match_length)

    server = ManagedServer(
        command,
        cwd=str(ROOT),
        env_updates=env_updates,
        stdout_path=log_dir / "stdout.log",
        stderr_path=log_dir / "stderr.log",
    )
    requests = load_trace_requests(str(trace_path))
    client = OpenAICompatClient(api_base=f"http://{args.host}:{server_ns.port}", model=resolve_model(args.model) or args.model, endpoint="chat")
    reference_latencies = load_reference_latencies(output_root, work_id, runtime)
    start_wall = time.perf_counter()
    per_rows: list[dict[str, Any]] = []
    call_results: list[ExternalCallResult] = []
    shadowkv_server_metrics_before: dict[str, Any] | None = None
    shadowkv_server_metrics_after: dict[str, Any] | None = None
    error: str | None = None

    try:
        server.start()
        try:
            client.wait_until_ready(timeout_s=args.server_ready_timeout_s)
        except Exception:
            raise RuntimeError(f"server did not become ready for {engine}; logs:\n{server.tail_logs()}")
        time.sleep(args.post_ready_sleep_s)
        for warmup_index in range(max(int(args.warmup_requests or 0), 0)):
            client.invoke(
                prompt=f"Warmup request {warmup_index}. Reply with ok.",
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.request_timeout_s,
            )
        if args.warmup_requests:
            reset_runtime_cache(client, runtime)
        if runtime == "sglang":
            reset_shadowkv_admission_metrics(client)
            shadowkv_server_metrics_before = collect_shadowkv_admission_metrics(client)

        for idx, req in enumerate(requests):
            request_wall_start = time.perf_counter()
            metadata = dict(req.metadata or {})
            metadata["arrival_time"] = req.arrival_time
            plan = None
            tokens: tuple[int, ...] = ()
            request_extra_body: dict[str, Any] | None = None
            native_bypass_store_allowed = False
            if admission is not None:
                prompt_hash = sha256_text(req.prompt)
                shared_prefix_text = metadata.get("shared_prefix_text")
                if shared_prefix_text and shared_prefix_text not in shared_prefix_token_cache:
                    shared_prefix_token_cache[str(shared_prefix_text)] = len(admission.tokenize(str(shared_prefix_text)))
                if shared_prefix_text:
                    metadata["shared_prefix_hint_tokens"] = shared_prefix_token_cache[str(shared_prefix_text)]
                planning_start = time.perf_counter()
                plan, tokens = admission.plan(req.prompt, metadata=metadata)
                planning_ms = (time.perf_counter() - planning_start) * 1000.0
                admission_metrics["shadowkv_planning_latency_total_ms"] += planning_ms
                admission_metrics["admission_plans_total"] += 1
                admission_reason_counts[str(plan.reason)] = admission_reason_counts.get(str(plan.reason), 0) + 1
                admission_strategy_counts[str(plan.strategy)] = admission_strategy_counts.get(str(plan.strategy), 0) + 1
                if plan.strategy == "bypass":
                    admission_metrics["admission_bypass_total"] += 1
                    if admission_mode == "native_sglang_hook":
                        native_bypass_store_allowed = admission.should_store_after_bypass(tokens, plan, metadata=metadata)
                        request_extra_body = {
                            "extra_key": make_sglang_native_admission_extra_key(
                                skip_lookup=True,
                                skip_write=not native_bypass_store_allowed,
                                tag=f"req_{req.request_id}_{prompt_hash[:8]}",
                            )
                        }
                        admission_metrics["admission_native_hook_bypass_total"] += 1
                        admission_metrics["admission_native_skip_lookup_total"] += 1
                        if native_bypass_store_allowed:
                            admission_metrics["admission_native_bypass_store_allowed_total"] += 1
                        else:
                            admission_metrics["admission_native_skip_write_total"] += 1
                    else:
                        admission_metrics["admission_write_through_bypass_total"] += 1
                else:
                    admission_metrics["admission_allow_total"] += 1

            metrics_before = request_metrics_delta(runtime, client)
            result = client.invoke(
                prompt=req.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.request_timeout_s,
                extra_body=request_extra_body,
            )
            metrics_after = request_metrics_delta(runtime, client)
            result.request_id = int(req.request_id)
            result.end_to_end_latency_ms = (time.perf_counter() - request_wall_start) * 1000.0
            metric_diff = diff_request_metrics(runtime, metrics_before, metrics_after)
            query_tokens, hit_tokens, runtime_detail = cache_tokens_for_request(runtime, result, metric_diff)
            if admission_mode == "native_sglang_hook" and plan is not None and plan.strategy == "bypass":
                # A native SGLang skip_lookup request does not transfer or miss
                # cache entries, so it should not count prompt tokens as cache
                # query waste.
                query_tokens = 0
                hit_tokens = 0
                runtime_detail["native_skip_lookup_accounting_adjusted"] = True
            result.cached_tokens = int(hit_tokens)
            feedback_ms, stored_after_bypass = update_admission_after_request(
                admission=admission,
                admission_mode=admission_mode,
                plan=plan,
                tokens=tokens,
                metadata=metadata,
                result=result,
                native_bypass_store_allowed=native_bypass_store_allowed,
            )
            admission_metrics["shadowkv_feedback_latency_total_ms"] += feedback_ms
            if stored_after_bypass and plan is not None and plan.strategy == "bypass":
                admission_metrics["admission_bypass_store_successes"] += 1

            reference_latency = reference_latencies.get(int(req.request_id))
            speedup_vs_no_cache = None
            speedup_lt_1 = False
            if reference_latency and result.end_to_end_latency_ms:
                speedup_vs_no_cache = float(reference_latency / result.end_to_end_latency_ms)
                speedup_lt_1 = speedup_vs_no_cache < 1.0

            bytes_per_token = int(kv_info["kv_bytes_per_token"])
            query_bytes = int(max(query_tokens, 0) * bytes_per_token)
            hit_bytes = int(max(hit_tokens, 0) * bytes_per_token)
            miss_bytes = int(max(query_bytes - hit_bytes, 0))
            failure_waste_bytes = int(query_bytes if speedup_lt_1 else 0)
            row = {
                "request_index": idx,
                "request_id": int(req.request_id),
                "prompt_sha256": sha256_text(req.prompt),
                "engine": engine,
                "runtime": runtime,
                "workload_id": work_id,
                "dataset": workload_spec["dataset"],
                "prompt_mode": workload_spec["prompt_mode"],
                "role": workload_spec["role"],
                "http_latency_ms": result.latency_ms,
                "end_to_end_latency_ms": result.end_to_end_latency_ms,
                "prompt_tokens": int(result.prompt_tokens or 0),
                "completion_tokens": int(result.completion_tokens or 0),
                "total_tokens": int(result.total_tokens or 0),
                "cache_query_tokens": int(query_tokens),
                "cache_hit_tokens": int(hit_tokens),
                "cache_miss_tokens": int(max(query_tokens - hit_tokens, 0)),
                "cache_query_bytes": query_bytes,
                "cache_query_kib": float(query_bytes / 1024.0),
                "cache_hit_bytes": hit_bytes,
                "cache_hit_kib": float(hit_bytes / 1024.0),
                "cache_miss_waste_bytes": miss_bytes,
                "cache_miss_waste_kib": float(miss_bytes / 1024.0),
                "speedup_reference_engine": "vllm_no_cache_reference" if runtime == "vllm" else "sglang_no_cache_reference",
                "speedup_reference_latency_ms": reference_latency,
                "speedup_vs_no_cache": speedup_vs_no_cache,
                "speedup_lt_1": speedup_lt_1,
                "failure_waste_bytes": failure_waste_bytes,
                "failure_waste_kib": float(failure_waste_bytes / 1024.0),
                "shadowkv_plan_strategy": str(getattr(plan, "strategy", "")) if plan is not None else "",
                "shadowkv_plan_reason": str(getattr(plan, "reason", "")) if plan is not None else "",
                "shadowkv_plan_score": float(getattr(plan, "score", 0.0)) if plan is not None else None,
                "shadowkv_admission_mode": admission_mode,
                "admission_selected_preset": admission_metrics["admission_selected_preset"],
                "admission_policy_util_min_ms": admission_metrics["admission_policy_util_min_ms"],
                "admission_policy_min_bootstrap_admissions": admission_metrics["admission_policy_min_bootstrap_admissions"],
                "shadowkv_native_bypass_store_allowed": native_bypass_store_allowed,
                "metadata": metadata,
            }
            row.update(runtime_detail)
            per_rows.append(row)
            call_results.append(result)
            print(
                f"[{utc_now()}] {work_id} {engine} {idx + 1}/{len(requests)} "
                f"lat={result.end_to_end_latency_ms:.1f}ms hit={hit_tokens} query={query_tokens} "
                f"speedup={speedup_vs_no_cache if speedup_vs_no_cache is not None else 'n/a'}",
                flush=True,
            )

        if runtime == "sglang":
            shadowkv_server_metrics_after = collect_shadowkv_admission_metrics(client)
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        server.stop()

    total_wall = time.perf_counter() - start_wall
    write_jsonl(per_request_path, per_rows)
    extra = dict(admission_metrics)
    extra.update(
        {
            "workload_id": work_id,
            "dataset": workload_spec["dataset"],
            "prompt_mode": workload_spec["prompt_mode"],
            "role": workload_spec["role"],
            "engine": engine,
            "runtime": runtime,
            "is_requested_engine": engine in REQUESTED_ENGINES,
            "kv_bytes_per_token": kv_info["kv_bytes_per_token"],
            "kv_kib_per_token": kv_info["kv_kib_per_token"],
            "total_cache_query_tokens": sum(int(row["cache_query_tokens"]) for row in per_rows),
            "total_cache_hit_tokens": sum(int(row["cache_hit_tokens"]) for row in per_rows),
            "total_cache_miss_tokens": sum(int(row["cache_miss_tokens"]) for row in per_rows),
            "total_cache_query_bytes": sum(int(row["cache_query_bytes"]) for row in per_rows),
            "total_cache_hit_bytes": sum(int(row["cache_hit_bytes"]) for row in per_rows),
            "total_cache_miss_waste_bytes": sum(int(row["cache_miss_waste_bytes"]) for row in per_rows),
            "total_failure_waste_bytes": sum(int(row["failure_waste_bytes"]) for row in per_rows),
            "speedup_lt_1_count": sum(1 for row in per_rows if row["speedup_lt_1"]),
            "speedup_observed_count": sum(1 for row in per_rows if row["speedup_vs_no_cache"] is not None),
            "speedup_mean_vs_no_cache": statistics.mean(
                [float(row["speedup_vs_no_cache"]) for row in per_rows if row["speedup_vs_no_cache"] is not None]
            )
            if any(row["speedup_vs_no_cache"] is not None for row in per_rows)
            else None,
            "cache_miss_waste_ratio": (
                sum(int(row["cache_miss_waste_bytes"]) for row in per_rows)
                / max(sum(int(row["cache_query_bytes"]) for row in per_rows), 1)
            ),
            "failure_waste_ratio": (
                sum(int(row["failure_waste_bytes"]) for row in per_rows)
                / max(sum(int(row["cache_query_bytes"]) for row in per_rows), 1)
            ),
            "per_request_path": str(per_request_path),
            "server_stdout_path": str(log_dir / "stdout.log"),
            "server_stderr_path": str(log_dir / "stderr.log"),
            "server_command": command,
            "server_env_updates": env_updates,
            "error": error,
            "admission_reason_counts": admission_reason_counts,
            "admission_strategy_counts": admission_strategy_counts,
        }
    )
    if shadowkv_server_metrics_before is not None and shadowkv_server_metrics_after is not None:
        extra["shadowkv_server_counters"] = diff_counter_metrics(shadowkv_server_metrics_before, shadowkv_server_metrics_after)
    summary = summarize_external_results(call_results, total_wall_time_s=total_wall, extra_metrics=extra)
    summary["p95_latency_ms"] = percentile([float(row["end_to_end_latency_ms"]) for row in per_rows], 95)
    summary["p95_http_latency_ms"] = percentile([float(row["http_latency_ms"]) for row in per_rows], 95)
    payload = {
        "config": {
            "created_at_utc": utc_now(),
            "model": resolve_model(args.model) or args.model,
            "engine": engine,
            "runtime": runtime,
            "workload": workload_spec,
            "trace_path": str(trace_path),
            "n_requests": args.n_requests,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "kv_info": kv_info,
            "admission_preset": args.admission_preset,
            "admission_util_min_ms": args.admission_util_min_ms,
            "admission_min_bootstrap_admissions": args.admission_min_bootstrap_admissions,
        },
        "summary": summary,
    }
    write_json(cell_dir / "result.json", payload)
    return {
        "workload_id": work_id,
        "dataset": workload_spec["dataset"],
        "prompt_mode": workload_spec["prompt_mode"],
        "role": workload_spec["role"],
        "engine": engine,
        "runtime": runtime,
        "is_requested_engine": engine in REQUESTED_ENGINES,
        **summary,
        "result_path": str(cell_dir / "result.json"),
        "per_request_path": str(per_request_path),
    }


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            out[key] = json.dumps(value, sort_keys=True)
        else:
            out[key] = value
    return out


def write_summary_tables(output_root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_json(output_root / "summary.json", rows)
    csv_rows = [flatten_for_csv(row) for row in rows]
    all_keys: list[str] = []
    preferred = [
        "workload_id",
        "dataset",
        "prompt_mode",
        "role",
        "engine",
        "runtime",
        "is_requested_engine",
        "requests_seen",
        "end_to_end_latency_mean_ms",
        "p95_latency_ms",
        "end_to_end_latency_p95_ms",
        "end_to_end_throughput_rps",
        "speedup_mean_vs_no_cache",
        "speedup_lt_1_count",
        "total_cache_query_bytes",
        "total_cache_hit_bytes",
        "total_cache_miss_waste_bytes",
        "cache_miss_waste_ratio",
        "total_failure_waste_bytes",
        "failure_waste_ratio",
        "admission_allow_total",
        "admission_bypass_total",
        "admission_selected_preset",
        "admission_policy_util_min_ms",
        "admission_policy_min_bootstrap_admissions",
        "admission_native_skip_lookup_total",
        "admission_native_skip_write_total",
        "cached_tokens_total",
        "kv_bytes_per_token",
    ]
    for key in preferred:
        if any(key in row for row in csv_rows):
            all_keys.append(key)
    for row in csv_rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(csv_rows)

    compact_keys = preferred
    with (output_root / "summary_compact.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=compact_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)


def load_existing_summary_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / "cells").glob("*/*/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = dict(payload.get("summary") or {})
            config = payload.get("config") or {}
            workload = config.get("workload") or {}
            row = {
                "workload_id": workload.get("workload_id") or summary.get("workload_id"),
                "dataset": workload.get("dataset") or summary.get("dataset"),
                "prompt_mode": workload.get("prompt_mode") or summary.get("prompt_mode"),
                "role": workload.get("role") or summary.get("role"),
                "engine": config.get("engine") or summary.get("engine"),
                "runtime": config.get("runtime") or summary.get("runtime"),
                "is_requested_engine": (config.get("engine") or summary.get("engine")) in REQUESTED_ENGINES,
                **summary,
                "result_path": str(path),
                "per_request_path": summary.get("per_request_path"),
            }
            rows.append(row)
        except Exception as exc:
            print(f"[{utc_now()}] warning: failed to load existing summary {path}: {exc}", flush=True)
    return rows


def replace_summary_row(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
    key = (new_row.get("workload_id"), new_row.get("engine"))
    kept = [row for row in rows if (row.get("workload_id"), row.get("engine")) != key]
    kept.append(new_row)
    return sorted(kept, key=lambda row: (str(row.get("workload_id")), str(row.get("runtime")), str(row.get("engine"))))


def write_report(output_root: Path, rows: list[dict[str, Any]], kv_info: dict[str, Any]) -> None:
    requested = [row for row in rows if row.get("is_requested_engine")]
    lines = [
        "# Qwen2.5-1.5B Reviewer Gap Runtime Waste Experiment",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Scope",
        "",
        "- Model: `Qwen/Qwen2.5-1.5B-Instruct`.",
        "- Workloads: AG News semantic, AG News templated, SAMSum templated control.",
        "- Requested engines: vLLM APC, vLLM APC + ShadowKV++, SGLang RadixAttention, SGLang RadixAttention + ShadowKV++.",
        "- Paired no-cache reference runs are included only to count requests with speedup below 1.0.",
        "",
        "## Instrumentation Notes",
        "",
        f"- KV byte estimate: `{kv_info['kv_bytes_per_token']}` bytes/token ({kv_info['kv_kib_per_token']:.2f} KiB/token), computed from model config.",
        "- vLLM per-request cache hits come from `/metrics` deltas: prefix-cache queried tokens and prompt cached tokens.",
        "- SGLang per-request cache hits come from OpenAI-compatible `usage.prompt_tokens_details.cached_tokens`.",
        "- `cache_miss_waste_bytes` is the query-token byte proxy not satisfied by a cache hit: `(cache_query_tokens - cache_hit_tokens) * kv_bytes_per_token`.",
        "- `failure_waste_bytes` is the stricter paired slowdown proxy: cache-query bytes on requests where latency was slower than the same request under the runtime's no-cache reference.",
        "- vLLM ShadowKV++ remains a write-through policy overlay; SGLang ShadowKV++ uses the native admission hook to request per-request skip-lookup/skip-write.",
        "",
        "## Compact Results",
        "",
        "| Workload | Engine | Mean ms | P95 ms | Mean speedup | Speedup<1 | Miss waste ratio | Failure waste ratio | Cache hit tokens | Admission bypass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in requested:
        lines.append(
            "| {workload_id} | {engine} | {mean:.2f} | {p95:.2f} | {speedup} | {slow} | {miss:.4f} | {failure:.4f} | {hits} | {bypass} |".format(
                workload_id=row.get("workload_id"),
                engine=row.get("engine"),
                mean=float(row.get("end_to_end_latency_mean_ms") or row.get("mean_latency_ms") or 0.0),
                p95=float(row.get("p95_latency_ms") or row.get("end_to_end_latency_p95_ms") or 0.0),
                speedup="n/a" if row.get("speedup_mean_vs_no_cache") is None else f"{float(row['speedup_mean_vs_no_cache']):.4f}",
                slow=int(row.get("speedup_lt_1_count") or 0),
                miss=float(row.get("cache_miss_waste_ratio") or 0.0),
                failure=float(row.get("failure_waste_ratio") or 0.0),
                hits=int(float(row.get("total_cache_hit_tokens") or row.get("cached_tokens_total") or 0)),
                bypass=int(float(row.get("admission_bypass_total") or 0)),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: full run-level metrics.",
            "- `summary_compact.csv`: reviewer-facing metric subset.",
            "- `per_request/<workload>/<engine>.jsonl`: request-level latency, cache-hit, waste, and speedup records.",
            "- `cells/<workload>/<engine>/result.json`: structured per-cell config and summary.",
            "- `server_logs/<workload>/<engine>/`: runtime stdout/stderr for each launched server.",
            "",
        ]
    )
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_metadata(output_root: Path, args: argparse.Namespace, kv_info: dict[str, Any]) -> None:
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        metadata_dir / "experiment_config.json",
        {
            "created_at_utc": utc_now(),
            "model": resolve_model(args.model) or args.model,
            "requested_engines": REQUESTED_ENGINES,
            "reference_engines": REFERENCE_ENGINES,
            "workloads": WORKLOADS,
            "n_requests": args.n_requests,
            "seed": args.seed,
            "context_length": args.context_length,
            "dtype": args.dtype,
            "kv_info": kv_info,
            "admission_preset": args.admission_preset,
            "admission_util_min_ms": args.admission_util_min_ms,
            "admission_min_bootstrap_admissions": args.admission_min_bootstrap_admissions,
            "notes": [
                "No-cache reference engines are used only for paired speedup<1 counts.",
                "vLLM ShadowKV++ is write-through admission overlay; SGLang ShadowKV++ uses native admission hook.",
                "Balanced rerun applies admission_preset=balanced, policy.utility.util_min_ms=2.0, and min_bootstrap_admissions=0.",
            ],
        },
    )


def build_job_order(args: argparse.Namespace) -> list[tuple[dict[str, str], str]]:
    rng = random.Random(args.seed)
    jobs: list[tuple[dict[str, str], str]] = []
    for workload in WORKLOADS:
        if args.runtime_filter in ("all", "vllm"):
            jobs.append((workload, "vllm_no_cache_reference"))
        if args.runtime_filter in ("all", "sglang"):
            jobs.append((workload, "sglang_no_cache_reference"))
        engines = list(REQUESTED_ENGINES)
        rng.shuffle(engines)
        for engine in engines:
            if args.runtime_filter != "all" and engine_runtime(engine) != args.runtime_filter:
                continue
            jobs.append((workload, engine))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted Qwen2.5-1.5B reviewer-gap runtime waste experiment.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-requests", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--sglang-attention-backend", default="triton")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--server-ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--post-ready-sleep-s", type=float, default=3.0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--admission-preset", "--admission_preset", dest="admission_preset", choices=ADMISSION_PRESETS, default="balanced")
    parser.add_argument("--admission-util-min-ms", "--admission_util_min_ms", "--policy.utility.util_min_ms", dest="admission_util_min_ms", type=float, default=2.0)
    parser.add_argument("--admission-min-bootstrap-admissions", "--admission_min_bootstrap_admissions", "--min_bootstrap_admissions", dest="admission_min_bootstrap_admissions", type=int, default=0)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--runtime-filter", choices=["all", "vllm", "sglang"], default="all")
    parser.add_argument("--only-job", default=None, help="Optional workload_id:engine selector for debugging.")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    kv_info = kv_bytes_per_token(args.model, args.dtype_bytes)
    write_metadata(output_root, args, kv_info)
    trace_paths = make_traces(args, output_root)

    rows: list[dict[str, Any]] = load_existing_summary_rows(output_root)
    if rows:
        write_summary_tables(output_root, rows)
        write_report(output_root, rows, kv_info)
    jobs = build_job_order(args)
    for index, (workload, engine) in enumerate(jobs, 1):
        selector = f"{workload['workload_id']}:{engine}"
        if args.only_job and args.only_job != selector:
            continue
        print(f"[{utc_now()}] START {index}/{len(jobs)} {selector}", flush=True)
        row = run_engine(
            args=args,
            output_root=output_root,
            workload_spec=workload,
            trace_path=trace_paths[workload["workload_id"]],
            engine=engine,
            kv_info=kv_info,
        )
        rows = replace_summary_row(rows, row)
        write_summary_tables(output_root, rows)
        write_report(output_root, rows, kv_info)
        print(f"[{utc_now()}] DONE {index}/{len(jobs)} {selector}", flush=True)

    write_summary_tables(output_root, rows)
    write_report(output_root, rows, kv_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
