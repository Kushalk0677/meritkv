from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
SRC = ROOT / "src"
for candidate in (ROOT, THIS_DIR, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from literature_accurate_baselines.adapter_lib import (
        ExternalAdmissionController,
        ManagedServer,
        OpenAICompatClient,
        add_workload_args,
        build_lmcache_command,
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
        make_output_filename,
        maybe_sleep,
        normalize_api_base,
        parse_command_string,
        reset_shadowkv_admission_metrics,
        reset_runtime_cache,
        resolve_model,
        resolve_prompt_mode,
        save_summary,
        summarize_external_results,
        vllm_compat_env_updates,
    )
    from proactive_kv_cache.energy import NvidiaEnergyMeter, measure_idle_baseline
except ModuleNotFoundError:
    from adapter_lib import (
        ExternalAdmissionController,
        ManagedServer,
        OpenAICompatClient,
        add_workload_args,
        build_lmcache_command,
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
        make_output_filename,
        maybe_sleep,
        normalize_api_base,
        parse_command_string,
        reset_shadowkv_admission_metrics,
        reset_runtime_cache,
        resolve_model,
        resolve_prompt_mode,
        save_summary,
        summarize_external_results,
        vllm_compat_env_updates,
    )
    from proactive_kv_cache.energy import NvidiaEnergyMeter, measure_idle_baseline


BASELINE_CHOICES = (
    "vllm_no_cache",
    "vllm_apc",
    "vllm_apc_shadowkv_plus",
    "sglang_radix_attention",
    "sglang_radix_attention_shadowkv_plus",
    "lmcache",
    "lmcache_shadowkv_plus",
)

ADMISSION_PRESETS = (
    "conservative",
    "balanced",
    "aggressive_prefix",
    "low_latency",
)

DEFAULT_ADMISSION_TUNING_PRESETS = (
    "balanced",
    "aggressive_prefix",
    "low_latency",
)


def _runtime_kind(args: argparse.Namespace) -> str:
    if args.baseline.startswith("sglang"):
        return "sglang"
    if args.baseline.startswith("lmcache") and args.lmcache_engine == "sglang":
        return "sglang"
    return "vllm"


def _uses_admission_controller(args: argparse.Namespace) -> bool:
    return args.baseline.endswith("_shadowkv_plus")


def _build_launch(args: argparse.Namespace):
    if args.baseline == "vllm_no_cache":
        return build_vllm_no_cache_command(args), vllm_compat_env_updates()
    if args.baseline.startswith("vllm_apc"):
        return build_vllm_apc_command(args), vllm_compat_env_updates()
    if args.baseline.startswith("sglang_radix_attention"):
        return build_sglang_radix_attention_command(args), {}
    if args.baseline.startswith("lmcache"):
        args.engine = args.lmcache_engine
        return build_lmcache_command(args)
    raise ValueError(f"Unsupported baseline: {args.baseline}")


def _apply_admission_preset(controller: ExternalAdmissionController, preset: str) -> ExternalAdmissionController:
    preset = str(preset or "balanced")
    if preset not in ADMISSION_PRESETS:
        raise ValueError(f"Unknown admission preset: {preset}")
    controller.admission_preset = preset
    if preset == "conservative":
        controller.full_ms_per_token = 0.30
        controller.reuse_overhead_ms = 2.0
        controller.bank.min_match_length = max(int(controller.bank.min_match_length), 16)
    elif preset == "aggressive_prefix":
        controller.full_ms_per_token = 0.45
        controller.reuse_overhead_ms = 0.55
        controller.bank.min_match_length = min(int(controller.bank.min_match_length), 8)
    elif preset == "low_latency":
        controller.full_ms_per_token = 0.50
        controller.reuse_overhead_ms = 0.25
        controller.bank.min_match_length = min(int(controller.bank.min_match_length), 6)
    return controller


def _new_admission_controller(model: str, runtime: str, preset: str) -> ExternalAdmissionController:
    controller = ExternalAdmissionController(model, runtime=runtime)
    return _apply_admission_preset(controller, preset)


def _runtime_admission_score(metrics: dict, metric: str) -> float:
    if metric == "mean_latency_ms":
        return float(metrics.get("mean_latency_ms", 0.0))
    if metric == "p95_latency_ms":
        return float(metrics.get("p95_latency_ms", 0.0))
    if metric == "cached_adjusted_latency":
        mean_latency = float(metrics.get("mean_latency_ms", 0.0))
        p95_latency = float(metrics.get("p95_latency_ms", mean_latency))
        cached_mean = float(metrics.get("cached_tokens_mean", 0.0))
        return mean_latency + 0.05 * p95_latency - 0.02 * cached_mean
    raise ValueError(f"Unsupported admission tuning metric: {metric}")


def _prepare_admission_metadata(admission: ExternalAdmissionController, req, shared_prefix_token_cache: dict[str, int]) -> dict:
    metadata = dict(req.metadata or {})
    metadata["arrival_time"] = req.arrival_time
    shared_prefix_text = metadata.get("shared_prefix_text")
    if shared_prefix_text:
        hint_len = shared_prefix_token_cache.get(shared_prefix_text)
        if hint_len is None:
            hint_len = len(admission.tokenize(str(shared_prefix_text)))
            shared_prefix_token_cache[shared_prefix_text] = hint_len
        metadata["shared_prefix_hint_tokens"] = hint_len
    return metadata


def _stabilize_gpu_power(
    meter: NvidiaEnergyMeter,
    *,
    timeout_s: float,
    sample_interval_s: float = 1.0,
    window_samples: int = 5,
    tolerance_w: float = 3.0,
) -> dict:
    """Wait until observed GPU power is stable enough for an idle baseline."""

    timeout_s = max(float(timeout_s or 0.0), 0.0)
    if timeout_s <= 0:
        return {
            "enabled": False,
            "stable": None,
            "samples": [],
            "timeout_s": 0.0,
            "tolerance_w": tolerance_w,
            "window_samples": window_samples,
        }

    deadline = time.perf_counter() + timeout_s
    samples: list[dict] = []
    stable = False
    while time.perf_counter() < deadline:
        snap = meter.snapshot()
        sample = {
            "elapsed_s": None,
            "power_w": snap.gpu_power_w,
            "memory_mb": snap.gpu_memory_used_mb,
            "source": snap.source,
            "error": snap.error,
        }
        if samples:
            sample["elapsed_s"] = snap.timestamp_s - samples[0]["timestamp_s"]
        else:
            sample["elapsed_s"] = 0.0
        sample["timestamp_s"] = snap.timestamp_s
        samples.append(sample)
        recent = [s for s in samples[-window_samples:] if s.get("power_w") is not None]
        if len(recent) >= window_samples:
            powers = [float(s["power_w"]) for s in recent]
            if max(powers) - min(powers) <= tolerance_w:
                stable = True
                break
        time.sleep(max(sample_interval_s, 0.1))

    compact_samples = [
        {k: v for k, v in sample.items() if k != "timestamp_s"}
        for sample in samples
    ]
    return {
        "enabled": True,
        "stable": stable,
        "samples": compact_samples,
        "timeout_s": timeout_s,
        "tolerance_w": tolerance_w,
        "window_samples": window_samples,
    }


def _wait_until_ready_or_server_exit(client: OpenAICompatClient, server: ManagedServer | None, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: str | None = None
    while time.time() < deadline:
        if server is not None:
            rc = server.poll()
            if rc is not None:
                logs = server.tail_logs(max_lines=240)
                detail = f"\n{logs}" if logs else ""
                raise RuntimeError(f"Runtime server exited before becoming ready (rc={rc}).{detail}")
        for path in ("/v1/models", "/health"):
            try:
                req = urllib.request.Request(client.api_base + path, method="GET")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if 200 <= resp.status < 300:
                        return
            except Exception as exc:
                last_error = str(exc)
        time.sleep(1.0)
    logs = server.tail_logs(max_lines=240) if server is not None else ""
    detail = f"\n{logs}" if logs else ""
    raise RuntimeError(
        f"Server at {client.api_base} did not become ready within {timeout_s:.0f}s. "
        f"Last error: {last_error}{detail}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run literature-accurate external runtime cache baselines.")
    parser.add_argument("--baseline", choices=BASELINE_CHOICES, required=True)
    add_workload_args(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--request_endpoint", choices=["auto", "chat", "completion"], default="auto")
    parser.add_argument("--launch_server", action="store_true")
    parser.add_argument("--server_command", default=None, help="Optional full launch command string. Overrides the built runtime command.")
    parser.add_argument("--python_executable", default="python")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention_backend", default=None)
    parser.add_argument("--lmcache_engine", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--lmcache_mode", choices=["transfer", "offload"], default="transfer")
    parser.add_argument("--lmcache_chunk_size", type=int, default=256)
    parser.add_argument("--lmcache_config_file", default=None)
    parser.add_argument("--kv_offloading_size", type=float, default=10.0)
    parser.add_argument("--server_extra_arg", action="append", default=[], help="Repeatable extra argument passed through to the runtime.")
    parser.add_argument("--server_ready_timeout_s", type=float, default=240.0)
    parser.add_argument("--trace_path", default=None, help="Optional JSON/JSONL request trace. Overrides generated workload requests.")
    parser.add_argument("--reset_external_cache", action="store_true", help="When resetting vLLM prefix cache, also reset connector-managed external cache.")
    parser.add_argument(
        "--admission_mode",
        choices=["strict_no_write", "write_through_admission", "native_sglang_hook"],
        default="write_through_admission",
        help="How ShadowKV++ admission enforces bypasses for external runtime caches.",
    )
    parser.add_argument("--admission_preset", choices=ADMISSION_PRESETS, default="balanced", help="Fixed external admission preset used for the measured run.")
    parser.add_argument("--enable_admission_tuning", action="store_true", help="Run a short unmeasured preset calibration before the measured runtime run.")
    parser.add_argument("--admission_tuning_requests", type=int, default=16, help="Number of leading requests used for unmeasured runtime admission calibration.")
    parser.add_argument("--admission_tuning_presets", nargs="+", choices=ADMISSION_PRESETS, default=list(DEFAULT_ADMISSION_TUNING_PRESETS), help="Candidate admission presets for --enable_admission_tuning.")
    parser.add_argument("--admission_tuning_metric", choices=["mean_latency_ms", "p95_latency_ms", "cached_adjusted_latency"], default="cached_adjusted_latency", help="Metric minimized during admission preset calibration.")
    parser.add_argument("--warmup_requests", type=int, default=0, help="Number of leading requests to run before measured metrics. Runtime cache is reset after warmup when supported.")
    parser.add_argument("--measure_energy", action="store_true", help="Record NVML energy/power around the measured request loop when supported.")
    parser.add_argument("--gpu_index", type=int, default=0, help="NVML GPU index used for energy/power sampling.")
    parser.add_argument("--idle_baseline_seconds", type=float, default=0.0, help="Optional idle baseline duration for adjusted energy estimates.")
    parser.add_argument("--idle_stabilization_seconds", type=float, default=0.0, help="Wait for GPU power to stabilize before measuring idle baseline.")
    parser.add_argument("--idle_stabilization_tolerance_w", type=float, default=3.0, help="Power range threshold used by --idle_stabilization_seconds.")
    args = parser.parse_args()
    args.resolved_prompt_mode = resolve_prompt_mode(args)
    args.admission_tuning_report = None

    runtime = _runtime_kind(args)
    if args.port is None:
        args.port = 30000 if runtime == "sglang" else 8000
    endpoint = args.request_endpoint
    if endpoint == "auto":
        endpoint = "completion" if runtime == "vllm" else "chat"

    api_base = normalize_api_base(args.api_base, args.host, args.port)
    resolved_model = resolve_model(args.model) or args.model
    client = OpenAICompatClient(api_base=api_base, model=resolved_model, endpoint=endpoint)
    requests = load_trace_requests(args.trace_path) if args.trace_path else build_requests(args)

    server = None
    if args.launch_server:
        command = parse_command_string(args.server_command)
        env_updates = {}
        if command is None:
            command, env_updates = _build_launch(args)
        out_dir = Path(args.output_dir)
        server = ManagedServer(
            command=command,
            cwd=str(Path.cwd()),
            env_updates=env_updates,
            stdout_path=out_dir / f"{args.baseline}.stdout.txt",
            stderr_path=out_dir / f"{args.baseline}.stderr.txt",
        )
        server.start()

    admission = _new_admission_controller(resolved_model, runtime, args.admission_preset) if _uses_admission_controller(args) else None
    admission_metrics = {
        "admission_controller_enabled": bool(admission is not None),
        "admission_plans_total": 0,
        "admission_allow_total": 0,
        "admission_bypass_total": 0,
        "admission_runtime_cache_resets": 0,
        "admission_pre_request_cache_resets": 0,
        "admission_post_request_cache_resets": 0,
        "admission_runtime_cache_reset_failures": 0,
        "admission_policy_net_utility_ms": 0.0,
        "admission_write_through_bypass_total": 0,
        "admission_native_hook_bypass_total": 0,
        "admission_native_bypass_store_allowed_total": 0,
        "admission_native_skip_lookup_total": 0,
        "admission_native_skip_write_total": 0,
        "admission_bypass_store_successes": 0,
        "admission_selected_preset": args.admission_preset,
        "admission_calibrated_full_ms_per_token": 0.0,
        "admission_calibrated_reuse_overhead_ms": 0.0,
        "admission_enforcement_mode": args.admission_mode,
    }
    vllm_metrics_report = None
    shadowkv_server_metrics_report = None
    admission_tuning_vllm_metrics = []
    energy_report = None
    idle_energy_baseline = None
    idle_stabilization_report = None

    try:
        _wait_until_ready_or_server_exit(client, server, args.server_ready_timeout_s)
        reset_external = args.reset_external_cache or args.baseline.startswith("lmcache")
        if admission is not None and args.enable_admission_tuning:
            tuning_requests = requests[: max(int(args.admission_tuning_requests), 0)]
            candidates = []
            for preset in args.admission_tuning_presets:
                if preset not in candidates:
                    candidates.append(preset)
            tuning_rows = []
            selected_preset = args.admission_preset
            best_score = None
            for preset in candidates:
                reset_runtime_cache(client, runtime, reset_external=reset_external)
                tuning_metrics_before = collect_vllm_cache_metrics(client) if runtime == "vllm" else None
                candidate_admission = _new_admission_controller(resolved_model, runtime, preset)
                candidate_results = []
                shared_prefix_token_cache = {}
                for req in tuning_requests:
                    metadata = _prepare_admission_metadata(candidate_admission, req, shared_prefix_token_cache)
                    plan, tokens = candidate_admission.plan(req.prompt, metadata=metadata)
                    result = client.invoke(
                        prompt=req.prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        timeout_s=args.request_timeout_s,
                    )
                    result.request_id = req.request_id
                    candidate_results.append(result)
                    candidate_admission.record_after_request(
                        tokens,
                        plan,
                        metadata=metadata,
                        result=result,
                        allow_bypass_store=args.admission_mode == "write_through_admission",
                    )
                metrics = summarize_external_results(candidate_results)
                if tuning_metrics_before is not None:
                    tuning_metrics_after = collect_vllm_cache_metrics(client)
                    admission_tuning_vllm_metrics.append(
                        {
                            "preset": preset,
                            "vllm_cache_metrics": diff_vllm_cache_metrics(tuning_metrics_before, tuning_metrics_after),
                        }
                    )
                metrics["admission_calibrated_full_ms_per_token"] = float(candidate_admission.full_ms_per_token)
                metrics["admission_calibrated_reuse_overhead_ms"] = float(candidate_admission.reuse_overhead_ms)
                score = _runtime_admission_score(metrics, args.admission_tuning_metric)
                row = {"preset": preset, "score": score, "metrics": metrics}
                tuning_rows.append(row)
                if best_score is None or score < best_score:
                    best_score = score
                    selected_preset = preset
            reset_runtime_cache(client, runtime, reset_external=reset_external)
            args.admission_preset = selected_preset
            admission_metrics["admission_selected_preset"] = selected_preset
            admission = _new_admission_controller(resolved_model, runtime, selected_preset)
            report = {
                "enabled": True,
                "method": "fixed_preset_calibration",
                "selected_preset": selected_preset,
                "selection_metric": args.admission_tuning_metric,
                "tuning_requests": len(tuning_requests),
                "candidate_presets": candidates,
                "candidates": tuning_rows,
                "vllm_cache_metrics_by_preset": admission_tuning_vllm_metrics,
                "excluded_from_measured_run": True,
            }
            report_file = Path(args.output_dir) / "admission_tuning_report.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report["report_file"] = str(report_file)
            args.admission_tuning_report = report

        warmup_count = max(int(args.warmup_requests or 0), 0)
        if warmup_count:
            warmup_requests = requests[:warmup_count]
            for req in warmup_requests:
                client.invoke(
                    prompt=req.prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout_s=args.request_timeout_s,
                )
            reset_runtime_cache(client, runtime, reset_external=reset_external)

        results = []
        shared_prefix_token_cache = {}
        workload_trace_class_counts: dict[str, int] = {}
        for req in requests:
            trace_class = str((req.metadata or {}).get("shadowkv_trace_class") or "unclassified")
            workload_trace_class_counts[trace_class] = workload_trace_class_counts.get(trace_class, 0) + 1
        measured_metrics_before = collect_vllm_cache_metrics(client) if runtime == "vllm" else None
        if runtime == "sglang":
            reset_shadowkv_admission_metrics(client)
            shadowkv_server_metrics_before = collect_shadowkv_admission_metrics(client)
        else:
            shadowkv_server_metrics_before = None
        energy_meter = NvidiaEnergyMeter(args.gpu_index) if args.measure_energy else None
        energy_before = None
        if energy_meter is not None:
            if args.idle_baseline_seconds and args.idle_baseline_seconds > 0:
                idle_stabilization_report = _stabilize_gpu_power(
                    energy_meter,
                    timeout_s=args.idle_stabilization_seconds,
                    tolerance_w=args.idle_stabilization_tolerance_w,
                )
                idle_energy_baseline = measure_idle_baseline(energy_meter, duration_s=args.idle_baseline_seconds)
            energy_before = energy_meter.snapshot()
        measured_loop_start_s = time.perf_counter()
        shadowkv_planning_latency_total_ms = 0.0
        shadowkv_feedback_latency_total_ms = 0.0
        admission_reason_counts: dict[str, int] = {}
        admission_strategy_counts: dict[str, int] = {}
        for idx, req in enumerate(requests):
            maybe_sleep(idx, requests, args.simulate_arrivals, args.max_arrival_sleep_ms)
            request_end_to_end_start_s = time.perf_counter()
            plan = None
            tokens = ()
            bypass_runtime_cache = False
            request_extra_body = None
            native_bypass_store_allowed = False
            if admission is not None:
                metadata = _prepare_admission_metadata(admission, req, shared_prefix_token_cache)
                planning_start_s = time.perf_counter()
                plan, tokens = admission.plan(req.prompt, metadata=metadata)
                shadowkv_planning_latency_total_ms += (time.perf_counter() - planning_start_s) * 1000.0
                admission_metrics["admission_plans_total"] += 1
                admission_metrics["admission_policy_net_utility_ms"] += float(plan.score)
                admission_reason_counts[str(plan.reason)] = admission_reason_counts.get(str(plan.reason), 0) + 1
                admission_strategy_counts[str(plan.strategy)] = admission_strategy_counts.get(str(plan.strategy), 0) + 1
                if plan.strategy == "bypass":
                    admission_metrics["admission_bypass_total"] += 1
                    if args.admission_mode == "native_sglang_hook":
                        if runtime != "sglang":
                            raise RuntimeError("native_sglang_hook admission mode requires an SGLang runtime")
                        native_bypass_store_allowed = admission.should_store_after_bypass(
                            tokens,
                            plan,
                            metadata=metadata,
                        )
                        request_extra_body = {
                            "extra_key": make_sglang_native_admission_extra_key(
                                skip_lookup=True,
                                skip_write=not native_bypass_store_allowed,
                                tag=f"req_{req.request_id}",
                            )
                        }
                        admission_metrics["admission_native_hook_bypass_total"] += 1
                        admission_metrics["admission_native_skip_lookup_total"] += 1
                        if native_bypass_store_allowed:
                            admission_metrics["admission_native_bypass_store_allowed_total"] += 1
                        else:
                            admission_metrics["admission_native_skip_write_total"] += 1
                    elif args.admission_mode == "strict_no_write":
                        bypass_runtime_cache = True
                        if reset_runtime_cache(client, runtime, reset_external=reset_external):
                            admission_metrics["admission_runtime_cache_resets"] += 1
                            admission_metrics["admission_pre_request_cache_resets"] += 1
                        else:
                            admission_metrics["admission_runtime_cache_reset_failures"] += 1
                    else:
                        admission_metrics["admission_write_through_bypass_total"] += 1
                else:
                    admission_metrics["admission_allow_total"] += 1
            else:
                metadata = dict(req.metadata or {})
                metadata["arrival_time"] = req.arrival_time
            result = client.invoke(
                prompt=req.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.request_timeout_s,
                extra_body=request_extra_body,
            )
            result.request_id = req.request_id
            results.append(result)
            if bypass_runtime_cache:
                if reset_runtime_cache(client, runtime, reset_external=reset_external):
                    admission_metrics["admission_runtime_cache_resets"] += 1
                    admission_metrics["admission_post_request_cache_resets"] += 1
                else:
                    admission_metrics["admission_runtime_cache_reset_failures"] += 1
            if admission is not None and plan is not None:
                feedback_start_s = time.perf_counter()
                stored = admission.record_after_request(
                    tokens,
                    plan,
                    metadata=metadata,
                    result=result,
                    allow_bypass_store=(
                        args.admission_mode == "write_through_admission"
                        or (
                            args.admission_mode == "native_sglang_hook"
                            and native_bypass_store_allowed
                        )
                    ),
                )
                shadowkv_feedback_latency_total_ms += (time.perf_counter() - feedback_start_s) * 1000.0
                if stored:
                    admission_metrics["store_successes"] = int(admission_metrics.get("store_successes", 0)) + 1
                    if plan.strategy == "bypass":
                        admission_metrics["admission_bypass_store_successes"] += 1
                admission_metrics["admission_calibrated_full_ms_per_token"] = float(admission.full_ms_per_token)
                admission_metrics["admission_calibrated_reuse_overhead_ms"] = float(admission.reuse_overhead_ms)
            result.end_to_end_latency_ms = (time.perf_counter() - request_end_to_end_start_s) * 1000.0
        measured_loop_wall_time_s = time.perf_counter() - measured_loop_start_s
        if measured_metrics_before is not None:
            measured_metrics_after = collect_vllm_cache_metrics(client)
            vllm_metrics_report = diff_vllm_cache_metrics(measured_metrics_before, measured_metrics_after)
        if shadowkv_server_metrics_before is not None:
            shadowkv_server_metrics_after = collect_shadowkv_admission_metrics(client)
            shadowkv_server_metrics_report = diff_counter_metrics(
                shadowkv_server_metrics_before,
                shadowkv_server_metrics_after,
            )
        if energy_meter is not None:
            energy_after = energy_meter.snapshot()
            energy_report = energy_meter.delta(energy_before, energy_after)
    finally:
        if server is not None:
            server.stop()

    end_to_end_metrics = {
        "shadowkv_planning_latency_total_ms": shadowkv_planning_latency_total_ms,
        "shadowkv_planning_latency_mean_ms": shadowkv_planning_latency_total_ms / max(len(results), 1),
        "shadowkv_feedback_latency_total_ms": shadowkv_feedback_latency_total_ms,
        "shadowkv_feedback_latency_mean_ms": shadowkv_feedback_latency_total_ms / max(len(results), 1),
        "end_to_end_measured_scope": "ShadowKV planning -> server request -> feedback",
        "workload_trace_class_counts": workload_trace_class_counts,
        "admission_reason_counts": admission_reason_counts,
        "admission_strategy_counts": admission_strategy_counts,
    }
    summary_metrics = summarize_external_results(
        results,
        total_wall_time_s=measured_loop_wall_time_s,
        extra_metrics=end_to_end_metrics,
    )
    summary_metrics.update(admission_metrics)
    if vllm_metrics_report is not None:
        summary_metrics["vllm_cache_metrics"] = vllm_metrics_report
        summary_metrics["vllm_prefix_cache_queries_delta"] = vllm_metrics_report["prefix_cache_queries_delta"]
        summary_metrics["vllm_prefix_cache_hits_delta"] = vllm_metrics_report["prefix_cache_hits_delta"]
        summary_metrics["vllm_prompt_tokens_cached_delta"] = vllm_metrics_report["prompt_tokens_cached_delta"]
        summary_metrics["vllm_local_cache_hit_tokens_delta"] = vllm_metrics_report["local_cache_hit_tokens_delta"]
        summary_metrics["vllm_external_prefix_cache_hits_delta"] = vllm_metrics_report["external_prefix_cache_hits_delta"]
    if shadowkv_server_metrics_report is not None:
        summary_metrics["shadowkv_server_counters"] = shadowkv_server_metrics_report
        for name, value in shadowkv_server_metrics_report.get("delta", {}).items():
            safe_name = str(name).replace("-", "_").replace(".", "_")
            summary_metrics[f"shadowkv_server_{safe_name}"] = value
    if energy_report is not None:
        summary_metrics.update(energy_report)
        gpu_energy_j = energy_report.get("gpu_energy_j")
        elapsed_s = float(energy_report.get("energy_elapsed_s") or 0.0)
        idle_power_w = None
        if idle_energy_baseline:
            idle_power_w = idle_energy_baseline.get("avg_power_w_from_energy")
            if idle_power_w is None:
                idle_power_w = idle_energy_baseline.get("avg_power_w")
        idle_adjusted = None
        if gpu_energy_j is not None and idle_power_w is not None and elapsed_s > 0:
            idle_adjusted = float(gpu_energy_j) - float(idle_power_w) * elapsed_s
        summary_metrics["idle_adjusted_gpu_energy_j"] = idle_adjusted
        summary_metrics["gpu_joules_per_request"] = float(gpu_energy_j) / max(len(results), 1) if gpu_energy_j is not None else None
        summary_metrics["idle_adjusted_joules_per_request"] = float(idle_adjusted) / max(len(results), 1) if idle_adjusted is not None else None
        prompt_tokens_total = int(summary_metrics.get("prompt_tokens_total") or 0)
        summary_metrics["gpu_joules_per_prompt_token"] = float(gpu_energy_j) / prompt_tokens_total if gpu_energy_j is not None and prompt_tokens_total > 0 else None
    summary = {
        args.baseline: summary_metrics,
        "config": vars(args),
        "runtime": {
            "runtime_kind": runtime,
            "api_base": api_base,
            "resolved_model": resolved_model,
            "launch_server": bool(args.launch_server),
            "request_endpoint": endpoint,
            "literature_accurate_external_runtime": True,
        },
    }
    if idle_energy_baseline is not None:
        summary["idle_energy_baseline"] = idle_energy_baseline
    if idle_stabilization_report is not None:
        summary["idle_stabilization"] = idle_stabilization_report
    out_file = save_summary(args.output_dir, make_output_filename(f"benchmark_{args.baseline}", args), summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
