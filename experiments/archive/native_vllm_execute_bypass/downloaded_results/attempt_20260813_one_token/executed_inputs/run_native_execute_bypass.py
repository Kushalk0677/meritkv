#!/usr/bin/env python3
"""Native vLLM per-request execute-or-bypass experiment for MeritKV.

The experiment uses vLLM's request-level ``cache_salt`` as the native lookup
actuator.  A stable salt permits automatic prefix-cache reuse.  A unique salt
forces the request to miss previously warmed prefix blocks without flushing or
mutating the rest of the runtime cache.

Four paired arms are measured on one live OpenAI-compatible vLLM server:

* forced_recompute: every measured request receives a unique salt;
* native_apc: every measured request receives the stable warm salt;
* meritkv_write_through: MeritKV scores the request but always uses the stable
  warm salt;
* meritkv_enforced: MeritKV-approved requests use the stable salt and MeritKV-
  bypassed requests receive a unique salt.

The script refuses to report a successful run unless an actuator self-test
observes a native cached-token hit with the stable salt and zero cached tokens
with a unique salt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ARMS = (
    "forced_recompute",
    "native_apc",
    "meritkv_write_through",
    "meritkv_enforced",
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values)) if len(values) > 1 else 0.0


def mean_ci95(values: list[float]) -> dict[str, Any]:
    mean, std = mean_std(values)
    n = len(values)
    # Exact Student-t critical values for the only configured replicate counts
    # likely to be used here; normal fallback is conservative enough for an
    # exploratory non-paper run with more than 30 seeds.
    t95 = {1: 0.0, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    critical = t95.get(n, 2.0 if n > 10 else 0.0)
    half = critical * std / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "ci95_lower": mean - half, "ci95_upper": mean + half}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VLLMClient:
    def __init__(self, api_base: str, model: str, timeout_s: float) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def get_json(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(self.api_base + path, method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_json(self, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.api_base + path,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {path}: {detail}") from exc
        return json.loads(raw), (time.perf_counter() - start) * 1000.0

    def wait_ready(self, timeout_s: float = 900.0) -> None:
        deadline = time.time() + timeout_s
        last_error = ""
        while time.time() < deadline:
            try:
                payload = self.get_json("/v1/models")
                if payload.get("data"):
                    return
            except Exception as exc:  # runtime readiness path
                last_error = str(exc)
            time.sleep(2.0)
        raise RuntimeError(f"vLLM server did not become ready: {last_error}")

    def complete(self, prompt_token_ids: list[int], cache_salt: str, max_tokens: int) -> dict[str, Any]:
        body = {
            "model": self.model,
            "prompt": prompt_token_ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "seed": 0,
            "cache_salt": cache_salt,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
        payload, latency_ms = self.post_json("/v1/completions", body)
        usage = payload.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        choice = (payload.get("choices") or [{}])[0]
        token_ids = choice.get("token_ids")
        if token_ids is None:
            token_ids = []
        logprob_tokens = ((choice.get("logprobs") or {}).get("tokens") or [])
        if not token_ids and logprob_tokens:
            parsed_ids = []
            for token in logprob_tokens:
                text = str(token)
                if text.startswith("token_id:"):
                    try:
                        parsed_ids.append(int(text.split(":", 1)[1]))
                    except ValueError:
                        parsed_ids = []
                        break
            token_ids = parsed_ids
        return {
            "latency_ms": float(latency_ms),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "cached_tokens": int(details.get("cached_tokens") or 0),
            "text": str(choice.get("text") or ""),
            "token_ids": [int(value) for value in token_ids],
            "logprob_tokens": [str(value) for value in logprob_tokens],
            "finish_reason": choice.get("finish_reason"),
        }


def fixed_token_prefix(tokenizer: Any, length: int, tag: str) -> list[int]:
    sentence = (
        f"MeritKV native cache experiment prefix {tag}. "
        "Automatic prefix caching stores key and value tensors for repeated token blocks. "
        "This controlled passage is repeated only to create a deterministic long prefix. "
    )
    text = sentence
    while len(tokenizer.encode(text, add_special_tokens=False)) < length + 16:
        text += sentence
    return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)[:length]]


def build_cases(tokenizer: Any, prefix_lengths: list[int], requests_per_length: int, seed: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for prefix_length in prefix_lengths:
        prefix = fixed_token_prefix(tokenizer, prefix_length, f"L{prefix_length}")
        stable_salt = hashlib.sha256(f"meritkv-prefix-{seed}-{prefix_length}".encode()).hexdigest()
        for index in range(requests_per_length):
            suffix = (
                f"\nRequest {index}: Give one concise implication for cache admission "
                f"when the shared prefix contains about {prefix_length} tokens."
            )
            suffix_ids = [int(value) for value in tokenizer.encode(suffix, add_special_tokens=False)]
            prompt_token_ids = prefix + suffix_ids
            cases.append(
                {
                    "case_id": f"L{prefix_length}_R{index:03d}",
                    "prefix_length_target": prefix_length,
                    "prefix_token_ids": prefix,
                    "prompt_token_ids": prompt_token_ids,
                    "stable_salt": stable_salt,
                }
            )
    random.Random(seed).shuffle(cases)
    return cases


def plan_meritkv(
    *,
    controller: Any,
    tokenizer: Any,
    case: dict[str, Any],
    full_ms_per_token: float,
    reuse_overhead_ms: float,
) -> dict[str, Any]:
    tokens = tuple(int(value) for value in case["prompt_token_ids"])
    reusable = min(int(case["prefix_length_target"]), len(tokens))
    plan = controller.plan(
        tokens=tokens,
        exact_match_len=reusable,
        semantic_similarity=0.0,
        semantic_prefix_len=0,
        shared_prefix_hint=reusable,
        full_ms_per_token=max(float(full_ms_per_token), 1e-6),
        reuse_overhead_ms=max(float(reuse_overhead_ms), 0.0),
        metadata={"prompt_mode": "templated", "shared_prefix_hint_tokens": reusable},
        tier="gpu",
        memory_bytes=0,
        observation_count=100,
    )
    return {
        "strategy": str(plan.strategy),
        "score_ms": float(plan.score),
        "expected_benefit_ms": float(plan.expected_benefit_ms),
        "expected_cost_ms": float(plan.expected_cost_ms),
        "expected_waste_ms": float(plan.expected_waste_ms),
        "reason": str(plan.reason),
        "reusable_prefix_tokens": int(plan.reusable_prefix_tokens),
    }


def calibrate(
    client: VLLMClient,
    tokenizer: Any,
    prefix_lengths: list[int],
    max_tokens: int,
    repetitions: int,
) -> dict[str, Any]:
    per_length: list[dict[str, Any]] = []
    for prefix_length in prefix_lengths:
        prefix = fixed_token_prefix(tokenizer, prefix_length, f"calibration-L{prefix_length}")
        suffix = tokenizer.encode("\nCalibration query: answer with one word.", add_special_tokens=False)
        prompt = prefix + [int(value) for value in suffix]
        stable = hashlib.sha256(f"calibration-{prefix_length}".encode()).hexdigest()
        client.complete(prompt, stable, max_tokens)
        cached_runs = []
        uncached_runs = []
        for repetition in range(repetitions):
            cached_runs.append(client.complete(prompt, stable, max_tokens))
            uncached_runs.append(client.complete(prompt, f"{stable}-unique-{repetition}", max_tokens))
        cached_latency = statistics.median(row["latency_ms"] for row in cached_runs)
        uncached_latency = statistics.median(row["latency_ms"] for row in uncached_runs)
        cached_tokens = int(statistics.median(row["cached_tokens"] for row in cached_runs))
        prompt_tokens = int(statistics.median(row["prompt_tokens"] for row in uncached_runs))
        saved_ms = float(uncached_latency - cached_latency)
        per_length.append(
            {
                "prefix_length_target": prefix_length,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "cached_latency_median_ms": cached_latency,
                "forced_recompute_latency_median_ms": uncached_latency,
                "cached_runs": cached_runs,
                "forced_recompute_runs": uncached_runs,
                "observed_saved_ms": saved_ms,
                "saved_ms_per_cached_token": saved_ms / max(cached_tokens, 1),
            }
        )

    # Fit the measured forced-recompute latency as a + beta*n.  Then estimate
    # the incremental native cache-consumption overhead above the same affine
    # latency evaluated on the uncached suffix.  This avoids calling the full
    # cached-request latency itself a reuse overhead.
    xs = [float(row["prompt_tokens"]) for row in per_length]
    ys = [float(row["forced_recompute_latency_median_ms"]) for row in per_length]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    beta = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator if denominator else 0.0
    beta = max(float(beta), 1e-6)
    intercept = float(y_mean - beta * x_mean)
    overhead_candidates = []
    for row in per_length:
        uncached_tokens = max(int(row["prompt_tokens"]) - int(row["cached_tokens"]), 0)
        expected_without_incremental_reuse_cost = intercept + beta * uncached_tokens
        row["incremental_reuse_cost_ms"] = float(
            row["cached_latency_median_ms"] - expected_without_incremental_reuse_cost
        )
        if row["cached_tokens"] > 0:
            overhead_candidates.append(max(row["incremental_reuse_cost_ms"], 0.0))
    full_ms_per_token = beta
    reuse_overhead_ms = statistics.median(overhead_candidates) if overhead_candidates else 0.0
    return {
        "per_length": per_length,
        "full_ms_per_token": float(full_ms_per_token),
        "reuse_overhead_ms": float(reuse_overhead_ms),
        "forced_recompute_fit_intercept_ms": intercept,
        "repetitions_per_length": repetitions,
        "excluded_from_measured_run": True,
    }


def actuator_self_test(client: VLLMClient, tokenizer: Any, max_tokens: int) -> dict[str, Any]:
    prefix = fixed_token_prefix(tokenizer, 512, "actuator-self-test")
    suffix = tokenizer.encode("\nSelf-test query: answer with one word.", add_special_tokens=False)
    prompt = prefix + [int(value) for value in suffix]
    stable = hashlib.sha256(b"meritkv-native-actuator-self-test").hexdigest()
    warm = client.complete(prompt, stable, max_tokens)
    hit = client.complete(prompt, stable, max_tokens)
    bypass = client.complete(prompt, stable + "-unique", max_tokens)
    passed = hit["cached_tokens"] > 0 and bypass["cached_tokens"] == 0
    report = {"warm": warm, "hit": hit, "bypass": bypass, "passed": passed}
    if not passed:
        raise RuntimeError(
            "Per-request cache-salt actuator self-test failed: expected stable-salt "
            f"cached_tokens>0 and unique-salt cached_tokens=0, observed {report}"
        )
    return report


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted({int(row["seed"]) for row in rows})

    def request_metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
        latencies = [float(row["latency_ms"]) for row in selected]
        cached = [int(row["cached_tokens"]) for row in selected]
        return {
            "requests": len(selected),
            "mean_latency_ms": float(statistics.mean(latencies)) if latencies else 0.0,
            "median_latency_ms": percentile(latencies, 0.50),
            "p95_latency_ms": percentile(latencies, 0.95),
            "cached_tokens_total": int(sum(cached)),
            "cached_tokens_mean": float(statistics.mean(cached)) if cached else 0.0,
            "cache_hit_requests": sum(1 for value in cached if value > 0),
            "meritkv_bypasses": sum(1 for row in selected if row.get("meritkv_strategy") == "bypass"),
            "enforced_bypasses": sum(1 for row in selected if row.get("bypass_enforced")),
        }

    per_seed: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        per_seed[str(seed)] = {
            arm: request_metrics([row for row in rows if int(row["seed"]) == seed and row["arm"] == arm])
            for arm in ARMS
        }

    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        seed_means = [per_seed[str(seed)][arm]["mean_latency_ms"] for seed in seeds]
        seed_p95s = [per_seed[str(seed)][arm]["p95_latency_ms"] for seed in seeds]
        latency_stats = mean_ci95(seed_means)
        p95_stats = mean_ci95(seed_p95s)
        pooled = request_metrics(selected)
        by_arm[arm] = {
            **pooled,
            "mean_latency_ms": latency_stats["mean"],
            "std_latency_ms": latency_stats["std"],
            "latency_ci95_lower_ms": latency_stats["ci95_lower"],
            "latency_ci95_upper_ms": latency_stats["ci95_upper"],
            "p95_latency_ms": p95_stats["mean"],
            "p95_std_ms": p95_stats["std"],
            "replicate_unit": "seed",
            "seed_count": len(seeds),
            "pooled_request_mean_latency_ms": pooled["mean_latency_ms"],
        }
    comparisons: dict[str, Any] = {}
    for left, right, label in (
        ("meritkv_enforced", "meritkv_write_through", "enforced_vs_write_through"),
        ("meritkv_enforced", "native_apc", "enforced_vs_native_apc"),
        ("native_apc", "forced_recompute", "native_apc_vs_forced_recompute"),
    ):
        speedups = []
        latency_changes = []
        for seed in seeds:
            left_mean = per_seed[str(seed)][left]["mean_latency_ms"]
            right_mean = per_seed[str(seed)][right]["mean_latency_ms"]
            speedups.append(right_mean / left_mean if left_mean else 0.0)
            latency_changes.append(100.0 * (left_mean - right_mean) / right_mean if right_mean else 0.0)
        comparisons[label] = {
            "speedup": mean_ci95(speedups),
            "latency_change_pct": mean_ci95(latency_changes),
            "per_seed_speedup": dict(zip((str(seed) for seed in seeds), speedups)),
            "per_seed_latency_change_pct": dict(zip((str(seed) for seed in seeds), latency_changes)),
        }

    by_prefix_length: dict[str, Any] = {}
    for prefix_length in sorted({int(row["prefix_length_target"]) for row in rows}):
        prefix_arms: dict[str, Any] = {}
        for arm in ARMS:
            seed_values = []
            for seed in seeds:
                selected = [
                    row for row in rows
                    if int(row["seed"]) == seed
                    and row["arm"] == arm
                    and int(row["prefix_length_target"]) == prefix_length
                ]
                seed_values.append(request_metrics(selected)["mean_latency_ms"])
            arm_rows = [
                row for row in rows
                if row["arm"] == arm and int(row["prefix_length_target"]) == prefix_length
            ]
            prefix_arms[arm] = {
                "latency_ms": mean_ci95(seed_values),
                "cached_tokens_mean": statistics.mean(float(row["cached_tokens"]) for row in arm_rows),
                "enforced_bypasses": sum(1 for row in arm_rows if row.get("bypass_enforced")),
            }
        by_prefix_length[str(prefix_length)] = prefix_arms

    reference = {
        (int(row["seed"]), row["case_id"]): row
        for row in rows
        if row["arm"] == "forced_recompute"
    }
    output_checks: dict[str, Any] = {}
    for arm in ARMS[1:]:
        selected = [row for row in rows if row["arm"] == arm]
        token_matches = 0
        text_matches = 0
        for row in selected:
            base = reference[(int(row["seed"]), row["case_id"])]
            token_matches += int(row["token_ids"] == base["token_ids"] and bool(row["token_ids"]))
            text_matches += int(row["text"] == base["text"])
        output_checks[arm] = {
            "cases": len(selected),
            "token_id_exact_matches": token_matches,
            "text_exact_matches": text_matches,
            "all_token_ids_exact": token_matches == len(selected) and len(selected) > 0,
            "all_text_exact": text_matches == len(selected) and len(selected) > 0,
        }
    return {
        "arms": by_arm,
        "per_seed": per_seed,
        "by_prefix_length": by_prefix_length,
        "comparisons": comparisons,
        "output_checks": output_checks,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Native vLLM Execute-or-Bypass Results",
        "",
        "> This file is generated from measured request records. It must not be",
        "> treated as a paper result until the run passes all validity checks.",
        "",
        "## Configuration",
        "",
        f"- Model: `{report['configuration']['model']}`",
        f"- Seeds: `{report['configuration']['seeds']}`",
        f"- Prefix lengths: `{report['configuration']['prefix_lengths']}`",
        f"- Requests per prefix length: `{report['configuration']['requests_per_length']}`",
        f"- Output tokens: `{report['configuration']['max_tokens']}`",
        "- Native actuator: stable versus unique per-request vLLM `cache_salt`.",
        "",
        "## Arm summary",
        "",
        "| Arm | Requests | Mean latency (ms) | P95 (ms) | Cached tokens/request | Cache-hit requests | MeritKV bypasses | Enforced bypasses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = summary["arms"][arm]
        lines.append(
            f"| {arm} | {row['requests']} | {row['mean_latency_ms']:.3f} | "
            f"{row['p95_latency_ms']:.3f} | {row['cached_tokens_mean']:.2f} | "
            f"{row['cache_hit_requests']} | {row['meritkv_bypasses']} | {row['enforced_bypasses']} |"
        )
    lines.extend(["", "## Paired interpretation", ""])
    for label, values in summary["comparisons"].items():
        lines.append(
            f"- `{label}`: mean paired-seed speedup `{values['speedup']['mean']:.4f}x` "
            f"(95% CI `{values['speedup']['ci95_lower']:.4f}` to "
            f"`{values['speedup']['ci95_upper']:.4f}`); mean latency change "
            f"`{values['latency_change_pct']['mean']:+.3f}%`."
        )
    lines.extend(["", "## Results by prefix length", ""])
    lines.extend(
        [
            "| Prefix tokens | Forced recompute (ms) | Native APC (ms) | Write-through (ms) | Enforced (ms) | Enforced bypasses |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for prefix_length, arms in summary["by_prefix_length"].items():
        lines.append(
            f"| {prefix_length} | {arms['forced_recompute']['latency_ms']['mean']:.3f} | "
            f"{arms['native_apc']['latency_ms']['mean']:.3f} | "
            f"{arms['meritkv_write_through']['latency_ms']['mean']:.3f} | "
            f"{arms['meritkv_enforced']['latency_ms']['mean']:.3f} | "
            f"{arms['meritkv_enforced']['enforced_bypasses']} |"
        )
    lines.extend(["", "## Output checks", ""])
    for arm, values in summary["output_checks"].items():
        lines.append(
            f"- `{arm}`: token IDs {values['token_id_exact_matches']}/{values['cases']}; "
            f"text {values['text_exact_matches']}/{values['cases']}."
        )
    lines.extend(
        [
            "",
            "## Validity checks",
            "",
            f"- Actuator self-test passed: `{report['validity']['actuator_self_test_passed']}`",
            f"- All requested arms completed: `{report['validity']['all_arms_complete']}`",
            f"- Enforced bypass telemetry consistent: `{report['validity']['enforcement_telemetry_consistent']}`",
            f"- Deterministic output check passed: `{report['validity']['output_check_passed']}`",
            "",
            "## Scope",
            "",
            "This experiment isolates the request-time choice between consuming a",
            "warmed native APC hit and forcing native recomputation. It does not test",
            "storage admission or eviction, and it does not replace the separate",
            "capacity-pressure experiment.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--requests-per-length", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 999])
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--calibration-repetitions", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--output-dir", default="results/native_vllm_execute_bypass")
    args = parser.parse_args()

    if any(length < 16 for length in args.prefix_lengths):
        raise ValueError("All prefix lengths must be at least one vLLM 16-token block")
    if args.requests_per_length < 2:
        raise ValueError("Use at least two requests per prefix length")

    source_root = Path(os.environ.get("MERITKV_SOURCE_ROOT", Path(__file__).resolve().parents[2] / "src"))
    if not (source_root / "proactive_kv_cache").is_dir():
        raise RuntimeError(f"Missing frozen MeritKV source package at {source_root}")
    sys.path.insert(0, str(source_root))
    from proactive_kv_cache.controller import AdaptiveReuseController
    from transformers import AutoTokenizer

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    client = VLLMClient(args.api_base, args.model, args.timeout_s)
    client.wait_ready()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    self_test = actuator_self_test(client, tokenizer, args.max_tokens)
    calibration = calibrate(
        client,
        tokenizer,
        args.prefix_lengths,
        args.max_tokens,
        args.calibration_repetitions,
    )
    (output_root / "actuator_self_test.json").write_text(json.dumps(self_test, indent=2), encoding="utf-8")
    (output_root / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        cases = build_cases(tokenizer, args.prefix_lengths, args.requests_per_length, seed)
        controller = AdaptiveReuseController()

        # Warm every stable prefix before measurement and verify at least one
        # cached block is observable for every tested prefix length.
        warm_rows = []
        for length in args.prefix_lengths:
            representative = next(case for case in cases if case["prefix_length_target"] == length)
            warm_suffix_a = [int(value) for value in tokenizer.encode(
                f"\nWarmup continuation A for prefix length {length}.",
                add_special_tokens=False,
            )]
            warm_suffix_b = [int(value) for value in tokenizer.encode(
                f"\nWarmup continuation B for prefix length {length}.",
                add_special_tokens=False,
            )]
            for warm_arm in ("native_apc", "meritkv_write_through", "meritkv_enforced"):
                arm_salt = f"{representative['stable_salt']}-{warm_arm}"
                first = client.complete(
                    representative["prefix_token_ids"] + warm_suffix_a,
                    arm_salt,
                    args.max_tokens,
                )
                second = client.complete(
                    representative["prefix_token_ids"] + warm_suffix_b,
                    arm_salt,
                    args.max_tokens,
                )
                warm_rows.append(
                    {"prefix_length": length, "arm": warm_arm, "warm": first, "verify": second}
                )
                if second["cached_tokens"] <= 0:
                    raise RuntimeError(f"Stable prefix L{length}/{warm_arm} did not produce a native APC hit")

        # Compute one frozen decision per case.  The write-through and enforced
        # arms consume the identical plan, so their only difference is whether
        # a bypass changes the native request salt.
        plans = {
            case["case_id"]: plan_meritkv(
                controller=controller,
                tokenizer=tokenizer,
                case=case,
                full_ms_per_token=calibration["full_ms_per_token"],
                reuse_overhead_ms=calibration["reuse_overhead_ms"],
            )
            for case in cases
        }
        scheduled = [(arm, case) for case in cases for arm in ARMS]
        random.Random(seed + 1000003).shuffle(scheduled)
        for arm, case in scheduled:
            plan = plans[case["case_id"]] if arm.startswith("meritkv_") else None
            bypass = bool(plan and plan["strategy"] == "bypass")
            if arm == "forced_recompute":
                salt = f"forced-{seed}-{case['case_id']}-{time.time_ns()}"
                enforced = True
            elif arm == "meritkv_enforced" and bypass:
                salt = f"meritkv-bypass-{seed}-{case['case_id']}-{time.time_ns()}"
                enforced = True
            else:
                salt = f"{case['stable_salt']}-{arm}"
                enforced = False
            result = client.complete(case["prompt_token_ids"], salt, args.max_tokens)
            row = {
                "seed": seed,
                "arm": arm,
                "case_id": case["case_id"],
                "prefix_length_target": case["prefix_length_target"],
                "cache_salt_mode": "unique" if enforced else "stable",
                "bypass_enforced": bool(arm == "meritkv_enforced" and bypass),
                "meritkv_strategy": plan["strategy"] if plan else None,
                "meritkv_score_ms": plan["score_ms"] if plan else None,
                "meritkv_expected_benefit_ms": plan["expected_benefit_ms"] if plan else None,
                "meritkv_expected_cost_ms": plan["expected_cost_ms"] if plan else None,
                "meritkv_expected_waste_ms": plan["expected_waste_ms"] if plan else None,
                "meritkv_reason": plan["reason"] if plan else None,
                **result,
            }
            all_rows.append(row)
            if arm == "meritkv_enforced" and bypass and result["cached_tokens"] != 0:
                raise RuntimeError(f"Enforced bypass unexpectedly reused {result['cached_tokens']} tokens")
            if arm in {"native_apc", "meritkv_write_through"} and result["cached_tokens"] <= 0:
                raise RuntimeError(f"{arm} failed to consume a warmed native APC hit for {case['case_id']}")

        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_rows = [row for row in all_rows if row["seed"] == seed]
        (seed_dir / "requests.json").write_text(json.dumps(seed_rows, indent=2), encoding="utf-8")
        (seed_dir / "warmup.json").write_text(json.dumps(warm_rows, indent=2), encoding="utf-8")

    summary = summarize_run(all_rows)
    all_arms_complete = all(summary["arms"][arm]["requests"] > 0 for arm in ARMS)
    telemetry_consistent = all(
        row["cached_tokens"] == 0
        for row in all_rows
        if row["arm"] == "meritkv_enforced" and row["bypass_enforced"]
    )
    output_check_passed = all(values["all_token_ids_exact"] or values["all_text_exact"] for values in summary["output_checks"].values())
    report = {
        "configuration": vars(args),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "actuator_self_test": self_test,
        "calibration": calibration,
        "summary": summary,
        "validity": {
            "actuator_self_test_passed": bool(self_test["passed"]),
            "all_arms_complete": all_arms_complete,
            "enforcement_telemetry_consistent": telemetry_consistent,
            "output_check_passed": output_check_passed,
        },
    }
    (output_root / "requests.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, output_root / "RESULTS.md")

    manifest_lines = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file() and item.name != "MANIFEST_SHA256.txt"):
        manifest_lines.append(f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}")
    (output_root / "MANIFEST_SHA256.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print(f"RESULTS={output_root}")
    if not all(report["validity"].values()):
        print(f"VALIDITY_FAILURE={report['validity']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
