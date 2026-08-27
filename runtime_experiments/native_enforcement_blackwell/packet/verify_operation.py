#!/usr/bin/env python3
"""Strict completeness and causal-action verifier for the overnight operation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
from itertools import combinations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trace_metrics import percentile


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "frozen_config.json"
CONFIG_HASH = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
FROZEN_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
DATASET_MANIFEST_PATH = HERE / "dataset_workload_manifest.json"
DATASET_MANIFEST_HASH = hashlib.sha256(DATASET_MANIFEST_PATH.read_bytes()).hexdigest()
DATASET_MANIFEST = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
SERVER_CONFIG_HASH = hashlib.sha256(
    json.dumps(
        FROZEN_CONFIG["runtime"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()
MODEL_CONFIGS = {model["id"]: model for model in FROZEN_CONFIG["models"]}
REQUIRED_ARMS = {
    "smoke": {
        "native_baseline",
        "meritkv_write_through",
        "meritkv_enforced",
        "forced_recompute",
    },
    "proof": {"native_baseline", "meritkv_write_through", "meritkv_enforced"},
    "full": {
        "native_baseline",
        "meritkv_write_through",
        "meritkv_enforced",
        "forced_recompute",
    },
}
PLAN_FILES = {
    "smoke": "smoke_plan_8.tsv",
    "proof": "proof_plan_6.tsv",
    "full": "cell_plan_400.tsv",
}
EXPECTED_COUNTS = {"smoke": 8, "proof": 6, "full": 400}
ARM_ORDER = (
    "native_baseline",
    "meritkv_write_through",
    "meritkv_enforced",
    "forced_recompute",
)


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExpectedCell:
    model_slug: str
    model_id: str
    arm: str
    seed: int
    dataset: str
    prompt_mode: str
    requests: int

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.model_id, self.seed, self.dataset, self.prompt_mode)

    def path(self, phase_root: Path) -> Path:
        return (
            phase_root
            / self.model_slug
            / f"seed_{self.seed}"
            / self.dataset
            / self.prompt_mode
            / self.arm
        )


def fail(message: str) -> None:
    raise VerificationError(message)


def verify_dataset_cache_probe(result_root: Path) -> dict[str, Any]:
    path = result_root / "control" / "dataset_cache_preflight.json"
    if not path.is_file():
        fail(f"missing dataset cache preflight receipt: {path}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid dataset cache preflight receipt: {exc}")
    required = {
        "schema_version": 1,
        "status": "pass",
        "offline": True,
        "frozen_config_sha256": CONFIG_HASH,
        "dataset_count": 5,
        "workload_count": 50,
        "rows_probed": 12800,
        "expected_manifest_sha256": DATASET_MANIFEST_HASH,
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            fail(f"dataset cache receipt {key}={receipt.get(key)!r}, expected {value!r}")
    if receipt.get("datasets") != DATASET_MANIFEST.get("datasets"):
        fail("dataset cache identity/fingerprint manifest mismatch")

    expected_keys = {
        (dataset, int(seed), prompt_mode)
        for dataset in FROZEN_CONFIG["workload"]["datasets"]
        for seed in FROZEN_CONFIG["workload"]["seeds"]
        for prompt_mode in FROZEN_CONFIG["workload"]["prompt_modes"]
    }
    workloads = receipt.get("workloads")
    if not isinstance(workloads, list):
        fail("dataset cache receipt workloads are missing")
    actual_keys: set[tuple[str, int, str]] = set()
    for row in workloads:
        if not isinstance(row, dict):
            fail("dataset cache workload receipt row is invalid")
        key = (str(row.get("dataset")), int(row.get("seed")), str(row.get("prompt_mode")))
        if key in actual_keys:
            fail(f"duplicate dataset cache workload receipt: {key}")
        actual_keys.add(key)
        if row.get("rows") != FROZEN_CONFIG["workload"]["full_requests_per_cell"]:
            fail(f"dataset cache workload row count mismatch: {key}")
        digest = row.get("rows_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            fail(f"dataset cache workload digest invalid: {key}")
        try:
            int(digest, 16)
        except ValueError:
            fail(f"dataset cache workload digest is not hexadecimal: {key}")
    if actual_keys != expected_keys:
        fail("dataset cache workload key set mismatch")

    core = {"datasets": receipt["datasets"], "workloads": workloads}
    core_hash = hashlib.sha256(
        json.dumps(
            core, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    if core_hash != receipt.get("core_sha256"):
        fail("dataset cache receipt core digest does not match its contents")
    if core_hash != DATASET_MANIFEST.get("core_sha256"):
        fail("dataset cache workload rows do not match frozen manifest")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected_manifest_sha256": DATASET_MANIFEST_HASH,
        "core_sha256": core_hash,
        "datasets": 5,
        "workloads": 50,
        "rows_probed": 12800,
    }


def load_plan(phase: str) -> list[ExpectedCell]:
    with (HERE / PLAN_FILES[phase]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    cells = [
        ExpectedCell(
            model_slug=row["model_slug"],
            model_id=row["model_id"],
            arm=row["arm"],
            seed=int(row["seed"]),
            dataset=row["dataset"],
            prompt_mode=row["prompt_mode"],
            requests=int(row["requests"]),
        )
        for row in rows
    ]
    if len(cells) != EXPECTED_COUNTS[phase]:
        fail(f"{phase}: plan count {len(cells)} != {EXPECTED_COUNTS[phase]}")
    if len(set(cells)) != len(cells):
        fail(f"{phase}: duplicate plan cells")
    return cells


def metric_payload(payload: dict[str, Any], cell: Path) -> dict[str, Any]:
    candidates = [
        value
        for key, value in payload.items()
        if key.startswith("sglang_") and isinstance(value, dict)
    ]
    if len(candidates) != 1:
        fail(f"{cell}: expected one SGLang metrics object, found {len(candidates)}")
    return candidates[0]


def load_trace(path: Path, expected: int) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"missing trace {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"{path}:{line_number}: invalid JSON: {exc}")
        if not isinstance(row, dict):
            fail(f"{path}:{line_number}: trace row is not an object")
        rows.append(row)
    if len(rows) != expected:
        fail(f"{path}: trace rows {len(rows)} != {expected}")
    return rows


def cell_identity(cell: ExpectedCell) -> tuple[str, str, str, str, str, str, str]:
    return (
        cell.model_slug,
        cell.model_id,
        cell.arm,
        str(cell.seed),
        cell.dataset,
        cell.prompt_mode,
        str(cell.requests),
    )


def verify_execution_ledger(
    phase_root: Path, expected_cells: list[ExpectedCell]
) -> dict[str, Any]:
    path = phase_root / "execution_ledger.tsv"
    if not path.is_file():
        fail(f"{phase_root.name}: missing append-only execution ledger")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_fields = [
            "timestamp",
            "event",
            "model_slug",
            "model_id",
            "arm",
            "seed",
            "dataset",
            "prompt_mode",
            "requests",
        ]
        if reader.fieldnames != expected_fields:
            fail(f"{path}: invalid ledger columns {reader.fieldnames!r}")
        rows = list(reader)
    if not rows:
        fail(f"{path}: empty execution ledger")
    allowed_events = {"start", "pass"}
    expected_order = [cell_identity(cell) for cell in expected_cells]
    expected_set = set(expected_order)
    seen_by_event: dict[str, list[tuple[str, ...]]] = {event: [] for event in allowed_events}
    for row_number, row in enumerate(rows, 2):
        event = row.get("event") or ""
        if event not in allowed_events:
            fail(f"{path}:{row_number}: invalid event {event!r}")
        if not row.get("timestamp"):
            fail(f"{path}:{row_number}: missing timestamp")
        identity = tuple(row.get(key, "") for key in expected_fields[2:])
        if identity not in expected_set:
            fail(f"{path}:{row_number}: unplanned cell {identity!r}")
        if identity not in seen_by_event[event]:
            seen_by_event[event].append(identity)
    for event in ("start", "pass"):
        if seen_by_event[event] != expected_order:
            fail(
                f"{path}: first-{event} order/completeness does not match "
                f"{PLAN_FILES[phase_root.name]}"
            )
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def expected_server_labels(phase: str) -> list[str]:
    if phase == "smoke":
        with (HERE / "smoke_plan_8.tsv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        return [
            f"smoke_{row['smoke_id']}_{row['model_slug']}_{row['arm']}" for row in rows
        ]
    if phase == "proof":
        return [
            "admission_sensitive_qwen25_32b_native_action_proof",
            "admission_sensitive_gemma4_31b_native_action_proof",
        ]
    with (HERE / "block_plan_40.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        f"block_{row['block_id']}_{row['model_slug']}_{row['arm']}_seed_{row['seed']}"
        for row in rows
    ]


def verify_runtime_artifacts(phase_root: Path, phase: str) -> dict[str, Any]:
    command_path = phase_root / "raw_commands.log"
    if not command_path.is_file() or command_path.stat().st_size == 0:
        fail(f"{phase}: missing raw command ledger")
    raw_commands = command_path.read_text(encoding="utf-8")
    required_launch_tokens = (
        "shadowkv-sglang-native-admission:2026-08-20-swa-counters",
        "sglang.launch_server",
        "--context-length 4096",
        "--mem-fraction-static 0.88",
        "--dtype float16",
        "--attention-backend triton",
        "--sampling-backend pytorch",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-cache-report",
        "--enable-metrics",
    )
    for token in required_launch_tokens:
        if token not in raw_commands:
            fail(f"{phase}: raw command ledger missing launch token {token!r}")
    labels = expected_server_labels(phase)
    server_logs = phase_root / "server_logs"
    artifact_rows = []
    for label in labels:
        if label not in raw_commands:
            fail(f"{phase}: raw command ledger missing server label {label}")
        action_path = server_logs / f"{label}_action_counters.json"
        if not action_path.is_file() or action_path.stat().st_size == 0:
            fail(f"{phase}: missing native action-counter artifact {action_path}")
        try:
            action_payload = json.loads(action_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"{action_path}: invalid action-counter JSON: {exc}")
        if not isinstance(action_payload.get("counters"), dict):
            fail(f"{action_path}: counters object missing")
        matching_logs = sorted(server_logs.glob(f"{label}_*.log"))
        if not matching_logs:
            fail(f"{phase}: missing server log for {label}")
        latest_log = matching_logs[-1]
        if latest_log.stat().st_size == 0:
            fail(f"{latest_log}: empty server log")
        log_lower = latest_log.read_text(encoding="utf-8", errors="replace").lower()
        if any(marker in log_lower for marker in ("out of memory", "oom-kill", "fatal", "traceback")):
            fail(f"{latest_log}: fatal/OOM/traceback marker")
        artifact_rows.append(
            {
                "label": label,
                "server_log": str(latest_log),
                "server_log_sha256": hashlib.sha256(latest_log.read_bytes()).hexdigest(),
                "action_counters": str(action_path),
                "action_counters_sha256": hashlib.sha256(action_path.read_bytes()).hexdigest(),
            }
        )
    return {
        "raw_commands": str(command_path),
        "raw_commands_sha256": hashlib.sha256(command_path.read_bytes()).hexdigest(),
        "servers": artifact_rows,
    }


def counter_int(metrics: dict[str, Any], key: str) -> int:
    try:
        return int(metrics[key])
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"missing or invalid counter {key}: {exc}")
    raise AssertionError("unreachable")


def server_counter_int(metrics: dict[str, Any], key: str) -> int:
    """Read an action delta, accepting omitted zeroes from the server endpoint."""

    if key in metrics:
        return counter_int(metrics, key)
    counters = metrics.get("shadowkv_server_counters") or {}
    delta = counters.get("delta")
    if counters.get("available") is True and isinstance(delta, dict) and key not in delta:
        return 0
    fail(f"missing or invalid native server counter {key}")
    raise AssertionError("unreachable")


def build_output_agreement_report(
    phase: str,
    groups: dict[tuple[str, int, str, str], dict[str, dict[str, Any]]],
    expected_cells: list[ExpectedCell],
) -> dict[str, Any]:
    """Compare every arm exactly while treating disagreement as a result.

    The caller has already recomputed every request text hash and ordered cell
    digest. This adds exhaustive, zero-tolerance cross-arm comparison. Missing
    or malformed evidence still fails; genuine cache/recompute differences are
    retained in the scientific receipt instead of masquerading as an unrun cell.
    """

    required = REQUIRED_ARMS[phase]
    arm_order = [arm for arm in ARM_ORDER if arm in required]
    if set(arm_order) != required:
        fail(f"{phase}: no canonical output-comparison order for required arms")

    expected_by_group: dict[
        tuple[str, int, str, str], dict[str, ExpectedCell]
    ] = {}
    for cell in expected_cells:
        bucket = expected_by_group.setdefault(cell.key, {})
        if cell.arm in bucket:
            fail(f"{phase} {cell.key}: duplicate expected output arm {cell.arm}")
        bucket[cell.arm] = cell
    if set(groups) != set(expected_by_group):
        fail(f"{phase}: output-agreement group coverage does not match the plan")

    group_reports: list[dict[str, Any]] = []
    pairwise_totals: dict[tuple[str, str, str], dict[str, Any]] = {}
    artifact_rows = 0
    logical_positions = 0
    group_pair_records = 0
    request_pair_comparisons = 0
    all_arm_matches = 0
    all_arm_differences = 0

    for key, expected_arms in expected_by_group.items():
        model, seed, dataset, prompt_mode = key
        arms = groups[key]
        if set(expected_arms) != required or set(arms) != required:
            fail(f"{phase} {key}: output comparison arms are incomplete")
        request_counts = {cell.requests for cell in expected_arms.values()}
        if len(request_counts) != 1:
            fail(f"{phase} {key}: planned request counts differ across arms")
        requests = request_counts.pop()
        if requests <= 0:
            fail(f"{phase} {key}: output comparison has no requests")

        for arm in arm_order:
            actual = arms[arm]
            trace = actual.get("trace")
            if actual.get("requests") != requests or not isinstance(trace, list):
                fail(f"{phase} {key} {arm}: output trace coverage is malformed")
            if len(trace) != requests:
                fail(f"{phase} {key} {arm}: output trace length is incomplete")
            if [row.get("request_index") for row in trace] != list(range(requests)):
                fail(f"{phase} {key} {arm}: malformed output request order")
            for row in trace:
                index = row.get("request_index")
                output_text = row.get("output_text")
                output_sha256 = row.get("output_sha256")
                completion_tokens = row.get("completion_tokens")
                if not isinstance(output_text, str) or not isinstance(output_sha256, str):
                    fail(f"{phase} {key} {arm}: missing output evidence at request {index}")
                recomputed = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
                if output_sha256 != recomputed:
                    fail(f"{phase} {key} {arm}: corrupt output evidence at request {index}")
                if (
                    isinstance(completion_tokens, bool)
                    or not isinstance(completion_tokens, int)
                    or not 0 <= completion_tokens <= FROZEN_CONFIG["workload"]["max_tokens"]
                ):
                    fail(
                        f"{phase} {key} {arm}: invalid completion-token evidence "
                        f"at request {index}"
                    )

        pairwise_rows: list[dict[str, Any]] = []
        for left, right in combinations(arm_order, 2):
            differing_indices = [
                index
                for index in range(requests)
                if (
                    arms[left]["trace"][index]["output_text"],
                    arms[left]["trace"][index]["output_sha256"],
                )
                != (
                    arms[right]["trace"][index]["output_text"],
                    arms[right]["trace"][index]["output_sha256"],
                )
            ]
            differing = len(differing_indices)
            matching = requests - differing
            pairwise_rows.append(
                {
                    "arm_a": left,
                    "arm_b": right,
                    "requests_compared": requests,
                    "matching_requests": matching,
                    "differing_requests": differing,
                    "exact_agreement_ratio": f"{matching}/{requests}",
                    "agreement_rate": matching / requests,
                    "exact_difference_ratio": f"{differing}/{requests}",
                    "difference_rate": differing / requests,
                    "differing_request_indices": differing_indices,
                }
            )
            total_key = (model, left, right)
            total = pairwise_totals.setdefault(
                total_key,
                {
                    "model": model,
                    "arm_a": left,
                    "arm_b": right,
                    "groups_compared": 0,
                    "groups_with_differences": 0,
                    "requests_compared": 0,
                    "matching_requests": 0,
                    "differing_requests": 0,
                },
            )
            total["groups_compared"] += 1
            total["groups_with_differences"] += int(differing > 0)
            total["requests_compared"] += requests
            total["matching_requests"] += matching
            total["differing_requests"] += differing
            group_pair_records += 1
            request_pair_comparisons += requests

        differences: list[dict[str, Any]] = []
        for index in range(requests):
            outputs = {
                (
                    arms[arm]["trace"][index]["output_text"],
                    arms[arm]["trace"][index]["output_sha256"],
                )
                for arm in arm_order
            }
            if len(outputs) == 1:
                continue
            observations: dict[str, dict[str, Any]] = {}
            for arm in arm_order:
                row = arms[arm]["trace"][index]
                observations[arm] = {
                    "output_text": row["output_text"],
                    "output_sha256": row["output_sha256"],
                    "completion_tokens": row["completion_tokens"],
                    "cached_tokens": row.get("cached_tokens"),
                    "strategy": row.get("strategy"),
                    "skip_lookup": row.get("skip_lookup"),
                    "skip_write": row.get("skip_write"),
                }
            differences.append(
                {"request_index": index, "arm_observations": observations}
            )

        differing_requests = len(differences)
        matching_requests = requests - differing_requests
        artifact_rows += requests * len(arm_order)
        logical_positions += requests
        all_arm_matches += matching_requests
        all_arm_differences += differing_requests
        group_reports.append(
            {
                "model": model,
                "seed": seed,
                "dataset": dataset,
                "prompt_mode": prompt_mode,
                "requests_compared": requests,
                "arm_output_digests": {
                    arm: arms[arm]["computed_output_digest"] for arm in arm_order
                },
                "all_agree": differing_requests == 0,
                "all_arms_matching_requests": matching_requests,
                "all_arms_differing_requests": differing_requests,
                "exact_all_arms_agreement_ratio": f"{matching_requests}/{requests}",
                "all_arms_agreement_rate": matching_requests / requests,
                "pairwise": pairwise_rows,
                "differences": differences,
            }
        )

    expected_groups = len(expected_by_group)
    expected_artifact_rows = sum(cell.requests for cell in expected_cells)
    expected_logical_positions = sum(
        next(iter(cells.values())).requests for cells in expected_by_group.values()
    )
    pairs_per_group = len(arm_order) * (len(arm_order) - 1) // 2
    expected_group_pair_records = expected_groups * pairs_per_group
    expected_request_pair_comparisons = expected_logical_positions * pairs_per_group
    coverage = {
        "expected_groups": expected_groups,
        "compared_groups": len(group_reports),
        "expected_verified_artifact_request_rows": expected_artifact_rows,
        "verified_artifact_request_rows": artifact_rows,
        "expected_logical_paired_request_positions": expected_logical_positions,
        "logical_paired_request_positions": logical_positions,
        "expected_group_pair_comparisons": expected_group_pair_records,
        "performed_group_pair_comparisons": group_pair_records,
        "expected_request_pair_comparisons": expected_request_pair_comparisons,
        "performed_request_pair_comparisons": request_pair_comparisons,
    }
    if any(
        coverage[expected] != coverage[performed]
        for expected, performed in (
            ("expected_groups", "compared_groups"),
            (
                "expected_verified_artifact_request_rows",
                "verified_artifact_request_rows",
            ),
            (
                "expected_logical_paired_request_positions",
                "logical_paired_request_positions",
            ),
            ("expected_group_pair_comparisons", "performed_group_pair_comparisons"),
            (
                "expected_request_pair_comparisons",
                "performed_request_pair_comparisons",
            ),
        )
    ):
        fail(f"{phase}: output-agreement comparison coverage is incomplete")

    aggregate_pairwise = []
    for total_key in sorted(pairwise_totals):
        total = pairwise_totals[total_key]
        compared = int(total["requests_compared"])
        if compared <= 0:
            fail(f"{phase}: output-agreement pair has no comparisons")
        matching = int(total["matching_requests"])
        total["exact_agreement_ratio"] = f"{matching}/{compared}"
        total["agreement_rate"] = matching / compared
        differing = int(total["differing_requests"])
        total["exact_difference_ratio"] = f"{differing}/{compared}"
        total["difference_rate"] = differing / compared
        aggregate_pairwise.append(total)

    primary_pairs = [
        row
        for row in aggregate_pairwise
        if row["arm_a"] == "meritkv_write_through"
        and row["arm_b"] == "meritkv_enforced"
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "completion_gate": False,
        "comparison_basis": "exact_utf8_output_text_and_sha256",
        "normalization": "none",
        "tolerance_applied": False,
        "arms": arm_order,
        "coverage": coverage,
        "all_agree": all_arm_differences == 0,
        "request_positions": logical_positions,
        "matching_request_positions": all_arm_matches,
        "differing_request_positions": all_arm_differences,
        "groups_with_differences": sum(
            not row["all_agree"] for row in group_reports
        ),
        "pairs_with_differences": sum(
            int(row["differing_requests"]) > 0 for row in aggregate_pairwise
        ),
        "group_pair_comparisons_with_differences": sum(
            int(pair["differing_requests"]) > 0
            for group in group_reports
            for pair in group["pairwise"]
        ),
        "exact_all_arms_agreement_ratio": f"{all_arm_matches}/{logical_positions}",
        "all_arms_agreement_rate": all_arm_matches / logical_positions,
        "exact_all_arms_difference_ratio": f"{all_arm_differences}/{logical_positions}",
        "all_arms_difference_rate": all_arm_differences / logical_positions,
        "primary_policy_pair": primary_pairs,
        "aggregate_pairwise": aggregate_pairwise,
        "groups": group_reports,
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    return report


def verify_cell(phase_root: Path, expected: ExpectedCell) -> dict[str, Any]:
    cell = expected.path(phase_root)
    files = list(cell.glob("benchmark_*.json")) if cell.is_dir() else []
    if len(files) != 1:
        fail(f"{cell}: expected one benchmark JSON, found {len(files)}")
    try:
        payload = json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"{files[0]}: unreadable benchmark JSON: {exc}")
    config = payload.get("config") or {}
    metrics = metric_payload(payload, cell)
    expected_config = {
        "model": expected.model_id,
        "seed": expected.seed,
        "dataset": expected.dataset,
        "prompt_mode": expected.prompt_mode,
        "n_requests": expected.requests,
    }
    for key, value in expected_config.items():
        actual = config.get(key)
        if actual != value:
            fail(f"{cell}: config {key}={actual!r}, expected {value!r}")
    expected_warmups = 16 if phase_root.name == "full" else 2
    fixed_config = {
        "max_tokens": 1,
        "temperature": 0.0,
        "simulate_arrivals": False,
        "warmup_requests": expected_warmups,
        "measure_energy": True,
        "admission_preset": "balanced",
        "enable_admission_tuning": False,
        "request_endpoint": "chat",
    }
    for key, value in fixed_config.items():
        if config.get(key) != value:
            fail(f"{cell}: frozen config {key}={config.get(key)!r}, expected {value!r}")
    stage = payload.get("meritkv_stage") or {}
    if stage.get("arm") != expected.arm:
        fail(f"{cell}: wrong arm receipt {stage.get('arm')!r}")
    if stage.get("frozen_config_sha256") != CONFIG_HASH:
        fail(f"{cell}: frozen config hash mismatch")
    if stage.get("runtime_image_id") != FROZEN_CONFIG["runtime"]["image_id"]:
        fail(f"{cell}: runtime image receipt mismatch")
    if stage.get("server_config_sha256") != SERVER_CONFIG_HASH:
        fail(f"{cell}: server config receipt mismatch")
    if metrics.get("meritkv_arm") != expected.arm:
        fail(f"{cell}: metrics arm mismatch")
    if metrics.get("config_frozen") is not True or metrics.get("retuning_from_results") is not False:
        fail(f"{cell}: frozen/no-retuning flags invalid")
    if counter_int(metrics, "requests_seen") != expected.requests:
        fail(f"{cell}: request count mismatch")
    if not metrics.get("output_agreement_digest"):
        fail(f"{cell}: missing output agreement digest")
    if float(metrics.get("ttft_mean_ms", -1)) < 0:
        fail(f"{cell}: invalid TTFT")
    if float(metrics.get("end_to_end_latency_mean_ms", -1)) < float(
        metrics.get("ttft_mean_ms", 0)
    ):
        fail(f"{cell}: end-to-end latency below TTFT")
    for key in ("ttft_p95_ms", "end_to_end_latency_p95_ms"):
        if float(metrics.get(key, -1)) < 0:
            fail(f"{cell}: missing or invalid {key}")
    if metrics.get("energy_source") != "nvml" or metrics.get("gpu_energy_j") is None:
        fail(f"{cell}: required NVML energy receipt missing")
    counters = metrics.get("shadowkv_server_counters") or {}
    if counters.get("available") is not True:
        fail(f"{cell}: native server counters unavailable")
    trace = load_trace(cell / "meritkv_request_trace.jsonl", expected.requests)
    if any(row.get("arm") != expected.arm for row in trace):
        fail(f"{cell}: request trace arm mismatch")
    if [row.get("request_index") for row in trace] != list(range(expected.requests)):
        fail(f"{cell}: request indices/order are not exact")
    trace_ttfts: list[float] = []
    trace_end_to_end: list[float] = []
    for row in trace:
        if "output_text" not in row:
            fail(f"{cell}: output text missing from request trace")
        observed_hash = hashlib.sha256(str(row["output_text"]).encode("utf-8")).hexdigest()
        if row.get("output_sha256") != observed_hash:
            fail(f"{cell}: output text/hash mismatch at request {row.get('request_index')}")
        try:
            ttft_ms = float(row["ttft_ms"])
            end_to_end_ms = float(row["end_to_end_latency_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"{cell}: invalid request timing: {exc}")
        if not math.isfinite(ttft_ms) or ttft_ms < 0:
            fail(f"{cell}: invalid request TTFT {ttft_ms}")
        if not math.isfinite(end_to_end_ms) or end_to_end_ms < ttft_ms:
            fail(f"{cell}: invalid request end-to-end latency {end_to_end_ms}")
        trace_ttfts.append(ttft_ms)
        trace_end_to_end.append(end_to_end_ms)
    computed_output_digest = hashlib.sha256(
        "\n".join(str(row["output_sha256"]) for row in trace).encode("utf-8")
    ).hexdigest()
    if metrics.get("output_agreement_digest") != computed_output_digest:
        fail(f"{cell}: self-reported output digest does not match ordered trace")
    recomputed_timings = {
        "ttft_mean_ms": sum(trace_ttfts) / len(trace_ttfts),
        "ttft_p95_ms": percentile(trace_ttfts, 0.95),
        "end_to_end_latency_mean_ms": sum(trace_end_to_end) / len(trace_end_to_end),
        "end_to_end_latency_p95_ms": percentile(trace_end_to_end, 0.95),
    }
    for key, recomputed in recomputed_timings.items():
        if not math.isclose(
            float(metrics[key]), recomputed, rel_tol=1e-9, abs_tol=1e-6
        ):
            fail(
                f"{cell}: {key}={metrics[key]!r} does not match request trace "
                f"value {recomputed}"
            )

    if expected.arm.startswith("meritkv_"):
        bypasses = counter_int(metrics, "admission_bypass_total")
        allows = counter_int(metrics, "admission_allow_total")
        if metrics.get("admission_controller_enabled") is not True:
            fail(f"{cell}: MeritKV controller not enabled")
        expected_mode = (
            "native_sglang_hook"
            if expected.arm == "meritkv_enforced"
            else "write_through_admission"
        )
        if metrics.get("admission_enforcement_mode") != expected_mode:
            fail(f"{cell}: wrong admission enforcement mode")
        if bypasses + allows != expected.requests:
            fail(f"{cell}: policy decisions do not sum to requests")
        if counter_int(metrics, "admission_plans_total") != expected.requests:
            fail(f"{cell}: policy plan count mismatch")
        if not metrics.get("policy_decision_digest"):
            fail(f"{cell}: missing policy decision digest")
        tokenizer = metrics.get("tokenizer_provenance") or {}
        expected_tokenizer = MODEL_CONFIGS[expected.model_id]["tokenizer_manifest_sha256"]
        if tokenizer.get("manifest_sha256") != expected_tokenizer:
            fail(f"{cell}: pinned tokenizer manifest mismatch")
        if tokenizer.get("fallback_used") is not False or tokenizer.get("local_files_only") is not True:
            fail(f"{cell}: tokenizer was not offline and fail-closed")
        utility_fields = {
            "utility_score",
            "expected_benefit_ms",
            "expected_cost_ms",
            "expected_waste_ms",
            "confidence",
            "layer_reuse_ratio",
        }
        for row in trace:
            missing = utility_fields.difference(row)
            if missing:
                fail(f"{cell}: utility breakdown missing fields {sorted(missing)}")
        decision_fields = (
            "strategy",
            "reason",
            "utility_score",
            "expected_benefit_ms",
            "expected_cost_ms",
            "expected_waste_ms",
            "confidence",
            "layer_reuse_ratio",
            "reusable_prefix_tokens",
        )
        decision_rows = [{key: row[key] for key in decision_fields} for row in trace]
        computed_policy_digest = hashlib.sha256(
            json.dumps(
                decision_rows, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if metrics.get("policy_decision_digest") != computed_policy_digest:
            fail(f"{cell}: self-reported policy digest does not match ordered trace")
        trace_bypasses = sum(row.get("strategy") == "bypass" for row in trace)
        if trace_bypasses != bypasses or len(trace) - trace_bypasses != allows:
            fail(f"{cell}: trace strategy counts do not match allow/bypass metrics")
        if metrics.get("admission_bypass_store_successes_scope") != (
            "policy_controller_bookkeeping_only"
        ):
            fail(f"{cell}: logical bypass-store metric scope is not explicit")
        logical_bypass_stores = counter_int(
            metrics, "admission_policy_bookkeeping_bypass_store_successes"
        )
        if logical_bypass_stores != counter_int(
            metrics, "admission_bypass_store_successes"
        ):
            fail(f"{cell}: scoped logical bypass-store metric changed its value")
        if any(
            row.get("stored_after_request_scope")
            != "policy_controller_bookkeeping_only"
            or not isinstance(row.get("policy_bookkeeping_stored_after_request"), bool)
            for row in trace
        ):
            fail(f"{cell}: policy bookkeeping trace scope is missing")
        if sum(
            row.get("strategy") == "bypass"
            and row.get("policy_bookkeeping_stored_after_request") is True
            for row in trace
        ) != logical_bypass_stores:
            fail(f"{cell}: logical bypass-store count does not match request trace")
        if expected.arm == "meritkv_enforced":
            exact = {
                "admission_native_skip_lookup_total": bypasses,
                "admission_native_skip_write_total": bypasses,
                "shadowkv_server_skip_lookup_requested_total": bypasses,
                "shadowkv_server_skip_write_requested_total": bypasses,
                "shadowkv_server_radix_skip_lookup_total": bypasses,
            }
            for key, value in exact.items():
                if counter_int(metrics, key) != value:
                    fail(f"{cell}: {key} did not match bypasses={value}")
            finished = server_counter_int(
                metrics, "shadowkv_server_radix_skip_write_finished_total"
            )
            unfinished = server_counter_int(
                metrics, "shadowkv_server_radix_skip_write_unfinished_total"
            )
            if finished + unfinished != bypasses:
                fail(f"{cell}: executed skip-write count did not match bypasses")
            if expected.model_id == "google/gemma-4-31B-it":
                if counter_int(
                    metrics, "shadowkv_server_swa_radix_skip_lookup_total"
                ) != bypasses:
                    fail(f"{cell}: Gemma SWARadixCache lookup bypass was not executed")
                swa_finished = server_counter_int(
                    metrics, "shadowkv_server_swa_radix_skip_write_finished_total"
                )
                swa_unfinished = server_counter_int(
                    metrics, "shadowkv_server_swa_radix_skip_write_unfinished_total"
                )
                if swa_finished + swa_unfinished != bypasses:
                    fail(f"{cell}: Gemma SWARadixCache write bypass was not executed")
            if counter_int(metrics, "admission_native_bypass_store_allowed_total") != 0:
                fail(f"{cell}: enforced bypass allowed a post-bypass store")
            if counter_int(
                metrics, "admission_physical_bypass_store_permitted_total"
            ) != 0:
                fail(f"{cell}: enforced bypass permitted a physical store")
            bypass_rows = [row for row in trace if row.get("strategy") == "bypass"]
            if len(bypass_rows) != bypasses:
                fail(f"{cell}: trace bypass count did not match metrics")
            if any(not row.get("skip_lookup") or not row.get("skip_write") for row in bypass_rows):
                fail(f"{cell}: enforced bypass trace did not request both native actions")
            if any(int(row.get("cached_tokens") or 0) != 0 for row in bypass_rows):
                fail(f"{cell}: an enforced bypass consumed cached tokens")
            if any(
                row.get("physical_store_permitted") is not False
                for row in bypass_rows
            ):
                fail(f"{cell}: bypass trace does not distinguish logical and physical stores")
    if expected.arm == "forced_recompute" and counter_int(metrics, "cached_tokens_total") != 0:
        fail(f"{cell}: forced recompute observed cached tokens")
    if expected.arm in {"native_baseline", "forced_recompute"}:
        if metrics.get("admission_controller_enabled") is not False:
            fail(f"{cell}: controller unexpectedly enabled for control arm")
    if expected.arm != "meritkv_enforced":
        for key in (
            "admission_native_hook_bypass_total",
            "admission_native_bypass_store_allowed_total",
            "admission_native_skip_lookup_total",
            "admission_native_skip_write_total",
        ):
            if counter_int(metrics, key) != 0:
                fail(f"{cell}: non-enforced arm leaked native action counter {key}")
        for key in (
            "shadowkv_server_skip_lookup_requested_total",
            "shadowkv_server_skip_write_requested_total",
            "shadowkv_server_radix_skip_lookup_total",
            "shadowkv_server_radix_skip_write_finished_total",
            "shadowkv_server_radix_skip_write_unfinished_total",
            "shadowkv_server_swa_radix_skip_lookup_total",
            "shadowkv_server_swa_radix_skip_write_finished_total",
            "shadowkv_server_swa_radix_skip_write_unfinished_total",
        ):
            if server_counter_int(metrics, key) != 0:
                fail(f"{cell}: non-enforced arm leaked native action counter {key}")

    return {
        "cell": str(cell),
        "arm": expected.arm,
        "model": expected.model_id,
        "seed": expected.seed,
        "dataset": expected.dataset,
        "prompt_mode": expected.prompt_mode,
        "requests": expected.requests,
        "metrics": metrics,
        "trace": trace,
        "computed_output_digest": computed_output_digest,
        "benchmark_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256((cell / "meritkv_request_trace.jsonl").read_bytes()).hexdigest(),
    }


def verify_phase(result_root: Path, phase: str) -> dict[str, Any]:
    phase_root = result_root / phase
    expected_cells = load_plan(phase)
    ledger = verify_execution_ledger(phase_root, expected_cells)
    runtime_artifacts = verify_runtime_artifacts(phase_root, phase)
    verified = [verify_cell(phase_root, cell) for cell in expected_cells]
    expected_benchmarks = {
        str(next(cell.path(phase_root).glob("benchmark_*.json")).resolve())
        for cell in expected_cells
    }
    observed_benchmarks = {str(path.resolve()) for path in phase_root.rglob("benchmark_*.json")}
    if observed_benchmarks != expected_benchmarks:
        extras = sorted(observed_benchmarks.difference(expected_benchmarks))
        missing = sorted(expected_benchmarks.difference(observed_benchmarks))
        fail(f"{phase}: unexpected/missing benchmark files extras={extras} missing={missing}")
    expected_traces = {
        str((cell.path(phase_root) / "meritkv_request_trace.jsonl").resolve())
        for cell in expected_cells
    }
    observed_traces = {
        str(path.resolve()) for path in phase_root.rglob("meritkv_request_trace*.jsonl")
    }
    if observed_traces != expected_traces:
        extras = sorted(observed_traces.difference(expected_traces))
        missing = sorted(expected_traces.difference(observed_traces))
        fail(f"{phase}: unexpected/missing trace files extras={extras} missing={missing}")
    groups: dict[tuple[str, int, str, str], dict[str, dict[str, Any]]] = {}
    for expected, actual in zip(expected_cells, verified, strict=True):
        groups.setdefault(expected.key, {})[expected.arm] = actual
    required = REQUIRED_ARMS[phase]
    for key, arms in groups.items():
        if set(arms) != required:
            fail(f"{phase} {key}: arms {sorted(arms)} != {sorted(required)}")
        write_metrics = arms["meritkv_write_through"]["metrics"]
        enforced_metrics = arms["meritkv_enforced"]["metrics"]
        if write_metrics["policy_decision_digest"] != enforced_metrics["policy_decision_digest"]:
            fail(f"{phase} {key}: write-through/enforced policy decisions differ")
        if counter_int(
            write_metrics, "admission_bypass_store_successes"
        ) != counter_int(enforced_metrics, "admission_bypass_store_successes"):
            fail(f"{phase} {key}: logical policy bookkeeping differs across MeritKV arms")
        write_store_pattern = [
            row["policy_bookkeeping_stored_after_request"]
            for row in arms["meritkv_write_through"]["trace"]
        ]
        enforced_store_pattern = [
            row["policy_bookkeeping_stored_after_request"]
            for row in arms["meritkv_enforced"]["trace"]
        ]
        if write_store_pattern != enforced_store_pattern:
            fail(f"{phase} {key}: logical policy-store patterns differ across arms")
    output_agreement = build_output_agreement_report(
        phase, groups, expected_cells
    )
    if phase == "proof":
        for enforced in (row for row in verified if row["arm"] == "meritkv_enforced"):
            metrics = enforced["metrics"]
            bypasses = counter_int(metrics, "admission_bypass_total")
            allows = counter_int(metrics, "admission_allow_total")
            if bypasses <= 1 or allows <= 0:
                fail(
                    f"proof {enforced['model']}: trace was not admission-sensitive "
                    f"(allows={allows}, bypasses={bypasses})"
                )
            reuse_rows = [
                row
                for row in enforced["trace"]
                if row.get("strategy") != "bypass" and int(row.get("cached_tokens") or 0) > 0
            ]
            if not reuse_rows:
                fail(f"proof {enforced['model']}: no allowed decision consumed a native hit")
    if phase == "smoke":
        for row in verified:
            if row["arm"] != "forced_recompute" and counter_int(
                row["metrics"], "cached_tokens_total"
            ) <= 0:
                fail(f"smoke {row['model']} {row['arm']}: no native cache hit observed")
    requests = sum(row["requests"] for row in verified)
    expected_requests = {"smoke": 128, "proof": 384, "full": 102400}[phase]
    if requests != expected_requests:
        fail(f"{phase}: requests {requests} != {expected_requests}")

    # Keep the terminal science receipt directly reviewable.  Timings are pooled
    # from independently verified per-request traces; energy and native action
    # counts are additive.  The manifest binds each aggregate to its exact cells.
    aggregates: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in verified}):
        for arm in sorted({row["arm"] for row in verified if row["model"] == model}):
            rows = [
                row for row in verified if row["model"] == model and row["arm"] == arm
            ]
            group_requests = sum(row["requests"] for row in rows)
            if group_requests <= 0:
                fail(f"{phase} {model} {arm}: aggregate has no requests")
            pooled_trace = [request for row in rows for request in row["trace"]]
            if len(pooled_trace) != group_requests:
                fail(f"{phase} {model} {arm}: pooled request trace length mismatch")
            pooled_ttfts = [float(request["ttft_ms"]) for request in pooled_trace]
            pooled_end_to_end = [
                float(request["end_to_end_latency_ms"]) for request in pooled_trace
            ]
            policy_digests = [
                str(row["metrics"].get("policy_decision_digest", "")) for row in rows
            ]
            aggregates.append(
                {
                    "model": model,
                    "arm": arm,
                    "cells": len(rows),
                    "measured_requests": group_requests,
                    "pooled_request_ttft_mean_ms": sum(pooled_ttfts)
                    / group_requests,
                    "pooled_request_ttft_p95_ms": percentile(pooled_ttfts, 0.95),
                    "pooled_request_end_to_end_mean_ms": sum(pooled_end_to_end)
                    / group_requests,
                    "pooled_request_end_to_end_p95_ms": percentile(
                        pooled_end_to_end, 0.95
                    ),
                    "gpu_energy_j_total": sum(
                        float(row["metrics"]["gpu_energy_j"]) for row in rows
                    ),
                    "cached_tokens_total": sum(
                        counter_int(row["metrics"], "cached_tokens_total") for row in rows
                    ),
                    "admission_allow_total": sum(
                        counter_int(row["metrics"], "admission_allow_total") for row in rows
                    ),
                    "admission_bypass_total": sum(
                        counter_int(row["metrics"], "admission_bypass_total") for row in rows
                    ),
                    "benchmark_input_manifest_sha256": hashlib.sha256(
                        "\n".join(sorted(row["benchmark_sha256"] for row in rows)).encode()
                    ).hexdigest(),
                    "policy_decision_manifest_sha256": hashlib.sha256(
                        "\n".join(policy_digests).encode()
                    ).hexdigest(),
                }
            )
    return {
        "schema_version": 1,
        "status": "pass",
        "phase": phase,
        "cells": len(verified),
        "requests": requests,
        "models": sorted({row["model"] for row in verified}),
        "arms": sorted({row["arm"] for row in verified}),
        "frozen_config_sha256": CONFIG_HASH,
        "execution_ledger": ledger,
        "runtime_artifacts": runtime_artifacts,
        "aggregate_metrics": aggregates,
        "output_agreement": output_agreement,
        "benchmark_manifest_sha256": hashlib.sha256(
            "\n".join(sorted(row["benchmark_sha256"] for row in verified)).encode()
        ).hexdigest(),
        "trace_manifest_sha256": hashlib.sha256(
            "\n".join(sorted(row["trace_sha256"] for row in verified)).encode()
        ).hexdigest(),
    }


def verify_all(result_root: Path) -> dict[str, Any]:
    phases = [verify_phase(result_root, phase) for phase in ("smoke", "proof", "full")]
    return {
        "schema_version": 1,
        "status": "pass",
        "experiment": "meritkv_blackwell_enforced_20260820",
        "phases": phases,
        "cells": sum(phase["cells"] for phase in phases),
        "requests": sum(phase["requests"] for phase in phases),
        "frozen_config_sha256": CONFIG_HASH,
        "dataset_cache_probe": verify_dataset_cache_probe(result_root),
    }


def verify_receipt(result_root: Path) -> dict[str, Any]:
    control = result_root / "control"
    receipt_path = control / "completion_receipt.json"
    receipt_sha_path = control / "completion_receipt.sha256"
    scientific_path = control / "scientific_verification.json"
    for path in (receipt_path, receipt_sha_path, scientific_path):
        if not path.is_file():
            fail(f"missing terminal artifact {path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        scientific = json.loads(scientific_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid terminal JSON: {exc}")
    sidecar = receipt_sha_path.read_text(encoding="utf-8").split()
    if not sidecar or sidecar[0] != hashlib.sha256(receipt_path.read_bytes()).hexdigest():
        fail("completion receipt checksum mismatch")
    required = {
        "schema_version": 1,
        "status": "complete_and_verified",
        "experiment": "meritkv_blackwell_enforced_20260820",
        "models": ["Qwen/Qwen2.5-32B-Instruct", "google/gemma-4-31B-it"],
        "cells": 414,
        "measured_requests": 102912,
        "no_retuning_from_results": True,
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            fail(f"completion receipt {key}={receipt.get(key)!r}, expected {value!r}")
    if receipt.get("scientific_verification") != scientific:
        fail("completion receipt embeds different scientific verification")
    if receipt.get("scientific_verification_sha256") != hashlib.sha256(
        scientific_path.read_bytes()
    ).hexdigest():
        fail("scientific verification checksum mismatch")
    current = verify_all(result_root)
    if scientific != current:
        fail("stored scientific verification does not match current artifacts")
    try:
        outage = datetime.fromisoformat(str(receipt["outage_started_at"]))
        restored = datetime.fromisoformat(str(receipt["production_restored_at"]))
    except (KeyError, ValueError) as exc:
        fail(f"invalid outage/restore chronology: {exc}")
    if restored < outage:
        fail("production restoration precedes outage")
    restore = Path(str(receipt.get("restore_evidence") or ""))
    required_restore_files = {
        "health.json",
        "models.json",
        "chat.json",
        "agentvm_relay_models.json",
        "agentvm_relay_chat.json",
        "litellm_readiness.json",
        "hermes_gateway_state.txt",
        "container.json",
        "logs.txt",
        "nvidia-smi.txt",
    }
    if not restore.is_dir():
        fail(f"restore evidence directory missing: {restore}")
    missing_restore = sorted(
        name for name in required_restore_files if not (restore / name).is_file()
    )
    if missing_restore:
        fail(f"restore evidence missing files: {missing_restore}")
    if "MERITKV_RESTORE_OK" not in (restore / "chat.json").read_text(encoding="utf-8"):
        fail("restore generation receipt is invalid")
    if "MERITKV_RELAY_OK" not in (restore / "agentvm_relay_chat.json").read_text(encoding="utf-8"):
        fail("relay generation receipt is invalid")
    return {
        "schema_version": 1,
        "status": "pass",
        "receipt": str(receipt_path),
        "receipt_sha256": sidecar[0],
        "restore_evidence": str(restore),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    phase_parser = subparsers.add_parser("phase")
    phase_parser.add_argument("phase", choices=sorted(PLAN_FILES))
    phase_parser.add_argument("result_root", type=Path)
    cell_parser = subparsers.add_parser("cell")
    cell_parser.add_argument("phase", choices=sorted(PLAN_FILES))
    cell_parser.add_argument("result_root", type=Path)
    cell_parser.add_argument("model_slug")
    cell_parser.add_argument("model_id")
    cell_parser.add_argument("arm")
    cell_parser.add_argument("seed", type=int)
    cell_parser.add_argument("dataset")
    cell_parser.add_argument("prompt_mode")
    cell_parser.add_argument("requests", type=int)
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("result_root", type=Path)
    receipt_parser = subparsers.add_parser("receipt")
    receipt_parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "phase":
            result = verify_phase(args.result_root, args.phase)
        elif args.command == "cell":
            expected = ExpectedCell(
                args.model_slug,
                args.model_id,
                args.arm,
                args.seed,
                args.dataset,
                args.prompt_mode,
                args.requests,
            )
            actual = verify_cell(args.result_root / args.phase, expected)
            result = {"status": "pass", "cell": actual["cell"]}
        elif args.command == "all":
            result = verify_all(args.result_root)
        else:
            result = verify_receipt(args.result_root)
    except (VerificationError, AssertionError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
