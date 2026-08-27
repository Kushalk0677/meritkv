#!/usr/bin/env python3
"""Build, hash, relocate, and verify Kushal's two requested artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DELIVERABLES = PROJECT / "deliverables"
KUSHAL_DIR = Path("/Users/evanleri/Desktop/Kushal Deliverables")
OLD_ZIP = DELIVERABLES / "MeritKV-Blackwell-native-enforcement-evidence-2026-08-20.zip"
OLD_EXTRACTED = DELIVERABLES / "MeritKV-Blackwell-native-enforcement-evidence-2026-08-20"
JUNE21 = KUSHAL_DIR / "shadowkv_qwen14b_native_admission_results_2026-06-21.zip"
JUNE22 = KUSHAL_DIR / "shadowkv_qwen14b_native_admission_gate_2026-06-22.zip"
ARMS = (
    "native_admit_all_lru",
    "meritkv_write_through_lru",
    "meritkv_skip_write_only_lru",
    "meritkv_joint_skip_write_skip_lookup_lru",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded(path: Path) -> bool:
    return (
        "__pycache__" in path.parts
        or path.suffix == ".pyc"
        or path.name.endswith((".partial", ".tmp"))
    )


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if excluded(path):
            continue
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def extract_member(archive: Path, member: str, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        data = handle.read(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def add_lineage(root: Path) -> None:
    extract_member(
        JUNE21,
        "shadowkv_qwen14b_native_admission_results_2026-06-21/session_files/"
        "Dockerfile.sglang_2026-06-21-native-admission",
        root / "implementation_lineage/2026-06-21/Dockerfile.sglang_2026-06-21-native-admission",
    )
    extract_member(
        JUNE22,
        "shadowkv_qwen14b_native_admission_gate_2026-06-22_bundle/session_files/"
        "Dockerfile.sglang_2026-06-22-native-admission-counters",
        root / "implementation_lineage/2026-06-22/Dockerfile.sglang_2026-06-22-native-admission-counters",
    )
    extract_member(
        JUNE22,
        "shadowkv_qwen14b_native_admission_gate_2026-06-22_bundle/session_files/"
        "test_sglang_native_admission_hooks_2026_06_22.py",
        root / "implementation_lineage/2026-06-22/test_sglang_native_admission_hooks_2026_06_22.py",
    )
    extract_member(
        JUNE22,
        "shadowkv_qwen14b_native_admission_gate_2026-06-22_bundle/"
        "results_sglang_native_admission_qwen14b_gate_2026-06-22/hook_tests/"
        "hook_test_report.json",
        root / "implementation_lineage/2026-06-22/hook_test_report.json",
    )
    copy_tree(
        PROJECT / "meritkv_blackwell_20260820/runtime_patch",
        root / "implementation_lineage/2026-08-20-gemma-swa",
    )
    lineage = {
        "schema_version": 1,
        "archives": {
            JUNE21.name: sha256(JUNE21),
            JUNE22.name: sha256(JUNE22),
        },
        "files": {
            str(path.relative_to(root)): sha256(path)
            for path in sorted((root / "implementation_lineage").rglob("*"))
            if path.is_file()
        },
        "claim": "June 21 supplies Req/scheduler/Radix request plumbing; June 22 adds requested/executed counters; August 20 adds Gemma SWA hooks. The capacity extension uses the pinned image's proven default Gemma SWARadixCache topology. The optional LFU baseline is omitted as infeasible because an ordinary RadixCache launch crashes on the first Gemma4 forward; the bound crash evidence is retained in packet/feasibility/.",
    }
    (root / "implementation_lineage/LINEAGE.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n"
    )


def add_runtime_dependencies(root: Path) -> None:
    for relative in (
        "meritkv_blackwell_20260817/run_meritkv_cell.py",
        "meritkv_blackwell_20260820/run_meritkv_cell.py",
        "meritkv_blackwell_20260820/trace_metrics.py",
        "FILE_MANIFEST_SHA256.txt",
    ):
        source = PROJECT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    copy_tree(PROJECT / "source_snapshot", root / "source_snapshot")


def write_manifest(root: Path) -> None:
    manifest = root / "MANIFEST_SHA256.txt"
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest and not excluded(path):
            records.append(f"{sha256(path)}  {path.relative_to(root)}\n")
    manifest.write_text("".join(records), encoding="utf-8")


def verify_manifest(root: Path) -> None:
    manifest = root / "MANIFEST_SHA256.txt"
    expected = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            raise SystemExit(f"manifest mismatch: {relative}")
        expected.add(relative)
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != manifest and not excluded(path)
    }
    if actual != expected:
        raise SystemExit(
            f"exact inventory mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )


def zip_root(root: Path) -> Path:
    archive = root.with_suffix(".zip")
    if archive.exists():
        raise SystemExit(f"refusing to overwrite {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not excluded(path):
                handle.write(path, f"{root.name}/{path.relative_to(root)}")
    archive.with_suffix(".zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n")
    return archive


def verify_old_package() -> dict:
    if sha256(OLD_ZIP) != "e7943267252fe160afdbbe37c72884eccd029cc66f22419688657d962d5f749b":
        raise SystemExit("accepted August 20 ZIP changed")
    verify_manifest(OLD_EXTRACTED)
    result = subprocess.run(
        [
            "python3", str(OLD_EXTRACTED / "packet/verify_operation.py"),
            "all", str(OLD_EXTRACTED / "results"),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    report = json.loads(result.stdout)
    if report.get("status") != "pass" or report.get("cells") != 414:
        raise SystemExit("accepted August 20 verifier changed")
    return {
        "zip": OLD_ZIP.name,
        "zip_sha256": sha256(OLD_ZIP),
        "cells": report["cells"],
        "requests": report["requests"],
        "status": "verified_unchanged",
    }


def aggregate(full: dict) -> tuple[dict[str, dict], list[dict]]:
    evaluation = next(row for row in full["phases"] if row["phase"] == "evaluation")
    by_arm: dict[str, list[dict]] = defaultdict(list)
    by_seed: dict[int, dict[str, dict]] = defaultdict(dict)
    for cell in evaluation["cells_detail"]:
        by_arm[cell["arm"]].append(cell)
        by_seed[int(cell["seed"])][cell["arm"]] = cell
    summary = {}
    for arm, cells in by_arm.items():
        summary[arm] = {
            "seeds": len(cells),
            "end_to_end_mean_ms": sum(c["end_to_end_latency_mean_ms"] for c in cells) / len(cells),
            "end_to_end_p95_ms_mean_across_seeds": sum(c["end_to_end_latency_p95_ms"] for c in cells) / len(cells),
            "ttft_mean_ms": sum(c["ttft_mean_ms"] for c in cells) / len(cells),
            "ttft_p95_ms_mean_across_seeds": sum(c["ttft_p95_ms"] for c in cells) / len(cells),
            "gpu_energy_j": sum(c["gpu_energy_j"] for c in cells) / len(cells),
            "evicted_tokens": sum(c["evicted_tokens"] for c in cells) / len(cells),
            "post_pressure_hot_survival": sum(
                c["phase_metrics"]["hot_set"]["post_pressure_survival_count"] for c in cells
            ) / len(cells),
        }
    paired = []
    target = "meritkv_skip_write_only_lru"
    for seed, cells in sorted(by_seed.items()):
        base = cells[target]
        row = {"seed": seed}
        for comparator in (
            "native_admit_all_lru",
            "meritkv_write_through_lru",
            "meritkv_joint_skip_write_skip_lookup_lru",
        ):
            other = cells[comparator]
            row[comparator] = {
                "end_to_end_mean_pct": 100 * (
                    base["end_to_end_latency_mean_ms"] / other["end_to_end_latency_mean_ms"] - 1
                ),
                "gpu_energy_pct": 100 * (base["gpu_energy_j"] / other["gpu_energy_j"] - 1),
                "evicted_tokens_delta": base["evicted_tokens"] - other["evicted_tokens"],
                "hot_survival_delta": (
                    base["phase_metrics"]["hot_set"]["post_pressure_survival_count"]
                    - other["phase_metrics"]["hot_set"]["post_pressure_survival_count"]
                ),
            }
        paired.append(row)
    return summary, paired


def build_extension(results: Path, old_reference: dict) -> Path:
    full = json.loads((results / "control/full_verification.json").read_text())
    completion = json.loads((results / "control/completion_receipt.json").read_text())
    if full.get("status") != "pass" or completion.get("status") != "complete_verified_restored":
        raise SystemExit("extension is not verified and restored")
    if full.get("cells") != 25 or full.get("requests") != 2688:
        raise SystemExit("extension totals are not the frozen 25 cells / 2688 requests")
    lfu_verification = (full.get("inputs") or {}).get("lfu_optional_baseline") or {}
    if (
        lfu_verification.get("status") != "pass"
        or lfu_verification.get("disposition") != "omitted_as_infeasible"
    ):
        raise SystemExit("LFU optional-baseline infeasibility evidence is not verified")
    name = f"MeritKV-Blackwell-storage-admission-extension-Gemma4-31B-{date.today().isoformat()}"
    root = DELIVERABLES / name
    if root.exists():
        raise SystemExit(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    copy_tree(HERE, root / "packet")
    copy_tree(results, root / "results")
    add_runtime_dependencies(root)
    add_lineage(root)
    summary, paired = aggregate(full)
    lfu_receipt = json.loads(
        (HERE / "feasibility/lfu_infeasibility_receipt.json").read_text()
    )
    report = {
        "schema_version": 1,
        "status": "verified",
        "old_result": old_reference,
        "experiment": "meritkv_gemma4_capacity_pressure_20260822",
        "cells": full["cells"],
        "requests": full["requests"],
        "arm_aggregate_across_five_paired_seeds": summary,
        "paired_seed_deltas_for_skip_write_only": paired,
        "lfu_optional_baseline": {
            "request_boundary": "LFU or another frequency-aware retention baseline, if feasible",
            "status": "omitted_as_infeasible_on_pinned_image",
            "reason": (lfu_receipt.get("observation") or {}).get("interpretation"),
            "attempted_topology": lfu_receipt.get("attempted_topology"),
            "failure_fingerprint": (lfu_receipt.get("observation") or {}).get(
                "failure_fingerprint"
            ),
            "receipt": "packet/feasibility/lfu_infeasibility_receipt.json",
            "receipt_sha256": lfu_verification.get("receipt_sha256"),
            "full_crash_log": "packet/feasibility/lfu_launch_crash.log",
            "full_crash_log_sha256": lfu_verification.get("crash_log_sha256"),
            "successful_lfu_cells": 0,
        },
        "exact_output_mismatch_positions": 0,
        "claim_boundary": "This separate frozen extension tests storage admission under measured cache pressure. It does not replace or retune the accepted August 20 native-enforcement result.",
    }
    (root / "SCIENTIFIC_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    conformance = {
        "schema_version": 1,
        "status": "pass",
        "requester": "Kushal Khemani",
        "items": [
            {"request": "preserve the completed experiment unchanged", "evidence": "PRIOR_RESULT_REFERENCE.json and verified original ZIP hash", "check": "old package verifier: 414 cells, 102912 requests"},
            {"request": "separate frozen storage-admission extension", "evidence": "packet/predeclared_protocol.json, results/control/evaluation_freeze.json", "check": "packet/verify_extension.py all"},
            {"request": "Gemma 4 31B priority", "evidence": "cell receipts and pinned model revision", "check": "model/config/index/tokenizer hashes"},
            {"request": "native admit-all, write-through, skip-write-only, joint skip", "evidence": "20 evaluation cells across five seeds", "check": "native action counter invariants on four required arms"},
            {"request": "LFU or another frequency-aware baseline, if feasible", "evidence": "packet/feasibility/lfu_infeasibility_receipt.json and full lfu_launch_crash.log", "check": "optional baseline honestly omitted: the pinned default SWARadixCache is effectively LRU; forcing ordinary RadixCache reaches ready then crashes on the first Gemma4 forward"},
            {"request": "recurring hot set plus long one-offs under constrained cache", "evidence": "packet/token_budget_manifest.json", "check": "4004-token hot set, >138k one-off tokens, 16384-token cap"},
            {"request": "separate calibration then freeze", "evidence": "results/control/calibration_receipt.json and evaluation_freeze.json", "check": "disjoint seed and chronological hash binding"},
            {"request": "five paired evaluation seeds without retuning", "evidence": "packet/evaluation_plan_20.tsv", "check": "one cell per required arm/seed; identical trace and policy decision digests"},
            {"request": "latency, P95, energy, evictions, recovery, useful hits", "evidence": "SCIENTIFIC_REPORT.json and per-request traces", "check": "recomputed finite metrics and physical SGLang eviction counters"},
            {"request": "skip-write executed with zero requested skip-lookup", "evidence": "skip-write-only traces and native counters", "check": "lookup=1;write=0 payload; all lookup counters zero; write counters equal bypasses"},
            {"request": "include legacy runner and base request-plumbing patch", "evidence": "meritkv_blackwell_20260817/, source_snapshot/, implementation_lineage/", "check": "manifest hashes and relocated verifier"},
        ],
    }
    (root / "KUSHAL_REQUEST_CONFORMANCE.json").write_text(
        json.dumps(conformance, indent=2, sort_keys=True) + "\n"
    )
    (root / "PRIOR_RESULT_REFERENCE.json").write_text(
        json.dumps(old_reference, indent=2, sort_keys=True) + "\n"
    )
    rows = [
        "# Gemma 4 31B storage-admission extension\n",
        "This is a separate frozen capacity-pressure extension. The accepted August 20 result remains unchanged.\n",
        f"Verified scope: {full['cells']} cells and {full['requests']} requests: one calibration cell, four all-arm smoke cells, and 20 evaluation cells over five paired seeds.\n",
        "The four required arms use the proven default Gemma `SWARadixCache` topology. LFU was optional if feasible; it is omitted because the pinned image's ordinary-Radix path crashed on the first Gemma4 forward. The exact receipt, launch command, and full crash log are preserved under `packet/feasibility/`.\n",
        "The complete claim, paired results, action-counter proof, hot-set survival/recovery metrics, and limitations are in `SCIENTIFIC_REPORT.json`; request-by-request conformance is in `KUSHAL_REQUEST_CONFORMANCE.json`.\n",
        "Verify after extraction with:\n\n",
        "```bash\nPYTHONDONTWRITEBYTECODE=1 python3 packet/verify_extension.py all results --packet packet\n```\n",
    ]
    (root / "README_FOR_KUSHAL.md").write_text("\n".join(rows))
    write_manifest(root)
    verify_manifest(root)
    with tempfile.TemporaryDirectory() as temporary:
        relocated = Path(temporary) / root.name
        shutil.copytree(root, relocated)
        subprocess.run(
            [
                "python3", str(relocated / "packet/verify_extension.py"),
                "all", str(relocated / "results"), "--packet", str(relocated / "packet"),
            ],
            check=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.DEVNULL,
        )
        verify_manifest(relocated)
    return root


def build_supplement(old_reference: dict) -> Path:
    name = f"MeritKV-Blackwell-native-enforcement-self-containment-supplement-{date.today().isoformat()}"
    root = DELIVERABLES / name
    if root.exists():
        raise SystemExit(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    add_runtime_dependencies(root)
    add_lineage(root)
    (root / "ORIGINAL_PACKAGE_REFERENCE.json").write_text(
        json.dumps(old_reference, indent=2, sort_keys=True) + "\n"
    )
    (root / "README.md").write_text(
        "# August 20 self-containment supplement\n\n"
        "Place these root-level directories beside the extracted August 20 `packet/` directory. "
        "They provide the preserved 20260817 runner, its complete source import closure, and the June 21/22 native request-plumbing/counter lineage requested by Kushal. "
        "The original result ZIP and its 970-file manifest remain byte-for-byte unchanged.\n"
    )
    write_manifest(root)
    verify_manifest(root)
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    old_reference = verify_old_package()
    extension = build_extension(args.results.resolve(), old_reference)
    supplement = build_supplement(old_reference)
    extension_zip = zip_root(extension)
    supplement_zip = zip_root(supplement)
    print(
        json.dumps(
            {
                "status": "pass",
                "extension": str(extension_zip),
                "extension_sha256": sha256(extension_zip),
                "supplement": str(supplement_zip),
                "supplement_sha256": sha256(supplement_zip),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
