#!/usr/bin/env python3
"""Build and independently verify Kushal's frozen Qwen2.5-32B extension ZIP."""

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
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DELIVERABLES = PROJECT / "deliverables"
OLD_ZIP = DELIVERABLES / "MeritKV-Blackwell-native-enforcement-evidence-2026-08-20.zip"
OLD_EXTRACTED = DELIVERABLES / "MeritKV-Blackwell-native-enforcement-evidence-2026-08-20"
GEMMA_ZIP = DELIVERABLES / (
    "MeritKV-Blackwell-storage-admission-extension-Gemma4-31B-2026-08-23.zip"
)
SUPPLEMENT_ZIP = DELIVERABLES / (
    "MeritKV-Blackwell-native-enforcement-self-containment-supplement-2026-08-23.zip"
)
OUTPUT_NAME = "MeritKV-Blackwell-storage-admission-extension-Qwen2.5-32B-2026-08-24"
EXPECTED_HASHES = {
    OLD_ZIP.name: "e7943267252fe160afdbbe37c72884eccd029cc66f22419688657d962d5f749b",
    GEMMA_ZIP.name: "8ea89aae9c21963a5983898333aebfef488edb45e83908f1755e021336d38797",
    SUPPLEMENT_ZIP.name: "1e8a7e28901921e50ca89d24a1d00f9e14eb71c123594e696e8848f6e0acb2fa",
}
ARMS = (
    "native_admit_all_lru",
    "meritkv_write_through_lru",
    "meritkv_skip_write_only_lru",
    "meritkv_joint_skip_write_skip_lookup_lru",
    "native_admit_all_lfu",
)
LEGACY_PACKET_FILES = {
    "evaluation_plan_20.tsv",
    "smoke_plan_4.tsv",
    "meritkv-capacity-pressure-20260822.service",
    "feasibility/lfu_infeasibility_receipt.json",
    "feasibility/lfu_launch_crash.log",
    "feasibility/server_commands.jsonl",
}


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


def copy_tree(
    source: Path, destination: Path, *, excluded_relatives: set[str] | None = None
) -> None:
    excluded_relatives = excluded_relatives or set()
    for path in sorted(source.rglob("*")):
        if excluded(path):
            continue
        relative = path.relative_to(source)
        if str(relative) in excluded_relatives:
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


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


def verify_manifest(root: Path) -> dict[str, object]:
    manifest = root / "MANIFEST_SHA256.txt"
    expected: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            raise SystemExit(f"manifest mismatch: {relative}")
        if relative in expected:
            raise SystemExit(f"duplicate manifest path: {relative}")
        expected.add(relative)
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != manifest and not excluded(path)
    }
    if actual != expected:
        raise SystemExit(
            f"exact inventory mismatch: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    return {"status": "pass", "manifest_entries": len(expected)}


def run_verifier(packet: Path, results: Path) -> dict:
    completed = subprocess.run(
        [
            "python3",
            str(packet / "verify_extension.py"),
            "delivery",
            str(results),
            "--packet",
            str(packet),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return json.loads(completed.stdout)


def verify_preserved_artifacts() -> dict[str, dict]:
    references: dict[str, dict] = {}
    for path in (OLD_ZIP, GEMMA_ZIP, SUPPLEMENT_ZIP):
        actual = sha256(path)
        if actual != EXPECTED_HASHES[path.name]:
            raise SystemExit(f"preserved artifact changed: {path.name}: {actual}")
        references[path.name] = {
            "zip": path.name,
            "zip_sha256": actual,
            "status": "verified_unchanged",
        }
    verify_manifest(OLD_EXTRACTED)
    old = subprocess.run(
        [
            "python3",
            str(OLD_EXTRACTED / "packet/verify_operation.py"),
            "all",
            str(OLD_EXTRACTED / "results"),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    old_report = json.loads(old.stdout)
    if (
        old_report.get("status") != "pass"
        or old_report.get("cells") != 414
        or old_report.get("requests") != 102912
    ):
        raise SystemExit("accepted August 20 verifier no longer reproduces acceptance")
    references[OLD_ZIP.name].update(
        {"cells": 414, "requests": 102912, "relocated_verifier": "pass"}
    )
    references[SUPPLEMENT_ZIP.name]["role"] = (
        "Previously supplied self-containment supplement containing the preserved "
        "legacy runner and base request-plumbing lineage; referenced, not rebuilt."
    )
    return references


def aggregate(full: dict) -> tuple[dict[str, dict], list[dict]]:
    evaluation = next(row for row in full["phases"] if row["phase"] == "evaluation")
    by_arm: dict[str, list[dict]] = defaultdict(list)
    by_seed: dict[int, dict[str, dict]] = defaultdict(dict)
    for cell in evaluation["cells_detail"]:
        by_arm[cell["arm"]].append(cell)
        by_seed[int(cell["seed"])][cell["arm"]] = cell
    summary: dict[str, dict] = {}
    for arm in ARMS:
        cells = by_arm[arm]
        if len(cells) != 5:
            raise SystemExit(f"aggregate lacks five paired cells for {arm}")
        summary[arm] = {
            "seeds": len(cells),
            "end_to_end_mean_ms": sum(
                cell["end_to_end_latency_mean_ms"] for cell in cells
            ) / len(cells),
            "end_to_end_p95_ms_mean_across_seeds": sum(
                cell["end_to_end_latency_p95_ms"] for cell in cells
            ) / len(cells),
            "ttft_mean_ms": sum(cell["ttft_mean_ms"] for cell in cells) / len(cells),
            "ttft_p95_ms_mean_across_seeds": sum(
                cell["ttft_p95_ms"] for cell in cells
            ) / len(cells),
            "gpu_energy_j": sum(cell["gpu_energy_j"] for cell in cells) / len(cells),
            "evicted_tokens": sum(cell["evicted_tokens"] for cell in cells) / len(cells),
            "post_pressure_hot_survival": sum(
                cell["phase_metrics"]["hot_set"]["post_pressure_survival_count"]
                for cell in cells
            ) / len(cells),
            "complete_hot_recovery_cells": sum(
                cell["phase_metrics"]["hot_set"]["complete_recovery"] is True
                for cell in cells
            ),
        }
    paired: list[dict] = []
    target = "meritkv_skip_write_only_lru"
    for seed, cells in sorted(by_seed.items()):
        if set(cells) != set(ARMS):
            raise SystemExit(f"seed {seed} does not contain all five paired arms")
        base = cells[target]
        comparisons: dict[str, dict] = {}
        for comparator in ARMS:
            if comparator == target:
                continue
            other = cells[comparator]
            comparisons[comparator] = {
                "end_to_end_mean_pct": 100
                * (
                    base["end_to_end_latency_mean_ms"]
                    / other["end_to_end_latency_mean_ms"]
                    - 1
                ),
                "gpu_energy_pct": 100
                * (base["gpu_energy_j"] / other["gpu_energy_j"] - 1),
                "evicted_tokens_delta": base["evicted_tokens"] - other["evicted_tokens"],
                "hot_survival_delta": (
                    base["phase_metrics"]["hot_set"]["post_pressure_survival_count"]
                    - other["phase_metrics"]["hot_set"]["post_pressure_survival_count"]
                ),
            }
        paired.append({"seed": seed, "comparisons": comparisons})
    return summary, paired


def build_extension(results: Path, references: dict[str, dict], staging: Path) -> Path:
    delivery = run_verifier(HERE, results)
    if (
        delivery.get("status") != "pass"
        or delivery.get("cells") != 31
        or delivery.get("requests") != 3344
    ):
        raise SystemExit("Qwen extension is not delivery-verified at 31 cells / 3344 requests")
    full = delivery["experiment"]
    evaluation = next(row for row in full["phases"] if row["phase"] == "evaluation")
    lfu = evaluation.get("lfu_feasibility") or {}
    if lfu.get("status") != "pass":
        raise SystemExit("excluded LFU feasibility gate is not verified")

    root = staging / OUTPUT_NAME
    root.mkdir(parents=True)
    copy_tree(HERE, root / "packet", excluded_relatives=LEGACY_PACKET_FILES)
    copy_tree(results, root / "results")
    add_runtime_dependencies(root)
    old_reference = references[OLD_ZIP.name]
    supplement_reference = references[SUPPLEMENT_ZIP.name]
    gemma_reference = references[GEMMA_ZIP.name]
    (root / "PRIOR_RESULT_REFERENCE.json").write_text(
        json.dumps(old_reference, indent=2, sort_keys=True) + "\n"
    )
    (root / "PRIOR_GEMMA_EXTENSION_REFERENCE.json").write_text(
        json.dumps(gemma_reference, indent=2, sort_keys=True) + "\n"
    )
    (root / "SELF_CONTAINMENT_SUPPLEMENT_REFERENCE.json").write_text(
        json.dumps(supplement_reference, indent=2, sort_keys=True) + "\n"
    )

    summary, paired = aggregate(full)
    report = {
        "schema_version": 1,
        "status": "verified",
        "experiment": "meritkv_qwen25_capacity_pressure_20260824",
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "cells": full["cells"],
        "requests": full["requests"],
        "measured_design": {
            "excluded_lfu_feasibility_preflight": 1,
            "excluded_calibration_cells": 1,
            "measured_smoke_cells": 5,
            "measured_evaluation_cells": 25,
            "paired_evaluation_seeds": [32001, 32002, 32003, 32004, 32005],
            "chronological_latin_balance_verified": True,
            "all_five_smoke_cells_completed_before_evaluation": True,
        },
        "arms": list(ARMS),
        "cache_topology": {
            "implementation": "RadixCache",
            "hybrid_swa": False,
            "lru_arms": 4,
            "lfu_arms": 1,
        },
        "lfu_frequency_aware_baseline": {
            "status": "feasible_included",
            "arm": "native_admit_all_lfu",
            "excluded_preflight_receipt": lfu,
            "measured_cells": 6,
        },
        "arm_aggregate_across_five_paired_seeds": summary,
        "paired_seed_deltas_for_skip_write_only": paired,
        "observed_evaluation_execution_order": evaluation[
            "observed_execution_order"
        ],
        "exact_output_mismatch_positions": evaluation[
            "exact_output_mismatch_positions"
        ],
        "restoration": delivery["restoration"],
        "claim_boundary": (
            "This is a separate frozen Qwen2.5-32B storage-admission extension. "
            "It does not replace or retune the accepted August 20 result or the "
            "completed Gemma extension. Metrics are reported as measured; no positive "
            "performance conclusion is implied by conformance alone."
        ),
    }
    (root / "SCIENTIFIC_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    conformance = {
        "schema_version": 1,
        "status": "pass",
        "requester": "Kushal Khemani",
        "items": [
            {
                "request": "retain completed work unchanged and report a separate extension",
                "evidence": "PRIOR_RESULT_REFERENCE.json and PRIOR_GEMMA_EXTENSION_REFERENCE.json",
                "check": "fixed SHA-256 values verified before and after packaging",
            },
            {
                "request": "Qwen2.5-32B if time permits",
                "evidence": "packet/predeclared_protocol.json and every cell receipt",
                "check": "pinned model revision plus config, weight-index, and tokenizer manifests",
            },
            {
                "request": "admit-all, write-through, skip-write-only, joint skip, and frequency-aware retention",
                "evidence": "five smoke cells and 25 paired evaluation cells",
                "check": "four LRU arms plus a genuine native LFU RadixCache arm",
            },
            {
                "request": "LFU only if feasible",
                "evidence": "results/control/lfu_feasibility_receipt.json",
                "check": "excluded launch, RadixCache topology, functional canary, and real physical eviction all passed before freeze",
            },
            {
                "request": "recurring hot set plus long one-offs under constrained cache",
                "evidence": "packet/token_budget_manifest.json and frozen traces",
                "check": "tokenizer-measured hot-set/capacity and pressure gates",
            },
            {
                "request": "separate calibration, freeze, then five paired seeds without retuning",
                "evidence": "calibration_receipt.json, evaluation_freeze.json, and evaluation_plan_25.tsv",
                "check": "disjoint seeds, one cell per arm/seed, and timestamp order exactly matching the frozen plan",
            },
            {
                "request": "smoke-test all arms before the full run",
                "evidence": "five smoke cell receipts and evaluation timestamps",
                "check": "last smoke completion precedes first evaluation start",
            },
            {
                "request": "latency, P95, energy, evictions, recovery, and useful resident hits",
                "evidence": "SCIENTIFIC_REPORT.json and per-request traces",
                "check": "finite metrics, emitted physical-eviction samples, frozen useful-hit thresholds, and complete per-family recovery",
            },
            {
                "request": "skip-write-only executes skip writes with zero requested skip lookups",
                "evidence": "skip-write-only request traces and native counter deltas",
                "check": "requested/executed write counts equal bypasses; lookup counters are zero",
            },
            {
                "request": "legacy runner and base request-plumbing evidence",
                "evidence": "SELF_CONTAINMENT_SUPPLEMENT_REFERENCE.json",
                "check": "previously supplied supplement is reused by exact SHA-256 and is not rebuilt",
            },
            {
                "request": "restore the captured Qwen production service on every exit",
                "evidence": "results/control/completion_receipt.json, recovery_state.json, production preimage, and restore probes",
                "check": "exact container/image binding, NVFP4 image/model mount, direct chat/vision/relay, and LiteLLM readiness",
            },
        ],
    }
    (root / "KUSHAL_REQUEST_CONFORMANCE.json").write_text(
        json.dumps(conformance, indent=2, sort_keys=True) + "\n"
    )
    (root / "README_FOR_KUSHAL.md").write_text(
        "# Qwen2.5-32B storage-admission extension\n\n"
        "This is a separate frozen capacity-pressure extension. The accepted August 20 "
        "result and the completed Gemma extension remain byte-for-byte unchanged.\n\n"
        "Verified scope: 31 cells and 3,344 requests: one excluded calibration cell, "
        "five measured smoke cells, and 25 evaluation cells over five paired seeds. "
        "A separate excluded LFU feasibility preflight passed before calibration and "
        "freeze. All five measured arms use ordinary `RadixCache` with `hybrid_swa=False`; "
        "the frequency-aware comparator uses genuine LFU retention.\n\n"
        "The verifier confirms non-overlapping evaluation timestamps in the exact frozen "
        "25-row plan order, as well as paired identical traces within each seed.\n\n"
        "The previously delivered self-containment supplement is referenced by exact hash "
        "in `SELF_CONTAINMENT_SUPPLEMENT_REFERENCE.json`; it was not rebuilt.\n\n"
        "Verify after extraction with:\n\n"
        "```bash\n"
        "PYTHONDONTWRITEBYTECODE=1 python3 packet/verify_extension.py delivery results --packet packet\n"
        "```\n",
        encoding="utf-8",
    )
    write_manifest(root)
    verify_manifest(root)
    relocated = run_verifier(root / "packet", root / "results")
    if relocated.get("status") != "pass":
        raise SystemExit("relocated delivery verifier failed")
    return root


def zip_root(root: Path, destination: Path) -> Path:
    archive = destination / f"{root.name}.zip"
    sidecar = archive.with_suffix(".zip.sha256")
    if archive.exists() or sidecar.exists():
        raise SystemExit(f"refusing to overwrite {archive} or its sidecar")
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not excluded(path):
                handle.write(path, f"{root.name}/{path.relative_to(root)}")
    sidecar.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    return archive


def independently_verify_zip(archive: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="meritkv-qwen-zip-audit-") as temporary:
        extraction = Path(temporary)
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            if len(names) != len(set(names)) or any(
                Path(name).is_absolute() or ".." in Path(name).parts for name in names
            ):
                raise SystemExit("ZIP contains duplicate or unsafe member names")
            handle.extractall(extraction)
        root = extraction / OUTPUT_NAME
        manifest_report = verify_manifest(root)
        delivery = run_verifier(root / "packet", root / "results")
        if delivery.get("status") != "pass":
            raise SystemExit("independently extracted relocated verifier failed")
        archive_files = {
            str(path.relative_to(extraction))
            for path in extraction.rglob("*")
            if path.is_file()
        }
        if archive_files != set(names):
            raise SystemExit("extracted ZIP inventory differs from central directory")
        return {
            "status": "pass",
            "zip_members": len(names),
            **manifest_report,
            "relocated_delivery_verifier": "pass",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    final_root = DELIVERABLES / OUTPUT_NAME
    final_zip = DELIVERABLES / f"{OUTPUT_NAME}.zip"
    final_sidecar = final_zip.with_suffix(".zip.sha256")
    if final_root.exists() or final_zip.exists() or final_sidecar.exists():
        raise SystemExit(f"refusing to overwrite existing Qwen deliverable: {final_root}")

    references = verify_preserved_artifacts()
    with tempfile.TemporaryDirectory(
        prefix=f".{OUTPUT_NAME}.staging-", dir=DELIVERABLES
    ) as temporary:
        staging = Path(temporary)
        staged_root = build_extension(results, references, staging)
        staged_zip = zip_root(staged_root, staging)
        independent = independently_verify_zip(staged_zip)
        after = verify_preserved_artifacts()
        if references != after:
            raise SystemExit("preserved artifact references changed during Qwen packaging")
        shutil.move(str(staged_root), final_root)
        shutil.move(str(staged_zip), final_zip)
        shutil.move(str(staged_zip.with_suffix(".zip.sha256")), final_sidecar)
    archive = final_zip
    print(
        json.dumps(
            {
                "status": "pass",
                "extension": str(archive),
                "extension_sha256": sha256(archive),
                "sidecar": str(archive.with_suffix(".zip.sha256")),
                "independent_extraction": independent,
                "preserved_artifacts": after,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
