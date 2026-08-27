#!/usr/bin/env python3
"""Colab-side driver for the native vLLM execute-or-bypass experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


ROOT = Path("/content/meritkv_native_bypass")
RESULTS = ROOT / "results" / "final"
ARCHIVE = Path("/content/meritkv_native_vllm_execute_bypass_results.zip")


def archive_results() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    shutil.make_archive(str(ARCHIVE.with_suffix("")), "zip", RESULTS)
    print(f"RESULT_ARCHIVE={ARCHIVE}", flush=True)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    status = {
        "dependency_install_completed": False,
        "benchmark_started": False,
        "benchmark_completed": False,
        "benchmark_return_code": None,
        "failure": None,
    }
    return_code = 1
    try:
        install_log = RESULTS / "dependency_install.log"
        with install_log.open("w", encoding="utf-8") as handle:
            install = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "-r",
                    str(ROOT / "requirements.txt"),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if install.returncode != 0:
            raise RuntimeError(
                f"Dependency installation failed with code {install.returncode}; "
                f"see {install_log}"
            )
        status["dependency_install_completed"] = True
        source_snapshot = ROOT / "source_snapshot"
        source_snapshot.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "python3",
                "-m",
                "zipfile",
                "-e",
                str(ROOT / "meritkv_source.zip"),
                str(source_snapshot),
            ],
            check=True,
        )
        os.chmod(ROOT / "run_experiment.sh", 0o755)
        environment = os.environ.copy()
        environment["MERITKV_OUTPUT_DIR"] = str(RESULTS)
        environment["MERITKV_SOURCE_ROOT"] = str(source_snapshot / "v10" / "src")
        environment["SHADOWKV_CONFIG"] = str(
            source_snapshot / "v10" / "config" / "config.yaml"
        )
        command = ["bash", str(ROOT / "run_experiment.sh")]
        (RESULTS / "remote_command.json").write_text(
            json.dumps({"command": command, "environment_overrides": {
                key: environment[key]
                for key in ("MERITKV_OUTPUT_DIR", "MERITKV_SOURCE_ROOT", "SHADOWKV_CONFIG")
            }}, indent=2),
            encoding="utf-8",
        )
        status["benchmark_started"] = True
        benchmark_log = RESULTS / "benchmark_driver.log"
        with benchmark_log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        status["benchmark_completed"] = True
        status["benchmark_return_code"] = int(completed.returncode)
        return_code = int(completed.returncode)
    except Exception:
        failure = traceback.format_exc()
        status["failure"] = failure
        (RESULTS / "REMOTE_DRIVER_FAILURE.txt").write_text(failure, encoding="utf-8")
        return_code = 1
    finally:
        (RESULTS / "RUN_STATUS.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        archive_results()
    return return_code


raise SystemExit(main())
