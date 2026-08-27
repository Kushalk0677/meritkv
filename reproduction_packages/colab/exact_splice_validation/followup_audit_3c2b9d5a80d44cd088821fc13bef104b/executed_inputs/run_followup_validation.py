#!/usr/bin/env python3
"""Run the targeted GPT-2 float32 and five-model natural-text controls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path("/content/meritkv_splice")
sys.path.insert(0, str(ROOT))
import run_remote_validation as base


OUTPUT = Path("/content/meritkv_exact_splice_followup_outputs")
ARCHIVE = Path("/content/meritkv_exact_splice_followup_outputs.zip")
base.OUTPUT = OUTPUT
base.ARCHIVE = ARCHIVE

MODELS = {
    "gpt2": "gpt2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen25_15b": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma2b": "google/gemma-2b-it",
    "phi3mini": "microsoft/Phi-3-mini-4k-instruct",
}


def run_job(
    model_key: str,
    model_id: str,
    revision: str,
    dtype: str,
    input_mode: str,
    samples: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    precision_tag = "f32" if dtype == "float32" else "f16"
    stem = f"{model_key}_t4_{precision_tag}_sdpa_{input_mode}"
    output_json = OUTPUT / f"{stem}.json"
    log_path = OUTPUT / f"{stem}.log"
    tolerance = "0.00001" if dtype == "float32" else "0.001"
    command = [
        sys.executable,
        str(ROOT / "validate_exact_splice.py"),
        "--model-id",
        model_id,
        "--revision",
        revision,
        "--device",
        "cuda:0",
        "--dtype",
        dtype,
        "--attn-implementation",
        "sdpa",
        "--input-mode",
        input_mode,
        "--sequence-length",
        "128",
        "--samples",
        str(samples),
        "--max-new-tokens",
        str(max_new_tokens),
        "--seed",
        "42",
        "--atol",
        tolerance,
        "--rtol",
        tolerance,
    ]
    if input_mode == "random":
        command.extend(["--shared-ratios", "0.5", "0.75"])
    command.extend(["--output", str(output_json)])
    return_code = base.run_and_capture(command, log_path)
    if return_code not in (0, 2):
        raise RuntimeError(f"{stem} execution failed with return code {return_code}")
    report = json.loads(output_json.read_text(encoding="utf-8"))
    return {
        "model": model_key,
        "dtype": dtype,
        "attention": "sdpa",
        "input_mode": input_mode,
        "validator_return_code": return_code,
        "strict_pass": report["summary"]["all_strict_pass"],
        "token_semantic_pass": report["summary"].get(
            "all_token_semantic_pass", False
        ),
        "report": output_json.name,
        "log": log_path.name,
    }


def record_inputs() -> None:
    destination = OUTPUT / "executed_inputs"
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "validate_exact_splice.py",
        "summarize_validation.py",
        "run_remote_validation.py",
        "run_followup_validation.py",
        "requirements-colab.txt",
    ):
        shutil.copy2(ROOT / name, destination / name)


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    status: dict[str, Any] = {
        "purpose": (
            "Targeted GPT-2 float32 control on the original random-token cases and "
            "five-model float16 SDPA validation on fixed natural-text cases."
        ),
        "models": MODELS,
        "environment": None,
        "model_revisions": {},
        "runs": [],
        "execution_failures": [],
    }

    token: str | None = None
    try:
        status["environment"] = base.record_environment()
        record_inputs()
        token = base.authenticate_hugging_face()

        from huggingface_hub import model_info

        for key, model_id in MODELS.items():
            status["model_revisions"][key] = model_info(model_id, token=token).sha
        (OUTPUT / "model_revisions.txt").write_text(
            "".join(
                f"{key}  {MODELS[key]}  {status['model_revisions'][key]}\n"
                for key in MODELS
            ),
            encoding="utf-8",
        )

        jobs = [
            ("gpt2", "float32", "random", 4, 8),
            *(
                (key, "float16", "natural", 8, 16)
                for key in MODELS
            ),
        ]
        for key, dtype, input_mode, samples, max_new_tokens in jobs:
            try:
                status["runs"].append(
                    run_job(
                        key,
                        MODELS[key],
                        status["model_revisions"][key],
                        dtype,
                        input_mode,
                        samples,
                        max_new_tokens,
                    )
                )
            except Exception as exc:
                status["execution_failures"].append(
                    {
                        "stage": "validation",
                        "model": key,
                        "dtype": dtype,
                        "input_mode": input_mode,
                        "error": repr(exc),
                    }
                )
    except Exception as exc:
        status["execution_failures"].append({"stage": "setup", "error": repr(exc)})
    finally:
        token = None
        os.environ.pop("HF_TOKEN", None)
        base.TOKEN_FILE.unlink(missing_ok=True)

    try:
        base.make_summary()
    except Exception as exc:
        status["execution_failures"].append({"stage": "summary", "error": repr(exc)})

    status["requested_run_count"] = 6
    status["all_requested_runs_completed"] = len(status["runs"]) == 6
    status["all_completed_runs_token_semantic_pass"] = bool(status["runs"]) and all(
        run["token_semantic_pass"] for run in status["runs"]
    )
    (OUTPUT / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    base.make_archive()
    print(json.dumps(status, indent=2), flush=True)
    return 0 if not status["execution_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
