#!/usr/bin/env python3
"""Run the complete MeritKV splice validation inside a Colab T4 session."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path("/content/meritkv_splice")
OUTPUT = Path("/content/meritkv_exact_splice_validation_outputs")
ARCHIVE = Path("/content/meritkv_exact_splice_validation_outputs.zip")
TOKEN_FILE = ROOT / ".hf_token"

AVAILABLE_MODELS = {
    "gpt2": "gpt2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gemma4_e2b": "google/gemma-4-E2B-it",
    "qwen25_15b": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma2b": "google/gemma-2b-it",
    "phi3mini": "microsoft/Phi-3-mini-4k-instruct",
}
AVAILABLE_ATTENTION_IMPLEMENTATIONS = ("eager", "sdpa")
COMMAND_LOG: list[str] = []
SELECTION_FILE = ROOT / "model_selection.txt"
ATTENTION_SELECTION_FILE = ROOT / "attention_selection.txt"


def selected_models() -> dict[str, str]:
    if not SELECTION_FILE.is_file():
        return dict(AVAILABLE_MODELS)
    requested = [
        item.strip()
        for item in SELECTION_FILE.read_text(encoding="utf-8").replace("\n", ",").split(",")
        if item.strip()
    ]
    if not requested:
        raise ValueError("model_selection.txt contains no model keys")
    unknown = sorted(set(requested).difference(AVAILABLE_MODELS))
    if unknown:
        raise ValueError(f"Unknown model selection: {unknown}")
    return {key: AVAILABLE_MODELS[key] for key in requested}


def selected_attention_implementations() -> tuple[str, ...]:
    if not ATTENTION_SELECTION_FILE.is_file():
        return AVAILABLE_ATTENTION_IMPLEMENTATIONS
    requested = tuple(
        item.strip()
        for item in ATTENTION_SELECTION_FILE.read_text(encoding="utf-8")
        .replace("\n", ",")
        .split(",")
        if item.strip()
    )
    if not requested:
        raise ValueError("attention_selection.txt contains no implementations")
    unknown = sorted(set(requested).difference(AVAILABLE_ATTENTION_IMPLEMENTATIONS))
    if unknown:
        raise ValueError(f"Unknown attention selection: {unknown}")
    return requested


def run_and_capture(command: list[str], log_path: Path) -> int:
    """Stream a command to the terminal while retaining its complete log."""
    rendered = "$ " + " ".join(shlex.quote(part) for part in command)
    COMMAND_LOG.append(rendered)
    (OUTPUT / "command_log.txt").write_text(
        "\n".join(COMMAND_LOG) + "\n", encoding="utf-8"
    )
    print(rendered, flush=True)
    lines: list[str] = []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return_code = process.wait()
    log_path.write_text("".join(lines), encoding="utf-8")
    return return_code


def record_environment() -> dict[str, Any]:
    import torch
    import transformers

    nvidia = subprocess.run(
        ["nvidia-smi"], text=True, capture_output=True, check=False
    )
    (OUTPUT / "nvidia_smi.txt").write_text(
        nvidia.stdout + nvidia.stderr, encoding="utf-8"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
        capture_output=True,
        check=False,
    )
    (OUTPUT / "pip_freeze.txt").write_text(
        freeze.stdout + freeze.stderr, encoding="utf-8"
    )
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name,
    }
    (OUTPUT / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    cli_version = ROOT / "colab_cli_version.txt"
    if cli_version.is_file():
        shutil.copy2(cli_version, OUTPUT / cli_version.name)
    if not torch.cuda.is_available():
        raise RuntimeError("The allocated Colab runtime has no CUDA GPU")
    if "T4" not in str(gpu_name):
        raise RuntimeError(f"Requested a T4 but Colab allocated {gpu_name!r}")
    return environment


def record_executed_inputs() -> None:
    destination = OUTPUT / "executed_inputs"
    destination.mkdir(parents=True, exist_ok=True)
    names = (
        "validate_exact_splice.py",
        "summarize_validation.py",
        "run_remote_validation.py",
        "requirements-colab.txt",
    )
    for name in names:
        source = ROOT / name
        if not source.is_file():
            raise RuntimeError(f"Missing executed input: {source}")
        shutil.copy2(source, destination / name)


def authenticate_hugging_face() -> str:
    from huggingface_hub import login

    if not TOKEN_FILE.is_file():
        raise RuntimeError(f"Missing uploaded Hugging Face token: {TOKEN_FILE}")
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    TOKEN_FILE.unlink(missing_ok=True)
    if not token:
        raise RuntimeError("The uploaded Hugging Face token is empty")
    login(token=token, add_to_git_credential=False)
    return token


def run_validator(
    model_key: str, model_id: str, revision: str, attention: str
) -> dict[str, Any]:
    output_json = OUTPUT / f"{model_key}_t4_f16_{attention}.json"
    log_path = OUTPUT / f"{model_key}_t4_f16_{attention}.log"
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
        "float16",
        "--attn-implementation",
        attention,
        "--sequence-length",
        "128",
        "--shared-ratios",
        "0.5",
        "0.75",
        "--samples",
        "4",
        "--max-new-tokens",
        "8",
        "--seed",
        "42",
        "--atol",
        "0.001",
        "--rtol",
        "0.001",
        "--output",
        str(output_json),
    ]
    return_code = run_and_capture(command, log_path)
    if return_code not in (0, 2):
        raise RuntimeError(
            f"{model_key}/{attention} execution failed with return code {return_code}"
        )
    report = json.loads(output_json.read_text(encoding="utf-8"))
    return {
        "model": model_key,
        "attention": attention,
        "validator_return_code": return_code,
        "strict_pass": report["summary"]["all_strict_pass"],
        "token_semantic_pass": report["summary"].get(
            "all_token_semantic_pass", False
        ),
        "report": output_json.name,
        "log": log_path.name,
    }


def make_summary() -> None:
    command = [
        sys.executable,
        str(ROOT / "summarize_validation.py"),
        str(OUTPUT),
        "--output-json",
        str(OUTPUT / "combined_summary.json"),
        "--output-md",
        str(OUTPUT / "VALIDATION_SUMMARY.md"),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Summary generation failed with return code {completed.returncode}"
        )


def make_archive() -> None:
    manifest_lines = []
    for path in sorted(path for path in OUTPUT.rglob("*") if path.is_file()):
        if path.name == "MANIFEST_SHA256.txt":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(
            f"{digest}  {path.relative_to(OUTPUT).as_posix()}"
        )
    (OUTPUT / "MANIFEST_SHA256.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    made = Path(shutil.make_archive(str(ARCHIVE.with_suffix("")), "zip", OUTPUT))
    if made != ARCHIVE:
        raise RuntimeError(f"Unexpected archive path: {made}")
    print(f"RESULT_ARCHIVE={ARCHIVE}", flush=True)


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    models = selected_models()
    attention_implementations = selected_attention_implementations()
    status: dict[str, Any] = {
        "models": models,
        "attention_implementations": list(attention_implementations),
        "configuration": {
            "dtype": "float16",
            "sequence_length": 128,
            "shared_ratios": [0.5, 0.75],
            "samples": 4,
            "max_new_tokens": 8,
            "seed": 42,
            "atol": 0.001,
            "rtol": 0.001,
        },
        "environment": None,
        "model_revisions": {},
        "runs": [],
        "execution_failures": [],
    }

    token: str | None = None
    try:
        status["environment"] = record_environment()
        record_executed_inputs()
        token = authenticate_hugging_face()

        from huggingface_hub import model_info

        for model_key, model_id in models.items():
            try:
                status["model_revisions"][model_key] = model_info(
                    model_id, token=token
                ).sha
            except Exception as exc:  # retain other model if one access check fails
                status["execution_failures"].append(
                    {
                        "stage": "resolve_revision",
                        "model": model_key,
                        "error": repr(exc),
                    }
                )

        revision_lines = [
            f"{key}  {models[key]}  {revision}"
            for key, revision in status["model_revisions"].items()
        ]
        (OUTPUT / "model_revisions.txt").write_text(
            "\n".join(revision_lines) + ("\n" if revision_lines else ""),
            encoding="utf-8",
        )

        self_test_dir = OUTPUT / "self_test"
        self_test_dir.mkdir(parents=True, exist_ok=True)
        self_test = [
            sys.executable,
            str(ROOT / "validate_exact_splice.py"),
            "--tiny-model",
            "tiny-qwen2",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--attn-implementation",
            "eager",
            "--sequence-length",
            "24",
            "--shared-ratios",
            "0.5",
            "--samples",
            "1",
            "--max-new-tokens",
            "4",
            "--atol",
            "1e-5",
            "--rtol",
            "1e-5",
            "--output",
            str(self_test_dir / "harness_self_test.json"),
        ]
        self_test_code = run_and_capture(
            self_test, self_test_dir / "harness_self_test.log"
        )
        if self_test_code != 0:
            raise RuntimeError(f"Harness self-test returned {self_test_code}")

        for model_key, model_id in models.items():
            revision = status["model_revisions"].get(model_key)
            if not revision:
                continue
            for attention in attention_implementations:
                try:
                    status["runs"].append(
                        run_validator(
                            model_key, model_id, revision, attention
                        )
                    )
                except Exception as exc:
                    status["execution_failures"].append(
                        {
                            "stage": "validation",
                            "model": model_key,
                            "attention": attention,
                            "error": repr(exc),
                        }
                    )
    except Exception as exc:
        status["execution_failures"].append(
            {"stage": "setup", "error": repr(exc)}
        )
    finally:
        token = None
        os.environ.pop("HF_TOKEN", None)
        TOKEN_FILE.unlink(missing_ok=True)

    try:
        make_summary()
    except Exception as exc:
        status["execution_failures"].append(
            {"stage": "summary", "error": repr(exc)}
        )

    status["requested_run_count"] = len(models) * len(attention_implementations)
    status["all_requested_runs_completed"] = (
        len(status["runs"]) == status["requested_run_count"]
    )
    status["all_completed_runs_strict_pass"] = bool(status["runs"]) and all(
        run["strict_pass"] for run in status["runs"]
    )
    status["all_completed_runs_token_semantic_pass"] = bool(status["runs"]) and all(
        run["token_semantic_pass"] for run in status["runs"]
    )
    (OUTPUT / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    make_archive()

    print(json.dumps(status, indent=2), flush=True)
    return 0 if not status["execution_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
