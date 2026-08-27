#!/usr/bin/env python3
"""Guarded First Light controller for the frozen Qwen2.5 capacity extension."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT = Path("/home/jade_hand/research/shadowkv/gemma4_runtime_overlay_matrix_20260718")
PACKET = PROJECT / "meritkv_qwen25_capacity_pressure_20260823"
RESULTS = Path("/datapool/experiments/meritkv/qwen25_capacity_pressure_20260824")
CONTROL = RESULTS / "control"
RECOVERY_STATE = CONTROL / "recovery_state.json"
IMAGE = "shadowkv-sglang-native-admission:2026-08-20-swa-counters"
IMAGE_ID = "sha256:f4593e56ec7f7858f465a62f36dfc42ba8a2d9e91b7700cba50e8b21809e4d3f"
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
MODEL_REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
MODEL_SNAPSHOT = Path(
    "/datapool/cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/"
    f"snapshots/{MODEL_REVISION}"
)
CONTAINER_SNAPSHOT = (
    "/hf_hub/models--Qwen--Qwen2.5-32B-Instruct/"
    f"snapshots/{MODEL_REVISION}"
)
TOKENIZER_MANIFEST = "a28a1734767030cbbdc1588ea53965c763bae58e47596a4f03b237ac15b2c8b0"
TOKENIZER_FILE_HASHES = {
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
MODEL_CONFIG_HASH = "9c6772f138ef9e5b3d1c18f2c87e451bbc01f5f1a4eabb36f9bf4f53829b903e"
MODEL_INDEX_HASH = "0183543f2e6e40d3d1e863ed0b2a8c9cdaf2ee4045ce724a45cea30e9995a0af"
PROD_NAME = "qwen38-27b-fp8-sglang"
PROD_URL = "http://127.0.0.1:8017"
PROD_DIR = Path("/home/jade_hand/active/services/qwen38-27b-fp8-sglang")
SERVER_NAME = "meritkv-capacity-qwen25-server"
CLIENT_NAME = "meritkv-capacity-qwen25-cell-client"
SERVER_PORT = 30000
MAX_TOTAL_TOKENS = 16384
PAGE_SIZE = 1
OUTAGE_NOT_BEFORE = datetime.fromisoformat("2026-08-23T23:30:00-04:00")
HERMES_DIR = Path("/home/jade_hand/.hermes/hermes-agent")
HEALTH_JOB_ID = "b8813f6b0489"
ARMS_LRU = (
    "native_admit_all_lru",
    "meritkv_write_through_lru",
    "meritkv_skip_write_only_lru",
    "meritkv_joint_skip_write_skip_lookup_lru",
)
LFU_ARM = "native_admit_all_lfu"
ARMS = (*ARMS_LRU, LFU_ARM)
EXPECTED_CELLS = 31
EXPECTED_REQUESTS = 3344


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def physical_eviction_metric_snapshot(text: str) -> dict[str, object]:
    """Extract actual physical-eviction samples from a Prometheus exposition."""
    definitions: set[str] = set()
    samples: dict[str, float] = {}
    relevant_lines: list[str] = []
    definition_names = {
        "sglang:evicted_tokens_total",
        "sglang:eviction_duration_seconds",
    }
    sample_names = {
        "sglang:evicted_tokens_total",
        "sglang:eviction_duration_seconds_count",
        "sglang:eviction_duration_seconds_sum",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("# HELP "):
            parts = line.split(None, 3)
            if len(parts) >= 3 and parts[2] in definition_names:
                definitions.add(parts[2])
                relevant_lines.append(line)
            continue
        if not line or line.startswith("#"):
            continue
        try:
            sample, raw_value = line.rsplit(None, 1)
            metric_name = sample.split("{", 1)[0]
            if metric_name not in sample_names:
                continue
            value = float(raw_value)
        except (ValueError, TypeError):
            continue
        if not math.isfinite(value):
            continue
        samples[sample] = value
        relevant_lines.append(line)
    evicted = sum(
        value
        for sample, value in samples.items()
        if sample.split("{", 1)[0] == "sglang:evicted_tokens_total"
    )
    calls = sum(
        value
        for sample, value in samples.items()
        if sample.split("{", 1)[0]
        == "sglang:eviction_duration_seconds_count"
    )
    return {
        "definitions": sorted(definitions),
        "samples": dict(sorted(samples.items())),
        "evicted_tokens_total": evicted,
        "eviction_duration_count": calls,
        "relevant_lines": relevant_lines,
    }


def physical_eviction_metrics_proven(snapshot: dict[str, object]) -> bool:
    return (
        set(snapshot.get("definitions") or [])
        == {
            "sglang:evicted_tokens_total",
            "sglang:eviction_duration_seconds",
        }
        and float(snapshot.get("evicted_tokens_total") or 0.0) > 0
        and float(snapshot.get("eviction_duration_count") or 0.0) > 0
    )


def normalize_config_record(value: dict) -> dict:
    """Canonicalize Docker fields whose list order is not semantically stable."""
    result = dict(value)
    mounts = list(result.get("Mounts") or [])
    result["Mounts"] = sorted(
        mounts,
        key=lambda row: (
            str(row.get("Destination") or ""),
            str(row.get("Source") or ""),
            str(row.get("Type") or ""),
        ),
    )
    return result


def now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def write_atomic_json(path: Path, payload: dict) -> None:
    """Durably replace a JSON receipt, including its directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class Controller:
    def __init__(self) -> None:
        CONTROL.mkdir(parents=True, exist_ok=True)
        self.log_path = CONTROL / "controller.log"
        self.lock_handle = (CONTROL / "controller.lock").open("a+")
        self.prod_id = ""
        self.prod_image_id = ""
        self.prod_compose_hash = ""
        self.prod_config_hash = ""
        self.restore_required = False
        self.monitor_paused = False
        self.server_argv: list[str] = []
        self.server_config_hash = ""
        self.server_active = False
        self.retention = ""
        self.server_label = ""
        self.server_launch_log_relative = ""
        self.server_launch_log_hash = ""
        self.canary_receipt_relative = ""
        self.canary_receipt_hash = ""
        self.primer_receipt_relative = ""
        self.primer_receipt_hash = ""
        self.outage_started_at = ""

    def log(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def state(self, status: str, phase: str, detail: str = "") -> None:
        payload = {
            "experiment": "meritkv_qwen25_capacity_pressure_20260824",
            "status": status,
            "phase": phase,
            "detail": detail,
            "updated_at": now(),
            "pid": os.getpid(),
        }
        temporary = CONTROL / "state.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, CONTROL / "state.json")

    def run(
        self,
        command: list[str],
        *,
        check: bool = True,
        capture: bool = False,
        timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        merged_env = os.environ.copy()
        merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
        if env:
            merged_env.update(env)
        return subprocess.run(
            command,
            check=check,
            text=True,
            capture_output=capture,
            timeout=timeout,
            cwd=cwd,
            env=merged_env,
        )

    def acquire_lock(self) -> None:
        fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def acquire_recovery_lock(self) -> None:
        self.recovery_lock_handle = (CONTROL / "recovery.lock").open("a+")
        fcntl.flock(
            self.recovery_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB
        )

    def docker_inspect(self, fmt: str, name: str = PROD_NAME) -> str:
        return self.run(
            ["docker", "inspect", "--format", fmt, name], capture=True
        ).stdout.strip()

    def normalized_production_config(self, target: str) -> dict:
        raw = json.loads(
            self.run(["docker", "inspect", target], capture=True).stdout
        )[0]
        return normalize_config_record({
            "Path": raw.get("Path"),
            "Args": raw.get("Args"),
            "Image": raw.get("Image"),
            "Config": raw.get("Config"),
            "HostConfig": raw.get("HostConfig"),
            "Mounts": raw.get("Mounts"),
            "Ports": (raw.get("NetworkSettings") or {}).get("Ports"),
        })

    def get_json(self, url: str, timeout: float = 10) -> object:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)

    def get_text(self, url: str, timeout: float = 10) -> str:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")

    def post_json(self, url: str, payload: dict, timeout: float = 300) -> object:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def production_healthy(self) -> bool:
        try:
            if self.docker_inspect("{{.State.Status}}") != "running":
                return False
            if self.docker_inspect("{{.State.Health.Status}}") != "healthy":
                return False
            if self.docker_inspect("{{.RestartCount}}") != "0":
                return False
            if self.docker_inspect("{{.State.OOMKilled}}") != "false":
                return False
            self.get_text(f"{PROD_URL}/health")
            models = self.get_json(f"{PROD_URL}/v1/models")
            return "qwen38-27b-fp8" in json.dumps(models)
        except Exception:
            return False

    def assert_only_production_on_gpu(self) -> None:
        top = self.run(
            ["docker", "top", PROD_NAME, "-eo", "pid"], capture=True
        ).stdout.splitlines()[1:]
        allowed = {int(row.strip().split()[0]) for row in top if row.strip()}
        query = self.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture=True,
        ).stdout
        unexpected = []
        for row in csv.reader(query.splitlines()):
            if len(row) >= 2 and row[0].strip().isdigit():
                if int(row[0]) not in allowed and float(row[1]) > 128:
                    unexpected.append(row)
        if unexpected:
            raise RuntimeError(f"unexpected GPU owners: {unexpected}")

    def assert_production_nvfp4_profile(
        self, normalized_config: dict, startup_logs: str
    ) -> None:
        image_reference = str((normalized_config.get("Config") or {}).get("Image") or "")
        if "nvfp4" not in image_reference.lower():
            raise RuntimeError(
                f"accepted production image is not the NVFP4 image: {image_reference}"
            )
        model_mounts = [
            row
            for row in normalized_config.get("Mounts") or []
            if row.get("Destination") == "/model"
        ]
        if (
            len(model_mounts) != 1
            or "nvfp4" not in str(model_mounts[0].get("Source") or "").lower()
            or model_mounts[0].get("RW") is not False
        ):
            raise RuntimeError(f"accepted production NVFP4 model mount drift: {model_mounts}")
        args = list(normalized_config.get("Args") or [])
        if "--model-path" not in args or "/model" not in args:
            raise RuntimeError("accepted production args do not bind --model-path /model")
        if "quant=compressed-tensors" not in startup_logs.lower():
            raise RuntimeError(
                "accepted production startup log does not prove compressed-tensors quantization"
            )

    def assert_no_foreign_waiter(self) -> None:
        result = self.run(
            ["pgrep", "-af", "wait_for_sglang_idle.py"],
            check=False,
            capture=True,
        )
        rows = [row for row in result.stdout.splitlines() if row.strip()]
        if rows:
            raise RuntimeError(f"foreign idle waiter is active: {rows}")

    def wait_for_foreign_waiter_clear(self, timeout_seconds: int = 7200) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_log = 0.0
        while True:
            result = self.run(
                ["pgrep", "-af", "wait_for_sglang_idle.py"],
                check=False,
                capture=True,
            )
            rows = [row for row in result.stdout.splitlines() if row.strip()]
            if not rows:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"foreign idle waiter did not clear: {rows}")
            if time.monotonic() - last_log >= 60:
                self.log(f"WAIT foreign production operator still active: {rows}")
                last_log = time.monotonic()
            time.sleep(10)

    def validate_tokenizer_files(self) -> None:
        """Bind the Qwen tokenizer to the frozen August manifest inputs."""
        observed = {
            name: sha256(MODEL_SNAPSHOT / name)
            for name in TOKENIZER_FILE_HASHES
        }
        if observed != TOKENIZER_FILE_HASHES:
            raise RuntimeError(
                f"Qwen tokenizer file drift: {observed} != {TOKENIZER_FILE_HASHES}"
            )

    def protocol_hash(self) -> str:
        return sha256(PACKET / "predeclared_protocol.json")

    def freeze_hash(self) -> str:
        path = CONTROL / "evaluation_freeze.json"
        return sha256(path) if path.is_file() else ""

    def preflight(self) -> None:
        self.state("preflight", "inputs", "validating frozen assets")
        self.run([sys.executable, str(PACKET / "verify_extension.py"), "inputs"])
        image_id = self.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE],
            capture=True,
        ).stdout.strip()
        if image_id != IMAGE_ID:
            raise RuntimeError(f"runtime image drift: {image_id}")
        if sha256(MODEL_SNAPSHOT / "config.json") != MODEL_CONFIG_HASH:
            raise RuntimeError("Qwen config hash drift")
        if sha256(MODEL_SNAPSHOT / "model.safetensors.index.json") != MODEL_INDEX_HASH:
            raise RuntimeError("Qwen weight-index hash drift")
        self.validate_tokenizer_files()
        if shutil.disk_usage("/datapool").free < 100 * 1024**3:
            raise RuntimeError("less than 100 GiB free on /datapool")
        for arm, expected in (
            ("meritkv_skip_write_only_lru", "lookup=1;write=0"),
            ("meritkv_joint_skip_write_skip_lookup_lru", "lookup=0;write=0"),
        ):
            code = (
                "import importlib.util;"
                "p='/workspace/project/meritkv_qwen25_capacity_pressure_20260823/run_capacity_cell.py';"
                "s=importlib.util.spec_from_file_location('c',p);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "print(m.capacity_extra_key(skip_lookup=True,skip_write=True,tag='probe'))"
            )
            output = self.run(
                [
                    "docker", "run", "--rm", "--network", "none",
                    "-e", f"MERITKV_EFFECTIVE_ARM={arm}",
                    "-e", "MERITKV_RETENTION_POLICY=lru",
                    "-e", f"MERITKV_MAX_TOTAL_TOKENS={MAX_TOTAL_TOKENS}",
                    "-v", f"{PROJECT}:/workspace/project:ro",
                    "--entrypoint", "python3", IMAGE, "-c", code,
                ],
                capture=True,
            ).stdout.strip()
            if expected not in output:
                raise RuntimeError(f"independent action probe failed for {arm}: {output}")
        self.run(
            [
                "docker", "run", "--rm", "--network", "none",
                "-v", f"{PROJECT}:/workspace/project:ro",
                "--entrypoint", "python3", IMAGE,
                "/workspace/project/meritkv_qwen25_capacity_pressure_20260823/tests/"
                "test_lazy_eviction_metrics.py",
            ],
            capture=True,
        )
        self.log(
            "PREFLIGHT PASS inputs, model, image, independent action mapping, "
            "and lazy eviction metrics"
        )

    def snapshot_production(self) -> None:
        if not self.production_healthy():
            raise RuntimeError("production is not healthy")
        self.run(["docker", "compose", "-f", str(PROD_DIR / "docker-compose.yml"), "config", "-q"])
        self.prod_id = self.docker_inspect("{{.Id}}")
        self.prod_image_id = self.docker_inspect("{{.Image}}")
        self.prod_compose_hash = sha256(PROD_DIR / "docker-compose.yml")
        normalized_config = self.normalized_production_config(self.prod_id)
        production_started_at = self.docker_inspect("{{.State.StartedAt}}", self.prod_id)
        log_result = self.run(
            ["docker", "logs", "--since", production_started_at, self.prod_id],
            capture=True,
        )
        startup_logs = log_result.stdout + log_result.stderr
        self.assert_production_nvfp4_profile(normalized_config, startup_logs)
        self.prod_config_hash = json_hash(normalized_config)
        destination = CONTROL / "production_preimage"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "container_inspect.json").write_text(
            self.run(["docker", "inspect", PROD_NAME], capture=True).stdout
        )
        (destination / "compose_rendered.yaml").write_text(
            self.run(
                ["docker", "compose", "-f", str(PROD_DIR / "docker-compose.yml"), "config"],
                capture=True,
            ).stdout
        )
        (destination / "normalized_container_config.json").write_text(
            json.dumps(normalized_config, indent=2, sort_keys=True) + "\n"
        )
        (destination / "accepted_nvfp4_startup.log").write_text(startup_logs)
        (destination / "identity.json").write_text(
            json.dumps(
                {
                    "captured_at": now(),
                    "container_id": self.prod_id,
                    "image_id": self.prod_image_id,
                    "compose_sha256": self.prod_compose_hash,
                    "normalized_config_sha256": self.prod_config_hash,
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )
        for script, name, arguments in (
            (
                "probe_openai_chat.py", "direct_chat.json",
                ["--base-url", PROD_URL, "--model", "qwen38-27b-fp8", "--marker", "MERITKV_PREOUTAGE_OK", "--timeout", "180"],
            ),
            (
                "probe_openai_vision.py", "direct_vision.json",
                ["--base-url", PROD_URL, "--model", "qwen38-27b-fp8", "--timeout", "180"],
            ),
        ):
            result = self.run([sys.executable, str(PROD_DIR / script), *arguments], capture=True, timeout=240)
            (destination / name).write_text(result.stdout)
        self.assert_only_production_on_gpu()
        self.log(
            f"PRODUCTION SNAPSHOT id={self.prod_id} image={self.prod_image_id} "
            f"compose={self.prod_compose_hash} config={self.prod_config_hash}"
        )

    def wait_for_idle_captured(self) -> None:
        self.assert_no_foreign_waiter()
        result = self.run(
            [
                sys.executable, str(PROD_DIR / "wait_for_sglang_idle.py"),
                "--url", f"{PROD_URL}/v1/loads", "--container", PROD_NAME,
                "--quiet-seconds", "120", "--poll-seconds", "2",
            ],
            capture=True,
            timeout=7200,
        )
        stream = CONTROL / "fresh_idle_gate.jsonl"
        stream.write_text(result.stdout)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("idle gate emitted no receipt")
        receipt = json.loads(lines[-1])
        if receipt.get("event") != "IDLE_GATE_PASSED" or receipt.get("container_id") != self.prod_id:
            raise RuntimeError(f"idle gate did not bind accepted container: {receipt}")
        receipt_path = CONTROL / "fresh_idle_gate_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        assertion = self.run(
            [
                sys.executable, str(PROD_DIR / "assert_idle_before_stop.py"),
                "--receipt", str(receipt_path), "--container", PROD_NAME,
                "--url", f"{PROD_URL}/v1/loads",
            ],
            capture=True,
        )
        (CONTROL / "final_idle_assertion.json").write_text(assertion.stdout)
        if self.docker_inspect("{{.Id}}") != self.prod_id:
            raise RuntimeError("production container changed after idle gate")
        if sha256(PROD_DIR / "docker-compose.yml") != self.prod_compose_hash:
            raise RuntimeError("production compose changed after snapshot")
        self.log("IDLE GATE PASS 120 continuous seconds")

    def monitor_listing(self) -> str:
        return self.run(
            [
                str(HERMES_DIR / "venv/bin/python"),
                "-m", "hermes_cli.main", "cron", "list", "--all",
            ],
            capture=True,
            cwd=HERMES_DIR,
        ).stdout

    def monitor_is_active(self) -> bool:
        return f"{HEALTH_JOB_ID} [active]" in self.monitor_listing()

    def pause_monitor(self) -> None:
        marker = CONTROL / "local_model_health_paused"
        listing = self.monitor_listing()
        (CONTROL / "local_model_health_before.txt").write_text(listing)
        if f"{HEALTH_JOB_ID} [active]" in listing:
            marker.write_text(f"{now()} pause_intent\n")
            output = self.run(
                [str(HERMES_DIR / "venv/bin/python"), "-m", "hermes_cli.main", "cron", "pause", HEALTH_JOB_ID],
                capture=True,
                cwd=HERMES_DIR,
            ).stdout
            (CONTROL / "local_model_health_pause.txt").write_text(output)
            self.monitor_paused = True

    def resume_monitor(self) -> None:
        marker = CONTROL / "local_model_health_paused"
        if not marker.exists():
            return
        output = self.run(
            [str(HERMES_DIR / "venv/bin/python"), "-m", "hermes_cli.main", "cron", "resume", HEALTH_JOB_ID],
            capture=True,
            cwd=HERMES_DIR,
        ).stdout
        listing = self.monitor_listing()
        if f"{HEALTH_JOB_ID} [active]" not in listing:
            raise RuntimeError("local-model-health did not resume")
        (CONTROL / "local_model_health_resume.txt").write_text(output + listing)
        marker.rename(CONTROL / f"local_model_health_paused.resumed.{int(time.time())}")
        self.monitor_paused = False

    def arm_recovery(self, monitor_was_active: bool) -> None:
        payload = {
            "schema_version": 1,
            "run_id": f"meritkv-capacity-{int(time.time())}",
            "phase": "restore_armed",
            "restore_required": True,
            "restore_status": "armed",
            "armed_at": now(),
            "production": {
                "name": PROD_NAME,
                "container_id": self.prod_id,
                "image_id": self.prod_image_id,
                "normalized_config_sha256": self.prod_config_hash,
                "compose_path": str(PROD_DIR / "docker-compose.yml"),
                "compose_sha256": self.prod_compose_hash,
                "url": PROD_URL,
                "model": "qwen38-27b-fp8",
            },
            "experiment": {
                "server_name": SERVER_NAME,
                "client_name": CLIENT_NAME,
            },
            "monitor": {
                "job_id": HEALTH_JOB_ID,
                "was_active": monitor_was_active,
                "resume_required": monitor_was_active,
            },
        }
        write_atomic_json(RECOVERY_STATE, payload)
        self.restore_required = True
        self.log(f"RECOVERY ARMED state={RECOVERY_STATE} run_id={payload['run_id']}")

    def update_recovery_state(self, **changes: object) -> None:
        if not RECOVERY_STATE.is_file():
            raise RuntimeError("recovery state disappeared while restore was armed")
        payload = json.loads(RECOVERY_STATE.read_text(encoding="utf-8"))
        payload.update(changes)
        write_atomic_json(RECOVERY_STATE, payload)

    def begin_outage(self) -> None:
        if datetime.now(ZoneInfo("America/New_York")) < OUTAGE_NOT_BEFORE:
            raise RuntimeError(f"outage may not begin before {OUTAGE_NOT_BEFORE.isoformat()}")
        self.wait_for_foreign_waiter_clear()
        self.snapshot_production()
        self.wait_for_idle_captured()
        self.assert_only_production_on_gpu()
        monitor_was_active = self.monitor_is_active()
        self.arm_recovery(monitor_was_active)
        self.pause_monitor()
        self.outage_started_at = now()
        self.update_recovery_state(
            phase="outage_starting", outage_started_at=self.outage_started_at
        )
        (CONTROL / "outage_started_at.txt").write_text(self.outage_started_at + "\n")
        self.state("running", "outage", "stopping exact accepted production container")
        result = self.run(["docker", "stop", "--time", "120", self.prod_id], capture=True, timeout=180)
        (CONTROL / "production_stop.txt").write_text(result.stdout)
        if self.docker_inspect("{{.Id}}") != self.prod_id:
            raise RuntimeError("production container identity changed during stop")
        for _ in range(120):
            query = self.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                capture=True,
            ).stdout.strip()
            if not query:
                self.log("OUTAGE STARTED GPU released")
                return
            time.sleep(2)
        raise RuntimeError("GPU remained allocated after production stop")

    def stop_server(self) -> None:
        if self.run(["docker", "inspect", SERVER_NAME], check=False).returncode == 0:
            result = self.run(["docker", "logs", SERVER_NAME], check=False, capture=True)
            logs = result.stdout + result.stderr
            destination = CONTROL / "server_logs"
            destination.mkdir(parents=True, exist_ok=True)
            label = self.server_label or self.retention or "unknown"
            (destination / f"{label}_{time.time_ns()}.log").write_text(logs)
        self.run(["docker", "rm", "-f", CLIENT_NAME], check=False, capture=True)
        self.run(["docker", "rm", "-f", SERVER_NAME], check=False, capture=True)
        self.server_active = False

    def reset_server_cache_and_counters(self) -> None:
        self.run(
            [
                "curl", "-fsS", "-X", "POST",
                f"http://127.0.0.1:{SERVER_PORT}/flush_cache?timeout=30",
            ],
            capture=True,
        )
        self.run(
            [
                "curl", "-fsS", "-X", "POST",
                f"http://127.0.0.1:{SERVER_PORT}/shadowkv_admission_metrics/reset",
            ],
            capture=True,
        )

    def prime_physical_eviction_metrics(self, label: str) -> None:
        """Force one non-measured eviction so lazy Prometheus children exist."""
        self.reset_server_cache_and_counters()
        metrics_url = f"http://127.0.0.1:{SERVER_PORT}/metrics"
        before = physical_eviction_metric_snapshot(self.get_text(metrics_url))
        trace = PACKET / "traces/calibration_seed_32000.jsonl"
        candidates = []
        for line in trace.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            if metadata.get("shadowkv_trace_class") == "long_one_off_prefix":
                candidates.append(row)
        if not candidates:
            raise RuntimeError("physical-eviction primer has no frozen one-off prompts")

        requests: list[dict[str, object]] = []
        proven: dict[str, object] | None = None
        for row in candidates:
            prompt = str(row["prompt"])
            response = self.post_json(
                f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
                {
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=300,
            )
            if not isinstance(response, dict) or not response.get("choices"):
                raise RuntimeError(
                    f"physical-eviction primer request failed: {response}"
                )
            choice = response["choices"][0]
            message = choice.get("message") if isinstance(choice, dict) else {}
            output = str((message or {}).get("content") or "")
            requests.append(
                {
                    "request_id": row.get("request_id"),
                    "input_prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "output_sha256": hashlib.sha256(
                        output.encode("utf-8")
                    ).hexdigest(),
                }
            )
            current = physical_eviction_metric_snapshot(self.get_text(metrics_url))
            if (
                physical_eviction_metrics_proven(current)
                and float(current["evicted_tokens_total"])
                > float(before["evicted_tokens_total"])
                and float(current["eviction_duration_count"])
                > float(before["eviction_duration_count"])
            ):
                proven = current
                break
        if proven is None:
            raise RuntimeError(
                "physical-eviction primer exhausted the frozen one-off prompts "
                "without registering real eviction metrics"
            )

        self.reset_server_cache_and_counters()
        post_flush = physical_eviction_metric_snapshot(self.get_text(metrics_url))
        if not physical_eviction_metrics_proven(post_flush):
            raise RuntimeError(
                "physical-eviction metric children disappeared after primer flush"
            )
        primer_dir = CONTROL / "eviction_metric_primers"
        primer_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = primer_dir / f"{label}_{time.time_ns()}.json"
        receipt = {
            "schema_version": 1,
            "status": "pass",
            "purpose": "non_measured_lazy_physical_eviction_metric_registration",
            "label": label,
            "completed_at": now(),
            "runtime_image_id": IMAGE_ID,
            "retention": self.retention,
            "cache_implementation": "RadixCache",
            "hybrid_swa_memory": False,
            "server_argv": self.server_argv,
            "server_config_sha256": self.server_config_hash,
            "predeclared_protocol_sha256": self.protocol_hash(),
            "source_trace": "traces/calibration_seed_32000.jsonl",
            "source_trace_sha256": sha256(trace),
            "candidate_class": "long_one_off_prefix",
            "requests_executed": len(requests),
            "requests": requests,
            "before": before,
            "registered": proven,
            "post_flush": post_flush,
            "physical_eviction_delta": {
                "evicted_tokens": float(proven["evicted_tokens_total"])
                - float(before["evicted_tokens_total"]),
                "eviction_calls": float(proven["eviction_duration_count"])
                - float(before["eviction_duration_count"]),
            },
            "cache_flushed_after": True,
            "native_action_counters_reset_after": True,
            "excluded_from_calibration_smoke_and_evaluation": True,
        }
        write_atomic_json(receipt_path, receipt)
        self.primer_receipt_relative = str(receipt_path.relative_to(RESULTS))
        self.primer_receipt_hash = sha256(receipt_path)
        self.log(
            "EVICTION METRIC PRIMER PASS "
            f"requests={len(requests)} evicted_delta="
            f"{receipt['physical_eviction_delta']['evicted_tokens']} "
            f"receipt={self.primer_receipt_relative}"
        )

    def launch_server(self, label: str, retention: str) -> None:
        if retention not in {"lru", "lfu"}:
            raise RuntimeError(f"unsupported frozen retention policy: {retention}")
        self.stop_server()
        self.retention = retention
        self.server_label = label
        self.server_launch_log_relative = ""
        self.server_launch_log_hash = ""
        self.canary_receipt_relative = ""
        self.canary_receipt_hash = ""
        self.primer_receipt_relative = ""
        self.primer_receipt_hash = ""
        self.server_argv = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", CONTAINER_SNAPSHOT,
            "--served-model-name", MODEL_ID,
            "--host", "127.0.0.1", "--port", str(SERVER_PORT),
            "--context-length", "4096", "--mem-fraction-static", "0.88",
            "--max-total-tokens", str(MAX_TOTAL_TOKENS), "--page-size", str(PAGE_SIZE),
            "--dtype", "float16", "--attention-backend", "triton",
            "--sampling-backend", "pytorch", "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--radix-eviction-policy", retention,
            "--enable-cache-report", "--enable-metrics",
        ]
        self.server_config_hash = json_hash(self.server_argv)
        command = [
            "docker", "run", "-d", "--name", SERVER_NAME, "--network", "host",
            "--device", "nvidia.com/gpu=all", "--ipc", "host",
            "-e", "HF_HOME=/cache/huggingface",
            "-e", "TRANSFORMERS_CACHE=/cache/huggingface",
            "-e", "USE_HUB_KERNELS=NO", "-e", "FLASHINFER_DISABLE_VERSION_CHECK=1",
            "-e", f"SHADOWKV_ADMISSION_METRICS_PATH=/workspace/results/control/{label}_action_counters.json",
            "-v", "/datapool/cache/huggingface:/cache/huggingface",
            "-v", "/datapool/cache/huggingface/hub:/hf_hub:ro",
            "-v", f"{RESULTS}:/workspace/results",
            "--entrypoint", "python3", IMAGE, *self.server_argv[1:],
        ]
        with (CONTROL / "server_commands.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": now(), "label": label, "argv": command}) + "\n")
        self.run(command, capture=True, timeout=60)
        for _ in range(300):
            ready = False
            try:
                models = self.get_json(f"http://127.0.0.1:{SERVER_PORT}/v1/models")
                metrics = urllib.request.urlopen(
                    f"http://127.0.0.1:{SERVER_PORT}/metrics", timeout=10
                ).read().decode()
                ready = (
                    MODEL_ID in json.dumps(models)
                    and "sglang:kv_evictable_tokens" in metrics
                )
            except Exception:
                pass
            if ready:
                result = self.run(["docker", "logs", SERVER_NAME], capture=True)
                logs = result.stdout + result.stderr
                if "impl=RadixCache" not in logs or "hybrid_swa=False" not in logs:
                    raise RuntimeError(
                        "Qwen startup did not prove ordinary RadixCache with hybrid SWA disabled"
                    )
                canary = self.post_json(
                    f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
                    {
                        "model": MODEL_ID,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Return one short readiness token.",
                            }
                        ],
                        "temperature": 0,
                        "max_tokens": 1,
                        "stream": False,
                    },
                    timeout=300,
                )
                if not isinstance(canary, dict) or not canary.get("choices"):
                    raise RuntimeError(f"Qwen functional startup canary failed: {canary}")
                canary_dir = CONTROL / "server_canaries"
                canary_dir.mkdir(parents=True, exist_ok=True)
                canary_path = canary_dir / f"{label}.json"
                canary_path.write_text(
                    json.dumps(canary, indent=2, sort_keys=True) + "\n"
                )
                self.canary_receipt_relative = str(canary_path.relative_to(RESULTS))
                self.canary_receipt_hash = sha256(canary_path)
                self.prime_physical_eviction_metrics(label)
                launch_dir = CONTROL / "server_launches" / label
                launch_dir.mkdir(parents=True, exist_ok=True)
                launch_log = launch_dir / "startup_and_primer.log"
                result = self.run(["docker", "logs", SERVER_NAME], capture=True)
                launch_text = result.stdout + result.stderr
                launch_log.write_text(launch_text)
                lowered = launch_text.lower()
                fatal_terms = (
                    "out of memory",
                    "oom-kill",
                    "fatal",
                    "traceback",
                    "retracting request",
                )
                observed_fatal_terms = [term for term in fatal_terms if term in lowered]
                if observed_fatal_terms:
                    raise RuntimeError(
                        f"Qwen launch log contains fatal/retraction evidence: "
                        f"{observed_fatal_terms}"
                    )
                self.server_launch_log_relative = str(launch_log.relative_to(RESULTS))
                self.server_launch_log_hash = sha256(launch_log)
                self.log(
                    f"SERVER READY retention={retention} label={label} "
                    f"config={self.server_config_hash} cache=RadixCache "
                    "hybrid_swa=False canary=pass"
                )
                self.server_active = True
                return
            running = self.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", SERVER_NAME],
                check=False,
                capture=True,
            ).stdout.strip()
            if running != "true":
                result = self.run(
                    ["docker", "logs", SERVER_NAME], check=False, capture=True
                )
                logs = result.stdout + result.stderr
                raise RuntimeError(f"Qwen server exited: {logs[-8000:]}")
            time.sleep(5)
        raise RuntimeError("Qwen server did not become ready")

    def write_lfu_feasibility_receipt(self) -> None:
        """Freeze the excluded LFU runtime gate before calibration/evaluation."""
        if self.retention != "lfu":
            raise RuntimeError("LFU feasibility receipt requires an active LFU launch")
        gate = PACKET / "feasibility/lfu_feasibility_gate.json"
        if not gate.is_file():
            raise RuntimeError(f"missing frozen LFU feasibility gate: {gate}")
        launch_log = RESULTS / self.server_launch_log_relative
        log_text = launch_log.read_text(encoding="utf-8", errors="replace")
        if "impl=RadixCache" not in log_text or "hybrid_swa=False" not in log_text:
            raise RuntimeError("LFU feasibility log does not prove RadixCache topology")
        receipt = {
            "schema_version": 1,
            "status": "pass",
            "purpose": "excluded_pre_freeze_lfu_runtime_feasibility_gate",
            "completed_at": now(),
            "static_gate": {
                "path": "feasibility/lfu_feasibility_gate.json",
                "sha256": sha256(gate),
            },
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "snapshot": str(MODEL_SNAPSHOT),
                "container_snapshot": CONTAINER_SNAPSHOT,
            },
            "runtime": {"image": IMAGE, "image_id": IMAGE_ID},
            "retention": "lfu",
            "cache": {"implementation": "RadixCache", "hybrid_swa": False},
            "server": {
                "label": self.server_label,
                "argv": self.server_argv,
                "config_sha256": self.server_config_hash,
                "startup_log": self.server_launch_log_relative,
                "startup_log_sha256": self.server_launch_log_hash,
            },
            "functional_canary": {
                "path": self.canary_receipt_relative,
                "sha256": self.canary_receipt_hash,
            },
            "physical_eviction_primer": {
                "path": self.primer_receipt_relative,
                "sha256": self.primer_receipt_hash,
            },
            "runtime_error_scan": {
                "no_oom_retraction_fatal_or_traceback": True,
                "startup_log_scanned": self.server_launch_log_relative,
            },
            "excluded_from_calibration_smoke_evaluation": True,
            "evaluated_cells_allowed": True,
        }
        write_atomic_json(CONTROL / "lfu_feasibility_receipt.json", receipt)
        self.log(
            "LFU FEASIBILITY PASS excluded launch/canary/real-eviction gate "
            f"config={self.server_config_hash}"
        )

    def ensure_server(self, retention: str, label: str) -> None:
        if self.server_active and self.retention == retention:
            return
        self.launch_server(label, retention)

    def cell_dir(self, phase: str, seed: int, arm: str) -> Path:
        return RESULTS / phase / f"seed_{seed}" / arm

    def verify_cell(self, phase: str, seed: int, arm: str, check: bool = True) -> bool:
        result = self.run(
            [sys.executable, str(PACKET / "verify_extension.py"), "cell", phase, str(RESULTS), str(seed), arm],
            check=False,
            capture=True,
        )
        if check and result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.returncode == 0

    def run_cell(
        self,
        phase: str,
        seed: int,
        arm: str,
        retention: str,
        trace_relative: str,
        requests: int,
    ) -> None:
        if arm not in ARMS:
            raise RuntimeError(f"cell arm is outside frozen design: {arm}")
        expected_retention = "lfu" if arm == LFU_ARM else "lru"
        if (
            retention != expected_retention
            or not self.server_active
            or self.retention != retention
        ):
            raise RuntimeError(
                f"cell topology mismatch arm={arm} requested={retention} "
                f"active={self.retention} expected={expected_retention}"
            )
        if self.verify_cell(phase, seed, arm, check=False):
            self.log(f"CELL SKIP verified phase={phase} seed={seed} arm={arm}")
            return
        output = self.cell_dir(phase, seed, arm)
        if output.exists() and any(output.iterdir()):
            attempts = RESULTS / "_attempts"
            attempts.mkdir(parents=True, exist_ok=True)
            output.rename(attempts / f"{phase}_seed{seed}_{arm}_{int(time.time())}")
        output.mkdir(parents=True, exist_ok=True)
        trace = PACKET / trace_relative
        baseline = "sglang_radix_attention_shadowkv_plus" if arm.startswith("meritkv_") else "sglang_radix_attention"
        admission = (
            "native_sglang_hook"
            if arm in {
                "meritkv_skip_write_only_lru",
                "meritkv_joint_skip_write_skip_lookup_lru",
            }
            else "write_through_admission"
        )
        self.reset_server_cache_and_counters()
        started = now()
        environment = {
            "MERITKV_EFFECTIVE_ARM": arm,
            "MERITKV_RETENTION_POLICY": retention,
            "MERITKV_MAX_TOTAL_TOKENS": str(MAX_TOTAL_TOKENS),
            "MERITKV_PAGE_SIZE": str(PAGE_SIZE),
            "MERITKV_PROTOCOL_SHA256": self.protocol_hash(),
            "MERITKV_EVALUATION_FREEZE_SHA256": self.freeze_hash(),
            "MERITKV_INPUT_TRACE_SHA256": sha256(trace),
            "MERITKV_RUNTIME_IMAGE_ID": IMAGE_ID,
            "MERITKV_SERVER_CONFIG_SHA256": self.server_config_hash,
            "MERITKV_FROZEN_CONFIG_SHA256": self.protocol_hash(),
            "MERITKV_MODEL_ID": MODEL_ID,
            "MERITKV_TOKENIZER_SNAPSHOT": CONTAINER_SNAPSHOT,
            "MERITKV_TOKENIZER_MANIFEST_SHA256": TOKENIZER_MANIFEST,
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
        }
        command = ["docker", "run", "--rm", "--name", CLIENT_NAME, "--network", "host"]
        command += ["--device", "nvidia.com/gpu=all", "--ipc", "host"]
        for key, value in environment.items():
            command += ["-e", f"{key}={value}"]
        command += [
            "-v", f"{PROJECT}:/workspace/project:ro",
            "-v", f"{RESULTS}:{RESULTS}",
            "-v", "/datapool/cache/huggingface:/cache/huggingface",
            "-v", "/datapool/cache/huggingface/hub:/hf_hub:ro",
            "--entrypoint", "python3", IMAGE,
            "/workspace/project/meritkv_qwen25_capacity_pressure_20260823/run_capacity_cell.py",
            "--baseline", baseline, "--model", MODEL_ID,
            "--workload", "public_dataset", "--dataset", "samsum",
            "--prompt_mode", "templated", "--n_requests", str(requests),
            "--seed", str(seed), "--disable_arrival_simulation",
            "--trace_path", f"/workspace/project/meritkv_qwen25_capacity_pressure_20260823/{trace_relative}",
            "--output_dir", str(output), "--max_tokens", "1", "--temperature", "0",
            "--api_base", f"http://127.0.0.1:{SERVER_PORT}", "--request_endpoint", "chat",
            "--admission_preset", "balanced", "--admission_mode", admission,
            "--warmup_requests", "0", "--measure_energy", "--gpu_index", "0",
            "--idle_baseline_seconds", "2", "--idle_stabilization_seconds", "5",
            "--idle_stabilization_tolerance_w", "3",
        ]
        (RESULTS / phase).mkdir(parents=True, exist_ok=True)
        with (RESULTS / phase / "raw_commands.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": started, "argv": command}) + "\n")
        self.log(f"CELL START phase={phase} seed={seed} arm={arm} requests={requests}")
        result = self.run(command, capture=True, timeout=2400)
        (output / "client_stdout.json").write_text(result.stdout)
        server_log_result = self.run(["docker", "logs", SERVER_NAME], capture=True)
        server_log = server_log_result.stdout + server_log_result.stderr
        (output / "server_snapshot.log").write_text(server_log)
        receipt = {
            "schema_version": 1,
            "status": "pass",
            "phase": phase,
            "seed": seed,
            "arm": arm,
            "retention": retention,
            "trace_source": trace_relative,
            "trace_sha256": sha256(trace),
            "requests": requests,
            "max_total_tokens": MAX_TOTAL_TOKENS,
            "page_size": PAGE_SIZE,
            "runtime_image_id": IMAGE_ID,
            "server_command": " ".join(self.server_argv),
            "server_argv": self.server_argv,
            "server_config_sha256": self.server_config_hash,
            "cache_type": "RadixCache",
            "hybrid_swa_memory": False,
            "server_log_sha256": sha256(output / "server_snapshot.log"),
            "physical_eviction_primer": self.primer_receipt_relative,
            "physical_eviction_primer_sha256": self.primer_receipt_hash,
            "evaluation_freeze_sha256": self.freeze_hash(),
            "started_at": started,
            "completed_at": now(),
        }
        (output / "cell_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        self.verify_cell(phase, seed, arm)
        self.log(f"CELL PASS phase={phase} seed={seed} arm={arm}")

    def verify_phase(self, phase: str) -> None:
        result = self.run(
            [sys.executable, str(PACKET / "verify_extension.py"), "phase", phase, str(RESULTS)],
            capture=True,
        )
        (CONTROL / f"{phase}_verification.json").write_text(result.stdout)

    def experiment(self) -> None:
        freeze_path = CONTROL / "evaluation_freeze.json"
        lfu_receipt = CONTROL / "lfu_feasibility_receipt.json"
        if freeze_path.is_file():
            if not lfu_receipt.is_file():
                raise RuntimeError(
                    "evaluation freeze exists without its pre-freeze LFU feasibility receipt"
                )
            self.log("LFU FEASIBILITY SKIP existing pre-freeze receipt retained")
        else:
            self.launch_server("lfu_feasibility_preflight", "lfu")
            self.write_lfu_feasibility_receipt()
            self.stop_server()

        self.state(
            "running",
            "calibration",
            "separate excluded seed; single predeclared cache cap",
        )
        if self.verify_cell(
            "calibration", 32000, "native_admit_all_lru", check=False
        ):
            self.log(
                "CELL SKIP verified phase=calibration seed=32000 "
                "arm=native_admit_all_lru"
            )
        else:
            self.ensure_server("lru", "calibration_lru")
            self.run_cell(
                "calibration", 32000, "native_admit_all_lru", "lru",
                "traces/calibration_seed_32000.jsonl", 64,
            )
        self.verify_phase("calibration")
        freeze = self.run(
            [sys.executable, str(PACKET / "freeze_evaluation.py"), str(RESULTS), "--packet", str(PACKET)],
            capture=True,
        )
        (CONTROL / "freeze_command_receipt.json").write_text(freeze.stdout)
        smoke_plan = list(
            csv.DictReader((PACKET / "smoke_plan_5.tsv").open(), delimiter="\t")
        )
        if (
            len(smoke_plan) != 5
            or {row["arm"] for row in smoke_plan} != set(ARMS)
            or [int(row["order"]) for row in smoke_plan] != list(range(1, 6))
            or tuple(row["arm"] for row in smoke_plan) != ARMS
        ):
            raise RuntimeError("smoke plan is not the frozen five-arm Qwen design")
        for row in smoke_plan:
            retention = row["retention"]
            expected_retention = "lfu" if row["arm"] == LFU_ARM else "lru"
            if (
                retention != expected_retention
                or int(row["seed"]) != 32010
                or int(row["requests"]) != 36
                or row["trace"] != "traces/smoke_seed_32010.jsonl"
            ):
                raise RuntimeError(f"smoke arm/retention mismatch: {row}")
        for position, row in enumerate(smoke_plan, start=1):
            retention = row["retention"]
            if self.verify_cell(
                "smoke", int(row["seed"]), row["arm"], check=False
            ):
                self.log(
                    f"CELL SKIP verified phase=smoke seed={row['seed']} "
                    f"arm={row['arm']}"
                )
                continue
            self.ensure_server(
                retention, f"smoke_position_{position}_{retention}"
            )
            self.run_cell(
                "smoke", int(row["seed"]), row["arm"], retention,
                row["trace"], int(row["requests"]),
            )
        self.verify_phase("smoke")
        plan = list(
            csv.DictReader((PACKET / "evaluation_plan_25.tsv").open(), delimiter="\t")
        )
        if len(plan) != 25 or {
            (int(row["seed"]), row["arm"]) for row in plan
        } != {
            (seed, arm) for seed in range(32001, 32006) for arm in ARMS
        }:
            raise RuntimeError(
                "evaluation plan is not the frozen five-seed five-arm Qwen design"
            )
        positions_by_seed: dict[int, list[int]] = {}
        for row in plan:
            seed = int(row["seed"])
            positions_by_seed.setdefault(seed, []).append(int(row["order"]))
            retention = row["retention"]
            expected_retention = "lfu" if row["arm"] == LFU_ARM else "lru"
            if (
                retention != expected_retention
                or int(row["requests"]) != 124
                or row["trace"] != f"traces/evaluation_seed_{seed}.jsonl"
                or row.get("reset_before") != "flush_cache+action_counters"
            ):
                raise RuntimeError(f"evaluation arm/retention mismatch: {row}")
        if (
            set(positions_by_seed) != set(range(32001, 32006))
            or any(
                orders != list(range(1, 6))
                for orders in positions_by_seed.values()
            )
        ):
            raise RuntimeError(
                f"evaluation plan does not preserve chronological Latin positions: "
                f"{positions_by_seed}"
            )
        expected_latin = {
            seed: ARMS[offset:] + ARMS[:offset]
            for offset, seed in enumerate(range(32001, 32006))
        }
        observed_latin = {
            seed: tuple(
                row["arm"] for row in plan if int(row["seed"]) == seed
            )
            for seed in range(32001, 32006)
        }
        if observed_latin != expected_latin:
            raise RuntimeError(
                f"evaluation plan Latin chronology drift: {observed_latin}"
            )
        for chronological_position, row in enumerate(plan, start=1):
            seed = int(row["seed"])
            retention = row["retention"]
            if self.verify_cell(
                "evaluation", seed, row["arm"], check=False
            ):
                self.log(
                    f"CELL SKIP verified phase=evaluation seed={seed} "
                    f"arm={row['arm']}"
                )
                continue
            self.ensure_server(
                retention,
                f"evaluation_position_{chronological_position}_{retention}",
            )
            self.run_cell(
                "evaluation", seed, row["arm"], retention,
                row["trace"], int(row["requests"]),
            )
        self.stop_server()
        self.verify_phase("evaluation")
        full = self.run(
            [sys.executable, str(PACKET / "verify_extension.py"), "all", str(RESULTS), "--packet", str(PACKET)],
            capture=True,
        )
        (CONTROL / "full_verification.json").write_text(full.stdout)

    def restore(self) -> None:
        if self.restore_required and RECOVERY_STATE.is_file():
            state = json.loads(RECOVERY_STATE.read_text(encoding="utf-8"))
            self.update_recovery_state(
                phase="restore_running",
                restore_status="running",
                restore_attempts=int(state.get("restore_attempts", 0)) + 1,
                restore_attempted_at=now(),
            )
        self.stop_server()
        if not self.restore_required:
            self.resume_monitor()
            return
        destination = CONTROL / f"restore_{int(time.time())}"
        destination.mkdir(parents=True, exist_ok=True)
        if self.docker_inspect("{{.Id}}", self.prod_id) != self.prod_id:
            raise RuntimeError("exact retained production container no longer exists")
        if self.docker_inspect("{{.State.Running}}", self.prod_id) != "true":
            result = self.run(
                ["docker", "start", self.prod_id], capture=True, timeout=60
            )
            (destination / "docker_start.txt").write_text(result.stdout)
        else:
            (destination / "docker_start.txt").write_text(
                f"{self.prod_id} already running\n"
            )
        for _ in range(240):
            if self.production_healthy():
                break
            time.sleep(5)
        if not self.production_healthy():
            raise RuntimeError("production did not return healthy")
        if self.docker_inspect("{{.Id}}") != self.prod_id:
            raise RuntimeError("production container identity changed during restore")
        if self.docker_inspect("{{.Image}}") != self.prod_image_id:
            raise RuntimeError("production image identity changed during restore")
        actual_config = self.normalized_production_config(self.prod_id)
        actual_config_hash = json_hash(actual_config)
        if actual_config_hash != self.prod_config_hash:
            raise RuntimeError(
                "production normalized container config changed during restore: "
                f"{actual_config_hash} != {self.prod_config_hash}"
            )
        if sha256(PROD_DIR / "docker-compose.yml") != self.prod_compose_hash:
            raise RuntimeError("production compose changed during outage")
        probes = (
            (
                "direct_chat.json", "probe_openai_chat.py",
                ["--base-url", PROD_URL, "--model", "qwen38-27b-fp8", "--marker", "MERITKV_RESTORE_OK", "--timeout", "180"],
            ),
            (
                "direct_vision.json", "probe_openai_vision.py",
                ["--base-url", PROD_URL, "--model", "qwen38-27b-fp8", "--timeout", "180"],
            ),
            (
                "agentvm_relay_chat.json", "probe_openai_chat.py",
                ["--base-url", "http://192.168.122.1:8017", "--model", "qwen38-27b-fp8", "--marker", "MERITKV_RELAY_OK", "--timeout", "180"],
            ),
        )
        for name, script, arguments in probes:
            output = self.run(
                [sys.executable, str(PROD_DIR / script), *arguments],
                capture=True,
                timeout=240,
            ).stdout
            (destination / name).write_text(output)
        readiness = urllib.request.urlopen(
            "http://127.0.0.1:8001/health/readiness", timeout=30
        ).read().decode()
        (destination / "litellm_readiness.json").write_text(readiness)
        self.run(["systemctl", "--user", "is-active", "hermes-gateway.service"])
        self.assert_only_production_on_gpu()
        restored_started_at = self.docker_inspect("{{.State.StartedAt}}", self.prod_id)
        logs_result = self.run(
            ["docker", "logs", "--since", restored_started_at, self.prod_id],
            capture=True,
        )
        logs = logs_result.stdout + logs_result.stderr
        (destination / "recent.log").write_text(logs)
        lowered = logs.lower()
        if any(term in lowered for term in ("out of memory", "oom-kill", "fatal", "traceback")):
            raise RuntimeError("restored production log contains fatal/OOM evidence")
        self.assert_production_nvfp4_profile(actual_config, logs)
        self.resume_monitor()
        (destination / "restored_at.txt").write_text(now() + "\n")
        self.update_recovery_state(
            phase="restore_verified",
            restore_status="verified",
            restore_required=False,
            restored_at=now(),
            restore_evidence_dir=str(destination),
        )
        self.restore_required = False
        self.log(
            f"PRODUCTION RESTORED exact_container={self.prod_id} image={self.prod_image_id}"
        )

    def restore_only(self) -> None:
        if not RECOVERY_STATE.is_file():
            self.log("RESTORE-ONLY no recovery state; no action required")
            return
        payload = json.loads(RECOVERY_STATE.read_text(encoding="utf-8"))
        if payload.get("restore_required") is not True:
            self.log("RESTORE-ONLY recovery already verified; no action required")
            return
        production = payload.get("production") or {}
        experiment = payload.get("experiment") or {}
        monitor = payload.get("monitor") or {}
        expected = {
            "production_name": (production.get("name"), PROD_NAME),
            "experiment_server": (experiment.get("server_name"), SERVER_NAME),
            "experiment_client": (experiment.get("client_name"), CLIENT_NAME),
            "monitor_job": (monitor.get("job_id"), HEALTH_JOB_ID),
        }
        drift = {
            key: values for key, values in expected.items() if values[0] != values[1]
        }
        if drift:
            raise RuntimeError(f"recovery state target drift: {drift}")
        self.prod_id = str(production["container_id"])
        self.prod_image_id = str(production["image_id"])
        recorded_config_hash = str(production["normalized_config_sha256"])
        preimage_path = CONTROL / "production_preimage" / "normalized_container_config.json"
        preimage = json.loads(preimage_path.read_text(encoding="utf-8"))
        if json_hash(preimage) != recorded_config_hash:
            raise RuntimeError("recorded production-config preimage no longer matches recovery state")
        self.prod_config_hash = json_hash(normalize_config_record(preimage))
        self.prod_compose_hash = str(production["compose_sha256"])
        self.restore_required = True
        self.log(
            f"RESTORE-ONLY armed state found; restoring exact container {self.prod_id}"
        )
        self.restore()

    def completion_receipt(self) -> None:
        verification_path = CONTROL / "full_verification.json"
        verification = json.loads(verification_path.read_text())
        if verification.get("status") != "pass":
            raise RuntimeError("full verifier did not pass")
        if (
            verification.get("cells") != EXPECTED_CELLS
            or verification.get("requests") != EXPECTED_REQUESTS
        ):
            raise RuntimeError(
                "full verifier totals do not match the frozen Qwen design: "
                f"cells={verification.get('cells')} requests={verification.get('requests')}"
            )
        payload = {
            "schema_version": 1,
            "status": "complete_verified_restored",
            "experiment": "meritkv_qwen25_capacity_pressure_20260824",
            "completed_at": now(),
            "outage_started_at": self.outage_started_at,
            "production_restored": True,
            "production_container_id": self.prod_id,
            "production_image_id": self.prod_image_id,
            "cells": verification["cells"],
            "requests": verification["requests"],
            "verification_sha256": sha256(verification_path),
        }
        (CONTROL / "completion_receipt.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )

    def execute(self, action: str) -> None:
        if action == "restore-only":
            self.acquire_recovery_lock()
            self.restore_only()
            return
        self.acquire_lock()
        completed = CONTROL / "completion_receipt.json"
        if action in {"run", "resume"} and completed.is_file():
            receipt = json.loads(completed.read_text(encoding="utf-8"))
            verification = CONTROL / "full_verification.json"
            valid = (
                receipt.get("status") == "complete_verified_restored"
                and receipt.get("cells") == EXPECTED_CELLS
                and receipt.get("requests") == EXPECTED_REQUESTS
                and verification.is_file()
                and receipt.get("verification_sha256") == sha256(verification)
            )
            if not valid:
                raise RuntimeError("existing completion receipt is inconsistent")
            if not self.production_healthy():
                raise RuntimeError(
                    "completed experiment exists but production is not healthy"
                )
            self.log("RUN SKIP existing complete_verified_restored receipt verified")
            return
        self.preflight()
        if action == "preflight":
            self.state("ready", "preflight", "offline and image checks passed")
            return
        if datetime.now(ZoneInfo("America/New_York")) < OUTAGE_NOT_BEFORE:
            raise RuntimeError(f"outage guard: current time is before {OUTAGE_NOT_BEFORE}")
        failure: BaseException | None = None
        try:
            self.begin_outage()
            self.experiment()
        except BaseException as exc:
            failure = exc
        try:
            self.restore()
        except BaseException as restore_exc:
            self.state("restore_failed", "restore", str(restore_exc))
            raise
        if failure is not None:
            self.state("failed_restored", "closeout", str(failure))
            raise failure
        self.completion_receipt()
        self.state(
            "complete", "verified",
            "31 cells and 3344 requests verified; production restored",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["preflight", "run", "resume", "restore-only"],
        nargs="?",
        default="run",
    )
    args = parser.parse_args()
    controller = Controller()

    def terminate(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    controller.execute(args.action)


if __name__ == "__main__":
    main()
