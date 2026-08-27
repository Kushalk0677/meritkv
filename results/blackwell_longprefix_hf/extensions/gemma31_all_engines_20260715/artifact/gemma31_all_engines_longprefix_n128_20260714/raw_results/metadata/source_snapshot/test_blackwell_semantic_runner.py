import argparse
import importlib.util
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parents[1] / "blackwell_semantic_n128" / "run_blackwell_semantic_n128.py"
SPEC = importlib.util.spec_from_file_location("blackwell_semantic_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def test_build_cmd_propagates_include_experimental(tmp_path):
    args = argparse.Namespace(
        python="python",
        device="cuda",
        dtype="float16",
        n_requests=16,
        mean_inter_arrival_ms=50.0,
        max_arrival_sleep_ms=500.0,
        max_memory_mb=512,
        speculative_k=2,
        idle_threshold_ms=30.0,
        policy_preset="balanced",
        early_layer_reuse_ratio=0.35,
        logit_guard_threshold=0.08,
        gpu_index=0,
        idle_baseline_seconds=5.0,
        measure_energy=True,
        include_experimental=True,
        simulate_arrivals=True,
        allow_unsafe_semantic_kv_reuse=True,
        enable_policy_trace=True,
        semantic_index_diagnostics=False,
        trust_remote_code=False,
        config_path=None,
    )
    job = {
        "model": "google/gemma-4-31B-it",
        "dataset": "ag_news",
        "prompt_mode": "semantic",
        "seed": 42,
        "engine": "shadow_kv_plus_best_latency",
    }

    cmd = RUNNER.build_cmd(tmp_path, job, args)

    assert "--include_experimental" in cmd
    assert cmd[cmd.index("--engines") + 1] == "shadow_kv_plus_best_latency"
