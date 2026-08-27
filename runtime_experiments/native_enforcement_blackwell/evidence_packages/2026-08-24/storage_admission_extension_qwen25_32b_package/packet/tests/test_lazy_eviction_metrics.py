#!/usr/bin/env python3
"""Offline regression for SGLang's lazy physical-eviction metric registration."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


os.environ["MERITKV_EFFECTIVE_ARM"] = "native_admit_all_lru"
os.environ["MERITKV_RETENTION_POLICY"] = "lru"
HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "run_capacity_cell.py"
spec = importlib.util.spec_from_file_location("capacity_cell", MODULE_PATH)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeClient:
    def __init__(self, metrics: str):
        self.metrics = metrics

    def get_json(self, path: str, timeout_s: float = 10.0):
        assert path == "/shadowkv_admission_metrics"
        return {"counters": {}}

    def get_text(self, path: str, timeout_s: float = 10.0):
        assert path == "/metrics"
        return self.metrics


before_text = """
# HELP sglang:kv_evictable_tokens Number of evictable KV slots.
sglang:kv_evictable_tokens 0
# HELP sglang:swa_evictable_tokens Number of evictable SWA slots.
sglang:swa_evictable_tokens 0
"""
after_text = before_text + """
# HELP sglang:evicted_tokens_total Device KV slots physically freed.
sglang:evicted_tokens_total{cache_type="RadixCache"} 42
# HELP sglang:eviction_duration_seconds Physical eviction duration.
sglang:eviction_duration_seconds_count{cache_type="RadixCache"} 1
sglang:eviction_duration_seconds_sum{cache_type="RadixCache"} 0.01
"""

before = module.collect_capacity_metrics(FakeClient(before_text))
after = module.collect_capacity_metrics(FakeClient(after_text))
required = {
    "sglang:evicted_tokens_total",
    "sglang:eviction_duration_seconds",
}
assert before["available"] is True
assert before["prometheus_lazy_registration_observed"] is True
assert set(before["prometheus_capacity_metric_definitions_missing"]) == required
assert not before["prometheus_observed_samples"]
assert not any("evicted_tokens_total" in key for key in before["values"])
assert after["available"] is True
assert after["prometheus_lazy_registration_observed"] is False
assert not after["prometheus_capacity_metric_definitions_missing"]
assert {
    sample.split("{", 1)[0] for sample in after["prometheus_observed_samples"]
} >= {
    "sglang:evicted_tokens_total",
    "sglang:eviction_duration_seconds_count",
}

delta = module.adapter.diff_counter_metrics(before, after)
assert delta["available"] is True
assert any(
    key.startswith("prometheus::sglang:evicted_tokens_total") and value == 42.0
    for key, value in delta["delta"].items()
)
assert any(
    key.startswith("prometheus::sglang:eviction_duration_seconds_count")
    and value == 1.0
    for key, value in delta["delta"].items()
)
print("PASS lazy physical-eviction metric registration")
