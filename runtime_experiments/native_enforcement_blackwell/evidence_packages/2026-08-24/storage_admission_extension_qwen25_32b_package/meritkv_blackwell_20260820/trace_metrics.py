#!/usr/bin/env python3
"""Pure helpers for binding final request timings into MeritKV traces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def percentile(values: Sequence[float], quantile: float) -> float:
    clean = sorted(float(value) for value in values)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(clean) - 1)
    fraction = position - low
    return clean[low] * (1.0 - fraction) + clean[high] * fraction


def synchronize_measured_timings(
    calls: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    results: Sequence[Any],
    *,
    require_decisions: bool,
) -> None:
    """Replace provisional HTTP timings with final planning-to-feedback timings."""

    measured_count = len(results)
    if len(calls) < measured_count:
        raise RuntimeError(
            f"measured result count {measured_count} exceeds call count {len(calls)}"
        )
    if require_decisions and len(decisions) != measured_count:
        raise RuntimeError(
            f"measured result count {measured_count} != decision count {len(decisions)}"
        )
    measured_calls = calls[-measured_count:] if measured_count else []
    for index, (call, result) in enumerate(zip(measured_calls, results, strict=True)):
        ttft_ms = float(getattr(result, "ttft_ms", result.latency_ms))
        final_e2e = getattr(result, "end_to_end_latency_ms", None)
        end_to_end_ms = float(result.latency_ms if final_e2e is None else final_e2e)
        call["ttft_ms"] = ttft_ms
        call["end_to_end_latency_ms"] = end_to_end_ms
        if index < len(decisions):
            decisions[index]["ttft_ms"] = ttft_ms
            decisions[index]["end_to_end_latency_ms"] = end_to_end_ms
