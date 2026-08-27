from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


HOT_TEMPLATED_PREFIXES = [
    "System: You are a regulated support triage assistant. Use the policy excerpt, identify risk, and return a concise action.\nPolicy: verify identity before account changes; escalate suspicious access; avoid promising irreversible actions.\nCase:\n",
    "System: You are a deployment reviewer. Summarize risk, rollback readiness, and missing checks from the release note.\nRelease checklist: canary, health checks, data migration guard, rollback owner, and post-deploy review.\nRelease note:\n",
    "System: You are a claims analyst. Extract facts, classify the claim, list missing evidence, and recommend next action.\nClaim rubric: date, location, claimant statement, coverage, exclusions, fraud signals.\nClaim:\n",
    "System: You are a database incident reviewer. Produce a short incident record with trigger, blast radius, containment, remediation, and prevention.\nIncident context:\n",
]

HOT_RAG_PREFIXES = [
    "RAG context:\nDocument A: Cache admission should avoid storing one-off prefixes when reuse probability is low.\nDocument B: Reusable long prefixes can benefit from Radix lookup when the same prefix recurs.\nDocument C: Evaluation must separate HTTP latency from end-to-end controller latency.\nQuestion:\n",
    "RAG context:\nDocument A: GPU idle energy should be measured after power stabilizes.\nDocument B: Server-side counters are required because client counters can drift from runtime behavior.\nDocument C: Global cache flush is not per-request admission control.\nQuestion:\n",
    "RAG context:\nDocument A: A deterministic skip-lookup test warms a prefix and then proves cached tokens become zero.\nDocument B: A deterministic skip-write test sends a new prefix, repeats it, and expects a miss.\nQuestion:\n",
    "RAG context:\nDocument A: Native Radix caches reusable prefixes by default.\nDocument B: ShadowKV admission can choose to skip cache lookup or skip cache write for individual requests.\nQuestion:\n",
]

ONE_OFF_ROOTS = [
    "One-off pressure packet. The runtime should not persist this unique prefix unless future reuse is plausible. ",
    "Unique audit memo. This item is intentionally unlikely to recur and should pressure cache admission. ",
    "Transient customer note. The prefix shares a generic shell but the body is unique enough to avoid useful reuse. ",
    "Disposable retrieval context. This synthetic record is designed to create write pressure without repeat value. ",
]

SUFFIXES = [
    "Return the next action.",
    "Give a one-line triage result.",
    "Classify the risk.",
    "State whether this should be cached.",
    "Summarize the evidence.",
    "Identify the missing control.",
]


def _hot_prefixes(mode: str) -> list[str]:
    return HOT_RAG_PREFIXES if mode == "rag" else HOT_TEMPLATED_PREFIXES


def _hot_prompt(prefix: str, index: int, mode: str) -> str:
    if mode == "rag":
        return (
            f"{prefix}"
            f"How should admission handle repeated prefix family {index % 4} when request {index} arrives?\n"
            f"Case details: reusable family={index % 4}; variant={index % 13}; desired output is concise."
        )
    return (
        f"{prefix}"
        f"Reusable family {index % 4}; request variant {index % 13}; operational details include queue depth, "
        f"identity state, and rollback owner. {SUFFIXES[index % len(SUFFIXES)]}"
    )


def _one_off_prompt(index: int, mode: str) -> str:
    root = ONE_OFF_ROOTS[index % len(ONE_OFF_ROOTS)]
    unique_terms = " ".join(f"nonce_{index}_{j}" for j in range(36))
    if mode == "rag":
        return (
            "RAG context:\n"
            f"{root}"
            f"Record id {index}. Unique evidence: {unique_terms}.\n"
            "Question: should this transient record be stored for future prefix reuse?"
        )
    return (
        f"{root}"
        f"Record id {index}. Unique evidence: {unique_terms}. "
        "Instruction: classify the cache admission decision in one phrase."
    )


def build_trace(mode: str, n_requests: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    prefixes = _hot_prefixes(mode)
    hot_count = max(1, int(n_requests * 0.40))
    one_off_count = max(0, n_requests - hot_count)

    for i in range(hot_count):
        prefix = prefixes[i % len(prefixes)]
        rows.append(
            {
                "request_id": len(rows),
                "arrival_time": len(rows) * 0.001,
                "prompt": _hot_prompt(prefix, i, mode),
                "metadata": {
                    "source_workload": "admission_sensitive_trace",
                    "prompt_mode": mode,
                    "shared_prefix_text": prefix,
                    "shadowkv_trace_class": "hot_reusable_prefix",
                    "expected_shadowkv_behavior": "allow_after_initial_store",
                    "reuse_family": f"hot_{i % len(prefixes)}",
                },
            }
        )

    for i in range(one_off_count):
        rows.append(
            {
                "request_id": len(rows),
                "arrival_time": len(rows) * 0.001,
                "prompt": _one_off_prompt(i, mode),
                "metadata": {
                    "source_workload": "admission_sensitive_trace",
                    "prompt_mode": mode,
                    "shadowkv_trace_class": "one_off_cache_pressure",
                    "expected_shadowkv_behavior": "bypass_skip_write",
                    "reuse_family": f"one_off_{i}",
                },
            }
        )

    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["request_id"] = i
        row["arrival_time"] = i * 0.001
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an admission-sensitive ShadowKV validation trace.")
    parser.add_argument("--mode", choices=["templated", "rag"], required=True)
    parser.add_argument("--n-requests", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_trace(args.mode, args.n_requests, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        key = row["metadata"]["shadowkv_trace_class"]
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "mode": args.mode,
        "n_requests": len(rows),
        "seed": args.seed,
        "class_counts": counts,
        "output": str(out),
    }
    (out.with_suffix(out.suffix + ".summary.json")).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
