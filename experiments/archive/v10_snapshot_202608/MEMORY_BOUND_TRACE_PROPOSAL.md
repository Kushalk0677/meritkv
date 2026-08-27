# Memory-Bound Trace Proposal

## Goal

Earn the claim: *"MeritKV prevents real bad decisions that production systems
make under realistic mixed traffic with memory pressure."*

Current gap: the mixed-traffic workloads show MeritKV makes *different*
decisions, but not that the prevented decisions were *costly* (no eviction,
no reload latency, no queueing delay). This proposal closes that gap.

## Design

### Trace Structure

A single interleaved trace (200 requests total) that exceeds KV cache
capacity midway through, forcing evictions.

**Phase 1 — Fill (requests 1-80).**
80 templated requests with 200-token shared prefix + 50-token unique suffix.
All use the same system prompt template (90% overlap). This fills the cache
with hot reusable entries. The remaining 20 requests are raw one-offs that
probe the cache without extending hot entries.

**Phase 2 — Churn (requests 81-140).**
60 requests alternating between:
- 30 requests with a *new* shared prefix (different system prompt, zero
  overlap with Phase 1). These force eviction of Phase 1 hot entries
  because capacity is exhausted.
- 30 requests that *would have* hit Phase 1's hot entries — but those
  entries are now evicted. These are the victims.

**Phase 3 — Recovery (requests 141-200).**
60 requests repeating Phase 1's shared prefix (the same hot template).
This measures whether the system has recovered from phase-2 churn.

### Eviction Cost Measurement

| Metric | What it captures |
|--------|-----------------|
| Victim cache misses | Requests 81-140 in Phase 1's template that miss because entries were evicted |
| Reload bytes | KV bytes re-computed for victims that previously hit |
| P99 victim latency | Latency inflation on victim requests vs their clean-phase baseline |
| Cache churn rate | Total store + evict events per 10-request window |
| MeritKV declined admissions | Count of prefix admissions MeritKV declined vs APC baseline |

### Per-Decision Ledger

Run **APC only** vs **APC + MeritKV** on identical trace.
Instrument per request:

```
request_id, phase, template_id, was_hit, was_admitted,
  declined_by_MeritKV (bool), victim_of_eviction (bool),
  latency_ms, reload_bytes
```

Then aggregate:

```
Total admissions APC made: N
Total admissions MeritKV declined: M
Of those M, how many caused an eviction: E
Victim requests harmed by those E evictions: V
Total reload bytes from those evictions: R
P99 victim latency: L
```

This is the per-decision ledger the critique demanded.

### Hardware & Parameters

| Parameter | Value |
|-----------|-------|
| GPU | RTX PRO 6000 Blackwell (96 GB) |
| Model | Qwen2.5-7B-Instruct |
| Cache capacity | 80% of max (leave headroom for model weights) |
| Max KV cache tokens | ~66K (per vLLM, configurable via --gpu-memory-utilization) |
| Prefix length | 200 tokens shared / 50 unique |
| Batch size | 1 (sequential, no concurrent) |
| Request count | 200 |
| Seeds | 42, 123, 456 |

### Expected Output

| Metric | APC only | APC+MeritKV | Delta |
|--------|:--------:|:-----------:|:-----:|
| Victim cache misses | ~45 | ~15 | −67% |
| Reload bytes | ~2.5 GB | ~0.8 GB | −68% |
| P99 victim latency | 180 ms | 140 ms | −22% |
| Cache churn rate (peak) | 12/10 req | 5/10 req | −58% |
| Declined admissions | 0 (none) | ~35 | — |

If MeritKV declines ~35 admissions in phases 1-2, and those declinations
prevent ~30 evictions that would have caused ~45 victim misses with ~2.5 GB
of reload data and 22% P99 inflation on the victims, the strong claim is
supported with concrete numbers.

### Implementation

The trace can be built using `run_mixed_traffic.py` as a template, with the
added constraint of cache capacity. Key changes:

1. Fix KV cache capacity to 80% of max via vLLM server args
2. Phase 1: 80 requests, same `shared_prefix_text`, different `suffix`
3. Phase 2: new `shared_prefix_text` for 30 requests, old `shared_prefix_text`
   for 30 requests
4. Phase 3: same as Phase 1 to measure recovery

Instrumentation: modify the vLLM benchmark script to emit per-request
`victim_of_eviction` and `declined_by_MeritKV` flags.

### Time Estimate

| Step | Time |
|------|------|
| Write trace generator | 2 hours |
| Integration test on FakeBackend | 1 hour |
| One Blackwell run (200 req, 3 seeds) | ~30 min |
| Analysis + ledger construction | 2 hours |
| **Total** | **~1 day** |
