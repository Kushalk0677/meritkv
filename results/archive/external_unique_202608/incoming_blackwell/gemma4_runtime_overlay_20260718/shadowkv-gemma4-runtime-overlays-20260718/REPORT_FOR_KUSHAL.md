# Gemma 4 Runtime Matrix With ShadowKV++ Overlays

## Scope

This corrected Blackwell matrix compares each native cache runtime with its
ShadowKV++ policy-overlay arm:

- Models: Gemma 4 E2B, E4B, 12B, 26B-A4B, and 31B
- Workloads: daily_dialog, samsum, ag_news, dolly, and xsum
- Modes: templated and RAG
- Arms: vLLM APC, SGLang RadixAttention, and LMCache, each with and without
  ShadowKV++
- Full cells: 300 (5 models x 5 datasets x 2 modes x 6 arms)
- Requests: 256 per cell, 76,800 total
- Energy: NVML, including idle stabilization before every cell
- Schedule: one randomized model/arm server-block order

The 30-cell smoke and 300-cell full matrix both completed without request or
reuse failures. Production `qwen36-27b-fp8-vllm` was restored and verified at
`127.0.0.1:8014` after the run.

## Paired Results

Each percentage below is the arithmetic mean of 50 matched
model/dataset/mode ratios. Negative latency, P95, and energy deltas favor the
ShadowKV++ arm; positive throughput favors it.

| Runtime | Mean latency | Mean P95 | Throughput | GPU energy | Latency wins | P95 wins |
|---|---:|---:|---:|---:|---:|---:|
| vLLM APC + ShadowKV++ | +2.61% | +3.19% | -2.08% | +0.03% | 19/50 | 16/50 |
| SGLang Radix + ShadowKV++ | -2.19% | -1.75% | +2.36% | -1.69% | 32/50 | 32/50 |
| LMCache + ShadowKV++ | +2.15% | +3.35% | -1.82% | +0.84% | 11/50 | 10/50 |

The SGLang overlay has the best aggregate paired result, but this is a
single-run result and not yet a causal ShadowKV++ performance claim. The
randomized schedule was not a replicated or position-balanced crossover.

## Model-Level Pattern

Mean paired latency deltas show that the direction is not consistent across
models:

| Model | vLLM APC overlay | SGLang overlay | LMCache overlay |
|---|---:|---:|---:|
| Gemma 4 E2B | -0.12% | +0.14% | +3.07% |
| Gemma 4 E4B | +5.75% | -3.52% | +8.13% |
| Gemma 4 12B | +12.10% | -3.52% | +2.08% |
| Gemma 4 26B-A4B | +0.19% | +0.91% | -6.21% |
| Gemma 4 31B | -4.87% | -4.94% | +3.65% |

This model dependence, combined with one run per cell, is consistent with a
mixture of observer overhead and execution-order/system noise rather than one
stable cross-runtime effect.

## Admission Behavior

All 150 overlay cells passed the controller gate:

- `admission_controller_enabled=true`
- 256 plans and 256 decisions per cell
- zero runtime-cache reset failures
- `admission_enforcement_mode=write_through_admission`

Each runtime's 50 overlay cells produced exactly 12,750 allows and 50
bypasses: 255 allows and one bypass per cell. The bypass was the initial
no-reuse request. After that, the balanced policy allowed every request.

Therefore, this workload did not create meaningful selective cache admission.
The overlay mostly measured policy planning/feedback overhead around the same
native cache behavior.

## Cache Evidence

- vLLM APC: 1,448,960 local hit tokens in native and overlay arms; positive in
  100/100 cells.
- SGLang RadixAttention: 1,629,265 cached tokens in native and overlay arms;
  positive in 100/100 cells.
- LMCache: 66,560 external hit tokens in native and overlay arms. All ten
  LMCache model/arm server logs contain both store and retrieve events.
- LMCache had zero external hits in the 20 AG News cells (ten native and ten
  overlay). Their reusable prefixes were below the configured 256-token
  external-cache chunk boundary. Those requests completed normally.

The identical cache-evidence totals within every native/overlay pair confirm
that write-through admission did not change runtime cache ownership or the
executed cache path.

## Interpretation Boundary

These ShadowKV++ arms are portable write-through policy observers. End-to-end
latency includes ShadowKV++ planning, the server request, and feedback. The
runtime still owns cache lookup and writes. A bypass records the policy
decision but does not natively suppress per-request lookup or write.

Accordingly:

- This matrix is valid evidence that the same external policy controller runs
  cleanly around all three cache systems with complete timing and energy data.
- It provides a preliminary SGLang parity/improvement signal.
- It does **not** show the benefit of enforced selective cache admission.
- A causal performance claim requires repeated, balanced native/overlay order
  and a native hook that enforces skip-lookup and skip-write on bypasses.

## Verification

- Full cells: 300/300
- Native/overlay pairs: 150/150
- Requests: 76,800/76,800
- Request failures: 0
- Reuse failures: 0
- Overlay reset failures: 0
- NVML energy: 300/300, no capture errors
- Idle stabilization timeouts: 0
- Smoke cells: 30/30
- Production restored: yes

See `analysis/ANOMALY_AUDIT.md`, `analysis/paired_summary.csv`,
`analysis/paired_summary_by_model.csv`, and
`analysis/paired_native_vs_shadowkv_plus.csv` for the auditable tables. Raw
JSON, server logs, source snapshot, runtime versions, image inspections, and
the randomized block plans are included.
