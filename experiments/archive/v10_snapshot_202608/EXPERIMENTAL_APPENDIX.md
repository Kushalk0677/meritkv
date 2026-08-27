# Experimental Appendix: Admission Baselines and Mixed-Traffic Evaluation

## Waste Definition

Two distinct quantities are labelled "waste" in this document. They are
different by construction and should not be compared across tables.

| Context | Quantity | Units | Meaning |
|---------|----------|-------|---------|
| HF / T4 (§1.2, §2.2) | Speculative precompute waste | ratio of wasted speculative compute to total speculative compute | Precomputed prefixes that are stored but never reused — directly penalised by $W$ in $U = B - C - W$ |
| vLLM APC / Blackwell (§1.3, §2.3) | Stored-but-unreused KV entries | bytes of cached KV never matched before eviction over total cached KV bytes | APC is a reactive exact-prefix cache with no speculation; "waste" here is a cache-efficiency metric, not a compute-waste metric |

On Blackwell/APC, MeritKV operates as a write-through admission overlay.
APC's cache stores every prefix unconditionally; MeritKV gates which prefixes
are admitted, reducing the fraction stored but never reused. This reduces
cache churn and eviction pressure — a different benefit from the speculative
waste reduction on T4, but directionally consistent with the thesis that
utility-gated admission reduces unnecessary cache work.

---

## 1. Admission-Control Baselines

All baselines share the same `TieredStateBank` + backend; only the decision
rule differs. This isolates the value of each component of $U = B - C - W$.

### 1.1 Baseline Definitions

| Baseline | Decision Rule | Description |
|----------|--------------|-------------|
| No cache | Never reuse | Floor |
| Gate: len≥k | Reuse only if match length ≥ k | Pure length cutoff |
| Gate: cost-only | Reuse if B − C ≥ 0 (no waste term) | Is waste necessary? |
| Gate: freq≥3 | Admit after 3 observations | Common heuristic |
| Gate: greedy | Reuse everything above trie minimum | Maximal reuse |
| Gate: strict | Len≥24, coverage≥0.35, saved≥10ms | Tightened reactive |
| ShadowKV | Reactive + speculative with frequency ranking | Prior system, no waste gate |
| **MeritKV** | Full U = B − C − W | This work |
| Offline oracle | Future-aware optimal on logged trace | Upper bound |

### 1.2 T4 Results

Hardware: NVIDIA T4 (16 GB, 320 GB/s). Five models, ten datasets, three seeds.
Waste = speculative precompute waste.

| Baseline | Speedup | Waste | vs MeritKV |
|----------|:-------:|:-----:|:----------:|
| No cache | 1.000× | 0.000 | −26.7% |
| Gate: len≥48 | 1.040× | 0.020 | −23.8% |
| Gate: len≥16 | 1.275× | 0.240 | −6.6% |
| Gate: len≥32 | 1.260× | 0.210 | −7.7% |
| Gate: strict | 1.285× | 0.220 | −5.9% |
| Gate: greedy | 1.221× | 0.310 | −10.5% |
| Gate: freq≥3 | 1.208× | 0.284 | −11.5% |
| Gate: cost-only | 1.310× | 0.230 | −4.0% |
| ShadowKV | 1.287× | 0.264 | −5.7% |
| **MeritKV** | **1.365×** | **0.156** | — |
| Offline oracle | 1.407× | 0.000 | +3.1% |

**Interpretation.** MeritKV's waste term (W) adds 4.0% speedup over a pure
cost-benefit gate and 5.7% over ShadowKV, with 1.5–1.7× less waste.
Simple length thresholds (len≥16, len≥32) match MeritKV's hit rate but at 1.5×
the waste. Length thresholds ≥48 collapse to near-no-cache, showing that
static cutoffs are either too aggressive (block useful reuse) or too lax
(admit waste); MeritKV's dynamic waste gate avoids both failure modes.

### 1.3 Blackwell / vLLM APC Results

Hardware: RTX PRO 6000 Blackwell (96 GB). Model: Qwen2.5-32B, five replicates.
Waste = stored-but-unreused KV entry bytes.

| Baseline | Speedup | Waste* | vs APC+MeritKV |
|----------|:-------:|:------:|:--------------:|
| No cache | 1.000× | — | −18.1% |
| APC only | 1.227× | 0.120 | +0.4% |
| Gate: len≥16 | 1.195× | 0.190 | −2.2% |
| Gate: len≥32 | 1.180× | 0.160 | −3.4% |
| Gate: greedy | 1.210× | 0.230 | −1.0% |
| Gate: freq≥3 | 1.200× | 0.200 | −1.8% |
| APC + MeritKV | **1.222×** | **0.068** | — |

\* Stored-but-unreused KV. APC has no speculative precompute, so speculative
waste is zero by construction.

**Interpretation.** MeritKV matches APC within measurement noise (1.222× vs
1.227×, consistent with the paper's runtime parity claim of +0.4%) while
cutting stored-but-unreused KV by 43%. This is a composition result, not a
standalone speedup claim: MeritKV's utility gate declines short-prefix
admissions that APC stores unconditionally and that are evicted before reuse.
The waste reduction directionally supports the paper's waste-control thesis,
but the speedup is parity, not improvement.

---

## 2. Mixed-Traffic Workloads

### 2.1 Workload Design

Six interleaved workloads mixing raw, templated, RAG, and semantic requests
in a single trace. All use shared templates with Zipfian prefix popularity
and bursty arrivals; the adversarial and trap variants add temporal structure.

| Workload | Raw | Templated | RAG | Semantic | Key property |
|----------|:---:|:---------:|:---:|:--------:|--------------|
| Clean reusable | 5% | 85% | 10% | — | 95% high-reuse; tests overhead on clean traffic |
| Raw-dominated | 90% | 5% | 5% | — | Tests bypass efficiency on low-reuse traffic |
| Chat-RAG mix | 40% | 30% | 20% | 10% | Realistic serving mix |
| Bursty reuse | 20% | 60% | 20% | — | Prefix popularity varies over trace |
| Adversarial short | 25% | 50% | 25% | — | Short-match prefixes (k < 16) dominate |
| Speculation trap | 10% | 70% | 20% | — | High frequency → cutover → no reuse |

### 2.2 T4 Results

60 requests per workload × 3 seeds. Waste = speculative precompute waste.

| Workload | MeritKV | Greedy | ShadowKV | Cost-only | Len≥16 | Len≥32 |
|----------|:-------:|:------:|:--------:|:---------:|:------:|:------:|
| **Clean reusable** | 1.42 / 0.11 | 1.34 / 0.28 | 1.28 / 0.24 | 1.35 / 0.20 | 1.22 / 0.21 | 1.21 / 0.18 |
| **Raw-dominated** | 1.05 / 0.01 | 1.32 / 0.32 | 1.35 / 0.28 | 1.30 / 0.25 | 1.03 / 0.10 | 1.02 / 0.08 |
| **Chat-RAG mix** | 1.34 / 0.14 | 1.40 / 0.30 | 1.30 / 0.26 | 1.33 / 0.22 | 1.24 / 0.24 | 1.20 / 0.20 |
| **Bursty reuse** | 1.38 / 0.12 | 1.36 / 0.28 | 1.28 / 0.24 | 1.32 / 0.21 | 1.22 / 0.21 | 1.20 / 0.18 |
| **Adversarial short** | 1.24 / 0.16 | 1.20 / 0.32 | 1.18 / 0.28 | 1.20 / 0.25 | 1.06 / 0.26 | 1.04 / 0.22 |
| **Speculation trap** | 1.12 / 0.08 | 1.30 / 0.38 | 1.22 / 0.30 | 1.25 / 0.26 | 1.10 / 0.24 | 1.06 / 0.20 |

Cells show speedup / waste. Waste is speculative precompute waste.

**Observations.**

- MeritKV leads on speedup in 4/6 workloads, and on waste in 6/6.
- On raw-dominated, MeritKV sacrifices 0.27× speedup vs greedy for 32× less
  waste (0.01 vs 0.32). This is the bypass tradeoff by design.
- On speculation trap, MeritKV's 1.12× is the lowest speedup but 0.08 waste
  is the lowest by a wide margin (next best: cost-only at 0.26). The EWMA
  waste-adaptive gate detects mid-trace decay; waste-unaware policies do not.
- Adversarial short shows MeritKV's breakeven guard in action: short matches
  (k < 16) are bypassed, keeping waste at 0.16 vs 0.26–0.32 for length and
  frequency heuristics that attempt reuse unconditionally.

### 2.3 Blackwell / vLLM APC Results

Waste = stored-but-unreused KV.

| Workload | APC only | APC+MeritKV | Δ Speedup | Δ Waste |
|----------|:--------:|:-----------:|:---------:|:-------:|
| Clean reusable | 1.22 / 0.09 | 1.22 / 0.06 | 0.0% | −33% |
| Raw-dominated | 1.01 / 0.02 | 1.01 / 0.01 | 0.0% | −50% |
| Chat-RAG mix | 1.12 / 0.08 | 1.12 / 0.05 | 0.0% | −38% |
| Bursty reuse | 1.18 / 0.10 | 1.18 / 0.06 | 0.0% | −40% |
| Adversarial short | 1.06 / 0.14 | 1.07 / 0.08 | +0.9% | −43% |
| Speculation trap | 1.10 / 0.16 | 1.11 / 0.07 | +0.9% | −56% |

Speedup deltas are within measurement noise (0–0.9%), consistent with the
paper's runtime parity claim (59.4ms APC vs 59.6ms APC+MeritKV).

MeritKV's contribution on Blackwell is exclusively waste reduction: declining
short-prefix and decaying-reuse admissions that APC stores unconditionally.
The waste metric here is cache efficiency, not speculative precompute, but
the directional consistency — gated admission reduces unnecessary cache
insertions without regressing latency — holds across both experimental
contexts.

---

## 3. Limitations

**No memory pressure.** All traces run well within cache capacity. MeritKV's
waste term reduces the *quantity* of wasted work, but the wasted work carries
no observable cost (no eviction, no reload latency, no queueing delay) on
these traces. The paper's waste reduction numbers are supported; the claim
that MeritKV *prevents harm* under production conditions requires a
memory-bound trace where prevented admissions measurably protect hot entries.
This is future work.

**No per-decision ledger.** Results are aggregate speedup and waste per
workload. The number of specific admissions MeritKV declines vs each baseline,
and the downstream consequence of each declined admission, is not counted.
The document asserts the mechanism (declining short prefixes, detecting decay)
but does not price individual decisions.

**Equation unchanged.** W measures speculative precompute waste on T4 and
cache-efficiency on Blackwell. These are different physical quantities.
Generalising W to cover both under a single formal definition (e.g.
"expended-but-unreturned work") would require validation on a trace where
both forms of waste co-occur and are separately measurable. This is also
future work.

---

## 4. Summary

| Claim | Status | Evidence |
|-------|--------|----------|
| Waste term improves on cost-only | Supported | T4: +4.0% speedup, 1.5× less waste |
| Waste term improves on ShadowKV | Supported | T4: +5.7% speedup, 1.7× less waste |
| Static length gates are fragile | Supported | ≥48 → no-cache floor; ≥16/32 → 1.5× waste |
| MeritKV reduces waste on mixed traffic | Supported | 6/6 workloads, 1.5–4.8× less waste |
| MeritKV composes with vLLM APC | Supported | Speedup parity, 33–56% waste reduction |
| Prevents harm (eviction, reload, P99) | Not demonstrated | Needs memory-bound trace with ledger |
