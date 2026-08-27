# Admission Baselines and Mixed-Traffic Results

## 1. Admission-Control Baselines

All baselines share the same `TieredStateBank` + backend infrastructure;
only the decision rule differs. This isolates the value of each component in
the utility model $U = B - C - W$.

### 1.1 What Each Baseline Tests

| Baseline | Rule | What It Tests |
|----------|------|--------------|
| **No cache** | Never reuse | Floor — 1.000× speedup, 0 waste |
| **Gate: len≥k** | Reuse only if match length ≥ k | Is MeritKV just a cheap length cutoff? |
| **Gate: cost-only** | Reuse if B − C ≥ 0 (W = 0) | Does the waste term matter? |
| **Gate: freq≥3** | Admit after 3 observations | Common heuristic admission |
| **MeritKV** | Full U = B − C − W | Reference |
| **Offline oracle** | Future-aware optimal | Upper bound (97% achievable) |

### 1.2 T4 Results (NVIDIA T4, 5 models × 10 datasets × 3 seeds)

| Baseline | Speedup | Waste | HitRate | vs MeritKV |
|----------|:-------:|:-----:|:-------:|:----------:|
| No cache | 1.000× | 0.000 | 0.000 | −26.7% |
| Gate: len≥48 | 1.040× | 0.020 | 0.010 | −23.8% |
| Gate: len≥16 | 1.275× | 0.240 | 0.580 | −6.6% |
| Gate: len≥32 | 1.260× | 0.210 | 0.520 | −7.7% |
| Gate: strict | 1.285× | 0.220 | 0.550 | −5.9% |
| Gate: greedy | 1.221× | 0.310 | 0.620 | −10.5% |
| Gate: freq≥3 | 1.208× | 0.284 | 0.617 | −11.5% |
| Gate: cost-only (B−C) | 1.310× | 0.230 | 0.580 | −4.0% |
| ShadowKV (no waste) | 1.287× | 0.264 | 0.606 | −5.7% |
| **MeritKV** | **1.365×** | **0.156** | **0.402** | — |
| Offline oracle | 1.407× | 0.000 | 0.700 | +3.1% |

**Key finding:** MeritKV's waste term (W) accounts for a 5.7% speedup advantage
over ShadowKV and a 4.0% advantage over a pure cost-only gate, with 1.5–1.7×
less waste than either. Simple gates that ignore waste (len≥16, strict) achieve
intermediate speedup (1.275–1.285×) but at 1.5× the waste.

### 1.3 Blackwell Results (RTX PRO 6000, vLLM APC, Qwen2.5-32B)

Waste in this section is stored-but-unreused KV entries (cache query bytes
minus cache hit bytes). APC is a reactive exact-prefix cache with no
speculative precompute, so speculative waste is zero by construction.
This is a different quantity from the speculative waste reported in §1.2;
both are labelled "waste" in their respective contexts.

| Baseline | Speedup | Waste* | HitRate | vs APC+MeritKV |
|----------|:-------:|:------:|:-------:|:--------------:|
| No cache | 1.000× | — | 0.000 | −18.1% |
| APC only | 1.227× | 0.120 | 0.610 | +0.4% |
| Gate: len≥16 | 1.195× | 0.190 | 0.580 | −2.2% |
| Gate: len≥32 | 1.180× | 0.160 | 0.520 | −3.4% |
| Gate: greedy | 1.210× | 0.230 | 0.620 | −1.0% |
| Gate: freq≥3 | 1.200× | 0.200 | 0.617 | −1.8% |
| APC + MeritKV | **1.222×** | **0.068** | 0.402 | — |

\* Waste = stored-but-unreused KV entry bytes / total cached KV bytes.
APC has no speculative precompute, so this is a cache-efficiency metric,
not a compute-waste metric.

On Blackwell with vLLM APC, the write-through MeritKV overlay matches APC
within measurement noise (1.222× vs 1.227×, +0.4% overhead) while lowering
the controller-labelled unused-store fraction by 43% (0.120 → 0.068). This is
decision-quality evidence: native APC still stores through in both arms, so it
is not a realised storage saving. The latency parity is consistent with the
paper's runtime result (59.4ms vs 59.6ms, §7).

---

## 2. Mixed-Traffic Workloads

### 2.1 Workload Definitions

| Workload | Raw | Templated | RAG | Semantic | Description |
|----------|:---:|:---------:|:---:|:--------:|-------------|
| **Clean reusable** | 5% | 85% | 10% | — | 95% templated/RAG with long shared prefixes |
| **Raw-dominated** | 90% | 5% | 5% | — | 90% diverse raw prompts |
| **Chat-RAG mix** | 40% | 30% | 20% | 10% | Realistic chat + RAG + semantic |
| **Bursty reuse** | 20% | 60% | 20% | — | Repeated prefix in bursts, then disappears |
| **Adversarial short** | 25% | 50% | 25% | — | Many matches below breakeven (k < 16) |
| **Speculation trap** | 10% | 70% | 20% | — | High early frequency → no future reuse |

### 2.2 T4 Results (60 req mixed, 3 seeds)

| Workload | Metric | MeritKV | Greedy | ShadowKV | Cost-only | k≥16 | k≥32 |
|----------|--------|:-------:|:------:|:--------:|:---------:|:----:|:----:|
| **Clean reusable** | Speedup | **1.42×** | 1.34× | 1.28× | 1.35× | 1.22× | 1.21× |
| | Waste | **0.11** | 0.28 | 0.24 | 0.20 | 0.21 | 0.18 |
| **Raw-dominated** | Speedup | 1.05× | 1.32× | **1.35×** | 1.30× | 1.03× | 1.02× |
| | Waste | **0.01** | 0.32 | 0.28 | 0.25 | 0.10 | 0.08 |
| **Chat-RAG mix** | Speedup | 1.34× | **1.40×** | 1.30× | 1.33× | 1.24× | 1.20× |
| | Waste | **0.14** | 0.30 | 0.26 | 0.22 | 0.24 | 0.20 |
| **Bursty reuse** | Speedup | **1.38×** | 1.36× | 1.28× | 1.32× | 1.22× | 1.20× |
| | Waste | **0.12** | 0.28 | 0.24 | 0.21 | 0.21 | 0.18 |
| **Adversarial short** | Speedup | **1.24×** | 1.20× | 1.18× | 1.20× | 1.06× | 1.04× |
| | Waste | **0.16** | 0.32 | 0.28 | 0.25 | 0.26 | 0.22 |
| **Speculation trap** | Speedup | 1.12× | **1.30×** | 1.22× | 1.25× | 1.10× | 1.06× |
| | Waste | **0.08** | 0.38 | 0.30 | 0.26 | 0.24 | 0.20 |

**Clean reusable** (95% templated/RAG): MeritKV's highest advantage at 1.42×.
Bypass on the 5% raw requests avoids lookup overhead.

**Raw-dominated** (90% raw): MeritKV activates its conservative raw-mode gate,
keeping waste near zero (0.01) and speedup at 1.05×. Waste-unaware policies
(ShadowKV, greedy) achieve 1.32–1.35× but waste 0.28–0.32.

**Chat-RAG mix**: Greedy reaches 1.40× versus MeritKV's 1.34×, but incurs
2.1× the waste (0.30 versus 0.14).

**Bursty reuse**: MeritKV's 1.38× matches greedy's peak (1.36×) with 0.12 vs
0.28 waste.

**Adversarial short**: MeritKV's break-even guard (k<16) blocks the harmful
short-prefix matches that inflate waste for other policies (0.16 vs 0.26–0.32).

**Speculation trap**: After the mid-trace cutover, waste-unaware policies keep
speculating on stale prefixes. MeritKV's waste-adaptive EWMA detects the shift
and throttles back, keeping waste at 0.08 vs 0.30–0.38.

### 2.3 Blackwell Results (RTX PRO 6000, vLLM APC, Qwen2.5-7B)

Waste definition follows §1.3: stored-but-unreused KV entry bytes.
This comparison is write-through: both runtime arms continue to store through
native APC. The rows therefore measure controller-labelled decision quality
alongside APC, not end-to-end acceleration caused by enforced admission.

| Workload | APC only | APC+MeritKV | Δ Speedup | Δ Waste |
|----------|:--------:|:-----------:|:---------:|:-------:|
| Clean reusable | 1.22× / 0.09 | **1.22×** / **0.06** | 0.0% | −33% |
| Raw-dominated | 1.01× / 0.02 | 1.01× / **0.01** | 0.0% | −50% |
| Chat-RAG mix | 1.12× / 0.08 | **1.12×** / **0.05** | 0.0% | −38% |
| Bursty reuse | 1.18× / 0.10 | **1.18×** / **0.06** | 0.0% | −40% |
| Adversarial short | 1.06× / 0.14 | **1.07×** / **0.08** | +0.9% | −43% |
| Speculation trap | 1.10× / 0.16 | **1.11×** / **0.07** | +0.9% | −56% |

On Blackwell, the overlay differs by 0–0.9% in speedup (within measurement
noise, consistent with the paper's parity claim) while its controller labels
33–56% less KV as worth storing but ultimately unused. Adversarial short and
speculation trap show the largest decision-quality difference because MeritKV
would decline short-prefix and decaying-reuse admissions if enforcement were
enabled. In this write-through trace, native APC still stores them.

---

## 3. Key Takeaways

1. **The waste term matters.** MeritKV (U = B − C − W) beats ShadowKV by 5.7%
   on T4 and a pure cost-only gate (B − C) by 4.0%, with 1.5–1.7× less waste.

2. **Length-only gates are fragile.** The aggressive len≥48 gate falls to
   1.040×, while len≥16/32 remain slower than MeritKV and carry substantially
   higher waste.

3. **MeritKV handles mixed traffic.** MeritKV leads speedup in three of six
   workloads and has the lowest waste in all six, exposing the intended
   speed-versus-waste tradeoff rather than claiming a universal speed lead.

4. **Waste control is where MeritKV separates.** Speculation trap: 0.08 waste
   for MeritKV vs 0.30–0.38 for waste-unaware policies. Adversarial short:
   0.16 vs 0.26–0.32.

5. **On Blackwell/vLLM**, the write-through overlay matches APC speedup within
   noise and labels 33–56% less stored-but-unreused KV. This is decision-quality
   evidence, not a realised storage saving: native APC still stores through in
   both arms.
