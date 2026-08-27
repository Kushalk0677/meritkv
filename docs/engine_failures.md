# Controlled-Run Failure Analysis

This is a secondary diagnostic of a 750-run subset, not the paper's aggregate
result table. The paper reports the ten worst controlled shared-process MeritKV
runs directly and treats the explanations below as observed mechanisms, not
architecture-level proofs.

## Observed Failures

The subset contains 17 runs with speedup below 1.0: 14 Qwen2.5-1.5B runs and
three Phi-3 runs. The worst observed row is Phi-3 on CNN/DailyMail in raw mode
at `0.895x` speedup and `1.000` waste. The remaining worst rows are dominated
by short-prefix Qwen semantic or templated cases.

## Phi-3 High-Waste Case

Phi-3 has the largest measured KV footprint in the primary study
(`kappa = 0.375 MB/token`). Its T4 transfer-only analytical breakeven is about
`9.6` tokens, not 449 tokens; 449K is a cache-capacity quantity from a different
model/hardware table and must not be used as a prefix breakeven.

The failed 48-token speculative precompute therefore was not structurally below
the transfer-only k*. It lost because the speculative work was not reused under
that request ordering. The memory-aware risk penalty assigns Phi-3 a larger
discount and reduces these high-waste admissions, but this diagnostic does not
claim a complete causal proof.

## Qwen Short-Prefix Cases

The observed Qwen failures have zero recorded speculative waste and matched
prefixes that are often only 8--12 tokens. With measured prefill cost
`beta = 0.741 ms/token` and reuse overhead `delta_r = 8.21 ms`, these cases sit
near the analytical boundary before suffix and planning costs. They illustrate
the paper's end-to-end qualification: a small transfer-only k* does not
guarantee a latency win for every model/backend/request combination.

The coupling penalty is modest for Qwen (`kappa = 0.0273 MB/token`) and is not
presented as a universal fix for short-prefix cases.

## Interpretation

| Observation | Paper-aligned interpretation |
|---|---|
| Phi-3 raw `0.895x`, waste `1.000` | An unused high-memory speculative precompute under one ordering. |
| Qwen semantic/templated rows below 1.0 | Short-prefix, near-boundary cases where fixed overhead can dominate. |
| Risk penalty blocks many Phi-3 admits | A measured admission effect, not proof of an architectural defect. |
| Ten worst-run table | Negative-result transparency; it does not replace aggregate statistics. |

These failures motivate utility gating and explicit waste reporting. They do not
support claims that any tokenizer or architecture is inherently unsuitable for
KV caching.
