# Native SGLang Storage Admission on Blackwell: Qwen2.5-32B

## Scope

This frozen extension tests MeritKV storage admission on a second production model. It is separate from, and does not modify, the accepted August 20 native-enforcement package or the Gemma-4-31B storage-admission extension.

- Hardware: NVIDIA RTX PRO 6000 Blackwell (96 GB).
- Model: `Qwen/Qwen2.5-32B-Instruct` at pinned revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`.
- Runtime: SGLang 0.5.13, float16, Triton attention, PyTorch sampling, and ordinary native `RadixCache` with `hybrid_swa=False`.
- Workload: a frozen, constructed SAMSum-templated capacity-pressure trace with four recurring hot families and long one-off prefixes.
- Cache capacity: 16,384 tokens, page size 1.
- Evaluation: five paired seeds, 124 measured requests per arm and seed, greedy generation, and one output token.
- Tuning: an excluded LFU feasibility preflight and one excluded calibration cell preceded the frozen smoke and evaluation plans; no result-driven evaluation retuning was performed.

The five paired arms are native admit-all LRU, MeritKV write-through LRU, MeritKV skip-write-only LRU, MeritKV joint skip-write plus skip-lookup LRU, and native admit-all LFU. The first four arms share the same native LRU retention policy, while the fifth is a genuine frequency-aware native retention baseline.

## Completion and verification

The complete private package contains 31 measured cells and 3,344 measured requests: one excluded calibration cell, five smoke cells, and 25 evaluation cells. A separate excluded LFU feasibility preflight passed before the freeze. The bundled Linux/WSL verifier passes all 31 cells, confirms the frozen chronological plan and paired traces, and reports zero exact UTF-8/SHA-256 output mismatches.

For every evaluation seed, each MeritKV arm recorded 56 allow and 68 bypass decisions. Skip-write-only requested 68 write skips and the native Radix counter recorded exactly 68 completed write skips. The joint arm also requested 68 lookup skips, but its native completed skip-lookup counter remained zero on this trace; the measured distinction between the enforced arms must therefore be interpreted from the recorded physical actions, not from requested actions alone.

## Measured result

| Arm | Mean end-to-end latency | Mean-of-seed P95 | Mean GPU energy per cell | Mean evicted tokens | Post-pressure hot survival |
|---|---:|---:|---:|---:|---:|
| Native admit-all LRU | 341.66 ms | 516.59 ms | 25,130.52 J | 150,580.2 | 4/4 |
| MeritKV write-through LRU | 346.05 ms | 521.92 ms | 25,460.69 J | 150,580.2 | 4/4 |
| MeritKV skip-write-only LRU | 318.48 ms | 530.53 ms | 23,230.21 J | 0 | 4/4 |
| MeritKV joint skip-write/lookup LRU | 321.85 ms | 534.15 ms | 23,481.61 J | 0 | 4/4 |
| Native admit-all LFU | 305.51 ms | 516.68 ms | 22,287.86 J | 126,940.6 | 4/4 |

Paired per-seed ratios show that skip-write-only, relative to MeritKV write-through, reduced mean end-to-end latency by `7.97%` (95% CI `[7.50%, 8.43%]`) and GPU energy by `8.76%` (`[8.33%, 9.19%]`). It eliminated measured physical evictions, but P95 increased by `1.65%` (`[0.97%, 2.33%]`). Relative to native admit-all LRU, skip-write-only reduced mean latency by `6.78%` and energy by `7.56%`, again with a higher P95.

The native LFU arm was stronger on this constructed trace: compared with native LFU, skip-write-only had `4.25%` higher mean latency and `4.23%` higher GPU energy. LFU still evicted a mean of 126,940.6 tokens per cell, whereas skip-write-only recorded zero. Because every arm preserved and recovered all four hot families, hot-set recovery is not a differentiating outcome in this Qwen extension.

## Interpretation and claim boundary

The Qwen result confirms across a second model that physically declining low-value writes can improve mean latency and energy over the same LRU backend operated as admit-all or write-through under a frozen capacity-pressure trace. It also shows that the frequency-aware native LFU baseline can outperform MeritKV on this workload and that the storage-admission gain does not imply a tail-latency gain.

The supported claim is therefore narrow:

> On a frozen Qwen2.5-32B native-SGLang capacity-pressure trace, MeritKV skip-write storage admission eliminated measured physical evictions and reduced mean end-to-end latency by 7.97% and GPU energy by 8.76% relative to MeritKV write-through LRU across five paired seeds; native LFU was approximately 4.2% faster and lower-energy than MeritKV skip-write-only on the same trace.

This extension does not establish general production acceleration under ordinary workloads, a P95 improvement, or superiority over frequency-aware retention. It is separate from the Hugging Face custom-splice and approximate semantic-reuse diagnostics.

## Files

- `aggregate_results.csv`: five-seed arm aggregates and paired 95% confidence intervals relative to native admit-all LRU.
- `paired_seed_results.csv`: one row per evaluation seed and arm with measured latency, energy, eviction, survival, decision, and physical-action fields.
- `SCIENTIFIC_REPORT.json`: machine-readable report supplied with the verified evidence package.
- `PROVENANCE.md`: package hash, verification status, curation boundary, and code inventory.
- `code/`: the frozen trace generator, cell wrapper, plans, verifier, feasibility gate, instrumentation wrappers, and compact source closure used by the executed package.

The public derivative intentionally omits request-level traces, server logs, host restoration records, transfer archives, and operator-specific orchestration. The complete unmodified package is preserved in the private ShadowKV evidence repository.
