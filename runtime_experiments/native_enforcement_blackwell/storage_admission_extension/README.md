# Native SGLang Storage Admission on Blackwell

## Scope

This frozen extension tests whether MeritKV can improve a production cache backend by declining low-value **writes** during capacity pressure. It complements the earlier two-model native-enforcement matrix, whose sparse bypass rate established physical enforcement and near-parity rather than acceleration.

- Hardware: NVIDIA RTX PRO 6000 Blackwell (96 GB).
- Model: Gemma-4-31B-it at pinned revision `b9ea41a2887d8607f594846523f94c6cc75ac8a4`.
- Runtime: SGLang 0.5.13, float16, Triton attention, PyTorch sampling, and the native `SWARadixCache` LRU topology.
- Workload: a frozen, constructed SAMSum-templated capacity-pressure trace with four recurring hot families and long one-off prefixes.
- Cache capacity: 16,384 tokens.
- Evaluation: five paired seeds, 124 measured requests per arm and seed, greedy generation, and one output token.
- Tuning: one calibration seed preceded the evaluation freeze; no result-driven evaluation retuning was performed.

The four paired arms are native admit-all LRU, MeritKV write-through LRU, MeritKV skip-write-only LRU, and MeritKV joint skip-write plus skip-lookup LRU. All arms use the same native LRU retention policy, so the comparison isolates admission rather than replacing the eviction policy.

## Completion and verification

The complete private package contains 25 cells and 2,688 requests: one calibration cell, four smoke cells, and 20 evaluation cells. Its relocatable verifier passes all cells. The evaluation subset contains 2,480 requests.

For every evaluation seed, the controller produced 56 reuse and 68 bypass decisions in each MeritKV arm. The skip-write-only arm requested 68 write skips, and both the native Radix and Gemma SWARadix execution counters recorded exactly 68. The joint arm additionally requested 68 lookup skips, with both native counters again recording exactly 68. This is physical native-cache enforcement, not write-through logging.

## Measured result

| Arm | Mean end-to-end latency | Mean-of-seed P95 | Mean GPU energy per cell | Mean evicted tokens | Post-pressure hot survival |
|---|---:|---:|---:|---:|---:|
| Native admit-all LRU | 399.32 ms | 623.32 ms | 29,395.99 J | 305,722.6 | 3.2/4 |
| MeritKV write-through LRU | 399.43 ms | 623.84 ms | 29,406.22 J | 305,722.6 | 3.2/4 |
| MeritKV skip-write-only LRU | 368.45 ms | 628.08 ms | 26,884.78 J | 0 | 4/4 |
| MeritKV joint skip-write/lookup LRU | 367.44 ms | 626.59 ms | 26,765.55 J | 0 | 4/4 |

Paired per-seed ratios give the following changes relative to native admit-all LRU:

- Skip-write-only: mean latency `-7.73%` (95% CI `[-8.43%, -7.03%]`) and GPU energy `-8.54%` (`[-9.25%, -7.83%]`).
- Joint enforcement: mean latency `-7.98%` (`[-8.71%, -7.25%]`) and GPU energy `-8.95%` (`[-9.80%, -8.09%]`).
- The P95 changes are `+0.77%` and `+0.53%`, respectively, with confidence intervals crossing zero. This experiment therefore does not establish a tail-latency improvement.

The causal comparison between the two enforced arms is especially informative. Joint skip-write plus skip-lookup improves mean latency by only `0.27%` and energy by `0.44%` relative to skip-write alone. Most of the measured benefit therefore comes from storage admission: preventing one-off writes preserves the learned hot set and avoids subsequent recomputation. Declining already-resident cache hits is not the source of the approximately 8% result.

## Output comparison

The package reports zero exact UTF-8/SHA-256 output mismatches across corresponding arms. Generation was limited to one deterministic output token, so this is a narrow execution check rather than a universal multi-token or logit-equivalence result. It is also separate from the Hugging Face custom-splice and approximate semantic-reuse diagnostics.

## Claim boundary

The supported claim is narrow:

> On a frozen Gemma-4-31B native-SGLang capacity-pressure trace, MeritKV storage admission prevented measured cache evictions and reduced mean end-to-end latency by 7.7--8.0% and GPU energy by 8.5--9.0% relative to native admit-all LRU across five paired seeds.

This result does not establish general production acceleration under ordinary workloads, cross-model generality, a P95 improvement, or a native LFU result. The controller initially declines unseen hot-family bootstrap writes; repeated requests then establish the hot set, after which all four families survive the pressure phase.

## Frequency-aware baseline boundary

The working Gemma `SWARadixCache` topology uses LRU internally. A genuine native LFU comparison required ordinary `RadixCache`, which failed on the first Gemma-4 forward because of an incompatible KV-buffer layout. No LFU cell is reported. The separate controlled-bank LRU/LFU experiment remains the frequency-aware comparison; this extension isolates native storage admission against native admit-all LRU.

## Files

- `aggregate_results.csv`: five-seed arm aggregates and paired 95% confidence intervals versus native admit-all.
- `paired_seed_results.csv`: one row per evaluation seed and arm, including latency, energy, eviction, survival, decisions, and native-action counters.
- `PROVENANCE.md`: package hashes, verification status, curation boundary, and code inventory.
- `code/`: the frozen trace generator, cell wrapper, verifier, native SWARadix hook, protocol files, and the source import closure required by the wrapper.

The public package intentionally omits raw request traces, server logs, host configuration, transfer archives, and operator-specific paths. Those materials remain bound by the hashes in `PROVENANCE.md` in the private evidence repository.
