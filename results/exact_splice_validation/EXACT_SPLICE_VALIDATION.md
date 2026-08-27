# Exact-Prefix Splice Validation

## Scope

This artifact validates the custom Hugging Face KV-cache crop-and-splice path used by the MeritKV exact-prefix experiments. It is a correctness-focused follow-up, not a latency benchmark. The test compares three paths on the same token sequence:

1. full recomputation without a cache;
2. continuation from a clean native prefix cache; and
3. continuation from a cache produced by a longer source request and cropped at a token-identical prefix boundary.

All cached continuations pass an all-ones attention mask covering the complete logical sequence, explicit logical `position_ids`, and `cache_position` when the model API accepts it. The tested supported path is Hugging Face `DynamicCache`-compatible execution with SDPA attention. Greedy continuations are evaluated under the same teacher-forced context so that a single argmax change does not contaminate later comparisons.

The validation does **not** claim bitwise equality. Float16 kernel shape changes produce ordinary rounding differences even between native-cache continuation and full recomputation. Consequently, the raw reports keep three comparisons separate: native cache versus full recomputation, splice versus full recomputation, and splice versus native cache.

## Frozen environment

- GPU: NVIDIA Tesla T4
- PyTorch: 2.11.0+cu128
- Transformers: 5.10.2
- CUDA reported by PyTorch: 12.8
- Attention implementation: SDPA
- Greedy decoding
- Explicit attention mask and logical positions
- Absolute/relative tensor tolerance: `1e-3` for float16 and `1e-5` for float32

Exact checkpoint revisions:

| Model | Revision |
|---|---|
| GPT-2 | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| TinyLlama-1.1B-Chat-v1.0 | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Gemma-2B-IT | `96988410cbdaeb8d5093d1ebdc5a8fb563e02bad` |
| Phi-3-mini-4k-Instruct | `f39ac1d28e925b323eae81227eaba4464caced4e` |

## Experiments

### Random-token stress test

The first run uses four deterministic random sequences at each of 50% and 75% shared-prefix ratios. Each of the eight cases compares eight generation steps, giving 64 token comparisons per model. Inputs have length 128 and use float16.

| Model | Splice vs. native tokens | Native vs. full cases | Splice vs. full cases | Mean splice/native generation TV | Maximum splice/native generation TV |
|---|---:|---:|---:|---:|---:|
| GPT-2 | 62/64 | 6/8 | 6/8 | 0.007215 | 0.01553 |
| TinyLlama | 64/64 | 8/8 | 8/8 | 0.000593 | 0.002013 |
| Qwen2.5-1.5B | 64/64 | 8/8 | 8/8 | 0.000777 | 0.002857 |
| Gemma-2B | 64/64 | 8/8 | 8/8 | 0.000199 | 0.005687 |
| Phi-3-mini | 64/64 | 7/8 | 7/8 | 0.000723 | 0.003740 |
| **Total** | **318/320 (99.375%)** | **37/40** | **37/40** | — | — |

Phi-3's single case-level comparison against full recomputation is not a splice/native disagreement: the native-cache and splice paths select the same token on all 64 steps. GPT-2 has two splice/native argmax differences. In one, the splice agrees with full recomputation while the native-cache path does not; in the other, the native-cache path agrees while the splice does not. The splice-to-full total variation is lower than the native-to-full total variation in both cases.

### GPT-2 float32 control

The exact eight GPT-2 random-token cases were repeated in float32. Native-cache and splice continuations both match full recomputation in all eight cases, and splice/native decoded tokens match on all 64 generation steps. Four of eight cases also meet the deliberately tight tensor-level `allclose` rule. Across the 64 generation steps, mean splice/native total variation is `2.145e-6` and the maximum is `3.974e-6`.

This control supports the interpretation that the isolated GPT-2 float16 argmax changes are precision-sensitive numerical effects rather than incorrect mask or position handling.

### Natural-text test

The follow-up uses eight fixed English prompt pairs per model. Each pair has a token-identical natural-language prefix followed by two different suffixes. Tokenized lengths vary by tokenizer; shared prefixes cover approximately 77%–89% of each sequence. Every case compares 16 generation steps, giving 128 comparisons per model. The exact text and token IDs are embedded in every raw JSON report.

| Model | Splice vs. native tokens | Native vs. full tokens | Splice vs. full tokens | Mean splice/native generation TV | Maximum splice/native generation TV |
|---|---:|---:|---:|---:|---:|
| GPT-2 | 128/128 | 127/128 | 127/128 | 0.007174 | 0.03100 |
| TinyLlama | 127/128 | 128/128 | 127/128 | 0.000511 | 0.003058 |
| Qwen2.5-1.5B | 128/128 | 128/128 | 128/128 | 0.001741 | 0.007172 |
| Gemma-2B | 128/128 | 128/128 | 128/128 | 0.000713 | 0.004430 |
| Phi-3-mini | 127/128 | 127/128 | 128/128 | 0.002472 | 0.008593 |
| **Total** | **638/640 (99.6875%)** | **638/640 (99.6875%)** | **638/640 (99.6875%)** | — | — |

The three exceptional reference comparisons have distinct directions:

- GPT-2, `natural_05`, step 2: native and splice agree with each other (`1029`) and differ from full recomputation (`1165`).
- TinyLlama, `natural_00`, step 8: native and full recomputation agree (`1438`); splice selects `963`.
- Phi-3, `natural_05`, step 11: splice and full recomputation agree (`9133`); native selects `12027`.

Thus, two of 640 natural-text steps differ between splice and native cache. One favors the native-cache argmax and one favors the splice/full-recompute argmax. All compared logits are finite. There is no model-family-wide divergence, and the previously problematic Qwen2.5 path matches on all 128 natural-text steps and all 64 random-token steps.

## Interpretation

Across the two float16 suites, explicit-state splice and equivalent native-cache decoding agree on 956 of 960 evaluated steps (99.583%). Qwen2.5 and Gemma agree on every tested step; the remaining four isolated argmax differences are distributed across GPT-2, TinyLlama, and Phi-3 and occur within a setting where native cached execution itself is not bitwise identical to full recomputation. This direct-equivalence check answers a different question from the original crop-and-replay diagnostic, whose Qwen2.5 disagreement remains reported separately.

These results support a scoped claim: with the recorded cache API, mask, logical positions, precision, checkpoint revisions, and SDPA backend, the explicit-state splice closely tracks the equivalent native-cache path. They do not establish bitwise equivalence for every float16 kernel, attention implementation, model revision, prompt, or hardware platform. Eager attention is not certified by this artifact.

## Raw artifacts

The internal ShadowKV repository includes the per-model JSON reports and JSON environment, status, and summary metadata under `raw/`. The curated anonymous MeritKV repository intentionally publishes only this concise report; the raw internal artifacts can be added to the reviewer release if artifact policy permits.

The downloaded archives passed their embedded SHA-256 manifest checks. No scientific execution failures were recorded. The `strict_pass` field is a tensor-level `allclose` diagnostic and should not be interpreted as the token-level outcome; all float16 reports retain the raw differences rather than relaxing tolerances after observing the results.
