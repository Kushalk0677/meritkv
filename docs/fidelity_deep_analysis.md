# KV Cache Reuse Fidelity — Deep Analysis

## 1. Qwen's Failure on GPU (float16)

### Observed Behavior

On GPU (T4, float16), Qwen 2.5 1.5B shows **0.20 ROUGE-L** between ref (clean generation) and reuse (with KV cache splice). This is near-random similarity, indicating the KV reuse produces completely different outputs.

On CPU (float32), earlier tests showed ~0.99 correspondence with divergence starting at token 7.

### Observed Precision Sensitivity in This Custom-Splice Path

The diagnostic shows a precision-sensitive difference, but it does not isolate
an architectural root cause or establish behavior of a native cache API.

**Potential contributor: float16 behavior in this DynamicCache splice**
- The KV cache stores attention key/value tensors. On GPU with float16, each cache entry has only ~3.3 decimal digits of precision (vs ~7.3 for float32).
- After `cache.crop(shared)` truncates the cache and new tokens are appended,
  the tested float16 path shows larger numerical differences than float32.
- The measurements are consistent with numerical amplification across the
  splice boundary, but they do not identify its cause or rule out position,
  mask, cache-API, or implementation effects.

**Observed Qwen2 trace**
- Layer-by-layer tracing on the tested path showed that hidden-state differences grow through the 28-layer stack:
  - Layer 1: diff ≈ 5e-7 (float32) / ≈ 1e-4 (float16)
  - Layer 10: diff ≈ 1e-5 (float32) / ≈ 1e-3 (float16)  
  - Layer 28: diff ≈ 2e-5 (float32) / ≈ 1e-2 (float16)
- In float32, the 2e-5 output diff only flips a token when the top-2 logits are exceptionally close.
- In float16, the 1e-2 output diff flips tokens **immediately** — often at the first or second generated token.

**Observed result**: In this GPU diagnostic, Qwen outputs often diverge from the
reference within the first generated tokens and have low aggregate ROUGE-L.

### Token-Level Divergence Pattern

```
Step  ref_token      reuse_token     Match
  0    come           come            ✓
  1    to             to              ✓  
  2    my             my              ✓
  3    house          house           ✓
  4    and            and             ✓
  5    we             we'll           ✗  (first divergence)
  6    can            play            ✗
  ...  (completely different after this point)
```

The first 4-5 tokens typically match (they're copied from the prompt suffix), but the first **generated** token often differs due to the accumulated float16 error in the hidden state.

The experiment does not determine why TinyLlama, Gemma, and Phi-3 have higher
agreement. Differences in model implementation, cache handling, position
metadata, precision, or other configuration details remain possible.

---

## 2. No-Reuse Control (ratio=0.0)

### What It Is

A control experiment where `shared_ratio = 0.0` — meaning **100% of the prompt is shuffled**, leaving zero shared prefix tokens between the original and modified prompts.

```python
shared = 0  # No shared tokens
cache.crop(0)  # Crop cache to empty
suffix = prompt_ids_mod[:, 0:]  # Prefill the ENTIRE modified prompt
```

With `shared = 0`, the cache crop removes everything. The suffix is the full modified prompt. Prefilling it on an **empty cache** is identical to a normal full prefill.

### Expected Result

**ROUGE-L = 1.0, 100% exact match**

Because no cache is actually reused — the operation degenerates to a clean generation from the modified prompt. Both `ref_text` and `reuse_text` are generated from the same prompt with the same `model.generate()` path (or the manual loop).

### Why Include It

1. **Validity check**: Confirms the pipeline doesn't produce false positives. If ratio=0.0 shows < 1.0, there's a bug in the code.
2. **Lower bound**: Demonstrates that when no reuse occurs, the output is identical.
3. **Paper narrative**: Shows that the engine only reuses when there's actual token overlap — for zero-overlap semantic matches, it falls through to a full prefill.

### Current Results (from local CPU tests)

| Model | Ratio=0.0 | Ratio=0.75 |
|-------|-----------|------------|
| TinyLlama | 1.0 | 1.0 |
| Qwen (CPU) | 1.0 | ~0.99 |

On CPU (float32), ratio=0.0 shows 1.0. This checks the no-splice control path;
it does not validate a nonempty splice.

---

## 3. Float32 vs Float16 Comparison

### Why It Matters

The paper's experiments are typically run on GPU (float16) for speed, but the numerical behavior differs from CPU (float32). Comparing both precisions reveals whether the fidelity measurement is **precision-dependent**.

### Expected Differences

| Aspect | float32 (CPU) | float16 (GPU) |
|--------|--------------|--------------|
| Precision | ~7 decimal digits | ~3 decimal digits |
| KV cache error growth | ~2e-5 across 28 layers | ~1e-2 across 28 layers |
| Token divergence onset | Token 7+ | Token 1-2 |
| TinyLlama fidelity | 1.0 | 0.966 |
| Qwen fidelity | ~0.99 | 0.20 |

### What This Means for the Paper

1. **The tested Qwen splice is precision-sensitive** — its float16 agreement is
   substantially lower than its float32 agreement.
2. **TinyLlama/Gemma are higher in this check** — their observed float16
   agreement is about 0.96--0.97.
3. **Recommendation**: Validate KV reuse fidelity in the deployment precision (typically float16 on GPU)
4. **Practical implication**: If using Qwen2-family models in float16, the `_partial_semantic_reuse` should use a conservative divergence threshold

### Recommended Experiment

Run TinyLlama at both precisions with identical prompts:

| Precision | Ratio | Exact Match | ROUGE-L | Notes |
|-----------|-------|-------------|---------|-------|
| float32 | 0.75 | 100% | 1.0 | CPU — already done |
| float16 | 0.75 | ~97% | ~0.97 | GPU — from Colab results |

The ~3% drop from float32 to float16 is the **precision cost** — acceptable for most applications but worth documenting.
