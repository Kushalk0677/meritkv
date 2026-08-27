# Cross-Model Analysis Review

**Reviewer:** worker-review  
**Date:** 2026-06-28  
**Scope:** `build_estimates.py`, `full_10dataset_5model_estimate.csv`, `engine_comparison_summary.csv`, `dataset_difficulty_factors.csv`

---

## 1. What Was Tested

- Full pipeline script (`build_estimates.py`, 966 lines)
- All three output CSVs (260-row full table, 30-row engine summary, 62-row difficulty factors)
- Methodology: scaling-law curve fitting (3-parameter power law), dataset imputation via token-length-adjusted analogy, validation cross-checks
- Spot-checked: ~20 estimated cells against their source values and token-ratio arithmetic

---

## 2. What's Good

### Pipeline structure
- Clean separation of loading, aggregation, fitting, imputation, and output generation.
- Scalable design that could easily add more model sizes or datasets.

### Curve-fitting strategy
- 3-parameter power-law (`a*x^b + c`) is a sensible physical model for latency vs. model parameters.
- Fallback to 2-parameter power law when R² < 0.5 is reasonable.
- Bounds on exponent `b ∈ [-5, 5]` prevent catastrophic fits without being overly restrictive.
- Fit quality is reported per (engine, dataset, mode, metric) — good observability.

### Dataset analogy mapping
- `banking77→ag_news` (classification), `alpaca_eval→dolly` (instruction), `oasst1→samsum` (chat), `ultrachat→daily_dialog` (chat), `cnn_dailymail→xsum` (summarization) are all well-motivated.

### Validation section
- 32B extrapolation cross-check against actual SGLang+LMCache measurements is included.
- Top-5 deviation reporting helps identify problematic fits.
- Full 5×10 speedup table printed with measured/estimated markers.

### Data integrity
- Estimated cells are marked with `method="estimated"` and `source_dataset` column.
- Replicate aggregation correctly computes means, stds, and 95% CIs.
- Speedups are computed vs. a proper LMCache baseline, not vs. the native radix attention baseline.

---

## 3. Issues Found

### Issue A (Critical): Difficulty factor computation mixes 32B data unevenly

**Location:** `build_estimates.py` lines 718–746

**Problem:** The difficulty factor computation averages over **all** rows matching (engine, mode, dataset). For the reference dataset `ag_news`, this includes 5 model sizes (1.5B, 3B, 7B, 14B, **and the 32B SGLang data**). For missing datasets like `banking77`, only 4 model sizes are available (32B has no banking77 data). The 32B ag_news latency (~92 ms) is much higher than the 1.5B–14B values (~10–43 ms), inflating the ag_news average by ~60%.

**Impact:** This creates artificially low difficulty multipliers for all missing datasets when compared to ag_news. For example:
- `banking77` shows `difficulty_multiplier=0.55` for `sglang_radix_attention, rag`
- The token-ratio-only expectation would be ~0.88 (150/180 tokens)^0.7
- The discrepancy (0.55 vs 0.88) is almost entirely an artifact of including 32B in the ag_news denominator

**Fix:** Either (a) exclude 32B data from difficulty factor computation, or (b) compute difficulty factors only on the subset of model sizes that exist for **both** the target and reference dataset.

### Issue B (Moderate): 32B data miscounted in summary

**Location:** `build_estimates.py` lines 681–684

**Problem:** The engine comparison summary counts:
```python
n_measured = (grp["method"] == "measured").sum()
n_estimated = (grp["method"] == "estimated").sum()
```

But 32B SGLang data has `method="measured_32b_sglang"`, which matches **neither** condition. The output CSV therefore shows `n_measured_datasets=0, n_estimated_datasets=0` for all 32B rows, even though those rows contain real measured data that contributes to the means.

**Impact:** Users reading the summary would conclude 32B has zero data and the numbers are pure extrapolation. In reality, 32B has 5 measured datasets for `sglang_radix_attention` and `lmcache_no_native_radix`. The summary is misleading.

**Fix:** Count `"measured_32b_sglang"` as measured, or rename the method tag to `"measured"` and distinguish by `model_slug`.

### Issue C (Moderate): Dead code — `impute_missing_dataset` is never used

**Location:** `build_estimates.py` lines 349–415 (function definition) and lines 535–540 (call site)

**Problem:** The function `impute_missing_dataset` is defined with a sophisticated interface (fits a scaling curve per source dataset, then computes difficulty), but it is called in a loop whose return value is **discarded**:

```python
for metric_key, metric_col in metric_means.items():
    imputed = impute_missing_dataset(...)
    if imputed is None:
        continue
# imputed is never used
```

The actual imputation happens in the subsequent inline code block (lines 542–577). The function `impute_missing_dataset` is effectively dead code.

**Impact:** Code maintenance risk — anyone reading the code will wonder why the function is defined and called but not used. Also suggests the imputation methodology may have evolved without removing the old version.

**Fix:** Either remove the dead function, or refactor the main pipeline to use it.

### Issue D (Moderate): 4-point, 3-parameter fits are statistically fragile

**Location:** `build_estimates.py` lines 284–326

**Problem:** The 3-parameter power law `a*x^b + c` has 3 free parameters but is fitted on only 4 data points (1.5B, 3B, 7B, 14B). This leaves only **1 degree of freedom**. The fit is highly sensitive to any single outlier.

**Observation from the validation output:** The code reports that 32B extrapolation errors are ~20–30% — this is consistent with overfitting to 4 points and then extrapolating 2× beyond the training range.

**Impact:** The scaling curves may not generalize well. The ~20–30% systematic overprediction at 32B is noted in codebase.md as "expected" but could be reduced with a simpler model (e.g., 2-parameter power law) or Bayesian priors.

**Fix:** Consider using the 2-parameter power law as the primary model and reserving 3-parameter for cases with ≥5 data points. Alternatively, report AIC/BIC to justify model complexity.

### Issue E (Moderate): Token-length scaling exponent (0.7) is unvalidated

**Location:** `build_estimates.py` lines 549, 556

**Problem:** The imputation uses `token_ratio ** 0.7` for latency/energy, `sqrt(token_ratio)` for cached tokens, and `1/lat_factor` for throughput. The 0.7 exponent is stated as "empirically" chosen but no citation or sensitivity analysis is provided.

**Impact:** If the true exponent is 0.5 or 1.0, the estimated values for missing datasets could be systematically off by 10–20%. This uncertainty is not propagated into any confidence intervals.

**Fix:** Add a comment justifying the exponent choice, or add a sensitivity analysis that shows how results change for exponents in [0.5, 1.0]. At minimum, flag this as a source of uncertainty in the output documentation.

### Issue F (Minor): Speedup trend partially inverted at 14B

**Observation from `engine_comparison_summary.csv`:**

| Model | ShadowKV++ speedup vs LMCache (rag) |
|-------|--------------------------------------|
| 1.5B  | 7.17% |
| 3B    | 8.84% |
| 7B    | **16.15%** |
| 14B   | **13.23%** |

The trend increases from 1.5B→7B (as expected — larger models benefit more from cache-aware scheduling), then **decreases** at 14B. While this could be a real effect (different hardware characteristics, different batch dynamics), it warrants investigation:

- Is the 14B ShadowKV++ data showing anomalous LMCache latency?
- The 14B LMCache latency for ag_news/templated (46.55 ms) is actually _lower_ than the 7B LMCache latency for xsum/templated (34.95 ms) adjusted for model size — this seems reasonable.
- But the speedup at 14B should be >= 7B if the technique works better on larger models.

**Recommendation:** Investigate whether the 14B speedup dip is a measurement artifact or a real physical effect. If real, document the reason.

### Issue G (Minor): No 32B ShadowKV++ data exists

**Impact:** The 32B analysis is limited to `sglang_radix_attention` and `lmcache_no_native_radix`. ShadowKV++ at 32B is entirely extrapolated from smaller models. The cross-check validation only applies to two of three engines. Users should be warned that the 32B ShadowKV++ speedup numbers are purely speculative.

---

## 4. Spot-Check Results

### Arithmetical verification

Checked 5 estimated cells against their source data and token-ratio formula:

| Target | Source | Metric | Source Val | Ratio | Factor | Expected | Actual | Match? |
|--------|--------|--------|-----------|-------|--------|---------|--------|--------|
| banking77, 15B, sglang_radix, rag | ag_news | latency | 10.71 | 150/180=0.833 | 0.833^0.7=0.88 | 9.43 | 9.43 | ✓ |
| cnn_dailymail, 7B, lmcache, rag | xsum | latency | 35.48 | 400/440=0.909 | 0.909^0.7=0.936 | 33.19 | 33.19 | ✓ |
| ultrachat, 14B, shadowkv++, templated | daily_dialog | latency | 43.05 | 350/210=1.667 | 1.667^0.7=1.42 | 61.55 | 61.55 | ✓ |
| alpaca_eval, 3B, sglang_radix, rag | dolly | cached_tokens | 142.45 | 250/200=1.25 | sqrt(1.25)=1.118 | 159.27 | 159.27 | ✓ |
| oasst1, 1.5B, lmcache, rag | samsum | cached_tokens | 4.33 | 300/280=1.071 | sqrt(1.071)=1.035 | 4.49 | 4.49 | ✓ |

All spot-checked estimated values are arithmetically correct.

### Reasonableness of estimated values

- **banking77** (150 tokens, classification): Estimated latency ~88% of ag_news. Makes sense — shorter prompts → lower prefill latency.
- **ultrachat** (350 tokens, chat): Estimated latency ~142% of daily_dialog (210 tokens). Slightly aggressive given similarly structured data, but not unreasonable.
- **cnn_dailymail** (400 tokens, summarization): Estimated latency ~94% of xsum (440 tokens). Reasonable — similar task type and comparable length.

---

## 5. Decision: "Best option"

**Ready to use with caveats — recommended fixes before publication.**

The analysis is **useful in its current form** for understanding cross-model trends, especially for the measured datasets (ag_news, daily_dialog, dolly, samsum, xsum) across 1.5B–14B. The estimated values for missing datasets are reasonable first-order approximations.

### Must-fix before publication
1. **Issue A** (Difficulty factor 32B mixing) — creates misleading difficulty multipliers
2. **Issue B** (32B data miscount) — misrepresents the summary to readers

### Should-fix
3. **Issue C** (Dead code) — maintenance hazard
4. **Issue D** (4-point fits) — add AIC/BIC or prefer 2-parameter model
5. **Issue E** (0.7 exponent) — add justification or sensitivity note

### Document as limitations
6. **Issue F** (14B speedup dip) — investigate or explain
7. **Issue G** (No 32B ShadowKV++ data) — clearly label as extrapolation-only

### Verdict

> **Best option: Fix issues A and B, then the analysis is ready.** The core methodology is sound, the code is well-structured, and the outputs are internally consistent. The main problems are in the summary/difficulty post-processing, not in the primary estimation pipeline. With those fixes, the analysis can be used with appropriate caveats about the untested 0.7 exponent and the lack of 32B ShadowKV++ validation data.
