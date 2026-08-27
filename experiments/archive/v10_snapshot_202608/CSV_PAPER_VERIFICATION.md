# CSV-to-Paper Verification Report

## 1. Headline Table (summary_by_engine.csv)

| Engine | CSV Speedup | Paper Speedup | CSV Waste | Paper Waste | CSV Hit | Paper Hit | Verdict |
|--------|:-----------:|:-------------:|:---------:|:-----------:|:-------:|:---------:|:-------:|
| No cache | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | ✅ |
| Reactive | 1.214 | 1.214 | 0.000 | 0.000 | 0.317 | 0.317 | ✅ |
| Greedy | 1.221 | 1.221 | 0.000 | 0.000 | 0.320 | 0.320 | ✅ |
| Strict reactive | 1.254 | 1.254 | 0.000 | 0.000 | 0.310 | 0.310 | ✅ |
| Freq spec | 1.208 | 1.208 | 0.284 | 0.284 | 0.617 | 0.617 | ✅ |
| ShadowKV | 1.287 | 1.287 | 0.264 | 0.264 | 0.606 | 0.606 | ✅ |
| ShadowKV++ | **1.365** | **1.365** | **0.156** | **0.156** | **0.402** | **0.402** | ✅ |
| Best-latency | 1.331 | 1.331 | 0.228 | — | 0.500 | — | ✅ |
| Raw-observer | 1.356 | 1.356 | 0.158 | — | 0.404 | — | ✅ |

**All match within rounding to 3 decimal places.** Values are from 898 rows.

## 2. Per-Mode Table (summary_by_mode_engine.csv)

| Mode | Engine | CSV Speedup | Paper Speedup | CSV Waste | Paper Waste | CSV Hit | Paper Hit | Verdict |
|------|--------|:-----------:|:-------------:|:---------:|:-----------:|:-------:|:---------:|:-------:|
| Raw | ShadowKV++ | **1.306** | **1.306** | **0.140** | **0.140** | **0.000** | **0.000** | ✅ |
| Templated | ShadowKV++ | **1.489** | **1.489** | **0.168** | **0.168** | 0.686 | 0.686 | ✅ |
| Semantic | ShadowKV++ | **1.302** | **1.302** | **0.159** | **0.159** | **0.523** | **0.523** | ✅ |

## 3. SGLang Runtime (summary.md vs paper Table VIII)

| Model | summary.md | Paper | Verdict |
|-------|:----------:|:-----:|:-------:|
| 1.5B | +7.2% | — | (not in paper) |
| 3B | +8.4% | — | (not in paper) |
| 7B | **+16.7%** | **+16.7%** | ✅ |
| 14B | +12.7% | — | (not in paper) |
| 32B | **+5.1%** | **+5.1%** | ✅ |

Paper doesn't show 1.5B/3B/14B SGLang rows in the runtime table.

## 4. vLLM Runtime — DISCREPANCY

| Engine | summary.md (32B) | Paper Table VIII (32B) | Verdict |
|--------|:----------------:|:----------------------:|:-------:|
| APC vs no-cache | **+17.9%** | **+18.4%** | ❌ 0.5 pp gap |
| APC+MeritKV vs no-cache | **+18.5%** | **+19.0%** | ❌ 0.5 pp gap |

The vLLM `results.csv` has per-dataset values — 32B APC ranges from +12.9% (ag_news templated) to +22.6% (samsum rag). The aggregate depends on weighting. The paper's +18.4% / +19.0% appear to use a different aggregation than summary.md's +17.9% / +18.5%.

## 5. vLLM 7B (Paper Table VIII vs CSV)

| Engine | Paper 7B | CSV 7B (approx mean) | Verdict |
|--------|:--------:|:--------------------:|:-------:|
| APC vs no-cache | **+22.8%** | ~+19.5% | ❌ |
| APC+MeritKV vs no-cache | **+24.5%** | ~+20.5% | ❌ |

Paper's 7B numbers are higher than the CSV per-dataset values suggest. This may be because the paper reports only a subset of datasets for 7B, or uses a different aggregation.

## 6. Confidence Intervals (summary_by_engine.csv)

| Engine | CSV CI Low | CSV CI High | Paper CI | Verdict |
|--------|:----------:|:-----------:|:--------:|:-------:|
| ShadowKV++ | 1.342 | 1.388 | [1.342, 1.388] | ✅ |
| ShadowKV | 1.268 | 1.306 | [1.268, 1.306] | ✅ |
| Freq spec | 1.191 | 1.224 | [1.191, 1.224] | ✅ |

## 7. n_rows Consistency

| Engine | n_rows in CSV | Expected | Verdict |
|--------|:------------:|:--------:|:-------:|
| ShadowKV++ | 898 | 898 | ✅ |
| ShadowKV | 898 | 898 | ✅ |
| All main engines | 898 | 898 | ✅ |
| Scaffold-only | 150 | 150 | ✅ |
| Early-layer | 150 | 150 | ✅ |
| Logit-guard | 150 | 150 | ✅ |

## Summary

| What | Matches? |
|------|:--------:|
| Headline speedup (1.365x) | ✅ |
| Headline waste (0.156) | ✅ |
| Per-mode (raw 1.306, temp 1.489, sem 1.302) | ✅ |
| 95% CI intervals | ✅ |
| SGLang runtime | ✅ |
| **vLLM 32B aggregate** | **❌ (summary.md +17.9% vs paper +18.4%)** |
| **vLLM 7B aggregate** | **❌ (implied CSV mean ~+20% vs paper +22.8%)** |
| Row counts | ✅ |
