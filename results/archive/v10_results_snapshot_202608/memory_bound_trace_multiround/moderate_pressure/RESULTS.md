# Four-Arm Multi-Round Admission Enforcement (56 GB budget (moderate pressure))

**Model**: google/gemma-4-31B-it on Blackwell RTX PRO 6000
**Trace**: 20 fill + [15 churn + 15] recovery x 4 rounds = 140 requests

| Arm | Admission mode | Enforcement |
|-----|---------------|-------------|
| **No cache** | Disabled | N/A |
| **Native (APC)** | Disabled | Always write |
| **Overlay** | U = B - C - W (logged) | Write-through |
| **Enforced** | U = B - C - W (enforced) | Skip-write + skip-lookup |

### Table 1: Overall Metrics (mean over 5 seeds)

| Metric | No cache | Native (APC) | Overlay | Enforced (skip-write) |
|--------|:--------:|:------------:|:-------:|:---------------------:|
| Mean latency | 85.75 $\pm$ 0.36 ms | 71.36 $\pm$ 0.32 ms | 71.37 $\pm$ 0.27 ms | 63.06 $\pm$ 0.28 ms |
| GPU energy | 3422.38 $\pm$ 23.31 J | 2826.00 $\pm$ 37.80 J | 2850.00 $\pm$ 55.17 J | 2523.52 $\pm$ 53.70 J |
| Final-round recovery | 0.0\% | 51 $\pm$ 1\% | 51 $\pm$ 2\% | 92 $\pm$ 1\% |
| Total evictions | 60 | 31.20 $\pm$ 1.04 | 30.80 $\pm$ 1.04 | 4.80 $\pm$ 1.36 |

### Table 2: Pairwise Decomposition

| Overlay - Native | +0.0 ms (+0.0\\%) | 0.0 pp | Controller overhead |
| Enforced - Overlay | -8.3 ms (-11.6\\%) | 41.0 pp | Enforcement benefit |
| Enforced - Native | -8.3 ms (-11.6\\%) | 41.0 pp | Net system value |

* 5 seeds, CIs use Student's t-distribution (t_{4,0.025} = 2.776).
* Per-request admission logs in seed-level trace.json files.
