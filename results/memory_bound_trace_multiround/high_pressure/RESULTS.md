# Four-Arm Multi-Round Admission Enforcement (28 GB, High Pressure)

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
| Mean latency | 85.90 $\pm$ 0.24 ms | 77.70 $\pm$ 0.43 ms | 77.89 $\pm$ 0.42 ms | 64.88 $\pm$ 0.35 ms |
| GPU energy | 3422.52 $\pm$ 60.66 J | 3117.22 $\pm$ 43.66 J | 3088.78 $\pm$ 35.62 J | 2569.08 $\pm$ 29.35 J |
| Final-round recovery | 0.0\% | 21 $\pm$ 2\% | 22 $\pm$ 2\% | 83 $\pm$ 1\% |
| Total evictions | 60 | 46.60 $\pm$ 1.42 | 45.60 $\pm$ 2.72 | 12.20 $\pm$ 1.04 |

### Table 2: Pairwise Decomposition

| Overlay - Native | +0.2 ms (+0.2%) | 1.0 pp | Controller overhead |
| Enforced - Overlay | -13.0 ms (-16.7%) | 61.0 pp | Enforcement benefit |
| Enforced - Native | -12.8 ms (-16.5%) | 62.0 pp | Net system value |

* 5 seeds, CIs use Student's t-distribution (t_{4,0.025} = 2.776).
* Per-request admission logs in seed-level trace.json files.

This deliberately capacity-pressured trace supports a narrow locality and
tail-behavior claim, not broad production acceleration across workloads.
