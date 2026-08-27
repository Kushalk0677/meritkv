# Historical Implementation Reports

These reports preserve the detailed implementation history behind the current
MeritKV codebase. They explain how the controller was integrated, how hot-path
overheads were reduced, how raw-mode gating evolved, and how semantic
self-matches were corrected.

They are included for engineering transparency. They are not independent
sources for paper claims, and their smoke-test or diagnostic numbers do not
override the manuscript, which is the sole authority for claims.

| Report | Scope |
|---|---|
| `INTEGRATION_REPORT.md` | Controller integration, interfaces, and tests. |
| `PERFORMANCE_FASTPATH_REPORT.md` | Fast-path and controller-overhead work. |
| `RAW_MITIGATION_VARIANTS_REPORT.md` | Raw-mode mitigation variants. |
| `RAW_MODE_CONSERVATIVE_GATE_REPORT.md` | Conservative raw-mode admission gate. |
| `SEMANTIC_FIX_REPORT.md` | Semantic-index self-match correction. |
| `README_MERITKV_BASE.md` | Detailed historical implementation overview. |

When a report describes an earlier experiment configuration or a local smoke
test, read it as development provenance rather than current quantitative
evidence.
