# Paper vs Repo Cross-Check

Checked: `paper/tmlr/`, `results/`, `runtime_experiments/`, `experiments/`, `src/`.

## Discrepancies Found

### 1. RESULTS.md uses "ShadowKV++" (needs rename)

`results/RESULTS.md` still refers to `shadow_kv_plus` throughout. Needs
updating to `MeritKV` to match the TMLR rename. The CSV column names
(`shadow_kv_plus`, `shadow_kv`) can stay as-is (they're internal engine
names) but the Markdown prose should use the paper name.

### 2. Controlled results: 898 files, 3 seeds

The paper says "750 benchmark runs" and "five seeds." The controlled result
bundle has 898 benchmark files across seeds {42, 123, 456}.

| Claim | Paper | Repo |
|-------|-------|------|
| Benchmark files | 750 | 898 |
| Seeds | 5 (42, 123, 456, 789, 999) | 3 (42, 123, 456) |

**Fix:** Update paper to say "898 benchmark files across 3 seeds." The
summary_by_engine.csv has `n_rows=898` for every engine.

### 3. vLLM runtime table: 1.5B-14B are projected

`runtime_experiments/vllm/README.md` states:
> "1.5B, 3B, and 14B rows are scaled from measured anchors using
> model-size ratios. Missing datasets are scaled from the nearest
> measured dataset by token length."

The paper's Table VIII shows a full 5-model table (1.5B–32B) without
distinguishing measured from projected rows. Only 7B and 32B are measured.

**Fix:** Add footnotes to the runtime table marking projected rows.

### 4. vLLM 32B: paper says +19.0%, CSV says +18.5%

Paper Table VIII: APC+MeritKV at 32B = +19.0% vs no-cache.
CSV `runtime_experiments/vllm/results.csv`: 32B APC+MeritKV = +18.5%.

| Claim | Paper | CSV |
|-------|-------|-----|
| vLLM 32B APC+MeritKV vs no-cache | +19.0% | +18.5% |
| vLLM 7B APC+MeritKV vs no-cache | +24.5% | +24.5% (matches) |

The 32B gap is 0.5 pp — small but present. Likely aggregation difference
(simple mean vs weighted). Paper should match the CSV.

### 5. SGLang CSV: 290 rows not 300

`runtime_experiments/sglang/README.md` confirms 290 rows — 10 missing
dataset cells. Paper presents a full 5×3×10×2 table without caveat.

**Fix:** Add a note that 10 of 300 cells are imputed from nearest measured.

### 6. SGLang 32B numbers reconciled

Earlier review claimed SGLang 7B = 15.7% not 16.7% and 32B = 2.7% not 5.1%.
This was based on a simple mean recomputation from the CSV. The paper's
values (16.7%, 5.1%) are confirmed by `runtime_experiments/summary.md`.
The discrepancy was an aggregation method difference — the paper uses
a weighted mean, the review used a simple mean. Both are valid; the paper
should document which aggregation it uses.

### 7. Memory-bound trace results pushed

`results/mixed_traffic/MEMORY_BOUND_RESULTS.md` now in repo (commit 6b129e7).
No conflicts with paper — the paper doesn't include this experiment yet.
When adding to the paper, ensure the occupancy cap baseline and Blackwell
headline framing are preserved.

### 8. Experimental scripts pushed

- `experiments/run_admission_baselines.py` — in repo
- `experiments/run_mixed_traffic.py` — in repo
- `experiments/run_memory_bound_trace.py` — in repo

These are utility scripts, not results — no paper conflict.

### 9. RESULTS.md waste interpretation notes

`results/RESULTS.md` has a correct caveat:
> "Raw-mode ShadowKV++ gains should be interpreted as bypass and
> overhead-avoidance gains, not as proof of exact KV reuse."

This is consistent with the paper's current framing. No change needed.

## Items That Check Out

| Claim | Status |
|-------|--------|
| shadow_kv_plus mean speedup 1.365x | ✓ CSV confirms (1.3652) |
| shadow_kv waste 0.264 | ✓ CSV confirms (0.264) |
| shadow_kv_plus waste 0.156 | ✓ CSV confirms (0.1556) |
| frequency_speculative waste 0.284 | ✓ CSV confirms (0.284) |
| reactive/greedy/strict waste = 0 | ✓ CSV confirms (0.000) |
| SGLang 7B +16.7% | ✓ summary.md confirms |
| SGLang 32B +5.1% | ✓ summary.md confirms |
| Blackwell parity (59.4 vs 59.6) | ✓ 5-replicate run confirms |
| Fidelity: TinyLlama 0.966 | ✓ all_results.json supports |
| Fidelity: Qwen2.5 0.200 | ✓ all_results.json supports |

## Priority Action Items

| Priority | Fix | File |
|----------|-----|------|
| P0 | Update seed/counts (898, 3 seeds) | `paper/tmlr/50platform.tex`, `60results.tex` |
| P0 | Mark projected runtime rows | `paper/tmlr/65runtime.tex` |
| P0 | Sync vLLM 32B (19.0% → 18.5%) | `paper/tmlr/65runtime.tex` |
| P1 | Rename ShadowKV++ → MeritKV in RESULTS.md | `results/RESULTS.md` |
| P1 | Document SGLang CSV aggregation method | `paper/tmlr/65runtime.tex` |
| P1 | Note 290/300 SGLang cells imputed | `paper/tmlr/65runtime.tex` |
| P2 | Add memory-bound trace to paper appendix | `paper/tmlr/` (new section) |
