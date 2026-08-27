# Native SGLang Balanced Admission

This directory contains the complete measured artifact bundle for the
paper-facing Qwen2.5-1.5B balanced-admission experiment on an NVIDIA RTX PRO
6000 Blackwell GPU.

## Paper-facing result

The primary cell is `pathological_semantic_ag_news`:

| Engine | Attempted cache-query traffic | Native skip-lookups | Mean latency |
|---|---:|---:|---:|
| SGLang RadixAttention | 957.788160 MB | 0 | 10.895 ms |
| SGLang RadixAttention + MeritKV | 6.766592 MB | 254/256 | 20.233 ms |

The attempted cache-query traffic decreases by 99.3%. The latency increase is
part of the result: this experiment demonstrates enforced waste avoidance, not
a latency improvement on this small model.

## Configuration

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Hardware: NVIDIA RTX PRO 6000 Blackwell
- Workload: pathological short-prefix semantic AG News
- Requests: 256
- Seed: 20260703
- Admission preset: `balanced`
- Utility threshold: `util_min_ms=2.0`
- Bootstrap admissions: `min_bootstrap_admissions=0`
- Enforcement: native SGLang per-request skip-lookup/skip-write hook

The bundle also includes pathological templated AG News and templated SAMSum
control workloads, plus vLLM write-through comparison cells. Only SGLang
enforces the native admission decisions in this harness.

## Directory contents

| Path | Contents |
|---|---|
| `results/summary.csv` | Full aggregate export for all 18 cells. |
| `results/summary_compact.csv` | Compact reviewer-facing aggregate. |
| `results/summary.json` | JSON aggregate with configuration and counters. |
| `results/cells/` | One `result.json` and server records per cell. |
| `results/per_request/` | 18 request-level JSONL files, each containing 256 measured requests. |
| `results/traces/` | Three input traces, each containing 256 requests. |
| `results/metadata/` | GPU, host, container, package, runtime, configuration, and captured source metadata. |
| `results/REPORT.md` | Full balanced-run aggregate report. |
| `results/KUSHAL_BALANCED_UTIL2_HANDOFF.md` | Detailed run interpretation and verification record. |
| `results/POSTPROCESS_SKIP_LOOKUP_ACCOUNTING.md` | Exact accounting correction applied to native skipped lookups. |
| `run_logs/` | Primary, nohup, and status logs for the run. |
| `session_files/` | Original Python harness and shell command used for the run. |
| `previous_low_latency_summary/` | Small retained summary from the preceding low-latency run used by the comparison CSV. |
| `SHA256SUMS.txt` | SHA-256 checksums for every retained artifact other than the checksum file itself. |

## Accounting note

For native SGLang requests marked `skip_lookup=True`, no cache lookup or
transfer was attempted. The initial harness still assigned those requests
cache-query and miss-waste bytes. The retained postprocess corrected those
fields to zero for skipped-lookup rows and regenerated the affected aggregates.
It did not alter latency or speedup fields. The correction is documented in
`results/POSTPROCESS_SKIP_LOOKUP_ACCOUNTING.md`, and each adjusted request is
marked in the per-request record.

Consequently, the traffic result is a deterministic accounting of attempted
cache-query bytes under enforced skip-lookup behavior. It is not an independent
hardware-link bandwidth measurement or a modelled latency result.

## Verification

The paper values can be checked directly in
`results/comparison_vs_low_latency_2026-07-03.csv`:

- `957.788160 MB` rounds to `957.8 MB`;
- `6.766592 MB` rounds to `6.8 MB`;
- `254` native skip-lookups are recorded;
- `10.895051 ms` rounds to `10.9 ms`;
- `20.232623 ms` rounds to `20.2 ms`; and
- the computed query-traffic reduction rounds to `99.3%`.

The measured run can also be audited from the corresponding `result.json`,
256 request-level records, server counters, and run logs. Environment versions
for this separate native-admission image are recorded under
`results/metadata/`; they should not be inferred from the broad write-through
runtime campaign.

## Scope

This bundle supports the narrow paper claim that native balanced admission can
enforce cache lookup avoidance on the tested pathological short-prefix
workload. It does not establish a latency benefit, generalize the result to
other workloads or models, or turn the broad vLLM/SGLang/LMCache write-through
campaign into an enforced-acceleration study.
