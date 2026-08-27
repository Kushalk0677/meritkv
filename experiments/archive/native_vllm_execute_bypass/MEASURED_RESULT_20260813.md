# Measured Native vLLM Execute-or-Bypass Result

## Status

The four-arm native vLLM pilot completed on a Tesla T4 on 13 August
2026. All requested arms, five paired seeds, and 2,240 measured requests
completed. The native `cache_salt` actuator and all cached-token enforcement
checks passed.

This experiment does **not** support a positive claim that MeritKV improves
latency by declining already-resident GPU-cache hits in this T4 configuration.
It is a pilot, not the final manuscript decision. The primary follow-up will
extend the coauthor's existing Blackwell write-through integration with native
enforcement and rerun the paper-aligned configuration before choosing the
revision branch. The standalone `cache_salt` harness is only a fallback.

## Frozen configuration

- Backend: vLLM 0.10.2 with Automatic Prefix Caching enabled.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Hardware: Tesla T4, 15,360 MiB.
- Precision: float16; eager execution; `max-num-seqs=1`.
- Prefix lengths: 16, 32, 64, 128, 256, 512, and 1,024 tokens.
- Replicates: five paired seeds, with 16 requests per prefix length and arm.
- Generation: one greedy output token.
- Arms: forced recomputation, native APC, MeritKV write-through, and MeritKV
  enforcement.
- Native actuator: stable per-arm `cache_salt` for a warmed hit and a unique
  per-request salt for forced recomputation.

## Measured result

| Arm | Requests | Mean latency (ms) | P95 (ms) | Cached tokens/request | Cache-hit requests | MeritKV bypasses | Enforced bypasses |
|---|---:|---:|---:|---:|---:|---:|---:|
| Forced recomputation | 560 | 92.380 | 275.583 | 0.00 | 0 | 0 | 0 |
| Native APC | 560 | 41.267 | 52.655 | 290.29 | 560 | 0 | 0 |
| MeritKV write-through | 560 | 40.919 | 53.314 | 290.29 | 560 | 160 | 0 |
| MeritKV enforced | 560 | 40.658 | 52.114 | 283.43 | 400 | 160 | 160 |

The main causal comparison is MeritKV enforcement versus MeritKV
write-through. Both arms compute the same frozen policy decisions, while only
the enforced arm applies them to native cache consumption.

- Enforced versus write-through: `1.0071x` mean paired-seed speedup, 95% CI
  `0.9649-1.0493`; mean latency change `-0.616%`, 95% CI
  `-4.653% to +3.422%`.
- Enforced versus native APC: `1.0153x` mean paired-seed speedup, 95% CI
  `0.9630-1.0676`; mean latency change `-1.372%`, 95% CI
  `-6.373% to +3.629%`.
- Native APC versus forced recomputation: `2.2434x` mean paired-seed speedup,
  95% CI `2.0580-2.4289`; mean latency change `-55.270%`.

MeritKV bypassed all measured 16- and 32-token resident hits and consumed all
hits at 64 tokens and above. Restricting the comparison to the 160 bypassed
cases still gives no benefit: `1.0041x` mean paired-seed speedup, 95% CI
`0.9093-1.0989`, with mean latency change `+0.066%`.

## Enforcement and output checks

- Actuator self-test: passed.
- All four arms and five seeds: completed.
- Every native APC and write-through request: nonzero cached-token hit.
- Every enforced MeritKV bypass: zero cached tokens.
- MeritKV decisions: nonzero bypass count and identical between the
  write-through and enforced arms.

The predeclared all-cases exact-output check did not pass. Against forced
recomputation, the one-token agreement counts were:

- native APC: 558/560;
- MeritKV write-through: 558/560; and
- MeritKV enforced: 559/560.

The disagreements also occur between repeated native APC executions, so they
are not specific to the MeritKV actuator. Nevertheless, the protocol required
all 560 cases to agree, and this result must not be described as having passed
that strict validity condition.

## Interpretation

The experiment successfully demonstrates native, per-request enforcement, but
its T4 performance estimate is statistically indistinguishable from write-
through/native APC. It cannot by itself retain a broad positive resident-hit
consume-or-recompute claim.

Pending the Blackwell result, the defensible decision rule is:

1. if Blackwell shows a positive paired effect, report it as a configuration-
   scoped native resident-hit result alongside the T4 null;
2. if Blackwell is also null or negative, narrow the positive systems claim to
   speculative/storage admission;
3. in either outcome, preserve the write-through overlays as compatibility and
   controller-overhead evidence; and
4. do not tune the controller or select a post-hoc workload to manufacture a
   positive result.

## Preserved evidence

- Original downloaded archive:
  `downloaded_results/meritkv_native_vllm_execute_bypass_results_20260813T044424Z.zip`
- Archive SHA-256:
  `79981da7f14a2fbe4d684b2e9f2d187a28bafee6b73f66160cffdc8a4f6ff407`
- Extracted inspection copy:
  `downloaded_results/attempt_20260813_one_token/`

The archive contains request-level JSON, per-seed ledgers, calibration probes,
the actuator self-test, exact executed source/configuration, server logs,
environment capture, and the generated aggregate report.

The original archive's inner manifest matches the result JSON, per-seed
ledgers, calibration, source snapshot, and environment files. Three streaming
logs (`benchmark_driver.log`, `server/stdout.txt`, and `server/stderr.txt`)
received final bytes after that manifest was written, so their three inner
hashes are stale. The original archive is preserved unchanged and identified
by the outer SHA-256 above. The runner has been corrected to generate the inner
manifest only after the benchmark process and logs close on future runs.
