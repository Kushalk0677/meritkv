# Native SGLang Enforcement on Blackwell

## Scope

This frozen experiment tests whether MeritKV decisions are physically enforced by a native SGLang cache backend. It is separate from the broad write-through SGLang, LMCache, and vLLM matrices.

- Hardware: NVIDIA RTX PRO 6000 Blackwell (96 GB).
- Models: Qwen2.5-32B-Instruct and Gemma-4-31B-it, with pinned checkpoint revisions.
- Workloads: DailyDialog, SAMSum, AG News, Dolly, and XSum in templated and RAG modes.
- Arms: native cache, MeritKV write-through, MeritKV enforced, and forced recomputation.
- Seeds: five paired seeds.
- Generation: greedy, one output token, fixed request order.
- Runtime: SGLang, float16 Triton attention, PyTorch sampling, and CUDA graphs disabled.
- Tuning: the frozen policy and workload were not retuned after results were observed.

## Completion

| Phase | Cells | Measured requests | Status |
|---|---:|---:|---|
| Smoke | 8 | 128 | complete |
| Native-action proof | 6 | 384 | complete |
| Full matrix | 400 | 102,400 | complete |
| **Total** | **414** | **102,912** | **verified** |

The causal proof produced 22 reuse and 42 bypass decisions per model. For every bypass, the requested 42 lookup skips and 42 write skips matched native execution counters. Gemma's SWARadix-specific counters also matched 42/42. The enforced arm therefore changed physical native-cache behavior rather than only logging a decision.

## Full-matrix result

Each model's full workload contained 12,800 requests. MeritKV allowed 12,750 and bypassed 50, so the frozen workload exercised enforcement but had a sparse bypass rate.

| Model | Enforced vs. write-through mean latency | P95 latency | GPU energy | Forced recomputation vs. enforced mean latency |
|---|---:|---:|---:|---:|
| Qwen2.5-32B | +0.197% | +0.106% | +0.114% | +15.745% |
| Gemma-4-31B | +0.144% | +0.102% | +0.102% | +13.575% |

This is native-action and parity evidence, not evidence that selective admission accelerated this workload. The sparse 50/12,800 bypass rate explains the near-parity outcome.

## Output comparison

Outputs were compared as exact UTF-8 text plus SHA-256, without normalization or tolerance. Write-through and enforced differed at 15/12,800 Qwen positions and 11/12,800 Gemma positions. Because generation was limited to one output token, each mismatch is a one-token difference. The result does not support a universal bitwise-equivalence claim.

## Artifact scope

The private ShadowKV repository retains the frozen configuration, implementation patch, accepted benchmark records, request-level artifacts, native action counters, server logs, verification tooling, and completion receipt. The public MeritKV reviewer repository retains this concise paper-facing summary only.

- Evidence-package SHA-256: `e7943267252fe160afdbbe37c72884eccd029cc66f22419688657d962d5f749b`
- Completion-receipt SHA-256: `476f33dab27e65e5e9f750fde3e25bd6dbccf085398d7791d1a6395eea9776ee`
- Original ZIP: `evidence_packages/2026-08-20/archives/MeritKV-Blackwell-native-enforcement-evidence-2026-08-20.zip`
- Complete byte-verified expansion: `evidence_packages/2026-08-20/native_enforcement_package/`

## Gemma-4-31B storage-admission extension

The 2026-08-23 extension uses a frozen capacity-pressure trace to separate
storage admission from lookup bypass. Against native admit-all LRU,
skip-write-only lowers mean latency by 7.73% and GPU energy by 8.54%, prevents
all measured evictions, and preserves 4/4 hot families. Joint enforcement
reaches 7.98% lower mean latency and 8.95% lower energy, but improves only
0.27% in mean latency over skip-write alone. Storage admission therefore
drives nearly all of the result. P95 does not improve reliably.

- Curated report, per-seed aggregates, and code: `storage_admission_extension/`
- Complete extracted packages and original ZIPs: `evidence_packages/2026-08-23/`
- Storage-package SHA-256: `8ea89aae9c21963a5983898333aebfef488edb45e83908f1755e021336d38797`
- Self-containment-supplement SHA-256: `1e8a7e28901921e50ca89d24a1d00f9e14eb71c123594e696e8848f6e0acb2fa`

## Qwen2.5-32B storage-admission extension

The separate 2026-08-24 extension repeats the frozen storage-admission design
on Qwen2.5-32B and adds a genuine native LFU arm. Relative to MeritKV
write-through LRU, skip-write-only reduces paired-seed mean latency by 7.97%
and GPU energy by 8.76% while eliminating measured physical evictions. P95
increases by 1.65%. Native LFU is approximately 4.2% faster and lower-energy
than MeritKV skip-write-only on this trace. Every arm preserves and recovers
all four hot families, so recovery is not a differentiating Qwen outcome.

- Curated report, per-seed aggregates, and code: `storage_admission_extension_qwen25_32b/`
- Complete extracted package and original ZIP: `evidence_packages/2026-08-24/`
- Package SHA-256: `5b2b81fd335ab89266591a94582b43b15bfa07e0037ff1caec5752cd8f242f7c`

The Qwen package's frozen source-closure paths use POSIX separators. Replay its
bundled verifier under Linux or WSL; native Windows Python reports a path-set
mismatch even though the files and hashes are identical.
