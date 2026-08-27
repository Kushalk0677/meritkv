# Native vLLM Execute-or-Bypass Experiment

## Purpose

This experiment directly addresses the TMLR request for one enforced native
backend evaluation of the decision to consume an existing cache hit or bypass
it and recompute.

It uses vLLM Automatic Prefix Caching (APC) and the native request-level
`cache_salt` field:

- requests sharing a stable salt can reuse warmed APC blocks;
- a request with a unique salt cannot match those warmed blocks and therefore
  recomputes the prefix;
- the server and all unrelated cache state remain live.

This is cleaner than flushing the full cache on every bypass. The runner checks
vLLM's reported `cached_tokens` and aborts if the native actuator does not
produce the expected hit/miss behavior.

## Experimental question

For a prefix that is already resident in vLLM's native APC, does enforcing
MeritKV's request-time consume-or-recompute decision improve latency relative
to always consuming the native hit?

This experiment tests only resident-hit consumption. It does not test storage
admission or eviction; those remain covered by the separate Gemma-4-31B
capacity-pressure experiment.

## Arms

| Arm | MeritKV decision computed | Native hit consumed | Purpose |
|---|---:|---:|---|
| `forced_recompute` | No | Never; unique salt per request | No-reuse latency and deterministic-output reference |
| `native_apc` | No | Always; stable warmed salt | Native cache baseline |
| `meritkv_write_through` | Yes | Always | Controller-overhead/decision control |
| `meritkv_enforced` | Yes | Only when allowed | Causal native execute-or-bypass result |

The comparison `meritkv_enforced` versus `meritkv_write_through` is the cleanest
estimate of the effect of enforcing the decision: both compute the same policy,
but only the former changes whether the native hit is consumed.

## Default protocol

- Backend: vLLM APC through its OpenAI-compatible completions endpoint.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Precision: float16.
- Prefix lengths: 16, 32, 64, 128, 256, 512, and 1024 tokens.
- Requests: 16 per prefix length per seed.
- Seeds: 42, 123, 456, 789, and 999.
- Generation: one greedy output token with explicit `top_p=1`, `top_k=-1`,
  `repetition_penalty=1`, and vLLM's neutral generation configuration. This
  tests direct next-token agreement without autoregressive amplification.
- Calibration: five excluded paired hit/recompute probes per prefix length;
  full-prefill slope and incremental reuse-cost proxy are fit before measurement.
- Execution: sequential, `max-num-seqs=1`, eager mode.
- Cache state: all stable prefixes are warmed and verified resident before
  measured requests. Warmup and verification use different suffixes so the
  observed hit must come from the intended shared prefix rather than an
  identical full-prompt replay.

Qwen2.5-1.5B fits on a T4 and is sufficient for the reviewer-requested native
mechanism test. A larger model is optional, not required to establish that the
native per-request actuator exists and is measured honestly.

The intended primary Blackwell confirmation is an enforced extension of the
coauthor's existing write-through production integration. That path should be
preferred because it modifies the same codebase used by the paper's Blackwell
runtime study. The standalone wrapper below is retained only as a fallback
native-actuator diagnostic using the paper-aligned
`Qwen/Qwen2.5-32B-Instruct` model:

```bash
bash run_blackwell.sh
```

The fallback wrapper writes to a timestamped
`results/blackwell_qwen25_32b_*` directory
and creates a ZIP plus SHA-256 beside it. It packages the evidence even when a
strict validity check returns nonzero. Do not tune the policy or select prefix
lengths after observing the Blackwell outcome. Keeping the T4 and Blackwell
protocols aligned makes the hardware comparison defensible.

## Run on a CUDA Linux machine

Create a clean environment with a CUDA-compatible PyTorch installation, then:

```bash
python3 -m pip install -r requirements.txt
bash run_experiment.sh
```

If the Hugging Face account is already authenticated, no token is placed in the
command line. Otherwise authenticate beforehand with `hf auth login`.

Environment overrides:

```bash
MERITKV_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
MERITKV_PORT=8000 \
MERITKV_GPU_MEMORY_UTILIZATION=0.90 \
bash run_experiment.sh
```

## Run through the Colab CLI

From WSL, with the official Colab CLI already authenticated:

```bash
cd /mnt/c/shadowkv/v10/experiments/native_vllm_execute_bypass
bash run_with_colab_cli.sh
```

The Colab wrapper uploads the experiment, installs the pinned environment from
inside the remote driver (avoiding the CLI helper's short installation timeout),
runs it on a T4, downloads the complete result archive, and stops the session.

## Mandatory validity conditions

The result is usable only if all of these pass:

1. The actuator self-test reports `cached_tokens > 0` for a stable-salt request.
2. The same prompt with a unique salt reports `cached_tokens == 0`.
3. Every `native_apc` and `meritkv_write_through` measured request reports a
   native cached-token hit.
4. Every enforced MeritKV bypass reports zero cached tokens.
5. Generated token IDs or generated text agree with forced recomputation for
   every paired case.
6. All four arms and all five seeds complete.

If MeritKV never chooses bypass under the frozen controller, the experiment is
valid but non-informative. Do not tune the decision threshold on measured
requests to force a favorable result. Any calibration is performed before the
measured run and is recorded separately.

## Interpretation

Possible outcomes are all reportable:

- If enforced MeritKV is faster than write-through and native APC, it directly
  supports the broad resident-hit execute-or-bypass claim in this configuration.
- If it is statistically indistinguishable, report that the native gate is
  compatible but does not improve resident-hit latency here.
- If it is slower, narrow the paper's positive claim to speculative/storage
  admission and report the resident-hit result as a useful negative finding.

Do not combine this result numerically with the capacity-pressure experiment;
they evaluate different cache actions.

## Output

Each run produces:

- `requests.json`: all request-level timing, cached-token, decision, and output
  records;
- `summary.json`: aggregate metrics and validity flags;
- `RESULTS.md`: compact human-readable report;
- `actuator_self_test.json` and `calibration.json`;
- per-seed request and warmup ledgers;
- vLLM stdout/stderr, `pip freeze`, Python version, and `nvidia-smi`; and
- the exact executed MeritKV controller source and configuration; and
- `MANIFEST_SHA256.txt`.

The completed 13 August 2026 T4 pilot and its interpretation are recorded in
`MEASURED_RESULT_20260813.md`; the original raw archive is retained under
`downloaded_results/`. The native actuator and request telemetry passed, but
the measured latency effect was statistically indistinguishable and the
predeclared all-cases exact-output condition did not pass. It does not support
a positive resident-hit claim on T4. The paper-level decision remains pending
the frozen Blackwell run.
