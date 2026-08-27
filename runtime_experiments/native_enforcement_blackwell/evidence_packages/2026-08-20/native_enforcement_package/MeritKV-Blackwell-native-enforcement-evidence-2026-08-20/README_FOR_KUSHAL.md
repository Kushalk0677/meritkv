# MeritKV Blackwell native-enforcement evidence

Prepared for Kushal Khemani on 2026-08-20 from the completed First Light run
`meritkv_blackwell_enforced_20260820`.

## Executive result

The experiment is complete and independently verified:

- Smoke: 8/8 cells, 128 measured requests.
- Two-model native-action proof: 6/6 cells, 384 measured requests.
- Full frozen matrix: 400/400 cells, 102,400 measured requests.
- Total: 414 cells and 102,912 measured requests.

The causal proof produced 22 reuse and 42 bypass decisions per model. For every
bypass, requested `skip_lookup` and `skip_write` counts matched executed native
cache counters at 42. This includes the Gemma-specific SWARadix lookup and write
paths. The enforced arm therefore declined physical native-cache consumption;
it did not merely log the MeritKV decision.

## Frozen scope

- Models:
  - `Qwen/Qwen2.5-32B-Instruct`, revision
    `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`.
  - `google/gemma-4-31B-it`, revision
    `b9ea41a2887d8607f594846523f94c6cc75ac8a4`.
- Arms: native cache, MeritKV write-through, MeritKV enforced, and forced
  recomputation/no-cache.
- Seeds: 1 through 5, paired across arms.
- Datasets: DailyDialog, SAMSum, AG News, Dolly, and XSum.
- Modes: templated and RAG.
- Generation: `max_tokens=1`, `temperature=0`.
- Runtime: RTX PRO 6000 Blackwell, SGLang, float16, Triton attention, PyTorch
  sampling, CUDA graphs disabled, frozen request order and server settings.
- No result-driven retuning was permitted or performed.

The exact model, tokenizer, dataset, image, policy, workload, runtime, and plan
hashes are in `packet/frozen_config.json` and
`packet/dataset_workload_manifest.json`.

## Full-matrix findings

For each model, both MeritKV arms selected 12,750 allows and 50 bypasses across
12,800 requests. The frozen workload therefore exercised enforcement but had a
sparse bypass rate.

Enforced versus write-through:

| Model | Mean end-to-end latency | P95 end-to-end latency | GPU energy |
| --- | ---: | ---: | ---: |
| Qwen2.5-32B | +0.1967% | +0.1061% | +0.1143% |
| Gemma 4 31B | +0.1437% | +0.1024% | +0.1015% |

This is native-enforcement and parity evidence, not evidence of a
selective-admission performance gain. As a control, forced recomputation was
15.745% slower for Qwen and 13.575% slower for Gemma in mean end-to-end
latency, with 24.583% and 24.088% more GPU energy, respectively, than enforced.

## Exact output comparison

Outputs were compared as exact UTF-8 text plus SHA-256, with no normalization
or tolerance. Write-through versus enforced differed at 15/12,800 Qwen
positions and 11/12,800 Gemma positions. Across all four arms, 60/25,600
logical request positions contained at least one difference. Because the run
generated one completion token per request, each reported difference is a
one-token difference. Every mismatch, arm value, and hash is preserved in
`results/control/full_verification.json` and the request traces.

## Package map

- `results/{smoke,proof,full}/`: accepted benchmark JSON, request-level traces,
  execution ledgers, raw launch-command ledgers, native action counters, and
  server logs used by the verifier.
- `results/control/`: completion receipt, immutable scientific report, phase
  reports, model compatibility proof, and the 50-workload offline dataset
  preflight receipt.
- `packet/`: frozen configuration and plans, dataset/model probes, runtime cell
  wrapper, Gemma SWARadix patch, trace utilities, and verifier.
- `MANIFEST_SHA256.txt`: SHA-256 for every other file in this package.

Failed/interrupted attempts, redundant `client_stdout.json` copies, container
inspection, production topology, restoration logs, service units, and
controller/watchdog state are intentionally excluded. The retained server logs
and commands are the minimum runtime evidence needed for independent phase
verification. Request traces preserve exact one-token public-dataset outputs.

## Verification

From the extracted package root:

```bash
shasum -a 256 -c MANIFEST_SHA256.txt
python3 packet/verify_operation.py all results \
  | jq '{status,cells,requests,phases:[.phases[]|{phase,status,cells,requests}]}'
```

Expected verifier summary:

```json
{
  "status": "pass",
  "cells": 414,
  "requests": 102912,
  "phases": [
    {"phase": "smoke", "status": "pass", "cells": 8, "requests": 128},
    {"phase": "proof", "status": "pass", "cells": 6, "requests": 384},
    {"phase": "full", "status": "pass", "cells": 400, "requests": 102400}
  ]
}
```

The completion receipt and stored scientific report are immutable source-run
artifacts. Their expected SHA-256 values are:

- `results/control/completion_receipt.json`:
  `476f33dab27e65e5e9f750fde3e25bd6dbccf085398d7791d1a6395eea9776ee`
- `results/control/scientific_verification.json`:
  `af590621fcd687769ab7af511e30302b65f0e81ed70cd667cf3bee30fd2cf227`

The original source run also passed the receipt verifier before packaging.
Receipt verification is path-bound to that source result root; use the `all`
command above for an independent verification after extraction or relocation.

## Operational closeout

After the experiment, the production Qwen service was restored and verified by
health, model discovery, real loopback generation, AgentVM relay generation,
LiteLLM readiness, Hermes gateway status, clean logs, restart/OOM state, GPU
ownership, and cron resumption. The experiment launch and watchdog timers were
then disabled.
