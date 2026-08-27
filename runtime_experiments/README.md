# Runtime Experiments

This directory is organized by model family. Each folder owns its curated
runtime tables, complete project-retained raw run files, commands, environment
captures, logs, and source snapshots.

## Layout

| Path | Contents |
|---|---|
| `qwen2.5/` | Qwen2.5 runtime results, curated tables, summaries, k-star outputs, balanced admission, and preserved dated raw campaigns. |
| `gemma4/` | Measured Gemma-4 Blackwell write-through campaign and overlay aggregates. |
| `native_enforcement_blackwell/` | Complete frozen two-model native SGLang enforcement package plus the Gemma-4-31B and Qwen2.5-32B storage-admission extensions, including raw request records, counters, logs, code, verifiers, original archives, and curated summaries. |
| `archive/` | Complete retained older root-workspace runtime snapshot, explicitly separated from current aggregates. |

## Notes

- Gemma-4 includes measured five-seed Blackwell records for vLLM APC,
  SGLang RadixAttention, and LMCache plus paper-facing overlay aggregates.
- Qwen2.5 remains organized with `vllm/`, `sglang/`, `lmcache/`, and `kstar/` outputs.
- The native Qwen2.5-1.5B balanced-admission bundle is at
  `qwen2.5/sglang/balanced_admission/`; it is separate from the broad
  write-through scale study.
- `native_enforcement_blackwell/` physically enforces skip-lookup and
  skip-write decisions. Its sparse-bypass full matrix is a parity result, not
  a selective-admission acceleration result.
- All raw runtime records retained in the project workspace are tracked. Model
  weights, dataset caches, virtual environments, and transient Python caches
  remain outside the release.
- The broad production-runtime integrations are write-through. Their deltas
  establish compatibility and observed overhead, not MeritKV acceleration.

See `../docs/EXPERIMENT_CATALOG.md` for the code-to-data map and
`../docs/HARDWARE_AND_ENVIRONMENTS.md` for machine and software captures.
