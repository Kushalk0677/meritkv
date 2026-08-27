# Runtime Experiments

This directory is organized by model family. Each model-family folder owns its curated runtime tables, summaries, and local raw run files.

## Layout

| Path | Contents |
|---|---|
| `qwen2.5/` | Qwen2.5 runtime results, curated tables, summaries, k-star outputs, and included run files. |
| `gemma4/` | Measured Gemma-4 Blackwell write-through campaign and overlay aggregates. |
| `native_enforcement_blackwell/` | Complete frozen two-model native SGLang enforcement package plus the Gemma-4-31B and Qwen2.5-32B storage-admission extensions, including raw request records, counters, logs, code, verifiers, original archives, and curated summaries. |

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
- Raw result trees may be kept locally and ignored by Git when they are too large for the public tracked tree.
- The broad production-runtime integrations are write-through. Their deltas
  establish compatibility and observed overhead, not MeritKV acceleration.
