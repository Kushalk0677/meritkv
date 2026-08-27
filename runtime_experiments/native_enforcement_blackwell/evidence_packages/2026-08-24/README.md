# Blackwell Qwen Storage-Admission Evidence: 2026-08-24

This private directory preserves the delivered Qwen2.5-32B archive byte-for-byte, its SHA-256 sidecar, and the complete extracted evidence tree.

## Archive

| File | SHA-256 |
|---|---|
| `archives/MeritKV-Blackwell-storage-admission-extension-Qwen2.5-32B-2026-08-24.zip` | `5b2b81fd335ab89266591a94582b43b15bfa07e0037ff1caec5752cd8f242f7c` |

## Extracted tree

`storage_admission_extension_qwen25_32b_package/` contains the full frozen package: protocol and plans, excluded LFU feasibility preflight and calibration, smoke and evaluation records, request traces, native action and eviction counters, server snapshots and logs, output text and hashes, source snapshot, implementation wrappers, restoration evidence, manifests, scientific report, and relocatable verifier.

The bundled verifier passes all 31 measured cells and 3,344 measured requests under Linux/WSL:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 packet/verify_extension.py delivery results --packet packet
```

The reviewer-facing derivative is mirrored at `../../storage_admission_extension_qwen25_32b/`. It contains only aggregate and per-seed scalar records, machine-readable summaries, provenance, the frozen protocol and verifier, executed instrumentation wrappers, and the compact source closure. The private extracted tree remains authoritative for request-level and server-level provenance.
