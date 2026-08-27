# Qwen2.5-32B storage-admission extension

This is a separate frozen capacity-pressure extension. The accepted August 20 result and the completed Gemma extension remain byte-for-byte unchanged.

Verified scope: 31 cells and 3,344 requests: one excluded calibration cell, five measured smoke cells, and 25 evaluation cells over five paired seeds. A separate excluded LFU feasibility preflight passed before calibration and freeze. All five measured arms use ordinary `RadixCache` with `hybrid_swa=False`; the frequency-aware comparator uses genuine LFU retention.

The verifier confirms non-overlapping evaluation timestamps in the exact frozen 25-row plan order, as well as paired identical traces within each seed.

The previously delivered self-containment supplement is referenced by exact hash in `SELF_CONTAINMENT_SUPPLEMENT_REFERENCE.json`; it was not rebuilt.

Verify after extraction with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 packet/verify_extension.py delivery results --packet packet
```
