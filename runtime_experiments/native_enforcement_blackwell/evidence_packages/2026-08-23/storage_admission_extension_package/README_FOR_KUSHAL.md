# Gemma 4 31B storage-admission extension

This is a separate frozen capacity-pressure extension. The accepted August 20 result remains unchanged.

Verified scope: 25 cells and 2688 requests: one calibration cell, four all-arm smoke cells, and 20 evaluation cells over five paired seeds.

The four required arms use the proven default Gemma `SWARadixCache` topology. LFU was optional if feasible; it is omitted because the pinned image's ordinary-Radix path crashed on the first Gemma4 forward. The exact receipt, launch command, and full crash log are preserved under `packet/feasibility/`.

The complete claim, paired results, action-counter proof, hot-set survival/recovery metrics, and limitations are in `SCIENTIFIC_REPORT.json`; request-by-request conformance is in `KUSHAL_REQUEST_CONFORMANCE.json`.

Verify after extraction with:


```bash
PYTHONDONTWRITEBYTECODE=1 python3 packet/verify_extension.py all results --packet packet
```
