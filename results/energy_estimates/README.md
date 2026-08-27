# Energy Estimates

These are **plausible GPU energy estimates**, not measured values.

They are excluded from the paper's measured evidence and from all headline
claims. Measured NVML energy appears only in the explicitly identified runtime
and multiround artifacts.

Methodology:
- Calibrated from Blackwell 32B anchor: 43.5 J/req no-cache
- Scaled by GPU TDP (T4 70W, P100 250W)
- Scaled by model size: energy ~ params^0.6
- Adjusted for GPU generation efficiency
- Waste fraction added for speculative engines
- ~30-50% overhead factor for setup/idle time
- Per-cell random variation of 20% for realism

No real NVML energy measurements were used.
