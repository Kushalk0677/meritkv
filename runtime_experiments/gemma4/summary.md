# Gemma-4 Blackwell Runtime Compatibility

The current paper reports the measured mean latency difference of the
write-through MeritKV overlay versus each native cache backend. E4B is the 4B
variant used only in this overlay study.

| Backend | E2B | E4B | 12B | 26B-A4B | 31B | Mean |
|---|---:|---:|---:|---:|---:|---:|
| SGLang Radix | +0.8% | +1.0% | +1.5% | +1.2% | +2.1% | +1.3% |
| vLLM APC | +0.3% | +0.4% | +0.6% | +0.5% | +0.8% | +0.5% |
| LMCache | +0.2% | +0.3% | +0.5% | +0.4% | +0.6% | +0.4% |

Each model/backend value summarizes 50 matched cells: five datasets, two modes,
and five seeds. This gives 250 matched cells per runtime and engine arm. The
fixed-width bands produced by the runtime scripts are descriptive, not
confidence intervals.

The integration remains write-through. These values are compatibility and
decision-layer overhead evidence, not enforced MeritKV acceleration. Raw
per-cell files are retained for provenance.
