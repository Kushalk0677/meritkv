# Four-Arm Multi-Round Admission Enforcement Experiment

## Goal

Test the causal link:
**controller decision → skipped cache operation → fewer evictions → better latency**

## Design

Four arms, multi-round trace, vLLM backend on Blackwell RTX PRO 6000.

### Arms

| # | Arm | Engine class | Admission | Write enforcement | Status |
|---|-----|-------------|-----------|-------------------|--------|
| 1 | No cache | `NoCacheEngine` | Disabled | N/A | Exists |
| 2 | Native APC | `RuntimeNativeCacheEngine` | Disabled | Always write | Exists |
| 3 | Overlay | `AdmissionControlledRuntimeCacheEngine` | U = B−C−W (logged) | Write-through | Exists |
| 4 | Enforced | **New**: `EnforcedAdmissionEngine` | U = B−C−W (enforced) | Skip-write + skip-lookup | **Needs implementation** |

### What the enforced arm must do

`EnforcedAdmissionEngine` extends `AdmissionControlledRuntimeCacheEngine` with one change:
when the gate decides `bypass`, it must **not call the backend's cache write/lookup**.

In `process_request()`:

```
if plan.strategy == 'exact' and plan.score >= 0.0:
    # Allow: normal path, cache lookup + write
    result = super().process_request(request)
else:
    # Bypass: skip cache entirely
    result = backend.generate(request)  # no cache involvement
    # Record bypass counters
    admission_native_skip_lookup_total += 1
    admission_native_skip_write_total += 1
```

### Trace

Multi-round: `20 fill + [15 churn + 15 recovery] × 4 rounds = 140 requests`.

Prefix structure:
- Fill: 20 requests, 220-token shared prefix + 50-token unique suffix
- Churn (each round): 15 requests with a new shared prefix (evictors) interleaved with 15 requests reusing the fill prefix (victims)
- Recovery (each round): 15 requests reusing the fill prefix

Pressure is set by the `max_memory_mb` parameter. Two levels:
- **High pressure**: `max_memory_mb=128` (tight cache budget)
- **Moderate pressure**: `max_memory_mb=256` (relaxed budget)

### Required implementation

**File**: `src/proactive_kv_cache/engines.py` — add class:

```python
class EnforcedAdmissionEngine(AdmissionControlledRuntimeCacheEngine):
    """Admission-controlled engine that enforces bypass (skips cache write/lookup)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_metrics['admission_native_skip_lookup_total'] = 0
        self.engine_metrics['admission_native_skip_write_total'] = 0

    def process_request(self, request: TokenizedRequest) -> GenerationResult:
        plan = self.controller.plan(request)
        self._record_admission_plan(plan)

        allowed = plan.strategy == 'exact' and plan.score >= 0.0

        if allowed:
            return super().process_request(request)
        else:
            self.engine_metrics['admission_native_skip_lookup_total'] += 1
            self.engine_metrics['admission_native_skip_write_total'] += 1
            return self.backend.generate(request)
```

### Run command

```bash
# High pressure (128 MB cache)
python experiments/run_four_arm_trace.py \
    --backend vllm \
    --model google/gemma-4-31B-it \
    --max_memory_mb 128 \
    --n_phase1 20 --n_phase2 15 --n_phase3 15 --n_rounds 4 \
    --output_dir results/four_arm_trace_high_pressure

# Moderate pressure (256 MB cache)
python experiments/run_four_arm_trace.py \
    --backend vllm \
    --model google/gemma-4-31B-it \
    --max_memory_mb 256 \
    --n_phase1 20 --n_phase2 15 --n_phase3 15 --n_rounds 4 \
    --output_dir results/four_arm_trace_moderate_pressure
```

### Analysis

For each arm, extract:

| Metric | Source |
|--------|--------|
| Mean latency | `engine.metrics['mean_latency_ms']` |
| Per-phase latency | Per-request timestamps in trace ledger |
| Phase 3 recovery rate | Fraction of recovery requests that hit cache |
| Evictions | `engine.metrics['evictions']` or cache delta |
| Bypass count | `engine_metrics['admission_bypass_total']` |
| Skip-write count | `engine_metrics['admission_native_skip_write_total']` |

### Predicted outcome (hypothesis, not result)

| Contrast | Expected sign | Bounding argument |
|----------|--------------|-------------------|
| Overlay − Native | ≈ 0 or slightly positive | Gate evaluation adds ∼0.3% overhead |
| Enforced − Overlay | Negative | Skipping writes reduces eviction pressure |
| Enforced − Native | Negative (smaller than enforced−overlay) | Net benefit after subtracting gate cost |

The **overlay arm** is the critical control: same cache state as native, same gate decisions as enforced. If enforced differs from overlay, the difference is purely from skipping writes — no other mechanism.

### Existing evidence (for context, not substitution)

- **Memory-bound trace** (`results/memory_bound_trace/`): shows 36−40 pp recovery improvement for enforced MeritKV vs native under single-round high pressure. Lacks overlay arm.
- **Qwen vLLM overlay** (`runtime_experiments/qwen2.5/vllm/`): shows 255/1 allow/bypass ratio. Gate rarely bypasses at runtime pressure levels.
