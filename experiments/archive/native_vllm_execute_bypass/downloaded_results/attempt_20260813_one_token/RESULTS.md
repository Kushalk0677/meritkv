# Native vLLM Execute-or-Bypass Results

> This file is generated from measured request records. It must not be
> treated as a paper result until the run passes all validity checks.

## Configuration

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Seeds: `[42, 123, 456, 789, 999]`
- Prefix lengths: `[16, 32, 64, 128, 256, 512, 1024]`
- Requests per prefix length: `16`
- Output tokens: `1`
- Native actuator: stable versus unique per-request vLLM `cache_salt`.

## Arm summary

| Arm | Requests | Mean latency (ms) | P95 (ms) | Cached tokens/request | Cache-hit requests | MeritKV bypasses | Enforced bypasses |
|---|---:|---:|---:|---:|---:|---:|---:|
| forced_recompute | 560 | 92.380 | 275.583 | 0.00 | 0 | 0 | 0 |
| native_apc | 560 | 41.267 | 52.655 | 290.29 | 560 | 0 | 0 |
| meritkv_write_through | 560 | 40.919 | 53.314 | 290.29 | 560 | 160 | 0 |
| meritkv_enforced | 560 | 40.658 | 52.114 | 283.43 | 400 | 160 | 160 |

## Paired interpretation

- `enforced_vs_write_through`: mean paired-seed speedup `1.0071x` (95% CI `0.9649` to `1.0493`); mean latency change `-0.616%`.
- `enforced_vs_native_apc`: mean paired-seed speedup `1.0153x` (95% CI `0.9630` to `1.0676`); mean latency change `-1.372%`.
- `native_apc_vs_forced_recompute`: mean paired-seed speedup `2.2434x` (95% CI `2.0580` to `2.4289`); mean latency change `-55.270%`.

## Results by prefix length

| Prefix tokens | Forced recompute (ms) | Native APC (ms) | Write-through (ms) | Enforced (ms) | Enforced bypasses |
|---:|---:|---:|---:|---:|---:|
| 16 | 39.679 | 41.332 | 39.569 | 38.884 | 80 |
| 32 | 40.807 | 39.408 | 40.574 | 41.162 | 80 |
| 64 | 42.427 | 41.162 | 39.309 | 39.044 | 0 |
| 128 | 46.049 | 41.577 | 40.056 | 40.619 | 0 |
| 256 | 73.769 | 40.051 | 39.713 | 41.121 | 0 |
| 512 | 130.194 | 40.711 | 41.216 | 39.866 | 0 |
| 1024 | 273.735 | 44.631 | 45.998 | 43.911 | 0 |

## Output checks

- `native_apc`: token IDs 558/560; text 558/560.
- `meritkv_write_through`: token IDs 558/560; text 558/560.
- `meritkv_enforced`: token IDs 559/560; text 559/560.

## Validity checks

- Actuator self-test passed: `True`
- All requested arms completed: `True`
- Enforced bypass telemetry consistent: `True`
- Deterministic output check passed: `False`

## Scope

This experiment isolates the request-time choice between consuming a
warmed native APC hit and forcing native recomputation. It does not test
storage admission or eviction, and it does not replace the separate
capacity-pressure experiment.
