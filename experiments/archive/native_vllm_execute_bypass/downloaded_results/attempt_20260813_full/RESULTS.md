# Native vLLM Execute-or-Bypass Results

> This file is generated from measured request records. It must not be
> treated as a paper result until the run passes all validity checks.

## Configuration

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Seeds: `[42, 123, 456, 789, 999]`
- Prefix lengths: `[16, 32, 64, 128, 256, 512, 1024]`
- Requests per prefix length: `16`
- Output tokens: `8`
- Native actuator: stable versus unique per-request vLLM `cache_salt`.

## Arm summary

| Arm | Requests | Mean latency (ms) | P95 (ms) | Cached tokens/request | Cache-hit requests | MeritKV bypasses | Enforced bypasses |
|---|---:|---:|---:|---:|---:|---:|---:|
| forced_recompute | 560 | 322.011 | 542.473 | 0.00 | 0 | 0 | 0 |
| native_apc | 560 | 259.085 | 334.830 | 290.29 | 560 | 0 | 0 |
| meritkv_write_through | 560 | 258.141 | 333.978 | 290.29 | 560 | 400 | 0 |
| meritkv_enforced | 560 | 266.797 | 344.042 | 219.43 | 160 | 400 | 400 |

## Paired interpretation

- `enforced_vs_write_through`: mean paired-seed speedup `0.9677x` (95% CI `0.9500` to `0.9854`); mean latency change `+3.359%`.
- `enforced_vs_native_apc`: mean paired-seed speedup `0.9712x` (95% CI `0.9547` to `0.9877`); mean latency change `+2.981%`.
- `native_apc_vs_forced_recompute`: mean paired-seed speedup `1.2430x` (95% CI `1.2153` to `1.2706`); mean latency change `-19.527%`.

## Results by prefix length

| Prefix tokens | Forced recompute (ms) | Native APC (ms) | Write-through (ms) | Enforced (ms) | Enforced bypasses |
|---:|---:|---:|---:|---:|---:|
| 16 | 259.460 | 256.683 | 249.011 | 252.887 | 80 |
| 32 | 257.669 | 253.117 | 251.491 | 253.142 | 80 |
| 64 | 258.017 | 254.295 | 263.649 | 255.608 | 80 |
| 128 | 265.308 | 248.377 | 243.613 | 258.672 | 80 |
| 256 | 301.015 | 259.826 | 253.023 | 298.100 | 80 |
| 512 | 364.699 | 253.172 | 253.434 | 261.505 | 0 |
| 1024 | 547.907 | 288.124 | 292.768 | 287.666 | 0 |

## Output checks

- `native_apc`: token IDs 549/560; text 549/560.
- `meritkv_write_through`: token IDs 545/560; text 545/560.
- `meritkv_enforced`: token IDs 555/560; text 555/560.

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
