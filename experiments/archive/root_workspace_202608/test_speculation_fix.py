#!/usr/bin/env python3
"""
Test whether the fixed speculation policy (with model-specific kappa)
blocks the Phi-3 failure that the default (kv_mb_per_token=0.0015) misses.
"""

import json

# The failed run: Phi-3, raw, CNN/DM, seed 123 on T4
# A 48-token speculative precompute was admitted, costing 292ms of waste
# The breakeven for Phi-3 on T4 is k* = 449 tokens

# Old policy (default kv_mb_per_token = 0.0015)
old_kappa = 0.0015
# New policy (uses Phi-3's actual kappa = 0.375)
new_kappa = 0.375

prefix_len = 48  # tokens speculated
memory_penalty_per_mb = 0.9
fixed_overhead = 5.0
ms_per_token = 1.2
idle_cost_fraction = 0.50

# Old cost estimate
old_memory_mb = max(prefix_len * old_kappa, 0.002)
old_prefill_cost = fixed_overhead + (prefix_len * ms_per_token)
old_memory_cost = old_memory_mb * memory_penalty_per_mb
old_cost = (old_prefill_cost * idle_cost_fraction + old_memory_cost)

# New cost estimate (with Phi-3's actual kappa)
new_memory_mb = max(prefix_len * new_kappa, 0.002)
new_prefill_cost = fixed_overhead + (prefix_len * ms_per_token)
new_memory_cost = new_memory_mb * memory_penalty_per_mb
new_cost = (new_prefill_cost * idle_cost_fraction + new_memory_cost)

print(f'=== Phi-3 48-token speculative precompute cost comparison ===')
print()
print(f'Old kv_mb_per_token: {old_kappa}')
print(f'Phi-3 actual kappa:  {new_kappa}')
print(f'Ratio:               {new_kappa/old_kappa:.0f}x')
print()
print(f'Memory estimate (old): {old_memory_mb:.3f} MB')
print(f'Memory estimate (new): {new_memory_mb:.1f} MB')
print(f'Prefill cost:          {old_prefill_cost:.1f} ms')
print(f'Memory cost (old):     {old_memory_cost:.3f}')
print(f'Memory cost (new):     {new_memory_cost:.1f}')
print(f'Total idle cost (old): {old_cost:.1f} ms')
print(f'Total idle cost (new): {new_cost:.1f} ms')
print(f'Cost ratio:            {new_cost/old_cost:.1f}x')
print()

# Check benefit_cost_ratio threshold (1.05)
# For the 48-token spec, what benefit would be needed?
for benefit in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    old_ratio = benefit / max(old_cost, 0.001)
    new_ratio = benefit / max(new_cost, 0.001)
    old_pass = old_ratio >= 1.05
    new_pass = new_ratio >= 1.05
    if old_pass or new_pass:
        print(f'  Benefit={benefit:3.0f}: old_ratio={old_ratio:.2f} pass={old_pass}  new_ratio={new_ratio:.2f} pass={new_pass}')

# What benefit would a 48-token prefix actually have?
# For CNN/DM raw mode, the prefix is a short unique prompt with no shared scaffold
# Typical benefit per token is beta ~ 1.2 ms/token
# So 48 tokens * 1.2 = 57.6 ms benefit
actual_benefit = prefix_len * ms_per_token
print(f'\nActual expected benefit for 48-token prefix: {actual_benefit:.1f} ms')
print(f'Old policy: benefit_cost_ratio = {actual_benefit/old_cost:.2f} (pass={actual_benefit/old_cost >= 1.05})')
print(f'Fixed policy: benefit_cost_ratio = {actual_benefit/new_cost:.2f} (pass={actual_benefit/new_cost >= 1.05})')
