#!/usr/bin/env python3
"""
Verify the memory-breakeven guard blocks the Phi-3 48-token speculation.
"""
import sys, os
sys.path.insert(0, 'v10/src')

# Simulate the breakeven calculation from the engines.py code
kappa_bytes_per_token = 0.375 * 1024 * 1024  # Phi-3: 0.375 MB/token in bytes
memory_bw_gbps = 320.0  # T4 bandwidth

kv_transfer_per_token_ms = kappa_bytes_per_token / (memory_bw_gbps * 1e9 / 1000)
beta = 1.198  # Phi-3 beta from paper Table II
decision_cost_ms = 47.5  # with correct kappa

breakeven = max(int(decision_cost_ms / max(beta - kv_transfer_per_token_ms, 0.01)), 0)

print(f'Phi-3 on T4 (kappa=0.375 MB/token, bandwidth=320 GB/s):')
print(f'  KV transfer per token: {kv_transfer_per_token_ms:.3f} ms')
print(f'  Beta (prefill per token): {beta:.3f} ms')
print(f'  Net benefit per token: {beta - kv_transfer_per_token_ms:.3f} ms')
print(f'  Decision cost: {decision_cost_ms:.1f} ms')
print(f'  Breakeven length k*: {breakeven} tokens')
print(f'  Candidate prefix: 48 tokens')
print(f'  Blocked by guard: {48 < breakeven}')
print()

# Also check that the guard doesn't block reasonable prefixes
# For a 200-token prefix with the same model
print(f'Candidate prefix: 200 tokens')
print(f'  Blocked by guard: {200 < breakeven}')
print()

# Check GPT-2 on T4 (kappa=0.0352 MB/token, beta=0.6)
kappa_gpt2 = 0.0352 * 1024 * 1024
kv_transfer_gpt2 = kappa_gpt2 / (memory_bw_gbps * 1e9 / 1000)
beta_gpt2 = 0.6
cost_gpt2 = 31.4  # old default cost for 48-token prefix
breakeven_gpt2 = max(int(cost_gpt2 / max(beta_gpt2 - kv_transfer_gpt2, 0.01)), 0)
print(f'GPT-2 on T4 (kappa=0.0352 MB/token, bandwidth=320 GB/s):')
print(f'  KV transfer per token: {kv_transfer_gpt2:.3f} ms')
print(f'  Net benefit per token: {beta_gpt2 - kv_transfer_gpt2:.3f} ms')
print(f'  Breakeven length k*: {breakeven_gpt2} tokens')
print(f'  Candidate prefix: 48 tokens')
print(f'  Blocked by guard: {48 < breakeven_gpt2}')
