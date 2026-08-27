#!/usr/bin/env python3
"""Verify the corrected breakeven guard blocks Phi-3 48-token speculation."""
kappa_mb = 0.375
memory_bw_gbps = 320.0
delta_r = 11.51  # Phi-3 from paper Table II
beta = 1.198      # Phi-3 from paper Table II

kv_cost = (kappa_mb * 1000.0) / memory_bw_gbps
denom = max(beta - kv_cost, 0.001)
breakeven_len = int(delta_r / denom)

print(f'Phi-3 on T4:')
print(f'  kappa = {kappa_mb} MB/token')
print(f'  bandwidth = {memory_bw_gbps} GB/s')
print(f'  KV transfer cost = {kv_cost:.3f} ms/token')
print(f'  Beta = {beta} ms/token')
print(f'  Net benefit = {beta - kv_cost:.3f} ms/token')
print(f'  Delta_r = {delta_r} ms')
print(f'  Breakeven k* = {breakeven_len} tokens')
print(f'  Candidate = 48 tokens')
print(f'  Blocked = {48 < breakeven_len}')
print()

# GPT-2 on T4
kappa_gpt2 = 0.0352
delta_gpt2 = 2.03
beta_gpt2 = 0.6
kv_gpt2 = (kappa_gpt2 * 1000.0) / memory_bw_gbps
denom_gpt2 = max(beta_gpt2 - kv_gpt2, 0.001)
bk_gpt2 = int(delta_gpt2 / denom_gpt2)
print(f'GPT-2 on T4:')
print(f'  KV transfer cost = {kv_gpt2:.3f}')
print(f'  Net benefit = {beta_gpt2 - kv_gpt2:.3f}')
print(f'  Breakeven k* = {bk_gpt2} tokens')
print(f'  Candidate = 48 tokens')
print(f'  Blocked = {48 < bk_gpt2}')
