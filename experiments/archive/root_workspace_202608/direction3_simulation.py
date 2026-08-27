#!/usr/bin/env python3
"""
Direction 3: Coupled Utility Model — Offline Simulation

Simulates the old (linear) and new (coupled) utility across all 898 benchmark
JSONs, comparing admission decisions and identifying which runs flip from
admit to bypass under the risk-aversion penalty.
"""

import json, os, sys, math
from collections import defaultdict

base = r'C:\Users\kusha\Documents\GitHub\shadowkv\results'

# Model parameter counts (for coupling calculation)
MODEL_PARAMS = {
    'gpt2': 0.124, 'TinyLlama_TinyLlama-1_1B-Chat-v1_0': 1.1,
    'Qwen_Qwen2_5-1_5B-Instruct': 1.54, 'google_gemma-2b-it': 2.0,
    'microsoft_Phi-3-mini-4k-instruct': 3.8
}
MODEL_KAPPA = {
    'gpt2': 0.0352, 'TinyLlama_TinyLlama-1_1B-Chat-v1_0': 0.0215,
    'Qwen_Qwen2_5-1_5B-Instruct': 0.0273, 'google_gemma-2b-it': 0.0176,
    'microsoft_Phi-3-mini-4k-instruct': 0.375
}

LAMBDA_RISK = 0.15  
BETA_DEFAULT = 0.6
MS_PER_TOKEN_DEFAULT = 1.2

results = []

for root_dir in ['final_p100','final_t4']:
    for model_dir in os.listdir(os.path.join(base, root_dir)):
        mpath = os.path.join(base, root_dir, model_dir)
        if not os.path.isdir(mpath): continue
        
        model_params = MODEL_PARAMS.get(model_dir, 1.0)
        kappa = MODEL_KAPPA.get(model_dir, 0.02)
        beta = BETA_DEFAULT + 0.15 * model_params  
        
        for mode_dir in os.listdir(mpath):
            modepath = os.path.join(mpath, mode_dir)
            if not os.path.isdir(modepath): continue
            for seed_dir in os.listdir(modepath):
                spath = os.path.join(modepath, seed_dir)
                if not os.path.isdir(spath): continue
                for ds_dir in os.listdir(spath):
                    dspath = os.path.join(spath, ds_dir)
                    if not os.path.isdir(dspath): continue
                    for f in os.listdir(dspath):
                        if not f.endswith('.json'): continue
                        fp = os.path.join(dspath,f)
                        if os.path.getsize(fp) < 100: continue
                        try:
                            with open(fp) as fh: d = json.load(fh)
                        except: continue
                        
                        skv = d.get('shadow_kv_plus', {})
                        nc = d.get('no_cache', {})
                        cfg = d.get('config', {})
                        
                        if not isinstance(skv, dict) or not isinstance(nc, dict):
                            continue
                        
                        # Extract metrics
                        nc_lat = nc.get('mean_latency_ms', 0)
                        skv_lat = skv.get('mean_latency_ms', 0)
                        spd = skv.get('speedup_vs_no_cache_mean', 1.0)
                        waste = skv.get('wasted_compute_ratio', 0)
                        rap = skv.get('reuse_attempts', 0) or 0
                        bp = skv.get('bypassed_matches', 0) or 0
                        hit = skv.get('hit_rate', 0)
                        rct = skv.get('recompute_tokens_total', 0) or 0
                        ret = skv.get('reused_prefix_tokens_total', 0) or 0
                        policy_plans = skv.get('policy_plans_total', 0) or 0
                        policy_bypass = skv.get('policy_bypass_total', 0) or 0
                        policy_exact = skv.get('policy_exact_total', 0) or 0
                        wasted_ms = skv.get('wasted_compute_ms', 0) or 0
                        useful_ms = skv.get('useful_speculative_savings_ms', 0) or 0
                        
                        if nc_lat <= 0 or skv_lat <= 0:
                            continue
                        
                        # --- Simulate old utility ---
                        # Benefit: saved prefill time
                        B = ret * max(beta, 0.05)
                        # Cost: reuse overhead
                        C = 2.0 + 0.02 * rct  
                        # Waste: proportional to cost
                        W = C * max(waste, 0.02)
                        # Memory cost
                        memory_mb = (rct + ret) * kappa
                        lambda_m = 0.02
                        M = lambda_m * memory_mb
                        
                        score_old = B - C - W - M
                        admitted_old = score_old >= 0
                        
                        # --- Simulate new coupled utility ---
                        vram_factor = memory_mb
                        coupling = LAMBDA_RISK * B * vram_factor * max(waste, 0.02)
                        score_new = B - C - W - M - coupling
                        admitted_new = score_new >= 0
                        
                        # Determine if this run flips
                        flip = 'none'
                        if admitted_old and not admitted_new:
                            flip = 'admit_to_bypass'
                        elif not admitted_old and admitted_new:
                            flip = 'bypass_to_admit'
                        
                        key = f'{model_dir}/{mode_dir}/{seed_dir}/{ds_dir}'
                        results.append({
                            'key': key, 'model': model_dir, 'mode': mode_dir,
                            'seed': seed_dir, 'dataset': ds_dir,
                            'gpu': root_dir, 'spd': spd, 'waste': waste,
                            'B': B, 'C': C, 'W': W, 'M': M, 'coupling': coupling,
                            'score_old': score_old, 'score_new': score_new,
                            'admitted_old': admitted_old, 'admitted_new': admitted_new,
                            'flip': flip, 'rap': rap, 'bp': bp, 'hit': hit,
                            'nc_lat': nc_lat, 'skv_lat': skv_lat,
                        })

# Analyze
print(f'Total runs analyzed: {len(results)}')
print()

# Overall flip statistics
flips = [r for r in results if r['flip'] != 'none']
print(f'Total flips: {len(flips)}')
by_model = defaultdict(list)
for r in flips:
    by_model[r['model']].append(r)
for m, runs in sorted(by_model.items()):
    print(f'  {m:50s}: {len(runs)} flips')

print()

# Phii-3 analysis
phi3_results = [r for r in results if 'Phi-3' in r['model']]
phi3_flips = [r for r in phi3_results if r['flip'] == 'admit_to_bypass']
print(f'Phi-3 total: {len(phi3_results)}')
print(f'Phi-3 flips (admit->bypass): {len(phi3_flips)}')
if phi3_flips:
    worst = sorted(phi3_flips, key=lambda r: r['spd'])[:10]
    print('Worst 10 flipped Phi-3 runs by speedup:')
    for r in worst:
        print(f'  spd={r["spd"]:.3f} waste={r["waste"]:.3f} B={r["B"]:.1f} coupling={r["coupling"]:.1f} old={r["score_old"]:.1f} new={r["score_new"]:.1f} {r["mode"]}/{r["seed"]}/{r["dataset"]}')

# Overall impact on average speedup
old_admit_rate = sum(1 for r in results if r['admitted_old']) / len(results) * 100
new_admit_rate = sum(1 for r in results if r['admitted_new']) / len(results) * 100
print(f'\nOld admit rate: {old_admit_rate:.1f}%')
print(f'New admit rate: {new_admit_rate:.1f}%')

# Speedup impact on flipped runs
if phi3_flips:
    avg_old_spd = sum(r['spd'] for r in phi3_results) / len(phi3_results)
    avg_flip_spd = sum(r['spd'] for r in phi3_flips) / len(phi3_flips) if phi3_flips else 0
    print(f'\nPhi-3 average speedup (all): {avg_old_spd:.3f}')
    print(f'Phi-3 average speedup (flipped): {avg_flip_spd:.3f}')

# Covariance of B, W across all runs
B_vals = [r['B'] for r in results]
W_vals = [r['W'] for r in results]
n = len(B_vals)
mean_B = sum(B_vals)/n
mean_W = sum(W_vals)/n
cov_BW = sum((B_vals[i] - mean_B) * (W_vals[i] - mean_W) for i in range(n)) / n
var_B = sum((b - mean_B)**2 for b in B_vals) / n
var_W = sum((w - mean_W)**2 for w in W_vals) / n
print(f'\nCovariance(B, W): {cov_BW:.3f}')
print(f'Correlation(B, W): {cov_BW/math.sqrt(var_B*var_W):.3f}' if var_B*var_W > 0 else 'Correlation: N/A')

# By-model coupling strength
print('\nBy-model coupling (avg coupling_penalty / B):')
for m in sorted(set(r['model'] for r in results)):
    m_runs = [r for r in results if r['model'] == m]
    avg_ratio = sum(r['coupling']/max(r['B'], 0.001) for r in m_runs) / len(m_runs)
    avg_B = sum(r['B'] for r in m_runs) / len(m_runs)
    avg_coupling = sum(r['coupling'] for r in m_runs) / len(m_runs)
    print(f'  {m:50s}: coupling/B ratio={avg_ratio:.4f} avg_B={avg_B:.1f} avg_coupling={avg_coupling:.1f} n={len(m_runs)}')
