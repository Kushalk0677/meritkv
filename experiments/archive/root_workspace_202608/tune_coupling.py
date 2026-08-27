#!/usr/bin/env python3
"""
Direction 3: Tune coupled utility parameters to find the best trade-off.
Target: block truly bad runs (speedup < 1.0, waste > 0.1) without blocking good runs.
"""

import json, os
from collections import defaultdict

base = r'C:\Users\kusha\Documents\GitHub\shadowkv\results'

MODEL_KAPPA = {
    'gpt2': 0.0352, 'TinyLlama_TinyLlama-1_1B-Chat-v1_0': 0.0215,
    'Qwen_Qwen2_5-1_5B-Instruct': 0.0273, 'google_gemma-2b-it': 0.0176,
    'microsoft_Phi-3-mini-4k-instruct': 0.375
}

all_runs = []
for root_dir in ['final_p100','final_t4']:
    for model_dir in os.listdir(os.path.join(base, root_dir)):
        mpath = os.path.join(base, root_dir, model_dir)
        if not os.path.isdir(mpath): continue
        kappa = MODEL_KAPPA.get(model_dir, 0.02)
        for mode_dir in os.listdir(mpath):
            for seed_dir in os.listdir(os.path.join(mpath, mode_dir)):
                for ds_dir in os.listdir(os.path.join(mpath, mode_dir, seed_dir)):
                    dspath = os.path.join(mpath, mode_dir, seed_dir, ds_dir)
                    for f in os.listdir(dspath):
                        if not f.endswith('.json'): continue
                        fp = os.path.join(dspath, f)
                        if os.path.getsize(fp) < 100: continue
                        try:
                            with open(fp) as fh: d = json.load(fh)
                        except: continue
                        skv = d.get('shadow_kv_plus', {})
                        nc = d.get('no_cache', {})
                        if not isinstance(skv, dict) or not isinstance(nc, dict): continue
                        nc_lat = nc.get('mean_latency_ms', 0)
                        if nc_lat <= 0: continue
                        spd = skv.get('speedup_vs_no_cache_mean', 1.0)
                        waste = skv.get('wasted_compute_ratio', 0)
                        rct = skv.get('recompute_tokens_total', 0) or 0
                        ret = skv.get('reused_prefix_tokens_total', 0) or 0
                        wasted_ms = skv.get('wasted_compute_ms', 0) or 0
                        rap = skv.get('reuse_attempts', 0) or 0
                        hit = skv.get('hit_rate', 0)
                        B = ret * 0.6  # benefit
                        C = 2.0 + 0.02 * rct  # cost
                        W = C * max(waste, 0.02)  # waste
                        memory_mb = (rct + ret) * kappa / 1024  
                        M = 0.02 * memory_mb
                        vram_mb_raw = (rct + ret) * kappa  
                        score_old = B - C - W - M
                        all_runs.append({
                            'model': model_dir, 'mode': mode_dir, 'seed': seed_dir,
                            'dataset': ds_dir, 'spd': spd, 'waste': waste,
                            'B': B, 'C': C, 'W': W, 'M': M,
                            'vram': vram_mb_raw, 'kappa': kappa,
                            'rap': rap, 'hit': hit, 'rct': rct, 'ret': ret,
                            'score_old': score_old, 'admitted': score_old >= 0,
                            'bad': spd < 1.0
                        })

phi3 = [r for r in all_runs if 'Phi-3' in r['model']]
bad_phi3 = [r for r in phi3 if r['bad']]
good_phi3 = [r for r in phi3 if not r['bad']]

print(f'Total runs: {len(all_runs)}')
print(f'Phi-3 total: {len(phi3)}, bad (spd<1.0): {len(bad_phi3)}, good: {len(good_phi3)}')
print()

# Test different coupling formulations
tests = []

# Formulation A: Original (lambda_risk * B * vram_mb * max(waste, floor))
for lr in [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
    blocked_bad = 0
    blocked_good = 0
    for r in phi3:
        cp = lr * r['B'] * max(r['vram'], 0) * max(r['waste'], 0.02)
        score = r['score_old'] - cp
        admitted = score >= 0
        if not admitted and r['admitted']:
            if r['bad']: blocked_bad += 1
            else: blocked_good += 1
    tests.append(('A: lambda_risk={:.2f}'.format(lr), blocked_bad, blocked_good, len(bad_phi3)))

# Formulation B: No waste floor (use actual waste only, min 0)
for lr in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    blocked_bad = 0
    blocked_good = 0
    for r in phi3:
        actual_waste = max(r['waste'], 0)  
        cp = lr * r['B'] * max(r['vram'], 0) * actual_waste
        score = r['score_old'] - cp
        admitted = score >= 0
        if not admitted and r['admitted']:
            if r['bad']: blocked_bad += 1
            else: blocked_good += 1
    tests.append(('B: actual_waste lambda={:.2f}'.format(lr), blocked_bad, blocked_good, len(bad_phi3)))

# Formulation C: Normalized by max kappa, actual waste
for lr in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    blocked_bad = 0
    blocked_good = 0
    for r in phi3:
        norm_vram = r['vram'] / 375.0  
        actual_waste = max(r['waste'], 0)
        cp = lr * r['B'] * norm_vram * actual_waste
        score = r['score_old'] - cp
        admitted = score >= 0
        if not admitted and r['admitted']:
            if r['bad']: blocked_bad += 1
            else: blocked_good += 1
    tests.append(('C: norm_vram lambda={:.1f}'.format(lr), blocked_bad, blocked_good, len(bad_phi3)))

# Formulation D: Cap coupling at 0.5*B, actual waste
for lr in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    blocked_bad = 0
    blocked_good = 0
    for r in phi3:
        actual_waste = max(r['waste'], 0)
        cp = lr * r['B'] * max(r['vram'], 0) * actual_waste
        cp = min(cp, 0.5 * r['B'])  
        score = r['score_old'] - cp
        admitted = score >= 0
        if not admitted and r['admitted']:
            if r['bad']: blocked_bad += 1
            else: blocked_good += 1
    tests.append(('D: capped50 lambda={:.2f}'.format(lr), blocked_bad, blocked_good, len(bad_phi3)))

# Print results sorted by best trade-off
# Best = high bad-blocked, low good-blocked
print(f'{"Formulation":45s} {"BadBlocked":>12s} {"GoodBlocked":>12s} {"BadTotal":>8s} {"Score":>6s}')
print('-'*85)
scored = []
for name, bb, bg, bt in tests:
    score = bb - 2*bg  # weight: blocking a good run is 2x worse than not blocking a bad one
    scored.append((score, name, bb, bg, bt))
scored.sort(reverse=True)

for s, name, bb, bg, bt in scored:
    print(f'{name:45s} {bb:>4d}/{bt:<2d} ({bb/max(bt,1)*100:5.1f}%) {bg:>4d}     {bg:>5.1f}%   {s:>6.1f}')

# Show the best candidate: which specific bad runs it blocks
best_params = ('formula_D', 0.08)
print(f'\n=== Detailed analysis of best candidate ===')
lr = 0.08
bad_blocked_detail = []
good_blocked_detail = []
for r in phi3:
    if not r['admitted']: continue  
    actual_waste = max(r['waste'], 0)
    cp = lr * r['B'] * max(r['vram'], 0) * actual_waste
    cp = min(cp, 0.5 * r['B'])
    score = r['score_old'] - cp
    admitted = score >= 0
    if not admitted:
        if r['bad']:
            bad_blocked_detail.append(r)
        else:
            good_blocked_detail.append(r)

print(f'Bad runs blocked by best candidate: {len(bad_blocked_detail)}')
for r in bad_blocked_detail:
    print(f'  spd={r["spd"]:.3f} waste={r["waste"]:.3f} {r["mode"]}/{r["seed"]}/{r["dataset"]}')
print(f'Good runs blocked: {len(good_blocked_detail)}')
for r in good_blocked_detail:
    print(f'  spd={r["spd"]:.3f} waste={r["waste"]:.3f} {r["mode"]}/{r["seed"]}/{r["dataset"]}')
