#!/usr/bin/env python3
"""Evaluate all fidelity results."""
import json, sys
from collections import defaultdict

sys.path.insert(0, 'v10/experiments')
from eval_fidelity import compute_rouge_l, compute_exact_match

results_dir = 'v10/experiments/fidelity_results'
model_keys = ['gpt2', 'tinyllama', 'qwen25_15b', 'gemma2b', 'phi3mini']

all_data = []
for mk in model_keys:
    try:
        with open(f'{results_dir}/{mk}_results.json') as f:
            data = json.load(f)
    except:
        print(f'{mk}: NOT FOUND')
        continue
    for r in data:
        r['rougeL'] = compute_rouge_l(r.get('exact_text',''), r.get('approx_text',''))
        r['exact_match'] = compute_exact_match(r.get('exact_text',''), r.get('approx_text',''))
        r['model'] = mk
    all_data.extend(data)
    print(f'{mk:15s}: {len(data):4d} samples')

print()
print(f'Total: {len(all_data)} samples')
print()

# Per-model + per-dataset
groups = defaultdict(list)
for r in all_data:
    groups[(r['model'], r['dataset'])].append(r)

print(f'{"Model":15s} {"Dataset":15s} {"n":>4s} {"ROUGE-L":>7s} {"EM":>5s}')
print('-'*50)
for key in sorted(groups.keys()):
    g = groups[key]
    avg_r = sum(x['rougeL']['rougeL_fmeasure'] for x in g)/len(g)
    avg_e = sum(x['exact_match'] for x in g)/len(g)
    print(f'{key[0]:15s} {key[1]:15s} {len(g):>4d} {avg_r:>7.4f} {avg_e:>5.3f}')

# Overall per model
print()
print(f'{"Model":15s} {"n":>4s} {"ROUGE-L":>7s} {"EM":>5s}')
print('-'*35)
for mk in model_keys:
    g = [r for r in all_data if r['model'] == mk]
    if not g: continue
    avg_r = sum(x['rougeL']['rougeL_fmeasure'] for x in g)/len(g)
    avg_e = sum(x['exact_match'] for x in g)/len(g)
    print(f'{mk:15s} {len(g):>4d} {avg_r:>7.4f} {avg_e:>5.3f}')
