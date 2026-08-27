import json, os
base = r'C:\Users\kusha\Documents\GitHub\shadowkv\results'
out = []
for root in ['final_p100','final_t4']:
    for model in os.listdir(os.path.join(base, root)):
        if 'Phi-3' not in model: continue
        for mode in os.listdir(os.path.join(base, root, model)):
            for seed in os.listdir(os.path.join(base, root, model, mode)):
                for ds in os.listdir(os.path.join(base, root, model, mode, seed)):
                    for f in os.listdir(os.path.join(base, root, model, mode, seed, ds)):
                        if not f.endswith('.json'): continue
                        fp = os.path.join(base, root, model, mode, seed, ds, f)
                        try:
                            with open(fp) as fh: d = json.load(fh)
                        except: continue
                        skv = d.get('shadow_kv_plus', {})
                        if not isinstance(skv, dict): continue
                        spd = skv.get('speedup_vs_no_cache_mean', 1)
                        if spd < 0.97:
                            wst = skv.get('wasted_compute_ratio', 0)
                            rap = skv.get('reuse_attempts', 0)
                            bp = skv.get('bypassed_matches', 0)
                            wcm = skv.get('wasted_compute_ms', 0)
                            policy_exact = skv.get('policy_exact_total', 0)
                            policy_sem = skv.get('policy_semantic_partial_total', 0)
                            policy_bp = skv.get('policy_bypass_total', 0)
                            out.append(f'spd={spd:.3f} waste={wst:.3f} rap={rap} bp={bp} wcm={wcm} policy_ex={policy_exact} sem={policy_sem} bypass={policy_bp} {root}/{mode}/{seed}/{ds}')
for line in out:
    print(line)
if not out:
    print('No Phi-3 runs with speedup < 0.97 found')
print(f'Total low-speedup Phi-3 runs: {len(out)}')
