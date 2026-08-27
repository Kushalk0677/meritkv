import json, os, sys
from collections import defaultdict

base = r'C:\Users\kusha\Documents\GitHub\shadowkv\results'

# Check Phi-3 failure runs
fails = []
for root_dir in ['final_p100','final_t4']:
    for model_dir in os.listdir(os.path.join(base, root_dir)):
        if 'Phi-3' not in model_dir: continue
        mpath = os.path.join(base, root_dir, model_dir)
        for mode_dir in os.listdir(mpath):
            for seed_dir in os.listdir(os.path.join(mpath, mode_dir)):
                for ds_dir in os.listdir(os.path.join(mpath, mode_dir, seed_dir)):
                    fp = os.path.join(mpath, mode_dir, seed_dir, ds_dir)
                    for f in os.listdir(fp):
                        if not f.endswith('.json'): continue
                        try:
                            with open(os.path.join(fp,f)) as fh: d = json.load(fh)
                        except: continue
                        skv = d.get('shadow_kv_plus',{})
                        if not isinstance(skv, dict): continue
                        spd = skv.get('speedup_vs_no_cache_mean', 1.0)
                        wst = skv.get('wasted_compute_ratio', 0)
                        rap = skv.get('reuse_attempts', 0)
                        bp = skv.get('bypassed_matches', 0)
                        rct = skv.get('recompute_tokens_total', 0)
                        hit = skv.get('hit_rate', 0)
                        key = f'{model_dir}/{mode_dir}/{seed_dir}/{ds_dir}'
                        fails.append((spd, wst, rap, bp, rct, hit, key))

fails.sort()
print('Phi-3 worst runs by speedup (all):')
for spd, wst, rap, bp, rct, hit, key in fails[:20]:
    print(f'  spd={spd:.3f} waste={wst:.3f} reuse_attempts={rap} bypass={bp} hit={hit:.3f} {key}')
print(f'  Total Phi-3 runs: {len(fails)}')
avg_spd = sum(f[0] for f in fails)/len(fails)
avg_wst = sum(f[1] for f in fails)/len(fails)
print(f'  Average speedup: {avg_spd:.3f}')
print(f'  Average waste: {avg_wst:.3f}')
