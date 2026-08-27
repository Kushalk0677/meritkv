import json, os
from collections import defaultdict

base = r'C:\Users\kusha\Documents\GitHub\shadowkv\results'

# Per-dataset + mode + model stats for ShadowKV++
data = defaultdict(lambda: {'speedups':[],'hit_rates':[],'wastes':[],'ch_tokens':[],'prompt_tokens':[],'reuse_attempts':[],'bypasses':[]})

for root_dir in ['final_p100','final_t4']:
    for model_dir in os.listdir(os.path.join(base, root_dir)):
        mpath = os.path.join(base, root_dir, model_dir)
        if not os.path.isdir(mpath): continue
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
                        # Get config for prompt token info
                        fp = os.path.join(dspath,f)
                        if os.path.getsize(fp) < 100: continue
                        try:
                            with open(fp) as fh: doc = json.load(fh)
                        except: continue
                        config = doc.get('config',{})
                        pcount = config.get('n_requests',64)
                        no_cache = doc.get('no_cache',{})
                        nc_tokens = 0
                        if isinstance(no_cache, dict):
                            nc_tokens = no_cache.get('recompute_tokens_total',0) or 0
                        prompt_tokens_per = nc_tokens / max(pcount,1)
                        
                        skv = doc.get('shadow_kv_plus',{})
                        if not isinstance(skv, dict): continue
                        spd = skv.get('speedup_vs_no_cache_mean')
                        hr = skv.get('hit_rate')
                        wst = skv.get('wasted_compute_ratio')
                        ct = skv.get('reused_prefix_tokens_total',0) or 0
                        rap = skv.get('reuse_attempts',0) or 0
                        bp = skv.get('bypassed_matches',0) or 0
                        key = (ds_dir, mode_dir)
                        data[key]['speedups'].append(spd)
                        data[key]['hit_rates'].append(hr)
                        data[key]['wastes'].append(wst)
                        data[key]['prompt_tokens'].append(prompt_tokens_per)
                        data[key]['ch_tokens'].append(ct)
                        data[key]['reuse_attempts'].append(rap)
                        data[key]['bypasses'].append(bp)

print('=== Per-dataset characteristics for ShadowKV++ ===')
print(f'{"Dataset":15s} {"Mode":12s} {"Spd":>6s} {"Hit":>6s} {"Waste":>6s} {"Reuse":>6s} {"Bypass":>7s} {"Tokens":>7s}')
print('-'*70)
for (ds, mode) in sorted(data.keys()):
    d = data[(ds, mode)]
    n = len(d['speedups'])
    ms = sum(d['speedups'])/n
    mh = sum(d['hit_rates'])/n
    mw = sum(d['wastes'])/n
    mr = sum(d['reuse_attempts'])/n
    mb = sum(d['bypasses'])/n
    mt = sum(d['prompt_tokens'])/n
    print(f'{ds:15s} {mode:12s} {ms:6.3f} {mh:6.3f} {mw:6.3f} {mr:6.1f} {mb:7.1f} {mt:7.0f} n={n}')
