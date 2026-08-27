import json, os
from collections import defaultdict

base = r'C:\Users\kusha\Documents\GitHub\shadowkv\results'
engine_labels = {
    'no_cache':'No Cache','reactive_prefix_cache':'Reactive','greedy_prefix_cache':'Greedy',
    'strict_reactive_prefix_cache':'Strict Reactive','frequency_speculative':'Freq Speculative',
    'shadow_kv':'ShadowKV','shadow_kv_plus':'ShadowKV++',
    'shadow_kv_plus_best_latency':'Best Latency','shadow_kv_plus_raw_observer':'Raw Observer'
}

# Per-engine aggregates for waste and reuse data
eng_data = defaultdict(lambda: {'waste':[],'reuse_attempts':[],'waste_cond':[],'bypass':[],'store':[],'latency':[],'hit_rate':[]})

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
                        fp = os.path.join(dspath,f)
                        if os.path.getsize(fp) < 100: continue
                        try:
                            with open(fp) as fh: d = json.load(fh)
                        except: continue
                        for ek, el in engine_labels.items():
                            eng = d.get(ek,{})
                            if not isinstance(eng, dict): continue
                            wst = eng.get('wasted_compute_ratio')
                            rap = eng.get('reuse_attempts',0)
                            wcm = eng.get('wasted_compute_ms',0)
                            bp = eng.get('bypassed_matches',0)
                            st = eng.get('store_successes',0)
                            lt = eng.get('mean_latency_ms')
                            hr = eng.get('hit_rate')
                            if wst is not None:
                                eng_data[ek]['waste'].append(wst)
                                eng_data[ek]['reuse_attempts'].append(rap)
                                eng_data[ek]['bypass'].append(bp)
                                eng_data[ek]['store'].append(st)
                                if lt: eng_data[ek]['latency'].append(lt)
                                if hr is not None: eng_data[ek]['hit_rate'].append(hr)
                                # Waste conditional on any attempt
                                wc = wcm / max(rap, 1)
                                eng_data[ek]['waste_cond'].append(wc)

print('=== Waste Analysis ===')
print(f'{"Engine":30s} {"Waste":>8s} {"WasteCond":>10s} {"ReuseAtt":>9s} {"Bypass":>7s} {"Store":>6s} {"HitRate":>8s}')
print('-'*80)
for ek in ['no_cache','reactive_prefix_cache','greedy_prefix_cache','strict_reactive_prefix_cache',
           'frequency_speculative','shadow_kv','shadow_kv_plus','shadow_kv_plus_best_latency','shadow_kv_plus_raw_observer']:
    d = eng_data[ek]
    n = len(d['waste'])
    if n == 0: continue
    mw = sum(d['waste'])/n
    mwc = sum(d['waste_cond'])/n
    mra = sum(d['reuse_attempts'])/n
    mbp = sum(d['bypass'])/n
    mst = sum(d['store'])/n
    mhr = sum(d['hit_rate'])/n
    print(f'{engine_labels[ek]:30s} {mw:8.3f} {mwc:10.3f} {mra:9.1f} {mbp:7.1f} {mst:6.1f} {mhr:8.3f} n={n}')
