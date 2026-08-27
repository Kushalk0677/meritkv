import json, os, sys
from collections import defaultdict

base = r'C:\Users\kusha\Documents\GitHub\shadowkv\results'
model_names = {'gpt2':'GPT-2','TinyLlama_TinyLlama-1_1B-Chat-v1_0':'TinyLlama','Qwen_Qwen2_5-1_5B-Instruct':'Qwen2.5','google_gemma-2b-it':'Gemma-2B','microsoft_Phi-3-mini-4k-instruct':'Phi-3'}
model_data = defaultdict(lambda: {'speedups':[],'wastes':[],'hit_rates':[]})

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
                        if f.endswith('.json'):
                            fp = os.path.join(dspath,f)
                            if os.path.getsize(fp) < 100: continue
                            try:
                                with open(fp) as fh: d = json.load(fh)
                            except: continue
                            for engine_key in ['shadow_kv_plus']:
                                eng = d.get(engine_key)
                                if eng and isinstance(eng, dict):
                                    spd = eng.get('speedup_vs_no_cache_mean')
                                    wst = eng.get('wasted_compute_ratio')
                                    hit = eng.get('hit_rate')
                                    if spd and spd > 0:
                                        model_data[model_dir]['speedups'].append(spd)
                                        if wst is not None: model_data[model_dir]['wastes'].append(wst)
                                        if hit is not None: model_data[model_dir]['hit_rates'].append(hit)

out = ['Per-model ShadowKV++ aggregates across all JSON files:']
for mdir in ['gpt2','TinyLlama_TinyLlama-1_1B-Chat-v1_0','Qwen_Qwen2_5-1_5B-Instruct','google_gemma-2b-it','microsoft_Phi-3-mini-4k-instruct']:
    d = model_data[mdir]
    n = len(d['speedups'])
    if n > 0:
        ms = sum(d['speedups'])/n
        mw = sum(d['wastes'])/n if d['wastes'] else 0
        mh = sum(d['hit_rates'])/n if d['hit_rates'] else 0
        out.append(f'{model_names[mdir]:15s} spd={ms:.3f} waste={mw:.3f} hit={mh:.3f} n={n}')
    else:
        out.append(f'{model_names[mdir]:15s} NO DATA FOUND')

out.append(f'')
out.append(f'Paper per-model table values (current):')
out.append(f'GPT-2:   1.533, waste 0.350')
out.append(f'Gemma:   1.475, waste 0.114')
out.append(f'Phi-3:   1.338, waste 0.125')
out.append(f'TinyLlama: 1.275, waste 0.106')
out.append(f'Qwen2.5:  1.204, waste 0.083')

with open(r'C:\shadowkv\per_model_verify.txt', 'w') as f:
    f.write('\n'.join(out))
print('Written to per_model_verify.txt')
