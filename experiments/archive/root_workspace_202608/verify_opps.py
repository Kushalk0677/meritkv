import json, os
from collections import defaultdict

base = r'C:\Users\kusha\Documents\GitHub\shadowkv\results'
opp_data = defaultdict(int)
miss_data = defaultdict(int)

for root_dir in ['final_p100','final_t4']:
    for model_dir in os.listdir(os.path.join(base, root_dir)):
        mpath = os.path.join(base, root_dir, model_dir)
        if not os.path.isdir(mpath): continue
        for mode_dir in os.listdir(mpath):
            if mode_dir != 'semantic': continue
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
                        skv = d.get('shadow_kv_plus',{})
                        if isinstance(skv, dict):
                            opp = skv.get('semantic_opportunity_plans_total',0) or 0
                            opp_data[model_dir] += int(opp)

# Also check the ablation table opportunity numbers
# scaffold-only, early-layer, logit-guarded should have opp counts
abl_opp = defaultdict(int)
for root_dir in ['final_p100','final_t4']:
    for model_dir in os.listdir(os.path.join(base, root_dir)):
        mpath = os.path.join(base, root_dir, model_dir)
        if not os.path.isdir(mpath): continue
        for mode_dir in os.listdir(mpath):
            if mode_dir != 'semantic': continue
            modepath = os.path.join(mpath, mode_dir)
            for seed_dir in os.listdir(modepath):
                spath = os.path.join(modepath, seed_dir)
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
                        for ek in ['shadow_kv_plus_scaffold_only','shadow_kv_plus_early_layer','shadow_kv_plus_logit_guard']:
                            eng = d.get(ek,{})
                            if isinstance(eng, dict):
                                o = eng.get('semantic_opportunity_plans_total',0) or 0
                                abl_opp[ek] += int(o)

total = sum(opp_data.values())
model_total = sum(opp_data.values())
print(f'=== Semantic opportunities (shadow_kv_plus, semantic mode only) ===')
for m,v in sorted(opp_data.items()):
    print(f'  {m}: {v}')
print(f'Total: {total}')
print()
print(f'=== Ablation opportunity counts ===')
for k,v in sorted(abl_opp.items()):
    print(f'  {k}: {v}')
