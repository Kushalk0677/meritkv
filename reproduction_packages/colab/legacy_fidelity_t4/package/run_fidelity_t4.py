#!/usr/bin/env python3
"""Standalone fidelity experiment runner for Colab T4.

Usage:
    python run_fidelity_t4.py --device cpu --models gpt2 phi3mini --n_samples 10
    python run_fidelity_t4.py --device cuda --models gpt2 --n_samples 32

Generates results matching the format in results/fidelity_examples/.
Float32 precision is guaranteed by --device cpu (HF loads in float32 on CPU).
"""
import argparse, json, os, sys, time, random, warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    'gpt2': 'gpt2',
    'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'qwen25_15b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'gemma2b': 'google/gemma-2b-it',
    'phi3mini': 'microsoft/Phi-3-mini-4k-instruct',
}

DATASETS = {
    'samsum': ('knkarthick/samsum', 'train', 'dialogue'),
    'xsum': ('xsum', 'train', 'document'),
    'cnn_dailymail': ('abisee/cnn_dailymail', '3.0.0', 'train', 'article'),
    'ag_news': ('fancyzhx/ag_news', 'train', 'text'),
    'banking77': ('mteb/banking77', 'train', 'text'),
    'alpaca_eval': ('Thanmay/alpaca_eval', 'eval', 'instruction'),
    'dolly': ('databricks/databricks-dolly-15k', 'train', 'instruction'),
    'daily_dialog': ('DeepPavlov/daily_dialog', 'train', 'dialog'),
    'oasst1': ('OpenAssistant/oasst1', 'train', 'text'),
    'ultrachat': ('HuggingFaceH4/ultrachat_200k', 'train_sft', 'messages'),
}

def load_model(m, d, dtype='auto'):
    t = AutoTokenizer.from_pretrained(m)
    if t.pad_token is None: t.pad_token = t.eos_token
    if dtype == 'float32':
        dt = torch.float32
    elif dtype == 'float16':
        dt = torch.float16
    else:
        dt = torch.float32 if d == 'cpu' else torch.float16
    return t, AutoModelForCausalLM.from_pretrained(
        m, torch_dtype=dt, low_cpu_mem_usage=True
    ).to(d).eval()

def gen_std(model, ids, mn):
    with torch.no_grad():
        o = model.generate(ids, max_new_tokens=mn, do_sample=False,
                           num_beams=1, pad_token_id=model.config.pad_token_id or 0)
    return o[0][ids.shape[1]:]

def gen_reuse(model, oi, mi, sh, mn, eos):
    sh = min(sh, len(oi[0]), len(mi[0]))
    with torch.no_grad():
        ca = model(oi, use_cache=True).past_key_values
    ca.crop(sh)
    sf = mi[:, sh:]
    if sf.shape[1] > 0:
        with torch.no_grad():
            cc = model(sf, past_key_values=ca, use_cache=True).past_key_values
        tp, st = sh + sf.shape[1], sf[:, -1:]
    else:
        cc, tp, st = ca, sh, oi[:, sh-1:sh]
    if tp < 1: return gen_std(model, mi, mn)
    cc.crop(tp-1)
    inp, kv, g = st, cc, []
    for _ in range(mn):
        with torch.no_grad():
            o = model(input_ids=inp, past_key_values=kv, use_cache=True)
        n = o.logits[0,-1].argmax(-1, keepdim=True).unsqueeze(0)
        inp, kv = n, o.past_key_values
        g.append(n.item())
        if n.item() == eos: break
    return torch.tensor(g)

def shuffle_ids(ids, r=0.75):
    sp = int(len(ids)*r)
    if len(ids)-sp < 4: sp = len(ids)-4
    if sp < 4: return ids, len(ids)
    return ids[:sp] + random.sample(ids[sp:], len(ids[sp:])), sp

def extract(row, k):
    if k in ('alpaca_eval','dolly'): return str(row.get('instruction',''))
    if k == 'samsum': return str(row.get('dialogue',''))
    if k == 'xsum': return str(row.get('document',''))
    if k == 'cnn_dailymail': return str(row.get('article',''))
    if k in ('ag_news','banking77'): return str(row.get('text',''))
    if k == 'daily_dialog':
        d = row.get('dialog',[])
        return ' '.join(d) if isinstance(d,list) else str(d)
    if k == 'oasst1': return str(row.get('text',''))
    if k == 'ultrachat':
        ms = row.get('messages',[])
        return ' '.join(m.get('content','') for m in ms if isinstance(m,dict))
    return str(row.get(list(row.keys())[0],''))

def load_samp(dk, n):
    from datasets import load_dataset
    info = DATASETS[dk]
    if len(info) == 4:
        ds = load_dataset(info[0], info[1], split=info[2], trust_remote_code=True)
    else:
        ds = load_dataset(info[0], split=info[1], trust_remote_code=True)
    return [extract(ds[i], dk)[:512] for i in range(min(n*2,len(ds)))
            if len(extract(ds[i], dk)) > 20][:n]

def run(args):
    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)
    ar = []; random.seed(42)
    for mk in args.models:
        print(f'\nLoading {mk} on {args.device}...', flush=True)
        t0 = time.time()
        tok, model = load_model(MODELS[mk], args.device, args.dtype)
        print(f'  loaded in {time.time()-t0:.0f}s', flush=True)
        eos = tok.eos_token_id or 0
        for dk in args.datasets:
            print(f'  === {mk} on {dk} ===', flush=True)
            sa = load_samp(dk, args.n_samples)
            print(f'  {len(sa)} samples', flush=True)
            if not sa: continue
            for idx, payload in enumerate(sa):
                i1 = tok.encode(payload, truncation=True, max_length=384)
                if len(i1) < 16: continue
                m1, sh = shuffle_ids(i1, r=0.75)
                io = torch.tensor([i1], device=args.device)
                im = torch.tensor([m1], device=args.device)
                ex = tok.decode(gen_std(model, io, args.max_gen_tokens), skip_special_tokens=True)
                rf = tok.decode(gen_std(model, im, args.max_gen_tokens), skip_special_tokens=True)
                ru = tok.decode(gen_reuse(model, io, im, sh, args.max_gen_tokens, eos), skip_special_tokens=True)
                ar.append({'model':mk,'dataset':dk,'shared_ratio':0.75,'shared_tokens':sh,
                    'total_tokens_orig':len(i1),'exact_text':ex,'ref_text':rf,'reuse_text':ru})
                if (idx+1) % 10 == 0:
                    print(f'    [{idx+1}/{len(sa)}]', flush=True)
        with open(od/f'{mk}_results.json','w') as f:
            json.dump([x for x in ar if x['model']==mk], f, indent=2)
        print(f'  Saved -> {od}/{mk}_results.json', flush=True)
        del model, tok
        if args.device.startswith('cuda'): torch.cuda.empty_cache()
    with open(od/'all_results.json','w') as f:
        json.dump(ar, f, indent=2)
    print(f'\nTotal: {len(ar)} samples -> {od}/all_results.json', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['float32','float16','auto'], default='auto')
    p.add_argument('--models', nargs='+', default=['gpt2'], choices=MODELS.keys())
    p.add_argument('--datasets', nargs='+', default=['samsum','alpaca_eval','banking77','daily_dialog','ag_news'], choices=DATASETS.keys())
    p.add_argument('--n_samples', type=int, default=10)
    p.add_argument('--max_gen_tokens', type=int, default=64)
    p.add_argument('--output_dir', default='fidelity_t4_results')
    args = p.parse_args()
    run(args)
