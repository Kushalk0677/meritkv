# P100 Run Steps

## 1. Transfer (Windows)
```powershell
scp C:\shadowkv\p100_transfer\shadowkv_p100.tgz kushalkhemani@192.168.104.77:~/research/
```

## 2. Unpack (server)
```bash
ssh kushalkhemani@192.168.104.77
cd ~/research
tar xzf shadowkv_p100.tgz
cd ~/research/shadowkv_p100
```

## 3. Environment
```bash
source ~/research/myenv/bin/activate
pip install -e .
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
```

## 4. Smoke test
```bash
python experiments/run_benchmark.py --backend fake --workload synthetic --n_requests 8 --include_experimental --disable_arrival_simulation --output_dir /tmp/smoke
```

## 5. tmux
```bash
TERM=xterm tmux new -s shadow
```

## 6. Preview (optional)
```bash
python experiments/run_p100_overnight.py --dry-run
```

## 7. Overnight run (n=64, 3 seeds, 3 engines, ~7-9h)
```bash
python experiments/run_p100_overnight.py
```

## 7b. Full run (n=256, all engines, energy, ~20-24h) -- alternative
```bash
python experiments/run_p100_full.py
```

## 8. Check progress (from a second shell / after reattach)
```bash
tail -f ~/research/shadowkv_p100/results_p100_n64_3seed/_sweep.log
```

## 9. Retrieve (Windows)
```powershell
scp kushalkhemani@192.168.104.77:~/research/shadowkv_p100/results_p100_n64_3seed.zip C:\Users\kusha\Downloads\
```

```powershell
scp kushalkhemani@192.168.104.77:~/research/shadowkv_p100/results_p100_n256_full_3seed.zip C:\Users\kusha\Downloads\
```

## tmux attach / kill
```bash
TERM=xterm tmux attach -t shadow
```

```bash
TERM=xterm tmux kill-session -t shadow
```
