# P100 Learned Baseline Phase 3 Steps

Use this for the learned-admission baseline run. The fixed package is
`shadowkv_p100.tgz`; it contains the real `shadow_kv_plus_learned` engine.

## 0. Rebuild The Package On Windows

Do this after any local code change, before copying to the P100 box:

```powershell
cd C:\shadowkv\p100_transfer
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = "C:\shadowkv\p100_transfer\_package_stage_$stamp"
New-Item -ItemType Directory -Path $stage | Out-Null
robocopy C:\shadowkv\p100_transfer\shadowkv_p100 "$stage\shadowkv_p100" /E /XD .pytest_cache __pycache__ results_p100_n64_3seed results_p100_n256_full_3seed _qf32 /XF *.pyc
if ($LASTEXITCODE -gt 7) { exit $LASTEXITCODE }
tar -czf C:\shadowkv\p100_transfer\shadowkv_p100.tgz -C $stage shadowkv_p100
```

## 1. Transfer To The P100 Box

```powershell
scp C:\shadowkv\p100_transfer\shadowkv_p100.tgz kushalkhemani@192.168.104.77:~/research/
```

## 2. Unpack On The Server

```bash
ssh kushalkhemani@192.168.104.77
cd ~/research
tar xzf shadowkv_p100.tgz
cd ~/research/shadowkv_p100
```

Do not delete `~/research/shadowkv_p100` if Phase 1/2 already produced
`results/learned_baseline/learned_policy_raw.json`,
`results/learned_baseline/learned_policy_utility.json`, or train traces.
Untarring over the existing folder updates the code while preserving result
files that are not inside the archive.

## 3. Environment

Use the existing lab environment if it already has PyTorch/CUDA/HF working:

```bash
source ~/research/myenv/bin/activate
pip install -e .
pip install 'scikit-learn>=1.3'
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
```

If you are making a fresh environment instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 4. Smoke Test

This is CPU/fake-backend and should finish quickly:

```bash
python experiments/run_benchmark.py --backend fake --workload synthetic --variant high_skew --n_requests 8 --include_experimental --disable_arrival_simulation --engines no_cache shadow_kv_plus --output_dir /tmp/shadowkv_smoke
```

## 5. Start tmux

```bash
TERM=xterm tmux new -s shadowkv
```

Detach with `Ctrl-b d`. Reattach later with:

```bash
TERM=xterm tmux attach -t shadowkv
```

## 6. If Phase 1 And 2 Are Already Done

Expected default policy paths:

```bash
~/research/shadowkv_p100/results/learned_baseline/learned_policy_raw.json
~/research/shadowkv_p100/results/learned_baseline/learned_policy_utility.json
```

The policy JSONs now include the validation-selected admit threshold. Do not
pass `--learned_policy_threshold` unless you intentionally want to override
that saved threshold.

Run only held-out Phase 3:

```bash
python experiments/run_learned_baseline.py --phase phase3
```

If the utility policy JSON is somewhere else:

```bash
python experiments/run_learned_baseline.py --phase phase3 --policy_path /path/to/learned_policy.json
```

If either policy JSON is missing but the train `policy_trace.jsonl` files are
still under `results/learned_baseline/train/`, the script will rebuild the
missing policy automatically and then start Phase 3.

## 7. If The Policy Does Not Exist

Run Phase 1 and 2 first:

```bash
python experiments/run_learned_baseline.py --phase train
```

This pools train traces across seeds 42 and 123, five training datasets, and
both templated and semantic modes. Epsilon exploration is enabled by default to
collect realized outcomes for some requests MeritKV would otherwise bypass, and
the exploration hash includes the request index so repeated templates do not all
share the same explore/no-explore draw. The trainer writes both raw-feature and
utility-component policies and tunes a small C/threshold grid.

Then run held-out Phase 3:

```bash
python experiments/run_learned_baseline.py --phase phase3
```

## 8. Full Learned Baseline From Scratch

This trains and evaluates in one command:

```bash
python experiments/run_learned_baseline.py
```

## 8A. Complete 5-Seed / 5-Model Matrix

Use this for the full learned-baseline rerun. It runs GPT-2 first and Phi-3
last, uses five seeds, trains on templated + semantic, evaluates on raw +
templated + semantic, and then runs the memory-bound
victim/distractor/recovery trace against the learned baselines. The memory
trace uses repeated victim warmup and repeated distractor pressure so recovery
is measured under actual cache pressure:

```bash
python experiments/run_learned_baseline_all_models.py
```

Outputs:

```bash
results/learned_baseline_5seed_all_models/
results/learned_baseline_5seed_all_models/summary_all_models.json
results/learned_baseline_5seed_all_models/summary_memory_bound_all_models.json
results/learned_baseline_5seed_all_models/<model_slug>/memory_bound/memory_bound_aggregate.json
```

The wrapper resumes by default. If it stops halfway through, rerun the same
command and it will skip completed benchmark cells.

## 9. Preview Commands Without Running

```bash
python experiments/run_learned_baseline.py --phase phase3 --dry-run
python experiments/run_learned_baseline_all_models.py --dry-run
```

## 10. Check Progress

The driver prints each benchmark cell. For output files:

```bash
find results/learned_baseline -name 'benchmark_*.json' | wc -l
find results/learned_baseline -name 'policy_trace.jsonl' | wc -l
```

The final summary is:

```bash
cat results/learned_baseline/phase3_summary.json
```

## 11. Retrieve Results To Windows

From Windows:

```powershell
scp -r kushalkhemani@192.168.104.77:~/research/shadowkv_p100/results/learned_baseline C:\shadowkv\incoming_p100\
```

Or make one archive on the server first:

```bash
cd ~/research/shadowkv_p100
tar czf learned_baseline_results.tgz results/learned_baseline
tar czf learned_baseline_5seed_all_models_results.tgz results/learned_baseline_5seed_all_models
```

Then from Windows:

```powershell
scp kushalkhemani@192.168.104.77:~/research/shadowkv_p100/learned_baseline_results.tgz C:\shadowkv\incoming_p100\
scp kushalkhemani@192.168.104.77:~/research/shadowkv_p100/learned_baseline_5seed_all_models_results.tgz C:\shadowkv\incoming_p100\
```

## 12. What To Report

Use `results/learned_baseline/phase3_summary.json`.

Compare:

- `shadow_kv_plus`: hand-derived MeritKV utility decision.
- `shadow_kv_plus_lite`: capacity/break-even-style comparator included in the same Phase 3 table.
- `shadow_kv_plus_learned` in the `raw` summary block: outcome-trained learned baseline without B/C/W fields.
- `shadow_kv_plus_learned` in the `utility` summary block: outcome-trained learned baseline with B/C/W fields.
- `no_cache` and `shadow_kv`: anchors.

The learned label is single-request net savings. The all-model wrapper now
also runs the separate memory-bound recovery check, so do not rely on the
mixed-traffic Phase 3 summary alone for the locality mechanism claim.

Key fields:

- `mean_speedup`
- `mean_waste`
- `mean_hit_rate`
- `learned_policy_admit_total`
- `learned_policy_bypass_total`
- `learned_policy_flip_to_admit_total`
- `learned_policy_flip_to_bypass_total`

Memory-bound fields to report from `summary_memory_bound_all_models.json`:

- `victim_recovery_hit_rate`
- `first_recovery_hit_rate`
- `victim_recovery_speedup_vs_no_cache`
- `evictions`
- `peak_memory_used_mb`

## 13. tmux Stop

```bash
TERM=xterm tmux kill-session -t shadowkv
```
