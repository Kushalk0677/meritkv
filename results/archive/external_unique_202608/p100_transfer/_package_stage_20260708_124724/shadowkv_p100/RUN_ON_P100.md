# ShadowKV++ — P100 re-run package

Self-contained. Everything needed to run the two GPU experiments is here;
no clone required. The memory-breakeven guard is already disabled (Path C)
and gated behind `SHADOWKV_BREAKEVEN_GUARD` for A/B testing.

## 0. Setup (once, on the P100 box)

```bash
cd shadowkv_p100
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
pip install pytest                      # optional, for the smoke check

# Use the cached models/datasets, never hit the network:
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# If the lab cache is not at ~/.cache/huggingface:
# export HF_HOME=/path/to/shared/hf_cache

nvidia-smi                              # confirm a P100 is free
export CUDA_VISIBLE_DEVICES=0           # pin to ONE free GPU
```

Sanity check (CPU, ~seconds — catches any env breakage before GPU time):
```bash
python experiments/run_benchmark.py --backend fake --workload synthetic \
  --n_requests 8 --include_experimental --disable_arrival_simulation \
  --output_dir /tmp/smoke
```

**Always launch the real sweeps inside tmux** (survives SSH drops):
```bash
tmux new -s shadowkv        # detach: Ctrl-b d   reattach: tmux attach -t shadowkv
```

## 1. Phase A — Phi-3 breakeven re-run (~90 runs)

Reproduces the committed final_p100 Phi-3 sweep exactly (n=256, simulated
arrivals, all 9 engines per process), guard OFF. Resumable — rerun to
continue after an interruption.

```bash
python experiments/run_p100_phi3.py            # -> results_p100_phi3_rerun/
```

Optional controlled A/B (reproduce the ORIGINAL guarded behaviour):
```bash
python experiments/run_p100_phi3.py --restore-guard \
  --output_dir results_p100_phi3_guarded
```

**What to look at:** does Phi-3 `raw` speedup stay >= 1.0 — especially
`cnn_dailymail / seed_42`, which was 0.895x with the guard? Also pull
`wasted_compute_ratio` and `speculation_cooldown_events` from the
shadow_kv_plus block of each JSON.

## 2. Phase B — Executed semantic-reuse re-run (~50 runs)

Turns on real approximate semantic reuse (`--allow_unsafe_semantic_kv_reuse`).
Default is the minimum defensible matrix: 5 models x 10 datasets x seed 42.

```bash
python experiments/run_p100_semantic.py        # -> results_p100_semantic_unsafe/
```

Expand to 3 seeds for CIs (only if the effect is worth reporting):
```bash
python experiments/run_p100_semantic.py --seeds 42 123 456
```

**What to look at:** in the shadow_kv_plus block, `semantic_partial_hits`
should now be > 0 (was 0 everywhere), with `semantic_partial_reused_tokens_total`
and `semantic_quality_divergence_sum/_events`. That is executed reuse +
a divergence proxy.

**Output fidelity (do this too):** the latency benchmark does not compute
ROUGE. After Phase B, run a targeted fidelity pass so you can report the
quality COST of approximate reuse (reviewers will ask):
```bash
python experiments/run_fidelity_equiv.py --device cuda:0 \
  --models tinyllama qwen25_15b gemma2b phi3mini \
  --datasets samsum alpaca_eval banking77 --n_samples 32 --max_gen_tokens 64
```
Expect Qwen2.5 to degrade badly in float16 (cf. the 0.20 ROUGE finding).

## 3. Phase C - Learned admission baseline

Compares MeritKV's hand-derived utility formula against real learned admission
engines. The driver trains from observed admitted-reuse outcomes, not from
MeritKV's own admit/bypass labels. Phase 1 uses deterministic epsilon
exploration (`--outcome_exploration_rate`, default 0.10) to collect some
counterfactual admit outcomes for candidates MeritKV would otherwise bypass.

It saves two policies:

```
results/learned_baseline/learned_policy_raw.json
results/learned_baseline/learned_policy_utility.json
```

`learned_policy_raw.json` uses raw request/cache features only.
`learned_policy_utility.json` adds MeritKV's B/C/W-style components to test
learned reweighting. A compatibility alias is also written to
`results/learned_baseline/learned_policy.json`.

Trains on seeds {42, 123} and 5 datasets, then evaluates on seed 456 and the
remaining 5 datasets. This split ensures the comparison is out-of-sample.

```bash
python experiments/run_learned_baseline.py     # -> results/learned_baseline/
```

If Phase 1/2 already finished and you only want the held-out learned-engine
run, use:
```bash
python experiments/run_learned_baseline.py --phase phase3
```
This uses both learned policy JSON files. If either JSON is missing but the
Phase 1 train `policy_trace.jsonl` files are still present, the driver will
rebuild the missing policy before starting Phase 3.

If the policy JSON is somewhere else, pass it explicitly:
```bash
python experiments/run_learned_baseline.py --phase phase3 --policy_path /path/to/learned_policy.json
```

**What to look at:** `phase3_summary.json` contains separate `raw` and
`utility` blocks. Each block reports held-out latency, speedup, waste, and
hit-rate for `no_cache`, `shadow_kv`, `shadow_kv_plus`, and
`shadow_kv_plus_learned`. The learned engine also reports
`learned_policy_admit_total`, `learned_policy_bypass_total`,
`learned_policy_flip_to_admit_total`, and
`learned_policy_flip_to_bypass_total`.

## 4. Bring results back

Copy these directories back to the workstation:
```
results_p100_phi3_rerun/
results_p100_semantic_unsafe/
results/learned_baseline/
fidelity_equiv_*/            # whatever run_fidelity_equiv wrote
```
(e.g. `tar czf p100_results.tgz results_p100_* fidelity_equiv_*` then scp.)

## Notes

- Every run is isolated to its own output dir and resumable; safe to Ctrl-C
  and relaunch.
- `--dry-run` on either driver prints the exact commands without executing.
- Phi-3 / Qwen do not need `--trust_remote_code` (the committed runs had it
  off); pass it only if a tokenizer load complains.
- The guard toggle lives in `src/proactive_kv_cache/engines.py` — search
  `SHADOWKV_BREAKEVEN_GUARD`.
```
