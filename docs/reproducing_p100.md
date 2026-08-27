# Reproducing P100 Runs

Raw artifacts keep stable engine IDs: `shadow_kv_plus` displays as MeritKV, `shadow_kv` displays as MeritKV-Sem, and `shadow_kv_plus_lite` displays as MeritKV-Lite.


Use `experiments/run_p100_isolated_sweep.py` for a conservative P100 rerun. The
frozen transfer package is expanded at `reproduction_packages/p100_hf/`, and
its byte-identical archive is retained under `reproduction_packages/archives/`.
Use the maintained runner for new work and the frozen package to reproduce or
audit the historical environment.

## Why This Runner Is Isolated

The historical P100 package included a private `--share_backend` experiment path. The public `experiments/run_benchmark.py` does not expose that flag, so the public runner uses one subprocess per engine cell instead. This is slower, but it is safer on a 12 GB P100 because each failed cell exits cleanly.

## Smoke Test

```bash
python experiments/run_p100_isolated_sweep.py \
  --models gpt2 \
  --datasets ag_news \
  --prompt_modes raw \
  --engines no_cache shadow_kv_plus  # shadow_kv_plus displays as MeritKV \
  --seeds 42 \
  --n_requests 8 \
  --results_root results_p100_smoke
```

## Public Isolated Rerun

```bash
python experiments/run_p100_isolated_sweep.py
```

Current runner default:

```text
5 models x 10 datasets x 3 prompt modes x 3 seeds x 3 engines
```

Default engines:

```text
no_cache
shadow_kv       # MeritKV-Sem
shadow_kv_plus  # MeritKV
```

This default is a convenient public sweep, not an assertion that every
configured cell appears in the paper. The released process-isolated comparison
contains four completed models (Gemma-2B, Qwen2.5-1.5B, TinyLlama-1.1B, and
GPT-2), for 360 cells per engine. The separate controlled study and its
availability accounting are documented in `results/controlled_results/`.

To execute approximate semantic KV reuse in semantic mode, add:

```bash
--allow_unsafe_semantic_kv_reuse
```

Use that flag only for controlled ablations and report it explicitly.

## Returned Files

```text
results_p100_isolated/_run_manifest.json
results_p100_isolated/_sweep.log
results_p100_isolated/_failures.json
results_p100_isolated/<model>/<mode>/seed_<seed>/<dataset>/<engine>/benchmark_*.json
```
