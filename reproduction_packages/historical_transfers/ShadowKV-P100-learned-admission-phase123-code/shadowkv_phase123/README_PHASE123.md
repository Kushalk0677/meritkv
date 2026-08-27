# Learned Baseline Phase 1-3 Code Package

This package contains only the code needed to run the learned admission baseline through Phase 3.

It includes:

- `experiments/run_benchmark.py`
- `experiments/run_learned_baseline.py`
- `experiments/run_learned_baseline_phase123_all_models.py`
- `src/proactive_kv_cache/`
- `config/config.yaml`
- `requirements.txt`
- `pyproject.toml`

It intentionally does not include result folders, logs, paper files, or the memory-bound victim/recovery trace.

## Setup

From the package root:

```bash
python -m venv myenv
source myenv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

## One Command: Full Phase 1-3 Matrix

This runs GPT-2, Qwen2.5-1.5B, TinyLlama-1.1B, Gemma-2B, and Phi-3-mini, with Phi-3 last.

It uses five seeds, `n_requests=128`, train modes `templated semantic`, and Phase 3 test modes `raw templated semantic`.

```bash
python experiments/run_learned_baseline_phase123_all_models.py
```

Outputs go to:

```text
results/learned_baseline_phase123_5seed_all_models/
```

The combined summary is written to:

```text
results/learned_baseline_phase123_5seed_all_models/summary_phase123_all_models.json
```

## Single Model

For a single model, run train and Phase 3 directly:

```bash
python experiments/run_learned_baseline.py --model Qwen/Qwen2.5-1.5B-Instruct --phase train --train_seeds 42 123 456 789 999 --test_seeds 42 123 456 789 999 --train_modes templated semantic --n_requests 128 --resume --out_root results/learned_baseline_qwen25_phase123
python experiments/run_learned_baseline.py --model Qwen/Qwen2.5-1.5B-Instruct --phase phase3 --train_seeds 42 123 456 789 999 --test_seeds 42 123 456 789 999 --train_modes templated semantic --test_modes raw templated semantic --n_requests 128 --resume --out_root results/learned_baseline_qwen25_phase123
```

## Notes

- Phase 1 collects policy traces using `no_cache`, `shadow_kv`, and `shadow_kv_plus`.
- Phase 2 trains two learned admission policies: raw features and utility-component features.
- Phase 3 evaluates `no_cache`, `shadow_kv`, `shadow_kv_plus_lite`, `shadow_kv_plus`, and `shadow_kv_plus_learned`.
- The run is resumable by default in the all-model wrapper. Use `--no-resume` to force reruns.
- On a 12 GB P100, Phi-3 may fail with CUDA OOM. The wrapper places Phi-3 last so the smaller models finish first.
