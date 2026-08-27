# Fidelity T4 Package

Standalone fidelity experiment runner for Colab T4 GPU.

## Contents

- `run_fidelity_t4.py` — Main experiment script
- `requirements.txt` — Python dependencies
- `pyproject.toml` — Package metadata

## Usage

```bash
pip install -r requirements.txt

# Float16 (GPU) — GPT-2 + Phi-3, 10 samples x 5 datasets
python run_fidelity_t4.py --device cuda:0 --models gpt2 phi3mini \
  --datasets samsum alpaca_eval banking77 daily_dialog ag_news \
  --n_samples 10 --output_dir fidelity_t4_results_f16

# Float32 (CPU) — same config, float32 precision
python run_fidelity_t4.py --device cpu --models gpt2 phi3mini \
  --datasets samsum alpaca_eval banking77 daily_dialog ag_news \
  --n_samples 10 --output_dir fidelity_t4_results_f32
```

## Output Format

Each run produces per-model JSON files + `all_results.json` with:
`{model, dataset, shared_ratio, shared_tokens, total_tokens_orig,
  exact_text, ref_text, reuse_text}`

Fidelity is computed as `ROUGE-L(ref_text, reuse_text)`.
