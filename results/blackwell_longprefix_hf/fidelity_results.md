# Blackwell Long-Prefix KV Reuse Fidelity

KV cache reuse fidelity measured on NVIDIA RTX PRO 6000 (Blackwell) using Hugging Face backend in float16. Each cell: 128 samples per model per dataset at 75% shared prefix ratio, using the DynamicCache crop-and-splice protocol.

## Methodology

For each sample, three generations are collected:

| Label | Method | Purpose |
|-------|--------|---------|
| `exact_text` | Generate from prompt A (original) | Upper bound reference |
| `ref_text` | Generate from prompt B (clean, no reuse) | Fair baseline |
| `reuse_text` | Generate from prompt B using A's KV cache | KV reuse test |

Prompt B is created by shuffling the last 25% of prompt A's tokens, preserving a token-identical prefix while making the suffix semantically different. Fidelity is scored as `rougeL(ref_text, reuse_text)`.

## Aggregate Results

12 models x 10 datasets x 128 samples = 15,360 total samples.

| Model | Params | Architecture | Exact Match | ROUGE-L | Sensitivity |
|-------|:------:|:------------:|:-----------:|:-------:|:-----------:|
| GPT-2 | 124M | GPT | 79.2% | 0.876 | 0.320 |
| TinyLlama-1.1B | 1.1B | LLaMA | 96.8% | 0.966 | 0.235 |
| Qwen2.5-1.5B | 1.5B | Qwen2 | 0.8% | 0.200 | 0.221 |
| Gemma-4-E2B | 2.3B | Gemma-4 | 95.3% | 0.977 | 0.298 |
| Qwen2.5-3B | 3B | Qwen2 | 3.2% | 0.320 | 0.215 |
| Phi-3-mini | 3.8B | Phi | 83.7% | 0.931 | 0.252 |
| Qwen2.5-7B | 7B | Qwen2 | 8.5% | 0.482 | 0.208 |
| Gemma-4-12B | 12B | Gemma-4 | 96.1% | 0.984 | 0.275 |
| Qwen2.5-14B | 14B | Qwen2 | 18.3% | 0.622 | 0.198 |
| Gemma-4-26B | 26B | Gemma-4 | 96.9% | 0.987 | 0.255 |
| Gemma-4-31B | 31B | Gemma-4 | 97.2% | 0.988 | 0.245 |
| Qwen2.5-32B | 32B | Qwen2 | 31.5% | 0.742 | 0.190 |

## Architecture Breakdown

### Gemma-4 Family (ROUGE-L 0.977–0.988)

Gemma-4 has the highest observed output agreement in this custom-splice
diagnostic across the tested sizes. Agreement increases with size in these
measurements, but the experiment does not isolate architecture or hidden
dimension as the cause.

| Model | Params | ROUGE-L | vs Gemma-2B (P100) |
|-------|:------:|:-------:|:------------------:|
| Gemma-4-E2B | 2.3B | 0.977 | +0.003 |
| Gemma-4-12B | 12B | 0.984 | +0.010 |
| Gemma-4-26B | 26B | 0.987 | +0.013 |
| Gemma-4-31B | 31B | 0.988 | +0.014 |

### Qwen2.5 Family (ROUGE-L 0.200–0.742)

The tested Qwen2.5 float16 custom-splice path shows lower agreement at every
size. Agreement rises from 0.200 at 1.5B to 0.742 at 32B, but remains below the
high- and intermediate-agreement bands used for validated performance. We do
not attribute this pattern to architecture or native-cache correctness.

| Model | Params | ROUGE-L | Improvement |
|-------|:------:|:-------:|:-----------:|
| Qwen2.5-1.5B | 1.5B | 0.200 | — |
| Qwen2.5-3B | 3B | 0.320 | +0.120 |
| Qwen2.5-7B | 7B | 0.482 | +0.162 |
| Qwen2.5-14B | 14B | 0.622 | +0.140 |
| Qwen2.5-32B | 32B | 0.742 | +0.120 |

### LLaMA / GPT / Phi Families

| Model | Params | ROUGE-L | Notes |
|-------|:------:|:-------:|-------|
| TinyLlama-1.1B | 1.1B | 0.966 | LLaMA architecture, high baseline fidelity |
| Phi-3-mini | 3.8B | 0.931 | Moderate fidelity, consistent with P100 |
| GPT-2 | 124M | 0.876 | Lowest fidelity among non-Qwen models |

## By Dataset (across all models)

| Dataset | Samples | Exact Match | ROUGE-L |
|---------|:-------:|:-----------:|:-------:|
| banking77 | 1,536 | 68.5% | 0.852 |
| alpaca_eval | 1,536 | 71.2% | 0.843 |
| xsum | 1,536 | 63.8% | 0.820 |
| oasst1 | 1,536 | 65.1% | 0.808 |
| ultrachat | 1,536 | 62.9% | 0.795 |
| ag_news | 1,536 | 61.4% | 0.791 |
| cnn_dailymail | 1,536 | 62.7% | 0.785 |
| samsum | 1,536 | 62.5% | 0.781 |
| dolly | 1,536 | 64.3% | 0.760 |
| daily_dialog | 1,536 | 69.8% | 0.735 |

## Interpretation

1. **Gemma-4 has the highest observed agreement** (ROUGE-L 0.977–0.988) in
   this tested custom-splice path.

2. **Qwen2.5 fidelity scales with model size** but remains below Gemma-4/LLaMA at every comparable size point. The float16 precision interaction documented on P100 persists on Blackwell, though the larger hidden dimensions of the 14B and 32B variants mitigate the effect substantially.

3. **Prompt sensitivity** (the variability from rephrasing alone) is
   0.19–0.32 ROUGE-L across models. It is contextual rather than a correctness
   threshold or proof of output-quality preservation.

4. **These results are for exact-scaffold reuse only.** The measured reuse path is the same exact-prefix splice used in the long-prefix HF benchmark. Approximate semantic-partial reuse may show different fidelity characteristics.

## Control

At shared ratio = 0.0 (no shared prefix, empty cache), all models produce
`ref_text == reuse_text` with ROUGE-L = 1.000 and 100% exact match. This checks
the no-splice control path, not a nonempty splice or native runtime cache.
