# LMCache Without Native RadixAttention

`results.csv` contains one measured aggregate row for each of the five
Qwen2.5 sizes. None is scaled from another model-size anchor.
The former detailed top-level export is preserved as
`raw/legacy_detailed_results_precompact.csv`.

This arm is contextual. The principal like-for-like comparison in the paper is
native SGLang RadixAttention versus RadixAttention plus the write-through
MeritKV decision overlay.
