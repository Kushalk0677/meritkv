# Process-Isolated Five-Model Summary

`summary_by_model_mode.csv` is recomputed from the measured process-isolated
P100 runs and contains one row per model and prompt mode. Each row aggregates 50
paired cells (ten datasets across five seeds).

The templated Qwen speedup is retained for transparency, but the paper excludes
it from validated performance because the separate float16 custom-splice
agreement diagnostic is materially lower. Semantic-mode rows are a
latency-only negative study because substitution output quality was not
evaluated there. Neither diagnostic is a native runtime-cache correctness test.
