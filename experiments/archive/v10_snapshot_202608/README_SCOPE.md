# V10 Source Snapshot Scope

This directory preserves experiment scripts, notebooks, analysis utilities,
and campaign notes retained from the V10 research workspace in August 2026.
Files keep their original names and contents so old commands and raw manifests
remain interpretable.

This is a provenance snapshot, not the canonical import location. Use
`src/proactive_kv_cache/` and the top-level `experiments/` scripts for new work.
Historical scripts may contain obsolete paths, defaults, or compatibility code;
their presence records what was available, not a recommendation to combine all
of them into one environment.

Security exception: two embedded Hugging Face token literals were replaced by
`hf_REDACTED`; see `docs/SECURITY_REDACTIONS.md`. No scientific content changed.
