# Historical Result Snapshots

This directory preserves complete retained result trees that are not the
current paper-facing namespace.

- `v10_results_snapshot_202608/` contains the V10 exact-splice, Gemma runtime,
  memory-bound, and multiround trees exactly as retained.
- `root_workspace_smoke_202608/` contains standalone HF, learned-policy, and
  sensitivity smoke outputs from the research workspace root.
- `external_unique_202608/` contains content-hash-unique files recovered from
  Blackwell/P100 handoffs, plus a path map for byte-identical deduplicated files.

These files are evidence and pipeline diagnostics, not alternate claim tables.
Use the canonical family directories and `results/paper_tables/` for the
current manuscript-facing aggregation.
