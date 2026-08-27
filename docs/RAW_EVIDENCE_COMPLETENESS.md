# Raw Evidence Completeness Audit

The August 2026 cleanup compared the release against every retained project
result, runtime, incoming-handoff, transfer, and V10 tree by SHA-256. This is a
content audit: renames and duplicate directory layouts do not count as missing
when identical bytes are preserved elsewhere.

## Audited Sources

| Source family | Unique contents absent before cleanup | Preservation action |
|---|---:|---|
| V10 results | 2,842 | Complete tree preserved at `results/archive/v10_results_snapshot_202608/`. |
| V10 runtime | 0 | Already represented in the release. |
| Root smoke/sensitivity results | 18 | Complete tree preserved at `results/archive/root_workspace_smoke_202608/`. |
| Root Gemma runtime | 90 | Complete tree preserved at `runtime_experiments/archive/root_workspace_snapshot_202608/`. |
| Original Blackwell long-prefix tree | 394 | Unique paths preserved below `results/archive/external_unique_202608/`. |
| Standalone k-star and vLLM summary trees | 3 | Unique summaries preserved in the same external snapshot. |
| Incoming Blackwell handoff | 444 | Unique files preserved in the external snapshot. |
| Incoming P100 handoff | 1,399 | Unique files preserved; transfer archives moved to named reproduction packages. |
| Incoming P100 five-seed tree | 906 | Unique raw cells preserved in the external snapshot. |
| P100 transfer tree | 468 | Unique files preserved; transfer archives moved to named reproduction packages. |
| Native-enforcement audit tree | 0 | Already represented in the native evidence package. |

After archive expansion, 706 byte-identical copies were removed only from the
new external snapshot. `results/archive/external_unique_202608/DEDUPLICATION_MAP.csv`
records each original path, preserved location, SHA-256, and size.

## Code and Pipeline Audit

All unique Python, shell, PowerShell, notebook, YAML, TOML, and experiment
configuration contents in the V10, Blackwell, incoming, P100, and robust-policy
workspaces are represented. Missing historical revisions were preserved under:

- `experiments/archive/v10_snapshot_202608/`
- `experiments/archive/root_workspace_202608/`
- `experiments/archive/policy_upgrade_snapshots/`
- `experiments/archive/blackwell_longprefix_source_snapshots/`
- `experiments/archive/p100_source_snapshots/`
- `experiments/archive/native_vllm_execute_bypass/`
- `reproduction_packages/colab/`

Frozen snapshots remain untouched. Concise engineer-facing section comments
were added only to maintained source and the maintained benchmark runner.

## Archive Handling

Historical ZIP/TGZ packages were moved to
`reproduction_packages/archives/historical/`, renamed descriptively without
changing their bytes, and expanded under
`reproduction_packages/historical_transfers/`. Package-internal archives whose
names are referenced by verifiers remain in place. See
`reproduction_packages/ARCHIVE_MAP.md` and the generated archive checksum files.

The four externally delivered Blackwell experiment archives are retained in
their original bytes and paired with complete expansions under
`runtime_experiments/native_enforcement_blackwell/evidence_packages/`. A
member-by-member SHA-256 audit found no missing, changed, or extra files in any
expansion. The bundled August 20 native-enforcement verifier, Gemma extension
verifier, and Qwen extension verifier pass; the Qwen verifier records POSIX
relative paths and must therefore be replayed on Linux or WSL rather than
native Windows Python.

## Defined Exclusions

The release does not contain model weights, downloaded dataset caches, virtual
environments, Python bytecode, pytest caches, GPU drivers, container layers, or
framework caches. These are dependencies or generated local state, not research
evidence. A run that failed before emitting a measurement remains documented as
failed; no raw cell is reconstructed from a summary.
