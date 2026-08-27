# Release Tools

- `generate_release_inventory.py` writes a deterministic file inventory and
  archive SHA-256 list without changing evidence files.
- `audit_release.py` checks required indexes, artifact-manifest paths, ignored
  evidence, oversized files, tracked caches, and archive checksums.
- `check_markdown_links.py` validates maintained repository-local Markdown
  targets. Frozen transfer-package documents are excluded because they retain
  original cross-repository provenance links.
- `audit_source_completeness.py` compares external handoff trees with the
  release by content hash while deduplicating repeated copies.
- `deduplicate_snapshot.py` removes byte-identical files only from a designated
  historical snapshot and records their preserved repository location.

Run both from the repository root after adding or relocating artifacts.
