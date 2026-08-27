# Repository Structure and Evidence Policy

This repository keeps the implementation, experiment entry points, complete
project-retained raw records, derived tables, and frozen reproduction packages
in separate namespaces. The separation is intentional: a reader should be able
to identify executable source, immutable evidence, and paper-facing aggregates
without guessing from a filename.

## Authoritative Layout

| Path | Role | Editing rule |
|---|---|---|
| `src/proactive_kv_cache/` | Current MeritKV implementation | Maintained source; changes require tests. |
| `experiments/` | Current experiment and analysis entry points | Maintained source; archived runners are explicitly scoped. |
| `literature_accurate_baselines/` | Runtime-baseline adapters and source notes | Maintained with the corresponding backend. |
| `results/` | HF, policy, fidelity, memory, and analytical evidence | Raw records are immutable; regenerate summaries from their documented inputs. |
| `runtime_experiments/` | vLLM, SGLang, LMCache, and native-enforcement evidence | Preserve commands, environment captures, logs, counters, and request records together. |
| `reproduction_packages/` | Expanded transfer/Colab bundles and byte-identical archives | Do not edit frozen package contents. |
| `docs/` | Cross-campaign design, methodology, and reproduction guidance | Keep paper-facing language aligned with the current manuscript. |
| `tests/` | Unit and regression tests | Run before changing maintained source. |
| `tools/` | Release inventory and integrity checks | Maintained source. |

## Evidence Layers

1. **Raw records** are the JSON, JSONL, CSV, log, command, environment, and
   counter files emitted or captured during a run. They are retained exactly as
   produced, including historical local paths and engine IDs.
2. **Campaign summaries** aggregate one experiment family. Their local README
   defines the run matrix, exclusions, and aggregation boundary.
3. **Paper tables** in `results/paper_tables/` are compact machine-readable
   counterparts of the manuscript tables. They do not replace raw evidence.
4. **Frozen packages** preserve the transfer artifact used for a campaign.
   Expanded copies are provided for navigation; the original ZIP and SHA-256
   establish byte-level provenance.
5. **Historical code** under `experiments/archive/` records runners used during
   development. It is not silently substituted for the maintained package.

## Completeness Boundary

The repository includes every raw result, log, script, notebook, environment
capture, and transfer package retained in the project workspace at release
preparation time. It does not redistribute model weights, downloaded datasets,
Python environments, framework caches, or GPU-driver installations. A campaign
that terminated before producing a cell is documented as incomplete; missing
measurements are never reconstructed from an aggregate.

Use `docs/EXPERIMENT_CATALOG.md` to locate each family and
`docs/HARDWARE_AND_ENVIRONMENTS.md` to locate machine captures. Run
`python tools/audit_release.py` for structural checks.
The content-hash audit and defined exclusions are recorded in
`docs/RAW_EVIDENCE_COMPLETENESS.md`.
Credential-redaction details are recorded in `docs/SECURITY_REDACTIONS.md`.
