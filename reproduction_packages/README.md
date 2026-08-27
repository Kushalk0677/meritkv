# Reproduction Packages

Transfer and Colab packages are provided both as navigable expanded trees and,
where retained, as byte-identical ZIP archives. Package contents are frozen.

| Package | Expanded tree | Archive | Scope |
|---|---|---|---|
| P100 HF | `p100_hf/` | `archives/ShadowKV-P100-HF-reproduction-package.zip` | Controlled/process-isolated P100 commands, source, and records. |
| Blackwell semantic n=128 | `blackwell_longprefix_semantic_n128/` | `archives/ShadowKV-Blackwell-long-prefix-semantic-n128-reproduction-package.zip` | Long-prefix semantic study package. |
| T4 legacy fidelity | `colab/legacy_fidelity_t4/` | `archives/ShadowKV-T4-fidelity-reproduction-package.zip` | Original T4 fidelity notebook/package. |
| Exact-splice validation | `colab/exact_splice_validation/` | Package-local downloaded archives are retained | Five-model Colab runner, requirements, execution audit, raw reports, and logs. |
| Legacy Gemma-4 E2B validation | `colab/gemma4_e2b_splice_validation_legacy/` | Expanded only | Earlier model-specific notebook and summary utility, retained for provenance. |
| Historical P100/Blackwell/policy transfers | `historical_transfers/` | `archives/historical/` | Retained source/result packages, descriptively renamed and safely expanded. |

Named native-enforcement archives remain beside their expanded evidence under
`runtime_experiments/native_enforcement_blackwell/`, because their verifiers
use package-relative paths.

`archives/SHA256SUMS.txt` and `ARCHIVE_INVENTORY.csv` record archive digests.
Python caches,
model weights, downloaded datasets, and complete virtual environments are not
release artifacts.

See `ARCHIVE_MAP.md` for the expansion corresponding to each archive family.
