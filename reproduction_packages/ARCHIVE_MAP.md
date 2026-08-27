# Archive-to-Expansion Map

Every retained archive is either paired with an expanded tree or lives inside a
package whose downloaded results are already expanded. Checksums are generated
in `archives/SHA256SUMS.txt` and `ARCHIVE_INVENTORY.csv`.

| Archive location | Expanded location |
|---|---|
| `archives/historical/ShadowKV-*` | Matching stem under `historical_transfers/` |
| `archives/ShadowKV-P100-HF-reproduction-package.zip` | `p100_hf/` |
| `archives/ShadowKV-Blackwell-long-prefix-semantic-n128-reproduction-package.zip` | `blackwell_longprefix_semantic_n128/` |
| `archives/ShadowKV-T4-fidelity-reproduction-package.zip` | `colab/legacy_fidelity_t4/` |
| `colab/exact_splice_validation/downloaded_results/*.zip` | Package-local execution audit/results plus `results/exact_splice_validation/raw/` |
| `experiments/archive/native_vllm_execute_bypass/downloaded_results/*.zip` | Sibling `attempt_*` directories |
| `results/blackwell_longprefix_hf/extensions/gemma31_all_engines_20260715/*.zip` | Sibling `artifact/` tree |
| `runtime_experiments/native_enforcement_blackwell/evidence_packages/2026-08-20/archives/*.zip` | `runtime_experiments/native_enforcement_blackwell/evidence_packages/2026-08-20/native_enforcement_package/` |
| `runtime_experiments/native_enforcement_blackwell/evidence_packages/2026-08-23/archives/*.zip` | Date-matched Gemma storage-admission and self-containment package trees |
| `runtime_experiments/native_enforcement_blackwell/evidence_packages/2026-08-24/archives/*.zip` | Date-matched Qwen storage-admission package tree |
| Qwen latency-diagnosis native-admission ZIP | Sibling `expanded_qwen14b_native_admission_gate_2026-06-22/` |

Historical archive names were normalized during cleanup without changing their
bytes. Package-internal archive names remain unchanged when verifiers or
provenance records refer to them.
