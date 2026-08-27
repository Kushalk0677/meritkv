# Dependencies and Package Records

The repository has a maintained portable environment and multiple frozen
campaign environments. There is intentionally no single lock file that claims
to reproduce T4/P100 Hugging Face, Blackwell vLLM, SGLang, and LMCache runs at
once.

## Maintained Package

- `pyproject.toml` defines the installable `proactive_kv_cache` package.
- `requirements.txt` lists the maintained benchmark dependencies.
- `pip install -e .` installs current source for development and smoke tests.

## Frozen Environments

| Family | Dependency record |
|---|---|
| P100 HF | `reproduction_packages/p100_hf/**/requirements.txt` and `pyproject.toml` |
| Blackwell long prefix | `reproduction_packages/blackwell_longprefix_semantic_n128/**/requirements.txt`, `pyproject.toml`, and fidelity requirements |
| Exact-splice Colab | `reproduction_packages/colab/exact_splice_validation/requirements-colab.txt`, execution-specific `pip_freeze.txt`, and `environment.json` |
| Legacy fidelity | `reproduction_packages/colab/legacy_fidelity_t4/` and the Gemma-specific legacy package |
| Qwen runtime | Campaign-local `pip_freeze.txt`, image inspection, launch commands, and runtime version files under `runtime_experiments/qwen2.5/` |
| Native enforcement | Source snapshots, container/runtime patches, commands, manifests, and verifiers under `runtime_experiments/native_enforcement_blackwell/` |
| Historical policy/P100 packages | Expanded trees under `reproduction_packages/historical_transfers/` |

## What Is Not Vendored

Model weights, dataset caches, CUDA, GPU drivers, container layers, and complete
virtual environments are not stored. Public identifiers, revisions, dependency
lists, environment captures, and launch commands are retained so those external
components can be reacquired. Never place access tokens in notebooks or command
files; see `SECURITY_REDACTIONS.md`.
