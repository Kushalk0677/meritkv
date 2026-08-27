# ShadowKV P100 Clean Package

This is a cleaned, sendable copy of the P100 runnable repository.

Included:

- `src/proactive_kv_cache/`
- `experiments/`
- `literature_accurate_baselines/`
- `tests/`
- `config/`
- `RUN_ON_P100.md`
- `STEPS.md`
- `pyproject.toml`
- `requirements.txt`

Excluded:

- prior result directories
- logs
- `.pytest_cache`
- `__pycache__`
- Python bytecode
- scratch `_qf32` outputs

For the learned-baseline Phase 1 to Phase 3 matrix, read `RUN_ON_P100.md` and `STEPS.md`.

Typical start:

```bash
python -m venv myenv
source myenv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

Then run the documented experiment command from `STEPS.md`.
