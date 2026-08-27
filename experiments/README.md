# Experiment Pipelines

The top-level scripts are the maintained entry points. Frozen campaign runners
and notebooks live under `archive/` so measured artifacts remain reproducible
without making historical code look like the current API.

| Entry point | Purpose |
|---|---|
| `run_benchmark.py` | Controlled and process-isolated HF benchmark runner. |
| `run_fidelity_equiv.py` | Cached-continuation fidelity evaluation. |
| `eval_comprehensive.py` | Exact-match and ROUGE-L aggregation. |
| `profile_plan.py` | Controller `Plan()` overhead profiling. |
| `analyze_shadowkv_results.py` | Controlled-result and policy-table aggregation. |
| `archive/v10_snapshot_202608/` | Complete retained V10 experiment/source snapshot, including debug and campaign-specific runners. |
| `archive/native_vllm_execute_bypass/` | Auxiliary measured native-vLLM execute-or-bypass development package. |
| `archive/root_workspace_202608/` | Standalone retained analysis, calibration, debugging, and notebook utilities. |
| `archive/policy_upgrade_snapshots/` | Successive robust-policy source, tests, configs, and experiment snapshots. |
| `archive/blackwell_longprefix_source_snapshots/` | Six campaign-captured Blackwell long-prefix source revisions. |
| `archive/p100_source_snapshots/` | Earlier P100 package stages and additional retained runners. |

Stable raw engine IDs are `shadow_kv_plus` (MeritKV), `shadow_kv`
(MeritKV-Sem), and `shadow_kv_plus_lite` (MeritKV-Lite).

Do not edit archived source to match current APIs. Reproduce a historical
campaign inside its recorded environment, or port it explicitly and report the
port as a new run. See `docs/EXPERIMENT_CATALOG.md` for code-to-evidence links.
