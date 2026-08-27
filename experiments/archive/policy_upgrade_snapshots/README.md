# Policy Upgrade Source Snapshots

These four source trees preserve successive controller-development workspaces.
Generated caches, temporary outputs, and duplicate result trees were excluded;
source, tests, experiment runners, configuration, and design notes were kept.

- `robust_policy_upgrade_v1/`: early robust-policy integration.
- `robust_policy_upgrade_v7/`: later semantic and policy optimization.
- `robust_policy_upgrade_v8-bad/`: explicitly named unsuccessful intermediate
  snapshot; retained so it cannot be mistaken for missing history.
- `robust_policy_upgrade_v9-exp/`: experimental successor snapshot.

The canonical release implementation remains `src/proactive_kv_cache/`.
