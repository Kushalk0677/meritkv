# Multiround Capacity-Pressure Trace

This family contains the four-arm enforced Gemma-4-31B capacity-pressure result
at two cache budgets. It distinguishes native admit-all, MeritKV write-through,
MeritKV enforcement, and the occupancy control, and reports latency, recovery,
and decision behavior across rounds.

The detailed measured values and methodology are in the Markdown reports in
this directory. Historical runner and analyzer sources are preserved in
`experiments/archive/v10_snapshot_202608/`.
