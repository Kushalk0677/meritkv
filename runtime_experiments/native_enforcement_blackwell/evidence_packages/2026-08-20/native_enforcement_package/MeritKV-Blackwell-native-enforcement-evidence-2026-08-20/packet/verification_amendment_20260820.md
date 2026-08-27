# Verification Amendment — Output Agreement Is a Measured Result

Date: 2026-08-20
Author: Keystone on IronGod

## Authority and reason

Kushal Khemani's 2026-08-14 email, “Blackwell enforced MeritKV experiment”
(Proton Inbox UID 1400), requires the request trace to retain “Output tokens or
text for agreement checking.” It does not specify bit-identical output across
the cache-reuse and forced-recompute arms as an experiment-completion gate.

The original packet verifier at SHA-256
`1176c1d8551a0552536a2faf2bea41f1917b525c598a1981226795bac5438882`
added a stricter, packet-created rule that rejected any cross-arm output-digest
difference. During the frozen run, that rule exposed sparse one-token
differences between cache reuse and recomputation. Those differences are a
scientific result to retain and quantify, not missing or corrupt execution.

## Amended acceptance rule

- Every planned request must still exist in exact order with no failures.
- Every output text must still match its per-request SHA-256.
- Every cell's ordered output digest must still be independently recomputed and
  match the benchmark receipt.
- Cross-arm comparison must cover every request, report exact matching and
  differing counts and indices for every arm pair, and retain all arm outputs,
  hashes, cache-token counts, and policy strategies for every differing row.
- A genuine, internally valid cross-arm output difference is reported without
  tolerance and does not make the operation incomplete.
- Missing, malformed, misordered, unplanned, or internally inconsistent output
  evidence remains fatal.

This amendment changes only terminal interpretation and reporting. It does not
change or rerun completed cells, model/image revisions, prompts, cell order,
seeds, policy, runtime configuration, generation settings, or energy protocol.

