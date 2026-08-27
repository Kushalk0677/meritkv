# ShadowKV++ Overlay Semantics

This corrected matrix contains six arms:

- `vllm_apc`
- `vllm_apc_shadowkv_plus`
- `sglang_radix_attention`
- `sglang_radix_attention_shadowkv_plus`
- `lmcache`
- `lmcache_shadowkv_plus`

The three `*_shadowkv_plus` arms use the same external ShadowKV++ admission and
policy controller with the `balanced` preset. End-to-end timing includes policy
planning, the server request, and feedback.

The overlay uses `write_through_admission` for all three runtimes. It records
allow/bypass decisions and policy overhead, but external runtimes retain cache
ownership. A bypass decision does not claim native per-request suppression of
cache lookup or cache write. This is the portable policy-overlay comparison,
not a direct KV injection path or a native SGLang admission-hook experiment.

The smoke gate requires every overlay arm to report:

- `admission_controller_enabled=true`
- one admission plan per measured request
- `allow + bypass == requests`
- zero runtime-cache reset failures
- `admission_enforcement_mode=write_through_admission`

It also requires positive runtime-native cache evidence for all six smoke arms.
