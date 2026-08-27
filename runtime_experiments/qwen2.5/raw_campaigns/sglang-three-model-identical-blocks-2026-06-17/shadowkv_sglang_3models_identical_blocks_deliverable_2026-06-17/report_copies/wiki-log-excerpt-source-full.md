---
title: Wiki Log
created: 2026-04-30
updated: 2026-06-13
author: Multiple
authors: [Multiple, Keystone, Rover]
last_editor: Keystone (Codex on IronGod)
type: journal
tags: [wiki, log, operations]
sources: []
---

## [2026-06-13] artifact | Obsidian Bases control dashboards
Author: Keystone (Codex on IronGod)
- Added six native `.base` dashboards: `infrastructure/keystone-startup.base`, `infrastructure/wiki-health.base`, `life/projects/project-control.base`, `infrastructure/jadewire-ops.base`, `infrastructure/loom-control.base`, and `research/research-watch.base`.
- Added `infrastructure/obsidian-bases-dashboard.md` with embedded views and linked it from `infrastructure/index.md`.
- Boundary: Bases are Obsidian-native query/control surfaces over Markdown and frontmatter. Runtime state remains in Postgres/service manifests.
- Verification: Ruby YAML parsing succeeded for all six `.base` files. `find` confirms six `.base` files in the active wiki. `.obsidian/core-plugins.json` has `"bases": true`.

## [2026-06-13] update | Darwin-28B-REASON deployed on first-light vLLM
Author: Keystone (Codex on IronGod)
- Added `/home/jade_hand/active/services/darwin28b-reason-vllm/docker-compose.yml` for `FINAL-Bench/Darwin-28B-REASON` using `vllm/vllm-openai:nightly`, BF16, `--language-model-only`, and `--max-model-len 65536`.
- Downloaded the public model into `/datapool/cache/huggingface`, stopped `qwen36-27b-fp8-vllm` to free VRAM, and started `darwin28b-reason-vllm` on `127.0.0.1:8015`.
- Verified direct vLLM `/health`, `/v1/models`, and chat completion; LiteLLM now exposes `darwin28b` and a routed chat smoke returned HTTP `200` with content.
- Retargeted `/home/jade_hand/.hermes/config.yaml` local-vLLM defaults from `qwen36` to `darwin28b`; restarted `hermes-gateway.service`, which remains active/running with `NRestarts=0`.
- Updated first-light `/home/jade_hand/ops/SERVICE_MANIFEST.md` and `/home/jade_hand/ops/AGENT_MANIFEST.md`.

## [2026-06-13] update | first-light Hermes gateway shell fixed
Author: Keystone (Codex on IronGod)
- Added first-light user-service drop-in `/home/jade_hand/.config/systemd/user/hermes-gateway.service.d/10-bash-shell.conf` for `hermes-gateway.service`.
- The drop-in sets `SHELL=/run/current-system/sw/bin/bash` and adds `/run/current-system/sw/bin` plus `/run/wrappers/bin` to the gateway PATH, preventing Hermes local terminal/file wrappers from falling back to fish.
- Restarted only `hermes-gateway.service`; it is active/running with `NRestarts=0`.
- Verified Hermes `tools.environments.local._find_bash()` returns `/run/current-system/sw/bin/bash` and a local execution smoke using `__hermes_ec=$?` returns `ec:0`.

## [2026-06-01] update | first-light desktop environment documented
Author: Rover (first-light)
- KDE Plasma 6.5.5 with SDDM 0.21.0 installed on first-light
- Configured for X11 (`plasmax11` default session, SDDM Wayland disabled), but user is running Wayland session (kwin_wayland + Xwayland) — selected at SDDM login
- Updated `life/entities/first-light.md`: added Desktop Environment section, NixOS version, GPU driver details, Docker CDI config, clarified Tailscale IP, improved tags
- Bumped `updated` date to 2026-06-01

## [2026-06-11] update | TensorRT-LLM Qwen3.6 run notes refreshed
Author: Keystone (Codex on IronGod)
- Refreshed [[life/entities/tensorrt-llm]] against NVIDIA TensorRT-LLM release notes, Qwen3 deployment guide, support matrix, and Qwen examples.
- Captured current first-light state: active `qwen36-27b-fp8-vllm` route serves `Qwen/Qwen3.6-27B-FP8` on `127.0.0.1:8014`, with LiteLLM alias `qwen36`.
- Recommendation: test TRT-LLM as a sidecar on a separate port before replacing vLLM, because vLLM is already integrated with LiteLLM, Langfuse, Loom, and ShadowKV workflows.

## [2026-06-01] update | first-light desktop environment documented
Author: Rover (first-light)
- KDE Plasma 6.5.5 with SDDM 0.21.0 installed on first-light
- Configured for X11 (`plasmax11` default session, SDDM Wayland disabled), but user is running Wayland session (kwin_wayland + Xwayland) — selected at SDDM login
- Updated `life/entities/first-light.md`: added Desktop Environment section, NixOS version, GPU driver details, Docker CDI config, clarified Tailscale IP, improved tags
- Bumped `updated` date to 2026-06-01

## [2026-06-01] ingest | Ousia datasets downloaded and inspected
Author: Rover (first-light)
- Downloaded `OusiaResearch/narrative-bench` (1,528 rows, MIT) and `OusiaResearch/Aureth-Agent-SFT-Robust` (243k rows, Apache 2.0) from Hugging Face
- Stored at `/datapool/datasets/ousia/` with symlinks in `~/research/ousia/`
- Schema inspected: narrative-bench is a 3×4×3 factorial evaluation benchmark (biography × persona × task). Aureth-SFT is a chat-format SFT curriculum with core/agentic/anti_sycophancy/func_call categories
- Created research entity page: `research/entities/ousia-research.md`
- Updated `life/projects/gpu-training-queue.md`: narrative-bench status READY, Aureth inspection status READY
- Updated `life/projects/gpu-training-log.md`: added full dataset intake details
- Updated `research/index.md`: added [[ousia-research]] entity entry
- Updated `life/index.md` and `research/index.md` frontmatter dates

## [2026-05-26] finding | ShadowKV vLLM APC observability clarified
Author: Keystone (Codex on IronGod)
- Ran a controlled exact-prefix APC probe against first-light's live `qwen36-27b-fp8-vllm` endpoint.
- vLLM `/metrics` showed a second-request cache hit of `15,680` prompt tokens, while the OpenAI-compatible response still returned `prompt_tokens_details: null`.
- Created [[life/projects/shadowkv-vllm-apc-observability-report-2026-05-26]] for Kushal and updated [[life/projects/nemotron-shadowkv-task-queue]].

### Follow-up | ShadowKV runtime patched and smoked
Author: Keystone (Codex on IronGod)
- Patched first-light `/home/jade_hand/research/shadowkv/literature_accurate_baselines/adapter_lib.py` and `run_runtime_cache_baseline.py` to scrape vLLM `/metrics` before/after measured runs.
- Added top-level vLLM cache deltas to benchmark JSON and optional `--trace_path` support for deterministic request traces.
- Container smoke passed; positive long-prefix smoke recorded nonzero vLLM APC deltas for both `vllm_apc` and `vllm_apc_shadowkv_plus`.

### Follow-up | Qwen2.5 full-attention matrix completed
Author: Keystone (Codex on IronGod)
- Stopped `qwen36-27b-fp8-vllm`, ran the patched 12-job Qwen2.5-14B vLLM APC matrix, then restored and health-checked `qwen36-27b-fp8-vllm`.
- Results landed under `/home/jade_hand/research/shadowkv/results_vllm_apc_qwen14b_metrics_2026-05-26`.
- Matrix total: 12 result JSONs, `82,816` vLLM prefix-cache hit tokens, and `0` response-level cached tokens.

## [2026-05-20] update | Root wiki log rotated
Author: Keystone (Codex on IronGod)
- Moved pre-2026-05-20 root log entries into [root-log-archive-2026-04-11-to-2026-05-19.md](_archive/logs/root-log-archive-2026-04-11-to-2026-05-19.md).
- Kept root [log.md](log.md) as the current rolling cross-domain ledger.
- Added [[logs/index]] as the log routing and activity index.

## [2026-05-20] update | Wiki maintenance routine candidate captured
Author: Keystone (Codex on IronGod)
- Created [[infrastructure/wiki-maintenance-routine-candidate]] so the Loom session can turn manual wiki health scans into a non-destructive routine.
- Added the routine candidate to [[radar]] under Infrastructure watch items.
- Added indexes for intentional empty staging folders and normalized Polaris control pages.
- Verification after cleanup: 280 active Markdown files, 36 missing frontmatter, 16 likely broken wikilinks, 0 ambiguous wikilinks, and 0 empty directories.

## [2026-05-20] update | Wiki cleanup pass 2 completed
Author: Keystone (Codex on IronGod)
- Migrated legacy `Wiki/Projects/` pages into canonical homes and left compatibility pointer stubs.
- Added frontmatter to root/domain logs and fixed active ambiguous links across A2A, Meta, schema, Nemotron, RAMS, and weekly-delta pages.
- Updated [[infrastructure/wiki-structure-audit-2026-05-20]] with pass 2 verification: 270 active Markdown files, 41 missing frontmatter, 34 likely broken wikilinks, 11 ambiguous wikilinks.

## [2026-05-20] update | Ousia benchmark and Qwen LoRA added to GPU queue
Author: Keystone (Codex on IronGod)
- Added `OusiaResearch/narrative-bench` eval on Qwen3.6-27B and Gemma 4 31B to [[life/projects/gpu-training-queue]].
- Added Qwen3.6-27B fine-tune on `OusiaResearch/Aureth-Agent-SFT-Robust` as the follow-on training item.
- Updated [[life/projects/gpu-training-log]] so the Ousia project has a durable baseline-before-training rule.

## [2026-05-20] create | GPU training log and queue created
Author: Keystone (Codex on IronGod)
- Created [[life/projects/gpu-training-log]] as the durable history page for first-light training, mining, eval, merge, and packaging work.
- Created [[life/projects/gpu-training-queue]] as the operational queue for GPU work, separating active/planned jobs from completed outcomes.
- Seeded the pages with the active Nemotron layered-teacher mining run and a blocked Qwen/Ousia personal LoRA intake lane.
- Restored Triton's NixOS `ldconfig` fallback in the first-light Unsloth venv after the layered-teacher mining queue failed during B eval, then restarted the queue.
- Updated [[life/index]] project links.

## [2026-05-20] cleanup | Wiki Projects legacy folder triage
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `Wiki/Projects/`, `Wiki/infrastructure/`, `Wiki/life/`, `Wiki/research/`, and `Wiki/Polaris/`.
- **Action:** Promoted the six legacy `Wiki/Projects/` notes into canonical homes and replaced the original files with compatibility pointer stubs.
- **Moved content:** Proton Bridge -> `infrastructure/proton-bridge-first-light.md`; biometric sleep sensor -> `life/projects/biometric-sleep-sensor.md`; biomimetic brain architecture -> `research/concepts/biomimetic-brain-architecture.md`; decision watcher and session rotation -> `Polaris/projects/`; wearable node power -> `life/concepts/wearable-agentic-node-power.md`.
- **Index updates:** Added canonical entries to infrastructure, life, research, and Polaris indexes.
- **Verification:** Follow-up checks after this pass: active wiki Markdown files `270`; missing frontmatter down to `45`; likely broken wikilinks still `24`, mostly old logs/raw references; ambiguous wikilinks down to `25`.
- **Reason:** `Wiki/Projects/` is now legacy/mixed; active/canonical material should live in routed domain folders while old links remain compatible.

## [2026-05-20] cleanup | Wiki structure cleanup pass 1
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `Wiki/index.md`, `Wiki/radar.md`, `Wiki/infrastructure/index.md`, `Wiki/life/index.md`, `Wiki/research/index.md`, `Wiki/life/SCHEMA.md`, `Wiki/research/SCHEMA.md`, `Wiki/research/raw/articles/index.md`, and selected active broken links.
- **Action:** Ran the first safe cleanup slice from `Wiki/infrastructure/wiki-structure-audit-2026-05-20.md`.
- **Changes:** Added frontmatter to active navigation/schema pages, deduplicated `infrastructure/index.md`, fixed root infrastructure link, fixed garage smart-lighting Jetson link, changed Nemotron experiment-log references to explicit external paths, added raw research articles index, and removed empty typo directory `research/raw/articlesmkdir`.
- **Verification:** Follow-up checks: active wiki Markdown files `264`; missing frontmatter down from `55` to `47`; likely broken wikilinks down from `29` to `24`; ambiguous wikilinks down from `30` to `28`. Remaining issues are mainly legacy/raw/log/Polaris cleanup and `Wiki/Projects/` migration.

## [2026-05-20] audit | Wiki structure audit and fix-list
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `Wiki/` active tree and `Wiki/infrastructure/wiki-structure-audit-2026-05-20.md`.
- **Action:** Ran a read-only structure/frontmatter/link/routing audit and wrote a prioritized fix-list.
- **Snapshot:** 262 active wiki Markdown files; 55 missing frontmatter; 23 with frontmatter but no `updated`; 29 likely broken wikilinks; 30 ambiguous wikilinks; 6 duplicate basename groups; 8 empty directories; root `Wiki/log.md` at 2,031 lines.
- **Main fix targets:** resolve legacy `Wiki/Projects/`, deduplicate `infrastructure/index.md`, add raw article index, rotate logs, remove typo directory `research/raw/articlesmkdir`, patch high-priority active broken links, and add frontmatter to navigation/control pages.

## [2026-05-20] planning | Unified project portfolio created
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `Wiki/life/projects/project-portfolio.md` and `Wiki/life/index.md`.
- **Action:** Created the canonical v1 project portfolio for Evan/JadeWire work.
- **Content:** Portfolio rules, status legend, operating priorities, overview table, active/incubating/future project cards, cross-project dependency map, candidate project data model, refresh protocol, and next actions.
- **Coverage:** Loom, first-light GPU training, tool/equipment inventory, lab renovation, JadeWire core app/data, spatial intelligence, drone lab, garage smart lighting, finance/Plaid, LoRA business, ShadowKV+, time-block manager, and MCP landscape.
- **Freshness caveat:** Portfolio is based on wiki state and recent logs, not a live re-check of every host/service/repo.

- **Author:** Keystone (Codex on IronGod)
- **Scope:** `/Users/evanleri/projects/loom`, Loom ops aggregation, operator CLI, docs, and `Wiki/infrastructure/loom-operating-doctrine.md`.
- **Context:** GPU remains reserved for training, so this pass avoided live runtime/vLLM probes by default.
- **Action:** Added `loom status` as a single read-only operator dashboard.
- **Behavior:** The command combines queue counts, stale running task details, scheduler routines blocked by active tasks, optimizer backlog, DB mirror state, service state, recent failed runs, recent low evaluations, blockers, and warnings.
- **Runtime safety:** `loom status` does not probe a runtime unless `--runtime <runtime_id>` is supplied.
- **Verification:** IronGod unittest suite passed with `63` tests. Focused tests cover the `operator_summary` aggregation path and `loom status --json` without runtime probing.
- **first-light check:** `LOOM_ACTIVE_STORE=postgres .venv/bin/python -m loom.cli status --json` completed without `--runtime`; service is active/running, DB mirror is current, stale running tasks are `0`, scheduler blockers are `0`, and runtime probing stayed disabled.
- **Deployment:** Synced to first-light without restarting `loom-worker-daemon.service`; no GPU-backed live test was run.
- **Next target:** Add a non-mutating `loom runs recent` or `loom failures` inspection command so optimizer/failure review can be done without hand-querying JSON/SQL.

## [2026-05-20] implementation | Loom agent-loop guardrail pipeline
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `/Users/evanleri/projects/loom`, `docs/agent-loop-guardrails-spec.md`, `loom/models.py`, `loom/agent_loop/guardrails.py`, `loom/agent_loop/executor.py`, `configs/base/loom.yaml`, and `Wiki/infrastructure/loom-operating-doctrine.md`.
- **Context:** Evan prioritized Forge-style guardrails as the useful pattern for Loom's agent loop.
- **Action:** Added a named guardrail pipeline around the existing agent-loop executor instead of creating a second runner.
- **Contracts:** Added `GuardrailFinding` and `GuardrailDecision` for typed runtime findings and decisions.
- **Guardrails:** Implemented default stack configuration for `tool_validity`, `step_budget`, `loop_detection`, `premature_stop`, `required_evidence`, `output_contract`, `safety_envelope`, and `escalation`.
- **Runtime behavior:** Unknown/disallowed tools are denied before execution; constrained arguments remain runtime-enforced; duplicate completed tool calls redirect toward final synthesis; final-quality checks now surface typed `guardrail:<code>` flags for evaluator/optimizer use.
- **Verification:** IronGod CPU-only unittest suite passed with `77` tests. No live model/vLLM testing was run because GPU capacity was reserved for training.
- **Next target:** Run `ops-observer-default` through a real guarded background routine when model capacity is available, then verify optimizer intake clustering by guardrail code.

## [2026-05-20] implementation | Loom OpenRouter DeepSeek V4 Flash fallback
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `/Users/evanleri/projects/loom`, `configs/runtimes/cloud-openai.yaml`, `configs/runtimes/openrouter-deepseek-v4-flash.yaml`, `loom/executors/runtime.py`, `loom/worker.py`, daemon/background docs, and Loom tests.
- **Context:** first-light GPU is occupied with dataset mining, so live Loom tests need a cloud backup that can replace Qwen3.6-27B temporarily.
- **Action:** Repointed the existing `cloud-openai` fallback runtime to OpenRouter `deepseek/deepseek-v4-flash` while preserving the legacy runtime id for existing specialist fallback policies.
- **Runtime:** Added explicit `openrouter-deepseek-v4-flash` runtime config. Both use `https://openrouter.ai/api`, model `deepseek/deepseek-v4-flash`, action protocol `loom-json`, context window `1048576`, and `OPENROUTER_API_KEY`.
- **Failover:** Added one-shot live executor fallback in the worker: if the selected live runtime raises before producing a result, Loom recompiles the same run against the configured fallback runtime and retries once.
- **Verification:** IronGod CPU-only unittest suite passed with `79` tests. No live OpenRouter call was run because `OPENROUTER_API_KEY` was not present in the current shell environment.
- **Next target:** Export `OPENROUTER_API_KEY` on first-light, then run a guarded `ops-observer-default` live smoke with `--runtime cloud-openai` before enabling broader background testing.

### Follow-up | `.hermes/.env` lookup
- **Author:** Keystone (Codex on IronGod)
- **Action:** Added `env_file: /home/jade_hand/.hermes/.env` to the OpenRouter runtime configs and taught Loom's runtime resolver to read `api_key_env` from that file when the key is not already exported.
- **Finding:** first-light `.hermes/.env` contains an active `OPENROUTER_API_KEY` line, but the active value currently resolves as empty/invalid after quote stripping, so Loom still reports `has_key: False` and OpenRouter returned `401 Unauthorized` during smoke.
- **Verification:** IronGod unittest suite passed with `80` tests. first-light unittest suite passed with `80` tests after sync.
- **Next target:** Replace the active `.hermes/.env` `OPENROUTER_API_KEY` value with a valid OpenRouter key, then rerun `scripts/runtime_smoke.py cloud-openai`.

### Follow-up | OpenRouter fallback live smoke passed
- **Author:** Keystone (Codex on IronGod)
- **Action:** After Evan updated the first-light OpenRouter key, reran runtime resolution and live smoke tests.
- **Verification:** Loom resolved `cloud-openai` with `has_key: true`; `scripts/runtime_smoke.py cloud-openai` passed plain chat and native tool-call checks against `deepseek/deepseek-v4-flash`.
- **Live Loom run:** Enqueued and processed read-only task `task_4d7481236f9940eebd47a5660e1c6715` with `loom worker once --executor agent-loop --runtime cloud-openai`.
- **Run result:** `run_40b96c85c3dc416fadeeda1d64a4c773` completed through `deepseek/deepseek-v4-flash`, used `read_recent_runs`, wrote one artifact, recorded `guardrail:evidence_budget_complete`, and scored `95.0`.
- **Maintenance:** Patched `scripts/runtime_smoke.py` to summarize OpenRouter `/v1/models` instead of dumping the full catalog.
- **Next target:** Use `cloud-openai` for guarded live Loom tests while first-light GPU is occupied.

## [2026-05-21] implementation | Loom prompt stack layer split
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `/Users/evanleri/projects/loom/prompts/`, specialist configs, policy compiler, prompt architecture docs, and first-light synced Loom repo.
- **Context:** Evan confirmed `time-keeper-default-briefing.md` had the right shape and asked to split it into layers, then build layer 4/5/6 equivalents for the rest of the agents.
- **Action:** Split the Time-Keeper sample structure into reusable prompt layers and extended that layer pattern across active Loom agents.
- **Layer 1:** Expanded `prompts/base/system.md` into the shared Loom operating frame.
- **Layer 2:** Added node frames under `prompts/nodes/` for `first-light`, `irongod`, `soren-edge`, and `wax-glove`; policy compilation now auto-inserts `prompts/nodes/<node>.md` after the base prompt when present.
- **Layer 3/4:** Expanded background and role prompts for Time-Keeper, Ops Observer, Research Scout, Wiki Gardener, Filesystem Janitor, Evaluator, and Optimizer.
- **Layer 5:** Added specialist frames for active default agents under `prompts/specialists/`.
- **Layer 6:** Added doctrine layers for calendar/timekeeping, ops observability, wiki hygiene, filesystem schema, research evidence, and evaluation.
- **Config:** Updated active specialist `prompt_stack`s to include specialist and doctrine layers while leaving node insertion automatic.
- **Verification:** `loom prompts render --agent time-keeper-default --node first-light --runtime cloud-openai` and `ops-observer-default` render in the intended layer order. IronGod and first-light unittest suites both passed with `81` tests.
- **Next target:** Implement `PromptDossier` and `PromptBriefing` contracts plus `loom prompts render --with-context`, then update the agent loop to use the rendered briefing as its system prompt.

## [2026-05-21] implementation | Loom prompt dossier and briefing renderer
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `/Users/evanleri/projects/loom/loom/models.py`, `loom/prompts/context.py`, `loom/cli.py`, prompt tests, and prompt architecture docs.
- **Context:** After splitting the prompt stack into clean base/node/role/specialist/doctrine layers, the next vertical step was to turn those layers into an auditable model-facing briefing rather than a raw concatenated stack.
- **Action:** Added `PromptBlock`, `PromptDossier`, and `PromptBriefing` contracts; implemented `loom prompts render --with-context`; implemented `loom prompts stats --with-context`; and added dynamic file-block loading with explicit uncertainty for unavailable sources.
- **Behavior:** The rendered prompt remains Markdown-first and human-readable. Source hashes, token estimates, context block statuses, and failure details live in the dossier metadata rather than being dumped into the prompt body.
- **Executor adoption:** Added run-time prompt attachment so worker-created runs store `prompt_dossier` and `prompt_briefing` metadata, persist the rendered system prompt under `data/prompt_briefings/<run_id>/system.md`, and route openai-compatible plus agent-loop execution through the rendered Markdown briefing.
- **Dynamic selectors:** Added renderer support for run-envelope context, effective allowed/forbidden tool scope, and runtime config summaries. These now become included briefing blocks during worker-created runs instead of remaining abstract planned selectors.
- **Verification:** IronGod focused prompt-render tests passed. Full CPU suite passed with `83` tests. Local render smoke passed for `time-keeper-default` with the calendar doctrine file loaded from the wiki; local stats smoke passed for `ops-observer-default`, correctly marking first-light-only service manifest context unavailable from IronGod.
- **first-light verification:** Synced the slice to `/home/jade_hand/projects/loom`; first-light CPU suite passed with `83` tests. `loom prompts stats --agent ops-observer-default --node first-light --runtime cloud-openai --with-context --json` loaded `/home/jade_hand/ops/SERVICE_MANIFEST.md` as an included dynamic file block.
- **Next target:** Add real dynamic selector adapters for run envelope, tool names, and runtime state; then run a guarded cloud fallback live task to evaluate richer prompts without touching the busy GPU.

## [2026-05-22] implementation | Loom small-agent draft workflow
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `/Users/evanleri/projects/loom/loom/cli.py`, `tests/test_contract_runtime.py`, `docs/agent-creation-guide.md`, and `docs/agent-factory-and-weave-assembly.md`.
- **Context:** Evan asked for the agent creation flow to become intuitive enough that smaller agents could carry it out without hand-building every stack layer.
- **Action:** Added `loom agents draft`, which creates a runnable specialist by inheriting scope, stack shape, model policy, role layers, and doctrine layers from an existing agent while replacing only the specialist manifest/prompt and optional prompt-context stub.
- **Guide:** Added `docs/agent-creation-guide.md` with the small-agent decision rule, draft command, validation gate, launch commands, and editing rules.
- **Verification:** IronGod CPU suite passed with `84` tests. Synced to first-light and first-light CPU suite passed with `84` tests.
- **Next target:** Add blueprint preview/promotion gates when the simple draft path proves stable, then connect drafted agents into Weave templates.
## [2026-05-23] update | Nemotron data-quality gate before GPU work
- **Author:** Keystone (Codex on IronGod)
- **Scope:** [[life/projects/nemotron-data-quality-gate]], [[life/projects/gpu-training-queue]], [[life/projects/gpu-training-log]], first-light `/home/jade_hand/projects/nemotron_pipeline`.
- **Context:** Evan halted further GPU use on generated/mined Nemotron data until the data engines and validation scripts are proven reliable.
- **Action:** Added a Bronze/Silver/Gold data-quality gate for Nemotron datasets. The queue now requires Gold validation before GPU mining or training.
- **Next target:** Implement `scripts/validate_candidate_dataset.py` on first-light, run it over current Huikang residual and generated hard-row datasets, and repair engines/selectors before any new GPU job.

### Follow-up | first CPU validation pass
- **Author:** Keystone (Codex on IronGod)
- **Action:** Installed `scripts/validate_candidate_dataset.py` and `scripts/build_validated_training_jsonl.py` on first-light.
- **Finding:** The Huikang residual source tier is Gold, but `train_huikang_residual_expanded_2026-05-22.jsonl` is only Bronze because it uses notebook-style boxed prompts outside original B distribution.
- **Correction:** Exported `data/prepared/train_huikang_residual_bstyle_gold_2026-05-23.jsonl`, preserving B-style prompts and B/Huikang difficulty metadata.
- **Validation:** Fresh 600-row balanced engine sample passed Silver with zero issues and engine oracle smoke passed for all six tasks. Filtered B5 hard-extension into `outputs/data_quality_gate/2026-05-23/B5_hard_extension_silver_rows.csv` with 44,597 clean Silver rows. No GPU job launched.

### Follow-up | B7 Gold-row continuation launched
- **Author:** Keystone (Codex on IronGod)
- **Action:** Added deterministic CPU solver promotion for B-miss rows and produced 31,775 Gold decrypt/gravity/unit rows.
- **Dataset:** Built `data/prepared/train_B7_gold_solver_replay_mix_2026-05-24.jsonl` with 100,349 rows: trusted original B replay plus Gold new rows only.
- **Training:** Launched `scripts/run_B7_gold_solver_train_eval_2026_05_24.sh` from `outputs/adapters/B_best_for_sweep`, LR `1e-6`, epochs `0.20`, batch `32`.
- **Runtime:** qwen is stopped by the queue while training/eval runs and should be restored afterward.

### Follow-up | B8 Gold-row v2 staged
- **Author:** Keystone (Codex on IronGod)
- **Action:** Extended the CPU Gold promotion gate with conservative bit and symbol solvers. Ambiguous rows are rejected; only rows whose prompt examples determine a single solver answer are promoted.
- **Result:** `data/prepared/train_B5_solver_verified_b_miss_gold_v2_2026-05-24.jsonl` has 42,767 B-miss Gold rows and passes validation with zero issues.
- **Next:** B7 eval is running. B8 is staged at `data/prepared/train_B8_gold_solver_v2_replay_mix_2026-05-24.jsonl`; launch after B7 eval only after choosing whether to initialize from B7 or original B.

### Follow-up | B9 pooled Gold staged
- **Author:** Keystone (Codex on IronGod)
- **Action:** Applied the same strict solver gate to older hard/B-miss pools and deduped them by stripped prompt plus normalized answer.
- **Result:** `data/prepared/train_solver_verified_b_miss_gold_pool_v2_2026-05-24.jsonl` has 85,122 unique Gold rows and passes validation with zero issues. A capped balanced training mix is staged at `data/prepared/train_B9_balanced_gold_pool_mix_2026-05-24.jsonl`.
- **Next:** Wait for B7 eval. If B7 is acceptable, use it as the B9 initializer; if B7 regresses broadly, launch B9 from `outputs/adapters/B_best_for_sweep` instead.

### Follow-up | B7 rejected, B9 launched
- **Author:** Keystone (Codex on IronGod)
- **Finding:** B7 mildly regressed versus B on A/B/C/E and only improved D: A `-0.009474`, B `-0.024000`, C `-0.002137`, D `+0.004000`, E `-0.008000`.
- **Action:** Rejected B7 as the next base and launched B9 from `outputs/adapters/B_best_for_sweep`, not from B7.
- **Runtime:** B9 queue is active at `outputs/b9_balanced_gold_pool_queue_2026-05-24/STATUS.txt`; qwen is stopped while training/eval runs.

### Follow-up | B9 rejected, B10 conservative fallback launched
- **Author:** Keystone (Codex on IronGod)
- **Finding:** B9 regressed versus original B on four of five slices: A `-0.010526`, B `-0.022000`, C `+0.002137`, D `-0.004000`, E `-0.010000`.
- **Action:** Rejected B9 as a continuation base and launched B10 from `outputs/adapters/B_best_for_sweep`.
- **Runtime:** B10 queue is active at `outputs/b10_conservative_gold_pool_queue_2026-05-24/STATUS.txt`; qwen is stopped while training/eval runs.

### Follow-up | B11 fallback staged from error transitions
- **Author:** Keystone (Codex on IronGod)
- **Finding:** B9 per-task transitions show the broad pooled-Gold regression is concentrated in `unit` and `gravity`; `decrypt` improves and `symbol` is mixed but non-destructive.
- **Action:** Staged B11 as a no-unit/no-gravity Gold fallback while B10 runs.
- **Artifacts:** `data/prepared/train_B11_no_unit_gravity_gold_pool_mix_2026-05-24.jsonl`; `scripts/run_B11_no_unit_gravity_gold_pool_train_eval_2026_05_24.sh`; analysis at `outputs/data_quality_gate/2026-05-24/B7_B9_error_transition_by_task.json`.

### Follow-up | B10 rejected, B11 launched
- **Author:** Keystone (Codex on IronGod)
- **Finding:** B10 improved C but still regressed A/B/D: A `-0.004210`, B `-0.010000`, C `+0.004274`, D `-0.006000`, E `+0.000000`.
- **Action:** Kept B10 as diagnostic only and launched B11 from `outputs/adapters/B_best_for_sweep`.
- **Runtime:** B11 queue is active at `outputs/b11_no_unit_gravity_gold_pool_queue_2026-05-24/STATUS.txt`; qwen is stopped while training/eval runs.

### Follow-up | B12 symbol-only fallback staged
- **Author:** Keystone (Codex on IronGod)
- **Action:** Staged a symbol-only Gold fallback in case B11 still regresses broad retention.
- **Artifacts:** `data/prepared/train_B12_symbol_gold_pool_mix_2026-05-24.jsonl`; `scripts/run_B12_symbol_gold_pool_train_eval_2026_05_24.sh`.

### Follow-up | B11 aborted, B12 launched
- **Author:** Keystone (Codex on IronGod)
- **Finding:** B11 failed the early retention gate: A `0.512632`, B `0.426000`, C `0.570513`.
- **Action:** Aborted the remaining B11 eval slices and launched B12 symbol-only Gold from `outputs/adapters/B_best_for_sweep`.
- **Runtime:** B12 queue is active at `outputs/b12_symbol_gold_pool_queue_2026-05-24/STATUS.txt`; qwen is stopped while training/eval runs.

### Follow-up | B12 recovery and B-alignment gate
- **Author:** Keystone (Codex on IronGod)
- **Finding:** B12 reached the end of training, but first-light `/home` hit 100% before status/eval handoff completed. The top-level adapter directory only had `adapter_model.safetensors`; the full PEFT package was under `checkpoint-303`.
- **Action:** Removed resume-only checkpoint payloads (`optimizer.pt`, `scheduler.pt`, `rng_state.pth`) to restore about `99G` free without deleting adapter weights, then restarted B12 eval from `outputs/adapters/B12_symbol_gold_pool_unsloth_qlora_b32_ga1_lr5e7_ep012_2026-05-24/checkpoint-303`.
- **Data gate:** Wrote `outputs/data_quality_gate/2026-05-24/b_alignment_profile_2026-05-24.md`. It records original B's near-even task mix and flags B12's symbol-heavy skew as `1.75x` original B, making B-style distribution fit an explicit pre-GPU gate.

### Follow-up | B12 rejected and fresh Silver pool prepared
- **Author:** Keystone (Codex on IronGod)
- **Finding:** B12 improved A to `0.517895`, but B hard dropped to `0.420000` and C dropped to `0.574786`. The early retention gate aborted D/E.
- **Action:** Rejected B12 as a continuation base and restarted qwen. Generated a fresh 300k balanced engine reservoir on `/datapool`; quality gate accepted `278,370` rows.
- **Validator fix:** Patched `scripts/validate_candidate_dataset.py` so symbol answers preserve backticks and punctuation. The 90k balanced sample now validates as Silver with `90,000/90,000` rows and zero issues.
- **Next:** Use the fresh Silver pool for difficulty mining/Qwen sanity checks, not direct training, because it lacks B-miss difficulty signatures.

### Follow-up | B13 conservative B-style Gold launched
- **Author:** Keystone (Codex on IronGod)
- **Finding:** B-mining the fresh 90k Silver sample produced `43,131` B-miss solver-Gold rows with Gold validation clean across all rows. B solved all sampled Roman rows, so the new Gold signal is non-Roman: bit `2,633`, decrypt `9,423`, gravity `14,104`, symbol `10,278`, unit `6,693`.
- **Action:** Built and launched B13 as a conservative retention test: `68,550` deduped original-B replay rows plus `10,000` capped new Gold rows, initialized from `outputs/adapters/B_best_for_sweep`, LR `3e-7`, epochs `0.08`, batch `32`.
- **Runtime:** B13 queue is active at `outputs/b13_ultraconservative_bstyle_gold_queue_2026-05-24/STATUS.txt`; qwen is stopped while training/eval runs.

### Follow-up | B13 rejected; solver-router path is stronger
- **Author:** Keystone (Codex on IronGod)
- **Finding:** B13 completed but failed the early B-hard retention gate: A `0.515789`, B-hard `0.430000` versus B baseline `0.434000`; D/E were skipped. The regression is mostly tiny unit rounding drift and a few bit flips.
- **Action:** Rejected B13 as a base, restored qwen, cleaned B13 resume-only checkpoint payloads, and installed `scripts/apply_solver_router.py`.
- **Result:** B plus validation-safe deterministic solvers projects far above adapter-only B without more training: A `.764021`, B `.752000`, C `.817204`, D `.779559`, E `.750503`, with zero solver-caused regressions in this check.
- **Next:** Improve bit/symbol residual handling and only train on residual rows that cannot be routed deterministically and have a verified independent oracle or stronger-teacher signal.

### Follow-up | Qwen residual probe is not useful
- **Author:** Keystone (Codex on IronGod)
- **Action:** Used qwen36-27B-FP8 as an answer-only teacher on `612` B-miss rows that the deterministic router could not solve (`112` bit, `500` symbol).
- **Result:** qwen solved only `2/612`, both symbol, and `0/112` bit. This is not a reliable teacher signal for the remaining residual class under the current prompt/serving setup.
- **Artifacts:** `outputs/data_quality_gate/2026-05-24/router_residual_b_miss_qwen_probe_results.csv` and `.manifest.json`.

### Follow-up | B fingerprint hard selector installed
- **Author:** Keystone (Codex on IronGod)
- **Action:** Reconstructed B back to source rows and installed B-distribution profiling/selection scripts.
- **Finding:** B is exactly `60,000` synthetic v4 rows plus `17,100` real-fit rows, with `10,000` synthetic rows per task and about `2,800-2,884` real rows per task.
- **Result:** First B-profile hard selection from the fresh 90k pool produced `5,000` B-miss rows; `4,789` promoted to solver-Gold and passed validation with zero issues.
- **Artifacts:** `outputs/data_quality_gate/2026-05-24/b_distribution_fingerprint_resolved.md`, `outputs/data_quality_gate/2026-05-24/b_profile_hard_rows_from_90k.solver_gold.validate.json`, and `data/prepared/train_b_profile_hard_rows_from_90k_solver_gold_2026-05-24.jsonl`.

### Follow-up | B14 B-profile checkpoint sweep launched
- **Author:** Keystone (Codex on IronGod)
- **Context:** Overnight Nemotron experiment on first-light after B13 failed but B-profile Gold rows validated cleanly.
- **Action:** Launched `scripts/run_B14_bprofile_gold_checkpoint_sweep_2026_05_24.sh` under nohup. The run stops `qwen36-27b-fp8-vllm`, trains from `outputs/adapters/B_best_for_sweep` on `data/prepared/train_b_profile_hard_rows_from_90k_solver_gold_2026-05-24.jsonl`, saves checkpoints every 50 steps, evaluates every checkpoint plus final adapter across A-E, ranks the best checkpoint, and restores qwen.
- **Config:** LR `3e-7`, epochs `3.0`, batch `32`, grad accum `1`, config `configs/blackwell_rtx6000_unsloth_qlora_bprofile_checkpoints_2026_05_24.yaml`.
- **Artifacts:** Queue `outputs/b14_bprofile_gold_sweep_queue_2026-05-24/STATUS.txt`; train log `outputs/logs/b14_bprofile_gold_sweep_train_2026-05-24/B14_bprofile_gold_sweep_from_B_b32_ga1_lr3e7_ep3_2026-05-24.train.log`; eval dir `outputs/eval_B14_bprofile_gold_checkpoint_sweep_b32_2026-05-24`.

### Follow-up | B14 checkpoint sweep complete
- **Author:** Keystone (Codex on IronGod)
- **Finding:** B14 completed and did not beat original B. The best point was `checkpoint-50`, with avg exact `0.502976`, avg delta `-0.002577`, B-hard delta `-0.004000`, and only C improved (`+0.004274`).
- **Decision:** Reject B14 as a base. The B-profile Gold rows still look useful as router/diagnostic data, but adapter-only continuation continues to introduce retention loss.
- **Runtime:** Training completed `2026-05-24T22:58:17-04:00`; eval completed `2026-05-25T01:15:06-04:00`; qwen36-27B-FP8 was restored and responds on `127.0.0.1:8014/v1/models`.
- **Artifacts:** `outputs/eval_B14_bprofile_gold_checkpoint_sweep_b32_2026-05-24/best_checkpoint.md`; `outputs/eval_B14_bprofile_gold_checkpoint_sweep_b32_2026-05-24/checkpoint_comparison_vs_B.md`.

### Follow-up | B15 base retrain sweep launched
- **Author:** Keystone (Codex on IronGod)
- **Context:** Evan asked whether training the base model on the same v4/original B dataset with more aggressive LR/epochs could outperform B.
- **Action:** Launched `scripts/run_B15_base_v4_lr_epoch_sweep_2026_05_25.sh` under nohup. The run stops qwen, trains from the base model on `data/prepared/train_B_synthetic_heavy.jsonl`, evaluates checkpoints across A-E, ranks against original B, and restores qwen after completion/failure.
- **Sweep:** `B15a` LR `5e-5` epoch `1.0`; `B15b` LR `1e-4` epoch `1.0`; `B15c` LR `1e-4` epochs `2.0`; `B15d` LR `2e-4` epoch `1.0`; `B15e` LR `2e-4` epochs `2.0`. Batch `32`, grad accum `1`, checkpoint every `600` steps.
- **Artifacts:** Queue `outputs/b15_base_v4_lr_epoch_sweep_2026-05-25/STATUS.txt`; adapter root `/datapool/datasets/nemotron/adapters/B15_base_v4_lr_epoch_sweep_2026-05-25`; eval dir `outputs/eval_B15_base_v4_lr_epoch_sweep_b32_2026-05-25`.

### Follow-up | B15 complete and plain Huikang compared
- **Author:** Keystone (Codex on IronGod)
- **Finding:** B15 completed and produced a new best adapter candidate: `B15e_base_v4_lr2e4_ep2__checkpoint-4820`. It scores A `0.560000`, B-hard `0.488000`, C `0.632479`, D `0.558000`, E `0.544000`, avg exact `0.556496`, avg delta `+0.050943` versus B.
- **Huikang check:** Plain Huikang without layering is far weaker on the same eval: A `0.195789`, B-hard `0.044000`, C `0.241453`, D `0.162000`, E `0.198000`.
- **Decision:** Promote B15e checkpoint-4820 as the next adapter candidate and evaluate solver-router projection on top of it. Do not use raw Huikang as a base under this evaluator.
- **Runtime:** B15 completed `2026-05-26T13:09:49-04:00`; Huikang plain eval completed `2026-05-26T13:31:53-04:00`; qwen36-27B-FP8 restored and responding.
- **Artifacts:** Best B15 report `outputs/eval_B15_base_v4_lr_epoch_sweep_b32_2026-05-25/best_checkpoint.md`; Huikang comparison `outputs/eval_huikang_plain_b32_2026-05-26/comparison_vs_B_and_best_B15.md`.

### Follow-up | ShadowKV main-variable gate checked
- **Author:** Keystone (Codex on IronGod)
- **Finding:** Qwen2.5 vLLM APC metrics are now real for exact-prefix caching, but the current vLLM backend cannot expose external `past_key_values` for ShadowKV semantic/partial KV reuse. The real external-KV path is HuggingFace; semantic/partial reuse is currently publishable only as fake-backend simulation/opportunity data unless a safe real-backend path is implemented and validated.
- **Action:** Added deterministic `--trace_path` support to `experiments/run_benchmark.py`, fixed nested trace metadata loading in `literature_accurate_baselines/adapter_lib.py`, and ran a fake-backend semantic smoke that produced `semantic_partial_hits=4` and `semantic_opportunity_reused_tokens_total=310`.
- **Artifacts:** Report `Wiki/life/projects/shadowkv-vllm-apc-observability-report-2026-05-26.md`; traces `/home/jade_hand/research/shadowkv/session_files/main_variable_traces_2026-05-26/`; smoke result `/home/jade_hand/research/shadowkv/results_main_variable_smoke_2026-05-26/fake_semantic_no_scaffold_long/benchmark_fake_default_synthetic_high_skew_cpu.json`.

### Follow-up | ShadowKV Qwen2.5-32B vLLM policy-overlay run complete
- **Author:** Keystone (Codex on IronGod)
- **Context:** Kushal clarified that Blackwell large-model results should use vLLM runtime baselines and treat ShadowKV++ as an admission/policy overlay, not direct `past_key_values` injection.
- **Action:** Ran Dockerized Qwen2.5-32B vLLM APC and `vllm_apc_shadowkv_plus` across `daily_dialog`, `samsum`, and `ag_news`, with `templated` and `rag` prompts at `256` measured requests/job. Added generic `--warmup_requests` to the runtime runner and used a warmup-parity result root.
- **Result:** 12/12 full jobs completed. Total prompt tokens `607,084`; vLLM prefix-cache hit tokens `352,352`; response-level cached tokens `0`; ShadowKV++ planned `256`, allowed `255`, bypassed `1`, and stored `256` in every overlay job. Production `qwen36-27b-fp8-vllm` was restored and health-checked.
- **Artifacts:** `/home/jade_hand/research/shadowkv/results_vllm_apc_qwen32b_policy_overlay_warmup_parity_2026-05-27`; aggregate CSV/JSON in that root; run log `/home/jade_hand/research/shadowkv/run_logs/qwen32b_policy_overlay_warmup_parity_20260527.log`; report updated at `Wiki/life/projects/shadowkv-vllm-apc-observability-report-2026-05-26.md`.

### Follow-up | first-light local KDE Plasma X11 enabled
- **Author:** Keystone (Codex on IronGod)
- **Context:** Evan asked to set up KDE Plasma locally on first-light, using SSH only as the control path.
- **Action:** Updated `/etc/nixos/configuration.nix` to disable `greetd`, enable `services.xserver`, enable SDDM with `wayland.enable = false`, enable Plasma 6, and set `services.displayManager.defaultSession = "plasmax11"`.
- **Verification:** `nixos-rebuild test` and `nixos-rebuild switch` succeeded. `display-manager.service` is active as SDDM, `/etc/sddm.conf` has `DisplayServer=x11` and `DefaultSession=plasmax11.desktop`, the X server is running on seat0, and `systemctl --failed` reports zero failed units.
- **Artifacts:** Config backups `/etc/nixos/configuration.nix.bak.20260601-114636.pre-kde-plasma-x11` and `/etc/nixos/configuration.nix.bak.20260601-114938.pre-plasmax11-default`; ops manifest updated at `/home/jade_hand/ops/SERVICE_MANIFEST.md`.

## [2026-06-10] artifact | Loom component progress dashboard
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `/Users/evanleri/projects/loom/docs/loom-component-progress.html`.
- **Context:** Evan asked for an HTML file to visualize Loom components and current progress.
- **Action:** Created a standalone static HTML dashboard with a system-flow diagram, progress scoreboard, filter/search controls, component cards, runtime posture, status legend, and next-vertical roadmap.
- **Sources:** Current Loom docs and repo state, especially `docs/v1-build-plan.md`, `docs/codebase-map.md`, `docs/agent-factory-and-weave-assembly.md`, `docs/rich-system-prompt-architecture.md`, and current config/module layout.
- **Verification:** Confirmed the file exists and contains the expected dashboard sections; no dev server is required.

### Follow-up | Loom component dashboard QA pass
- **Author:** Keystone (Codex on IronGod)
- **Action:** Reviewed `/Users/evanleri/projects/loom/docs/loom-component-progress.html` for UI behavior, component accuracy, and visual structure.
- **Accuracy checks:** Compared dashboard claims against `loom agents list --json`, `loom policy validate --all --node first-light --runtime cloud-openai --json`, `loom prompts stats --agent ops-observer-default --node first-light --runtime cloud-openai --with-context --json`, runtime configs, current docs, and tool catalog state.
- **Corrections:** Clarified that policy validation has warnings for uncataloged planned tools; downgraded the tool registry from functional to partial; added a Specialist Agent Set component; adjusted affected evaluator/optimizer/prompt wording; added explicit verification notes and confidence labels.
- **UI checks:** Added accessible filter state, reset button, responsive metric grid, status-colored progress bars, focus styles, and clearer responsive behavior. Static DOM-level tests confirmed six filters, seven metrics, 20 component cards, working filter/search/reset behavior, and no stale in-app-browser verification claim.
- **Limit:** The in-app Browser tool refused direct interaction with the `file://` page under its URL policy, so browser-click testing was not performed through that tool.

## [2026-06-10] artifact | Loom system canvas populated
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `/Users/evanleri/CONTINUITY/Wiki/infrastructure/loom-system-canvas.canvas`.
- **Context:** Evan created an Obsidian Canvas beside the Loom ideas inbox and asked Keystone to fill it with Loom components as cards.
- **Action:** Replaced the starter blank/Postgres canvas with a structured Loom map containing foundation, runtime, optimization, orchestration, and future-interface component cards.
- **Contents:** Added cards for contracts, policy compiler, prompt briefing, specialist agents, task queue, Postgres, agent loop, tools, scheduler, runtime matrix, evaluator, optimizer, benchmarks, Agent Factory, Weaves, Watchpoints, user inbox, decision-agent RAG, capability expansion swarm, Hermes/Pi adapters, and next vertical priorities.
- **Verification:** JSON validated successfully. Canvas has `27` nodes, `24` edges, and no broken edge references.

## [2026-06-10] artifact | Evan Loom system skeleton canvas populated
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `/Users/evanleri/CONTINUITY/Wiki/infrastructure/loom-system-evan-canvas.canvas`.
- **Context:** Evan created a second Obsidian Canvas with his name in it and asked for an idealized system map describing each entity's purpose, inputs, and outputs.
- **Action:** Populated the canvas as a design skeleton rather than a current-state progress map. Each major entity is represented as a card with `Purpose`, `Consumes`, and `Outputs` sections.
- **Contents:** Includes user intent, user-facing surfaces, verified message gateway, canonical inbox repository, inbox manager, signal router, task queue, worker, policy/prompt compiler, runtime/model layer, tool layer, agent loop, artifact store, Loom DB, JadeWire DB, wiki, scheduler, Weave Factory, Agent Factory, Watchpoints, Evaluator, Optimizer, Decision RAG, Capability Expansion Swarm, Research Swarms, and Security/Permission Swarm.
- **Verification:** JSON validated successfully. Canvas has `27` nodes, `32` edges, and no broken edge references.

## [2026-06-13] benchmark | Darwin-28B Loom Trial-By-Fire smoke
- **Author:** Keystone (Codex on IronGod)
- **Scope:** `FINAL-Bench/Darwin-28B-REASON` served by first-light vLLM as Loom runtime `darwin28b-reason`.
- **Action:** Added Loom runtime config `configs/runtimes/darwin28b-reason.yaml` and ran `loom bench trial-by-fire --runtime darwin28b-reason --profile smoke --seed 1` on first-light.
- **Result:** 10 scenarios, 6 passed, 4 failed, pass rate `60.00%`, average score `90.50`, elapsed `796.90s`.
- **Passed:** handoff, multimodal, optimizer, time-keeper, wiki-gardener, research-scout.
- **Failed:** evaluator (`75.00`, missed pytest/no-test-evidence/hard-failure concepts), code-worker (`86.67`, missed pytest and do-not-rewrite concepts), filesystem-janitor (`50.00`, parser errors and missing compatibility/symlink/rollback/protected concepts), ops-observer (`93.33`, missed database-locked concept).
- **Artifacts:** first-light `/home/jade_hand/projects/loom/data/benchmarks/trial_by_fire_darwin28b_reason_smoke_20260613-202600.{jsonl,md}`; IronGod copy `/Users/evanleri/projects/loom/data/benchmarks/trial_by_fire_darwin28b_reason_smoke_20260613-202600.{jsonl,md}`.

## [2026-06-14] benchmark | SGLang LMCache no-native-Radix baseline
- **Author:** Keystone (Codex on IronGod)
- **Scope:** first-light ShadowKV/SGLang experiment with `Qwen/Qwen2.5-14B-Instruct`.
- **Action:** Built patched Docker image `shadowkv-sglang-lmcache:2026-06-14-no-native-radix` so `--enable-lmcache --disable-radix-cache` keeps the LMCache path active while preventing native SGLang Radix from satisfying hits. Ran a controlled smoke and the 5 dataset x 2 mode Qwen14B matrix.
- **Result:** Smoke passed without manual `/flush_cache`: repeated long-prefix requests returned `cached_tokens=2560`, with LMCache log evidence for one store and two retrieves. Matrix completed `10/10`; aggregate mean latency `51.56 ms`, P95 `62.97 ms`, throughput `19.61 rps`, idle-adjusted energy `23.74` J/request, cached tokens `7168`, LMCache retrieve events `27`, store events `242`.
- **Artifacts:** Result root `/home/jade_hand/research/shadowkv/results_lmcache_no_native_radix_qwen14b_matrix_2026-06-14`; smoke root `/home/jade_hand/research/shadowkv/results_lmcache_no_native_radix_qwen14b_smoke_2026-06-14`; project report `Wiki/life/projects/shadowkv-sglang-small-model-test-2026-06-13.md`.
- **Verification:** Production `darwin28b-reason-vllm` was restored and `http://127.0.0.1:8015/v1/models` returned `darwin28b-reason`; no no-native-Radix benchmark container remained active.

## [2026-06-16] benchmark | ShadowKV SGLang three-model repeated run
- **Author:** Keystone (Codex on IronGod)
- **Scope:** first-light ShadowKV/SGLang matrix across Qwen2.5 `1.5B`, `3B`, and `7B`, with native Radix, ShadowKV++ overlay, and patched LMCache no-native-Radix.
- **Action:** Ran 270 randomized-order jobs: 3 models x 5 datasets x 2 prompt modes x 3 baselines x 3 repetitions, 256 requests per cell, shared runtime image `shadowkv-sglang-lmcache:2026-06-14-no-native-radix`, shared runtime settings, NVML energy, and metadata capture.
- **Result:** 270/270 jobs completed. ShadowKV++ was parity/slightly slower on 1.5B and 3B, faster on 7B; patched LMCache no-native-Radix worked across all models but was slower than native Radix because it lost most fine-grained Radix prefix reuse.
- **Artifacts:** Result root `/home/jade_hand/research/shadowkv/results_sglang_lmcache_shadowkv_3models_3reps_2026-06-16`; report updated at `Wiki/life/projects/shadowkv-sglang-small-model-test-2026-06-13.md`; task queue updated at `Wiki/life/projects/nemotron-shadowkv-task-queue.md`.
- **Verification:** Aggregate CSV/JSON/Markdown artifacts exist, LMCache summaries are 90/90, and production `darwin28b-reason-vllm` was restored on `127.0.0.1:8015`.

## [2026-06-17] benchmark | ShadowKV SGLang identical-block rerun
- **Author:** Keystone (Codex on IronGod)
- **Scope:** Same three-model SGLang/Radix/ShadowKV++/patched-LMCache matrix as 2026-06-16, rerun with matched workload blocks to remove model-major ordering as a caveat.
- **Action:** Ran 30 matched rep/dataset/mode blocks. Each block contained all three models and all three baselines; model and baseline positions were balanced exactly across the run.
- **Result:** 270/270 jobs completed, 90/90 LMCache summaries generated, aggregate CSV/JSON/Markdown artifacts generated, and audit artifact `ANOMALY_AUDIT_IDENTICAL_BLOCKS_2026-06-17.md` recorded zero error scan hits and balanced schedule counts.
- **Artifacts:** Result root `/home/jade_hand/research/shadowkv/results_sglang_lmcache_shadowkv_3models_identical_blocks_3reps_2026-06-16`; report updated at `Wiki/life/projects/shadowkv-sglang-small-model-test-2026-06-13.md`; task queue updated at `Wiki/life/projects/nemotron-shadowkv-task-queue.md`.
- **Verification:** Production `darwin28b-reason-vllm` was restored and `http://127.0.0.1:8015/v1/models` returned `darwin28b-reason`.
