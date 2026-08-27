#!/usr/bin/env python3
"""Run the complete learned-baseline matrix.

Default matrix:
  - 5 models, with GPT-2 first and Phi-3 last
  - 5 seeds
  - train on templated + semantic
  - evaluate on raw + templated + semantic
  - run the memory-bound victim/distractor/recovery trace against the learned
    baselines

Each model gets its own output directory under the matrix root, so the command
can be rerun with --resume without clobbering completed cells.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = [
    "gpt2",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/gemma-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
]

DEFAULT_SEEDS = [42, 123, 456, 789, 999]
DEFAULT_TRAIN_MODES = ["templated", "semantic"]
DEFAULT_TEST_MODES = ["raw", "templated", "semantic"]


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")
    aliases = {
        "gpt2": "gpt2",
        "Qwen_Qwen2_5_1_5B_Instruct": "qwen25_15b",
        "TinyLlama_TinyLlama_1_1B_Chat_v1_0": "tinyllama_11b",
        "google_gemma_2b_it": "gemma2b",
        "microsoft_Phi_3_mini_4k_instruct": "phi3mini",
    }
    return aliases.get(slug, slug.lower())


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("\n" + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def write_combined_summary(root: Path) -> None:
    combined: dict[str, object] = {}
    for summary_path in sorted(root.glob("*/phase3_summary.json")):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        combined[summary_path.parent.name] = data
    if not combined:
        return
    output = root / "summary_all_models.json"
    output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nSaved combined summary to {output}", flush=True)


def write_combined_memory_summary(root: Path) -> None:
    combined: dict[str, object] = {}
    for summary_path in sorted(root.glob("*/memory_bound/memory_bound_aggregate.json")):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        combined[summary_path.parents[1].name] = data
    if not combined:
        return
    output = root / "summary_memory_bound_all_models.json"
    output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Saved combined memory-bound summary to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete learned-baseline all-model matrix.")
    parser.add_argument("--backend", default="hf", choices=["hf", "fake"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--n_requests", type=int, default=128)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--train_modes", nargs="+", default=DEFAULT_TRAIN_MODES)
    parser.add_argument("--test_modes", nargs="+", default=DEFAULT_TEST_MODES)
    parser.add_argument("--out_root", default=str(REPO / "results" / "learned_baseline_5seed_all_models"))
    parser.add_argument("--resume", action="store_true", default=True, help="Skip completed benchmark cells. Enabled by default.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Rerun cells even when outputs already exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-share-backend", action="store_true")
    parser.add_argument("--skip-memory-bound", action="store_true", help="Skip the memory-pressure victim/recovery trace.")
    parser.add_argument("--memory_n_victims", type=int, default=6)
    parser.add_argument("--memory_n_distractors", type=int, default=32)
    parser.add_argument("--memory_victim_warmup_repeats", type=int, default=2)
    parser.add_argument("--memory_pressure_repeats", type=int, default=2)
    parser.add_argument("--memory_recovery_repeats", type=int, default=2)
    parser.add_argument("--memory_capacity_entries", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Complete learned-baseline matrix")
    print("=" * 72)
    print(f"models={args.models}")
    print(f"seeds={args.seeds}")
    print(f"train_modes={args.train_modes}")
    print(f"test_modes={args.test_modes}")
    print(f"out_root={root}")
    print(f"resume={args.resume}")

    for model in args.models:
        model_root = root / model_slug(model)
        common = [
            sys.executable,
            str(REPO / "experiments" / "run_learned_baseline.py"),
            "--backend", args.backend,
            "--model", model,
            "--device", args.device,
            "--dtype", args.dtype,
            "--n_requests", str(args.n_requests),
            "--train_seeds", *[str(seed) for seed in args.seeds],
            "--test_seeds", *[str(seed) for seed in args.seeds],
            "--out_root", str(model_root),
        ]
        if args.resume:
            common.append("--resume")
        if args.no_share_backend:
            common.append("--no-share-backend")

        train_cmd = [
            *common,
            "--phase", "train",
            "--train_modes", *args.train_modes,
        ]
        phase3_cmd = [
            *common,
            "--phase", "phase3",
            "--test_modes", *args.test_modes,
        ]
        run_command(train_cmd, dry_run=args.dry_run)
        run_command(phase3_cmd, dry_run=args.dry_run)
        if not args.skip_memory_bound:
            memory_cmd = [
                sys.executable,
                str(REPO / "experiments" / "run_learned_memory_bound.py"),
                "--backend", args.backend,
                "--model", model,
                "--device", args.device,
                "--dtype", args.dtype,
                "--seeds", *[str(seed) for seed in args.seeds],
                "--out_root", str(model_root / "memory_bound"),
                "--policy_raw", str(model_root / "learned_policy_raw.json"),
                "--policy_utility", str(model_root / "learned_policy_utility.json"),
                "--n_victims", str(args.memory_n_victims),
                "--n_distractors", str(args.memory_n_distractors),
                "--victim_warmup_repeats", str(args.memory_victim_warmup_repeats),
                "--pressure_repeats", str(args.memory_pressure_repeats),
                "--recovery_repeats", str(args.memory_recovery_repeats),
                "--capacity_entries", str(args.memory_capacity_entries),
            ]
            if args.resume:
                memory_cmd.append("--resume")
            run_command(memory_cmd, dry_run=args.dry_run)
        if not args.dry_run:
            write_combined_summary(root)
            write_combined_memory_summary(root)

    if not args.dry_run:
        write_combined_summary(root)
        write_combined_memory_summary(root)


if __name__ == "__main__":
    main()
