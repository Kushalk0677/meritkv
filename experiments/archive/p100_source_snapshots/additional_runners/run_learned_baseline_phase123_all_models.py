#!/usr/bin/env python3
"""Run only the learned-baseline Phase 1 to Phase 3 matrix.

This wrapper intentionally excludes the memory-bound victim/recovery trace.
It runs, per model:
  1. train traces on templated + semantic prompts,
  2. learned-policy fitting from those traces,
  3. held-out Phase 3 evaluation on raw + templated + semantic prompts.
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
    output = root / "summary_phase123_all_models.json"
    output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nSaved combined Phase 3 summary to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learned-baseline Phase 1 to Phase 3 only.")
    parser.add_argument("--backend", default="hf", choices=["hf", "fake"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--n_requests", type=int, default=128)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--train_modes", nargs="+", default=DEFAULT_TRAIN_MODES)
    parser.add_argument("--test_modes", nargs="+", default=DEFAULT_TEST_MODES)
    parser.add_argument("--out_root", default=str(REPO / "results" / "learned_baseline_phase123_5seed_all_models"))
    parser.add_argument("--resume", action="store_true", default=True, help="Skip completed cells. Enabled by default.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Rerun cells even if outputs exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-share-backend", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Learned-baseline Phase 1 to Phase 3 matrix")
    print("=" * 72)
    print(f"models={args.models}")
    print(f"seeds={args.seeds}")
    print(f"train_modes={args.train_modes}")
    print(f"test_modes={args.test_modes}")
    print(f"n_requests={args.n_requests}")
    print(f"out_root={root}")

    common = [
        sys.executable,
        str(REPO / "experiments" / "run_learned_baseline.py"),
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--n_requests",
        str(args.n_requests),
        "--train_seeds",
        *[str(seed) for seed in args.seeds],
        "--test_seeds",
        *[str(seed) for seed in args.seeds],
    ]
    if args.resume:
        common.append("--resume")
    if args.no_share_backend:
        common.append("--no-share-backend")

    for model in args.models:
        slug = model_slug(model)
        model_root = root / slug

        train_cmd = [
            *common,
            "--model",
            model,
            "--out_root",
            str(model_root),
            "--phase",
            "train",
            "--train_modes",
            *args.train_modes,
        ]
        phase3_cmd = [
            *common,
            "--model",
            model,
            "--out_root",
            str(model_root),
            "--phase",
            "phase3",
            "--train_modes",
            *args.train_modes,
            "--test_modes",
            *args.test_modes,
        ]

        run_command(train_cmd, dry_run=args.dry_run)
        run_command(phase3_cmd, dry_run=args.dry_run)
        write_combined_summary(root)

    write_combined_summary(root)


if __name__ == "__main__":
    main()
