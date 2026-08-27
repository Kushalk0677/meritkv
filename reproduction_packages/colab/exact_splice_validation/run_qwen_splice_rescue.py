#!/usr/bin/env python3
"""Paired old-vs-corrected Qwen splice fidelity and prefill timing check."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import sys
import time
from typing import Any

import torch
import transformers
from datasets import load_dataset
from huggingface_hub import login, model_info
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/content/meritkv_splice")
OUTPUT = Path("/content/meritkv_qwen_splice_rescue_outputs")
ARCHIVE = Path("/content/meritkv_qwen_splice_rescue_outputs.zip")
TOKEN_FILE = ROOT / ".hf_token"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASETS = {
    "samsum": ("knkarthick/samsum", None, "train", "dialogue"),
    "xsum": ("EdinburghNLP/xsum", None, "train", "document"),
    "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0", "train", "article"),
    "ag_news": ("fancyzhx/ag_news", None, "train", "text"),
    "banking77": ("mteb/banking77", None, "train", "text"),
    "alpaca_eval": ("Thanmay/alpaca_eval", None, "eval", "instruction"),
    "dolly": ("databricks/databricks-dolly-15k", None, "train", "instruction"),
    "daily_dialog": ("DeepPavlov/daily_dialog", None, "train", "dialog"),
    "oasst1": ("OpenAssistant/oasst1", None, "train", "text"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", "messages"),
}
SAMPLES_PER_DATASET = 8
MAX_NEW_TOKENS = 64
TIMING_REPEATS = 3
SHARED_RATIO = 0.75
STRIP_PATTERN = re.compile(
    r"^(Response:|Answer:|Output:|Instruction:|Script:|Task:|Assistant:|"
    r"System:|Note:|Explanation:|Result:|Summary:|Solution:|Dialogue:|"
    r"Conversation:|Messages:|Chat:|Description:|Overview:|Info:|Information:)\s*",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = text.strip()
    while True:
        match = STRIP_PATTERN.match(text)
        if match is None:
            return text
        text = text[match.end() :].strip()


def supports_argument(model: torch.nn.Module, name: str) -> bool:
    signature = inspect.signature(model.forward)
    return name in signature.parameters or any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )


def model_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any | None,
    logical_start: int,
    mode: str,
) -> Any:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": True,
        "return_dict": True,
    }
    if mode != "old":
        positions = torch.arange(
            logical_start,
            logical_start + input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        )
        kwargs["attention_mask"] = torch.ones(
            (input_ids.shape[0], logical_start + input_ids.shape[1]),
            dtype=torch.long,
            device=input_ids.device,
        )
        if supports_argument(model, "position_ids"):
            kwargs["position_ids"] = positions.unsqueeze(0)
        if mode == "corrected" and supports_argument(model, "cache_position"):
            kwargs["cache_position"] = positions
    with torch.inference_mode():
        return model(**kwargs)


def crop_cache(cache: Any, length: int) -> Any:
    result = cache.crop(length)
    return cache if result is None else result


def generate_clean(model: torch.nn.Module, ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Greedy full-recompute reference using the same explicit forward contract."""
    output = model_forward(model, ids, None, 0, "corrected")
    cache = output.past_key_values
    token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated: list[int] = []
    logical_position = ids.shape[1]
    for step in range(MAX_NEW_TOKENS):
        token_id = int(token.item())
        generated.append(token_id)
        if token_id == eos_id or step == MAX_NEW_TOKENS - 1:
            break
        update = model_forward(model, token, cache, logical_position, "corrected")
        cache = update.past_key_values
        token = update.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        logical_position += 1
    return torch.tensor(generated, dtype=torch.long)


def generate_reuse(
    model: torch.nn.Module,
    source_cache: Any,
    modified_ids: torch.Tensor,
    shared_length: int,
    eos_id: int,
    mode: str,
) -> torch.Tensor:
    cache = crop_cache(copy.deepcopy(source_cache), shared_length)
    suffix = modified_ids[:, shared_length:]
    continuation = model_forward(model, suffix, cache, shared_length, mode)
    cache = continuation.past_key_values
    token = continuation.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated: list[int] = []
    logical_position = modified_ids.shape[1]
    for step in range(MAX_NEW_TOKENS):
        token_id = int(token.item())
        generated.append(token_id)
        if token_id == eos_id or step == MAX_NEW_TOKENS - 1:
            break
        update = model_forward(model, token, cache, logical_position, mode)
        cache = update.past_key_values
        token = update.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        logical_position += 1
    return torch.tensor(generated, dtype=torch.long)


def timed_prefill(
    model: torch.nn.Module,
    modified_ids: torch.Tensor,
    source_cache: Any,
    shared_length: int,
    mode: str,
) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    if mode == "full":
        _ = model_forward(model, modified_ids, None, 0, "corrected")
    else:
        cache = crop_cache(copy.deepcopy(source_cache), shared_length)
        _ = model_forward(
            model,
            modified_ids[:, shared_length:],
            cache,
            shared_length,
            mode,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def extract_text(row: dict[str, Any], dataset: str, field: str) -> str:
    if dataset == "daily_dialog":
        value = row.get(field, [])
        return " ".join(value) if isinstance(value, list) else str(value)
    if dataset == "ultrachat":
        messages = row.get(field, [])
        return " ".join(
            str(item.get("content", ""))
            for item in messages
            if isinstance(item, dict)
        )
    return str(row.get(field, ""))


def load_samples(dataset: str) -> list[str]:
    repo, config, split, field = DATASETS[dataset]
    if config is None:
        loaded = load_dataset(repo, split=split, streaming=True)
    else:
        loaded = load_dataset(repo, config, split=split, streaming=True)
    samples: list[str] = []
    for row in loaded:
        text = extract_text(row, dataset, field)[:512]
        if len(text) > 20:
            samples.append(text)
        if len(samples) == SAMPLES_PER_DATASET:
            break
    return samples


def make_pair(tokenizer: Any, payload: str, rng: random.Random) -> tuple[list[int], list[int], int]:
    source = tokenizer.encode(payload, truncation=True, max_length=384)
    shared = int(len(source) * SHARED_RATIO)
    if len(source) - shared < 4:
        shared = len(source) - 4
    if shared < 4:
        raise ValueError("Prompt is too short after tokenization")
    modified = source[:shared] + rng.sample(source[shared:], len(source[shared:]))
    return source, modified, shared


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    old_rouge = [float(row["old_rouge_l"]) for row in rows]
    corrected_rouge = [float(row["corrected_rouge_l"]) for row in rows]
    backend_rouge = [float(row["paper_backend_rouge_l"]) for row in rows]
    old_speedup = [float(row["old_speedup"]) for row in rows]
    corrected_speedup = [float(row["corrected_speedup"]) for row in rows]
    backend_speedup = [float(row["paper_backend_speedup"]) for row in rows]
    old_latency = [float(row["old_prefill_ms"]) for row in rows]
    corrected_latency = [float(row["corrected_prefill_ms"]) for row in rows]
    backend_latency = [float(row["paper_backend_prefill_ms"]) for row in rows]
    return {
        "samples": len(rows),
        "old_exact_matches": sum(row["old_exact_match"] for row in rows),
        "corrected_exact_matches": sum(row["corrected_exact_match"] for row in rows),
        "paper_backend_exact_matches": sum(row["paper_backend_exact_match"] for row in rows),
        "old_rouge_l_mean": mean(old_rouge),
        "corrected_rouge_l_mean": mean(corrected_rouge),
        "paper_backend_rouge_l_mean": mean(backend_rouge),
        "old_speedup_mean_of_ratios": mean(old_speedup),
        "corrected_speedup_mean_of_ratios": mean(corrected_speedup),
        "paper_backend_speedup_mean_of_ratios": mean(backend_speedup),
        "old_prefill_ms_mean": mean(old_latency),
        "corrected_prefill_ms_mean": mean(corrected_latency),
        "paper_backend_prefill_ms_mean": mean(backend_latency),
        "corrected_vs_old_prefill_latency_ratio": mean(corrected_latency) / mean(old_latency),
        "corrected_minus_old_speedup": mean(corrected_speedup) - mean(old_speedup),
        "corrected_vs_paper_backend_prefill_latency_ratio": mean(corrected_latency) / mean(backend_latency),
        "corrected_minus_paper_backend_speedup": mean(corrected_speedup) - mean(backend_speedup),
    }


def write_report(summary: dict[str, Any], by_dataset: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Qwen2.5 Corrected-Splice Rescue Check",
        "",
        "Paired T4 comparison of the manuscript's original mask/position-implicit custom splice and the corrected SDPA path with a complete attention mask, explicit logical `position_ids`, and `cache_position`.",
        "",
        "## Aggregate",
        "",
        "| Metric | Original diagnostic | Frozen paper backend | Fully explicit corrected |",
        "|---|---:|---:|---:|",
        f"| Exact decoded-output matches | {summary['old_exact_matches']}/{summary['samples']} | {summary['paper_backend_exact_matches']}/{summary['samples']} | {summary['corrected_exact_matches']}/{summary['samples']} |",
        f"| Mean ROUGE-L | {summary['old_rouge_l_mean']:.4f} | {summary['paper_backend_rouge_l_mean']:.4f} | {summary['corrected_rouge_l_mean']:.4f} |",
        f"| Mean cached-prefill speedup vs full prefill | {summary['old_speedup_mean_of_ratios']:.4f}x | {summary['paper_backend_speedup_mean_of_ratios']:.4f}x | {summary['corrected_speedup_mean_of_ratios']:.4f}x |",
        f"| Mean cached-prefill latency | {summary['old_prefill_ms_mean']:.4f} ms | {summary['paper_backend_prefill_ms_mean']:.4f} ms | {summary['corrected_prefill_ms_mean']:.4f} ms |",
        "",
        f"Corrected/original cached-prefill latency ratio: `{summary['corrected_vs_old_prefill_latency_ratio']:.6f}`.",
        f"Corrected/frozen-paper-backend latency ratio: `{summary['corrected_vs_paper_backend_prefill_latency_ratio']:.6f}`.",
        "",
        "## Per dataset",
        "",
        "| Dataset | N | Original ROUGE-L | Paper-backend ROUGE-L | Corrected ROUGE-L | Paper-backend speedup | Corrected speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, values in by_dataset.items():
        lines.append(
            f"| {dataset} | {values['samples']} | {values['old_rouge_l_mean']:.4f} | "
            f"{values['paper_backend_rouge_l_mean']:.4f} | {values['corrected_rouge_l_mean']:.4f} | "
            f"{values['paper_backend_speedup_mean_of_ratios']:.4f}x | "
            f"{values['corrected_speedup_mean_of_ratios']:.4f}x |"
        )
    lines.extend(
        [
            "",
            "Speedup is full-prefill latency divided by cached-suffix-prefill latency on the same prompt. Source-cache creation is excluded because the cache is assumed to pre-exist; cache isolation and cropping are included. Absolute T4 ratios are not substituted for the paper's Blackwell values. The paired old/new difference tests whether the correctness fix changes the reuse economics.",
            "",
        ]
    )
    (OUTPUT / "QWEN_SPLICE_RESCUE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    TOKEN_FILE.unlink(missing_ok=True)
    if not token:
        raise RuntimeError("Uploaded Hugging Face token is empty")
    login(token=token, add_to_git_credential=False)
    resolved_revision = model_info(MODEL_ID, token=token).sha
    if resolved_revision != REVISION:
        raise RuntimeError(f"Revision mismatch: expected {REVISION}, got {resolved_revision}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        token=token,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda:0").eval()
    token = ""
    os.environ.pop("HF_TOKEN", None)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rng = random.Random(42)
    rows: list[dict[str, Any]] = []
    dataset_failures: list[dict[str, str]] = []

    for dataset in DATASETS:
        try:
            samples = load_samples(dataset)
        except Exception as exc:
            dataset_failures.append({"dataset": dataset, "error": repr(exc)})
            print(f"DATASET_FAILURE {dataset}: {exc!r}", flush=True)
            continue
        print(f"DATASET {dataset}: {len(samples)} samples", flush=True)
        for sample_index, payload in enumerate(samples):
            try:
                source_tokens, modified_tokens, shared = make_pair(tokenizer, payload, rng)
            except ValueError:
                continue
            source_ids = torch.tensor([source_tokens], dtype=torch.long, device="cuda:0")
            modified_ids = torch.tensor([modified_tokens], dtype=torch.long, device="cuda:0")
            source_output = model_forward(model, source_ids, None, 0, "corrected")
            source_cache = source_output.past_key_values

            reference_tokens = generate_clean(model, modified_ids, tokenizer.eos_token_id)
            old_tokens = generate_reuse(
                model, source_cache, modified_ids, shared, tokenizer.eos_token_id, "old"
            )
            paper_backend_tokens = generate_reuse(
                model, source_cache, modified_ids, shared, tokenizer.eos_token_id, "paper_backend"
            )
            corrected_tokens = generate_reuse(
                model, source_cache, modified_ids, shared, tokenizer.eos_token_id, "corrected"
            )
            reference_text = tokenizer.decode(reference_tokens, skip_special_tokens=True)
            old_text = tokenizer.decode(old_tokens, skip_special_tokens=True)
            paper_backend_text = tokenizer.decode(paper_backend_tokens, skip_special_tokens=True)
            corrected_text = tokenizer.decode(corrected_tokens, skip_special_tokens=True)
            reference_clean = clean_text(reference_text)
            old_clean = clean_text(old_text)
            paper_backend_clean = clean_text(paper_backend_text)
            corrected_clean = clean_text(corrected_text)

            timings: dict[str, list[float]] = {"full": [], "old": [], "paper_backend": [], "corrected": []}
            orders = (
                ("full", "old", "paper_backend", "corrected"),
                ("corrected", "paper_backend", "full", "old"),
                ("old", "corrected", "paper_backend", "full"),
            )
            for repeat in range(TIMING_REPEATS):
                for mode in orders[(sample_index + repeat) % len(orders)]:
                    timings[mode].append(
                        timed_prefill(model, modified_ids, source_cache, shared, mode)
                    )
            full_ms = mean(timings["full"])
            old_ms = mean(timings["old"])
            paper_backend_ms = mean(timings["paper_backend"])
            corrected_ms = mean(timings["corrected"])
            row = {
                "dataset": dataset,
                "sample": sample_index,
                "shared_tokens": shared,
                "total_tokens": len(modified_tokens),
                "reference_text": reference_text,
                "old_reuse_text": old_text,
                "paper_backend_reuse_text": paper_backend_text,
                "corrected_reuse_text": corrected_text,
                "old_exact_match": reference_clean == old_clean,
                "paper_backend_exact_match": reference_clean == paper_backend_clean,
                "corrected_exact_match": reference_clean == corrected_clean,
                "old_rouge_l": scorer.score(reference_clean, old_clean)["rougeL"].fmeasure,
                "paper_backend_rouge_l": scorer.score(reference_clean, paper_backend_clean)["rougeL"].fmeasure,
                "corrected_rouge_l": scorer.score(reference_clean, corrected_clean)["rougeL"].fmeasure,
                "full_prefill_ms": full_ms,
                "old_prefill_ms": old_ms,
                "paper_backend_prefill_ms": paper_backend_ms,
                "corrected_prefill_ms": corrected_ms,
                "old_speedup": full_ms / old_ms,
                "paper_backend_speedup": full_ms / paper_backend_ms,
                "corrected_speedup": full_ms / corrected_ms,
                "timing_repeats_ms": timings,
                "source_token_ids": source_tokens,
                "modified_token_ids": modified_tokens,
            }
            rows.append(row)
            print(
                f"  {sample_index + 1}/{len(samples)} old_RL={row['old_rouge_l']:.3f} "
                f"paper_RL={row['paper_backend_rouge_l']:.3f} new_RL={row['corrected_rouge_l']:.3f} "
                f"paper_S={row['paper_backend_speedup']:.3f} "
                f"new_S={row['corrected_speedup']:.3f}",
                flush=True,
            )

    summary = aggregate(rows)
    by_dataset = {
        dataset: aggregate([row for row in rows if row["dataset"] == dataset])
        for dataset in DATASETS
        if any(row["dataset"] == dataset for row in rows)
    }
    report = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "configuration": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "dtype": "float16",
            "attention_implementation": "sdpa",
            "samples_per_dataset": SAMPLES_PER_DATASET,
            "max_new_tokens": MAX_NEW_TOKENS,
            "timing_repeats": TIMING_REPEATS,
            "shared_ratio": SHARED_RATIO,
            "seed": 42,
            "passes_position_ids": supports_argument(model, "position_ids"),
            "passes_cache_position": supports_argument(model, "cache_position"),
            "passes_full_attention_mask": True,
        },
        "summary": summary,
        "by_dataset": by_dataset,
        "dataset_failures": dataset_failures,
        "rows": rows,
    }
    (OUTPUT / "qwen_splice_rescue_raw.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_report(summary, by_dataset)
    (OUTPUT / "environment.json").write_text(
        json.dumps(report["environment"], indent=2), encoding="utf-8"
    )
    shutil.copy2(ROOT / "run_qwen_splice_rescue.py", OUTPUT / "executed_run_qwen_splice_rescue.py")

    manifest = []
    for path in sorted(item for item in OUTPUT.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(f"{digest}  {path.relative_to(OUTPUT).as_posix()}")
    (OUTPUT / "MANIFEST_SHA256.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    shutil.make_archive(str(ARCHIVE.with_suffix("")), "zip", OUTPUT)
    print(json.dumps({"summary": summary, "dataset_failures": dataset_failures}, indent=2))
    print(f"RESULT_ARCHIVE={ARCHIVE}")
    return 0 if rows and not dataset_failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        TOKEN_FILE.unlink(missing_ok=True)
        os.environ.pop("HF_TOKEN", None)
