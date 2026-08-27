#!/usr/bin/env python3
"""Validate token-identical KV-cache crop-and-splice reuse.

This script compares three execution paths on the same modified token sequence:

1. full recompute with no cache;
2. a clean prefix-only cache followed by the suffix; and
3. a cache built from a longer source sequence, cropped to the token-identical
   prefix, then followed by the same suffix.

It reports prefix-KV differences, suffix-logit differences, and greedy-token
agreement under a shared teacher-forced context.  It does not use decoded-text
similarity as a correctness criterion.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM


TINY_MODELS = ("tiny-llama", "tiny-qwen2")


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def supports_argument(model: torch.nn.Module, name: str) -> bool:
    signature = inspect.signature(model.forward)
    if name in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def build_tiny_model(kind: str, dtype: torch.dtype) -> torch.nn.Module:
    common = {
        "vocab_size": 257,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    if kind == "tiny-llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(**common)
        model = LlamaForCausalLM(config)
    elif kind == "tiny-qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(**common)
        model = Qwen2ForCausalLM(config)
    else:
        raise ValueError(f"Unknown tiny model: {kind}")
    return model.to(dtype=dtype)


def load_model(args: argparse.Namespace, dtype: torch.dtype) -> torch.nn.Module:
    if args.tiny_model:
        model = build_tiny_model(args.tiny_model, dtype)
        model.config._attn_implementation = args.attn_implementation
    else:
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": args.trust_remote_code,
            "attn_implementation": args.attn_implementation,
        }
        if args.revision:
            kwargs["revision"] = args.revision
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    return model.to(args.device).eval()


def forward_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Any | None,
    use_cache: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "return_dict": True,
    }
    if supports_argument(model, "cache_position"):
        kwargs["cache_position"] = cache_position
    if supports_argument(model, "position_ids"):
        kwargs["position_ids"] = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)
    with torch.inference_mode():
        return model(**kwargs)


def cache_layers(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "layers"):
        layers = []
        for layer in cache.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None and values is not None:
                layers.append((keys, values))
        if layers:
            return layers
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    if isinstance(cache, (tuple, list)):
        return [(layer[0], layer[1]) for layer in cache]
    raise TypeError(f"Unsupported cache representation: {type(cache)!r}")


def crop_cache(cache: Any, length: int) -> Any:
    crop = getattr(cache, "crop", None)
    if crop is None:
        raise TypeError(f"Cache type {type(cache)!r} has no crop() method")
    result = crop(length)
    return cache if result is None else result


def cache_sequence_length(cache: Any) -> int | None:
    getter = getattr(cache, "get_seq_length", None)
    if getter is not None:
        return int(getter())
    layers = cache_layers(cache)
    return int(layers[0][0].shape[-2]) if layers else None


def tensor_metrics(left: torch.Tensor, right: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    same_shape = tuple(left.shape) == tuple(right.shape)
    if not same_shape:
        return {
            "same_shape": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "allclose": False,
        }
    left32 = left.detach().float()
    right32 = right.detach().float()
    left_finite = bool(torch.isfinite(left32).all().item())
    right_finite = bool(torch.isfinite(right32).all().item())
    if not left_finite or not right_finite:
        return {
            "same_shape": True,
            "shape": list(left.shape),
            "left_finite": left_finite,
            "right_finite": right_finite,
            "max_abs": None,
            "mean_abs": None,
            "allclose": False,
        }
    difference = (left32 - right32).abs()
    return {
        "same_shape": True,
        "shape": list(left.shape),
        "left_finite": True,
        "right_finite": True,
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "allclose": bool(torch.allclose(left32, right32, atol=atol, rtol=rtol)),
    }


def compare_caches(
    clean: Any, spliced: Any, active_length: int, atol: float, rtol: float
) -> dict[str, Any]:
    clean_layers = cache_layers(clean)
    spliced_layers = cache_layers(spliced)
    layer_results = []
    for index, (clean_layer, spliced_layer) in enumerate(zip(clean_layers, spliced_layers)):
        clean_key = clean_layer[0][..., :active_length, :]
        clean_value = clean_layer[1][..., :active_length, :]
        spliced_key = spliced_layer[0][..., :active_length, :]
        spliced_value = spliced_layer[1][..., :active_length, :]
        layer_results.append(
            {
                "layer": index,
                "key": tensor_metrics(clean_key, spliced_key, atol, rtol),
                "value": tensor_metrics(clean_value, spliced_value, atol, rtol),
            }
        )
    layer_count_equal = len(clean_layers) == len(spliced_layers)
    allclose = layer_count_equal and all(
        row["key"]["allclose"] and row["value"]["allclose"] for row in layer_results
    )
    return {
        "clean_layers": len(clean_layers),
        "spliced_layers": len(spliced_layers),
        "layer_count_equal": layer_count_equal,
        "active_length": active_length,
        "allclose": allclose,
        "layers": layer_results,
    }


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    metrics = tensor_metrics(reference, candidate, atol, rtol)
    if not metrics["same_shape"]:
        metrics.update({"top1_match_rate": 0.0, "mean_total_variation": None})
        return metrics
    if not metrics["left_finite"] or not metrics["right_finite"]:
        metrics.update({"top1_match_rate": None, "mean_total_variation": None})
        return metrics
    reference32 = reference.detach().float()
    candidate32 = candidate.detach().float()
    reference_top1 = reference32.argmax(dim=-1)
    candidate_top1 = candidate32.argmax(dim=-1)
    metrics["top1_match_rate"] = float((reference_top1 == candidate_top1).float().mean().item())
    reference_prob = torch.softmax(reference32, dim=-1)
    candidate_prob = torch.softmax(candidate32, dim=-1)
    metrics["mean_total_variation"] = float(
        (0.5 * (reference_prob - candidate_prob).abs().sum(dim=-1)).mean().item()
    )
    return metrics


def make_pair(
    vocab_size: int,
    sequence_length: int,
    shared_length: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = torch.randint(3, vocab_size, (1, sequence_length), generator=generator)
    modified = source.clone()
    suffix = modified[:, shared_length:]
    modified[:, shared_length:] = ((suffix - 3 + 17) % (vocab_size - 3)) + 3
    if not torch.equal(source[:, :shared_length], modified[:, :shared_length]):
        raise AssertionError("Constructed prefixes are not token-identical")
    if torch.equal(source[:, shared_length:], modified[:, shared_length:]):
        raise AssertionError("Constructed suffixes are unexpectedly identical")
    return source.to(device), modified.to(device)


def update_cache_with_token(
    model: torch.nn.Module,
    cache: Any,
    token: torch.Tensor,
    logical_position: int,
) -> Any:
    attention_mask = torch.ones(
        (token.shape[0], logical_position + 1), dtype=torch.long, device=token.device
    )
    position = torch.tensor([logical_position], dtype=torch.long, device=token.device)
    return forward_model(model, token, attention_mask, position, cache, True)


def validate_case(
    model: torch.nn.Module,
    source: torch.Tensor,
    modified: torch.Tensor,
    shared_length: int,
    max_new_tokens: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    device = modified.device
    total_length = modified.shape[1]
    prefix_mask = torch.ones((1, shared_length), dtype=torch.long, device=device)
    full_mask = torch.ones_like(modified, dtype=torch.long)
    prefix_positions = torch.arange(shared_length, dtype=torch.long, device=device)
    full_positions = torch.arange(total_length, dtype=torch.long, device=device)
    suffix_positions = torch.arange(shared_length, total_length, dtype=torch.long, device=device)

    full_output = forward_model(model, modified, full_mask, full_positions, None, False)
    clean_prefix_output = forward_model(
        model, modified[:, :shared_length], prefix_mask, prefix_positions, None, True
    )
    source_output = forward_model(model, source, torch.ones_like(source), full_positions, None, True)
    spliced_cache = crop_cache(source_output.past_key_values, shared_length)
    clean_cache = clean_prefix_output.past_key_values

    cache_comparison = compare_caches(
        clean_cache, spliced_cache, shared_length, atol, rtol
    )
    clean_length_before = cache_sequence_length(clean_cache)
    splice_length_before = cache_sequence_length(spliced_cache)

    suffix = modified[:, shared_length:]
    native_continuation = forward_model(
        model, suffix, full_mask, suffix_positions, clean_cache, True
    )
    splice_continuation = forward_model(
        model, suffix, full_mask, suffix_positions, spliced_cache, True
    )

    reference_suffix_logits = full_output.logits[:, shared_length:, :]
    suffix_metrics = {
        "native_vs_full": compare_logits(
            reference_suffix_logits, native_continuation.logits, atol, rtol
        ),
        "splice_vs_full": compare_logits(
            reference_suffix_logits, splice_continuation.logits, atol, rtol
        ),
        "splice_vs_native": compare_logits(
            native_continuation.logits, splice_continuation.logits, atol, rtol
        ),
    }

    native_cache = native_continuation.past_key_values
    continued_splice_cache = splice_continuation.past_key_values
    reference_ids = modified.clone()
    native_logits = native_continuation.logits[:, -1, :]
    splice_logits = splice_continuation.logits[:, -1, :]
    reference_logits = full_output.logits[:, -1, :]
    generation_steps = []

    for step in range(max_new_tokens):
        native_step = compare_logits(reference_logits, native_logits, atol, rtol)
        splice_step = compare_logits(reference_logits, splice_logits, atol, rtol)
        splice_native_step = compare_logits(native_logits, splice_logits, atol, rtol)
        reference_finite = bool(torch.isfinite(reference_logits).all().item())
        native_finite = bool(torch.isfinite(native_logits).all().item())
        splice_finite = bool(torch.isfinite(splice_logits).all().item())
        reference_token = reference_logits.argmax(dim=-1, keepdim=True)
        native_token = native_logits.argmax(dim=-1, keepdim=True)
        splice_token = splice_logits.argmax(dim=-1, keepdim=True)
        generation_steps.append(
            {
                "step": step,
                "reference_token": int(reference_token.item()),
                "native_token": int(native_token.item()),
                "splice_token": int(splice_token.item()),
                "reference_logits_finite": reference_finite,
                "native_logits_finite": native_finite,
                "splice_logits_finite": splice_finite,
                "native_token_match": bool(
                    reference_finite
                    and native_finite
                    and torch.equal(reference_token, native_token)
                ),
                "splice_token_match": bool(
                    reference_finite
                    and splice_finite
                    and torch.equal(reference_token, splice_token)
                ),
                "native_vs_full": native_step,
                "splice_vs_full": splice_step,
                "splice_vs_native": splice_native_step,
            }
        )
        if step == max_new_tokens - 1:
            break

        logical_position = reference_ids.shape[1]
        reference_ids = torch.cat([reference_ids, reference_token], dim=1)
        native_update = update_cache_with_token(
            model, native_cache, reference_token, logical_position
        )
        splice_update = update_cache_with_token(
            model, continued_splice_cache, reference_token, logical_position
        )
        native_cache = native_update.past_key_values
        continued_splice_cache = splice_update.past_key_values
        native_logits = native_update.logits[:, -1, :]
        splice_logits = splice_update.logits[:, -1, :]
        expanded_mask = torch.ones_like(reference_ids, dtype=torch.long)
        expanded_positions = torch.arange(
            reference_ids.shape[1], dtype=torch.long, device=device
        )
        full_output = forward_model(
            model, reference_ids, expanded_mask, expanded_positions, None, False
        )
        reference_logits = full_output.logits[:, -1, :]

    native_token_pass = all(row["native_token_match"] for row in generation_steps)
    splice_token_pass = all(row["splice_token_match"] for row in generation_steps)
    suffix_logits_finite = all(
        metrics.get("left_finite", False) and metrics.get("right_finite", False)
        for metrics in suffix_metrics.values()
    )
    token_semantic_pass = (
        clean_length_before == shared_length
        and splice_length_before == shared_length
        and suffix_logits_finite
        and native_token_pass
        and splice_token_pass
    )
    strict_pass = (
        cache_comparison["allclose"]
        and suffix_metrics["native_vs_full"]["allclose"]
        and suffix_metrics["splice_vs_native"]["allclose"]
        and native_token_pass
        and splice_token_pass
    )
    return {
        "shared_tokens": shared_length,
        "total_tokens": total_length,
        "cache_length_before_continuation": {
            "clean": clean_length_before,
            "spliced": splice_length_before,
        },
        "prefix_cache": cache_comparison,
        "suffix_logits": suffix_metrics,
        "generation": {
            "steps": generation_steps,
            "native_token_pass": native_token_pass,
            "splice_token_pass": splice_token_pass,
        },
        "suffix_logits_finite": suffix_logits_finite,
        "token_semantic_pass": token_semantic_pass,
        "strict_pass": strict_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-id")
    source.add_argument("--tiny-model", choices=TINY_MODELS)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--shared-ratios", nargs="+", type=float, default=(0.25, 0.5, 0.75))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.sequence_length < 4:
        parser.error("--sequence-length must be at least 4")
    if any(ratio <= 0 or ratio >= 1 for ratio in args.shared_ratios):
        parser.error("Every shared ratio must be strictly between 0 and 1")
    dtype = parse_dtype(args.dtype)
    if args.device == "cpu" and dtype == torch.float16:
        parser.error("float16 CPU execution is not supported; use float32/bfloat16 or CUDA")

    torch.manual_seed(args.seed)
    model = load_model(args, dtype)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "num_embeddings"):
        raise AttributeError("Model input embeddings do not expose num_embeddings")
    vocab_size = int(input_embeddings.num_embeddings)
    cases = []

    for sample in range(args.samples):
        for ratio in args.shared_ratios:
            shared_length = max(1, min(args.sequence_length - 1, int(args.sequence_length * ratio)))
            source_ids, modified_ids = make_pair(
                vocab_size,
                args.sequence_length,
                shared_length,
                args.seed + sample * 1009 + int(ratio * 1000),
                args.device,
            )
            print(
                f"sample={sample} ratio={ratio:.2f} shared={shared_length}/"
                f"{args.sequence_length}",
                flush=True,
            )
            result = validate_case(
                model,
                source_ids,
                modified_ids,
                shared_length,
                args.max_new_tokens,
                args.atol,
                args.rtol,
            )
            result.update({"sample": sample, "shared_ratio": ratio})
            cases.append(result)

    report = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
        "configuration": {
            "model_id": args.model_id,
            "tiny_model": args.tiny_model,
            "revision": args.revision,
            "device": args.device,
            "dtype": args.dtype,
            "attention_implementation": args.attn_implementation,
            "sequence_length": args.sequence_length,
            "shared_ratios": args.shared_ratios,
            "samples": args.samples,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
            "passes_cache_position": supports_argument(model, "cache_position"),
            "passes_position_ids": supports_argument(model, "position_ids"),
        },
        "summary": {
            "cases": len(cases),
            "strict_passes": sum(case["strict_pass"] for case in cases),
            "token_semantic_passes": sum(
                case["token_semantic_pass"] for case in cases
            ),
            "cache_passes": sum(case["prefix_cache"]["allclose"] for case in cases),
            "native_token_passes": sum(
                case["generation"]["native_token_pass"] for case in cases
            ),
            "splice_token_passes": sum(
                case["generation"]["splice_token_pass"] for case in cases
            ),
        },
        "cases": cases,
    }
    report["summary"]["all_strict_pass"] = (
        report["summary"]["strict_passes"] == report["summary"]["cases"]
    )
    report["summary"]["all_token_semantic_pass"] = (
        report["summary"]["token_semantic_passes"]
        == report["summary"]["cases"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0 if report["summary"]["all_strict_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
