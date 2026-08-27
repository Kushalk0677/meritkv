#!/usr/bin/env python3
"""CPU-only exact-image compatibility probe for both pinned model snapshots."""

from __future__ import annotations

import json
import sys

from transformers import AutoConfig, AutoTokenizer


EXPECTED = {
    "/hf_hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd": (
        "qwen2",
        "Qwen2ForCausalLM",
    ),
    "/hf_hub/models--google--gemma-4-31B-it/snapshots/b9ea41a2887d8607f594846523f94c6cc75ac8a4": (
        "gemma4",
        "Gemma4ForConditionalGeneration",
    ),
}


def main() -> None:
    results = []
    for path, (model_type, architecture) in EXPECTED.items():
        config = AutoConfig.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        if config.model_type != model_type:
            raise RuntimeError(f"{path}: {config.model_type} != {model_type}")
        if architecture not in list(getattr(config, "architectures", []) or []):
            raise RuntimeError(f"{path}: missing architecture {architecture}")
        encoded = tokenizer("MeritKV pinned tokenizer probe", truncation=True)["input_ids"]
        if not encoded:
            raise RuntimeError(f"{path}: tokenizer returned no tokens")
        results.append(
            {
                "path": path,
                "config_class": type(config).__name__,
                "model_type": config.model_type,
                "architecture": architecture,
                "tokenizer_class": type(tokenizer).__name__,
                "probe_tokens": len(encoded),
                "fallback_used": False,
            }
        )
    print(json.dumps({"status": "pass", "models": results}, sort_keys=True))


if __name__ == "__main__":
    main()
