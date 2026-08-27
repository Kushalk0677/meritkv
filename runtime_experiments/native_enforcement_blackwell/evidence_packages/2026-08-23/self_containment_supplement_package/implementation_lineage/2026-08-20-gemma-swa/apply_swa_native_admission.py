#!/usr/bin/env python3
"""Fail-closed port of the native admission hook into SGLang SWARadixCache."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "MERITKV_SWA_RADIX_TARGET",
        "/usr/local/lib/python3.12/dist-packages/sglang/srt/mem_cache/swa_radix_cache.py",
    )
)
EXPECTED_ORIGINAL_SHA256 = (
    "96c07d8f05a71eb4a70eef394db0c92783128ad8e9fc3cc84c6beaf8504e542f"
)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(old, new, 1)


def main() -> None:
    original = TARGET.read_bytes()
    observed = hashlib.sha256(original).hexdigest()
    if observed != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            f"unexpected SWARadixCache source {observed}; expected "
            f"{EXPECTED_ORIGINAL_SHA256}"
        )
    source = original.decode("utf-8")
    source = replace_once(
        source,
        "from sglang.srt.mem_cache.radix_cache import RadixKey\n"
        "from sglang.srt.mem_cache.utils import split_node_hash_value\n",
        "from sglang.srt.mem_cache.radix_cache import RadixKey\n"
        "from sglang.srt.mem_cache.utils import split_node_hash_value\n"
        "from sglang.srt.shadowkv_admission_metrics import inc as _shadowkv_metric_inc\n",
        "metrics import",
    )
    source = replace_once(
        source,
        "        \"\"\"\n\n"
        "        key = self._match_pre_processor(params)\n",
        "        \"\"\"\n\n"
        "        if params.req is not None and getattr(\n"
        "            params.req, \"shadowkv_skip_cache_lookup\", False\n"
        "        ):\n"
        "            _shadowkv_metric_inc(\"radix_skip_lookup_total\")\n"
        "            _shadowkv_metric_inc(\"swa_radix_skip_lookup_total\")\n"
        "            return MatchResult(\n"
        "                device_indices=torch.empty(\n"
        "                    (0,), dtype=torch.int64, device=self.device\n"
        "                ),\n"
        "                last_device_node=self.root_node,\n"
        "                last_host_node=self.root_node,\n"
        "                best_match_node=self.root_node,\n"
        "            )\n\n"
        "        key = self._match_pre_processor(params)\n",
        "lookup bypass",
    )
    source = replace_once(
        source,
        "    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:\n"
        "        \"\"\"Cache request when it finishes.\"\"\"\n"
        "        kv_committed_len = req.pop_committed_kv_cache()\n",
        "    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:\n"
        "        \"\"\"Cache request when it finishes.\"\"\"\n"
        "        if getattr(req, \"shadowkv_skip_cache_write\", False):\n"
        "            _shadowkv_metric_inc(\"radix_skip_write_finished_total\")\n"
        "            _shadowkv_metric_inc(\"swa_radix_skip_write_finished_total\")\n"
        "            is_insert = False\n"
        "        kv_committed_len = req.pop_committed_kv_cache()\n",
        "finished write bypass",
    )
    source = replace_once(
        source,
        "    def cache_unfinished_req(self, req: Req, chunked=False) -> None:\n"
        "        \"\"\"Cache request when it is unfinished.\"\"\"\n"
        "        if self.disable:\n",
        "    def cache_unfinished_req(self, req: Req, chunked=False) -> None:\n"
        "        \"\"\"Cache request when it is unfinished.\"\"\"\n"
        "        if getattr(req, \"shadowkv_skip_cache_write\", False):\n"
        "            _shadowkv_metric_inc(\"radix_skip_write_unfinished_total\")\n"
        "            _shadowkv_metric_inc(\"swa_radix_skip_write_unfinished_total\")\n"
        "            return\n"
        "        if self.disable:\n",
        "unfinished write bypass",
    )
    TARGET.write_text(source, encoding="utf-8")
    print(f"patched_sha256={hashlib.sha256(TARGET.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
