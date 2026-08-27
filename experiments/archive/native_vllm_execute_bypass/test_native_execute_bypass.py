from __future__ import annotations

import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import unittest


MODULE_PATH = Path(__file__).with_name("run_native_execute_bypass.py")
SPEC = importlib.util.spec_from_file_location("native_vllm_execute_bypass", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Handler(BaseHTTPRequestHandler):
    bodies: list[dict] = []

    def log_message(self, *_args) -> None:
        return

    def do_GET(self) -> None:
        payload = {"data": [{"id": "test-model"}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length).decode())
        self.bodies.append(body)
        salt = str(body.get("cache_salt", ""))
        cached = 0 if "unique" in salt or "forced" in salt or "bypass" in salt else 16
        payload = {
            "choices": [
                {
                    "text": " answer",
                    "token_ids": [42],
                    "logprobs": {"tokens": ["token_id:42"]},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(body["prompt"]),
                "completion_tokens": 1,
                "total_tokens": len(body["prompt"]) + 1,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())


class NativeBypassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_client_sends_native_cache_salt_and_token_prompt(self) -> None:
        client = MODULE.VLLMClient(
            f"http://127.0.0.1:{self.server.server_port}", "test-model", 5
        )
        result = client.complete([1, 2, 3], "stable", 1)
        self.assertEqual(result["cached_tokens"], 16)
        self.assertEqual(result["token_ids"], [42])
        self.assertEqual(Handler.bodies[-1]["prompt"], [1, 2, 3])
        self.assertEqual(Handler.bodies[-1]["cache_salt"], "stable")
        self.assertEqual(Handler.bodies[-1]["temperature"], 0.0)
        self.assertEqual(Handler.bodies[-1]["top_p"], 1.0)
        self.assertEqual(Handler.bodies[-1]["top_k"], -1)
        self.assertEqual(Handler.bodies[-1]["repetition_penalty"], 1.0)

    def test_actuator_self_test_requires_hit_and_forced_miss(self) -> None:
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return list(range(max(len(text.split()), 600)))

        client = MODULE.VLLMClient(
            f"http://127.0.0.1:{self.server.server_port}", "test-model", 5
        )
        report = MODULE.actuator_self_test(client, Tokenizer(), 1)
        self.assertTrue(report["passed"])
        self.assertGreater(report["hit"]["cached_tokens"], 0)
        self.assertEqual(report["bypass"]["cached_tokens"], 0)

    def test_summary_uses_seed_replicates_and_paired_outputs(self) -> None:
        rows = []
        for seed, offset in ((42, 0.0), (123, 1.0)):
            for arm, latency, cached in (
                ("forced_recompute", 20.0 + offset, 0),
                ("native_apc", 10.0 + offset, 16),
                ("meritkv_write_through", 11.0 + offset, 16),
                ("meritkv_enforced", 12.0 + offset, 0),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "case_id": "L16_R000",
                        "prefix_length_target": 16,
                        "latency_ms": latency,
                        "cached_tokens": cached,
                        "token_ids": [42],
                        "text": " answer",
                        "meritkv_strategy": "bypass" if arm.startswith("meritkv") else None,
                        "bypass_enforced": arm == "meritkv_enforced",
                    }
                )
        summary = MODULE.summarize_run(rows)
        self.assertEqual(summary["arms"]["native_apc"]["seed_count"], 2)
        self.assertEqual(summary["output_checks"]["meritkv_enforced"]["token_id_exact_matches"], 2)
        self.assertGreater(
            summary["comparisons"]["enforced_vs_write_through"]["speedup"]["mean"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
