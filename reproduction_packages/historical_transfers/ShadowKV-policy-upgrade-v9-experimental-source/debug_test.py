#!/usr/bin/env python3
"""
Debug the test failure to understand why policy plans are being created.
"""

import sys
sys.path.insert(0, 'C:\\shadowkv\\robust_policy_upgrade_v6\\src')

from proactive_kv_cache.engines import ShadowKVPlusEngine
from proactive_kv_cache.models import FakeBackend

def debug_test():
    backend = FakeBackend()
    engine = ShadowKVPlusEngine(backend=backend)
    requests = [
        tuple([1, 2, 3, 4, 5, 6, i]) for i in range(10)
    ]
    
    print("Debugging ShadowKV+ policy plan creation")
    print("=" * 50)
    
    for i, tokens in enumerate(requests):
        print(f"\nRequest {i}: {tokens}")
        print(f"  Metadata: {{'prompt_mode': 'templated', 'shared_prefix_hint_tokens': 6}}")
        
        # Check what happens before serving
        match_before = engine.bank.peek_match(tokens)
        print(f"  Match before: {match_before}")
        
        result = engine.serve_tokens(i, tokens, metadata={'prompt_mode': 'templated', 'shared_prefix_hint_tokens': 6})
        
        print(f"  Result - cache hit: {result.was_cache_hit}, matched len: {result.matched_prefix_length}")
        print(f"  Policy plans so far: {engine.engine_metrics.get('policy_plans_total', 0)}")
        print(f"  Fast exact path hits: {engine.engine_metrics.get('fast_exact_path_hits', 0)}")
        
        # Check the bank after
        match_after = engine.bank.peek_match(tokens)
        print(f"  Match after: {match_after}")
        
        if i >= 1:  # After first request, check if fast path should work
            print(f"  Should use fast path: {engine._has_scaffold_hint(tokens, {'prompt_mode': 'templated', 'shared_prefix_hint_tokens': 6})}")

if __name__ == "__main__":
    debug_test()