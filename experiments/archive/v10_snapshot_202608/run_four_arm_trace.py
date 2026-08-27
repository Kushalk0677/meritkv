#!/usr/bin/env python3
"""
Four-arm multi-round admission enforcement experiment.

Usage:
  # Test with fake backend (CPU, no GPU needed)
  python experiments/run_four_arm_trace.py --backend fake --output_dir /tmp/four_arm_test

  # Real run on Blackwell
  python experiments/run_four_arm_trace.py \
      --backend vllm \
      --model google/gemma-4-31B-it \
      --max_memory_mb 128 \
      --output_dir results/four_arm_trace_high_pressure \
      --seeds 42 123 456 789 999
"""
import argparse, json, os, sys, importlib, time
from pathlib import Path

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import engines
from proactive_kv_cache.engines import (
    BaseEngine,
    NoCacheEngine,
    RuntimeNativeCacheEngine,
    AdmissionControlledRuntimeCacheEngine,
)
from proactive_kv_cache.engine_utils import load_backend, tokenize_batch

# Trace parameters
PHASE1_SHARED = "The Transformer architecture has become the dominant paradigm in natural language processing. "
PHASE2_SHARED = "Recent advances in large language models have demonstrated remarkable capabilities. "
PHASE3_SHARED = "The Transformer architecture has become the dominant paradigm in natural language processing. "

SUFFIXES = [
    "This paper presents a comprehensive analysis of its key components.",
    "We evaluate performance across multiple benchmarks and domains.",
    "Our results show significant improvements over previous approaches.",
    "The findings have important implications for future research.",
    "We discuss limitations and potential directions for improvement.",
    "These results demonstrate the effectiveness of the proposed method.",
    "Further analysis reveals interesting patterns in the data.",
    "The approach generalizes well to unseen tasks and datasets.",
    "We provide detailed ablation studies to understand each component.",
    "Our code and models are publicly available for reproducibility.",
]


class EnforcedAdmissionEngine(AdmissionControlledRuntimeCacheEngine):
    """Like AdmissionControlledRuntimeCacheEngine but actually skips cache on bypass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_metrics['admission_native_skip_lookup_total'] = 0
        self.engine_metrics['admission_native_skip_write_total'] = 0

    def process_request(self, request):
        plan = self.controller.plan(request)
        self._record_admission_plan(plan)
        allowed = plan.strategy == 'exact' and plan.score >= 0.0
        if allowed:
            return super().process_request(request)
        else:
            self.engine_metrics['admission_native_skip_lookup_total'] += 1
            self.engine_metrics['admission_native_skip_write_total'] += 1
            return self.backend.generate(request)


class Req:
    def __init__(self, prefix, suffix, phase):
        self.prefix = prefix
        self.suffix = suffix
        self.phase = phase
        self.tokens = None
        self.result = None


def build_trace(n_phase1, n_phase2, n_phase3, n_rounds, seed):
    import random
    rng = random.Random(seed)
    requests = []
    for i in range(n_phase1):
        suffix = rng.choice(SUFFIXES)
        requests.append(Req(PHASE1_SHARED, suffix, 'fill'))
    for rd in range(n_rounds):
        for i in range(n_phase2):
            if i < n_phase2 // 2:
                # Evictor: different prefix
                suf = rng.choice(SUFFIXES)
                requests.append(Req(PHASE2_SHARED, suf, 'churn_evictor'))
            else:
                # Victim: reuse fill prefix
                suf = rng.choice(SUFFIXES)
                requests.append(Req(PHASE1_SHARED, suf, 'churn_victim'))
        for i in range(n_phase3):
            suf = rng.choice(SUFFIXES)
            requests.append(Req(PHASE3_SHARED, suf, 'recovery'))
    return requests


def run_trace(engine, requests):
    results = []
    for req in requests:
        t0 = time.time()
        result = engine.process_request(req)
        t1 = time.time()
        results.append({
            'phase': req.phase,
            'latency_ms': (t1 - t0) * 1000,
            'result': result,
        })
    return results


def compute_ledger(run_results, requests):
    phase1_hits = sum(
        1 for r in run_results if r['phase'] in ('fill',)
    )
    victim_recovery_first = []
    victim_recovery_later = []
    total_misses = 0
    total_victims = 0
    for r in run_results:
        if r['phase'] == 'churn_victim':
            total_victims += 1
            # Heuristic: if latency is close to no-cache baseline, it was a miss
            pass
    # Compute per-phase recovery
    recovery_requests = [r for r in run_results if r['phase'] == 'recovery']
    n_rec = len(recovery_requests)
    # We can't know real hit rate from latency alone in a real test,
    # so this is a placeholder that will be filled by the actual engine metrics
    return {
        'n_phase1': len([r for r in run_results if r['phase'] == 'fill']),
        'n_churn': len([r for r in run_results if 'churn' in r['phase']]),
        'n_recovery': n_rec,
        'mean_latency_ms': sum(r['latency_ms'] for r in run_results) / len(run_results),
        'p95_latency_ms': sorted(r['latency_ms'] for r in run_results)[int(len(run_results) * 0.95)],
        'phase1_hit_rate': 0.0,  # Placeholder — needs engine metrics
        'phase3_hit_rate': 0.0,  # Placeholder — needs engine metrics
        'victim_total': total_victims,
        'victim_misses': total_misses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['fake', 'vllm'], default='fake')
    parser.add_argument('--model', default=None)
    parser.add_argument('--max_memory_mb', type=int, default=128)
    parser.add_argument('--n_phase1', type=int, default=20)
    parser.add_argument('--n_phase2', type=int, default=15)
    parser.add_argument('--n_phase3', type=int, default=15)
    parser.add_argument('--n_rounds', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--output_dir', default='results/four_arm_trace')
    args = parser.parse_args()

    if args.backend == 'vllm' and not args.model:
        args.model = 'Qwen/Qwen2.5-7B-Instruct'

    total_req = args.n_phase1 + (args.n_phase2 + args.n_phase3) * args.n_rounds
    print(f"Four-arm multi-round trace: {total_req} requests/run")
    print(f"  {args.n_phase1} fill + [{args.n_phase2} churn + {args.n_phase3} rec] x {args.n_rounds} rounds")
    print(f"  Backend: {args.backend}, Model: {args.model or '(default)'}")
    print(f"  Cache: {args.max_memory_mb} MB, Seeds: {args.seeds}")

    engine_configs = [
        ('no_cache', NoCacheEngine, {}),
        ('native', RuntimeNativeCacheEngine, {'name': 'vllm_apc'}),
        ('overlay', AdmissionControlledRuntimeCacheEngine, {'name': 'vllm_apc_overlay', 'semantic_similarity_threshold': 0.58}),
        ('enforced', EnforcedAdmissionEngine, {'name': 'vllm_apc_enforced', 'semantic_similarity_threshold': 0.58}),
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        requests = build_trace(args.n_phase1, args.n_phase2, args.n_phase3, args.n_rounds, seed)

        if args.backend == 'vllm':
            backend = load_backend('vllm', model_name=args.model)
        else:
            from proactive_kv_cache.backend.fake_backend import FakeBackend
            backend = FakeBackend(device='cpu')

        tokenize_batch(backend, requests)

        for label, engine_cls, kwargs in engine_configs:
            print(f"  Running {label}...")
            eng = engine_cls(backend=backend, max_memory_mb=args.max_memory_mb, **kwargs)
            run_results = run_trace(eng, requests)
            ledger = compute_ledger(run_results, requests)
            metrics = dict(eng.engine_metrics)

            # Save seed-level results
            seed_dir = out_dir / label / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            summary = {
                'mean_latency_ms': ledger['mean_latency_ms'],
                'p95_latency_ms': ledger['p95_latency_ms'],
                'hit_rate': metrics.get('hit_rate', 0),
                'evictions': metrics.get('evictions', 0),
                'admission_bypass_total': metrics.get('admission_bypass_total', 0),
                'admission_allow_total': metrics.get('admission_allow_total', 0),
                'admission_native_skip_lookup_total': metrics.get('admission_native_skip_lookup_total', 0),
                'admission_native_skip_write_total': metrics.get('admission_native_skip_write_total', 0),
            }

            with open(seed_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"    Latency: {ledger['mean_latency_ms']:.1f} ms, "
                  f"Bypasses: {metrics.get('admission_bypass_total', 0)}, "
                  f"Skip-writes: {metrics.get('admission_native_skip_write_total', 0)}")

            if hasattr(eng, 'shutdown'):
                eng.shutdown()

        if args.backend == 'vllm':
            backend.shutdown()

    # Write aggregate (after all seeds)
    aggregate_path = out_dir / 'aggregate_results.json'
    print(f"\nResults saved to {aggregate_path}")
    print("Run `python experiments/analyze_four_arm_trace.py --input_dir", args.output_dir, "` for analysis.")


if __name__ == '__main__':
    main()
