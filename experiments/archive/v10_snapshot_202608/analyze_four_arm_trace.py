#!/usr/bin/env python3
"""
Analyze four-arm trace results: aggregate across seeds, compute CIs, produce tables.
"""
import argparse, json, math, sys
from pathlib import Path
from collections import defaultdict

ARMS = ['no_cache', 'native', 'overlay', 'enforced']
ARM_LABELS = {
    'no_cache': 'No cache',
    'native': 'Native (APC)',
    'overlay': 'Overlay (write-through)',
    'enforced': 'Enforced (skip-write)',
}
STUDENT_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def load_results(input_dir):
    """Load per-arm, per-seed summaries."""
    data = {arm: {} for arm in ARMS}
    in_dir = Path(input_dir)
    for arm in ARMS:
        arm_dir = in_dir / arm
        if not arm_dir.exists():
            print(f"  Warning: {arm_dir} not found")
            continue
        for seed_dir in sorted(arm_dir.glob('seed_*')):
            seed = int(seed_dir.name.split('_')[1])
            sf = seed_dir / 'summary.json'
            if sf.exists():
                with open(sf) as f:
                    data[arm][seed] = json.load(f)
    return data


def compute_aggregate(data):
    """Compute means and 95% CIs for each arm."""
    agg = {}
    for arm in ARMS:
        seed_data = data[arm]
        if not seed_data:
            continue
        n = len(seed_data)
        fields = list(next(iter(seed_data.values())).keys())
        result = {'seeds': n, 'seeds_list': sorted(seed_data.keys())}
        for field in fields:
            vals = [seed_data[s][field] for s in sorted(seed_data)]
            mu = sum(vals) / n
            if n > 1:
                var = sum((x - mu) ** 2 for x in vals) / (n - 1)
                se = math.sqrt(var / n)
                t = STUDENT_T.get(n - 1, 2.0)
                ci = t * se
            else:
                ci = 0.0
            result[field] = round(mu, 4)
            result[f'{field}_ci95'] = round(ci, 4)
        agg[arm] = result
    return agg


def print_table(agg):
    """Print paper-ready markdown tables."""
    def cell(arm, field, fmt='.2f'):
        v = agg[arm][field]
        ci = agg[arm].get(f'{field}_ci95', 0)
        if ci > 0 and abs(ci / v) > 0.001:
            return f"{v:{fmt}} $\\\\pm$ {ci:{fmt}}"
        return f"{v:{fmt}}"

    def pct(arm, field):
        v = agg[arm][field]
        ci = agg[arm].get(f'{field}_ci95', 0)
        if ci > 0:
            return f"{v*100:.1f}\\\% $\\\\pm$ {ci*100:.1f}\\\%"
        return f"{v*100:.1f}\\\%"

    print("## Results")
    print()
    print("| Metric | " + " | ".join(ARM_LABELS[a] for a in ARMS if a in agg) + " |")
    print("|--------|" + "|".join(":--------:" for _ in agg) + "|")

    if all(a in agg for a in ARMS):
        print("| Mean latency (ms) | " + " | ".join(cell(a, 'mean_latency_ms') for a in ARMS if a in agg) + " |")
        print("| P95 latency (ms) | " + " | ".join(cell(a, 'p95_latency_ms') for a in ARMS if a in agg) + " |")
        print("| Hit rate | " + " | ".join(pct(a, 'hit_rate') for a in ARMS if a in agg) + " |")
        print("| Evictions | " + " | ".join(cell(a, 'evictions', '.0f') for a in ARMS if a in agg) + " |")
        print("| Bypasses | " + " | ".join(cell(a, 'admission_bypass_total', '.0f') for a in ARMS if a in agg) + " |")
        print("| Skip-lookups | " + " | ".join(cell(a, 'admission_native_skip_lookup_total', '.0f') for a in ARMS if a in agg) + " |")
        print("| Skip-writes | " + " | ".join(cell(a, 'admission_native_skip_write_total', '.0f') for a in ARMS if a in agg) + " |")

    # Pairwise decomposition
    if all(a in agg for a in ['native', 'overlay', 'enforced']):
        print()
        print("### Pairwise Decomposition")
        print()
        l_n = agg['native']['mean_latency_ms']
        l_o = agg['overlay']['mean_latency_ms']
        l_e = agg['enforced']['mean_latency_ms']
        b_o = agg['overlay']['admission_bypass_total']
        b_e = agg['enforced']['admission_bypass_total']
        sw_e = agg['enforced']['admission_native_skip_write_total']

        print(f"| Contrast | Latency $\\\\Delta$ | Description |")
        print(f"|----------|:-----------------:|-------------|")
        print(f"| Overlay $-$ Native | $+{l_o - l_n:.1f}$ ms ($+{(l_o - l_n)/l_n*100:.1f}\\\%$) | Controller overhead |")
        print(f"| Enforced $-$ Overlay | ${l_e - l_o:.1f}$ ms (${(l_e - l_o)/l_o*100:.1f}\\\%$) | Enforcement benefit |")
        print(f"| Enforced $-$ Native | ${l_e - l_n:.1f}$ ms (${(l_e - l_n)/l_n*100:.1f}\\\%$) | Net value |")
        print()
        print(f"Bypass decisions: overlay={b_o:.0f}, enforced={b_e:.0f}")
        print(f"Writes actually skipped (enforced): {sw_e:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='results/four_arm_trace', help='Directory with per-arm/seed results')
    parser.add_argument('--output', help='Path to write aggregate JSON (optional)')
    args = parser.parse_args()

    print(f"Loading results from {args.input_dir}")
    data = load_results(args.input_dir)
    agg = compute_aggregate(data)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(agg, f, indent=2)
        print(f"Saved aggregate to {args.output}")

    print()
    print_table(agg)

    # Summary
    if all(a in agg for a in ['native', 'overlay', 'enforced']):
        l_n = agg['native']['mean_latency_ms']
        l_o = agg['overlay']['mean_latency_ms']
        l_e = agg['enforced']['mean_latency_ms']
        print()
        print("---")
        print("**Summary**")
        print(f"Gate cost: +{l_o - l_n:.1f} ms (+{(l_o - l_n)/l_n*100:.1f}%)")
        print(f"Enforcement benefit: {l_e - l_o:.1f} ms ({(l_e - l_o)/l_o*100:.1f}%)")
        print(f"Net value: {l_e - l_n:.1f} ms ({(l_e - l_n)/l_n*100:.1f}%)")
        n_seeds = agg['native'].get('seeds', 1)
        print(f"Based on {n_seeds} seed(s). CIs use Student's t-distribution.")


if __name__ == '__main__':
    main()
