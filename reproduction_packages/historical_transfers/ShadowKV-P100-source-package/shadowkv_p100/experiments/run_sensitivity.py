#!/usr/bin/env python3
"""
run_sensitivity.py -- one-at-a-time (OAT) sensitivity analysis of the policy
constants, on the fake backend (CPU, fast).

For each tuned constant we perturb it around its default and measure the
change in headline metrics (speedup, waste, hit-rate) for shadow_kv_plus.
Small changes => the net-utility framework is robust to the hand-set values.

Runs on --backend fake so it needs no GPU and does not compete with the P100.
Semantic-specific constants exercise weakly on the synthetic fake workload
(no real paraphrase families); those are marked and want a small HF sweep.

    python experiments/run_sensitivity.py                 # full sweep
    python experiments/run_sensitivity.py --quick         # baseline + 2 knobs
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

BASE_CONFIG = ROOT / "config" / "config.yaml"

# Tuned constants (bucket C): dotted config path -> which prompt modes exercise it.
KNOBS = {
    "policy.signals.templated_signal":               ["templated"],
    "policy.signals.entropy_penalty_weight":         ["raw", "templated"],
    "policy.health.base":                            ["raw", "templated"],
    "policy.health.hit_rate_weight":                 ["templated"],
    "policy.health.waste_ratio_weight":              ["templated"],
    "policy.health.ewma_alpha":                      ["templated"],
    "policy.utility.suffix_cost_ms_per_token":       ["templated"],
    "policy.utility.risk_aversion":                  ["templated"],   # already ablated; sanity
    "policy.utility.semantic_threshold":             ["semantic"],
    "policy.utility.semantic_prompt_benefit_discount": ["semantic"],
    "policy.utility.semantic_prompt_reuse_fraction": ["semantic"],
    "semantic.index.equivalence_key_boost":          ["semantic"],
    "semantic.sandbox.max_divergence":               ["semantic"],
}
FACTORS = [0.5, 0.75, 1.25, 1.5]
DATASET = "ag_news"        # real prompt structure (templated scaffolds + semantic families)
N_REQUESTS = 48


def load_base() -> dict:
    with open(BASE_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_dotted(cfg: dict, path: str, value) -> None:
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def get_dotted(cfg: dict, path: str):
    d = cfg
    for k in path.split("."):
        d = d[k]
    return d


def run_one(cfg: dict, mode: str, tmpdir: Path, tag: str) -> dict | None:
    cfg_path = tmpdir / f"cfg_{tag}.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    out = tmpdir / f"out_{tag}"
    cmd = [
        sys.executable, "experiments/run_benchmark.py",
        "--backend", "fake", "--workload", "public_dataset", "--dataset", DATASET,
        "--prompt_mode", mode, "--n_requests", str(N_REQUESTS),
        "--engines", "no_cache", "shadow_kv_plus",
        "--disable_arrival_simulation",
        "--config_path", str(cfg_path), "--output_dir", str(out),
    ]
    if mode == "semantic":
        cmd.append("--allow_unsafe_semantic_kv_reuse")
    rc = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if rc != 0:
        return None
    jf = sorted(out.glob("benchmark_*.json"))
    if not jf:
        return None
    d = json.loads(jf[-1].read_text(encoding="utf-8"))
    nc = d.get("no_cache", {}); skp = d.get("shadow_kv_plus", {})
    nc_lat = nc.get("mean_latency_ms"); skp_lat = skp.get("mean_latency_ms")
    if not nc_lat or not skp_lat:
        return None
    return {
        "speedup": nc_lat / skp_lat,
        "waste": skp.get("wasted_compute_ratio", 0.0),
        "hit_rate": skp.get("hit_rate", 0.0),
        "sem_hits": skp.get("semantic_partial_hits", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "sensitivity_results.csv"))
    args = ap.parse_args()

    base = load_base()
    knobs = dict(list(KNOBS.items())[:2]) if args.quick else KNOBS
    factors = [0.5, 1.5] if args.quick else FACTORS

    tmp = Path(tempfile.mkdtemp(prefix="sens_"))
    rows = []

    # Baselines per mode
    modes = sorted({m for ms in knobs.values() for m in ms})
    baseline = {}
    for mode in modes:
        r = run_one(base, mode, tmp, f"base_{mode}")
        baseline[mode] = r
        print(f"BASELINE {mode:10s} speedup={r['speedup']:.4f} waste={r['waste']:.4f} "
              f"hit={r['hit_rate']:.3f} sem_hits={r['sem_hits']}" if r else f"BASELINE {mode} FAILED")

    for knob, kmodes in knobs.items():
        default = get_dotted(base, knob)
        for mode in kmodes:
            b = baseline.get(mode)
            for fac in factors:
                cfg = copy.deepcopy(base)
                val = round(default * fac, 6)
                set_dotted(cfg, knob, val)
                tag = f"{knob.replace('.', '_')}_{mode}_{fac}"
                r = run_one(cfg, mode, tmp, tag)
                if not r or not b:
                    print(f"  {knob} @x{fac} {mode}: FAILED"); continue
                d_spd = (r["speedup"] - b["speedup"]) / b["speedup"] * 100 if b["speedup"] else 0
                d_wst = r["waste"] - b["waste"]
                rows.append({
                    "knob": knob, "default": default, "factor": fac, "value": val,
                    "mode": mode, "speedup": round(r["speedup"], 4),
                    "d_speedup_pct": round(d_spd, 2), "waste": round(r["waste"], 4),
                    "d_waste": round(d_wst, 4), "hit_rate": round(r["hit_rate"], 3),
                    "sem_hits": r["sem_hits"],
                })
                print(f"  {knob:48s} x{fac:<4} {mode:10s} "
                      f"speedup={r['speedup']:.4f} (d{d_spd:+.2f}%)  waste={r['waste']:.4f} (dw{d_wst:+.4f})")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.out}")

    # Tornado: max |Δspeedup| and max |Δwaste| per knob
    import collections
    infl = collections.defaultdict(lambda: {"spd": 0.0, "wst": 0.0})
    for r in rows:
        infl[r["knob"]]["spd"] = max(infl[r["knob"]]["spd"], abs(r["d_speedup_pct"]))
        infl[r["knob"]]["wst"] = max(infl[r["knob"]]["wst"], abs(r["d_waste"]))
    print("\n=== Influence ranking (max |Δ| over perturbations) ===")
    for k, v in sorted(infl.items(), key=lambda kv: kv[1]["spd"], reverse=True):
        print(f"  {k:50s} max|Δspeedup|={v['spd']:5.2f}%   max|Δwaste|={v['wst']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
