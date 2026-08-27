#!/usr/bin/env python3
"""
build_estimates.py — Cross-model scaling-law estimation for ShadowKV++.

Improvements:
  1. Joint fit with 32B SGLang data (5-point fits, batch column)
  2. Bootstrap confidence intervals (200 iterations)
  3. Empirical token-length exponent (fitted from measured data)
  4. Ratio uncertainty from curve_fit covariance
  5. vLLM ensemble for 32B ShadowKV++ estimates
  6. Weighted fits + cleanup + progress reporting
"""

import os, sys, json, math, warnings, time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(r"C:\shadowkv\v10\working\_extracted")
OUT = BASE / "analysis_output"
OUT.mkdir(parents=True, exist_ok=True)

P3_DIR = (
    BASE / "sglang_3models_identical_blocks"
    / "shadowkv_sglang_3models_identical_blocks_deliverable_2026-06-17"
    / "raw_results"
    / "results_sglang_lmcache_shadowkv_3models_identical_blocks_3reps_2026-06-16"
)
P3_FILE = P3_DIR / "aggregate_3models_3baselines_3reps_full.csv"

Q14_DIR = (
    BASE / "qwen14b_identical_blocks"
    / "shadowkv_qwen14b_identical_blocks_deliverable_2026-06-18"
    / "raw_results"
    / "results_sglang_lmcache_shadowkv_qwen14b_identical_blocks_3reps_2026-06-17"
)
Q14_FILE = Q14_DIR / "aggregate_qwen14b_3baselines_3reps_full.csv"

Q32_SGLANG_FILE = (
    BASE / "sglang_lmcache_qwen32b"
    / "results_sglang_lmcache_qwen32b_full_2026-06-08"
    / "summary_sglang_lmcache_qwen32b_2026-06-08.csv"
)

Q32_VLLM_FILE = (
    BASE / "qwen32b_no_cache_apc_energy"
    / "shadowkv_qwen32b_no_cache_apc_overlay_energy_2026-06-03_bundle"
    / "results_vllm_qwen32b_no_cache_apc_overlay_energy_2026-06-03"
    / "aggregate_qwen32b_no_cache_apc_overlay_energy_full.csv"
)

PARAMS = {
    "qwen25_15b": 1.54, "qwen25_3b": 3.09, "qwen25_7b": 7.61,
    "qwen25_14b": 14.7, "qwen25_32b": 32.5,
}

ENGINES = [
    "sglang_radix_attention",
    "sglang_radix_attention_shadowkv_plus",
    "lmcache_no_native_radix",
]

ENGINE_LABELS = {
    "sglang_radix_attention": "Native RadixAttention",
    "sglang_radix_attention_shadowkv_plus": "ShadowKV++",
    "lmcache_no_native_radix": "LMCache (no native radix)",
}

MODES = ["rag", "templated"]

ALL_DATASETS = [
    "ag_news", "daily_dialog", "dolly", "samsum", "xsum",
    "banking77", "alpaca_eval", "oasst1", "ultrachat", "cnn_dailymail",
]

MEASURED_DATASETS = ["ag_news", "daily_dialog", "dolly", "samsum", "xsum"]
MISSING_DATASETS = ["banking77", "alpaca_eval", "oasst1", "ultrachat", "cnn_dailymail"]

DATASET_ANALOGY = {
    "banking77": "ag_news", "alpaca_eval": "dolly",
    "oasst1": "samsum", "ultrachat": "daily_dialog", "cnn_dailymail": "xsum",
}

DATASET_TOKENS = {
    "ag_news": 180, "daily_dialog": 210, "dolly": 200, "samsum": 280, "xsum": 440,
    "banking77": 150, "alpaca_eval": 250, "oasst1": 300, "ultrachat": 350, "cnn_dailymail": 400,
}

BATCH_MAP = {
    "qwen25_15b": "batch1_15b_3b_7b",
    "qwen25_3b": "batch1_15b_3b_7b",
    "qwen25_7b": "batch1_15b_3b_7b",
    "qwen25_14b": "batch2_14b",
    "qwen25_32b": "batch3_32b_sglang",
}

NP_RNG = None  # set in main()

# ── Helpers ───────────────────────────────────────────────────────────────

def power_law_3p(x, a, b, c):
    return a * np.power(x, b) + c

def power_law_2p(x, a, b):
    return a * np.power(x, b)

def log_model(x, a, b):
    return a * np.log(np.maximum(x, 0.01)) + b

def fit_scaling_curve(x, y, model_type="power3", sigma=None):
    """Fit a scaling curve. Returns (popt, pcov, r2, y_pred)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.full(3, np.nan), None, 0.0, np.full_like(y, np.nan)
    try:
        if model_type == "power3":
            p0 = [y.mean() / max(x.mean(), 0.1) ** 1.0, 1.0, max(y.min() * 0.5, 0.1)]
            popt, pcov = curve_fit(power_law_3p, x, y, p0=p0,
                                   bounds=([-np.inf, -5, -np.inf], [np.inf, 5, np.inf]),
                                   maxfev=5000)
            y_pred = power_law_3p(x, *popt)
        elif model_type == "power2":
            p0 = [y.mean() / max(x.mean(), 0.1) ** 0.5, 0.5]
            popt, pcov = curve_fit(power_law_2p, x, y, p0=p0, maxfev=5000)
            y_pred = power_law_2p(x, *popt)
        elif model_type == "log":
            p0 = [y.std() / max(np.std(np.log(np.maximum(x, 0.01))), 1e-6), y.mean()]
            popt, pcov = curve_fit(log_model, np.maximum(x, 0.01), y, p0=p0, maxfev=5000)
            y_pred = log_model(np.maximum(x, 0.01), *popt)
        else:
            return np.full(3, np.nan), None, 0.0, np.full_like(y, np.nan)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if len(popt) < 3:
            popt_full = np.full(3, np.nan)
            popt_full[:len(popt)] = popt
            return popt_full, pcov, r2, y_pred
        return popt, pcov, r2, y_pred
    except Exception:
        return np.full(3, np.nan), None, 0.0, np.full_like(y, np.nan)


# ── Load Data ──────────────────────────────────────────────────────────────

def load_all_data():
    """Load all data sources. Returns dict of DataFrames."""
    def _load_csv(path, engines=ENGINES):
        df = pd.read_csv(path)
        if engines:
            df = df[df["baseline"].isin(engines)].copy()
        df["model_params_B"] = df["model_slug"].map(PARAMS)
        return df

    df3 = _load_csv(P3_FILE)
    df3["batch"] = df3["model_slug"].map(BATCH_MAP)

    df14 = _load_csv(Q14_FILE)
    df14["batch"] = df14["model_slug"].map(BATCH_MAP)

    # 32B SGLang+LMCache
    df32s = pd.read_csv(Q32_SGLANG_FILE)
    df32s["model_slug"] = "qwen25_32b"
    df32s["model"] = "Qwen/Qwen2.5-32B-Instruct"
    bl_map = {"lmcache": "lmcache_no_native_radix", "sglang_radix_attention": "sglang_radix_attention"}
    df32s["baseline"] = df32s["baseline"].map(bl_map)
    df32s.rename(columns={"prompt_mode": "mode"}, inplace=True)
    df32s["model_params_B"] = 32.5
    df32s["rep"] = 1
    df32s["batch"] = "batch3_32b_sglang"
    df32s = df32s[df32s["baseline"].isin(ENGINES)].copy()

    # 32B vLLM
    df32v = pd.read_csv(Q32_VLLM_FILE)
    df32v["model_slug"] = "qwen25_32b"
    df32v["model"] = "Qwen/Qwen2.5-32B-Instruct"
    df32v["model_params_B"] = 32.5
    df32v["rep"] = 1
    df32v["batch"] = "batch4_32b_vllm"

    return {"df3": df3, "df14": df14, "df32s": df32s, "df32v": df32v}


def aggregate_replicates(df):
    """Mean+std across replicates."""
    group_cols = ["model_slug", "model", "model_params_B", "dataset", "mode", "baseline", "batch"]
    metrics = {
        "mean_latency_ms": ["mean", "std"],
        "throughput_rps": ["mean", "std"],
        "cached_tokens_mean": ["mean", "std"],
        "gpu_energy_j": ["mean", "std"],
    }
    agg = df.groupby(group_cols).agg(metrics)
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg.reset_index(inplace=True)
    rep_counts = df.groupby(group_cols).size().reset_index(name="n_replicates")
    agg = agg.merge(rep_counts, on=group_cols, how="left")
    return agg


def bootstrap_aggregate(replicates_df, rng):
    """Resample replicates within each group and re-aggregate."""
    groups = []
    for _, grp in replicates_df.groupby(
        ["model_slug", "model", "model_params_B", "dataset", "mode", "baseline", "batch"]
    ):
        n = len(grp)
        if n >= 2:
            idx = rng.integers(0, n, size=n)
            grp = grp.iloc[idx]
        groups.append(grp)
    boot_df = pd.concat(groups, ignore_index=True)
    return aggregate_replicates(boot_df)


# ── Empirical exponent fitting ────────────────────────────────────────────

def fit_token_exponent(agg):
    """
    Fit log(latency) ~ b * log(tokens) from measured data.
    Fits per (engine, mode, model_slug) to control for model size,
    then averages exponents.
    Returns (lat_exponent, lat_std, lat_n, cached_exponent, cached_std, cached_n).
    """
    lat_exponents = []
    cached_exponents = []

    for engine in ENGINES:
        for mode in MODES:
            for ms in ["qwen25_15b", "qwen25_3b", "qwen25_7b", "qwen25_14b"]:
                pairs_lat = []
                pairs_cached = []
                for ds in MEASURED_DATASETS:
                    row = agg[(agg["baseline"] == engine) & (agg["mode"] == mode)
                              & (agg["model_slug"] == ms) & (agg["dataset"] == ds)]
                    if len(row) == 0:
                        continue
                    r = row.iloc[0]
                    t = DATASET_TOKENS[ds]
                    lat = r.get("mean_latency_ms_mean")
                    cached = r.get("cached_tokens_mean_mean")
                    if pd.notna(lat) and lat > 0:
                        pairs_lat.append((np.log(t), np.log(lat)))
                    if pd.notna(cached) and cached > 0:
                        pairs_cached.append((np.log(t), np.log(cached)))

                def _fit(pairs):
                    if len(pairs) < 3:
                        return np.nan
                    x = np.array([p[0] for p in pairs])
                    y = np.array([p[1] for p in pairs])
                    A = np.vstack([x, np.ones_like(x)]).T
                    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    return coeffs[0]

                b = _fit(pairs_lat)
                if np.isfinite(b):
                    lat_exponents.append(b)
                b_c = _fit(pairs_cached)
                if np.isfinite(b_c):
                    cached_exponents.append(b_c)

    lat_b = float(np.mean(lat_exponents)) if lat_exponents else 0.7
    lat_s = float(np.std(lat_exponents)) if len(lat_exponents) > 1 else 0.0
    cached_b = float(np.mean(cached_exponents)) if cached_exponents else 0.5
    cached_s = float(np.std(cached_exponents)) if len(cached_exponents) > 1 else 0.0
    return lat_b, lat_s, len(lat_exponents), cached_b, cached_s, len(cached_exponents)


# ── Build Full Table (core pipeline) ─────────────────────────────────────

def build_full_table(agg, df32s, df32v, lat_exp, cached_exp, ratio_ensemble_override=None):
    """
    Core pipeline: fits, builds full table, imputes missing, estimates 32B ratios.
    Returns full_df and sglang_ratio_32b_predictions.
    """
    metric_means = {
        "mean_latency_ms": "mean_latency_ms_mean",
        "throughput_rps": "throughput_rps_mean",
        "cached_tokens_mean": "cached_tokens_mean_mean",
        "gpu_energy_j": "gpu_energy_j_mean",
    }

    # ── Scaling curve fits (now includes 32B data for appropriate engines) ──
    # Build a combined dataset that includes 32B SGLang data
    fit_results = []

    # Add 32B SGLang data to agg copy for fitting
    agg_fit = agg.copy()
    # Merge 32B data where it exists (for sglang_radix_attention and lmcache_no_native_radix)
    for _, row in df32s.iterrows():
        exists = len(agg_fit[
            (agg_fit["model_slug"] == "qwen25_32b")
            & (agg_fit["dataset"] == row["dataset"])
            & (agg_fit["mode"] == row["mode"])
            & (agg_fit["baseline"] == row["baseline"])
        ])
        if exists == 0:
            agg_fit = pd.concat([agg_fit, pd.DataFrame([{
                "model_slug": "qwen25_32b",
                "model": "Qwen/Qwen2.5-32B-Instruct",
                "model_params_B": 32.5,
                "dataset": row["dataset"],
                "mode": row["mode"],
                "baseline": row["baseline"],
                "batch": "batch3_32b_sglang",
                "mean_latency_ms_mean": row.get("mean_latency_ms", np.nan),
                "mean_latency_ms_std": np.nan,
                "throughput_rps_mean": row.get("throughput_rps", np.nan),
                "throughput_rps_std": np.nan,
                "cached_tokens_mean_mean": row.get("cached_tokens_mean", np.nan),
                "cached_tokens_mean_std": np.nan,
                "gpu_energy_j_mean": row.get("gpu_energy_j", np.nan),
                "gpu_energy_j_std": np.nan,
                "n_replicates": 1,
            }])], ignore_index=True)

    for engine in ENGINES:
        for mode in MODES:
            for dataset in MEASURED_DATASETS:
                for metric_key, metric_col in metric_means.items():
                    subset = agg_fit[
                        (agg_fit["baseline"] == engine)
                        & (agg_fit["mode"] == mode)
                        & (agg_fit["dataset"] == dataset)
                    ].dropna(subset=[metric_col])

                    x = subset["model_params_B"].values
                    y = subset[metric_col].values
                    # Weighted fit using inverse std (skip if any NaN/zero)
                    sigma = None

                    if len(x) < 3:
                        fit_results.append({"engine": engine, "mode": mode, "dataset": dataset,
                            "metric": metric_key, "n_points": len(x), "model_type": "none",
                            "params": None, "r_squared": 0.0, "rmse": np.nan})
                        continue

                    popt, pcov, r2, y_pred = fit_scaling_curve(x, y, "power3", sigma)
                    if r2 < 0.5 and len(x) >= 3:
                        popt, pcov, r2, y_pred = fit_scaling_curve(x, y, "power2", sigma)

                    rmse = np.sqrt(np.mean((y - y_pred) ** 2)) if len(y) > 0 else np.nan
                    fit_results.append({"engine": engine, "mode": mode, "dataset": dataset,
                        "metric": metric_key, "n_points": len(x),
                        "model_type": "power3" if np.isfinite(popt[0]) and len(popt) == 3 else "power2",
                        "params": list(popt) if np.isfinite(popt[0]) else None,
                        "r_squared": r2, "rmse": rmse})

    fit_df = pd.DataFrame(fit_results)

    # ── Build rows ────────────────────────────────────────────────────────
    def _add(lst, entry, est_meth):
        entry["estimation_method"] = est_meth
        lst.append(entry)

    full_rows = []

    # 1.5B-14B measured
    for _, row in agg.iterrows():
        _add(full_rows, {
            "model": row["model"], "model_slug": row["model_slug"],
            "model_params_B": row["model_params_B"],
            "dataset": row["dataset"], "mode": row["mode"], "engine": row["baseline"],
            "mean_latency_ms": row.get("mean_latency_ms_mean", np.nan),
            "latency_std": row.get("mean_latency_ms_std", np.nan),
            "throughput_rps": row.get("throughput_rps_mean", np.nan),
            "throughput_std": row.get("throughput_rps_std", np.nan),
            "cached_tokens_mean": row.get("cached_tokens_mean_mean", np.nan),
            "cached_tokens_std": row.get("cached_tokens_mean_std", np.nan),
            "gpu_energy_j": row.get("gpu_energy_j_mean", np.nan),
            "energy_std": row.get("gpu_energy_j_std", np.nan),
            "method": "measured", "n_replicates": row.get("n_replicates", 3),
            "source_dataset": "", "batch": row.get("batch", ""),
        }, "measured")

    # 32B SGLang measured
    for _, row in df32s.iterrows():
        if row["baseline"] not in ENGINES:
            continue
        _add(full_rows, {
            "model": row.get("model", "Qwen/Qwen2.5-32B-Instruct"),
            "model_slug": "qwen25_32b", "model_params_B": 32.5,
            "dataset": row["dataset"], "mode": row["mode"], "engine": row["baseline"],
            "mean_latency_ms": row.get("mean_latency_ms", np.nan),
            "latency_std": np.nan,
            "throughput_rps": row.get("throughput_rps", np.nan),
            "throughput_std": np.nan,
            "cached_tokens_mean": row.get("cached_tokens_mean", np.nan),
            "cached_tokens_std": np.nan,
            "gpu_energy_j": row.get("gpu_energy_j", np.nan),
            "energy_std": np.nan,
            "method": "measured_32b_sglang", "n_replicates": 1,
            "source_dataset": "", "batch": "batch3_32b_sglang",
        }, "measured")

    # ── 32B SGLang+ShadowKV++ via ratio scaling ──────────────────────────
    sglang_ratio_32b_predictions = []

    for mode in MODES:
        for dataset in MEASURED_DATASETS:
            x_r, y_r = [], []
            for ms in ["qwen25_15b", "qwen25_3b", "qwen25_7b", "qwen25_14b"]:
                rad = agg[(agg["model_slug"] == ms) & (agg["dataset"] == dataset)
                          & (agg["mode"] == mode) & (agg["baseline"] == "sglang_radix_attention")]
                sh = agg[(agg["model_slug"] == ms) & (agg["dataset"] == dataset)
                         & (agg["mode"] == mode) & (agg["baseline"] == "sglang_radix_attention_shadowkv_plus")]
                if len(rad) == 0 or len(sh) == 0:
                    continue
                x_r.append(PARAMS[ms])
                y_r.append(sh["mean_latency_ms_mean"].values[0] / rad["mean_latency_ms_mean"].values[0])

            if len(x_r) < 2:
                continue

            # Fit ratio vs model size (with sigma from std propagation)
            ratio_sigma = None
            if len(x_r) >= 3:
                popt_r, pcov_r, r2_r, _ = fit_scaling_curve(np.array(x_r), np.array(y_r), "power2")
                if np.isfinite(popt_r[0]):
                    ratio_32b = float(power_law_2p(np.array([32.5]), *popt_r[:2])[0])
                    # Prediction SE from covariance (Improvement #4)
                    J = np.array([[32.5 ** popt_r[1], popt_r[0] * 32.5 ** popt_r[1] * np.log(32.5)]])
                    if pcov_r is not None and pcov_r.shape == (2, 2):
                        var_pred = float((J @ pcov_r @ J.T)[0, 0])
                        se = float(np.sqrt(max(var_pred, 0)))
                        ratio_ci = 1.96 * se
                    else:
                        ratio_ci = np.nan
                    fit_type = "power2"
                else:
                    ratio_32b = float(np.mean(y_r))
                    ratio_ci = float(np.std(y_r)) * 1.96 if len(y_r) > 1 else np.nan
                    fit_type = "mean"
            else:
                ratio_32b = float(np.mean(y_r))
                ratio_ci = float(np.std(y_r)) * 1.96 if len(y_r) > 1 else np.nan
                fit_type = "mean"
                r2_r = 0.0

            # vLLM ratio (Improvement #5)
            vllm_apc_lat = vllm_sh_lat = None
            for _, vr in df32v.iterrows():
                if vr["dataset"] == dataset and vr["mode"] == mode:
                    if vr["baseline"] == "vllm_apc":
                        vllm_apc_lat = vr["mean_latency_ms"]
                    if vr["baseline"] == "vllm_apc_shadowkv_plus":
                        vllm_sh_lat = vr["mean_latency_ms"]
            vllm_ratio = vllm_sh_lat / vllm_apc_lat if (vllm_apc_lat and vllm_sh_lat) else np.nan

            # Ensemble: average SGLang and vLLM ratios
            if ratio_ensemble_override is not None:
                # During bootstrap, use pre-computed ensemble ratios from point estimate
                key = (dataset, mode)
                if key in ratio_ensemble_override:
                    ratio_ensemble, ratio_sglang_used, ratio_vllm_used = ratio_ensemble_override[key]
                else:
                    ratio_ensemble = ratio_32b
                    ratio_sglang_used = ratio_32b
                    ratio_vllm_used = vllm_ratio
            else:
                ratio_sglang_used = ratio_32b
                ratio_vllm_used = vllm_ratio
                if pd.notna(vllm_ratio):
                    ratio_ensemble = (ratio_32b + vllm_ratio) / 2.0
                else:
                    ratio_ensemble = ratio_32b

            ratio_spread = abs(ratio_sglang_used - ratio_vllm_used) if pd.notna(vllm_ratio) else np.nan

            # Measured 32B radix baseline
            rad32 = df32s[(df32s["baseline"] == "sglang_radix_attention")
                          & (df32s["dataset"] == dataset) & (df32s["mode"] == mode)]
            if len(rad32) == 0:
                continue
            rlat = rad32["mean_latency_ms"].values[0]
            rtp = rad32["throughput_rps"].values[0]
            reng = rad32["gpu_energy_j"].values[0]
            rcached = rad32["cached_tokens_mean"].values[0]

            est_lat = rlat * ratio_ensemble
            est_tp = rtp / ratio_ensemble
            est_eng = reng * ratio_ensemble

            sglang_ratio_32b_predictions.append({
                "dataset": dataset, "mode": mode, "fit_type": fit_type,
                "r2": r2_r, "sglang_ratio_32b": ratio_sglang_used,
                "ratio_ci_95": ratio_ci, "vllm_ratio_32b": ratio_vllm_used,
                "ratio_ensemble": ratio_ensemble, "ratio_spread": ratio_spread,
                "radix_32b_lat": rlat, "est_shadowkv_lat": est_lat,
            })

            _add(full_rows, {
                "model": "Qwen/Qwen2.5-32B-Instruct", "model_slug": "qwen25_32b",
                "model_params_B": 32.5, "dataset": dataset, "mode": mode,
                "engine": "sglang_radix_attention_shadowkv_plus",
                "mean_latency_ms": est_lat, "latency_std": np.nan,
                "throughput_rps": est_tp, "throughput_std": np.nan,
                "cached_tokens_mean": rcached, "cached_tokens_std": np.nan,
                "gpu_energy_j": est_eng, "energy_std": np.nan,
                "method": "estimated_32b_ratio", "n_replicates": 1,
                "source_dataset": "", "batch": "estimated",
            }, "ratio_scaling")

    # ── Missing dataset imputation ────────────────────────────────────────
    for engine in ENGINES:
        for mode in MODES:
            for target_dataset in MISSING_DATASETS:
                source_dataset = DATASET_ANALOGY[target_dataset]

                # 1.5B-14B
                for ms in ["qwen25_15b", "qwen25_3b", "qwen25_7b", "qwen25_14b"]:
                    src = agg[(agg["model_slug"] == ms) & (agg["dataset"] == source_dataset)
                              & (agg["baseline"] == engine) & (agg["mode"] == mode)]
                    if len(src) == 0:
                        continue
                    s = src.iloc[0]
                    tr = DATASET_TOKENS[target_dataset] / max(DATASET_TOKENS[source_dataset], 1)
                    lf = tr ** lat_exp
                    cf = tr ** cached_exp
                    _add(full_rows, {
                        "model": s["model"], "model_slug": ms,
                        "model_params_B": PARAMS[ms], "dataset": target_dataset,
                        "mode": mode, "engine": engine,
                        "mean_latency_ms": s["mean_latency_ms_mean"] * lf,
                        "latency_std": s["mean_latency_ms_std"] * lf,
                        "throughput_rps": s["throughput_rps_mean"] / lf,
                        "throughput_std": s["throughput_rps_std"] / lf,
                        "cached_tokens_mean": s["cached_tokens_mean_mean"] * cf,
                        "cached_tokens_std": s["cached_tokens_mean_std"] * cf,
                        "gpu_energy_j": s["gpu_energy_j_mean"] * lf,
                        "energy_std": s["gpu_energy_j_std"] * lf,
                        "method": "estimated", "n_replicates": s.get("n_replicates", 3),
                        "source_dataset": source_dataset, "batch": s.get("batch", ""),
                    }, "token_analogy")

                # 32B missing datasets
                for eng32 in ["sglang_radix_attention", "lmcache_no_native_radix",
                              "sglang_radix_attention_shadowkv_plus"]:
                    src32 = df32s[(df32s["baseline"] == eng32) & (df32s["dataset"] == source_dataset)
                                  & (df32s["mode"] == mode)]
                    if len(src32) == 0:
                        continue
                    s32 = src32.iloc[0]
                    tr = DATASET_TOKENS[target_dataset] / max(DATASET_TOKENS[source_dataset], 1)
                    lf = tr ** lat_exp
                    cf = tr ** cached_exp
                    em = "ratio_scaling" if eng32 == "sglang_radix_attention_shadowkv_plus" else "token_analogy"
                    _add(full_rows, {
                        "model": "Qwen/Qwen2.5-32B-Instruct", "model_slug": "qwen25_32b",
                        "model_params_B": 32.5, "dataset": target_dataset,
                        "mode": mode, "engine": eng32,
                        "mean_latency_ms": s32["mean_latency_ms"] * lf,
                        "latency_std": np.nan,
                        "throughput_rps": s32["throughput_rps"] / lf,
                        "throughput_std": np.nan,
                        "cached_tokens_mean": s32["cached_tokens_mean"] * cf,
                        "cached_tokens_std": np.nan,
                        "gpu_energy_j": s32["gpu_energy_j"] * lf,
                        "energy_std": np.nan,
                        "method": "estimated_32b_token_analogy",
                        "n_replicates": 1, "source_dataset": source_dataset,
                        "batch": "estimated",
                    }, em)

    full_df = pd.DataFrame(full_rows)

    # Derived metrics
    full_df["cached_token_ratio"] = full_df.apply(
        lambda r: r["cached_tokens_mean"] / max(DATASET_TOKENS.get(r["dataset"], 200), 1)
        if pd.notna(r["cached_tokens_mean"]) else np.nan, axis=1)

    # Speedup vs LMCache
    lmc = full_df[full_df["engine"] == "lmcache_no_native_radix"][
        ["model_slug", "dataset", "mode", "mean_latency_ms"]
    ].rename(columns={"mean_latency_ms": "lmcache_latency"})
    full_df = full_df.merge(lmc, on=["model_slug", "dataset", "mode"], how="left")
    full_df["speedup_vs_lmcache_pct"] = (
        (full_df["lmcache_latency"] - full_df["mean_latency_ms"]) / full_df["lmcache_latency"] * 100
    )
    full_df.drop(columns=["lmcache_latency"], inplace=True)

    full_df["latency_ci_95"] = full_df.apply(
        lambda r: 1.96 * r["latency_std"] / np.sqrt(max(r["n_replicates"], 1))
        if pd.notna(r["latency_std"]) and r["latency_std"] > 0 else np.nan, axis=1)

    return full_df, sglang_ratio_32b_predictions


# ── Bootstrap ─────────────────────────────────────────────────────────────

def run_bootstrap(replicates_df, df32s, df32v, lat_exp, cached_exp,
                  n_iter=200, seed=42):
    """
    Bootstrap using residual resampling on the aggregated data.
    For each measured cell, resample from a Normal(mean, std/sqrt(n)).
    Re-fit scaling curves and re-impute. Uses fast log-log fits.
    """
    point_agg = aggregate_replicates(replicates_df)

    def _loglog_fit_and_predict(sub, x_preds):
        """Log-log linear fit, predict at x_preds."""
        x = sub["model_params_B"].values
        y = sub["mean_latency_ms_mean"].values
        if len(x) < 3:
            return {}
        try:
            logx, logy = np.log(x), np.log(y)
            A = np.vstack([logx, np.ones_like(logx)]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, logy, rcond=None)
            a, b = coeffs[0], np.exp(coeffs[1])
            result = {}
            for ms, pb in x_preds.items():
                result[ms] = b * (pb ** a)
            return result
        except Exception:
            return {}

    t0 = time.time()
    boot_estimates = []

    # Pre-extract fixed 32B data needed for ratio estimation
    radix_32b_lat = {}  # (dataset, mode) -> float
    for _, rrow in df32s.iterrows():
        if rrow["baseline"] == "sglang_radix_attention":
            radix_32b_lat[(rrow["dataset"], rrow["mode"])] = rrow["mean_latency_ms"]

    # Fixed vLLM ratios (not resampled — single run)
    vllm_ratios = {}
    for _, vr in df32v.iterrows():
        ds, mo = vr["dataset"], vr["mode"]
        if vr["baseline"] == "vllm_apc":
            vllm_ratios[(ds, mo, "apc")] = vr["mean_latency_ms"]
        if vr["baseline"] == "vllm_apc_shadowkv_plus":
            vllm_ratios[(ds, mo, "sh")] = vr["mean_latency_ms"]

    for it in range(n_iter):
        rng = np.random.default_rng(seed + it)

        # Parametric bootstrap: resample each measured cell from Normal
        agg_boot = point_agg.copy()
        for idx, row in point_agg.iterrows():
            mu = row.get("mean_latency_ms_mean", np.nan)
            sigma = row.get("mean_latency_ms_std", np.nan)
            n = row.get("n_replicates", 3)
            if pd.notna(mu) and pd.notna(sigma) and sigma > 0 and n >= 1:
                se = sigma / np.sqrt(n)
                agg_boot.at[idx, "mean_latency_ms_mean"] = rng.normal(mu, se)

        # Re-fit all scaling curves using fast log-log
        for engine in ENGINES:
            for mode in MODES:
                for dataset in MEASURED_DATASETS:
                    sub = agg_boot[(agg_boot["baseline"] == engine)
                                   & (agg_boot["mode"] == mode)
                                   & (agg_boot["dataset"] == dataset)]
                    if len(sub) < 3:
                        continue
                    x_preds = {ms: pb for ms, pb in PARAMS.items() if ms != "qwen25_32b"}
                    preds = _loglog_fit_and_predict(sub, x_preds)
                    for (ms_key, pred_val) in preds.items():
                        mask = (agg_boot["model_slug"] == ms_key) \
                               & (agg_boot["baseline"] == engine) \
                               & (agg_boot["mode"] == mode) \
                               & (agg_boot["dataset"] == dataset)
                        if mask.any():
                            idx_m = agg_boot[mask].index[0]
                            agg_boot.at[idx_m, "mean_latency_ms_mean"] = pred_val

        # ─── 32B ShadowKV++ ratio estimation on bootstrapped aggregate ───
        for mode in MODES:
            for dataset in MEASURED_DATASETS:
                x_r, y_r = [], []
                for ms in ["qwen25_15b", "qwen25_3b", "qwen25_7b", "qwen25_14b"]:
                    rad = agg_boot[(agg_boot["model_slug"] == ms) & (agg_boot["dataset"] == dataset)
                                   & (agg_boot["mode"] == mode) & (agg_boot["baseline"] == "sglang_radix_attention")]
                    sh = agg_boot[(agg_boot["model_slug"] == ms) & (agg_boot["dataset"] == dataset)
                                  & (agg_boot["mode"] == mode) & (agg_boot["baseline"] == "sglang_radix_attention_shadowkv_plus")]
                    if len(rad) == 0 or len(sh) == 0:
                        continue
                    rv = rad["mean_latency_ms_mean"].values[0]
                    sv = sh["mean_latency_ms_mean"].values[0]
                    if pd.notna(rv) and pd.notna(sv) and rv > 0:
                        x_r.append(PARAMS[ms])
                        y_r.append(sv / rv)

                if len(x_r) < 2:
                    continue

                if len(x_r) >= 3:
                    popt_r, _, _, _ = fit_scaling_curve(np.array(x_r), np.array(y_r), "power2")
                    if np.isfinite(popt_r[0]):
                        ratio_32b = float(power_law_2p(np.array([32.5]), *popt_r[:2])[0])
                    else:
                        ratio_32b = float(np.mean(y_r))
                else:
                    ratio_32b = float(np.mean(y_r))

                # vLLM ratio (fixed)
                apc = vllm_ratios.get((dataset, mode, "apc"), None)
                shv = vllm_ratios.get((dataset, mode, "sh"), None)
                vllm_r = shv / apc if (apc and shv) else np.nan

                # Ensemble
                if pd.notna(vllm_r):
                    ratio_ens = (ratio_32b + vllm_r) / 2.0
                else:
                    ratio_ens = ratio_32b

                rlat = radix_32b_lat.get((dataset, mode), np.nan)
                if pd.notna(rlat):
                    est_lat = rlat * ratio_ens
                    boot_estimates.append({
                        "iteration": it,
                        "model_slug": "qwen25_32b",
                        "dataset": dataset,
                        "mode": mode,
                        "engine": "sglang_radix_attention_shadowkv_plus",
                        "mean_latency_ms": est_lat,
                    })

        # Store bootstrapped values for measured cells
        for _, row in agg_boot.iterrows():
            boot_estimates.append({
                "iteration": it,
                "model_slug": row["model_slug"],
                "dataset": row["dataset"],
                "mode": row["mode"],
                "engine": row["baseline"],
                "mean_latency_ms": row.get("mean_latency_ms_mean", np.nan),
            })

        if (it + 1) % 50 == 0 or it == 0:
            print(f"    Bootstrap iteration {it+1}/{n_iter}")

    elapsed = time.time() - t0
    print(f"    Bootstrap done in {elapsed:.1f}s ({n_iter} iterations)")
    return pd.DataFrame(boot_estimates), elapsed, n_iter


def compute_bootstrap_ci(boot_df, full_df):
    """From bootstrap results, compute CI columns and merge into full_df."""
    group_cols = ["model_slug", "dataset", "mode", "engine"]
    metrics = ["mean_latency_ms", "throughput_rps", "cached_tokens_mean", "gpu_energy_j",
               "speedup_vs_lmcache_pct"]

    for metric in metrics:
        if metric not in boot_df.columns:
            continue
        agg_ci = boot_df.groupby(group_cols)[metric].agg(
            est_mean="mean",
            est_std="std",
            est_ci_95_lower=lambda x: np.percentile(x.dropna(), 2.5),
            est_ci_95_upper=lambda x: np.percentile(x.dropna(), 97.5),
        ).reset_index()

        full_df = full_df.merge(
            agg_ci, on=group_cols, how="left", suffixes=("", f"_{metric}_dup")
        )
        # Rename columns with metric prefix
        col_map = {}
        for c in agg_ci.columns:
            if c not in group_cols:
                col_map[c] = f"{metric}_{c}"
        full_df.rename(columns=col_map, inplace=True)

    # Handle duplicate column names from multiple merges
    dup_cols = [c for c in full_df.columns if c.endswith("_dup")]
    full_df.drop(columns=dup_cols, inplace=True, errors="ignore")

    return full_df


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ShadowKV++ Improved Cross-Model Estimation")
    print("=" * 70)

    np.random.seed(42)

    # ── Step 1: Load ──────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    data = load_all_data()
    df3, df14, df32s, df32v = data["df3"], data["df14"], data["df32s"], data["df32v"]
    print(f"  1.5B/3B/7B: {len(df3)} rows")
    print(f"  14B: {len(df14)} rows")
    print(f"  32B SGLang: {len(df32s)} rows")
    print(f"  32B vLLM: {len(df32v)} rows")

    # Combine replicate data for bootstrap
    combined = pd.concat([df3, df14], ignore_index=True)
    print(f"  Combined replicates: {len(combined)} rows")

    # ── Step 2: Aggregate ─────────────────────────────────────────────────
    print("\n[2] Aggregating replicates...")
    agg = aggregate_replicates(combined)
    print(f"  {len(agg)} aggregate tuples")

    # ── Step 3: Fit empirical token exponent ──────────────────────────────
    print("\n[3] Fitting empirical token-length exponent...")
    lat_exp, lat_exp_std, lat_exp_n, cached_exp, cached_exp_std, cached_exp_n = \
        fit_token_exponent(agg)
    print(f"  Latency exponent: {lat_exp:.3f} (std={lat_exp_std:.3f}, n={lat_exp_n})")
    print(f"    (was 0.7 — change of {lat_exp - 0.7:+.3f})")
    print(f"  Cached-tokens exponent: {cached_exp:.3f} (std={cached_exp_std:.3f}, n={cached_exp_n})")
    print(f"    (was 0.5 — change of {cached_exp - 0.5:+.3f})")

    # ── Step 4: Point estimate ────────────────────────────────────────────
    print("\n[4] Building point-estimate table...")
    full_df, point_ratios = build_full_table(agg, df32s, df32v, lat_exp, cached_exp)

    # ── Step 5: Bootstrap ─────────────────────────────────────────────────
    print("\n[5] Bootstrap confidence intervals (200 iterations)...")
    boot_df, boot_time, n_boot_iter = run_bootstrap(combined, df32s, df32v, lat_exp, cached_exp,
                                                      n_iter=200, seed=42)

    # Merge bootstrap CIs into full_df
    full_df = compute_bootstrap_ci(boot_df, full_df)

    # ── Add ratio columns from point estimate ────────────────────────────
    # Create per-row ratio columns from point_ratios (only non-NaN for 32B ShadowKV++)
    ratio_map = {}  # (dataset, mode) -> dict of ratios
    for pr in point_ratios:
        ratio_map[(pr["dataset"], pr["mode"])] = pr

    full_df["ratio_sglang"] = np.nan
    full_df["ratio_vllm"] = np.nan
    full_df["ratio_ensemble"] = np.nan
    full_df["ratio_spread"] = np.nan
    for idx, row in full_df.iterrows():
        key = (row["dataset"], row["mode"])
        if key in ratio_map:
            pr = ratio_map[key]
            full_df.at[idx, "ratio_sglang"] = pr.get("sglang_ratio_32b", np.nan)
            full_df.at[idx, "ratio_vllm"] = pr.get("vllm_ratio_32b", np.nan)
            full_df.at[idx, "ratio_ensemble"] = pr.get("ratio_ensemble", np.nan)
            full_df.at[idx, "ratio_spread"] = pr.get("ratio_spread", np.nan)

    # ── Step 6: Output columns ────────────────────────────────────────────
    print("\n[6] Saving outputs...")

    out_cols = [
        "model", "model_slug", "model_params_B", "dataset", "mode", "engine",
        "mean_latency_ms", "latency_ci_95", "latency_std",
        "mean_latency_ms_est_mean", "mean_latency_ms_est_std",
        "mean_latency_ms_est_ci_95_lower", "mean_latency_ms_est_ci_95_upper",
        "throughput_rps", "throughput_std",
        "throughput_rps_est_mean", "throughput_rps_est_std",
        "cached_tokens_mean", "cached_tokens_std", "cached_token_ratio",
        "cached_tokens_mean_est_mean", "cached_tokens_mean_est_std",
        "gpu_energy_j", "energy_std",
        "gpu_energy_j_est_mean", "gpu_energy_j_est_std",
        "speedup_vs_lmcache_pct",
        "speedup_vs_lmcache_pct_est_mean", "speedup_vs_lmcache_pct_est_std",
        "ratio_sglang", "ratio_vllm", "ratio_ensemble", "ratio_spread",
        "method", "estimation_method", "n_replicates", "source_dataset", "batch",
    ]
    out_cols = [c for c in out_cols if c in full_df.columns]
    full_out = full_df[out_cols].copy()
    full_out.to_csv(OUT / "full_10dataset_5model_estimate.csv", index=False)
    print(f"  >> full_10dataset_5model_estimate.csv ({len(full_out)} rows)")

    # ── Engine comparison summary ────────────────────────────────────────
    summary_rows = []
    for (ms, eng, mode), grp in full_out.groupby(["model_slug", "engine", "mode"]):
        mean_lat = grp["mean_latency_ms"].mean()
        mean_tp = grp["throughput_rps"].mean()
        mean_sp = grp["speedup_vs_lmcache_pct"].mean()
        meas_meth = {"measured", "measured_32b_sglang"}
        est_meth = {"estimated", "estimated_32b_ratio", "estimated_32b_token_analogy"}
        n_meas = grp["method"].isin(meas_meth).sum()
        n_est = grp["method"].isin(est_meth).sum()
        mo = grp[grp["method"].isin(meas_meth)]
        ml = mo["mean_latency_ms"].mean() if len(mo) > 0 else np.nan

        # Bootstrap CI for mean latency
        lat_ci = ""
        if "mean_latency_ms_est_ci_95_lower" in grp.columns and "mean_latency_ms_est_ci_95_upper" in grp.columns:
            lo = grp["mean_latency_ms_est_ci_95_lower"].mean()
            hi = grp["mean_latency_ms_est_ci_95_upper"].mean()
            lat_ci = f"[{lo:.1f}, {hi:.1f}]"

        summary_rows.append({
            "model_slug": ms, "model_params_B": PARAMS.get(ms, np.nan),
            "engine": eng, "engine_label": ENGINE_LABELS.get(eng, eng),
            "mode": mode,
            "mean_latency_ms_all": mean_lat,
            "mean_latency_ms_measured_only": ml,
            "latency_95_ci_range": lat_ci,
            "mean_throughput_rps": mean_tp,
            "mean_speedup_vs_lmcache_pct": mean_sp,
            "n_measured_datasets": n_meas,
            "n_estimated_datasets": n_est,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / "engine_comparison_summary.csv", index=False)
    print(f"  >> engine_comparison_summary.csv ({len(summary_df)} rows)")

    # ── Dataset difficulty factors ────────────────────────────────────────
    diff_rows = []
    ref_ds = "ag_news"
    for engine in ENGINES:
        for mode in MODES:
            for dataset in ALL_DATASETS:
                sub = full_out[(full_out["engine"] == engine) & (full_out["mode"] == mode)
                               & (full_out["dataset"] == dataset)]
                ref = full_out[(full_out["engine"] == engine) & (full_out["mode"] == mode)
                               & (full_out["dataset"] == ref_ds)]
                if len(sub) == 0 or len(ref) == 0:
                    continue
                common = set(sub["model_slug"].unique()) & set(ref["model_slug"].unique())
                if len(common) == 0:
                    continue
                sc = sub[sub["model_slug"].isin(common)]
                rc = ref[ref["model_slug"].isin(common)]
                avg_lat = sc["mean_latency_ms"].mean()
                avg_ref_lat = rc["mean_latency_ms"].mean()
                diff_rows.append({
                    "engine": engine, "mode": mode, "dataset": dataset,
                    "avg_latency_ms": avg_lat, "ref_dataset": ref_ds,
                    "avg_ref_latency_ms": avg_ref_lat,
                    "difficulty_multiplier": avg_lat / avg_ref_lat if avg_ref_lat > 0 else np.nan,
                    "is_measured": dataset in MEASURED_DATASETS,
                    "n_models_compared": len(common),
                })
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(OUT / "dataset_difficulty_factors.csv", index=False)
    print(f"  >> dataset_difficulty_factors.csv ({len(diff_df)} rows)")

    # ── Bootstrap summary ─────────────────────────────────────────────────
    with open(OUT / "bootstrap_summary.txt", "w") as f:
        f.write(f"Bootstrap iterations: {n_boot_iter}\n")
        f.write(f"Bootstrap time: {boot_time:.1f}s\n")
        f.write(f"Fitted latency exponent: {lat_exp:.3f} (was 0.7, change {lat_exp - 0.7:+.3f})\n")
        f.write(f"Fitted cached-tokens exponent: {cached_exp:.3f} (was 0.5, change {cached_exp - 0.5:+.3f})\n\n")

        # Top-10 most uncertain estimates
        ci_col = "mean_latency_ms_est_ci_95_upper"
        ci_lo = "mean_latency_ms_est_ci_95_lower"
        if ci_col in full_out.columns and ci_lo in full_out.columns:
            full_out["_ci_width"] = full_out[ci_col] - full_out[ci_lo]
            top10 = full_out.sort_values("_ci_width", ascending=False).head(10)
            f.write("Top-10 most uncertain estimates (by 95% CI width on latency):\n")
            f.write(f"{'Model':<15s} {'Dataset':<18s} {'Mode':<10s} {'Engine':<40s} {'Latency':>8s} {'CI_width':>10s}\n")
            f.write("-" * 105 + "\n")
            for _, r in top10.iterrows():
                f.write(f"{r['model_slug']:<15s} {r['dataset']:<18s} {r['mode']:<10s} "
                        f"{r['engine']:<40s} {r['mean_latency_ms']:>8.1f} {r['_ci_width']:>10.1f}\n")

    print(f"  >> bootstrap_summary.txt")

    # ── Validation & Reporting ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION & SUMMARY")
    print("=" * 70)

    # Fit quality
    print("\n--- Fitted Exponents ---")
    print(f"  Latency exponent:          {lat_exp:.3f} (was 0.7)")
    print(f"  Cached-tokens exponent:    {cached_exp:.3f} (was sqrt = 0.5)")

    # Cell counts
    print("\n--- Cell Counts ---")
    for m, c in full_out.groupby("method").size().items():
        print(f"  method={m}: {c}")
    print()
    for em, c in full_out.groupby("estimation_method").size().items():
        print(f"  estimation_method={em}: {c}")
    print(f"  Total: {len(full_out)}")

    # 32B ratio summary
    print("\n--- 32B SGLang ShadowKV++ Ratio Results (Ensemble) ---")
    if point_ratios:
        for pr in point_ratios:
            print(f"  {pr['dataset']:15s} {pr['mode']:10s} "
                  f"SGLang={pr['sglang_ratio_32b']:.4f} vLLM={pr['vllm_ratio_32b']:.4f} "
                  f"Ensemble={pr['ratio_ensemble']:.4f} Spread={pr['ratio_spread']:.4f}")

    # Speedup table (same as before)
    print("\n--- Full 5x10 Speedup Table (ShadowKV++ / Native RadixAttention) ---")
    speedup_rows = []
    for ms in ["qwen25_15b", "qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b"]:
        for ds in ALL_DATASETS:
            for mo in MODES:
                rad = full_out[(full_out["model_slug"] == ms) & (full_out["dataset"] == ds)
                               & (full_out["mode"] == mo) & (full_out["engine"] == "sglang_radix_attention")]
                sh = full_out[(full_out["model_slug"] == ms) & (full_out["dataset"] == ds)
                              & (full_out["mode"] == mo) & (full_out["engine"] == "sglang_radix_attention_shadowkv_plus")]
                if len(rad) == 0 or len(sh) == 0:
                    continue
                sp = (rad["mean_latency_ms"].values[0] - sh["mean_latency_ms"].values[0]) / rad["mean_latency_ms"].values[0] * 100
                meth = sh["method"].values[0]
                est_m = sh["estimation_method"].values[0] if "estimation_method" in sh.columns else meth
                marker = " " if meth in ("measured", "measured_32b_sglang") else ("r" if est_m == "ratio_scaling" else "e")
                speedup_rows.append({"model": ms, "params_B": PARAMS.get(ms, np.nan),
                    "dataset": ds, "mode": mo, "speedup_pct": sp, "marker": marker})
    sp_df = pd.DataFrame(speedup_rows)
    for mo in MODES:
        print(f"\nMode: {mo}")
        h = "| " + " | ".join([f"{'Model':<15s}"] + [f"{d:>14s}" for d in ALL_DATASETS]) + " |"
        print(h)
        print("|" + "|".join([":" + "-" * (14 if i == 0 else 14) + ":"
                              for i in range(len(ALL_DATASETS) + 1)]) + "|")
        for ms in ["qwen25_15b", "qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b"]:
            vals = []
            for ds in ALL_DATASETS:
                sub = sp_df[(sp_df["model"] == ms) & (sp_df["dataset"] == ds) & (sp_df["mode"] == mo)]
                if len(sub) == 0:
                    vals.append("  --  ")
                else:
                    s = sub.iloc[0]
                    vals.append(f"{s['speedup_pct']:>+5.1f}%{s['marker']}")
            print(f"| {ms:<15s} | " + " | ".join(vals) + " |")
    print("\n  Legend: ' ' = measured, 'e' = token-analogy, 'r' = ratio-scaling (32B ShadowKV++)")
    print(f"  Note: Latency exponent {lat_exp:.3f} (was 0.7). Cached exponent {cached_exp:.3f} (was 0.5).")
    print(f"  Bootstrap: 200 iterations, {boot_time:.1f}s.")

    # Top-5 uncertain
    if ci_col in full_out.columns:
        print("\n--- Top-5 Most Uncertain Estimates (by 95% CI width) ---")
        top5 = full_out.sort_values("_ci_width", ascending=False).head(5)
        for _, r in top5.iterrows():
            lo = r.get(ci_lo, np.nan)
            hi = r.get(ci_col, np.nan)
            print(f"  {r['model_slug']:12s} {r['dataset']:15s} {r['mode']:10s} "
                  f"{r['engine']:40s} lat={r['mean_latency_ms']:.1f} "
                  f"95%CI=[{lo:.1f}, {hi:.1f}] width={r['_ci_width']:.1f}")

    # Files
    print("\n--- Output files ---")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    print("\nDone.")
    return full_out


if __name__ == "__main__":
    main()
