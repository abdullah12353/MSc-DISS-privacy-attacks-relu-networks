#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
x3_aggregate.py — Build tidy, plot-ready master tables from X.3 trainer outputs.

This script ingests per-run artifacts written by x3_train.py and produces the
"master/*.csv" files that the plotting step expects. It is robust to partial
runs (e.g., no post-100% data), safe-divides, and missing optional files.

INPUT LAYOUT (under <OUT>/array_<JID>/):
  per_run/
    seed_<seed>_reg_<REG>/
      per_epoch_all.csv
      per_epoch_post100.csv          # optional: only if 100% reached
      slow_candidates_buffer.csv     # rolling last-K slow-eval candidates
      candidates_lastK.csv           # optional convenience for last-K
      per_seed_meta.csv
      model_config.json / model_final.pt
      snap_seed=<S>_reg=<REG>.png
  master/
    precision_series_alpha0.csv       # precision@α0 vs τ
    endpoint_precision_recall.csv     # endpoint precision/recall with counts
    alpha_curve_endpoint.csv          # per-seed precision vs α curve (endpoint)
    aupa030_endpoint.csv              # per-seed AUP@0.3 (normalized)
    ecdf_endpoint.csv                 # per-candidate z=d/g (endpoint), tidy
    e_grid_timeseries.csv             # τ series: cand_count, dup_ratio, coverage, HHI, top1_share
    time_to_099_alpha0.csv            # per-seed τ* to sustain ≥0.99 precision@α0 (if achievable)
    time_to_full_coverage.csv         # per-seed τ* to reach coverage==1 (if achievable)
    A_table.csv                       # log(mean_d_support) ~ a + b τ fits per seed

USAGE:
  python x3_aggregate.py --out-root <OUT> [--alpha0 0.20] [--alpha-grid 0.10,0.15,0.20,0.25,0.30]

Notes:
- Uses SVM-style margin during training only to set candidates; aggregation does not
  recompute margins. It reads logged counts and candidate-level distances.
- τ ("tau") is epochs since the first time accuracy reached 100% (t*). If post-100%
  CSV is missing, we derive τ≥0 by filtering per_epoch_all with its tau column.
- Safe-divides: precision is NaN when cand_count==0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Avoid glob shadowing bugs:
import glob as globmod

# ----------------------------- helpers -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def list_run_dirs(out_root: str) -> List[str]:
    per_run = os.path.join(out_root, "per_run")
    if not os.path.isdir(per_run):
        return []
    return sorted([p for p in globmod.glob(os.path.join(per_run, "seed_*_reg_*")) if os.path.isdir(p)])

def read_csv_coerce(path: str, **kwargs) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"[WARN] failed reading: {path} ({e})")
        return None

def safe_div(num, den):
    try:
        num = float(num)
        den = float(den)
        return np.nan if den == 0 else num / den
    except Exception:
        return np.nan

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% CI for binomial proportion (k successes, n trials)."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return (center - half, center + half)

def load_meta(run_dir: str) -> Dict:
    meta = read_csv_coerce(os.path.join(run_dir, "per_seed_meta.csv"))
    if meta is None or meta.empty:
        return {}
    # keep first row as dict
    return {k: meta.iloc[0][k] for k in meta.columns if k in meta.columns}

# ----------------------------- builders -----------------------------

def build_precision_series_alpha0(run_dirs: List[str], alpha0: float, out_csv: str) -> None:
    rows = []
    for rd in run_dirs:
        pea = read_csv_coerce(os.path.join(rd, "per_epoch_all.csv"))
        if pea is None or pea.empty:
            continue
        regime = None
        seed = None
        if "regime" in pea.columns:
            regime = str(pea["regime"].iloc[0])
        if "seed_id" in pea.columns:
            seed = int(pea["seed_id"].iloc[0])
        # Expect hits_a020 etc.; recompute precision safely
        if "hits_a020" in pea.columns and "cand_count" in pea.columns:
            df = pea.copy()
            if "tau" not in df.columns:
                # Try to derive tau from per_epoch_post100 if present; else mark NaN
                df["tau"] = np.nan
            df = df[["tau", "hits_a020", "cand_count"]].copy()
            df["precision_a020"] = df.apply(lambda r: safe_div(r["hits_a020"], r["cand_count"]), axis=1)
            for _, r in df.iterrows():
                rows.append({"regime": regime, "seed_id": seed, "tau": r["tau"], "precision_a020": r["precision_a020"]})
        else:
            print(f"[WARN] missing hits_a020/cand_count in {rd}/per_epoch_all.csv; skipping precision series.")
    if rows:
        pd.DataFrame(rows, columns=["regime","seed_id","tau","precision_a020"]).to_csv(out_csv, index=False)
        print(f"[OK] wrote {out_csv} ({len(rows)} rows)")
    else:
        print(f"[WARN] no precision series rows; did not write {out_csv}")

def build_e_grid_timeseries(out_root: str) -> None:
    """Combine per-run per_epoch_all.csv into τ-aligned e-grid series."""
    run_dirs = list_run_dirs(out_root)
    out_csv = os.path.join(out_root, "master", "e_grid_timeseries.csv")
    rows = []
    for rd in run_dirs:
        pea = read_csv_coerce(os.path.join(rd, "per_epoch_all.csv"))
        if pea is None or pea.empty:
            continue
        regime = pea.get("regime", pd.Series(["UNK"])).iloc[0]
        seed = int(pea.get("seed_id", pd.Series([-1])).iloc[0])
        cols_needed = ["tau","epoch","cand_count","dup_ratio","coverage","unique_supports_hit","HHI","top1_share","mean_d_support"]
        df = pea[[c for c in cols_needed if c in pea.columns]].copy()
        # Backfill missing fields if needed
        for c in cols_needed:
            if c not in df.columns:
                df[c] = np.nan
        for _, r in df.iterrows():
            rows.append({
                "regime": regime, "seed_id": seed,
                "tau": r["tau"], "epoch": r["epoch"],
                "cand_count": r["cand_count"], "dup_ratio": r["dup_ratio"],
                "coverage": r["coverage"], "unique_supports_hit": r["unique_supports_hit"],
                "HHI": r["HHI"], "top1_share": r["top1_share"],
                "mean_d_support": r["mean_d_support"]
            })
    if rows:
        pd.DataFrame(rows, columns=[
            "regime","seed_id","tau","epoch",
            "cand_count","dup_ratio","coverage","unique_supports_hit","HHI","top1_share","mean_d_support"
        ]).to_csv(out_csv, index=False)
        print(f"[OK] wrote {out_csv} ({len(rows)} rows)")
    else:
        print(f"[WARN] no e-grid rows; did not write {out_csv}")

def build_time_to_099(run_dirs: List[str], alpha0: float, out_csv: str) -> None:
    """Compute τ* to reach and sustain precision@α0 ≥ 0.99 per seed (Kaplan–Meier-like utility)."""
    rows = []
    for rd in run_dirs:
        pea = read_csv_coerce(os.path.join(rd, "per_epoch_all.csv"))
        if pea is None or pea.empty or "precision_a020" not in pea.columns:
            # Recompute if needed
            if pea is None or pea.empty or "hits_a020" not in pea.columns or "cand_count" not in pea.columns:
                continue
            df = pea[["tau","hits_a020","cand_count"]].copy()
            df["precision_a020"] = df.apply(lambda r: safe_div(r["hits_a020"], r["cand_count"]), axis=1)
        else:
            df = pea[["tau","precision_a020"]].copy()
        df = df.dropna(subset=["tau"]).sort_values("tau")
        if df.empty:
            continue
        regime = pea.get("regime", pd.Series(["UNK"])).iloc[0]
        seed = int(pea.get("seed_id", pd.Series([-1])).iloc[0])

        # First τ where precision≥0.99 and remains ≥0.99 for the rest
        tau_star = np.nan
        vals = df["precision_a020"].values
        taus = df["tau"].values
        for i in range(len(vals)):
            if np.isfinite(vals[i]) and vals[i] >= 0.99:
                if np.all(vals[i:] >= 0.99):
                    tau_star = taus[i]
                    break
        rows.append({"regime": regime, "seed_id": seed, "tau_099": tau_star})
    if rows:
        pd.DataFrame(rows, columns=["regime","seed_id","tau_099"]).to_csv(out_csv, index=False)
        print(f"[OK] wrote {out_csv} ({len(rows)} rows)")
    else:
        print(f"[WARN] no time-to-0.99 rows; did not write {out_csv}")

def build_A_table(out_root: str) -> None:
    """Fit log(mean_d_support) = a + b τ per seed (robust linear fit)."""
    egrid = read_csv_coerce(os.path.join(out_root, "master", "e_grid_timeseries.csv"))
    if egrid is None or egrid.empty:
        print("[WARN] cannot build A_table (no e_grid_timeseries).")
        return
    rows = []
    for (regime, seed), g in egrid.groupby(["regime","seed_id"]):
        g = g.dropna(subset=["tau","mean_d_support"])
        g = g[g["mean_d_support"] > 0]
        if g.empty:
            rows.append({"regime": regime,"seed_id": seed,"a": np.nan,"b": np.nan,"r2": np.nan,"n": 0})
            continue
        x = g["tau"].values.astype(float)
        y = np.log(g["mean_d_support"].values.astype(float))
        # linear fit y = a + b x
        A = np.vstack([np.ones_like(x), x]).T
        try:
            coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b = coeff[0], coeff[1]
            yhat = A @ coeff
            ss_res = np.sum((y - yhat)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
            rows.append({"regime": regime,"seed_id": seed,"a": a,"b": b,"r2": r2,"n": int(len(x))})
        except Exception as e:
            print(f"[WARN] fit failed for regime={regime}, seed={seed}: {e}")
            rows.append({"regime": regime,"seed_id": seed,"a": np.nan,"b": np.nan,"r2": np.nan,"n": int(len(x))})
    out_csv = os.path.join(out_root, "master", "A_table.csv")
    pd.DataFrame(rows, columns=["regime","seed_id","a","b","r2","n"]).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} ({len(rows)} rows)")

def build_endpoint_stats(per_run_dirs: List[str], alpha0: float,
                         out_prec_recall_csv: str,
                         out_alpha_curve_csv: str,
                         out_aupa_csv: str,
                         out_ecdf_csv: str,
                         alpha_grid: List[float]) -> None:
    """
    From candidates_lastK.csv build:
      - per-seed precision@α0 and recall@α0 (with counts, Wilson CI)
      - per-seed precision vs α curve on the chosen grid
      - per-seed AUP@0.3 (normalized)
      - tidy ECDF z-list (z = d_support / g_local)
    """
    prec_rows, curve_rows, aupa_rows, ecdf_rows = [], [], [], []
    for rd in per_run_dirs:
        meta = load_meta(rd)
        regime = meta.get("regime", "UNK")
        seed = int(meta.get("seed_id", -1))
        supp_count = int(meta.get("support_count", meta.get("n_supports", -1)))

        # Prefer candidates_lastK.csv; else derive from slow_candidates_buffer.csv filtering last K by epoch if present
        lastk_path = os.path.join(rd, "candidates_lastK.csv")
        df = read_csv_coerce(lastk_path)
        if df is None or df.empty:
            df = read_csv_coerce(os.path.join(rd, "slow_candidates_buffer.csv"))
            if df is None or df.empty:
                print(f"[WARN] no endpoint candidates for {rd}; skipping.")
                continue

        # Expect columns: d_support and g_local → z = d/g
        if "d_support" not in df.columns or "g_local" not in df.columns:
            print(f"[WARN] missing d_support/g_local in {rd}; skipping.")
            continue

        # Clean & compute z; keep only post-100% slow evals if tau present and >=0
        df = df.copy()
        if "tau" in df.columns:
            df = df[df["tau"].astype(float).fillna(np.inf) >= 0]

        # hits at alpha0 and totals for precision
        df["z"] = df["d_support"].astype(float) / df["g_local"].astype(float)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["z"])
        H = int((df["z"] <= alpha0).sum())
        M = int(len(df))
        precision = safe_div(H, M)

        # recall via supports covered by at least one candidate within α0
        covered_supports = len(set(df.loc[df["z"] <= alpha0, "support_index"].astype(int).tolist())) if "support_index" in df.columns else np.nan
        recall = np.nan if supp_count <= 0 or not np.isfinite(covered_supports) else covered_supports / supp_count
        lo, hi = wilson_ci(H, M) if M > 0 else (np.nan, np.nan)

        prec_rows.append({
            "regime": regime, "seed_id": seed,
            "alpha": alpha0, "precision": precision, "hits": H, "cands": M,
            "precision_wilson_lo": lo, "precision_wilson_hi": hi,
            "recall": recall, "supports_hit": covered_supports, "support_count": supp_count
        })

        # Precision–alpha curve & AUP@0.3
        alphas_sorted = sorted(alpha_grid)
        for a in alphas_sorted:
            k = int((df["z"] <= a).sum())
            n = int(len(df))
            p = safe_div(k, n)
            curve_rows.append({"regime": regime, "seed_id": seed, "alpha": a, "precision": p, "hits": k, "cands": n})
        # Normalize AUP@0.3 by max α (0.3)
        xs = np.array(alphas_sorted, dtype=float)
        ys = np.array([safe_div(int((df["z"] <= a).sum()), len(df)) for a in xs], dtype=float)
        # Trapezoidal rule up to 0.3; normalize by 0.3 to keep in [0,1]
        try:
            mask = xs <= 0.30 + 1e-12
            area = np.trapz(ys[mask], xs[mask]) / 0.30
        except Exception:
            area = np.nan
        aupa_rows.append({"regime": regime, "seed_id": seed, "AUP_at_0p30": area})

        # ECDF rows (tidy): each row is one z
        ecdf_rows += [{"regime": regime, "seed_id": seed, "z": z} for z in df["z"].tolist()]

    # ---------- WRITE FILES ----------
    # Option A: include BOTH the original columns and the plotter's alias columns
    df_prec = pd.DataFrame(prec_rows, columns=[
        "regime","seed_id","alpha","precision","hits","cands",
        "precision_wilson_lo","precision_wilson_hi",
        "recall","supports_hit","support_count"
    ])
    # Aliases expected by plotter
    df_prec["prec_hits"]   = df_prec["hits"]
    df_prec["prec_cands"]  = df_prec["cands"]
    df_prec["recall_hits"] = df_prec["supports_hit"]
    df_prec["n_supports"]  = df_prec["support_count"]

    # Friendly column order (both styles)
    cols_order = [
        "regime","seed_id","alpha",
        "precision","hits","cands","prec_hits","prec_cands",
        "precision_wilson_lo","precision_wilson_hi",
        "recall","supports_hit","support_count","recall_hits","n_supports"
    ]
    df_prec = df_prec[[c for c in cols_order if c in df_prec.columns]]
    df_prec.to_csv(out_prec_recall_csv, index=False)

    pd.DataFrame(curve_rows, columns=["regime","seed_id","alpha","precision","hits","cands"]).to_csv(out_alpha_curve_csv, index=False)
    pd.DataFrame(aupa_rows, columns=["regime","seed_id","AUP_at_0p30"]).to_csv(out_aupa_csv, index=False)
    pd.DataFrame(ecdf_rows, columns=["regime","seed_id","z"]).to_csv(out_ecdf_csv, index=False)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="Path to array_<JID> (contains per_run/)")
    ap.add_argument("--alpha0", type=float, default=0.20, help="Primary alpha for precision/recall")
    ap.add_argument("--alpha-grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30", help="Comma list for endpoint curves")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    master_root = os.path.join(out_root, "master")
    ensure_dir(master_root)
    ensure_dir(os.path.join(master_root, "figs"))

    # Gather per-run dirs
    run_dirs = list_run_dirs(out_root)
    if not run_dirs:
        print(f"[WARN] No per_run dirs under {out_root}")
    alpha0 = float(args.alpha0)
    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]

    # 1) Precision series (τ)
    build_precision_series_alpha0(run_dirs, alpha0, out_csv=os.path.join(master_root, "precision_series_alpha0.csv"))

    # 2) Endpoint stats (precision/recall@α0 with counts + curves/AUP/ECDF)
    build_endpoint_stats(
        per_run_dirs=run_dirs,
        alpha0=alpha0,
        out_prec_recall_csv=os.path.join(master_root, "endpoint_precision_recall.csv"),
        out_alpha_curve_csv=os.path.join(master_root, "alpha_curve_endpoint.csv"),
        out_aupa_csv=os.path.join(master_root, "aupa030_endpoint.csv"),
        out_ecdf_csv=os.path.join(master_root, "ecdf_endpoint.csv"),
        alpha_grid=alpha_grid,
    )

    # 3) A-table (log mean distance fits)
    build_e_grid_timeseries(out_root)
    build_A_table(out_root)

    # 4) Time-to-0.99 precision @ alpha0 (sustained)
    build_time_to_099(run_dirs, alpha0, out_csv=os.path.join(master_root, "time_to_099_alpha0.csv"))

    print("[DONE] master tables written under:", master_root)

if __name__ == "__main__":
    main()

