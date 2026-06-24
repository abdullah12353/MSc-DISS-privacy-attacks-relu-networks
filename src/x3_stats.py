#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, argparse, glob
import numpy as np
import pandas as pd

# ------------------ small utils ------------------

def _num(s):
    try: return pd.to_numeric(s, errors="coerce")
    except: return pd.Series([np.nan]*len(s))

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    k = float(min(max(k, 0.0), n)); n = float(n)
    p = k / n
    denom = 1.0 + (z**2)/n
    rad = (p*(1-p) + (z**2)/(4*n)) / n
    rad = max(rad, 0.0)
    center = (p + (z**2)/(2*n)) / denom
    half   = z * math.sqrt(rad) / denom
    return (center - half, center + half)

def onesided_t_norm_p_greater(xs, mu0):
    xs = np.asarray([x for x in xs if np.isfinite(x)])
    n = xs.size
    if n == 0: return np.nan, np.nan, 0
    m = xs.mean(); s = xs.std(ddof=1) if n > 1 else 0.0
    if n == 1 or s == 0:
        return (0.0 if m > mu0 else 1.0), float("inf") if s==0 else (m-mu0)/(s/np.sqrt(n)), n
    z = (m - mu0) / (s / math.sqrt(n))
    p = 0.5 * math.erfc(-z / math.sqrt(2.0))
    return p, z, n

def onesided_t_norm_p_less(xs, mu0):
    xs = np.asarray([x for x in xs if np.isfinite(x)])
    n = xs.size
    if n == 0: return np.nan, np.nan, 0
    m = xs.mean(); s = xs.std(ddof=1) if n > 1 else 0.0
    if n == 1 or s == 0:
        return (0.0 if m < mu0 else 1.0), float("-inf") if s==0 else (m-mu0)/(s/np.sqrt(n)), n
    z = (m - mu0) / (s / math.sqrt(n))
    p = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return p, z, n

def binom_sign_test_p(r, S):
    if S <= 0: return np.nan
    from math import comb
    return sum(comb(S,k) for k in range(r, S+1)) / (2.0**S)

def iqr(x):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if x.size == 0: return (np.nan, np.nan)
    return (float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75)))

def safe_div(a, b):
    a = float(a); b = float(b)
    return np.nan if b == 0 else a / b

# ------------------ data access ------------------

def per_run_dirs(out_root):
    d = os.path.join(out_root, "per_run")
    for p in glob.glob(os.path.join(d, "seed_*_reg_*")):
        base = os.path.basename(p)
        try:
            seed = int(base.split("_")[1]); reg  = base.split("_")[3].upper()
        except Exception:
            continue
        yield reg, seed, p

def read_csv(path):
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()

def load_meta(out_root):
    meta = {}
    for reg, seed, d in per_run_dirs(out_root):
        m = read_csv(os.path.join(d, "per_seed_meta.csv"))
        if m.empty:
            meta[(reg,seed)] = dict(bar_g=np.nan, n=np.nan, support_count=np.nan)
            continue
        def first(col, default=np.nan):
            return (m[col].iloc[0] if col in m.columns and pd.notna(m[col].iloc[0]) else default)
        g  = float(first("bar_g"))
        n  = float(first("n", default=np.nan))
        sc = first("support_count", default=first("n_supports", default=np.nan))
        sc = float(sc) if pd.notna(sc) else np.nan
        meta[(reg,seed)] = dict(bar_g=g, n=n, support_count=sc)
    return meta

# ------------------ per-seed computations ------------------

def per_seed_endpoint_counts(df_epoch, alpha0=0.20, lastK=5):
    cols_hits = {0.05:"hits_a005", 0.10:"hits_a010", 0.20:"hits_a020", 0.30:"hits_a030"}
    hk = cols_hits.get(round(alpha0,2), "hits_a020")
    df = df_epoch.copy()
    for c in [hk, "cand_count", "tau", "dup_ratio", "HHI", "unique_supports_hit", "coverage"]:
        if c not in df.columns: df[c] = np.nan
    df = df[_num(df["tau"]).fillna(np.inf) >= 0]
    if df.empty:
        return dict(hits=0.0, cands=0.0, p_series=[], dup_med=np.nan, hhi_med=np.nan,
                    unique_hits=np.nan, coverage_med=np.nan)
    slow_like = df[[c for c in cols_hits.values() if c in df.columns]].fillna(0).sum(axis=1) > 0
    df = df[ slow_like | (_num(df["cand_count"]) > 0) ].sort_values("epoch")
    if df.empty:
        return dict(hits=0.0, cands=0.0, p_series=[], dup_med=np.nan, hhi_med=np.nan,
                    unique_hits=np.nan, coverage_med=np.nan)
    wnd = df.tail(lastK).copy()
    H = float(_num(wnd[hk]).sum()); M = float(_num(wnd["cand_count"]).sum())
    p_series = []
    for _, r in wnd.iterrows():
        cands = float(r.get("cand_count", np.nan)); hits  = float(r.get(hk, 0.0))
        p_series.append((hits / cands) if cands > 0 else np.nan)
    dup_med = float(np.nanmedian(_num(wnd["dup_ratio"]).values)) if "dup_ratio" in wnd.columns else np.nan
    hhi_med = float(np.nanmedian(_num(wnd["HHI"]).values)) if "HHI" in wnd.columns else np.nan
    unique_hits = float(np.nanmax(_num(wnd["unique_supports_hit"]).values)) if "unique_supports_hit" in wnd.columns else np.nan
    coverage_med = float(np.nanmedian(_num(wnd["coverage"]).values)) if "coverage" in wnd.columns else np.nan
    return dict(hits=H, cands=M, p_series=p_series, dup_med=dup_med, hhi_med=hhi_med,
                unique_hits=unique_hits, coverage_med=coverage_med)

def per_seed_q95_endpoint(d_candidates):
    if d_candidates.empty or not {"d_support","g_local"}.issubset(d_candidates.columns): return np.nan
    z = (_num(d_candidates["d_support"]) / _num(d_candidates["g_local"])).replace([np.inf,-np.inf], np.nan).dropna().values
    return float(np.quantile(z, 0.95)) if z.size else np.nan

def per_seed_alpha_curve(d_candidates, alpha_grid):
    if d_candidates.empty or not {"d_support","g_local"}.issubset(d_candidates.columns):
        return np.full_like(alpha_grid, np.nan, dtype=float), 0
    z = (_num(d_candidates["d_support"]) / _num(d_candidates["g_local"])).replace([np.inf,-np.inf], np.nan).dropna().values
    if z.size == 0: return np.full_like(alpha_grid, np.nan, dtype=float), 0
    z = np.sort(z)
    idx = np.searchsorted(z, alpha_grid, side="right")
    pvals = idx / z.size
    return pvals.astype(float), int(z.size)

def per_seed_logdecay(df_epoch, bar_g):
    if df_epoch.empty or "tau" not in df_epoch.columns: return (np.nan, np.nan, np.nan)
    df = df_epoch.copy(); df = df[_num(df["tau"]).fillna(np.inf) >= 0]
    if df.empty or "mean_d_support" not in df.columns: return (np.nan, np.nan, np.nan)
    y = _num(df["mean_d_support"]).values
    dnorm = (y / float(bar_g)) if (np.isfinite(bar_g) and bar_g > 0) else y
    mask = (dnorm > 0) & np.isfinite(dnorm) & np.isfinite(_num(df["tau"]).values)
    if not mask.any(): return (np.nan, np.nan, np.nan)
    x = _num(df.loc[mask, "tau"]).values.astype(float); Y = np.log(dnorm[mask].astype(float))
    if x.size < 2: return (np.nan, np.nan, np.nan)
    A = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    Yhat = a + b*x
    ss_res = float(np.sum((Y - Yhat)**2)); ss_tot = float(np.sum((Y - np.mean(Y))**2))
    R2 = 1.0 - (ss_res/ss_tot if ss_tot>0 else 0.0)
    half = (math.log(2.0) / abs(b)) if (np.isfinite(b) and b != 0.0) else np.nan
    return (b, R2, half)

def per_seed_hitting_times(out_root, reg, seed, eps_list=(0.30,0.20,0.15,0.10)):
    ddir = next((d for r,s,d in per_run_dirs(out_root) if r==reg and s==seed), None)
    if ddir is None:
        return {f"tau_hit_q95_{e:.2f}": np.nan for e in eps_list}
    cand = read_csv(os.path.join(ddir, "slow_candidates_buffer.csv"))
    if cand.empty: cand = read_csv(os.path.join(ddir, "candidates_lastK.csv"))
    if cand.empty or "epoch" not in cand.columns or not {"d_support","g_local"}.issubset(cand.columns):
        return {f"tau_hit_q95_{e:.2f}": np.nan for e in eps_list}
    pea = read_csv(os.path.join(ddir, "per_epoch_all.csv"))
    if pea.empty or "epoch" not in pea.columns: 
        return {f"tau_hit_q95_{e:.2f}": np.nan for e in eps_list}
    ep2tau = pea.set_index("epoch")["tau"] if "tau" in pea.columns else pd.Series(dtype=float)
    cand = cand.dropna(subset=["epoch"])
    cand["z"] = (_num(cand["d_support"]) / _num(cand["g_local"])).replace([np.inf,-np.inf], np.nan)
    q = cand.dropna(subset=["z"]).groupby("epoch")["z"].quantile(0.95)
    hits = {}
    for eps in eps_list:
        hit_epoch = q.index[q.values <= eps]
        if len(hit_epoch) == 0:
            hits[f"tau_hit_q95_{eps:.2f}"] = np.nan
        else:
            e = float(hit_epoch[0])
            tau = float(ep2tau.get(e, np.nan)) if not ep2tau.empty else np.nan
            hits[f"tau_hit_q95_{eps:.2f}"] = tau
    return hits

# ------------------ main aggregation ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--alpha0", type=float, default=0.20)
    ap.add_argument("--lastK", type=int, default=5)
    ap.add_argument("--alpha-grid", type=str, default="0.000:0.001:0.300")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    stats_dir = _ensure_dir(os.path.join(out_root, "master", "stats"))

    try:
        a0, step, a1 = [float(x) for x in args.alpha_grid.split(":")]
    except Exception:
        a0, step, a1 = 0.0, 0.001, 0.30
    alpha_grid = np.round(np.arange(a0, a1 + step/2.0, step), 6)

    meta = load_meta(out_root)

    # -------- per-seed rows --------
    per_seed_rows = []
    for reg, seed, d in per_run_dirs(out_root):
        pea = read_csv(os.path.join(d, "per_epoch_all.csv"))
        end_counts = per_seed_endpoint_counts(pea, alpha0=args.alpha0, lastK=args.lastK)

        H = end_counts["hits"]; M = end_counts["cands"]
        p = safe_div(H, M); plo, phi = wilson_ci(H, M)

        sup_total = meta.get((reg,seed), {}).get("support_count", np.nan)
        unique_hits = end_counts["unique_hits"]
        if np.isfinite(sup_total) and sup_total > 0 and np.isfinite(unique_hits):
            R = int(round(min(unique_hits, sup_total)))
            q = R / sup_total; qlo, qhi = wilson_ci(R, sup_total)
        else:
            cov = end_counts["coverage_med"]; q = cov if np.isfinite(cov) else np.nan; qlo = qhi = np.nan

        cand = read_csv(os.path.join(d, "candidates_lastK.csv"))
        if cand.empty: cand = read_csv(os.path.join(d, "slow_candidates_buffer.csv"))
        q95 = per_seed_q95_endpoint(cand)

        prec_curve, n_z = per_seed_alpha_curve(cand, alpha_grid)
        if n_z > 0 and np.isfinite(prec_curve).any():
            valid = np.isfinite(prec_curve); x = alpha_grid[valid]; y = prec_curve[valid]
            if x.size >= 2: area = np.trapz(y, x); au_p03 = area / (x[-1] - x[0]) if (x[-1] > x[0]) else np.nan
            else: au_p03 = np.nan
            ge = np.where(y >= 0.99)[0]; alpha_star = float(x[ge[0]]) if ge.size > 0 else np.nan
        else:
            au_p03 = np.nan; alpha_star = np.nan

        g = meta.get((reg,seed), {}).get("bar_g", np.nan)
        b, r2, t12 = per_seed_logdecay(pea, bar_g=g)

        p_series = [v for v in end_counts["p_series"] if np.isfinite(v)]
        p_sd = float(np.std(p_series, ddof=1)) if len(p_series) >= 2 else (0.0 if len(p_series)==1 else np.nan)

        dup_med = end_counts["dup_med"]; hhi_med = end_counts["hhi_med"]
        dedup_minus_supports = (unique_hits - sup_total) if (np.isfinite(unique_hits) and np.isfinite(sup_total)) else np.nan

        hits_tau = per_seed_hitting_times(out_root, reg, seed, eps_list=(0.30,0.20,0.15,0.10))

        per_seed_rows.append({
            "regime": reg, "seed_id": int(seed),
            "precision_a02": p, "precision_low95": plo, "precision_high95": phi,
            "hits": H, "M_cands": M,
            "recall": q, "recall_low95": qlo, "recall_high95": qhi,
            "K_supports": sup_total, "R_recalled": unique_hits,
            "q95_normdist": q95, "nAUP_0_0.3": au_p03, "alpha_star_099": alpha_star,
            "logdist_slope_b": b, "logdist_r2": r2, "half_life_epochs": t12,
            "dup_ratio_med": dup_med, "HHI_med": hhi_med, "dedup_minus_supports": dedup_minus_supports,
            "sd_precision_lastK": p_sd, **hits_tau
        })

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed_path = os.path.join(stats_dir, "per_seed.csv")
    per_seed.to_csv(per_seed_path, index=False, float_format="%.6f")

    # -------- summaries & tests --------
    def summarize(df):
        out = {}
        for col in ["precision_a02","recall","q95_normdist","nAUP_0_0.3","alpha_star_099",
                    "logdist_slope_b","logdist_r2","half_life_epochs","dup_ratio_med","HHI_med",
                    "sd_precision_lastK","dedup_minus_supports"]:
            x = _num(df[col]) if col in df.columns else pd.Series(dtype=float)
            x = x[np.isfinite(x)]
            if x.empty:
                out[col+"_mean"] = out[col+"_median"] = out[col+"_q25"] = out[col+"_q75"] = np.nan
            else:
                out[col+"_mean"]   = float(x.mean())
                out[col+"_median"] = float(np.median(x))
                q25, q75 = iqr(x); out[col+"_q25"] = q25; out[col+"_q75"] = q75
        H = float(_num(df.get("hits", np.nan)).sum()); M = float(_num(df.get("M_cands", np.nan)).sum())
        plo, phi = wilson_ci(H, M)
        out["pooled_hits"] = H; out["pooled_cands"] = M
        out["pooled_prec_lo95"] = plo; out["pooled_prec_hi95"] = phi
        cov1 = np.mean((_num(df.get("recall", np.nan)) >= 1.0).fillna(False))*100.0 if "recall" in df.columns else np.nan
        out["pct_coverage_1_0"] = float(cov1) if not np.isnan(cov1) else np.nan
        out["prop_p_ge_0.99"] = float(np.mean((_num(df.get("precision_a02", np.nan)) >= 0.99).fillna(False)))
        out["prop_q_ge_0.95"] = float(np.mean((_num(df.get("recall", np.nan)) >= 0.95).fillna(False)))
        out["prop_q95_le_0.2"] = float(np.mean((_num(df.get("q95_normdist", np.nan)) <= 0.2).fillna(False)))
        out["n_seeds"] = int(df.shape[0])
        return out

    def tests_block(df):
        out = {}
        pvals = _num(df.get("precision_a02", np.nan))
        p_t, tstat, n = onesided_t_norm_p_greater(pvals, 0.99)
        r = int(np.sum(pvals > 0.99)); p_sign = binom_sign_test_p(r, int(np.sum(np.isfinite(pvals))))
        H = float(_num(df.get("hits", np.nan)).sum()); M = float(_num(df.get("M_cands", np.nan)).sum())
        plo, phi = wilson_ci(H, M)
        out.update({
            "A1_mean_precision": float(np.nanmean(pvals)) if pvals.size else np.nan,
            "A1_median_precision": float(np.nanmedian(pvals)) if pvals.size else np.nan,
            "A1_pooled_hits": H, "A1_pooled_cands": M,
            "A1_pooled_prec_lo95": plo, "A1_pooled_prec_hi95": phi,
            "A1_t_one_sided_p_vs_0.99": p_t, "A1_t_stat": tstat, "A1_n": n,
            "A1_sign_r_over_S": f"{r}/{int(np.sum(np.isfinite(pvals)))}",
            "A1_sign_p": p_sign
        })
        qvals = _num(df.get("recall", np.nan))
        p_t2, tstat2, n2 = onesided_t_norm_p_greater(qvals, 0.95)
        r2 = int(np.sum(qvals > 0.95)); p_sign2 = binom_sign_test_p(r2, int(np.sum(np.isfinite(qvals))))
        out.update({
            "A2_mean_recall": float(np.nanmean(qvals)) if qvals.size else np.nan,
            "A2_median_recall": float(np.nanmedian(qvals)) if qvals.size else np.nan,
            "A2_t_one_sided_p_vs_0.95": p_t2, "A2_t_stat": tstat2, "A2_n": n2,
            "A2_sign_r_over_S": f"{r2}/{int(np.sum(np.isfinite(qvals)))}",
            "A2_sign_p": p_sign2
        })
        q95 = _num(df.get("q95_normdist", np.nan))
        p_t3, tstat3, n3 = onesided_t_norm_p_less(q95, 0.2)
        r3 = int(np.sum(q95 < 0.2)); p_sign3 = binom_sign_test_p(r3, int(np.sum(np.isfinite(q95))))
        out.update({
            "A3_median_q95": float(np.nanmedian(q95)) if q95.size else np.nan,
            "A3_t_one_sided_p_q95_lt_0.2": p_t3, "A3_t_stat": tstat3, "A3_n": n3,
            "A3_sign_r_over_S": f"{r3}/{int(np.sum(np.isfinite(q95)))}",
            "A3_sign_p": p_sign3
        })
        bvals = _num(df.get("logdist_slope_b", np.nan))
        p_t4, tstat4, n4 = onesided_t_norm_p_less(bvals, 0.0)
        out.update({
            "A4_median_b": float(np.nanmedian(bvals)) if bvals.size else np.nan,
            "A4_median_R2": float(np.nanmedian(_num(df.get("logdist_r2", np.nan)))),
            "A4_t_one_sided_p_b_lt_0": p_t4, "A4_t_stat": tstat4, "A4_n": n4,
            "A4_median_half_life": float(np.nanmedian(_num(df.get("half_life_epochs", np.nan))))
        })
        return out

    per_seed.to_csv(os.path.join(stats_dir, "per_seed.csv"), index=False, float_format="%.6f")

    summaries = []; tests=[]
    regs = sorted(per_seed["regime"].dropna().unique().tolist())
    for reg in regs + ["Pooled"]:
        sub = per_seed if reg=="Pooled" else per_seed[per_seed["regime"]==reg]
        s = summarize(sub); s["group"] = reg; summaries.append(s)
        t = tests_block(sub); t["group"] = reg; tests.append(t)

    scheme_summary = pd.DataFrame(summaries)
    tests_A = pd.DataFrame(tests)
    scheme_summary.to_csv(os.path.join(stats_dir, "scheme_summary.csv"), index=False, float_format="%.6f")
    tests_A.to_csv(os.path.join(stats_dir, "tests_A.csv"), index=False, float_format="%.6f")

    # Hitting-time table
    eps_list = [0.30, 0.20, 0.15, 0.10]
    rows_ht = []
    for reg in regs + ["Pooled"]:
        sub = per_seed if reg=="Pooled" else per_seed[per_seed["regime"]==reg]
        for eps in eps_list:
            col = f"tau_hit_q95_{eps:.2f}"
            vals = _num(sub[col]) if col in sub.columns else pd.Series([])
            med = float(np.nanmedian(vals)) if vals.size else np.nan
            q25, q75 = iqr(vals) if vals.size else (np.nan, np.nan)
            succ = float(np.mean(np.isfinite(vals))) * 100.0 if vals.size else np.nan
            rows_ht.append({"group":reg, "epsilon":eps, "tau_hit_median":med,
                            "tau_hit_q25":q25, "tau_hit_q75":q75, "success_pct":succ})
    hitting_times = pd.DataFrame(rows_ht)
    hitting_times.to_csv(os.path.join(stats_dir, "hitting_times.csv"), index=False, float_format="%.6f")

    # Headlines
    pooled = scheme_summary[scheme_summary["group"]=="Pooled"].iloc[0] if not scheme_summary.empty else None
    pooled_test = tests_A[tests_A["group"]=="Pooled"].iloc[0] if not tests_A.empty else None
    if pooled is not None and pooled_test is not None:
        htP = hitting_times[hitting_times["group"]=="Pooled"].set_index("epsilon")
        med_tau_020 = float(htP.loc[0.20,"tau_hit_median"]) if 0.20 in htP.index else np.nan
        med_tau_010 = float(htP.loc[0.10,"tau_hit_median"]) if 0.10 in htP.index else np.nan
        lines = [
            f"Endpoint precision at α₀=0.20 is {pooled['precision_a02_mean']:.3f} (median {pooled['precision_a02_median']:.3f}); pooled 95% CI [{pooled['pooled_prec_lo95']:.3f}–{pooled['pooled_prec_hi95']:.3f}] ({int(round(pooled['pooled_hits']))}/{int(round(pooled['pooled_cands']))}); one-sided p (μ≤0.99) = {pooled_test['A1_t_one_sided_p_vs_0.99']:.3g}.",
            f"α-free tightness: median q95 = {pooled['q95_normdist_median']:.3f}; one-sided p (E[q95] < 0.2) = {tests_A.loc[tests_A['group']=='Pooled','A3_t_one_sided_p_q95_lt_0.2'].values[0]:.3g}.",
            f"Post-100% log-distance slope median b = {pooled['logdist_slope_b_median']:.4f}; half-life ≈ {pooled['half_life_epochs_median']:.2f} τ.",
            f"95th-percentile z crosses 0.20 by τ≈{med_tau_020:.1f} and 0.10 by τ≈{med_tau_010:.1f} (pooled medians)."
        ]
        pd.DataFrame({"headline": lines}).to_csv(os.path.join(stats_dir, "headlines.csv"), index=False)

    # ------------------ PRINT-READY TABLES ------------------

    tables_dir = _ensure_dir(os.path.join(stats_dir, "tables"))

    # X3-A
    def fmt_ci(lo, hi): 
        return f"[{lo:.3f}–{hi:.3f}]" if np.isfinite(lo) and np.isfinite(hi) else ""
    rowsA=[]
    for reg in regs + ["Pooled"]:
        summ = scheme_summary[scheme_summary["group"]==reg].iloc[0]
        tst  = tests_A[tests_A["group"]==reg].iloc[0]
        rowsA.append({
            "Scheme": reg,
            "Seeds": int(summ["n_seeds"]),
            "Mean precision": round(summ["precision_a02_mean"], 3) if pd.notna(summ["precision_a02_mean"]) else "",
            "Median precision": round(summ["precision_a02_median"], 3) if pd.notna(summ["precision_a02_median"]) else "",
            "Pooled hits/total": f"{int(round(summ['pooled_hits']))}/{int(round(summ['pooled_cands']))}",
            "Pooled 95% CI": fmt_ci(summ["pooled_prec_lo95"], summ["pooled_prec_hi95"]),
            "t-test p vs 0.99 (one-sided)": f"{tst['A1_t_one_sided_p_vs_0.99']:.3g}" if pd.notna(tst["A1_t_one_sided_p_vs_0.99"]) else "",
            "Mean recall": round(summ["recall_mean"], 3) if pd.notna(summ["recall_mean"]) else "",
            "Median recall": round(summ["recall_median"], 3) if pd.notna(summ["recall_median"]) else "",
            "% coverage=1.0": round(summ["pct_coverage_1_0"], 1) if pd.notna(summ["pct_coverage_1_0"]) else ""
        })
    pd.DataFrame(rowsA).to_csv(os.path.join(tables_dir, "table_X3_A.csv"), index=False)

    # X3-B
    def fmt_med_iqr(m, q25, q75, nd=3):
        return "" if not np.isfinite(m) else f"{m:.{nd}f} [{q25:.{nd}f}–{q75:.{nd}f}]"
    rowsB=[]
    for reg in regs + ["Pooled"]:
        summ = scheme_summary[scheme_summary["group"]==reg].iloc[0]
        tst  = tests_A[tests_A["group"]==reg].iloc[0]
        rowsB.append({
            "Scheme": reg,
            "Median q95 (IQR)": fmt_med_iqr(summ["q95_normdist_median"], summ["q95_normdist_q25"], summ["q95_normdist_q75"]),
            "p (q95 < 0.2, one-sided)": f"{tst['A3_t_one_sided_p_q95_lt_0.2']:.3g}" if pd.notna(tst["A3_t_one_sided_p_q95_lt_0.2"]) else "",
            "nAUP@0.3 (median, IQR)": fmt_med_iqr(summ["nAUP_0_0.3_median"], summ["nAUP_0_0.3_q25"], summ["nAUP_0_0.3_q75"]),
            "Median α* (IQR)": fmt_med_iqr(summ["alpha_star_099_median"], summ["alpha_star_099_q25"], summ["alpha_star_099_q75"])
        })
    pd.DataFrame(rowsB).to_csv(os.path.join(tables_dir, "table_X3_B.csv"), index=False)

    # X3-C
    ht = hitting_times.set_index(["group","epsilon"])
    rowsC=[]
    for reg in regs + ["Pooled"]:
        summ = scheme_summary[scheme_summary["group"]==reg].iloc[0]
        def ht_fmt(eps):
            if (reg, eps) not in ht.index: return ""
            row = ht.loc[(reg,eps)]
            return f"{row['tau_hit_median']:.2f} [{row['tau_hit_q25']:.2f}–{row['tau_hit_q75']:.2f}]"
        rowsC.append({
            "Scheme": reg,
            "Slope b (median, IQR)": fmt_med_iqr(summ["logdist_slope_b_median"], summ["logdist_slope_b_q25"], summ["logdist_slope_b_q75"], nd=4),
            "R^2 (median)": f"{summ['logdist_r2_median']:.3f}" if pd.notna(summ["logdist_r2_median"]) else "",
            "Half-life (median τ)": f"{summ['half_life_epochs_median']:.2f}" if pd.notna(summ["half_life_epochs_median"]) else "",
            "τ(0.20) med [IQR]": ht_fmt(0.20),
            "τ(0.10) med [IQR]": ht_fmt(0.10)
        })
    pd.DataFrame(rowsC).to_csv(os.path.join(tables_dir, "table_X3_C.csv"), index=False)

    # X3-D
    rowsD=[]
    for reg in regs + ["Pooled"]:
        summ = scheme_summary[scheme_summary["group"]==reg].iloc[0]
        rowsD.append({
            "Scheme": reg,
            "Coverage median": f"{summ['recall_median']:.3f}" if pd.notna(summ["recall_median"]) else "",
            "% coverage=1.0": f"{summ['pct_coverage_1_0']:.1f}" if pd.notna(summ["pct_coverage_1_0"]) else "",
            "Dup ratio median": f"{summ['dup_ratio_med_median']:.3f}" if pd.notna(summ["dup_ratio_med_median"]) else "",
            "HHI median": f"{summ['HHI_med_median']:.3f}" if pd.notna(summ["HHI_med_median"]) else "",
            "Max multiplicity med": "",  # not available from current logs
            "Dedup−supports (median)": f"{summ['dedup_minus_supports_median']:.2f}" if pd.notna(summ["dedup_minus_supports_median"]) else ""
        })
    pd.DataFrame(rowsD).to_csv(os.path.join(tables_dir, "table_X3_D.csv"), index=False)

    print(f"[DONE] Wrote stats to: {stats_dir}")
    for f in [
        "per_seed.csv","scheme_summary.csv","tests_A.csv",
        "hitting_times.csv","headlines.csv",
        "tables/table_X3_A.csv","tables/table_X3_B.csv",
        "tables/table_X3_C.csv","tables/table_X3_D.csv"
    ]:
        print(" -", os.path.join(stats_dir, f))

if __name__ == "__main__":
    main()

