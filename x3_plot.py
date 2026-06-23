#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, argparse, glob as globmod
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REG_ORDER_PREF = ["DEF","MF","NTK"]

# ---------- basic utils ----------

def seed_color(i: int):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not base:
        base = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                "#9467bd","#8c564b","#e377c2","#7f7f7f",
                "#bcbd22","#17becf"]
    return base[i % len(base)]

BLUE = "#1f77b4"

def wilson_ci(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan)
    k = float(min(max(k,0.0), n)); n=float(n)
    p = k/n
    denom = 1.0 + (z**2)/n
    rad = (p*(1-p) + (z**2)/(4*n)) / n
    rad = max(rad, 0.0)
    center = (p + (z**2)/(2*n)) / denom
    half   = z*math.sqrt(rad)/denom
    return (center-half, center+half)

def _num(s):
    try: return pd.to_numeric(s, errors="coerce")
    except: return pd.Series([np.nan]*len(s))

def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def _savefig(fig, png, pdf=None, dpi=180):
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    if pdf: fig.savefig(pdf, bbox_inches="tight")

def _present_regimes(out_root):
    regs=set()
    for d in globmod.glob(os.path.join(out_root,"per_run","seed_*_reg_*")):
        try: regs.add(os.path.basename(d).split("_")[3].upper())
        except: pass
    ordered=[r for r in REG_ORDER_PREF if r in regs]
    for r in sorted(regs):
        if r not in ordered: ordered.append(r)
    return ordered

def _iter_per_run_dir(out_root):
    for d in globmod.glob(os.path.join(out_root,"per_run","seed_*_reg_*")):
        base=os.path.basename(d)
        try:
            seed=int(base.split("_")[1]); reg=base.split("_")[3].upper()
        except: continue
        yield reg, seed, d

def _iter_per_run_csv(out_root, name):
    for reg, seed, d in _iter_per_run_dir(out_root):
        p=os.path.join(d,name)
        if not os.path.exists(p): continue
        try: df=pd.read_csv(p)
        except: continue
        yield reg, seed, p, df

# ---------- meta / normalization ----------

def _load_bar_g_map(out_root):
    mp={}
    for reg, seed, _, meta in _iter_per_run_csv(out_root,"per_seed_meta.csv"):
        try: bar_g=float(meta.get("bar_g", pd.Series([np.nan])).iloc[0])
        except: bar_g=np.nan
        try: n=int(meta.get("n", pd.Series([np.nan])).iloc[0])
        except: n=np.nan
        mp[(reg,seed)]=(bar_g,n)
    return mp

def load_tau_series_norm(out_root):
    """Return per-epoch rows with tau>=0 and mean_d_norm = mean_d_support / bar_g (if bar_g>0)."""
    bar=_load_bar_g_map(out_root)
    rows=[]
    for reg, seed, _, df in _iter_per_run_csv(out_root,"per_epoch_all.csv"):
        have=[c for c in ["epoch","tau","loss","cand_count","mean_d_support"] if c in df.columns]
        if "tau" not in have: continue
        d=df[have].copy()
        d["regime"]=reg; d["seed_id"]=seed
        d=d[_num(d["tau"]).fillna(np.inf)>=0].copy()
        if "mean_d_support" in d.columns:
            g=bar.get((reg,seed),(np.nan,np.nan))[0]
            if np.isfinite(g) and g>0:
                d["mean_d_norm"]=_num(d["mean_d_support"])/g
            else:
                d["mean_d_norm"]=_num(d["mean_d_support"])
        else:
            d["mean_d_norm"]=np.nan
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ---------- t_mm (based on loss < 1/n) ----------

def t_mm_per_regime(out_root, default_n=20):
    meta=_load_bar_g_map(out_root)
    tmm={}
    for reg in _present_regimes(out_root):
        firsts=[]
        for r, seed, _, df in _iter_per_run_csv(out_root,"per_epoch_all.csv"):
            if r!=reg or "epoch" not in df.columns or "loss" not in df.columns: continue
            n=meta.get((r,seed),(np.nan,np.nan))[1]
            thr=1.0/ (n if (n and n>0) else default_n)
            s=df[["epoch","loss"]].dropna().sort_values("epoch")
            idx=s.index[_num(s["loss"])<thr]
            if len(idx)>0:
                firsts.append(float(s.loc[idx[0],"epoch"]))
        tmm[reg]=float(np.mean(firsts)) if firsts else None
    return tmm

# ---------- A (epoch-norm) removed; AplusF7 (tau) requested ----------

def build_AplusF7_stitched_tau(out_root):
    """
    Stitched plot (per regime rows, 2 columns), x = tau (aligned at 0):
      Left: mean normalized distance vs tau
      Right: log(mean normalized distance) vs tau
    """
    df = load_tau_series_norm(out_root)
    if df.empty:
        print("[WARN] A+F7: no tau series; skip."); return
    regs = _present_regimes(out_root)
    figs = _ensure_dir(os.path.join(out_root,"master","figs"))

    # common y-limit for left column across all regimes
    yvals=_num(df["mean_d_norm"]).dropna().values
    y_min=0.0; y_max=float(np.nanmax(yvals)) if yvals.size>0 else 1.0
    y_max=max(y_max,1e-6)

    fig, axes = plt.subplots(len(regs), 2, figsize=(14, 4.2*len(regs)))
    for i, reg in enumerate(regs):
        sub=df[df["regime"]==reg].copy()
        # Left: mean normalized distance vs tau (τ≥0)
        axL=axes[i,0]
        for seed, s in sub.groupby("seed_id"):
            s=s.dropna(subset=["mean_d_norm","tau"]).sort_values("tau")
            if s.empty: continue
            axL.plot(s["tau"], s["mean_d_norm"], lw=1.8, alpha=0.9, color=seed_color(int(seed)), label=f"seed={int(seed)}")
        axL.set_title(f"{reg} — mean normalized distance vs $\\tau$")
        axL.set_xlabel("$\\tau$"); axL.set_ylabel(r"$\overline{{d}}/\bar g$")
        axL.set_ylim(y_min, y_max*1.02); axL.set_xlim(left=0); axL.grid(True, ls="--", alpha=0.3)

        # Right: log(mean normalized distance) vs tau
        axR=axes[i,1]
        for seed, s in sub.groupby("seed_id"):
            s=s.dropna(subset=["mean_d_norm","tau"]); s=s[s["mean_d_norm"]>0].sort_values("tau")
            if s.empty: continue
            axR.plot(s["tau"], np.log(s["mean_d_norm"]), lw=1.8, alpha=0.9, color=seed_color(int(seed)), label=f"seed={int(seed)}")
        axR.set_title(f"{reg} — $\\log(\\overline{{d}}/\\bar g)$ vs $\\tau$")
        axR.set_xlabel("$\\tau$"); axR.set_ylabel("$\\log(\\overline{{d}}/\\bar g)$")
        axR.set_xlim(left=0); axR.grid(True, ls="--", alpha=0.3)

    # Legend across whole figure
    handles, labels = [], []
    for ax in axes.ravel():
        h,l = ax.get_legend_handles_labels()
        for hi,li in zip(h,l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6,len(labels)),
                   bbox_to_anchor=(0.5,1.02), frameon=False)

    plt.tight_layout(rect=[0,0,1,0.98])
    _savefig(fig, os.path.join(figs,"AplusF7_stitched_X3.png"), os.path.join(figs,"AplusF7_stitched_X3.pdf"))
    plt.close(fig)

# ---------- F2 (precision-only, blue) ----------

def build_F2_precision_only(out_root, alpha0=0.20):
    rows=[]
    grid=[(0.05,"hits_a005"),(0.10,"hits_a010"),(0.20,"hits_a020"),(0.30,"hits_a030")]
    alpha_key=dict(grid).get(round(alpha0,2),"hits_a020")
    for reg, seed, _, d in _iter_per_run_csv(out_root,"per_epoch_all.csv"):
        for c in ["tau","cand_count",alpha_key]:
            if c not in d.columns: d[c]=np.nan
        slow_like=d[[c for _,c in grid if c in d.columns]].fillna(0).sum(axis=1)>0
        sub=d[(_num(d["tau"]).fillna(np.inf)>=0) & (slow_like | (_num(d["cand_count"])>0))].sort_values("epoch")
        if sub.empty: continue
        wnd=sub.tail(5)
        H=float(_num(wnd[alpha_key]).sum()); C=float(_num(wnd["cand_count"]).sum())
        rows.append({"regime":reg,"seed_id":seed,"hits":H,"cands":C})
    df=pd.DataFrame(rows)
    if df.empty:
        print("[WARN] F2: no endpoint data; skip."); return
    regs=_present_regimes(out_root)
    figs=_ensure_dir(os.path.join(out_root,"master","figs"))

    fig, axes = plt.subplots(len(regs), 1, figsize=(9.5, 4.2*len(regs)))
    if len(regs)==1: axes=np.array([axes])
    for i, reg in enumerate(regs):
        ax=axes[i]
        sub=df[df["regime"]==reg]
        seeds=sorted(sub["seed_id"].unique())
        for s in seeds:
            row=sub[sub["seed_id"]==s].iloc[0]
            k=float(row["hits"]); n=float(row["cands"])
            if not (np.isfinite(k) and np.isfinite(n)) or n<=0: continue
            ki,ni=int(round(min(max(k,0.0),n))), int(round(max(n,0.0)))
            p=0.0 if ni==0 else ki/ni
            lo,hi=wilson_ci(ki,ni)
            yerr=[[max(p-lo,0.0)],[max(hi-p,0.0)]]
            ax.errorbar([s],[p], yerr=yerr, fmt="o", color=BLUE, ecolor=BLUE)
        ax.set_title(f"{reg} — precision @ $\\alpha={alpha0:.2f}$")
        ax.set_ylabel("precision"); ax.set_ylim(0,1.05); ax.grid(True, ls="--", alpha=0.3)
        ax.set_xticks(seeds); ax.set_xticklabels([str(int(s)) for s in seeds])
        ax.legend([plt.Line2D([],[],color=BLUE,marker='o',linestyle='None')],[f"seeds (CI)"], loc="upper right", frameon=False)
    plt.tight_layout()
    _savefig(fig, os.path.join(figs,"F2_prec_only_X3.png"), os.path.join(figs,"F2_prec_only_X3.pdf"))
    plt.close(fig)

# ---------- F6 (precision@α vs tau) with all x labels ----------

def build_F6_precision_time(out_root, alpha0=0.20):
    rows=[]
    for reg, seed, _, d in _iter_per_run_csv(out_root,"per_epoch_all.csv"):
        if {"tau","hits_a020","cand_count"}.issubset(d.columns):
            sub=d[["tau","hits_a020","cand_count"]].copy()
            sub=sub[_num(sub["tau"]).notna()]
            sub["precision_a020"]=sub.apply(lambda r: np.nan if float(r["cand_count"])==0 else float(r["hits_a020"])/float(r["cand_count"]), axis=1)
            sub["regime"]=reg; sub["seed_id"]=seed
            rows.append(sub)
    df=pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if df.empty:
        print("[WARN] F6: no precision(τ) data; skip."); return

    regs=_present_regimes(out_root)
    figs=_ensure_dir(os.path.join(out_root,"master","figs"))

    fig, axes = plt.subplots(len(regs), 1, figsize=(9, 4.5*len(regs)), sharex=False)
    if len(regs)==1: axes=np.array([axes])
    for i, reg in enumerate(regs):
        ax=axes[i]
        sub=df[df["regime"]==reg]
        for seed, s in sub.groupby("seed_id"):
            s=s.dropna(subset=["precision_a020","tau"]).sort_values("tau")
            if s.empty: continue
            ax.plot(s["tau"], s["precision_a020"], lw=1.8, alpha=0.9, color=seed_color(int(seed)), label=f"seed={int(seed)}")
        G=sub.dropna(subset=["precision_a020"]).groupby("tau")["precision_a020"]
        if len(G)>0:
            x=G.median().index.values
            q25,q75=G.quantile(0.25).values, G.quantile(0.75).values
            ax.fill_between(x,q25,q75,alpha=0.12,linewidth=0)
        ax.set_title(f"{reg} — precision @ $\\alpha={alpha0:.2f}$ vs $\\tau$")
        ax.set_xlabel("$\\tau$")
        ax.set_ylabel("precision"); ax.set_ylim(0,1.05); ax.grid(True, ls="--", alpha=0.3)
        ax.tick_params(labelbottom=True)
    # Global legend
    handles, labels = [], []
    for ax in axes.ravel():
        h,l = ax.get_legend_handles_labels()
        for hi,li in zip(h,l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6,len(labels)),
                   bbox_to_anchor=(0.5,1.02), frameon=False)

    plt.tight_layout(rect=[0,0,1,0.98], h_pad=1.0)
    _savefig(fig, os.path.join(figs,"F6_precision_time_X3.png"), os.path.join(figs,"F6_precision_time_X3.pdf"))
    plt.close(fig)

# ---------- Grid E back to tau, keep orbs ----------

def build_gridE_tau(out_root):
    rows=[]
    for reg, seed, _, d in _iter_per_run_csv(out_root,"per_epoch_all.csv"):
        have=[c for c in ["tau","cand_count","unique_supports_hit"] if c in d.columns]
        if "tau" not in have: continue
        dd=d[have].copy()
        dd["regime"]=reg; dd["seed_id"]=seed
        dd=dd[_num(dd["tau"]).fillna(np.inf)>=0]
        rows.append(dd)
    if not rows:
        print("[WARN] GridE: no tau data; skip."); return
    df=pd.concat(rows, ignore_index=True)
    regs=_present_regimes(out_root)
    figs=_ensure_dir(os.path.join(out_root,"master","figs"))

    # supports map for "orbs"
    supports={}
    for reg, seed, _, m in _iter_per_run_csv(out_root,"per_seed_meta.csv"):
        n_sup = m.get("support_count", m.get("n_supports"))
        if n_sup is not None and len(n_sup)>0:
            try: supports[(reg,seed)]=int(n_sup.iloc[0])
            except: pass

    fig, axes = plt.subplots(len(regs), 2, figsize=(14, 4.2*len(regs)))
    for i, reg in enumerate(regs):
        axC, axD = axes[i]
        sub=df[df["regime"]==reg].copy()

        # E1 candidates vs tau
        for seed, s in sub.groupby("seed_id"):
            m=_num(s["cand_count"])>0
            if m.any():
                axC.plot(_num(s.loc[m,"tau"]), _num(s.loc[m,"cand_count"]),
                         lw=1.6, alpha=0.9, color=seed_color(int(seed)), label=f"seed={int(seed)}")
        # orbs at small negative tau to not overlap curves; curves still start at 0
        seeds_sorted=sorted(sub["seed_id"].dropna().unique().tolist())
        offsets=np.linspace(-0.12, -0.02, num=len(seeds_sorted)) if seeds_sorted else []
        for off, s in zip(offsets, seeds_sorted):
            n_sup=supports.get((reg,int(s)))
            if n_sup is not None:
                axC.plot([off],[n_sup], "o", mec="black", mfc=seed_color(seeds_sorted.index(s)), ms=6, alpha=0.95)
        axC.set_title(f"{reg} — candidates vs $\\tau$")
        axC.set_xlabel("$\\tau$"); axC.set_ylabel("#candidates")
        left = -0.15 if len(seeds_sorted)>0 else 0.0
        axC.set_xlim(left=left); axC.grid(True, ls="--", alpha=0.3)

        # E2 dup ratio vs tau
        for seed, s in sub.groupby("seed_id"):
            cc=_num(s["cand_count"]); uh=_num(s["unique_supports_hit"])
            valid=(cc>0) & np.isfinite(uh)
            if valid.any():
                tau=_num(s.loc[valid,"tau"]); dup=1.0 - (uh.loc[valid]/cc.loc[valid])
                axD.plot(tau, dup, lw=1.6, alpha=0.9, color=seed_color(int(seed)), label=f"seed={int(seed)}")
        axD.set_ylim(0,1.05); axD.axhline(0.5, color="#aaa", ls="--", lw=1.0, alpha=0.6)
        axD.set_title(f"{reg} — duplicate ratio vs $\\tau$")
        axD.set_xlabel("$\\tau$"); axD.set_ylabel("dup. ratio")
        axD.set_xlim(left=0); axD.grid(True, ls="--", alpha=0.3)

    # legend
    handles, labels=[],[]
    for ax in axes.ravel():
        h,l=ax.get_legend_handles_labels()
        for hi,li in zip(h,l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6,len(labels)),
                   bbox_to_anchor=(0.5,1.02), frameon=False)

    plt.tight_layout(rect=[0,0,1,0.98])
    _savefig(fig, os.path.join(figs,"gridE_candidates_dup_X3.png"), os.path.join(figs,"gridE_candidates_dup_X3.pdf"))
    plt.close(fig)

# ---------- F3F4 (alpha curves + ECDF to z=1.0) ----------

def build_F3F4_alpha_ecdf(out_root, alpha0=0.20):
    figs=_ensure_dir(os.path.join(out_root,"master","figs"))
    regs=_present_regimes(out_root)

    # collect z from endpoint candidates
    data=[]
    for reg, seed, _, d in _iter_per_run_csv(out_root,"candidates_lastK.csv"):
        if {"d_support","g_local"}.issubset(d.columns):
            z=(_num(d["d_support"])/_num(d["g_local"])).replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
            for v in z: data.append({"regime":reg,"seed_id":seed,"z":float(v)})
    if not data:
        for reg, seed, _, d in _iter_per_run_csv(out_root,"slow_candidates_buffer.csv"):
            if {"d_support","g_local"}.issubset(d.columns):
                z=(_num(d["d_support"])/_num(d["g_local"])).replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
                for v in z: data.append({"regime":reg,"seed_id":seed,"z":float(v)})
    dfe=pd.DataFrame(data)
    if dfe.empty:
        print("[WARN] F3F4: no endpoint z; skip."); return
    dfe["regime"]=dfe["regime"].astype(str).str.upper()

    # dense alpha grid 0..0.30
    alpha_grid=np.round(np.arange(0.0,0.300+0.001,0.001),3)

    # precision(α) per seed via ECDF
    curves=[]
    for reg in regs:
        seeds=sorted(dfe.loc[dfe["regime"]==reg,"seed_id"].dropna().unique().tolist())
        for seed in seeds:
            z=_num(dfe[(dfe["regime"]==reg)&(dfe["seed_id"]==seed)]["z"]).dropna().to_numpy(float)
            if z.size==0: continue
            z=np.sort(z)
            idx=np.searchsorted(z, alpha_grid, side="right")
            pvals=idx/z.size
            curves.extend({"regime":reg,"seed_id":int(seed),"alpha":float(a),"precision":float(p)}
                          for a,p in zip(alpha_grid,pvals))
    dfc=pd.DataFrame(curves)

    fig, (axL,axR) = plt.subplots(1,2, figsize=(13,5))
    reg_colors={reg: seed_color(i) for i,reg in enumerate(regs)}

    # left: precision vs alpha
    for reg in regs:
        sub=dfc[dfc["regime"]==reg]
        if sub.empty: continue
        G=sub.groupby(["alpha","seed_id"])["precision"].mean().reset_index()
        GG=G.groupby("alpha")["precision"]
        x=np.array(sorted(GG.groups.keys()),dtype=float)
        med=GG.median().reindex(x).to_numpy(float)
        q25=GG.quantile(0.25).reindex(x).to_numpy(float)
        q75=GG.quantile(0.75).reindex(x).to_numpy(float)
        axL.plot(x,med,lw=2.0,alpha=0.95,color=reg_colors[reg],label=reg)
        axL.fill_between(x,q25,q75,alpha=0.15,linewidth=0)
    axL.set_title("Endpoint precision vs $\\alpha$ (median ± IQR)")
    axL.set_xlabel("$\\alpha$"); axL.set_ylabel("precision")
    axL.set_xlim(0.0,0.30); axL.set_ylim(0.0,1.05); axL.grid(True, ls="--", alpha=0.3)
    axL.axvline(alpha0,color="black",ls=":",lw=1.2,alpha=0.8)

    # right: ECDF to z=1.0
    for reg in regs:
        z=_num(dfe.loc[dfe["regime"]==reg,"z"]).dropna().sort_values().to_numpy()
        if z.size==0: continue
        y=np.arange(1,z.size+1)/z.size
        axR.step(z,y,where="post",lw=2.0,alpha=0.95,color=reg_colors[reg],label=reg)
        # continue horizontally to 1.0 at y=1
        last_x=float(z[-1])
        if last_x < 1.0:
            axR.hlines(1.0, last_x, 1.0, colors=reg_colors[reg], linestyles="-", linewidth=2.0, alpha=0.95)
    axR.set_title("Endpoint ECDF of $z=d/g$")
    axR.set_xlabel("$z$"); axR.set_ylabel("ECDF")
    axR.set_xlim(0.0,1.0); axR.set_ylim(0.0,1.02); axR.grid(True, ls="--", alpha=0.3)
    axR.axvline(alpha0,color="black",ls=":",lw=1.2,alpha=0.8)

    # single 3-color legend
    handles,labels=axL.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(regs),
                   bbox_to_anchor=(0.5,1.08), frameon=False)

    plt.tight_layout(rect=[0,0,1,0.94])
    _savefig(fig, os.path.join(figs,"F3F4_alpha_ecdf_X3.png"), os.path.join(figs,"F3F4_alpha_ecdf_X3.pdf"))
    plt.close(fig)

# ---------- stitched F6 + F2 ----------

def build_F6F2_stitched(out_root, alpha0=0.20):
    """One figure with two columns: left = F6 (precision@α vs τ), right = F2 (precision@α points with CI)."""
    # prep F6 data
    rows=[]
    for reg, seed, _, d in _iter_per_run_csv(out_root,"per_epoch_all.csv"):
        if {"tau","hits_a020","cand_count"}.issubset(d.columns):
            sub=d[["tau","hits_a020","cand_count"]].copy()
            sub=sub[_num(sub["tau"]).notna()]
            sub["precision_a020"]=sub.apply(lambda r: np.nan if float(r["cand_count"])==0 else float(r["hits_a020"])/float(r["cand_count"]), axis=1)
            sub["regime"]=reg; sub["seed_id"]=seed
            rows.append(sub)
    f6=pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # prep F2 endpoint
    rows=[]
    grid=[(0.05,"hits_a005"),(0.10,"hits_a010"),(0.20,"hits_a020"),(0.30,"hits_a030")]
    alpha_key=dict(grid).get(round(alpha0,2),"hits_a020")
    for reg, seed, _, d in _iter_per_run_csv(out_root,"per_epoch_all.csv"):
        for c in ["tau","cand_count",alpha_key]:
            if c not in d.columns: d[c]=np.nan
        slow_like=d[[c for _,c in grid if c in d.columns]].fillna(0).sum(axis=1)>0
        sub=d[(_num(d["tau"]).fillna(np.inf)>=0) & (slow_like | (_num(d["cand_count"])>0))].sort_values("epoch")
        if sub.empty: continue
        wnd=sub.tail(5)
        H=float(_num(wnd[alpha_key]).sum()); C=float(_num(wnd["cand_count"]).sum())
        rows.append({"regime":reg,"seed_id":seed,"hits":H,"cands":C})
    f2=pd.DataFrame(rows)

    if f6.empty and f2.empty:
        print("[WARN] F6F2: nothing to plot; skip."); return

    regs=_present_regimes(out_root)
    figs=_ensure_dir(os.path.join(out_root,"master","figs"))

    fig, axes = plt.subplots(len(regs), 2, figsize=(16, 4.6*len(regs)))
    for i, reg in enumerate(regs):
        # left: F6
        axL=axes[i,0]
        sub=f6[f6["regime"]==reg]
        for seed, s in sub.groupby("seed_id"):
            s=s.dropna(subset=["precision_a020","tau"]).sort_values("tau")
            if s.empty: continue
            axL.plot(s["tau"], s["precision_a020"], lw=1.8, alpha=0.9, color=seed_color(int(seed)), label=f"seed={int(seed)}")
        G=sub.dropna(subset=["precision_a020"]).groupby("tau")["precision_a020"]
        if len(G)>0:
            x=G.median().index.values
            q25,q75=G.quantile(0.25).values, G.quantile(0.75).values
            axL.fill_between(x,q25,q75,alpha=0.12,linewidth=0)
        axL.set_title(f"{reg} — precision @ $\\alpha={alpha0:.2f}$ vs $\\tau$")
        axL.set_xlabel("$\\tau$"); axL.set_ylabel("precision"); axL.set_ylim(0,1.05); axL.grid(True, ls="--", alpha=0.3)

        # right: F2 precision-only, blue
        axR=axes[i,1]
        sub2=f2[f2["regime"]==reg]
        seeds=sorted(sub2["seed_id"].unique()) if not sub2.empty else []
        for s in seeds:
            row=sub2[sub2["seed_id"]==s].iloc[0]
            k=float(row["hits"]); n=float(row["cands"])
            if not (np.isfinite(k) and np.isfinite(n)) or n<=0: continue
            ki,ni=int(round(min(max(k,0.0),n))), int(round(max(n,0.0)))
            p=0.0 if ni==0 else ki/ni
            lo,hi=wilson_ci(ki,ni)
            yerr=[[max(p-lo,0.0)],[max(hi-p,0.0)]]
            axR.errorbar([s],[p], yerr=yerr, fmt="o", color=BLUE, ecolor=BLUE)
        axR.set_title(f"{reg} — precision @ $\\alpha={alpha0:.2f}$")
        axR.set_ylabel("precision"); axR.set_ylim(0,1.05); axR.grid(True, ls="--", alpha=0.3)
        axR.set_xticks(seeds); axR.set_xticklabels([str(int(s)) for s in seeds])

    # legends
    handles,labels=[],[]
    for ax in axes[:,0]:
        h,l=ax.get_legend_handles_labels()
        for hi,li in zip(h,l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6,len(labels)),
                   bbox_to_anchor=(0.5,1.02), frameon=False)

    plt.tight_layout(rect=[0,0,1,0.98])
    _savefig(fig, os.path.join(figs,"F6F2_stitched_X3.png"), os.path.join(figs,"F6F2_stitched_X3.pdf"))
    plt.close(fig)

# ---------- main ----------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--alpha0", type=float, default=0.20)
    args=ap.parse_args()
    out_root=os.path.abspath(args.out_root)

    build_AplusF7_stitched_tau(out_root)                 # A+F7 stitched, τ axis
    build_F2_precision_only(out_root, alpha0=args.alpha0)# F2 precision-only, blue
    build_F6_precision_time(out_root, alpha0=args.alpha0)# F6 each row labeled, more spacing
    build_gridE_tau(out_root)                            # Grid E back to τ (+ orbs)
    build_F3F4_alpha_ecdf(out_root, alpha0=args.alpha0)  # ECDF to z=1.0
    build_F6F2_stitched(out_root, alpha0=args.alpha0)    # New stitched F6 + F2

    print(f"[DONE] figures written to: {os.path.join(out_root,'master','figs')}")

if __name__=="__main__":
    main()

