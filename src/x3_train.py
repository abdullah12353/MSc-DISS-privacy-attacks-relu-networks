#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
x3_train.p yX.3 trainer (BCE-with-logits), SVM-style margin targets,
piecewise-linear candidate extraction, ETA logging, and snapshot grids.

Key updates in this version:
  - Uses BCEWithLogitsLoss (labels in {0,1}), accuracy by sign(logit) >= 0.
  - SVM-style margin m_svm = min_i y_i * phi(x_i); targets are +- m_target,
    where m_target = max(0, m_svm) * alpha_m (configurable).
  - Support points are training points whose left OR right neighbor has a
    different label (on x-sorted data). Distances and precision map to the
    NEAREST SUPPORT (not nearest train point).
  - Stable ETA printed on every fast eval.
  - Writes per-run CSVs and snapshots; also builds per-scheme snapshot grids
    (2 columns wide) under master/figs.

Outputs under: <OUT>/per_run/seed_<seed>_reg_<REG>/
  - per_epoch_all.csv                      (fast+slow rows, from epoch 0)
  - per_epoch_post100.csv                  (tau >= 0 only; present if 100% reached)
  - per_seed_meta.csv
  - slow_candidates_buffer.csv             (rolling buffer of last BUFFER_K evals)
  - candidates_lastK.csv                   (last K post-100% slow evals)
  - model_final.pt, model_config.json
  - snap_seed=<seed>_reg=<REG>.png         (final endpoint snapshot)

And scheme-level grids (trainer assembles them on exit):
  <OUT>/master/figs/snapshots_X3_<REG>.png   (2 columns wide, high-DPI)

Run example (smoke):
  python x3_train.py --out-root ~/privacy-attack/results_.../array_SMOKE --regime DEF --seed 0 \
    --n 20 --x-min -1 --x-max 1 --epochs 2000 --hidden 256 \
    --fast-eval-every 200 --slow-eval-every 1000 --jitter-frac 0.02 \
    --merge-tol-frac 0.05 --bp-eps 1e-12 --root-tol 1e-9 --alpha-m 1.0 --lastK 5 --buffer-K 50
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def to_cpu(x):
    return x.detach().cpu().numpy()


def fmt_hms(seconds):
    if not (isinstance(seconds, (int, float)) and math.isfinite(seconds)) or seconds < 0:
        return "--:--:--"
    s = int(seconds + 0.5)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ETATracker:
    """Smoothed ETA based on last few fast ticks."""
    def __init__(self, total_epochs: int, smooth_k: int = 10):
        self.total = int(total_epochs)
        self.hist = deque(maxlen=max(3, smooth_k))  # (epoch, wall_time)
        self.t0 = time.time()

    def update(self, epoch: int) -> str:
        now = time.time()
        self.hist.append((epoch, now))
        if len(self.hist) < 2 or epoch <= 0:
            return f"ETA={fmt_hms(float('nan'))} elapsed={fmt_hms(now - self.t0)}"
        e0, t0 = self.hist[0]
        e1, t1 = self.hist[-1]
        de = max(1, e1 - e0)
        dt = max(1e-6, t1 - t0)
        eps = de / dt  # epochs/sec
        remaining = max(0, self.total - epoch)
        eta = remaining / eps
        return f"ETA={fmt_hms(eta)} elapsed={fmt_hms(now - self.t0)}"


def even_plus_jitter(n, x_min, x_max, jitter_frac, rng):
    xs = np.linspace(x_min, x_max, n)
    bar_g = (x_max - x_min) / max(1, n - 1)
    jitter = rng.uniform(-jitter_frac * bar_g, jitter_frac * bar_g, size=n) if jitter_frac > 0 else 0.0
    xs = xs + jitter
    xs = np.clip(xs, x_min, x_max)
    xs.sort()
    # labels in {-1,+1}
    ys = rng.binomial(1, 0.5, size=n)
    ys = 2 * ys - 1
    return xs.astype(np.float64), ys.astype(np.int64), float(bar_g)


def compute_support_indices(x_sorted, y_sorted_pm1):
    """Support = training point with at least one neighbor of different label."""
    n = len(x_sorted)
    sup = []
    for i in range(n):
        left_diff = (i > 0 and y_sorted_pm1[i - 1] != y_sorted_pm1[i])
        right_diff = (i < n - 1 and y_sorted_pm1[i + 1] != y_sorted_pm1[i])
        if left_diff or right_diff:
            sup.append(i)
    return np.array(sorted(set(sup)), dtype=int)


def local_spacing_g_at_support(x_sorted, s_idx):
    """g(x*) = 0.5 * ((x* - left_train) + (right_train - x*)) with endpoint guards."""
    n = len(x_sorted)
    x = x_sorted
    i = int(s_idx)
    if n == 1:
        return 1.0
    if i == 0:
        left_gap = x[1] - x[0]
        right_gap = x[1] - x[0]
    elif i == n - 1:
        left_gap = x[-1] - x[-2]
        right_gap = x[-1] - x[-2]
    else:
        left_gap = x[i] - x[i - 1]
        right_gap = x[i + 1] - x[i]
    return 0.5 * float(left_gap + right_gap)


def nearest_support(support_x, cand_x):
    """Index of nearest support in sorted support_x using searchsorted."""
    if len(support_x) == 0:
        return -1
    pos = np.searchsorted(support_x, cand_x)
    if pos <= 0:
        return 0
    if pos >= len(support_x):
        return len(support_x) - 1
    left = pos - 1
    right = pos
    if abs(cand_x - support_x[left]) <= abs(cand_x - support_x[right]):
        return left
    return right


# ----------------------------
# Model
# ----------------------------

class MLP2(nn.Module):
    def __init__(self, hidden, out_bias=False):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, 1, bias=out_bias)

    def forward(self, x):
        # x: (B,1)
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return out.view(-1)

    def extract_breakpoints(self, bp_eps: float = 1e-12):
        # bp_j = -b1_j / w1_j for |w|>eps
        w = self.fc1.weight.detach().cpu().numpy().reshape(-1)
        b = self.fc1.bias.detach().cpu().numpy().reshape(-1)
        mask = np.where(np.abs(w) > bp_eps)[0]
        if mask.size == 0:
            return np.array([], dtype=np.float64)
        bp = -b[mask] / w[mask]
        bp = np.unique(np.asarray(bp, dtype=np.float64))
        return np.sort(bp)


def init_regime(model: MLP2, regime: str):
    with torch.no_grad():
        if regime == "DEF":
            # leave PyTorch defaults
            return
        elif regime == "MF":
            for p in [model.fc1.weight, model.fc1.bias, model.fc2.weight]:
                p.copy_(torch.randn_like(p) * 0.1)
            if model.fc2.bias is not None:
                model.fc2.bias.zero_()
        elif regime == "NTK":
            model.fc1.weight.copy_(torch.randn_like(model.fc1.weight) * 10.0)
            model.fc1.bias.copy_(torch.randn_like(model.fc1.bias) * 10.0)
            model.fc2.weight.copy_(torch.randn_like(model.fc2.weight) * 0.1)
            if model.fc2.bias is not None:
                model.fc2.bias.zero_()
        else:
            raise ValueError("unknown regime: " + str(regime))


def regime_lr(regime: str) -> float:
    if regime == "NTK":
        return 0.01
    return 0.001


# ----------------------------
# Candidate extraction
# ----------------------------

def extract_candidates_piecewise(model: MLP2, m_target: float,
                                 x_min: float, x_max: float,
                                 merge_tol_abs: float,
                                 bp_eps: float,
                                 root_tol: float,
                                 device: torch.device):
    """
    Deterministic extraction:
      - compute breakpoints from fc1
      - sort unique, clamp to [x_min,x_max]
      - evaluate logits at bps
      - scan segments linearly and solve for r where y = +- m_target
      - include endpoints that lie on target (within tol)
      - dedup roots with absolute merge tolerance
    """
    if m_target <= 0:
        return []

    bps = model.extract_breakpoints(bp_eps)
    # include span ends to form segments
    grid = np.concatenate([[x_min], bps[(bps >= x_min) & (bps <= x_max)], [x_max]])
    grid = np.unique(np.clip(grid, x_min, x_max))
    if grid.size < 2:
        return []

    with torch.no_grad():
        xs = torch.tensor(grid, dtype=torch.float64, device=device).view(-1, 1)
        ys = model(xs).double().cpu().numpy()

    targets = [+m_target, -m_target]
    roots = []

    for i in range(len(grid) - 1):
        x0, x1 = grid[i], grid[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        # near-constant segment?
        if abs(dy) < root_tol:
            # record endpoints if they coincide with target within tolerance
            for t in targets:
                if abs(y0 - t) <= root_tol:
                    roots.append(x0)
                if abs(y1 - t) <= root_tol:
                    roots.append(x1)
            continue
        # linear solve for each target
        for t in targets:
            r = x0 + (t - y0) * (dx / dy)
            if (r >= x0 - root_tol) and (r <= x1 + root_tol):
                roots.append(r)

    if not roots:
        return []

    roots = np.sort(np.asarray(roots, dtype=np.float64))
    # deduplicate with absolute tolerance
    dedup = []
    for r in roots:
        if len(dedup) == 0 or abs(r - dedup[-1]) > merge_tol_abs:
            dedup.append(r)
    # clamp to span
    dedup = [min(max(x_min, float(r)), x_max) for r in dedup]
    return dedup


# ----------------------------
# Snapshots
# ----------------------------

def draw_snapshot(model, regime, seed, x_sorted, y_sorted_pm1,
                  candidates, m_target, out_path_png,
                  device, dpi=300, show_supports=False):  # default False now
    x_min, x_max = float(x_sorted[0]), float(x_sorted[-1])
    xs_line = np.linspace(x_min, x_max, 1024, dtype=np.float64)
    with torch.no_grad():
        xv = torch.tensor(xs_line, dtype=torch.float64, device=device).view(-1, 1)
        yv = to_cpu(model(xv).double())

    # breakpoints on-span
    bp = model.extract_breakpoints(1e-12)
    bp = bp[(bp >= x_min) & (bp <= x_max)]

    # y-lims: ONLY the model output, padded slightly
    y_min, y_max = float(np.min(yv)), float(np.max(yv))
    yr = max(1e-6, y_max - y_min)
    pad = 0.06 * yr
    y0 = y_min - pad
    y1 = y_max + pad

    fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
    ax.plot(xs_line, yv, lw=1.6)

    # dotted ±m lines (may sit off-frame; that's fine)
    ax.axhline(+m_target, ls=":", lw=1.0)
    ax.axhline(-m_target, ls=":", lw=1.0)

    # x-axis line at 0
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)

    # training points (lighter blue, same red)
    y01 = (y_sorted_pm1 + 1) // 2
    for xi, yi01 in zip(x_sorted, y01):
        color = "red" if int(yi01) == 0 else "#6fa8dc"  # lighter blue
        ax.scatter([xi], [0.0], s=48, marker="o", edgecolors="none",
                   c=color, alpha=0.85, zorder=2)

    # candidates: black crosses, slightly thicker lines
    if candidates:
        ax.scatter(candidates, [0.0] * len(candidates),
                   marker="x", s=36, c="k", alpha=0.65, linewidths=1.8, zorder=3)


    # breakpoints as short ticks along the bottom edge (not at y=0)
    for t in bp:
        ax.plot([t, t], [y0, y0 + 0.08 * (y1 - y0)], lw=0.9, color="k", alpha=0.6, zorder=1)

    # NO support triangles anymore

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y0, y1)
    ax.set_title(f"reg={regime}  seed={seed}")
    ax.set_xlabel("x")
    ax.set_ylabel("logit f(x)")
    fig.tight_layout()
    fig.savefig(out_path_png, bbox_inches="tight")
    plt.close(fig)


def update_scheme_snapshot_grid(out_root, regime, cols=2, show_supports=True):
    """
    Assemble a per-regime grid of final per-seed snapshots found in per_run/.
    Writes: <OUT>/master/figs/snapshots_X3_<REG>.png (high DPI)
    """
    per_run_root = os.path.join(out_root, "per_run")
    figs_root = os.path.join(out_root, "master", "figs")
    ensure_dir(figs_root)
    pat = os.path.join(per_run_root, "seed_*_reg_%s" % regime)
    run_dirs = sorted([d for d in os.listdir(per_run_root) if d.startswith("seed_") and d.endswith("reg_" + regime)])
    if not run_dirs:
        return
    # collect per-seed snapshot paths
    entries = []
    for base in run_dirs:
        d = os.path.join(per_run_root, base)
        snaps = [f for f in os.listdir(d) if f.startswith("snap_seed=")]
        if not snaps:
            continue
        # take the last (or the only) snapshot
        snap = sorted(snaps)[-1]
        seed = int(base.split("_")[1])
        entries.append((seed, os.path.join(d, snap)))
    if not entries:
        return
    entries.sort(key=lambda t: t[0])

    # layout
    n = len(entries)
    cols = max(1, int(cols))
    rows = int(math.ceil(n / float(cols)))

    # build a blank canvas by stitching images side-by-side using matplotlib
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3), dpi=300)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    k = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if k >= n:
                continue
            _, img_path = entries[k]
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.set_title(os.path.basename(img_path), fontsize=9)
            k += 1

    fig.suptitle(f"Snapshot grid (regime={regime})", fontsize=12)
    fig.tight_layout()
    out_png = os.path.join(figs_root, f"snapshots_X3_{regime}.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Training / Eval
# ----------------------------

def train_one(args):
    # dirs
    out_root = os.path.abspath(os.path.expanduser(args.out_root))
    per_run_root = os.path.join(out_root, "per_run")
    run_dir = os.path.join(per_run_root, f"seed_{args.seed}_reg_{args.regime}")
    ensure_dir(run_dir)
    ensure_dir(os.path.join(out_root, "master", "figs"))

    # device
    dev = torch.device(args.device)

    # dataset
    rng = np.random.default_rng(args.seed)
    x_sorted, y_sorted_pm1, bar_g = even_plus_jitter(args.n, args.x_min, args.x_max, args.jitter_frac, rng)
    y01 = (y_sorted_pm1 + 1) // 2
    support_idx = compute_support_indices(x_sorted, y_sorted_pm1)
    support_x = x_sorted[support_idx]

    # model
    torch.set_default_dtype(torch.float64)
    model = MLP2(hidden=args.hidden, out_bias=False).to(dev)
    init_regime(model, args.regime)
    lr = regime_lr(args.regime)
   # BEFORE: opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, nesterov=False)
    # (and remove/disable any LR scheduler)

    loss_fn = nn.BCEWithLogitsLoss()

    # meta & config
    with open(os.path.join(run_dir, "model_config.json"), "w") as f:
        json.dump({
            "regime": args.regime,
            "seed": args.seed,
            "n": args.n,
            "x_min": args.x_min,
            "x_max": args.x_max,
            "hidden": args.hidden,
            "lr": lr,
            "alpha_m": args.alpha_m
        }, f)

    # CSV writers
    all_path = os.path.join(run_dir, "per_epoch_all.csv")
    post_path = os.path.join(run_dir, "per_epoch_post100.csv")
    meta_path = os.path.join(run_dir, "per_seed_meta.csv")
    slow_buf_path = os.path.join(run_dir, "slow_candidates_buffer.csv")
    lastK_path = os.path.join(run_dir, "candidates_lastK.csv")

    def open_writer(path, header):
        newfile = not os.path.exists(path)
        f = open(path, "a", buffering=1)
        if newfile:
            f.write(header + "\n")
        return f

    all_hdr = "regime,seed_id,epoch,tau,train_acc,loss,margin,cand_count,sum_d_support,mean_d_support,hits_a005,hits_a010,hits_a020,hits_a030,unique_supports_hit,dup_ratio,coverage,HHI,top1_share"
    post_hdr = all_hdr
    slow_hdr = "regime,seed_id,eval_id,epoch,tau,cand_x,support_index,support_x_star,gap_left,gap_right,g_local,d_support,hit_a005,hit_a010,hit_a020,hit_a030"

    all_f = open_writer(all_path, all_hdr)
    post_f = open_writer(post_path, post_hdr)  # may remain empty if never 100%
    slow_f = open_writer(slow_buf_path, slow_hdr)

    # write meta
    meta_row = {
        "regime": args.regime,
        "seed_id": args.seed,
        "n": args.n,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "bar_g": bar_g,
        "median_gap": float(np.median(np.diff(x_sorted))) if args.n > 1 else float("nan"),
        "support_count": int(len(support_idx)),
        "t_star": "",
        "epochs": args.epochs,
        "hidden": args.hidden,
        "lr": lr,
        "fc1_bias": True,
        "fc2_bias": False,
        "jitter_frac": args.jitter_frac,
        "train_x_sorted": json.dumps(x_sorted.tolist()),
        "train_y_sorted": json.dumps(y_sorted_pm1.tolist())
    }
    with open(meta_path, "w") as f:
        f.write(",".join(map(str, meta_row.keys())) + "\n")
        f.write(",".join(map(str, meta_row.values())) + "\n")

    # tensors
    tx = torch.tensor(x_sorted, dtype=torch.float64, device=dev).view(-1, 1)
    ty01 = torch.tensor(y01, dtype=torch.float64, device=dev)  # for BCE
    typm1 = torch.tensor(y_sorted_pm1, dtype=torch.float64, device=dev)  # for margin/acc

    # helpers
    def eval_logits():
        with torch.no_grad():
            return model(tx).double()

    def svm_margin_value():
        # m_svm = min_i y_i * phi(x_i)
        logits = eval_logits()
        m_svm = torch.min(typm1 * logits).item()
        m_target = max(0.0, m_svm) * float(args.alpha_m)
        return m_svm, m_target

    def acc_and_loss():
        logits = eval_logits()
        preds01 = (logits >= 0).to(torch.float64)
        acc = float((preds01 == ty01).to(torch.float64).mean().item())
        loss = float(loss_fn(logits, ty01).item())
        return acc, loss

    # slow metrics compute
    def compute_slow_metrics(m_target):
        # candidates
        merge_tol_abs = args.merge_tol_frac * bar_g
        cands = extract_candidates_piecewise(model, m_target,
                                             x_sorted[0], x_sorted[-1],
                                             merge_tol_abs=merge_tol_abs,
                                             bp_eps=args.bp_eps,
                                             root_tol=args.root_tol,
                                             device=dev)
        cand_count = int(len(cands))
        if cand_count == 0 or len(support_x) == 0:
            return {
                "cands": [],
                "cand_count": 0,
                "sum_d": 0.0,
                "mean_d": float("nan"),
                "hits": {0.05: 0, 0.10: 0, 0.20: 0, 0.30: 0},
                "unique_supports_hit": 0,
                "dup_ratio": 0.0,
                "coverage": 0.0,
                "HHI": float("nan"),
                "top1_share": 0.0
            }

        # map candidates to nearest support and distances
        indices = []
        dists = []
        glocals = []
        for cx in cands:
            j = nearest_support(support_x, cx)
            sj = support_idx[j]
            sx = support_x[j]
            d = abs(float(cx) - float(sx))
            g_loc = local_spacing_g_at_support(x_sorted, sj)
            indices.append(int(j))
            dists.append(d)
            glocals.append(g_loc)

        # hits@alpha by local spacing
        alphas = [0.05, 0.10, 0.20, 0.30]
        hits = {a: 0 for a in alphas}
        for d, g in zip(dists, glocals):
            if g <= 0:
                continue
            z = d / g
            for a in alphas:
                if z <= a:
                    hits[a] += 1

        # duplication / coverage
        from collections import Counter
        c = Counter(indices)
        unique_supports_hit = len(c)
        dup_ratio = 0.0 if cand_count == 0 else (1.0 - unique_supports_hit / float(cand_count))
        coverage = 0.0 if len(support_x) == 0 else unique_supports_hit / float(len(support_x))
        # HHI and top1_share
        if cand_count == 0:
            HHI = float("nan")
            top1 = 0.0
        else:
            shares = [cnt / float(cand_count) for cnt in c.values()]
            HHI = float(sum(s * s for s in shares))
            top1 = float(max(shares))

        return {
            "cands": cands,
            "cand_count": cand_count,
            "sum_d": float(sum(dists)),
            "mean_d": float(sum(dists) / cand_count) if cand_count > 0 else float("nan"),
            "hits": hits,
            "unique_supports_hit": int(unique_supports_hit),
            "dup_ratio": float(dup_ratio),
            "coverage": float(coverage),
            "HHI": float(HHI),
            "top1_share": float(top1),
            "indices": indices,
            "dists": dists,
            "glocals": glocals
        }

    # training loop
    t_star = None
    eta = ETATracker(total_epochs=args.epochs, smooth_k=10)
    slow_eval_id = 0
    slow_ring = deque(maxlen=max(1, args.buffer_K))  # store per-candidate rows; we dump at end

    for epoch in range(args.epochs + 1):
        # step
        model.train()
        logits = model(tx).double()
        loss = loss_fn(logits, ty01)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # fast eval?
        do_fast = (epoch % args.fast_eval_every == 0)
        do_slow = (epoch % args.slow_eval_every == 0)

        if not (do_fast or do_slow):
            continue

        # metrics
        acc, loss_val = acc_and_loss()
        m_svm, m_target = svm_margin_value()  # use m_target for candidates and snapshot
        # track t_star only when first time acc==1.0
        if t_star is None and acc >= 0.999999:
            t_star = epoch
            # update meta on disk with t_star
            meta_row["t_star"] = t_star
            with open(meta_path, "w") as f:
                f.write(",".join(map(str, meta_row.keys())) + "\n")
                f.write(",".join(map(str, meta_row.values())) + "\n")

        # slow metrics
        cand_count = 0
        sum_d = 0.0
        mean_d = float("nan")
        hits = {0.05: 0, 0.10: 0, 0.20: 0, 0.30: 0}
        uniq_hit = 0
        dup_ratio = 0.0
        coverage = 0.0
        HHI = float("nan")
        top1_share = 0.0
        tau = float("nan") if t_star is None else float(epoch - t_star)

        if do_slow:
            sm = compute_slow_metrics(m_target)
            cand_count = sm["cand_count"]
            sum_d = sm["sum_d"]
            mean_d = sm["mean_d"]
            hits = sm["hits"]
            uniq_hit = sm["unique_supports_hit"]
            dup_ratio = sm["dup_ratio"]
            coverage = sm["coverage"]
            HHI = sm["HHI"]
            top1_share = sm["top1_share"]

            # write candidate-level rows into slow buffer csv and ring
            if cand_count > 0:
                for cx, j, d, g in zip(sm["cands"], sm.get("indices", []), sm.get("dists", []), sm.get("glocals", [])):
                    # support index j refers to index into support_x; keep raw support index in dataset space too
                    s_idx = int(support_idx[j]) if len(support_idx) > 0 and j >= 0 else -1
                    gap_left = (x_sorted[s_idx] - x_sorted[max(0, s_idx - 1)]) if s_idx > 0 else (x_sorted[1] - x_sorted[0]) if len(x_sorted) > 1 else 1.0
                    gap_right = (x_sorted[min(len(x_sorted) - 1, s_idx + 1)] - x_sorted[s_idx]) if s_idx < len(x_sorted) - 1 else (x_sorted[-1] - x_sorted[-2]) if len(x_sorted) > 1 else 1.0
                    z = d / g if g > 0 else float("inf")
                    row = [
                        args.regime, args.seed, slow_eval_id, epoch, tau,
                        float(cx), int(s_idx), float(x_sorted[s_idx]) if s_idx >= 0 else float("nan"),
                        float(gap_left), float(gap_right), float(g),
                        float(d),
                        int(z <= 0.05), int(z <= 0.10), int(z <= 0.20), int(z <= 0.30)
                    ]
                    slow_f.write(",".join(map(str, row)) + "\n")
                    slow_ring.append(row)
            slow_eval_id += 1

        # write fast+slow row to per_epoch_all
        row_all = [
            args.regime, args.seed, epoch, tau,
            acc, loss_val, m_target,
            cand_count, sum_d, mean_d,
            hits[0.05], hits[0.10], hits[0.20], hits[0.30],
            uniq_hit, dup_ratio, coverage, HHI, top1_share
        ]
        all_f.write(",".join(map(str, row_all)) + "\n")
        # write post100 row if tau>=0
        if not math.isnan(tau) and tau >= 0:
            post_f.write(",".join(map(str, row_all)) + "\n")

        # ETA print at fast ticks
        if do_fast:
            eta_msg = ETATracker.update if isinstance(ETA := eta, ETATracker) else lambda e: ""
            print(f"[FAST] ep={epoch} acc={acc:.3f} loss={loss_val:.4g} m_target={m_target:.4g}  {eta.update(epoch)}",
                  flush=True)

    # save model
    torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pt"))

    # finalize lastK (last K post-100% slow evals)
    if len(slow_ring) > 0:
        # filter to last K with tau>=0
        # slow_ring rows: [reg, seed, eval_id, epoch, tau, cand_x, support_index, support_x_star, gap_left, gap_right, g_local, d_support, hit_a005, hit_a010, hit_a020, hit_a030]
        rows = [r for r in list(slow_ring) if isinstance(r[4], (int, float)) and (not math.isnan(r[4])) and r[4] >= 0]
        # keep only last K eval_ids among these
        if rows:
            eval_ids = sorted(set([int(r[2]) for r in rows]))[-args.lastK:]
            rows = [r for r in rows if int(r[2]) in eval_ids]
            with open(lastK_path, "w") as f:
                f.write("regime,seed_id,epoch,tau,cand_x,support_index,support_x_star,gap_left,gap_right,g_local,d_support,hit_a005,hit_a010,hit_a020,hit_a030\n")
                for r in rows:
                    # reorder to requested schema
                    out = [r[0], r[1], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15]]
                    f.write(",".join(map(str, out)) + "\n")

    # draw per-seed snapshot
    snap_path = os.path.join(run_dir, f"snap_seed={args.seed}_reg={args.regime}.png")
    m_svm, m_target = svm_margin_value()
    cands = extract_candidates_piecewise(model, m_target, x_sorted[0], x_sorted[-1],
                                         merge_tol_abs=args.merge_tol_frac * bar_g,
                                         bp_eps=args.bp_eps, root_tol=args.root_tol, device=dev)
    draw_snapshot(model, args.regime, args.seed, x_sorted, y_sorted_pm1,
                  cands, m_target, snap_path, device=dev, dpi=300, show_supports=True)

    # assemble per-regime grid (2 columns)
    if args.stitch_snapshots:
        update_scheme_snapshot_grid(out_root, args.regime, cols=2, show_supports=False)

    # close files
    all_f.close()
    post_f.close()
    slow_f.close()


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--regime", type=str, choices=["DEF","MF","NTK"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    # data
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--x-min", type=float, default=-1.0)
    ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--jitter-frac", type=float, default=0.02)
    ap.add_argument("--stitch-snapshots", action="store_true",
                help="If set, also build per-regime snapshot grids at the end.")
    # model
    ap.add_argument("--hidden", type=int, default=2048)
    # train
    ap.add_argument("--epochs", type=int, default=10000000)
    ap.add_argument("--fast-eval-every", type=int, default=10000)
    ap.add_argument("--slow-eval-every", type=int, default=50000)
    ap.add_argument("--alpha-m", type=float, default=1.0, help="scale factor on SVM margin for +-m targets")
    # numerics
    ap.add_argument("--merge-tol-frac", type=float, default=0.05)
    ap.add_argument("--bp-eps", type=float, default=1e-12)
    ap.add_argument("--root-tol", type=float, default=1e-9)
    ap.add_argument("--device", type=str, default="cpu")
    # buffers
    ap.add_argument("--lastK", type=int, default=5)
    ap.add_argument("--buffer-K", type=int, default=50)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    try:
        train_one(args)
    except Exception as e:
        print("[FATAL] exception:", repr(e), file=sys.stderr)
        raise

