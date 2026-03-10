#!/usr/bin/env python3
"""
plot_multisweep_dependencies.py

Reads the long-format CSV produced by run_multisweep_parallel.py and draws
dependency plots for bias/RMSE/coverage as functions of (p_sel, q_sel, n, theta).

Outputs:
  - Heatmaps over (p_sel, q_sel) faceted by n and theta (for each method)
  - Line plots over n and theta with shading (aggregated over p_sel,q_sel)
  - Optional det(K) stratification summaries

No seaborn (matplotlib only).

Example:
  python plot_multisweep_dependencies.py \
    --csv multisweep_long.csv \
    --out_dir plots \
    --agg over_replicates \
    --facet_n 20000 50000 \
    --facet_theta 0.693147181 1.098612289
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ("bias", "rmse", "coverage")


def ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")


def add_metrics_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "bias" not in df.columns:
        ensure_cols(df, ["ln_hat", "ln_true"])
        df["bias"] = df["ln_hat"] - df["ln_true"]
    if "cover_95" in df.columns and "coverage" not in df.columns:
        df["coverage"] = df["cover_95"].astype(float)
    if "coverage" not in df.columns:
        ensure_cols(df, ["ln_hat", "ln_true", "se_hat"])
        df["coverage"] = (
            (df["ln_true"] >= df["ln_hat"] - 1.96 * df["se_hat"])
            & (df["ln_true"] <= df["ln_hat"] + 1.96 * df["se_hat"])
        ).astype(float)
    return df


def agg_over_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate replicate-level rows into mean metrics at each (method,p,q,n,theta).
    """
    df = add_metrics_if_needed(df)
    ensure_cols(df, ["method", "p_sel", "q_sel", "n", "theta"])
    grp = df.groupby(["method", "p_sel", "q_sel", "n", "theta"], as_index=False)

    out = grp.agg(
        bias=("bias", "mean"),
        rmse=("bias", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
        coverage=("coverage", "mean"),
        se_mean=("se_hat", "mean"),
        feasible_rate=("feasible", "mean") if "feasible" in df.columns else ("bias", "size"),
        lam_median=("lam", "median") if "lam" in df.columns else ("bias", "size"),
    )

    # det(K) if available or computable
    if "detK" in df.columns:
        det = df.groupby(["p_sel", "q_sel"], as_index=False)["detK"].first()
        out = out.merge(det, on=["p_sel", "q_sel"], how="left")
    else:
        out["detK"] = out["p_sel"] + out["q_sel"] - 1.0
    return out


def heatmap_pq(
    df: pd.DataFrame,
    method: str,
    metric: str,
    n_val: int,
    theta_val: float,
    outpath: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Heatmap of metric over (p_sel, q_sel) for fixed (n, theta) and method.
    """
    sub = df[(df["method"] == method) & (df["n"] == n_val) & (df["theta"] == theta_val)].copy()
    if sub.empty:
        return

    ps = np.sort(sub["p_sel"].unique())
    qs = np.sort(sub["q_sel"].unique())
    pivot = sub.pivot_table(index="q_sel", columns="p_sel", values=metric, aggfunc="mean")

    Z = pivot.reindex(index=qs, columns=ps).to_numpy()

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"{method} | {metric} | n={n_val} | theta={theta_val:.6g}")
    ax.set_xlabel(r"$p_{\mathrm{sel}}$")
    ax.set_ylabel(r"$q_{\mathrm{sel}}$")

    ax.set_xticks(np.arange(len(ps)))
    ax.set_xticklabels([f"{p:.2f}" for p in ps], rotation=60, ha="right")
    ax.set_yticks(np.arange(len(qs)))
    ax.set_yticklabels([f"{q:.2f}" for q in qs])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def line_over_n(
    df: pd.DataFrame,
    method: str,
    metric: str,
    theta_val: float,
    outpath: Path,
) -> None:
    """
    Line plot of metric vs n at fixed theta, aggregated over (p_sel,q_sel).
    Shows mean and +-1 SD across (p,q) grid.
    """
    sub = df[(df["method"] == method) & (df["theta"] == theta_val)].copy()
    if sub.empty:
        return

    grp = sub.groupby("n")[metric]
    mean = grp.mean().sort_index()
    sd = grp.std(ddof=0).reindex(mean.index).fillna(0.0)

    x = mean.index.to_numpy()
    y = mean.to_numpy()
    s = sd.to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x, y, marker="o")
    ax.fill_between(x, y - s, y + s, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Sample size $n$")
    ax.set_ylabel(metric)
    ax.set_title(f"{method} | {metric} vs n | theta={theta_val:.6g} (avg over p,q)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def line_over_theta(
    df: pd.DataFrame,
    method: str,
    metric: str,
    n_val: int,
    outpath: Path,
) -> None:
    """
    Line plot of metric vs theta at fixed n, aggregated over (p_sel,q_sel).
    Shows mean and +-1 SD across (p,q) grid.
    """
    sub = df[(df["method"] == method) & (df["n"] == n_val)].copy()
    if sub.empty:
        return

    grp = sub.groupby("theta")[metric]
    mean = grp.mean().sort_index()
    sd = grp.std(ddof=0).reindex(mean.index).fillna(0.0)

    x = mean.index.to_numpy()
    y = mean.to_numpy()
    s = sd.to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x, y, marker="o")
    ax.fill_between(x, y - s, y + s, alpha=0.2)
    ax.set_xlabel(r"True log-OR $\theta$")
    ax.set_ylabel(metric)
    ax.set_title(f"{method} | {metric} vs theta | n={n_val} (avg over p,q)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="multisweep_long.csv")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--agg", choices=["over_replicates", "none"], default="over_replicates")
    ap.add_argument("--facet_n", nargs="*", default=[], help="Optional: list of n values to facet")
    ap.add_argument("--facet_theta", nargs="*", default=[], help="Optional: list of theta values to facet")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Ensure required columns exist
    ensure_cols(df, ["method", "p_sel", "q_sel", "n", "theta"])
    if args.agg == "over_replicates":
        df = agg_over_replicates(df)
    else:
        df = add_metrics_if_needed(df)

    # Determine facet values
    n_vals = sorted({int(x) for x in (args.facet_n if args.facet_n else df["n"].unique())})
    theta_vals = sorted({float(x) for x in (args.facet_theta if args.facet_theta else df["theta"].unique())})

    # Metric-specific color limits for heatmaps (helps comparability)
    # Leave None to autoscale; you can set explicit if desired.
    heat_limits = {
        "bias": (None, None),
        "rmse": (None, None),
        "coverage": (0.0, 1.0),
    }

    methods = sorted(df["method"].unique())

    # 1) Heatmaps over (p,q) for each (n,theta)
    hm_dir = out_dir / "heatmaps_pq"
    hm_dir.mkdir(exist_ok=True)

    for method in methods:
        for metric in METRICS:
            vmin, vmax = heat_limits[metric]
            for n_val in n_vals:
                for th in theta_vals:
                    outpath = hm_dir / f"hm_{metric}_{method}_n{n_val}_th{th:.6g}.png"
                    heatmap_pq(df, method, metric, n_val, th, outpath, vmin=vmin, vmax=vmax)

    # 2) Lines over n (fix theta)
    ln_dir = out_dir / "lines_over_n"
    ln_dir.mkdir(exist_ok=True)
    for method in methods:
        for metric in METRICS:
            for th in theta_vals:
                outpath = ln_dir / f"lineN_{metric}_{method}_th{th:.6g}.png"
                line_over_n(df, method, metric, th, outpath)

    # 3) Lines over theta (fix n)
    lt_dir = out_dir / "lines_over_theta"
    lt_dir.mkdir(exist_ok=True)
    for method in methods:
        for metric in METRICS:
            for n_val in n_vals:
                outpath = lt_dir / f"lineTH_{metric}_{method}_n{n_val}.png"
                line_over_theta(df, method, metric, n_val, outpath)

    # 4) Quick det(K) dependency summary (optional, single plot)
    #    Shows coverage vs det(K), aggregated over n,theta,p,q for each method.
    if "detK" not in df.columns:
        df["detK"] = df["p_sel"] + df["q_sel"] - 1.0

    dk_dir = out_dir / "detK"
    dk_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method in methods:
        sub = df[df["method"] == method].copy()
        # bin detK for smoother curve
        sub["detK_bin"] = pd.cut(sub["detK"], bins=20)
        g = sub.groupby("detK_bin", observed=True).agg(
            detK_mid=("detK", "mean"),
            coverage=("coverage", "mean"),
            bias=("bias", "mean"),
            rmse=("rmse", "mean"),
        ).sort_values("detK_mid")
        ax.plot(g["detK_mid"], g["coverage"], marker="o", label=method)
    ax.set_xlabel(r"$\det(K)=p_{\mathrm{sel}}+q_{\mathrm{sel}}-1$")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Coverage vs det(K) (binned, aggregated over n,theta,p,q)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dk_dir / "coverage_vs_detK.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] plots written to: {out_dir}")


if __name__ == "__main__":
    main()
