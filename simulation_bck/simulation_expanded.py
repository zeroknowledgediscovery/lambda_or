#!/usr/bin/env python3
"""
lambda-OR Simulation Study (expanded, 15-point grids, det(K) column, 4-panel figures)
====================================================================================

CORRECTION NOTE (important):
---------------------------
This version fixes the transpose mismatch in the correction step.

Manuscript definition:
  p_sel = P(Ŷ=1 | Y=1, I=1)   (selection-conditional sensitivity)
  q_sel = P(Ŷ=0 | Y=0, I=1)   (selection-conditional specificity)

Misclassification operator (true -> observed):
  K_{y,ŷ} = P(Ŷ=ŷ | Y=y) =
      [[ p_sel,     1-p_sel ],
       [ 1-q_sel,   q_sel   ]]

If the 2x2 table matrix is arranged with columns indexed by Y-state, then
  E[tilde_T] = T * K.
Therefore the correction should right-multiply by (K + λI)^{-1}, NOT (K^T + λI)^{-1}.

The original symmetric sweep (p_sel=q_sel) is unaffected, but asymmetric sweeps are.

RNG NOTE (threads/processes):
----------------------------
This version uses per-replicate RNG streams derived from a SeedSequence so that:
  - thread execution is safe (no shared RNG state)
  - results are reproducible for a given Config (including cfg.seed)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng, SeedSequence

OUTPUT_DIR = os.environ.get("LAMBDAOR_SIM_OUTPUT", "./lambdaor_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class Config:
    # Core design constants
    n: int = 20000               # sample size per replicate
    varphi: float = 0.5          # P(X=1)
    theta: float = math.log(2)   # true log(OR)
    pi: float = 0.2              # target marginal P(Y=1)

    # Misclassification parameters
    p_sel: float = 0.90
    q_sel: float = 0.90

    # Monte Carlo
    R: int = 250                 # replicates per grid point

    # Ridge feasibility grid (minimal feasible)
    lam_grid: Tuple[float, ...] = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)

    # Numerical floors
    eps_pos: float = 1e-9        # positivity floor for corrected counts
    cc: float = 0.5              # continuity correction for naive OR table

    # RNG
    seed: int = 20251005         # base seed; change per grid-point if desired


# -------------------------
# Helpers: DGP calibration
# -------------------------
def find_alpha_for_target_pi(varphi: float, theta: float, target_pi: float) -> float:
    """
    Choose alpha so that:
        target_pi ≈ varphi * σ(alpha + theta) + (1 - varphi) * σ(alpha)
    by deterministic grid search (stable + reproducible).
    """
    alphas = np.linspace(-6.0, 2.0, 8001)
    best_val = float("inf")
    best_alpha = None
    for a in alphas:
        p1 = 1.0 / (1.0 + np.exp(-(a + theta)))
        p0 = 1.0 / (1.0 + np.exp(-a))
        pi = varphi * p1 + (1.0 - varphi) * p0
        v = abs(pi - target_pi)
        if v < best_val:
            best_val = v
            best_alpha = a
    return float(best_alpha)


def gen_data(cfg: Config, alpha: float, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate:
        X ~ Bernoulli(varphi)
        Y|X ~ Bernoulli(σ(alpha + theta X))
    """
    X = rng.binomial(1, cfg.varphi, size=cfg.n).astype(np.int64)
    logits = alpha + cfg.theta * X
    p = 1.0 / (1.0 + np.exp(-logits))
    Y = rng.binomial(1, p).astype(np.int64)
    return X, Y


def misclassify_parametric(Y: np.ndarray, p_sel: float, q_sel: float, rng) -> np.ndarray:
    """
    Parametric outcome misclassification using manuscript notation:
      If Y=1, flip to 0 with prob 1-p_sel  => P(Ŷ=1|Y=1)=p_sel
      If Y=0, flip to 1 with prob 1-q_sel  => P(Ŷ=0|Y=0)=q_sel
    """
    Ytil = Y.copy()
    idx1 = np.where(Y == 1)[0]
    idx0 = np.where(Y == 0)[0]

    flip_1_to_0 = rng.binomial(1, 1.0 - p_sel, size=idx1.size)
    flip_0_to_1 = rng.binomial(1, 1.0 - q_sel, size=idx0.size)

    Ytil[idx1] = 1 - flip_1_to_0
    Ytil[idx0] = flip_0_to_1
    return Ytil.astype(np.int64)


# -------------------------
# Tables + estimators
# -------------------------
def detK(p_sel: float, q_sel: float) -> float:
    # det([[p,1-p],[1-q,q]]) = p+q-1
    return float(p_sel + q_sel - 1.0)


def table_counts(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Counts in the paper's orientation:
        a = #{X=1, Y=1}
        b = #{X=0, Y=1}
        c = #{X=1, Y=0}
        d = #{X=0, Y=0}
    """
    a = float(((x == 1) & (y == 1)).sum())
    b = float(((x == 0) & (y == 1)).sum())
    c = float(((x == 1) & (y == 0)).sum())
    d = float(((x == 0) & (y == 0)).sum())
    return a, b, c, d


def log_or_and_se_from_counts(a: float, b: float, c: float, d: float) -> Tuple[float, float]:
    log_or = math.log((a * d) / (b * c))
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    return log_or, se


def naive_log_or(x: np.ndarray, ytil: np.ndarray, cc: float) -> Tuple[float, float]:
    """
    Naive log OR from noisy table with continuity correction (cc).
    """
    a, b, c, d = table_counts(x, ytil)
    a += cc
    b += cc
    c += cc
    d += cc
    return log_or_and_se_from_counts(a, b, c, d)


def lambda_or_corrected(
    x: np.ndarray,
    ytil: np.ndarray,
    p_sel: float,
    q_sel: float,
    lam: float,
    eps_pos: float
) -> Tuple[float, float, bool, Tuple[float, float, float, float]]:
    """
    Correct noisy table via:
        T^(lam) = Ttil * (K + lam I)^(-1)
    with K_{y,ŷ} = P(Ŷ=ŷ | Y=y) = [[p,1-p],[1-q,q]].

    Table matrix convention:
      rows = X (1 then 0)
      cols = observed Ŷ (1 then 0)

      Ttil = [[a_t, c_t],
              [b_t, d_t]]

    Under this convention: E[Ttil] = T_true * K, so right-multiply by K^{-1}.
    """
    a_t, b_t, c_t, d_t = table_counts(x, ytil)

    # cols are observed Ŷ states (1 then 0)
    Ttil = np.array([[a_t, c_t],
                     [b_t, d_t]], dtype=float)

    # K: true Y (rows) -> observed Ŷ (cols), ordered as (1,0)
    K = np.array([[p_sel, 1.0 - p_sel],
                  [1.0 - q_sel, q_sel]], dtype=float)

    K_reg = K + lam * np.eye(2)

    try:
        K_inv = np.linalg.inv(K_reg)
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(K_reg)

    Tcorr = Ttil @ K_inv

    # Map back to (a,b,c,d) with same convention
    a = float(Tcorr[0, 0])  # X=1, Y=1
    c = float(Tcorr[0, 1])  # X=1, Y=0
    b = float(Tcorr[1, 0])  # X=0, Y=1
    d = float(Tcorr[1, 1])  # X=0, Y=0

    feasible = (a > eps_pos) and (b > eps_pos) and (c > eps_pos) and (d > eps_pos)

    if not feasible:
        a = max(a, eps_pos)
        b = max(b, eps_pos)
        c = max(c, eps_pos)
        d = max(d, eps_pos)

    log_or, se = log_or_and_se_from_counts(a, b, c, d)
    return log_or, se, feasible, (a, b, c, d)


def _grid_seed_sequence(cfg: Config) -> SeedSequence:
    """
    Deterministic seed sequence for a grid point.
    Uses cfg.seed plus discretized parameter values so that independent
    streams are obtained even if caller forgets to vary cfg.seed by grid.
    """
    # Discretize floats to stable integers
    def f2i(x: float, scale: float) -> int:
        return int(round(x * scale))

    entropy = [
        int(cfg.seed),
        int(cfg.n),
        f2i(cfg.varphi, 10**6),
        f2i(cfg.theta, 10**6),
        f2i(cfg.pi, 10**6),
        f2i(cfg.p_sel, 10**6),
        f2i(cfg.q_sel, 10**6),
        int(cfg.R),
    ]
    return SeedSequence(entropy)


def run_cell(cfg: Config) -> pd.DataFrame:
    """
    Run one (p_sel,q_sel,n,theta,pi,varphi) configuration for cfg.R replicates.
    RNG is per-replicate and thread-safe.
    """
    alpha = find_alpha_for_target_pi(cfg.varphi, cfg.theta, cfg.pi)
    rows: List[Dict] = []
    det_val = detK(cfg.p_sel, cfg.q_sel)

    ss = _grid_seed_sequence(cfg)
    child_seeds = ss.spawn(int(cfg.R))

    for r in range(int(cfg.R)):
        rng_r = default_rng(child_seeds[r])

        X, Y = gen_data(cfg, alpha, rng=rng_r)
        Ytil = misclassify_parametric(Y, cfg.p_sel, cfg.q_sel, rng=rng_r)

        ln_naive, se_naive = naive_log_or(X, Ytil, cc=cfg.cc)

        chosen_lam = None
        chosen = None
        for lam_try in cfg.lam_grid:
            ln_lam, se_lam, feasible, counts = lambda_or_corrected(
                X, Ytil, cfg.p_sel, cfg.q_sel, lam=lam_try, eps_pos=cfg.eps_pos
            )
            if feasible:
                chosen_lam = lam_try
                chosen = (ln_lam, se_lam, True, counts)
                break

        if chosen_lam is None:
            lam_try = cfg.lam_grid[-1]
            ln_lam, se_lam, feasible, counts = lambda_or_corrected(
                X, Ytil, cfg.p_sel, cfg.q_sel, lam=lam_try, eps_pos=cfg.eps_pos
            )
            chosen_lam = lam_try
            chosen = (ln_lam, se_lam, feasible, counts)

        ln_lam, se_lam, feasible, (a_hat, b_hat, c_hat, d_hat) = chosen

        rows.append(dict(
            ln_true=cfg.theta,
            ln_naive=ln_naive, se_naive=se_naive,
            ln_lam=ln_lam, se_lam=se_lam,
            lam=chosen_lam, feasible=bool(feasible),
            n=cfg.n, varphi=cfg.varphi, pi=cfg.pi, theta=cfg.theta,
            p_sel=cfg.p_sel, q_sel=cfg.q_sel, detK=det_val,
            a_hat=a_hat, b_hat=b_hat, c_hat=c_hat, d_hat=d_hat
        ))

    return pd.DataFrame(rows)


def summarize(rep_df: pd.DataFrame, method: str) -> pd.Series:
    """
    method in {"naive","lam"}.
    """
    est = rep_df[f"ln_{method}"]
    se = rep_df[f"se_{method}"]
    true = rep_df["ln_true"]

    bias = float((est - true).mean())
    rmse = float(np.sqrt(np.mean((est - true) ** 2)))
    cov = float(np.mean((est - 1.96 * se <= true) & (est + 1.96 * se >= true)))

    ridge_frac = float((rep_df["lam"] > 0).mean()) if method == "lam" else float("nan")
    infeas_frac = float((~rep_df["feasible"]).mean()) if method == "lam" else float("nan")
    dK = float(rep_df["detK"].iloc[0])

    return pd.Series(dict(bias=bias, rmse=rmse, coverage=cov,
                          ridge_frac=ridge_frac, infeas_frac=infeas_frac, detK=dK))


# -------------------------
# Experiments (15 points)
# -------------------------
def sweep_symmetric_psel(base: Config, m_grid: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Symmetric sweep (Fig1):
        q_sel = p_sel = 1 - m
        vary m over m_grid (length 15)
        x-axis uses m = 1 - p_sel
    """
    rep_all = []
    summ_all = []

    for m in m_grid:
        p = float(1.0 - m)
        cfg = Config(**{**base.__dict__, "p_sel": p, "q_sel": p})
        rep = run_cell(cfg)
        rep["m"] = float(m)
        rep_all.append(rep)

        sn = summarize(rep, "naive")
        sl = summarize(rep, "lam")

        s = pd.DataFrame([
            dict(method="naive", p_sel=p, q_sel=p, m=float(m), **sn.to_dict()),
            dict(method="lambda-OR", p_sel=p, q_sel=p, m=float(m), **sl.to_dict()),
        ])
        summ_all.append(s)

    return pd.concat(rep_all, ignore_index=True), pd.concat(summ_all, ignore_index=True)


def sweep_qsel_fixed_psel(base: Config, p_fixed: float, x_grid: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    q_sel sweep (Fig2):
        fix p_sel = p_fixed
        vary x = 1 - q_sel over x_grid (length 15)
        x-axis uses x
    """
    rep_all = []
    summ_all = []

    for x in x_grid:
        q = float(1.0 - x)
        cfg = Config(**{**base.__dict__, "p_sel": float(p_fixed), "q_sel": q})
        rep = run_cell(cfg)
        rep["x"] = float(x)
        rep_all.append(rep)

        sn = summarize(rep, "naive")
        sl = summarize(rep, "lam")

        s = pd.DataFrame([
            dict(method="naive", p_sel=float(p_fixed), q_sel=q, x=float(x), **sn.to_dict()),
            dict(method="lambda-OR", p_sel=float(p_fixed), q_sel=q, x=float(x), **sl.to_dict()),
        ])
        summ_all.append(s)

    return pd.concat(rep_all, ignore_index=True), pd.concat(summ_all, ignore_index=True)


def sweep_n(
    base: Config,
    n_list: List[int],
    p_sel: float,
    q_sel: float,
    theta: float = math.log(2),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sensitivity sweep over sample size n at fixed (p_sel, q_sel, theta).
    """
    rep_all = []
    summ_all = []

    for n in n_list:
        cfg = Config(**{**base.__dict__, "n": int(n), "theta": float(theta), "p_sel": float(p_sel), "q_sel": float(q_sel)})
        rep = run_cell(cfg)
        rep["n_sweep"] = int(n)
        rep_all.append(rep)

        sn = summarize(rep, "naive")
        sl = summarize(rep, "lam")

        s = pd.DataFrame([
            dict(method="naive", n=int(n), theta=float(theta), p_sel=float(p_sel), q_sel=float(q_sel), **sn.to_dict()),
            dict(method="lambda-OR", n=int(n), theta=float(theta), p_sel=float(p_sel), q_sel=float(q_sel), **sl.to_dict()),
        ])
        summ_all.append(s)

    return pd.concat(rep_all, ignore_index=True), pd.concat(summ_all, ignore_index=True)


def sweep_theta(
    base: Config,
    theta_list: List[float],
    p_sel: float,
    q_sel: float,
    n: int = 20000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sensitivity sweep over true log-OR theta at fixed (p_sel, q_sel, n).
    """
    rep_all = []
    summ_all = []

    for th in theta_list:
        cfg = Config(**{**base.__dict__, "n": int(n), "theta": float(th), "p_sel": float(p_sel), "q_sel": float(q_sel)})
        rep = run_cell(cfg)
        rep["theta_sweep"] = float(th)
        rep_all.append(rep)

        sn = summarize(rep, "naive")
        sl = summarize(rep, "lam")

        s = pd.DataFrame([
            dict(method="naive", n=int(n), theta=float(th), p_sel=float(p_sel), q_sel=float(q_sel), **sn.to_dict()),
            dict(method="lambda-OR", n=int(n), theta=float(th), p_sel=float(p_sel), q_sel=float(q_sel), **sl.to_dict()),
        ])
        summ_all.append(s)

    return pd.concat(rep_all, ignore_index=True), pd.concat(summ_all, ignore_index=True)


def plot_three_panel(
    summ_df: pd.DataFrame,
    x_key: str,
    x_levels: List[float],
    x_label: str,
    outpath: str,
    xscale: str = "linear",
    x_ticklabels: List[str] | None = None,
) -> None:
    """
    3-panel figure: Bias, RMSE, Coverage. (No det(K) panel because it is constant here.)
    """
    x_levels = list(x_levels)

    def extract(metric, method):
        vals = []
        for xv in x_levels:
            v = summ_df[(summ_df[x_key] == xv) & (summ_df["method"] == method)][metric]
            vals.append(float(v.iloc[0]))
        return vals

    bias_naive = extract("bias", "naive")
    bias_lam   = extract("bias", "lambda-OR")
    rmse_naive = extract("rmse", "naive")
    rmse_lam   = extract("rmse", "lambda-OR")
    cov_naive  = extract("coverage", "naive")
    cov_lam    = extract("coverage", "lambda-OR")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))

    ax = axes[0]
    ax.plot(x_levels, bias_naive, "o-", label="Naive OR")
    ax.plot(x_levels, bias_lam, "s-", label=r"$\lambda$-OR")
    ax.set_xlabel(x_label); ax.set_ylabel("Bias"); ax.set_title("a.")
    ax.set_xscale(xscale)

    ax = axes[1]
    ax.plot(x_levels, rmse_naive, "o-", label="Naive OR")
    ax.plot(x_levels, rmse_lam, "s-", label=r"$\lambda$-OR")
    ax.set_xlabel(x_label); ax.set_ylabel("RMSE"); ax.set_title("b.")
    ax.set_xscale(xscale)

    ax = axes[2]
    ax.plot(x_levels, cov_naive, "o-", label="Naive OR")
    ax.plot(x_levels, cov_lam, "s-", label=r"$\lambda$-OR")
    ax.set_xlabel(x_label); ax.set_ylabel("95% CI coverage"); ax.set_ylim(0.0, 1.02); ax.set_title("c.")
    ax.set_xscale(xscale)

    if x_ticklabels is not None:
        for ax in axes:
            ax.set_xticks(x_levels)
            ax.set_xticklabels(x_ticklabels)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Plotting (4 panels)
# -------------------------
def plot_four_panel(
    summ_df: pd.DataFrame,
    x_key: str,
    x_levels: List[float],
    x_label: str,
    outpath: str
) -> None:
    """
    4-panel figure: (a) Bias, (b) RMSE, (c) Coverage, (d) det(K).
    det(K) uses the single value stored per grid-point (identical for both methods).
    """
    x_levels = list(x_levels)

    def extract(metric, method):
        vals = []
        for xv in x_levels:
            v = summ_df[(summ_df[x_key] == xv) & (summ_df["method"] == method)][metric]
            vals.append(float(v.iloc[0]))
        return vals

    bias_naive = extract("bias", "naive")
    bias_lam   = extract("bias", "lambda-OR")
    rmse_naive = extract("rmse", "naive")
    rmse_lam   = extract("rmse", "lambda-OR")
    cov_naive  = extract("coverage", "naive")
    cov_lam    = extract("coverage", "lambda-OR")

    det_vals = []
    for xv in x_levels:
        rows = summ_df[summ_df[x_key] == xv]
        det_vals.append(float(rows["detK"].iloc[0]))

    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.2))

    ax = axes[0]
    ax.plot(x_levels, bias_naive, "o-", label="Naive OR")
    ax.plot(x_levels, bias_lam, "s-", label=r"$\lambda$-OR")
    ax.set_xlabel(x_label); ax.set_ylabel("Bias"); ax.set_title("a.")

    ax = axes[1]
    ax.plot(x_levels, rmse_naive, "o-", label="Naive OR")
    ax.plot(x_levels, rmse_lam, "s-", label=r"$\lambda$-OR")
    ax.set_xlabel(x_label); ax.set_ylabel("RMSE"); ax.set_title("b.")

    ax = axes[2]
    ax.plot(x_levels, cov_naive, "o-", label="Naive OR")
    ax.plot(x_levels, cov_lam, "s-", label=r"$\lambda$-OR")
    ax.set_xlabel(x_label); ax.set_ylabel("95% CI coverage"); ax.set_ylim(0.0, 1.02); ax.set_title("c.")

    ax = axes[3]
    ax.plot(x_levels, det_vals, "k--", linewidth=2.0,
            label=r"$\det(K)=p_{\mathrm{sel}}+q_{\mathrm{sel}}-1$")
    ax.set_xlabel(x_label); ax.set_ylabel(r"$\det(K)$"); ax.set_title("d.")
    pad = 0.05
    ax.set_ylim(min(det_vals) - pad, max(det_vals) + pad)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    base = Config()

    # 15-point grids
    m_grid = np.linspace(0.02, 0.35, 15)     # symmetric: m = 1 - p_sel
    x_grid = np.linspace(0.02, 0.65, 15)     # q-sweep: x = 1 - q_sel
    p_fixed = 0.95

    rep_symm, summ_symm = sweep_symmetric_psel(base, m_grid=m_grid)
    rep_symm.to_csv(os.path.join(OUTPUT_DIR, "replicates_symm.csv"), index=False)
    summ_symm.to_csv(os.path.join(OUTPUT_DIR, "summary_symm.csv"), index=False)
    plot_four_panel(
        summ_df=summ_symm,
        x_key="m",
        x_levels=list(m_grid),
        x_label=r"Misclassification rate $(1-p_{\mathrm{sel}})$ (with $q_{\mathrm{sel}}=p_{\mathrm{sel}}$)",
        outpath=os.path.join(OUTPUT_DIR, "Fig1_symm_4panel.png"),
    )

    rep_q, summ_q = sweep_qsel_fixed_psel(base, p_fixed=p_fixed, x_grid=x_grid)
    rep_q.to_csv(os.path.join(OUTPUT_DIR, "replicates_qsweep.csv"), index=False)
    summ_q.to_csv(os.path.join(OUTPUT_DIR, "summary_qsweep.csv"), index=False)
    plot_four_panel(
        summ_df=summ_q,
        x_key="x",
        x_levels=list(x_grid),
        x_label=rf"Misclassification rate $(1-q_{{\mathrm{{sel}}}})$ (with $p_{{\mathrm{{sel}}}}={p_fixed:.2f}$)",
        outpath=os.path.join(OUTPUT_DIR, "Fig2_qsweep_4panel.png"),
    )

    print("Done. Outputs written to:", os.path.abspath(OUTPUT_DIR))
    print("Main figures:")
    print("  - Fig1_symm_4panel.png")
    print("  - Fig2_qsweep_4panel.png")

    # -------------------------
    # Sensitivity to n and theta
    # -------------------------
    p_sens = 0.95
    q_sens = 0.65

    n_list = [5000, 10000, 20000, 50000]
    rep_n, summ_n = sweep_n(base, n_list=n_list, p_sel=p_sens, q_sel=q_sens, theta=math.log(2))
    rep_n.to_csv(os.path.join(OUTPUT_DIR, "replicates_nsweep.csv"), index=False)
    summ_n.to_csv(os.path.join(OUTPUT_DIR, "summary_nsweep.csv"), index=False)
    plot_three_panel(
        summ_df=summ_n,
        x_key="n",
        x_levels=[float(n) for n in n_list],
        x_label=r"Sample size $n$",
        outpath=os.path.join(OUTPUT_DIR, "Fig3_nsweep_3panel.png"),
        xscale="log",
        x_ticklabels=[str(n) for n in n_list],
    )

    theta_list = [math.log(1.25), math.log(1.5), math.log(2.0), math.log(3.0), math.log(4.0)]
    rep_th, summ_th = sweep_theta(base, theta_list=theta_list, p_sel=p_sens, q_sel=q_sens, n=20000)
    rep_th.to_csv(os.path.join(OUTPUT_DIR, "replicates_thetasweep.csv"), index=False)
    summ_th.to_csv(os.path.join(OUTPUT_DIR, "summary_thetasweep.csv"), index=False)
    plot_three_panel(
        summ_df=summ_th,
        x_key="theta",
        x_levels=[float(t) for t in theta_list],
        x_label=r"True log-OR $\theta$",
        outpath=os.path.join(OUTPUT_DIR, "Fig4_thetasweep_3panel.png"),
        xscale="linear",
        x_ticklabels=[r"$\log(1.25)$", r"$\log(1.5)$", r"$\log(2)$", r"$\log(3)$", r"$\log(4)$"],
    )


if __name__ == "__main__":
    main()
