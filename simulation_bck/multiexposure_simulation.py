#!/usr/bin/env python3
"""
Multi-exposure simulation for simulated volcano plots (naive OR vs lambda-OR)
============================================================================

Changes implemented (per your request):
1) Use a less extreme tail-selection default so selected n is not tiny.
   - alpha_hi = alpha_lo = 0.10 by default (keeps ~20% of sample).
   You can still set them to 0.01 if you want, but expect terrible FDR at p<0.05.

2) Volcano plots include multiple-testing reference lines:
   - nominal p = 0.05
   - Bonferroni p = 0.05 / m

3) Report discoveries and FDR at:
   - nominal p<0.05
   - Bonferroni
   - BH (FDR q=0.05)

4) Score construction uses a fraction of true signal features
   (score_signal_frac) so selection is informative.

Note: This script still uses the simple Wald SE for lambda-OR computed from the
corrected counts. For a publication-grade simulation, you should replace the
lambda SE with the delta-method variance you describe in the manuscript.
The multiple-testing thresholds are still necessary regardless.

Outputs:
  ./lambdaor_outputs/
    volcano_sim_results.csv
    volcano_sim_naive.png
    volcano_sim_lambda.png
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng, SeedSequence


OUTPUT_DIR = os.environ.get("LAMBDAOR_SIM_OUTPUT", "./lambdaor_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Config:
    # Sizes
    n: int = 20000
    nval: int = 20000

    # Multi-exposure design
    m: int = 1000
    m_signal: int = 500
    theta_signal: float = math.log(1.2)
    r_min: float = 0.005
    r_max: float = 0.15

    # Outcome prevalence control
    pi_target: float = 0.20

    # Predictive score model S=f(X) (imperfect)
    m_score: int = 200
    score_beta_scale: float = 0.35
    score_noise_sd: float = 1.0
    score_signal_frac: float = 0.5  # fraction of score features drawn from true signals

    # Two-gate selection (changed default to avoid n≈400)
    alpha_hi: float = 0.10
    alpha_lo: float = 0.10

    # Selection-conditional label noise targets (applied within I=1)
    p_sel_target: float = 0.8
    q_sel_target: float = 0.7

    # Ridge feasibility grid for correction
    lam_grid: Tuple[float, ...] = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
    eps_pos: float = 1e-9

    # Naive OR continuity correction
    cc: float = 0.5

    # RNG
    seed: int = 20251005


# -------------------------
# Latent DGP: X and Y
# -------------------------
def gen_latent_XY(cfg: Config, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate:
      X: (n,m) binary exposures
      is_signal: (m,) boolean truth mask
      Y: (n,) latent outcome via logistic model from signal exposures
    """
    m = cfg.m
    idx = np.arange(m)
    rng.shuffle(idx)
    signal_idx = np.sort(idx[: cfg.m_signal])

    is_signal = np.zeros(m, dtype=bool)
    is_signal[signal_idx] = True

    r = rng.uniform(cfg.r_min, cfg.r_max, size=m)
    U = rng.random((cfg.n, m))
    X = (U < r[None, :]).astype(np.int8)

    lin_part = cfg.theta_signal * X[:, is_signal].astype(float).sum(axis=1)

    def prevalence_for_intercept(b0: float) -> float:
        return float(sigmoid(b0 + lin_part).mean())

    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if prevalence_for_intercept(mid) > cfg.pi_target:
            hi = mid
        else:
            lo = mid
    b0 = 0.5 * (lo + hi)

    pY = sigmoid(b0 + lin_part)
    Y = rng.binomial(1, pY).astype(np.int64)

    return X, is_signal, Y


# -------------------------
# Predictive score and selection
# -------------------------
def gen_score(cfg: Config, X: np.ndarray, is_signal: np.ndarray, rng) -> np.ndarray:
    """
    Generate an imperfect predictive score S in [0,1] that uses some true signals.
    """
    m = X.shape[1]
    m_score = min(cfg.m_score, m)

    sig_idx = np.where(is_signal)[0]
    null_idx = np.where(~is_signal)[0]

    k_sig = min(len(sig_idx), int(round(cfg.score_signal_frac * m_score)))
    k_null = m_score - k_sig

    score_sig = rng.choice(sig_idx, size=k_sig, replace=False) if k_sig > 0 else np.array([], dtype=int)
    score_null = rng.choice(null_idx, size=k_null, replace=False) if k_null > 0 else np.array([], dtype=int)

    score_idx = np.sort(np.concatenate([score_sig, score_null]))
    w = rng.normal(0.0, cfg.score_beta_scale, size=m_score)

    lin = X[:, score_idx].astype(float) @ w
    lin_noisy = lin + rng.normal(0.0, cfg.score_noise_sd, size=X.shape[0])
    return sigmoid(lin_noisy).astype(float)


def two_gate_select(S: np.ndarray, alpha_hi: float, alpha_lo: float) -> Tuple[np.ndarray, float, float]:
    t_hi = float(np.quantile(S, 1.0 - alpha_hi))
    t_lo = float(np.quantile(S, alpha_lo))
    I = (S >= t_hi) | (S <= t_lo)
    return I, t_lo, t_hi


# -------------------------
# Controlled outcome misclassification within selected sample
# -------------------------
def misclassify_within_selected(Y: np.ndarray, I: np.ndarray, p_sel: float, q_sel: float, rng) -> np.ndarray:
    """
    Apply selection-conditional misclassification only to I=1 records:
      P(Ytil=1|Y=1,I=1)=p_sel, P(Ytil=0|Y=0,I=1)=q_sel.
    """
    Ytil = Y.copy().astype(np.int64)

    idx1 = np.where(I & (Y == 1))[0]
    idx0 = np.where(I & (Y == 0))[0]

    flip_1_to_0 = rng.binomial(1, 1.0 - p_sel, size=idx1.size).astype(np.int64)
    Ytil[idx1] = 1 - flip_1_to_0

    flip_0_to_1 = rng.binomial(1, 1.0 - q_sel, size=idx0.size).astype(np.int64)
    Ytil[idx0] = flip_0_to_1

    return Ytil


def estimate_pq(Y: np.ndarray, Ytil: np.ndarray, I: np.ndarray) -> Tuple[float, float]:
    Ys = Y[I]
    Yt = Ytil[I]

    denom1 = max(int((Ys == 1).sum()), 1)
    denom0 = max(int((Ys == 0).sum()), 1)

    p_hat = float(((Yt == 1) & (Ys == 1)).sum() / denom1)
    q_hat = float(((Yt == 0) & (Ys == 0)).sum() / denom0)
    return p_hat, q_hat


# -------------------------
# OR computations
# -------------------------
def table_counts(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    a = float(((x == 1) & (y == 1)).sum())
    b = float(((x == 0) & (y == 1)).sum())
    c = float(((x == 1) & (y == 0)).sum())
    d = float(((x == 0) & (y == 0)).sum())
    return a, b, c, d


def log_or_and_se(a: float, b: float, c: float, d: float) -> Tuple[float, float]:
    ln = math.log((a * d) / (b * c))
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    return ln, se


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def pval_from_z(z: float) -> float:
    return 2.0 * norm_sf(abs(z))


def naive_stats(x: np.ndarray, ytil: np.ndarray, cc: float) -> Tuple[float, float, float]:
    a, b, c, d = table_counts(x, ytil)
    a += cc; b += cc; c += cc; d += cc
    ln, se = log_or_and_se(a, b, c, d)
    p = pval_from_z(ln / se)
    return ln, se, p


def lambda_corrected_stats(
    x: np.ndarray,
    ytil: np.ndarray,
    p_sel: float,
    q_sel: float,
    lam_grid: Tuple[float, ...],
    eps_pos: float
) -> Tuple[float, float, float, float, bool]:
    """
    Minimal-feasible ridge correction:
      Tcorr = Ttil @ (K + lam I)^(-1)
    with Ttil built from observed (x, ytil) on selected sample.
    """
    a_t, b_t, c_t, d_t = table_counts(x, ytil)
    Ttil = np.array([[a_t, c_t],
                     [b_t, d_t]], dtype=float)

    K = np.array([[p_sel, 1.0 - p_sel],
                  [1.0 - q_sel, q_sel]], dtype=float)

    chosen = None
    chosen_lam = None

    for lam in lam_grid:
        Kreg = K + lam * np.eye(2)
        try:
            Kinv = np.linalg.inv(Kreg)
        except np.linalg.LinAlgError:
            Kinv = np.linalg.pinv(Kreg)

        Tcorr = Ttil @ Kinv
        a = float(Tcorr[0, 0])
        c = float(Tcorr[0, 1])
        b = float(Tcorr[1, 0])
        d = float(Tcorr[1, 1])

        feasible = (a > eps_pos) and (b > eps_pos) and (c > eps_pos) and (d > eps_pos)
        if feasible:
            chosen = (a, b, c, d)
            chosen_lam = float(lam)
            break

    if chosen is None:
        lam = lam_grid[-1]
        Kreg = K + lam * np.eye(2)
        try:
            Kinv = np.linalg.inv(Kreg)
        except np.linalg.LinAlgError:
            Kinv = np.linalg.pinv(Kreg)
        Tcorr = Ttil @ Kinv
        a = max(float(Tcorr[0, 0]), eps_pos)
        c = max(float(Tcorr[0, 1]), eps_pos)
        b = max(float(Tcorr[1, 0]), eps_pos)
        d = max(float(Tcorr[1, 1]), eps_pos)
        chosen = (a, b, c, d)
        chosen_lam = float(lam)
        feasible = False
    else:
        a, b, c, d = chosen

    ln, se = log_or_and_se(a, b, c, d)
    p = pval_from_z(ln / se)
    return ln, se, p, chosen_lam, bool(feasible)


# -------------------------
# Multiple-testing helpers
# -------------------------
def bh_threshold(pvals: np.ndarray, q: float = 0.05) -> Optional[float]:
    """
    Benjamini-Hochberg threshold for given q. Returns p-threshold or None.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    crit = (np.arange(1, m + 1) / m) * q
    ok = p_sorted <= crit
    if not np.any(ok):
        return None
    k = int(np.max(np.where(ok)[0]))
    return float(p_sorted[k])


def summarize_discoveries(pvals: np.ndarray, truth: np.ndarray, alpha: float) -> Tuple[int, int, float]:
    sel = pvals < alpha
    tp = int((truth & sel).sum())
    fp = int((~truth & sel).sum())
    fdr = 0.0 if (tp + fp) == 0 else float(fp) / float(tp + fp)
    return tp, fp, fdr


# -------------------------
# Plotting
# -------------------------
def volcano_plot(
    df: pd.DataFrame,
    x_col: str,
    p_col: str,
    xlab: str,
    outpath: str,
    alpha_nom: float,
    alpha_bonf: float,
    alpha_bh: Optional[float]
) -> None:
    x = df[x_col].to_numpy(float)
    p = df[p_col].to_numpy(float)
    y = -np.log10(np.clip(p, 1e-300, 1.0))

    truth = df["is_signal"].to_numpy(bool)
    null = ~truth

    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.scatter(x[null], y[null], s=8, alpha=0.55, label="null")
    ax.scatter(x[truth], y[truth], s=10, alpha=0.75, label="non-null")

    # Reference lines
    ax.axhline(-math.log10(alpha_nom), linewidth=1.2, linestyle="--")
    ax.axhline(-math.log10(alpha_bonf), linewidth=1.2, linestyle="-")
    if alpha_bh is not None:
        ax.axhline(-math.log10(alpha_bh), linewidth=1.2, linestyle=":")

    ax.set_xlabel(xlab)
    ax.set_ylabel(r"significance $(-\log_{10} p)$")
    ax.legend(frameon=False)

    # Small caption inside plot
    txt = f"p<0.05 (--)   Bonferroni (-)\n"
    if alpha_bh is not None:
        txt += "BH q=0.05 (:)"
    else:
        txt += "BH q=0.05: none"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    cfg = Config()
    rng = default_rng(SeedSequence(cfg.seed))

    # --- Analysis sample ---
    X, is_signal, Y = gen_latent_XY(cfg, rng)
    S = gen_score(cfg, X, is_signal, rng)
    I, t_lo, t_hi = two_gate_select(S, cfg.alpha_hi, cfg.alpha_lo)

    Ytil = misclassify_within_selected(Y, I, cfg.p_sel_target, cfg.q_sel_target, rng)
    X_sel = X[I, :]
    Ytil_sel = Ytil[I]

    # --- Validation sample (estimate p_sel, q_sel) ---
    cfg_val = Config(**{**cfg.__dict__})
    cfg_val.n = cfg.nval

    Xv, is_signal_v, Yv = gen_latent_XY(cfg_val, rng)
    Sv = gen_score(cfg, Xv, is_signal_v, rng)
    Iv = (Sv >= t_hi) | (Sv <= t_lo)

    Ytil_v = misclassify_within_selected(Yv, Iv, cfg.p_sel_target, cfg.q_sel_target, rng)
    p_sel_hat, q_sel_hat = estimate_pq(Yv, Ytil_v, Iv)
    detK = p_sel_hat + q_sel_hat - 1.0

    # --- Per-exposure stats on selected sample ---
    rows: List[Dict] = []
    for j in range(cfg.m):
        xj = X_sel[:, j].astype(np.int64)

        ln_naive, se_naive, p_naive = naive_stats(xj, Ytil_sel, cfg.cc)
        ln_lam, se_lam, p_lam, lam_used, feasible = lambda_corrected_stats(
            xj, Ytil_sel, p_sel_hat, q_sel_hat, cfg.lam_grid, cfg.eps_pos
        )

        rows.append(dict(
            j=j,
            is_signal=bool(is_signal[j]),
            ln_true=float(cfg.theta_signal) if is_signal[j] else 0.0,
            ln_naive=ln_naive, se_naive=se_naive, p_naive=p_naive,
            ln_lam=ln_lam, se_lam=se_lam, p_lam=p_lam,
            lam=lam_used, feasible=feasible
        ))

    df = pd.DataFrame(rows)

    # Sanity check truth mask
    truth = df["is_signal"].to_numpy(bool)
    if int(truth.sum()) != int(cfg.m_signal):
        print(f"WARNING: df truth sum={int(truth.sum())} != cfg.m_signal={int(cfg.m_signal)}")

    out_csv = os.path.join(OUTPUT_DIR, "volcano_sim_results.csv")
    df.to_csv(out_csv, index=False)

    # Thresholds
    alpha_nom = 0.05
    alpha_bonf = 0.05 / float(cfg.m)

    # BH thresholds per method
    bh_naive = bh_threshold(df["p_naive"].to_numpy(float), q=0.05)
    bh_lam = bh_threshold(df["p_lam"].to_numpy(float), q=0.05)

    # Summaries
    p_naive = df["p_naive"].to_numpy(float)
    p_lam = df["p_lam"].to_numpy(float)

    tp_n, fp_n, fdr_n = summarize_discoveries(p_naive, truth, alpha_nom)
    tp_nb, fp_nb, fdr_nb = summarize_discoveries(p_naive, truth, alpha_bonf)
    tp_nbh, fp_nbh, fdr_nbh = (0, 0, 0.0) if bh_naive is None else summarize_discoveries(p_naive, truth, bh_naive)

    tp_l, fp_l, fdr_l = summarize_discoveries(p_lam, truth, alpha_nom)
    tp_lb, fp_lb, fdr_lb = summarize_discoveries(p_lam, truth, alpha_bonf)
    tp_lbh, fp_lbh, fdr_lbh = (0, 0, 0.0) if bh_lam is None else summarize_discoveries(p_lam, truth, bh_lam)

    print("Selection thresholds:")
    print(f"  t_lo={t_lo:.6f}, t_hi={t_hi:.6f}, kept={int(I.sum())}/{cfg.n} ({I.mean():.3f})")
    print("Target (selection-conditional) label noise (applied within I=1):")
    print(f"  p_sel_target={cfg.p_sel_target:.4f}, q_sel_target={cfg.q_sel_target:.4f}")
    print("Estimated (selection-conditional) error rates on validation:")
    print(f"  p_sel_hat={p_sel_hat:.4f}, q_sel_hat={q_sel_hat:.4f}, det(K)={detK:.4f}")
    print("Multiple-testing thresholds:")
    print(f"  nominal alpha={alpha_nom:.3g}")
    print(f"  Bonferroni alpha={alpha_bonf:.3g}  (m={cfg.m})")
    print(f"  BH naive alpha={bh_naive if bh_naive is not None else 'none'}")
    print(f"  BH lambda alpha={bh_lam if bh_lam is not None else 'none'}")

    print("Discoveries (naive):")
    print(f"  p<0.05        TP={tp_n}   FP={fp_n}   FDR={fdr_n:.3f}")
    print(f"  Bonferroni    TP={tp_nb}  FP={fp_nb}  FDR={fdr_nb:.3f}")
    if bh_naive is None:
        print("  BH q=0.05     none")
    else:
        print(f"  BH q=0.05     TP={tp_nbh} FP={fp_nbh} FDR={fdr_nbh:.3f}")

    print("Discoveries (lambda):")
    print(f"  p<0.05        TP={tp_l}   FP={fp_l}   FDR={fdr_l:.3f}")
    print(f"  Bonferroni    TP={tp_lb}  FP={fp_lb}  FDR={fdr_lb:.3f}")
    if bh_lam is None:
        print("  BH q=0.05     none")
    else:
        print(f"  BH q=0.05     TP={tp_lbh} FP={fp_lbh} FDR={fdr_lbh:.3f}")

    print("Outputs written to:", os.path.abspath(OUTPUT_DIR))
    print("  ", out_csv)

    # Plots (use method-specific BH line)
    volcano_plot(
        df, "ln_naive", "p_naive",
        xlab=r"effect size (log OR, naive)",
        outpath=os.path.join(OUTPUT_DIR, "volcano_sim_naive.png"),
        alpha_nom=alpha_nom,
        alpha_bonf=alpha_bonf,
        alpha_bh=bh_naive
    )
    volcano_plot(
        df, "ln_lam", "p_lam",
        xlab=r"effect size (log $\lambda$-OR)",
        outpath=os.path.join(OUTPUT_DIR, "volcano_sim_lambda.png"),
        alpha_nom=alpha_nom,
        alpha_bonf=alpha_bonf,
        alpha_bh=bh_lam
    )


if __name__ == "__main__":
    main()
