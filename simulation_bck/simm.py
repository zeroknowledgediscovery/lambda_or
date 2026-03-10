#!/usr/bin/env python3
"""
Multi-exposure scan with selection-conditional outcome misclassification.
Generates volcano plots for:
  1) naive log-OR computed on misclassified outcome (within selected set)
  2) lambda-OR (ridge-stabilized inversion of misclassification operator)

Core design (interpretable DGP, no classifier training):
  Latent severity Z -> true outcome Y via logistic link
  Triage score S = Z + noise, selection I by ROC-targeted threshold on S predicting Y
  Exposures X_j:
    - "signal" features depend on Z (thus associated with Y)
    - "null" features independent of Z
  Misclassification applied only within I=1 with (p_sel, q_sel)

p-values:
  Uses bootstrap-based Wald z-tests for BOTH naive and lambda-OR for comparability.
  (Fisher exact p-values for naive are optionally computed for reference.)

Outputs:
  - volcano_naive.png
  - volcano_lambda.png
  - scan_results.csv
  - summary printed to stdout
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Optional

from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import roc_curve


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def bh_threshold(pvals: np.ndarray, q: float = 0.05) -> Optional[float]:
    """Benjamini-Hochberg threshold. Returns largest p s.t. p <= (k/m) q, or None if no rejections."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    thresh_line = (np.arange(1, m + 1) / m) * q
    ok = p_sorted <= thresh_line
    if not np.any(ok):
        return None
    kmax = np.max(np.where(ok)[0])
    return float(p_sorted[kmax])

def bonferroni_alpha(alpha: float, m: int) -> float:
    return alpha / float(m)

def roc_threshold_for_specificity(y_true: np.ndarray, score: np.ndarray, target_spec: float) -> float:
    """
    Choose threshold t such that specificity approximately >= target_spec.
    Using sklearn roc_curve: returns fpr, tpr, thresholds where predicted positive is score >= thr.
    specificity = 1 - fpr.
    We choose the threshold with minimal |spec - target_spec|, preferring spec >= target_spec.
    """
    fpr, tpr, thr = roc_curve(y_true, score)
    spec = 1.0 - fpr
    # prefer thresholds meeting spec >= target_spec
    meets = np.where(spec >= target_spec)[0]
    if meets.size > 0:
        # among those, choose closest specificity to target_spec (highest recall among close ones)
        idx = meets[np.argmin(np.abs(spec[meets] - target_spec))]
        return float(thr[idx])
    # fallback: closest overall
    idx = int(np.argmin(np.abs(spec - target_spec)))
    return float(thr[idx])

def roc_threshold_for_sensitivity(y_true: np.ndarray, score: np.ndarray, target_sens: float) -> float:
    fpr, tpr, thr = roc_curve(y_true, score)
    sens = tpr
    meets = np.where(sens >= target_sens)[0]
    if meets.size > 0:
        idx = meets[np.argmin(np.abs(sens[meets] - target_sens))]
        return float(thr[idx])
    idx = int(np.argmin(np.abs(sens - target_sens)))
    return float(thr[idx])

def compute_2x2_counts(x: np.ndarray, y: np.ndarray) -> Tuple[int, int, int, int]:
    """
    2x2 table with rows = X (0,1) and cols = Y (0,1):
      a = n(X=1,Y=1)
      b = n(X=1,Y=0)
      c = n(X=0,Y=1)
      d = n(X=0,Y=0)
    """
    x = x.astype(int)
    y = y.astype(int)
    a = int(np.sum((x == 1) & (y == 1)))
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    d = int(np.sum((x == 0) & (y == 0)))
    return a, b, c, d

def log_or_from_counts(a: float, b: float, c: float, d: float, eps: float = 1e-12) -> float:
    """log( (a*d)/(b*c) ) with safe guards."""
    a = max(a, eps)
    b = max(b, eps)
    c = max(c, eps)
    d = max(d, eps)
    return float(np.log((a * d) / (b * c)))

def apply_misclassification(y: np.ndarray, p: float, q: float, rng: np.random.Generator) -> np.ndarray:
    """
    Outcome misclassification:
      P(Ytil=1 | Y=1) = p   (sensitivity)
      P(Ytil=0 | Y=0) = q   (specificity)
    """
    y = y.astype(int)
    u = rng.random(y.shape[0])
    ytil = y.copy()
    # True positives retained with prob p; otherwise flip to 0
    mask1 = (y == 1)
    ytil[mask1] = (u[mask1] <= p).astype(int)
    # True negatives retained as 0 with prob q; otherwise flip to 1
    mask0 = (y == 0)
    ytil[mask0] = (u[mask0] > q).astype(int)
    return ytil

def correction_matrix(p: float, q: float) -> np.ndarray:
    """
    K maps true outcome counts -> observed outcome counts, column-wise.
    For a fixed X level:
      [obs Y=0]   [ q      (1-p) ] [true Y=0]
      [obs Y=1] = [ (1-q)   p    ] [true Y=1]
    """
    return np.array([[q, 1.0 - p],
                     [1.0 - q, p]], dtype=float)

def lambda_correct_table(obs_table: np.ndarray, p: float, q: float, lam: float) -> np.ndarray:
    """
    obs_table: shape (2,2) as:
      rows X in {0,1}, cols observed Ytil in {0,1}
    Returns corrected (estimated true) table with same shape.
    Uses ridge-stabilized inverse (K + lam I)^{-1} on the outcome axis, applied per X row.
    """
    # outcome axis vectors are [Y=0, Y=1] per X row
    K = correction_matrix(p, q)
    inv = np.linalg.inv(K + lam * np.eye(2))
    # apply to each X row: true_counts = inv @ obs_counts_vector
    corr = np.zeros_like(obs_table, dtype=float)
    for xi in (0, 1):
        obs_vec = np.array([obs_table[xi, 0], obs_table[xi, 1]], dtype=float)
        tru_vec = inv @ obs_vec
        # clip to small positive to avoid logOR blowups and sign flips from numerical noise
        tru_vec = np.clip(tru_vec, 1e-6, None)
        corr[xi, 0] = tru_vec[0]
        corr[xi, 1] = tru_vec[1]
    return corr

def logor_naive_from_xy(x: np.ndarray, ytil: np.ndarray, ha: float = 0.5) -> float:
    a, b, c, d = compute_2x2_counts(x, ytil)
    return log_or_from_counts(a + ha, b + ha, c + ha, d + ha)

def logor_lambda_from_xy(x: np.ndarray, ytil: np.ndarray, p: float, q: float, lam: float, ha: float = 0.0) -> float:
    """
    Build observed 2x2 table, correct it, then compute log-OR on corrected table.
    We do not add HA by default after correction (ha=0.0) because we already clip.
    """
    a, b, c, d = compute_2x2_counts(x, ytil)
    # obs_table rows X=0,1 ; cols Ytil=0,1
    obs_table = np.array([[d, c],   # X=0: Ytil=0 -> d, Ytil=1 -> c
                          [b, a]],  # X=1: Ytil=0 -> b, Ytil=1 -> a
                         dtype=float)
    corr = lambda_correct_table(obs_table, p, q, lam)
    # corrected counts:
    # X=1,Y=1 is corr[1,1], etc.
    a2 = corr[1, 1] + ha
    b2 = corr[1, 0] + ha
    c2 = corr[0, 1] + ha
    d2 = corr[0, 0] + ha
    return log_or_from_counts(a2, b2, c2, d2)

def bootstrap_se_and_pvalue(
    x: np.ndarray,
    ytil: np.ndarray,
    stat_fn,
    rng: np.random.Generator,
    B: int = 200
) -> Tuple[float, float, float]:
    """
    Bootstrap SE for stat_fn(x,ytil) within given sample.
    Returns (theta_hat, se, p_two_sided) using z = theta_hat / se.
    """
    n = x.shape[0]
    theta_hat = float(stat_fn(x, ytil))
    if B <= 1 or n <= 5:
        return theta_hat, np.nan, np.nan

    thetas = np.empty(B, dtype=float)
    idx = np.arange(n)
    for b in range(B):
        samp = rng.choice(idx, size=n, replace=True)
        thetas[b] = float(stat_fn(x[samp], ytil[samp]))
    se = float(np.std(thetas, ddof=1))
    if se <= 0 or not np.isfinite(se):
        return theta_hat, se, np.nan
    z = theta_hat / se
    p = float(2.0 * norm.sf(np.abs(z)))
    return theta_hat, se, p


# ----------------------------
# Main simulation
# ----------------------------

@dataclass
class SimConfig:
    n: int = 20000
    m: int = 1000
    n_signal: int = 60

    # latent outcome model
    eta0: float = -0.2
    eta1: float = 1.2

    # triage score noise and selection via ROC
    score_noise: float = 1.0
    selection_mode: str = "spec"  # "spec" or "sens"
    target_spec: float = 0.95
    target_sens: float = 0.80
    two_gate: bool = False
    keep_frac_two_gate: float = 0.20  # used only if two_gate=True and you want quantile-based fallback

    # exposure generation
    null_pi_low: float = 0.05
    null_pi_high: float = 0.25
    alpha_signal: float = -1.0
    gamma_signal: float = 1.0  # strength of Z->X association for signals

    # selection-conditional label noise within I=1
    p_sel: float = 0.80
    q_sel: float = 0.70

    # lambda ridge
    lam: float = 0.05

    # bootstrap
    boot_B: int = 200

    # multiple testing
    alpha: float = 0.05

    seed: int = 1
    out_dir: str = "./lambdaor_outputs_latent"


def run(cfg: SimConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    ensure_dir(cfg.out_dir)

    # 1) latent severity and true outcome
    Z = rng.normal(0.0, 1.0, size=cfg.n)
    pY = expit(cfg.eta0 + cfg.eta1 * Z)
    Y = (rng.random(cfg.n) < pY).astype(int)

    # 2) triage score and selection thresholds from ROC
    S = Z + rng.normal(0.0, cfg.score_noise, size=cfg.n)

    # split for threshold selection (simple holdout)
    idx = rng.permutation(cfg.n)
    n_val = cfg.n // 3
    val = idx[:n_val]
    trn = idx[n_val:]

    if cfg.selection_mode == "spec":
        t_hi = roc_threshold_for_specificity(Y[val], S[val], cfg.target_spec)
    elif cfg.selection_mode == "sens":
        t_hi = roc_threshold_for_sensitivity(Y[val], S[val], cfg.target_sens)
    else:
        raise ValueError("selection_mode must be 'spec' or 'sens'")

    if not cfg.two_gate:
        I = (S >= t_hi).astype(int)
        t_lo = None
    else:
        # symmetric two-gate: pick low threshold by matching kept fraction on the low tail
        # This is simple and interpretable; if you want ROC-based low gate, target sensitivity with inverted score.
        kept_target = int(round(cfg.keep_frac_two_gate * cfg.n))
        # choose low and high quantiles with total kept approx keep_frac_two_gate
        # keep_frac = q + (1-q2); pick q = keep_frac/2
        q = cfg.keep_frac_two_gate / 2.0
        t_lo = float(np.quantile(S, q))
        t_hi2 = float(np.quantile(S, 1.0 - q))
        t_hi = t_hi2  # override so reporting is consistent
        I = ((S <= t_lo) | (S >= t_hi)).astype(int)

    n_sel = int(I.sum())
    frac_sel = n_sel / cfg.n

    # 3) exposures X: n x m
    # choose which features are signal
    signal_idx = np.sort(rng.choice(cfg.m, size=cfg.n_signal, replace=False))
    is_signal = np.zeros(cfg.m, dtype=bool)
    is_signal[signal_idx] = True

    # null prevalences
    pi = rng.uniform(cfg.null_pi_low, cfg.null_pi_high, size=cfg.m)

    X = np.zeros((cfg.n, cfg.m), dtype=np.int8)

    # null features independent of Z
    null_cols = np.where(~is_signal)[0]
    if null_cols.size > 0:
        U = rng.random((cfg.n, null_cols.size))
        X[:, null_cols] = (U < pi[null_cols][None, :]).astype(np.int8)

    # signal features depend on Z
    sig_cols = np.where(is_signal)[0]
    if sig_cols.size > 0:
        # allow modest variation in alpha across signals for heterogeneity
        alpha_j = cfg.alpha_signal + rng.normal(0.0, 0.3, size=sig_cols.size)
        # probability depends on Z
        pX = expit(alpha_j[None, :] + cfg.gamma_signal * Z[:, None])
        U = rng.random((cfg.n, sig_cols.size))
        X[:, sig_cols] = (U < pX).astype(np.int8)

    # 4) apply misclassification within selected set
    Ytil = Y.copy()
    sel_idx = np.where(I == 1)[0]
    Ytil[sel_idx] = apply_misclassification(Y[sel_idx], cfg.p_sel, cfg.q_sel, rng)

    # 5) scan within selected set only
    Xs = X[sel_idx, :]
    Ys = Y[sel_idx]
    Yt = Ytil[sel_idx]

    # (optional) sanity check of achieved selection-conditional rates if we use latent Y
    # This is for simulation reporting only.
    # p_sel_hat = P(Ytil=1|Y=1,I=1)
    # q_sel_hat = P(Ytil=0|Y=0,I=1)
    mask1 = (Ys == 1)
    mask0 = (Ys == 0)
    p_sel_hat = float(np.mean(Yt[mask1] == 1)) if mask1.any() else np.nan
    q_sel_hat = float(np.mean(Yt[mask0] == 0)) if mask0.any() else np.nan
    detK = (p_sel_hat + q_sel_hat - 1.0) if np.isfinite(p_sel_hat) and np.isfinite(q_sel_hat) else np.nan

    # 6) per-feature stats with bootstrap p-values (naive and lambda)
    naive_logor = np.empty(cfg.m, dtype=float)
    naive_se = np.empty(cfg.m, dtype=float)
    naive_p = np.empty(cfg.m, dtype=float)

    lam_logor = np.empty(cfg.m, dtype=float)
    lam_se = np.empty(cfg.m, dtype=float)
    lam_p = np.empty(cfg.m, dtype=float)

    # separate RNG for bootstrap reproducibility
    boot_rng = np.random.default_rng(cfg.seed + 12345)

    def stat_naive(xb, yb):
        return logor_naive_from_xy(xb, yb, ha=0.5)

    def stat_lam(xb, yb):
        return logor_lambda_from_xy(xb, yb, p=cfg.p_sel, q=cfg.q_sel, lam=cfg.lam, ha=0.0)

    for j in range(cfg.m):
        xj = Xs[:, j].astype(int)

        th, se, p = bootstrap_se_and_pvalue(xj, Yt, stat_naive, boot_rng, B=cfg.boot_B)
        naive_logor[j] = th
        naive_se[j] = se
        naive_p[j] = p

        th, se, p = bootstrap_se_and_pvalue(xj, Yt, stat_lam, boot_rng, B=cfg.boot_B)
        lam_logor[j] = th
        lam_se[j] = se
        lam_p[j] = p

    # 7) multiple testing thresholds and discovery summaries
    m = cfg.m
    bonf = bonferroni_alpha(cfg.alpha, m)
    bh_naive = bh_threshold(naive_p, q=cfg.alpha)
    bh_lam = bh_threshold(lam_p, q=cfg.alpha)

    def discovery_counts(pvals: np.ndarray, thr: Optional[float]) -> Tuple[int, int, float]:
        if thr is None:
            return 0, 0, float("nan")
        disc = pvals <= thr
        tp = int(np.sum(disc & is_signal))
        fp = int(np.sum(disc & (~is_signal)))
        fdr = fp / max(tp + fp, 1)
        return tp, fp, fdr

    tp_n_005, fp_n_005, fdr_n_005 = discovery_counts(naive_p, cfg.alpha)
    tp_n_bonf, fp_n_bonf, fdr_n_bonf = discovery_counts(naive_p, bonf)
    tp_n_bh, fp_n_bh, fdr_n_bh = discovery_counts(naive_p, bh_naive)

    tp_l_005, fp_l_005, fdr_l_005 = discovery_counts(lam_p, cfg.alpha)
    tp_l_bonf, fp_l_bonf, fdr_l_bonf = discovery_counts(lam_p, bonf)
    tp_l_bh, fp_l_bh, fdr_l_bh = discovery_counts(lam_p, bh_lam)

    # 8) write results table
    df = pd.DataFrame({
        "feature": np.arange(cfg.m),
        "is_signal": is_signal.astype(int),
        "naive_logOR": naive_logor,
        "naive_se_boot": naive_se,
        "naive_p_boot": naive_p,
        "lambda_logOR": lam_logor,
        "lambda_se_boot": lam_se,
        "lambda_p_boot": lam_p,
    })
    out_csv = os.path.join(cfg.out_dir, "scan_results.csv")
    df.to_csv(out_csv, index=False)

    # 9) volcano plots
    def volcano_plot(x: np.ndarray, p: np.ndarray, title: str, out_png: str) -> None:
        y = -np.log10(np.clip(p, 1e-300, 1.0))
        plt.figure(figsize=(9, 7))
        # plot nulls then signals
        plt.scatter(x[~is_signal], y[~is_signal], s=10, alpha=0.6)
        plt.scatter(x[is_signal], y[is_signal], s=16, alpha=0.9)
        # reference lines
        plt.axhline(-np.log10(cfg.alpha), linestyle="--", linewidth=1.0)
        plt.axvline(0.0, linestyle=":", linewidth=1.0)
        plt.xlabel("log(OR)")
        plt.ylabel("-log10(p)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    out_naive_png = os.path.join(cfg.out_dir, "volcano_naive.png")
    out_lam_png = os.path.join(cfg.out_dir, "volcano_lambda.png")
    volcano_plot(naive_logor, naive_p, "Volcano: naive log-OR on misclassified outcome (selected set)", out_naive_png)
    volcano_plot(lam_logor, lam_p, f"Volcano: lambda-OR (lam={cfg.lam}) using (p_sel,q_sel) within selected set", out_lam_png)

    # 10) print summary
    print("\nSelection thresholds:")
    if cfg.two_gate:
        print(f"  two_gate=True, t_lo={t_lo:.6f}, t_hi={t_hi:.6f}, kept={n_sel}/{cfg.n} ({frac_sel:.3f})")
    else:
        mode_desc = "specificity" if cfg.selection_mode == "spec" else "sensitivity"
        target = cfg.target_spec if cfg.selection_mode == "spec" else cfg.target_sens
        print(f"  two_gate=False, ROC-targeted {mode_desc}={target:.3f}, t_hi={t_hi:.6f}, kept={n_sel}/{cfg.n} ({frac_sel:.3f})")

    print("Target (selection-conditional) label noise (applied within I=1):")
    print(f"  p_sel_target={cfg.p_sel:.4f}, q_sel_target={cfg.q_sel:.4f}")

    print("Estimated (selection-conditional) error rates on selected set (simulation check):")
    print(f"  p_sel_hat={p_sel_hat:.4f}, q_sel_hat={q_sel_hat:.4f}, det(K)={detK:.4f}")

    print("Multiple-testing thresholds:")
    print(f"  nominal alpha={cfg.alpha}")
    print(f"  Bonferroni alpha={bonf}  (m={m})")
    print(f"  BH naive alpha={bh_naive if bh_naive is not None else 'none'}")
    print(f"  BH lambda alpha={bh_lam if bh_lam is not None else 'none'}")

    print("Discoveries (naive, bootstrap p-values):")
    print(f"  p<0.05        TP={tp_n_005:4d}  FP={fp_n_005:4d}  FDR={fdr_n_005:.3f}")
    print(f"  Bonferroni    TP={tp_n_bonf:4d}  FP={fp_n_bonf:4d}  FDR={fdr_n_bonf:.3f}")
    if bh_naive is None:
        print("  BH q=0.05     none")
    else:
        print(f"  BH q=0.05     TP={tp_n_bh:4d}  FP={fp_n_bh:4d}  FDR={fdr_n_bh:.3f}")

    print("Discoveries (lambda, bootstrap p-values):")
    print(f"  p<0.05        TP={tp_l_005:4d}  FP={fp_l_005:4d}  FDR={fdr_l_005:.3f}")
    print(f"  Bonferroni    TP={tp_l_bonf:4d}  FP={fp_l_bonf:4d}  FDR={fdr_l_bonf:.3f}")
    if bh_lam is None:
        print("  BH q=0.05     none")
    else:
        print(f"  BH q=0.05     TP={tp_l_bh:4d}  FP={fp_l_bh:4d}  FDR={fdr_l_bh:.3f}")

    print(f"Outputs written to: {os.path.abspath(cfg.out_dir)}")
    print(f"  - {out_csv}")
    print(f"  - {out_naive_png}")
    print(f"  - {out_lam_png}")


def parse_args() -> SimConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./lambdaor_outputs_latent")
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--n_signal", type=int, default=60)

    ap.add_argument("--eta0", type=float, default=-0.2)
    ap.add_argument("--eta1", type=float, default=1.2)

    ap.add_argument("--score_noise", type=float, default=1.0)
    ap.add_argument("--selection_mode", type=str, choices=["spec", "sens"], default="spec")
    ap.add_argument("--target_spec", type=float, default=0.95)
    ap.add_argument("--target_sens", type=float, default=0.80)
    ap.add_argument("--two_gate", action="store_true")
    ap.add_argument("--keep_frac_two_gate", type=float, default=0.20)

    ap.add_argument("--null_pi_low", type=float, default=0.05)
    ap.add_argument("--null_pi_high", type=float, default=0.25)
    ap.add_argument("--alpha_signal", type=float, default=-1.0)
    ap.add_argument("--gamma_signal", type=float, default=1.0)

    ap.add_argument("--p_sel", type=float, default=0.80)
    ap.add_argument("--q_sel", type=float, default=0.70)

    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--boot_B", type=int, default=200)

    ap.add_argument("--alpha", type=float, default=0.05)

    a = ap.parse_args()
    return SimConfig(
        n=a.n,
        m=a.m,
        n_signal=a.n_signal,
        eta0=a.eta0,
        eta1=a.eta1,
        score_noise=a.score_noise,
        selection_mode=a.selection_mode,
        target_spec=a.target_spec,
        target_sens=a.target_sens,
        two_gate=a.two_gate,
        keep_frac_two_gate=a.keep_frac_two_gate,
        null_pi_low=a.null_pi_low,
        null_pi_high=a.null_pi_high,
        alpha_signal=a.alpha_signal,
        gamma_signal=a.gamma_signal,
        p_sel=a.p_sel,
        q_sel=a.q_sel,
        lam=a.lam,
        boot_B=a.boot_B,
        alpha=a.alpha,
        seed=a.seed,
        out_dir=a.out_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
