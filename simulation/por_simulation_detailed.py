#!/usr/bin/env python3
"""
POR Simulation Study (annotated)
================================

End-to-end Monte Carlo simulation illustrating how the Posterior Odds Ratio (POR)
corrects bias under label misclassification.

Pipeline overview:
------------------
1. Generate X ~ Bernoulli(pi)
2. Generate latent Y | X ~ Bernoulli(σ(α + βX))
3. Apply outcome misclassification with sensitivity (sens) and specificity (spec)
4. Estimate exposure effect using:
   - Naive log-OR (from noisy 2x2 table)
   - POR (bias-corrected table inversion with small ridge λ if necessary)
5. Aggregate bias, RMSE, and 95% CI coverage across replicates.

Outputs:
---------
- replicate-level and summary CSVs
- publication-quality Matplotlib figures
"""

from __future__ import annotations
import math, os
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng

RNG_SEED = 20251005
rng = default_rng(RNG_SEED)

OUTPUT_DIR = os.environ.get("POR_SIM_OUTPUT", "./por_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class Config:
    """
    Container for simulation settings.
    -------------------
    n : int
        Sample size per replicate.
    pi : float
        Exposure prevalence (P(X=1)).
    beta : float
        True log-odds effect size (log(OR_true)).
    target_prev : float
        Desired marginal prevalence for Y; used to solve α.
    sens, spec : float
        Sensitivity and specificity for label noise.
    reps : int
        Number of Monte Carlo replicates.
    """
    n: int = 20000
    pi: float = 0.5
    beta: float = math.log(2)
    target_prev: float = 0.2
    sens: float = 0.4
    spec: float = 0.6
    reps: int = 150


def find_alpha_for_prevalence(pi: float, beta: float, target: float = 0.2) -> float:
    """
    Solve for α such that marginal prevalence of Y ≈ target.

    α is found by grid search so that:
        π * σ(α + β) + (1 - π) * σ(α) ≈ target
    """
    alphas = np.linspace(-4, 0, 4001)
    best = None
    for a in alphas:
        p1 = 1 / (1 + np.exp(-(a + beta)))
        p0 = 1 / (1 + np.exp(-a))
        prev = pi * p1 + (1 - pi) * p0
        val = abs(prev - target)
        if best is None or val < best[0]:
            best = (val, a, prev)
    return best[1]


def gen_data(n: int, pi: float, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (X, Y) pairs from logistic model.

    Returns
    -------
    X : exposure (0/1)
    Y : latent outcome (0/1)
    """
    X = rng.binomial(1, pi, size=n)
    logits = alpha + beta * X
    p = 1 / (1 + np.exp(-logits))
    Y = rng.binomial(1, p)
    return X.astype(np.int64), Y.astype(np.int64)


def misclassify(Y: np.ndarray, sens: float, spec: float) -> np.ndarray:
    """
    Simulate outcome misclassification.

    Each Y=1 flips to 0 with prob (1-sens).
    Each Y=0 flips to 1 with prob (1-spec).
    """
    Ytil = Y.copy()
    idx1 = np.where(Y == 1)[0]
    idx0 = np.where(Y == 0)[0]
    flip1 = rng.binomial(1, 1 - sens, size=idx1.size)
    flip0 = rng.binomial(1, 1 - spec, size=idx0.size)
    Ytil[idx1] = 1 - flip1
    Ytil[idx0] = flip0
    return Ytil.astype(np.int64)


def log_or_from_table(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Naive log-odds ratio from observed 2x2 table.

    Adds 0.5 continuity correction to avoid division by zero.
    """
    a = int(((x == 1) & (y == 1)).sum())
    b = int(((x == 0) & (y == 1)).sum())
    c = int(((x == 1) & (y == 0)).sum())
    d = int(((x == 0) & (y == 0)).sum())
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = math.log((a * d) / (b * c))
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    return log_or, se


def por_log_or(x: np.ndarray, y_tilde: np.ndarray, sens: float, spec: float,
               lam: float = 0.0, eps: float = 1e-9) -> Tuple[float, float, bool]:
    """
    Correct the noisy table via matrix inversion of K^T.

    Tilde table: [[a_t, c_t], [b_t, d_t]]
    Correction:  T_corr = T_tilde @ (K^T + λI)^(-1)

    Returns log-OR and SE; feasible=False if any count < eps.
    """
    a_t = int(((x == 1) & (y_tilde == 1)).sum())
    b_t = int(((x == 0) & (y_tilde == 1)).sum())
    c_t = int(((x == 1) & (y_tilde == 0)).sum())
    d_t = int(((x == 0) & (y_tilde == 0)).sum())
    Ttil = np.array([[a_t, c_t],
                     [b_t, d_t]], dtype=float)

    K = np.array([[sens, 1 - sens],
                  [1 - spec, spec]], dtype=float)
    Kt_reg = K.T + lam * np.eye(2)

    try:
        Kt_inv = np.linalg.inv(Kt_reg)
    except np.linalg.LinAlgError:
        Kt_inv = np.linalg.pinv(Kt_reg)

    Tcorr = Ttil @ Kt_inv
    a, c = Tcorr[0, 0], Tcorr[0, 1]
    b, d = Tcorr[1, 0], Tcorr[1, 1]
    feasible = np.all(Tcorr > eps)
    if not feasible:
        a, b, c, d = [max(val, eps) for val in (a, b, c, d)]

    log_or = math.log((a * d) / (b * c))
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    return log_or, se, feasible


def summarize(df: pd.DataFrame, method: str) -> pd.Series:
    """
    Compute bias, RMSE, and 95% CI coverage for estimator across replicates.
    """
    est = df[f"ln_{method}"]
    se = df[f"se_{method}"]
    true = df["ln_true"]
    bias_mean = (est - true).mean()
    rmse = float(np.sqrt(np.mean((est - true) ** 2)))
    cov = float(np.mean((est - 1.96 * se <= true) & (est + 1.96 * se >= true)))
    return pd.Series(dict(bias_mean=bias_mean, rmse=rmse, coverage=cov))


def run_cell(cfg: Config) -> pd.DataFrame:
    """
    Execute one configuration (fixed sens/spec) for cfg.reps Monte Carlo replicates.
    """
    alpha = find_alpha_for_prevalence(cfg.pi, cfg.beta, target=cfg.target_prev)
    records = []
    lam_grid = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]

    for _ in range(cfg.reps):
        X, Y = gen_data(cfg.n, cfg.pi, alpha, cfg.beta)
        Yt = misclassify(Y, cfg.sens, cfg.spec)
        ln_naive, se_naive = log_or_from_table(X, Yt)

        chosen_lam = None
        for lam_try in lam_grid:
            ln_por, se_por, feasible = por_log_or(X, Yt, cfg.sens, cfg.spec, lam=lam_try)
            if feasible:
                chosen_lam = lam_try
                break
        if chosen_lam is None:
            chosen_lam = lam_grid[-1]
            ln_por, se_por, feasible = por_log_or(X, Yt, cfg.sens, cfg.spec, lam=chosen_lam)

        records.append(dict(
            ln_true=cfg.beta, ln_naive=ln_naive, se_naive=se_naive,
            ln_por=ln_por, se_por=se_por, lam=chosen_lam,
            n=cfg.n, pi=cfg.pi, beta=cfg.beta, sens=cfg.sens, spec=cfg.spec
        ))

    return pd.DataFrame(records)


def primary_sweep() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Symmetric noise sweep: sens=spec in {0.95, 0.90, 0.85, 0.80}.
    Returns replicate- and summary-level dataframes.
    """
    levels = [0.95, 0.90, 0.85, 0.80]
    base = Config()
    dfs, sums = [], []

    for s in levels:
        cfg = Config(n=base.n, pi=base.pi, beta=base.beta, target_prev=base.target_prev,
                     sens=s, spec=s, reps=base.reps)
        df = run_cell(cfg)
        df["sens"] = s
        dfs.append(df)

        sn, sp = summarize(df, "naive"), summarize(df, "por")
        sm = pd.concat({"naive": sn, "por": sp}, axis=1).T.reset_index().rename(columns={"index": "method"})
        sm["sens"] = s
        sm["lam_used_frac"] = float((df["lam"] > 0).mean())
        sums.append(sm)

    return pd.concat(dfs, ignore_index=True), pd.concat(sums, ignore_index=True)


def asymmetric_sweep() -> pd.DataFrame:
    """
    Asymmetric noise cases for SI table: (sens,spec)=(0.90,0.80),(0.85,0.95).
    """
    base = Config()
    results = []
    for p, q in [(0.90, 0.80), (0.85, 0.95)]:
        cfg = Config(n=base.n, pi=base.pi, beta=base.beta, target_prev=base.target_prev,
                     sens=p, spec=q, reps=base.reps)
        df = run_cell(cfg)
        sn, sp = summarize(df, "naive"), summarize(df, "por")
        sm = pd.concat({"naive": sn, "por": sp}, axis=1).T.reset_index().rename(columns={"index": "method"})
        sm["sens"], sm["spec"], sm["lam_used_frac"] = p, q, float((df["lam"] > 0).mean())
        results.append(sm)
    return pd.concat(results, ignore_index=True)


def plot_primary_curves(summ_df: pd.DataFrame, outdir: str):
    """
    Plot bias, RMSE, and coverage vs misclassification rate.
    """
    levels = sorted(summ_df["sens"].unique())
    xvals = [1 - s for s in levels]

    def extract(metric, method):
        return [float(summ_df[(summ_df["sens"] == s) & (summ_df["method"] == method)][metric]) for s in levels]

    bias_n, bias_p = extract("bias_mean", "naive"), extract("bias_mean", "por")
    rmse_n, rmse_p = extract("rmse", "naive"), extract("rmse", "por")
    cov_n, cov_p = extract("coverage", "naive"), extract("coverage", "por")

    plt.figure(); plt.plot(xvals, bias_n, "o-", label="Naive OR"); plt.plot(xvals, bias_p, "o-", label="POR")
    plt.xlabel("Misclassification rate (1-sensitivity)"); plt.ylabel("Bias"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "FigS1a_bias.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(xvals, rmse_n, "o-", label="Naive OR"); plt.plot(xvals, rmse_p, "o-", label="POR")
    plt.xlabel("Misclassification rate"); plt.ylabel("RMSE"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "FigS1b_rmse.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(xvals, cov_n, "o-", label="Naive OR"); plt.plot(xvals, cov_p, "o-", label="POR")
    plt.xlabel("Misclassification rate"); plt.ylabel("95% CI coverage"); plt.legend(); plt.ylim(0, 1)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "FigS1c_coverage.png"), dpi=200); plt.close()


def plot_ridge_use(rep_df: pd.DataFrame, outdir: str):
    """
    Plot fraction of replicates requiring ridge regularization (λ>0).
    """
    ridge_use = rep_df.groupby("sens")["lam"].apply(lambda s: float((s > 0).mean())).reset_index(name="ridge_frac")
    plt.figure(); plt.bar([str(1 - s) for s in ridge_use["sens"]], ridge_use["ridge_frac"])
    plt.xlabel("Misclassification rate"); plt.ylabel("Fraction with λ>0")
    plt.title("Ridge usage for POR"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "FigS2_ridge_use.png"), dpi=200); plt.close()


def main():
    rep_df, summ_df = primary_sweep()
    rep_df.to_csv(os.path.join(OUTPUT_DIR, "replicates_primary.csv"), index=False)
    summ_df.to_csv(os.path.join(OUTPUT_DIR, "summary_primary.csv"), index=False)
    plot_primary_curves(summ_df, OUTPUT_DIR)
    plot_ridge_use(rep_df, OUTPUT_DIR)
    asym = asymmetric_sweep()
    asym.to_csv(os.path.join(OUTPUT_DIR, "summary_asymmetric.csv"), index=False)
    print("POR simulation complete. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
