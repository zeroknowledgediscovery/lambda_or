#!/usr/bin/env python3
"""
Simulation at the 2x2-table level.

Pipeline per feature j:
  1) Draw a "true" 2x2 table (a,b,c,d) with balanced marginals using beta_true.
  2) Corrupt the table by outcome misclassification with (p,q) to get (a~,b~,c~,d~).
     This is done *within each X stratum* via binomial flips.
  3) Compute:
       - true logOR from (a,b,c,d)
       - naive logOR from (a~,b~,c~,d~)
       - lambda-corrected logOR from (a~,b~,c~,d~) using (K + lam I)^(-1)
     plus Wald p-values for naive and lambda (plug-in SE on the respective counts).
  4) Make volcano plots for naive and lambda, and a recovery scatter plot.

Outputs:
  - scan_noisy_lambda.csv
  - volcano_naive_noisy.png
  - volcano_lambda.png
  - recovery_scatter.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def logor_wald_p(a: float, b: float, c: float, d: float, ha: float = 0.5):
    # Haldane-Anscombe correction for stability
    a2, b2, c2, d2 = a + ha, b + ha, c + ha, d + ha
    logor = float(np.log((a2 * d2) / (b2 * c2)))
    se = float(np.sqrt(1.0 / a2 + 1.0 / b2 + 1.0 / c2 + 1.0 / d2))
    if not np.isfinite(se) or se <= 0:
        return logor, np.nan, np.nan
    z = logor / se
    p = float(2.0 * norm.sf(abs(z)))
    return logor, se, p


def probs_from_or_balanced(beta: float) -> np.ndarray:
    """
    Balanced marginals: P(X=1)=P(Y=1)=0.5 and symmetric table:
      p11 = p00 = t
      p10 = p01 = 0.5 - t
    Then OR = (t^2)/((0.5 - t)^2) => t/(0.5 - t)=exp(beta/2).
    Returns probabilities for [00,01,10,11].
    """
    r = float(np.exp(beta / 2.0))
    t = 0.5 * r / (1.0 + r)
    p11 = t
    p00 = t
    p10 = 0.5 - t
    p01 = 0.5 - t
    return np.array([p00, p01, p10, p11], dtype=float)


def misclassify_table_counts(a: int, b: int, c: int, d: int, p: float, q: float, rng: np.random.Generator):
    """
    Apply outcome misclassification within each X stratum.

    True table:
      a=n(X=1,Y=1), b=n(X=1,Y=0), c=n(X=0,Y=1), d=n(X=0,Y=0)

    Observed after misclassification (Y -> Y~):
      sensitivity p = P(Y~=1|Y=1)
      specificity q = P(Y~=0|Y=0)
    """
    # X=1
    tp1 = rng.binomial(a, p)         # from true positives retained
    fp1 = rng.binomial(b, 1.0 - q)   # from true negatives flipped to 1
    a_t = int(tp1 + fp1)             # X=1, Y~=1
    b_t = int((a - tp1) + (b - fp1)) # X=1, Y~=0

    # X=0
    tp0 = rng.binomial(c, p)
    fp0 = rng.binomial(d, 1.0 - q)
    c_t = int(tp0 + fp0)             # X=0, Y~=1
    d_t = int((c - tp0) + (d - fp0)) # X=0, Y~=0

    return a_t, b_t, c_t, d_t


def correction_matrix(p: float, q: float) -> np.ndarray:
    """
    K maps true outcome counts -> observed outcome counts for a fixed X:
      [obs0]   [ q      (1-p) ] [true0]
      [obs1] = [ (1-q)   p    ] [true1]
    """
    return np.array([[q, 1.0 - p],
                     [1.0 - q, p]], dtype=float)


def lambda_corrected_counts_per_stratum(obs0: float, obs1: float, p: float, q: float, lam: float):
    """
    Given observed counts for Y~ in one X stratum: [obs0, obs1],
    return corrected counts for Y: [true0, true1] using (K + lam I)^(-1).
    """
    K = correction_matrix(p, q)
    inv = np.linalg.inv(K + lam * np.eye(2))
    tru = inv @ np.array([obs0, obs1], dtype=float)
    tru = np.clip(tru, 1e-6, None)
    return float(tru[0]), float(tru[1])


def lambda_corrected_logor_and_p(a_t: int, b_t: int, c_t: int, d_t: int, p: float, q: float, lam: float, ha: float = 0.5):
    """
    Lambda-correct (a~,b~,c~,d~) to estimated true counts and compute:
      logOR_lambda, SE (plug-in on corrected counts), Wald p-value.
    """
    # X=1: obs0=b_t, obs1=a_t
    b_hat, a_hat = lambda_corrected_counts_per_stratum(b_t, a_t, p, q, lam)
    # X=0: obs0=d_t, obs1=c_t
    d_hat, c_hat = lambda_corrected_counts_per_stratum(d_t, c_t, p, q, lam)

    lo, se, pv = logor_wald_p(a_hat, b_hat, c_hat, d_hat, ha=ha)
    return lo, se, pv, a_hat, b_hat, c_hat, d_hat


def volcano_plot(x: np.ndarray, p: np.ndarray, out_png: str, title: str, xlabel: str):
    y = -np.log10(np.clip(p, 1e-300, 1.0))
    plt.figure(figsize=(9, 7))
    plt.scatter(x, y, s=8, alpha=0.6)
    plt.axvline(0.0, linestyle=":", linewidth=1.0)
    plt.axhline(-np.log10(0.05), linestyle="--", linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel("-log10(p)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def recovery_scatter(true_lo: np.ndarray, naive_lo: np.ndarray, lam_lo: np.ndarray, out_png: str, title: str):
    plt.figure(figsize=(8, 8))
    plt.scatter(true_lo, naive_lo, s=8, alpha=0.35, label="naive (noisy table)")
    plt.scatter(true_lo, lam_lo, s=8, alpha=0.35, label="lambda-corrected")
    mn = float(np.nanmin(true_lo))
    mx = float(np.nanmax(true_lo))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.0)
    plt.xlabel("true log(OR)")
    plt.ylabel("estimated log(OR)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./noisy_lambda_outputs")
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--n", type=int, default=600000,
                    help="Total sample size per feature.")
    ap.add_argument("--m", type=int, default=4000,
                    help="Number of features (independent contingency tables).")

    ap.add_argument("--k", type=float, default=6.0,
                    help="Effect scale in SE units: beta_j ~ N(0, k*SE).")
    ap.add_argument("--ha", type=float, default=0.5,
                    help="Haldane-Anscombe pseudo-count for OR/p-values.")
    ap.add_argument("--clip_beta", type=float, default=0.08,
                    help="Clip |beta|.")

    # Noise parameters (table corruption)
    ap.add_argument("--p", type=float, default=0.80, help="Sensitivity for Y~.")
    ap.add_argument("--q", type=float, default=0.70, help="Specificity for Y~.")

    # Lambda regularization for inversion
    ap.add_argument("--lam", type=float, default=0.05, help="Ridge for (K + lam I)^(-1).")

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    n, m = args.n, args.m

    # SE scale under balanced expected counts n/4
    se_target = 4.0 / np.sqrt(float(n))
    sigma_beta = args.k * se_target

    beta_true = rng.normal(0.0, sigma_beta, size=m)
    if args.clip_beta is not None and args.clip_beta > 0:
        beta_true = np.clip(beta_true, -args.clip_beta, args.clip_beta)

    # Storage
    a = np.empty(m, dtype=int)
    b = np.empty(m, dtype=int)
    c = np.empty(m, dtype=int)
    d = np.empty(m, dtype=int)

    a_t = np.empty(m, dtype=int)
    b_t = np.empty(m, dtype=int)
    c_t = np.empty(m, dtype=int)
    d_t = np.empty(m, dtype=int)

    logor_true = np.empty(m, dtype=float)

    logor_naive = np.empty(m, dtype=float)
    se_naive = np.empty(m, dtype=float)
    p_naive = np.empty(m, dtype=float)

    logor_lam = np.empty(m, dtype=float)
    se_lam = np.empty(m, dtype=float)
    p_lam = np.empty(m, dtype=float)

    # corrected counts (optional to save)
    a_hat = np.empty(m, dtype=float)
    b_hat = np.empty(m, dtype=float)
    c_hat = np.empty(m, dtype=float)
    d_hat = np.empty(m, dtype=float)

    for j in range(m):
        probs = probs_from_or_balanced(beta_true[j])
        cnt = rng.multinomial(n, probs)

        d[j] = int(cnt[0])  # X=0,Y=0
        c[j] = int(cnt[1])  # X=0,Y=1
        b[j] = int(cnt[2])  # X=1,Y=0
        a[j] = int(cnt[3])  # X=1,Y=1

        logor_true[j] = logor_wald_p(a[j], b[j], c[j], d[j], ha=args.ha)[0]

        # corrupt table
        a_t[j], b_t[j], c_t[j], d_t[j] = misclassify_table_counts(a[j], b[j], c[j], d[j], args.p, args.q, rng)

        # naive on corrupted table
        lo, s, pv = logor_wald_p(a_t[j], b_t[j], c_t[j], d_t[j], ha=args.ha)
        logor_naive[j], se_naive[j], p_naive[j] = lo, s, pv

        # lambda correction from corrupted table
        lo2, s2, pv2, ah, bh, ch, dh = lambda_corrected_logor_and_p(
            a_t[j], b_t[j], c_t[j], d_t[j],
            p=args.p, q=args.q, lam=args.lam, ha=args.ha
        )
        logor_lam[j], se_lam[j], p_lam[j] = lo2, s2, pv2
        a_hat[j], b_hat[j], c_hat[j], d_hat[j] = ah, bh, ch, dh

    df = pd.DataFrame({
        "feature": np.arange(m),
        "beta_true": beta_true,
        "logOR_true": logor_true,

        "a": a, "b": b, "c": c, "d": d,

        "a_obs": a_t, "b_obs": b_t, "c_obs": c_t, "d_obs": d_t,

        "logOR_naive": logor_naive,
        "se_naive": se_naive,
        "p_naive": p_naive,

        "logOR_lambda": logor_lam,
        "se_lambda": se_lam,
        "p_lambda": p_lam,

        "a_hat": a_hat, "b_hat": b_hat, "c_hat": c_hat, "d_hat": d_hat,
    })

    out_csv = os.path.join(args.out_dir, "scan_noisy_lambda.csv")
    df.to_csv(out_csv, index=False)

    detK = args.p + args.q - 1.0

    out_naive = os.path.join(args.out_dir, "volcano_naive_noisy.png")
    volcano_plot(
        logor_naive, p_naive, out_naive,
        title=f"Naive volcano from corrupted table (p={args.p:.2f}, q={args.q:.2f}, detK={detK:.2f})",
        xlabel="naive log(OR) on (a~,b~,c~,d~)"
    )

    out_lam_png = os.path.join(args.out_dir, "volcano_lambda.png")
    volcano_plot(
        logor_lam, p_lam, out_lam_png,
        title=f"Lambda-corrected volcano (lam={args.lam:g}, detK={detK:.2f})",
        xlabel="lambda-corrected log(OR)"
    )

    out_scatter = os.path.join(args.out_dir, "recovery_scatter.png")
    recovery_scatter(
        logor_true, logor_naive, logor_lam,
        out_scatter,
        title="Recovery: true vs naive and lambda log(OR)"
    )

    # numeric summaries
    def _mse(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        return float(np.mean((x[m] - y[m]) ** 2)) if np.any(m) else np.nan

    mse_naive = _mse(logor_true, logor_naive)
    mse_lam = _mse(logor_true, logor_lam)

    corr_naive = float(np.corrcoef(logor_true, logor_naive)[0,1])
    corr_lam = float(np.corrcoef(logor_true, logor_lam)[0,1])

    print("Run summary:")
    print(f"  n={n}, m={m}, seed={args.seed}")
    print(f"  Noise: p={args.p}, q={args.q}, det(K)={detK:.4f}, lam={args.lam}")
    print(f"  True beta sd≈{sigma_beta:.6g} (k={args.k}, SE_target≈{se_target:.6g})")
    print(f"  MSE(logOR): naive={mse_naive:.6g} ; lambda={mse_lam:.6g}")
    print(f"  Corr(logOR,true): naive={corr_naive:.4f} ; lambda={corr_lam:.4f}")
    print(f"Outputs written to: {os.path.abspath(args.out_dir)}")
    print(f"  - {out_csv}")
    print(f"  - {out_naive}")
    print(f"  - {out_lam_png}")
    print(f"  - {out_scatter}")


if __name__ == "__main__":
    main()
