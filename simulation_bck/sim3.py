#!/usr/bin/env python3
"""
Naive-only volcano with DGP engineered to "close the internal gap".

Key idea:
  - Fix balanced marginals: P(X=1)=0.5 and P(Y=1)=0.5 for every feature.
    -> all four expected cell counts ~ n/4, so SE(logOR) ~ 4/sqrt(n) is tiny and stable.
  - Draw true log-OR per feature at the scale of that SE:
        beta_j ~ Normal(0, k * SE)
    so beta_j is very small (x near 0), but z = beta_j / SE ~ Normal(0, k) can be large.
Result:
  - Many points have very small |logOR| but very large -log10(p).
  - The central "wedge" fills in (blob/column).

Outputs:
  - volcano_naive_blob_closed.png
  - scan_naive_blob_closed.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def logor_wald_p(a: int, b: int, c: int, d: int, ha: float = 0.5):
    # Haldane-Anscombe correction for stability
    a2, b2, c2, d2 = a + ha, b + ha, c + ha, d + ha
    logor = float(np.log((a2 * d2) / (b2 * c2)))
    se = float(np.sqrt(1.0 / a2 + 1.0 / b2 + 1.0 / c2 + 1.0 / d2))
    if not np.isfinite(se) or se <= 0:
        return logor, np.nan, np.nan
    z = logor / se
    p = float(2.0 * norm.sf(abs(z)))
    return logor, se, p


def probs_from_or_balanced(beta: float):
    """
    Balanced marginals: P(X=1)=P(Y=1)=0.5 and symmetric table:
      p11 = p00 = t
      p10 = p01 = 0.5 - t
    Then OR = (t^2)/((0.5 - t)^2) => t/(0.5 - t)=exp(beta/2).
    """
    r = float(np.exp(beta / 2.0))
    t = 0.5 * r / (1.0 + r)
    p11 = t
    p00 = t
    p10 = 0.5 - t
    p01 = 0.5 - t
    # order: [p00, p01, p10, p11] sums to 1
    return np.array([p00, p01, p10, p11], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./naive_blob_closed_outputs")
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--n", type=int, default=600000,
                    help="Total sample size per feature (drives SE ~ 4/sqrt(n)).")
    ap.add_argument("--m", type=int, default=4000,
                    help="Number of features (independent contingency tables).")

    ap.add_argument("--k", type=float, default=6.0,
                    help="Effect scale in SE units: beta_j ~ N(0, k*SE). Larger -> more vertical spread.")
    ap.add_argument("--ha", type=float, default=0.5,
                    help="Haldane-Anscombe pseudo-count.")
    ap.add_argument("--clip_beta", type=float, default=0.08,
                    help="Clip |beta| to keep x extremely tight near 0 (helps blob).")

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    n, m = args.n, args.m

    # For balanced expected cell counts n/4, the plug-in SE scale is ~ 4/sqrt(n).
    se_target = 4.0 / np.sqrt(float(n))
    sigma_beta = args.k * se_target

    beta = rng.normal(0.0, sigma_beta, size=m)
    if args.clip_beta is not None and args.clip_beta > 0:
        beta = np.clip(beta, -args.clip_beta, args.clip_beta)

    a = np.empty(m, dtype=int)
    b = np.empty(m, dtype=int)
    c = np.empty(m, dtype=int)
    d = np.empty(m, dtype=int)

    logor = np.empty(m, dtype=float)
    se = np.empty(m, dtype=float)
    pval = np.empty(m, dtype=float)

    for j in range(m):
        p = probs_from_or_balanced(beta[j])
        # multinomial draw: counts for [00,01,10,11]
        cnt = rng.multinomial(n, p)
        d[j] = int(cnt[0])  # X=0,Y=0
        c[j] = int(cnt[1])  # X=0,Y=1
        b[j] = int(cnt[2])  # X=1,Y=0
        a[j] = int(cnt[3])  # X=1,Y=1

        lo, s, pv = logor_wald_p(a[j], b[j], c[j], d[j], ha=args.ha)
        logor[j] = lo
        se[j] = s
        pval[j] = pv

    neglog10p = -np.log10(np.clip(pval, 1e-300, 1.0))

    df = pd.DataFrame({
        "feature": np.arange(m),
        "beta_true": beta,
        "a": a, "b": b, "c": c, "d": d,
        "logOR_hat": logor,
        "se_wald": se,
        "p_wald": pval,
        "neglog10p": neglog10p,
    })

    out_csv = os.path.join(args.out_dir, "scan_naive_blob_closed.csv")
    df.to_csv(out_csv, index=False)

    out_png = os.path.join(args.out_dir, "volcano_naive_blob_closed.png")
    plt.figure(figsize=(9, 7))
    plt.scatter(logor, neglog10p, s=8, alpha=0.6)
    plt.axvline(0.0, linestyle=":", linewidth=1.0)
    plt.axhline(-np.log10(0.05), linestyle="--", linewidth=1.0)
    plt.xlabel("log(OR)")
    plt.ylabel("-log10(p)")
    plt.title(f"Naive blob (gap closed): n={n}, m={m}, k={args.k}, SE≈{se_target:.4g}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

    med_abs = float(np.median(np.abs(logor[np.isfinite(logor)])))
    q95_abs = float(np.quantile(np.abs(logor[np.isfinite(logor)]), 0.95))
    frac_hi = float(np.mean(np.isfinite(pval) & (pval < 1e-12)))

    print("Run summary:")
    print(f"  n={n}, m={m}, seed={args.seed}")
    print(f"  SE_target≈{se_target:.6g}; beta_sd≈{sigma_beta:.6g} (k={args.k})")
    print(f"  median |logOR_hat|={med_abs:.6g}, 95% |logOR_hat|={q95_abs:.6g}")
    print(f"  fraction p<1e-12 = {frac_hi:.3f}")
    print(f"Outputs written to: {os.path.abspath(args.out_dir)}")
    print(f"  - {out_csv}")
    print(f"  - {out_png}")


if __name__ == "__main__":
    main()
