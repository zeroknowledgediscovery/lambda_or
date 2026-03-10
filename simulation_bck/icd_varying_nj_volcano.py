#!/usr/bin/env python3
"""
ICD-like many-covariate volcano demo with varying n_j (heavy-tailed prevalence), small effects,
and outcome misclassification applied by flipping cases/controls at the table level.

Generates two volcano plots:
  1) Perfect (true tables)
  2) Noisy (tables after outcome misclassification with sensitivity p and specificity q)

Also writes a CSV with per-covariate summaries.

Default design:
  - Many covariates (m large)
  - Exposure prevalences heavy-tailed and mostly small (ICD-code-like)
  - True logOR effects small
  - n_j varies automatically as exposed counts vary across covariates

Run:
  python icd_varying_nj_volcano.py --out_dir ./out
"""

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def logor_wald_vec(a,b,c,d,ha=0.5):
    a = a.astype(float) + ha
    b = b.astype(float) + ha
    c = c.astype(float) + ha
    d = d.astype(float) + ha
    lo = np.log((a*d)/(b*c))
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = lo / se
    p = 2.0 * norm.sf(np.abs(z))
    return lo, se, p

def misclassify_tables(a,b,c,d,p,q,rng):
    """
    Outcome misclassification within each X stratum:
      P(Ytil=1|Y=1)=p, P(Ytil=0|Y=0)=q
    Applied to counts (a,b,c,d) elementwise.
    """
    # X=1 stratum: (Y=1 count a, Y=0 count b)
    tp1 = rng.binomial(a, p)
    fp1 = rng.binomial(b, 1.0 - q)
    aT = tp1 + fp1
    bT = (a - tp1) + (b - fp1)

    # X=0 stratum: (Y=1 count c, Y=0 count d)
    tp0 = rng.binomial(c, p)
    fp0 = rng.binomial(d, 1.0 - q)
    cT = tp0 + fp0
    dT = (c - tp0) + (d - fp0)

    return aT, bT, cT, dT

def volcano(x, p, out_png, title):
    y = -np.log10(np.clip(p, 1e-300, 1.0))
    plt.figure(figsize=(8.2, 6.2))
    plt.scatter(x, y, s=4, alpha=0.55)
    plt.axvline(0.0, color="black", lw=1)
    plt.axhline(-np.log10(0.05), ls="--", lw=1)
    plt.xlabel("log(OR)")
    plt.ylabel("-log10(p)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./icd_volcano_outputs")
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--m", type=int, default=30000, help="Number of covariates (very large).")
    ap.add_argument("--N_cases", type=int, default=20000, help="Total cases in cohort.")
    ap.add_argument("--N_ctrl", type=int, default=20000, help="Total controls in cohort.")

    ap.add_argument("--N_cases_big", type=int, default=500000, help="Cases for equal-large-N panel.")
    ap.add_argument("--N_ctrl_big", type=int, default=500000, help="Controls for equal-large-N panel.")
    ap.add_argument("--n_exposed_big", type=int, default=20000, help="Fixed exposed count per covariate for equal-n_j panel.")

    ap.add_argument("--alpha_mean", type=float, default=-6.5,
                    help="Mean logit prevalence for controls (mostly rare codes).")
    ap.add_argument("--alpha_sd", type=float, default=2.0,
                    help="High variance -> heavy-tailed prevalence -> varying n_j.")
    ap.add_argument("--beta_sd", type=float, default=0.05,
                    help="Small true effect size (logOR) SD.")

    ap.add_argument("--p", type=float, default=0.80, help="Sensitivity for outcome misclassification.")
    ap.add_argument("--q", type=float, default=0.70, help="Specificity for outcome misclassification.")

    ap.add_argument("--min_prev", type=float, default=1e-6, help="Clip prevalence floor.")
    ap.add_argument("--max_prev", type=float, default=0.20, help="Clip prevalence ceiling.")

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    m = args.m
    Nc = args.N_cases
    N0 = args.N_ctrl

    # Control prevalence (heavy-tailed on probability scale)
    alpha = rng.normal(args.alpha_mean, args.alpha_sd, size=m)
    pi0 = 1.0/(1.0+np.exp(-alpha))
    pi0 = np.clip(pi0, args.min_prev, args.max_prev)

    # Small true effects: beta_j ~ N(0, beta_sd^2)
    beta = rng.normal(0.0, args.beta_sd, size=m)

    # Case prevalence via logit shift: logit(pi1)=logit(pi0)+beta
    logit0 = np.log(pi0/(1.0-pi0))
    pi1 = 1.0/(1.0+np.exp(-(logit0 + beta)))
    pi1 = np.clip(pi1, args.min_prev, args.max_prev)

    # True tables (a,b,c,d)
    # a: X=1 among cases; c: X=0 among cases
    a = rng.binomial(Nc, pi1).astype(np.int64)
    c = (Nc - a).astype(np.int64)
    # b: X=1 among controls; d: X=0 among controls
    b = rng.binomial(N0, pi0).astype(np.int64)
    d = (N0 - b).astype(np.int64)

    # n_j = exposed count across cohort (ICD code frequency proxy)
    n_j = (a + b).astype(np.int64)

    # Perfect volcano
    logor_true, se_true, p_true = logor_wald_vec(a,b,c,d,ha=0.5)

    # Misclassify outcome counts at the table level
    aT,bT,cT,dT = misclassify_tables(a,b,c,d,args.p,args.q,rng)

    # Noisy volcano (naive on corrupted tables)
    logor_noisy, se_noisy, p_noisy = logor_wald_vec(aT,bT,cT,dT,ha=0.5)

        # Equal-large-n_exposed panel: enforce the SAME exposed count n_j for every covariate.
    # This collapses SE heterogeneity and yields a clean parabola-like volcano.
    Nc_big = args.N_cases_big
    N0_big = args.N_ctrl_big
    n_exposed_big = args.n_exposed_big

    if n_exposed_big < 4:
        raise ValueError("--n_exposed_big must be at least 4.")
    if n_exposed_big > (Nc_big + N0_big - 4):
        raise ValueError("--n_exposed_big too large for the specified cohort sizes.")

    OR = np.exp(beta)

    def solve_a_for_or(or_val, Nc, N0, n1):
        # Solve: or*(n1-a)*(Nc-a) = a*(N0-n1+a)
        A = (or_val - 1.0)
        B = (-or_val*(n1 + Nc) - (N0 - n1))
        C = (or_val * n1 * Nc)
        if abs(A) < 1e-10:
            return n1 * (Nc / (Nc + N0))
        disc = B*B - 4*A*C
        disc = max(disc, 0.0)
        rdisc = np.sqrt(disc)
        a1 = (-B + rdisc) / (2*A)
        a2 = (-B - rdisc) / (2*A)
        target = n1 * (Nc / (Nc + N0))
        cand = [a for a in (a1, a2) if 0.0 <= a <= n1]
        if cand:
            return min(cand, key=lambda x: abs(x-target))
        return float(np.clip(a1, 0.0, n1))

    a_big = np.zeros(m, dtype=np.int64)
    b_big = np.zeros(m, dtype=np.int64)
    for j in range(m):
        a_est = solve_a_for_or(float(OR[j]), Nc_big, N0_big, n_exposed_big)
        a_int = int(np.round(a_est))
        # Keep all cells nonzero and within cohort bounds
        a_int = max(1, min(a_int, n_exposed_big-1, Nc_big-1))
        b_int = n_exposed_big - a_int
        b_int = max(1, min(b_int, N0_big-1))
        # Re-enforce sum after clamping b
        a_int = n_exposed_big - b_int
        a_int = max(1, min(a_int, Nc_big-1))
        a_big[j] = a_int
        b_big[j] = b_int

    c_big = (Nc_big - a_big).astype(np.int64)
    d_big = (N0_big - b_big).astype(np.int64)

    logor_big, se_big, p_big = logor_wald_vec(a_big, b_big, c_big, d_big, ha=0.5)

# Write outputs
    volcano(logor_true, p_true,
            os.path.join(args.out_dir, "volcano_true.png"),
            f"Perfect volcano (m={m}, Nc={Nc}, N0={N0}, beta_sd={args.beta_sd})")

    detK = args.p + args.q - 1.0
    volcano(logor_noisy, p_noisy,
            os.path.join(args.out_dir, "volcano_noisy.png"),
            f"Noisy volcano after table flips (p={args.p}, q={args.q}, detK={detK:.3f})")

    volcano(logor_big, p_big,
            os.path.join(args.out_dir, "volcano_bigN.png"),
            f"Equal-n_j volcano (n_exposed={n_exposed_big}, Nc={Nc_big}, N0={N0_big})")

    df = pd.DataFrame({
        "j": np.arange(m),
        "n_exposed": n_j,
        "pi0_ctrl": pi0,
        "pi1_case": pi1,
        "beta_true": beta,
        "a": a, "b": b, "c": c, "d": d,
        "a_noisy": aT, "b_noisy": bT, "c_noisy": cT, "d_noisy": dT,
        "logOR_true": logor_true,
        "p_true": p_true,
        "logOR_noisy": logor_noisy,
        "p_noisy": p_noisy,
        "logOR_bigN": logor_big,
        "p_bigN": p_big,
    })
    df.to_csv(os.path.join(args.out_dir, "scan_icd_like.csv"), index=False)

    # Quick printed diagnostics
    qtiles = np.quantile(n_j, [0.5, 0.9, 0.99])
    print("Wrote:", os.path.abspath(args.out_dir))
    print(f"n_exposed quantiles: median={qtiles[0]:.0f}, 90%={qtiles[1]:.0f}, 99%={qtiles[2]:.0f}")
    print(f"det(K)=p+q-1 = {detK:.3f}")

if __name__ == "__main__":
    main()