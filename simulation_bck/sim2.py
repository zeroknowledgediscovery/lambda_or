#!/usr/bin/env python3
"""
Naive-only volcano designed to look like a "blob" (vertical column near logOR ~ 0).

Mechanism:
  - Latent Z drives outcome Y strongly enough to create structure.
  - Each exposure X_j depends on Z with the SAME small gamma (tiny effect size ~ constant).
  - Each exposure has a very different intercept alpha_j (wide prevalence spread),
    causing the SE of logOR to vary a lot across features.
  - Large n collapses SE enough that many tiny effects become significant.
Result:
  - logOR values cluster tightly near 0 (x-axis compressed).
  - -log10(p) varies widely (y-axis spread) mostly because SE varies with prevalence.
  - Volcano looks like a blob/vertical smear rather than classic arms.

Outputs:
  - volcano_naive_blob.png
  - scan_naive_blob.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import norm


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def counts_2x2(x: np.ndarray, y: np.ndarray):
    x = x.astype(np.int8)
    y = y.astype(np.int8)
    a = int(np.sum((x == 1) & (y == 1)))
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    d = int(np.sum((x == 0) & (y == 0)))
    return a, b, c, d


def logor_wald_p(a: int, b: int, c: int, d: int, ha: float = 0.5):
    a2, b2, c2, d2 = a + ha, b + ha, c + ha, d + ha
    logor = float(np.log((a2 * d2) / (b2 * c2)))
    se = float(np.sqrt(1.0 / a2 + 1.0 / b2 + 1.0 / c2 + 1.0 / d2))
    if not np.isfinite(se) or se <= 0:
        return logor, np.nan, np.nan
    z = logor / se
    p = float(2.0 * norm.sf(abs(z)))
    return logor, se, p


def volcano_plot(logor: np.ndarray, p: np.ndarray, out_png: str, title: str):
    y = -np.log10(np.clip(p, 1e-300, 1.0))
    plt.figure(figsize=(9, 7))
    plt.scatter(logor, y, s=8, alpha=0.6)
    plt.axvline(0.0, linestyle=":", linewidth=1.0)
    plt.axhline(-np.log10(0.05), linestyle="--", linewidth=1.0)
    plt.xlabel("log(OR)")
    plt.ylabel("-log10(p)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()



#======================================================

import numpy as np

def apply_misclassification(y: np.ndarray, p: float, q: float, rng: np.random.Generator) -> np.ndarray:
    """
    Misclassify a binary outcome y -> ytil.
      P(ytil=1 | y=1) = p  (sensitivity)
      P(ytil=0 | y=0) = q  (specificity)
    """
    y = y.astype(np.int8)
    u = rng.random(y.shape[0])
    ytil = y.copy()

    m1 = (y == 1)
    ytil[m1] = (u[m1] <= p).astype(np.int8)

    m0 = (y == 0)
    ytil[m0] = (u[m0] > q).astype(np.int8)
    return ytil


def correction_matrix(p: float, q: float) -> np.ndarray:
    """
    Outcome misclassification operator K mapping true outcome counts to observed counts:
      [obs0]   [ q      (1-p) ] [true0]
      [obs1] = [ (1-q)   p    ] [true1]
    """
    return np.array([[q, 1.0 - p],
                     [1.0 - q, p]], dtype=float)


def lambda_corrected_logor_from_counts(a: int, b: int, c: int, d: int, p: float, q: float, lam: float) -> float:
    """
    Compute lambda-corrected log OR from the observed 2x2 table counts.

    Inputs are observed counts with Y being the OBSERVED outcome (either Y or Ytil),
    with the standard layout:
      a = n(X=1, Y=1)
      b = n(X=1, Y=0)
      c = n(X=0, Y=1)
      d = n(X=0, Y=0)

    Correction is applied separately within X=1 and X=0 strata:
      true_vec = (K + lam I)^{-1} obs_vec, with obs_vec=[obs0, obs1]^T.

    Returns:
      log( (a_corr * d_corr) / (b_corr * c_corr) )
    """
    K = correction_matrix(p, q)
    inv = np.linalg.inv(K + lam * np.eye(2))

    # X=1 observed outcome vector: [obs0=b, obs1=a]
    tru1 = inv @ np.array([float(b), float(a)])
    # X=0 observed outcome vector: [obs0=d, obs1=c]
    tru0 = inv @ np.array([float(d), float(c)])

    # clip to avoid negative/zero corrected counts
    tru1 = np.clip(tru1, 1e-6, None)
    tru0 = np.clip(tru0, 1e-6, None)

    a2 = float(tru1[1])  # X=1,Y=1
    b2 = float(tru1[0])  # X=1,Y=0
    c2 = float(tru0[1])  # X=0,Y=1
    d2 = float(tru0[0])  # X=0,Y=0

    return float(np.log((a2 * d2) / (b2 * c2)))

#==========================================================================



#   ap.add_argument("--out_dir", type=str, default="./naive_blob_outputs")
#    ap.add_argument("--seed", type=int, default=1)
#
#    ap.add_argument("--n", type=int, default=100000,
#                    help="Large n makes tiny effects very significant (vertical blob).")
#    ap.add_argument("--m", type=int, default=60000,
#                    help="Number of exposures/features scanned.")
#
#    ap.add_argument("--eta0", type=float, default=-1.6,
#                    help="Outcome intercept.")
#    ap.add_argument("--eta1", type=float, default=4.5,
#                    help="Z->Y strength.")
#
#    ap.add_argument("--alpha_mean", type=float, default=-8.2,
#                    help="Mean exposure intercept.")
#    ap.add_argument("--alpha_sd", type=float, default=10.0,
#                    help="Large alpha_sd creates huge prevalence heterogeneity (drives blob).")
#
#    ap.add_argument("--gamma", type=float, default=0.005,
#                    help="Constant small Z->X coupling for ALL features. Smaller -> tighter x blob.")
#    ap.add_argument("--signed", action="store_true",
#                    help="If set, random sign per feature (two-sided blob around 0).")
#
#    ap.add_argument("--min_prev", type=float, default=0.0001,
#                    help="Clip exposure probabilities from below (avoid ultra-rare).")
#    ap.add_argument("--max_prev", type=float, default=5,
#                    help="Clip exposure probabilities from above.")
#
#    ap.add_argument("--p", type=float, default=0.975)
#    ap.add_argument("--q", type=float, default=0.65)
#    ap.add_argument("--lam", type=float, default=0.05)

    
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, default="./naive_blob_outputs")
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--n", type=int, default=10000,
                    help="Large n makes tiny effects very significant (vertical blob).")
    ap.add_argument("--m", type=int, default=6000,
                    help="Number of exposures/features scanned.")

    ap.add_argument("--eta0", type=float, default=-1.6,
                    help="Outcome intercept.")
    ap.add_argument("--eta1", type=float, default=3,
                    help="Z->Y strength.")

    ap.add_argument("--alpha_mean", type=float, default=-8.2,
                    help="Mean exposure intercept.")
    ap.add_argument("--alpha_sd", type=float, default=20.0,
                    help="Large alpha_sd creates huge prevalence heterogeneity (drives blob).")

    ap.add_argument("--gamma", type=float, default=0.2,
                    help="Constant small Z->X coupling for ALL features. Smaller -> tighter x blob.")
    ap.add_argument("--signed", action="store_true",
                    help="If set, random sign per feature (two-sided blob around 0).")

    ap.add_argument("--min_prev", type=float, default=0.0001,
                    help="Clip exposure probabilities from below (avoid ultra-rare).")
    ap.add_argument("--max_prev", type=float, default=5,
                    help="Clip exposure probabilities from above.")

    ap.add_argument("--p", type=float, default=0.875)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--lam", type=float, default=0.05)

    args = ap.parse_args()
    
    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    n, m = args.n, args.m

    # Latent Z and outcome Y
    Z = rng.normal(0.0, 1.0, size=n)
    pY = expit(args.eta0 + args.eta1 * Z)
    Y = (rng.random(n) < pY).astype(np.int8)

    # Feature intercepts with very wide spread -> prevalence heterogeneity -> SE heterogeneity
    alpha = args.alpha_mean + rng.normal(0.0, args.alpha_sd, size=m)

    # Constant tiny coupling; optionally random sign per feature
    gamma = np.full(m, args.gamma, dtype=float)
    if args.signed:
        gamma *= rng.choice([-1.0, 1.0], size=m)

    logor = np.empty(m, dtype=float)
    se = np.empty(m, dtype=float)
    pval = np.empty(m, dtype=float)
    prev = np.empty(m, dtype=float)

    # Generate each X_j on the fly (no n x m allocation)
    u = rng.random(n)  # reused buffer for speed; we will refresh per feature

    logor_lambda = np.empty(m, dtype=float)

    for j in range(m):
        # p(X=1|Z) = sigmoid(alpha_j + gamma_j Z)
        pX = expit(alpha[j] + gamma[j] * Z)
        pX = np.clip(pX, args.min_prev, args.max_prev)

        # refresh randomness per feature without allocating a new vector
        u[:] = rng.random(n)
        Xj = (u < pX).astype(np.int8)

        prev[j] = float(Xj.mean())
        a, b, c, d = counts_2x2(Xj, Y)
        lo, s, p = logor_wald_p(a, b, c, d, ha=0.5)

        logor_lambda[j] = lambda_corrected_logor_from_counts(a, b, c, d, p=args.p, q=args.q, lam=args.lam)

        
        logor[j] = lo
        se[j] = s
        pval[j] = p

    df = pd.DataFrame({
        "feature": np.arange(m),
        "logOR": logor,
        "se_wald": se,
        "p_wald": pval,
        "neglog10p": -np.log10(np.clip(pval, 1e-300, 1.0)),
        "alpha": alpha,
        "gamma": gamma,
        "prev": prev,
    })

    out_csv = os.path.join(args.out_dir, "scan_naive_blob.csv")
    df.to_csv(out_csv, index=False)

    out_png = os.path.join(args.out_dir, "volcano_naive_blob.png")
    title = f"Naive blob volcano: n={n}, m={m}, gamma={args.gamma}, alpha_sd={args.alpha_sd}"
    volcano_plot(logor, pval, out_png, title)

    med_abs = float(np.median(np.abs(logor[np.isfinite(logor)])))
    q90_abs = float(np.quantile(np.abs(logor[np.isfinite(logor)]), 0.90))
    frac_hi = float(np.mean(np.isfinite(pval) & (pval < 1e-12)))

    print("Run summary:")
    print(f"  n={n}, m={m}, seed={args.seed}")
    print(f"  eta0={args.eta0}, eta1={args.eta1}")
    print(f"  alpha_mean={args.alpha_mean}, alpha_sd={args.alpha_sd}")
    print(f"  gamma={args.gamma}, signed={bool(args.signed)}")
    print(f"  median |logOR|={med_abs:.6f}, 90% |logOR|={q90_abs:.6f}")
    print(f"  fraction p<1e-12 = {frac_hi:.3f}")
    print(f"Outputs written to: {os.path.abspath(args.out_dir)}")
    print(f"  - {out_csv}")
    print(f"  - {out_png}")

    
    out_png2 = os.path.join(args.out_dir, "volcano_lambda.png")
    title2 = f"Lambda OR volcano: lambda={args.lam}, detK={args.p+args.q-1:.3f}"
    volcano_plot(logor_lambda, pval, out_png2, title2)

if __name__ == "__main__":
    main()
