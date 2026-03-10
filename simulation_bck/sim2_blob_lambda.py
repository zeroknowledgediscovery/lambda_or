#!/usr/bin/env python3
"""
Blobified naive volcano plus lambda-OR volcano, without changing the DGP.

DGP (unchanged from your sim2.py):
  - Z ~ N(0,1)
  - Y ~ Bernoulli(sigmoid(eta0 + eta1*Z))
  - For each feature j:
      alpha_j ~ N(alpha_mean, alpha_sd^2)
      gamma_j = +/- gamma (optional signed)
      X_j ~ Bernoulli(sigmoid(alpha_j + gamma_j*Z)) with clipping

To keep the plot blob-like:
  - y-axis uses a chi-square p-value computed on the observed (X, Ytil) 2x2 table
  - naive x-axis uses a shrunken log(OR) with a large pseudo-count (pseudo)
  - lambda x-axis uses a ridge-stabilized inverse of the misclassification operator (K + lam I)^{-1}

Outcome misclassification:
  - Observed outcome Ytil is a misclassified version of Y with sensitivity p and specificity q.

Outputs:
  - volcano_naive_blob.png
  - volcano_lambda.png
  - scan_blob_lambda.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import chi2


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def apply_misclassification(y: np.ndarray, p: float, q: float, rng: np.random.Generator) -> np.ndarray:
    """P(Ytil=1|Y=1)=p, P(Ytil=0|Y=0)=q."""
    y = y.astype(np.int8)
    u = rng.random(y.shape[0])
    ytil = y.copy()

    m1 = (y == 1)
    ytil[m1] = (u[m1] <= p).astype(np.int8)

    m0 = (y == 0)
    ytil[m0] = (u[m0] > q).astype(np.int8)

    return ytil


def counts_2x2(x: np.ndarray, y: np.ndarray):
    x = x.astype(np.int8)
    y = y.astype(np.int8)
    a = int(np.sum((x == 1) & (y == 1)))
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    d = int(np.sum((x == 0) & (y == 0)))
    return a, b, c, d


def logor_shrunken(a: int, b: int, c: int, d: int, pseudo: float) -> float:
    a2, b2, c2, d2 = a + pseudo, b + pseudo, c + pseudo, d + pseudo
    return float(np.log((a2 * d2) / (b2 * c2)))


def chisq_pvalue(a: int, b: int, c: int, d: int):
    """Pearson chi-square test for 2x2 independence, df=1."""
    n = a + b + c + d
    if n <= 0:
        return np.nan, np.nan

    row1 = a + b
    row0 = c + d
    col1 = a + c
    col0 = b + d

    ea = row1 * col1 / n
    eb = row1 * col0 / n
    ec = row0 * col1 / n
    ed = row0 * col0 / n

    eps = 1e-12
    chi = ((a - ea) ** 2) / max(ea, eps) + ((b - eb) ** 2) / max(eb, eps) + \
          ((c - ec) ** 2) / max(ec, eps) + ((d - ed) ** 2) / max(ed, eps)

    p = float(chi2.sf(chi, df=1))
    return float(chi), p


def correction_matrix(p: float, q: float) -> np.ndarray:
    """
    Outcome axis [Y=0, Y=1]^T:
      [obs0]   [ q      (1-p) ] [true0]
      [obs1] = [ (1-q)   p    ] [true1]
    """
    return np.array([[q, 1.0 - p],
                     [1.0 - q, p]], dtype=float)


def lambda_corrected_logor(a: int, b: int, c: int, d: int, p: float, q: float, lam: float) -> float:
    """
    Observed table (X by Ytil):
      X=1: (Ytil=1)=a, (Ytil=0)=b
      X=0: (Ytil=1)=c, (Ytil=0)=d

    Correct outcome counts separately within X=1 and X=0 using (K + lam I)^{-1},
    then compute log(OR) on corrected counts.
    """
    K = correction_matrix(p, q)
    inv = np.linalg.inv(K + lam * np.eye(2))

    tru1 = inv @ np.array([b, a], dtype=float)  # X=1: [obs0, obs1]
    tru0 = inv @ np.array([d, c], dtype=float)  # X=0: [obs0, obs1]

    tru1 = np.clip(tru1, 1e-6, None)
    tru0 = np.clip(tru0, 1e-6, None)

    a2 = float(tru1[1])
    b2 = float(tru1[0])
    c2 = float(tru0[1])
    d2 = float(tru0[0])

    return float(np.log((a2 * d2) / (b2 * c2)))


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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, default="./blob_lambda_outputs")
    ap.add_argument("--seed", type=int, default=1)

    # DGP parameters (kept the same defaults as your sim2.py)
    ap.add_argument("--n", type=int, default=90000,
                    help="Large n makes tiny effects very significant (vertical blob).")
    ap.add_argument("--m", type=int, default=3000,
                    help="Number of exposures/features scanned.")

    ap.add_argument("--eta0", type=float, default=-1.6,
                    help="Outcome intercept.")
    ap.add_argument("--eta1", type=float, default=4.5,
                    help="Z->Y strength.")

    ap.add_argument("--alpha_mean", type=float, default=-8.2,
                    help="Mean exposure intercept.")
    ap.add_argument("--alpha_sd", type=float, default=10.0,
                    help="Large alpha_sd creates huge prevalence heterogeneity (drives blob).")

    ap.add_argument("--gamma", type=float, default=0.005,
                    help="Constant small Z->X coupling for ALL features. Smaller -> tighter x blob.")
    ap.add_argument("--signed", action="store_true",
                    help="If set, random sign per feature (two-sided blob around 0).")

    ap.add_argument("--min_prev", type=float, default=0.0001,
                    help="Clip exposure probabilities from below (avoid ultra-rare).")
    ap.add_argument("--max_prev", type=float, default=5.0,
                    help="Clip exposure probabilities from above.")

    # Blobification control for naive x-axis
    ap.add_argument("--pseudo", type=float, default=200.0,
                    help="Large pseudo-count shrinks naive log(OR) toward 0 (blob).")

    # Misclassification and lambda-OR parameters
    ap.add_argument("--p", type=float, default=0.75, help="Sensitivity for Y -> Ytil.")
    ap.add_argument("--q", type=float, default=0.65, help="Specificity for Y -> Ytil.")
    ap.add_argument("--lam", type=float, default=0.05, help="Ridge lambda for (K + lam I)^{-1}.")

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    n, m = args.n, args.m

    # Latent Z and true outcome Y
    Z = rng.normal(0.0, 1.0, size=n)
    pY = expit(args.eta0 + args.eta1 * Z)
    Y = (rng.random(n) < pY).astype(np.int8)

    # Misclassified observed outcome
    Ytil = apply_misclassification(Y, args.p, args.q, rng)

    # Feature intercepts with wide spread -> prevalence heterogeneity
    alpha = args.alpha_mean + rng.normal(0.0, args.alpha_sd, size=m)

    # Constant tiny coupling; optionally random sign per feature
    gamma = np.full(m, args.gamma, dtype=float)
    if args.signed:
        gamma *= rng.choice([-1.0, 1.0], size=m)

    # Results
    x_naive_blob = np.empty(m, dtype=float)
    x_lambda = np.empty(m, dtype=float)
    chi = np.empty(m, dtype=float)
    pval = np.empty(m, dtype=float)
    prev = np.empty(m, dtype=float)

    # Generate each X_j on the fly (no n x m allocation)
    u = rng.random(n)  # reused buffer for speed

    for j in range(m):
        pX = expit(alpha[j] + gamma[j] * Z)
        pX = np.clip(pX, args.min_prev, args.max_prev)

        u[:] = rng.random(n)
        Xj = (u < pX).astype(np.int8)

        prev[j] = float(Xj.mean())
        a, b, c, d = counts_2x2(Xj, Ytil)

        # y-axis: chi-square p-value on observed (X, Ytil)
        chi[j], pval[j] = chisq_pvalue(a, b, c, d)

        # x-axis (naive blob): shrunken logOR on observed table
        x_naive_blob[j] = logor_shrunken(a, b, c, d, pseudo=args.pseudo)

        # x-axis (lambda): corrected logOR using p,q,lam
        x_lambda[j] = lambda_corrected_logor(a, b, c, d, p=args.p, q=args.q, lam=args.lam)

    df = pd.DataFrame({
        "feature": np.arange(m),
        "logOR_naive_blob": x_naive_blob,
        "logOR_lambda": x_lambda,
        "chisq": chi,
        "p_chisq": pval,
        "neglog10p": -np.log10(np.clip(pval, 1e-300, 1.0)),
        "alpha": alpha,
        "gamma": gamma,
        "prev": prev,
    })

    out_csv = os.path.join(args.out_dir, "scan_blob_lambda.csv")
    df.to_csv(out_csv, index=False)

    detK = args.p + args.q - 1.0

    out_naive = os.path.join(args.out_dir, "volcano_naive_blob.png")
    volcano_plot(
        x_naive_blob,
        pval,
        out_naive,
        title=f"Naive blob volcano: n={n}, m={m}, pseudo={args.pseudo:g}",
        xlabel=f"log(OR) with pseudo={args.pseudo:g}"
    )

    out_lam = os.path.join(args.out_dir, "volcano_lambda.png")
    volcano_plot(
        x_lambda,
        pval,
        out_lam,
        title=f"Lambda volcano: det(K)={detK:.3f}, lam={args.lam:g}",
        xlabel="lambda-corrected log(OR)"
    )

    print("Run summary:")
    print(f"  n={n}, m={m}, seed={args.seed}, signed={bool(args.signed)}")
    print(f"  DGP: eta0={args.eta0}, eta1={args.eta1}, alpha_mean={args.alpha_mean}, alpha_sd={args.alpha_sd}, gamma={args.gamma}")
    print(f"  Misclass: p={args.p}, q={args.q}, det(K)={detK:.3f}, lam={args.lam}")
    print(f"  Naive x shrink pseudo={args.pseudo}")
    print(f"Outputs written to: {os.path.abspath(args.out_dir)}")
    print(f"  - {out_csv}")
    print(f"  - {out_naive}")
    print(f"  - {out_lam}")


if __name__ == "__main__":
    main()
