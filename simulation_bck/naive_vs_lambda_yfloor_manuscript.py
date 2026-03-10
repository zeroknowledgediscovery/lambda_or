#!/usr/bin/env python3
"""
Naive vs Lambda-OR y-axis deflation demo, with var_extra COMPUTED (not assumed).

We simulate true 2x2 tables for many features (ICD-like heterogeneity), then apply
nondifferential outcome misclassification with known (p,q) to generate observed tables.
Naive: compute logOR + Wald p-values on observed outcome (Y~).
Lambda-OR: ridge-invert misclassification operator to estimate corrected table,
           compute delta-method sampling variance from observed counts PLUS
           an extra variance contribution from estimating (p,q) on a validation set
           of size n_val:
               Var_extra = g_p^2 Var(p_hat) + g_q^2 Var(q_hat),
           where g_p, g_q are sensitivities of logOR to (p,q) under the correction
           (implemented via closed-form derivative of K^{-1}).

This produces: naive volcano with ridiculous y at huge n, lambda volcano with y deflated
because Var_extra does not shrink with per-feature n_j.
"""

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def ensure_dir(d): os.makedirs(d, exist_ok=True)
def expit(x): return 1/(1+np.exp(-x))

def logor_wald(a,b,c,d,ha=0.5):
    a=a.astype(float)+ha; b=b.astype(float)+ha; c=c.astype(float)+ha; d=d.astype(float)+ha
    logor=np.log((a*d)/(b*c))
    se=np.sqrt(1/a + 1/b + 1/c + 1/d)
    z=logor/se
    p=2*norm.sf(np.abs(z))
    return logor,se,p

def Kmat(p,q):
    # mapping true -> observed for outcome label (neg,pos) ordering
    return np.array([[q, 1-p],[1-q, p]], dtype=float)

def inv_ridge(p,q,lam):
    return np.linalg.inv(Kmat(p,q) + lam*np.eye(2))

def dK_dp():
    # derivative of K wrt p
    return np.array([[0, -1],[0, 1]], dtype=float)

def dK_dq():
    # derivative of K wrt q
    return np.array([[1, 0],[-1, 0]], dtype=float)

def apply_misclass_counts(a_true,b_true,c_true,d_true,p,q,rng):
    """
    True table:
      cases: exposed=a_true, unexposed=c_true
      ctrls: exposed=b_true, unexposed=d_true
    Misclassify outcome: observed cases include true cases kept (p) + true ctrls flipped (1-q).
    Return observed table (aT,bT,cT,dT) under Y~ strata.
    """
    aT = rng.binomial(a_true, p) + rng.binomial(b_true, 1-q)
    n_case_obs = rng.binomial(a_true + c_true, p) + rng.binomial(b_true + d_true, 1-q)
    cT = n_case_obs - aT

    bT = rng.binomial(b_true, q) + rng.binomial(a_true, 1-p)
    n_ctrl_obs = (a_true+c_true+b_true+d_true) - n_case_obs
    dT = n_ctrl_obs - bT
    return aT,bT,cT,dT

def lambda_or_and_var(aT,bT,cT,dT,p,q,lam,n_val,ha=0.5):
    """
    Ridge-correct observed stratum-specific exposure counts by inverting (K+lam I).
    Then compute:
      - corrected logOR
      - delta-method sampling variance from observed counts (binomial within strata)
      - var_extra from estimating (p,q) on validation set of size n_val using delta method:
            d(logOR)/dp and d(logOR)/dq computed analytically via dA = -A (dK) A
    """
    A = inv_ridge(p,q,lam)

    # observed exposure counts within observed outcome strata as 2-vectors [neg,pos]
    obs1 = np.vstack([bT.astype(float), aT.astype(float)])  # among observed cases
    obs0 = np.vstack([dT.astype(float), cT.astype(float)])  # among observed ctrls

    tru1 = np.clip(A @ obs1, 1e-6, None)
    tru0 = np.clip(A @ obs0, 1e-6, None)

    b_hat, a_hat = tru1[0,:], tru1[1,:]
    d_hat, c_hat = tru0[0,:], tru0[1,:]
    logor = np.log((a_hat*d_hat)/(b_hat*c_hat))

    # Sampling variance (delta) from observed exposure counts within each stratum.
    n1 = np.maximum((aT+bT).astype(float), 1.0)
    n0 = np.maximum((cT+dT).astype(float), 1.0)
    pa = aT.astype(float)/n1; pb = bT.astype(float)/n1
    pc = cT.astype(float)/n0; pd = dT.astype(float)/n0
    v1 = n1*(pa*pb)  # for 2-cat cov rank-1
    v0 = n0*(pc*pd)

    u = (A[:,0]-A[:,1])  # because Cov([neg,pos]) = v [[1,-1],[-1,1]]

    dot1 = (-1.0/b_hat)*u[0] + (1.0/a_hat)*u[1]
    dot0 = ( 1.0/d_hat)*u[0] + (-1.0/c_hat)*u[1]
    var_counts = v1*(dot1**2) + v0*(dot0**2)

    # Extra variance from (p,q) estimation on validation set.
    n_val = max(int(n_val), 1)
    var_p = p*(1-p)/n_val
    var_q = q*(1-q)/n_val

    # dA/dtheta = -A (dK/dtheta) A
    dA_p = -A @ dK_dp() @ A
    dA_q = -A @ dK_dq() @ A

    # Sensitivity of corrected counts to theta: d(tru) = dA * obs  (obs treated fixed for this component)
    dtru1_p = dA_p @ obs1
    dtru0_p = dA_p @ obs0
    dtru1_q = dA_q @ obs1
    dtru0_q = dA_q @ obs0

    # unpack derivatives for each feature
    db_p, da_p = dtru1_p[0,:], dtru1_p[1,:]
    dd_p, dc_p = dtru0_p[0,:], dtru0_p[1,:]
    db_q, da_q = dtru1_q[0,:], dtru1_q[1,:]
    dd_q, dc_q = dtru0_q[0,:], dtru0_q[1,:]

    # derivative of logOR = log(a)+log(d)-log(b)-log(c)
    dlogor_dp = (da_p/a_hat) + (dd_p/d_hat) - (db_p/b_hat) - (dc_p/c_hat)
    dlogor_dq = (da_q/a_hat) + (dd_q/d_hat) - (db_q/b_hat) - (dc_q/c_hat)

    var_extra = (dlogor_dp**2)*var_p + (dlogor_dq**2)*var_q

    var_total = var_counts + var_extra
    se = np.sqrt(var_total)
    z = logor/se
    pval = 2*norm.sf(np.abs(z))
    eps = p+q-1.0
    return logor, se, pval, var_counts, var_extra, eps

def volcano(x, p, out_png, title, xlim=(-0.5,0.5), ylim=None):
    y=-np.log10(np.clip(p, 1e-300, 1.0))
    plt.figure(figsize=(8.0,6.0))
    plt.scatter(x, y, s=6, alpha=0.55)
    plt.axvline(0, color="black", lw=1)
    plt.axhline(-np.log10(0.05), color="black", lw=1, ls="--")
    plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("log(OR)")
    plt.ylabel("-log10(p)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./yfloor_manuscript_demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m", type=int, default=20000)
    ap.add_argument("--n_case", type=int, default=400000)
    ap.add_argument("--n_ctrl", type=int, default=400000)
    ap.add_argument("--prev0_mean", type=float, default=-3.3)
    ap.add_argument("--prev_sd", type=float, default=1.0)
    ap.add_argument("--beta_mean", type=float, default=0.06)
    ap.add_argument("--beta_sd", type=float, default=0.03)
    ap.add_argument("--p", type=float, default=0.76)
    ap.add_argument("--q", type=float, default=0.26)
    ap.add_argument("--lam", type=float, default=0.12)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--xlim", type=float, default=0.6)
    args=ap.parse_args()
    ensure_dir(args.out_dir)
    rng=np.random.default_rng(args.seed)

    m=args.m
    # True effects mostly positive
    beta = np.abs(rng.normal(args.beta_mean, args.beta_sd, size=m))
    logit_prev0 = rng.normal(args.prev0_mean, args.prev_sd, size=m)
    prev0 = expit(logit_prev0)
    prev1 = expit(logit_prev0 + beta)

    a_true = rng.binomial(args.n_case, prev1)
    b_true = rng.binomial(args.n_ctrl, prev0)
    c_true = args.n_case - a_true
    d_true = args.n_ctrl - b_true

    log_true, se_true, p_true = logor_wald(a_true,b_true,c_true,d_true)

    aT,bT,cT,dT = apply_misclass_counts(a_true,b_true,c_true,d_true,args.p,args.q,rng)
    log_naive, se_naive, p_naive = logor_wald(aT,bT,cT,dT)

    log_lam, se_lam, p_lam, var_counts, var_extra, eps = lambda_or_and_var(aT,bT,cT,dT,args.p,args.q,args.lam,args.n_val)

    # common y-limit for comparability
    y_naive = -np.log10(np.clip(p_naive, 1e-300, 1.0))
    ymax = float(np.quantile(y_naive[np.isfinite(y_naive)], 0.995))
    ymax = max(30.0, min(ymax, 400.0))
    xlim = (-args.xlim, args.xlim)

    volcano(log_naive, p_naive, os.path.join(args.out_dir,"volcano_naive.png"),
            f"Naive OR on observed Y~ (n huge) det(K)={eps:.3f}", xlim=xlim, ylim=(0,ymax))
    volcano(log_lam, p_lam, os.path.join(args.out_dir,"volcano_lambda.png"),
            f"Lambda-OR (ridge + var_extra from n_val={args.n_val}) det(K)={eps:.3f}", xlim=xlim, ylim=(0,ymax))

    df = pd.DataFrame({
        "beta_true": beta,
        "prev0": prev0, "prev1": prev1,
        "logOR_true": log_true, "p_true": p_true,
        "logOR_naive": log_naive, "p_naive": p_naive,
        "logOR_lambda": log_lam, "p_lambda": p_lam,
        "se_naive": se_naive, "se_lambda": se_lam,
        "var_counts_lambda": var_counts, "var_extra_lambda": var_extra
    })
    df.to_csv(os.path.join(args.out_dir,"summary.csv"), index=False)

    print(f"det(K)={eps:.6f}")
    print(f"median(var_extra)={np.median(var_extra):.4g}  median(var_counts)={np.median(var_counts):.4g}")
    print("Wrote:", os.path.abspath(args.out_dir))

if __name__=="__main__":
    main()
