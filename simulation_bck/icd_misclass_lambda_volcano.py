#!/usr/bin/env python3
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
    tp1 = rng.binomial(a, p)
    fp1 = rng.binomial(b, 1.0-q)
    aT = tp1 + fp1
    bT = (a - tp1) + (b - fp1)

    tp0 = rng.binomial(c, p)
    fp0 = rng.binomial(d, 1.0-q)
    cT = tp0 + fp0
    dT = (c - tp0) + (d - fp0)
    return aT,bT,cT,dT

def Kmat(p,q):
    return np.array([[q, 1.0-p],
                     [1.0-q, p]], dtype=float)

def lambda_correct_logor(aT,bT,cT,dT,p,q,lam):
    inv = np.linalg.inv(Kmat(p,q) + lam*np.eye(2))
    obs1 = np.stack([bT.astype(float), aT.astype(float)], axis=0)
    obs0 = np.stack([dT.astype(float), cT.astype(float)], axis=0)
    tru1 = inv @ obs1
    tru0 = inv @ obs0
    tru1 = np.clip(tru1, 1e-6, None)
    tru0 = np.clip(tru0, 1e-6, None)
    a_hat = tru1[1,:]; b_hat = tru1[0,:]
    c_hat = tru0[1,:]; d_hat = tru0[0,:]
    return np.log((a_hat*d_hat)/(b_hat*c_hat))


def lambda_correct_stats(aT,bT,cT,dT,p,q,lam,n_val):
    """
    Lambda-OR stats with variance propagation through the linear correction (delta method on counts)
    plus the manuscript Var_extra term for estimating (p,q).

    Steps:
      1) Correct observed 2-vectors within X=1 and X=0 strata:
           tru1 = A [bT,aT], tru0 = A [dT,cT],  A=(K+lam I)^{-1}
      2) Approximate Cov([bT,aT]) and Cov([dT,cT]) as multinomial with fixed totals n1, n0
         using plug-in probabilities from observed tables.
      3) Propagate covariance: Cov(tru) = A Cov(obs) A^T
      4) Delta method for logOR = log(a)+log(d)-log(b)-log(c):
           grad1 = [-1/b, 1/a] for tru1=[b,a]
           grad0 = [ 1/d,-1/c] for tru0=[d,c]
         Var_counts = grad1^T Cov(tru1) grad1 + grad0^T Cov(tru0) grad0
      5) Add Var_extra (manuscript) from estimating (p,q) with size n_val.
    """
    A = np.linalg.inv(Kmat(p,q) + lam*np.eye(2))

    # Observed vectors (2,m)
    obs1 = np.stack([bT.astype(float), aT.astype(float)], axis=0)  # [b,a]
    obs0 = np.stack([dT.astype(float), cT.astype(float)], axis=0)  # [d,c]

    tru1 = A @ obs1
    tru0 = A @ obs0

    # Feasible pseudo-counts
    tru1 = np.clip(tru1, 1e-6, None)
    tru0 = np.clip(tru0, 1e-6, None)

    b_hat = tru1[0,:]; a_hat = tru1[1,:]
    d_hat = tru0[0,:]; c_hat = tru0[1,:]

    logor = np.log((a_hat*d_hat)/(b_hat*c_hat))

    # ---- Covariance propagation for counts (plug-in multinomial) ----
    n1 = (aT + bT).astype(float)
    n0 = (cT + dT).astype(float)
    n1 = np.maximum(n1, 1.0)
    n0 = np.maximum(n0, 1.0)

    # plug-in probs for obs vectors
    pb = (bT.astype(float) / n1)
    pa = (aT.astype(float) / n1)
    pd = (dT.astype(float) / n0)
    pc = (cT.astype(float) / n0)

    # Cov matrices for each j are 2x2:
    # Cov([b,a]) = n1 * [[pb(1-pb), -pb pa],[-pb pa, pa(1-pa)]]
    # For binary multinomial pb+pa=1 so pb(1-pb)=pb pa and pa(1-pa)=pb pa, cov=-pb pa
    v1 = n1 * (pb * pa)
    v0 = n0 * (pd * pc)

    # For each j, Cov_obs1 = [[v1, -v1],[-v1, v1]]
    # Propagate through A: Cov_tru1 = A Cov_obs1 A^T, similarly for tru0.
    # We can do this without building full (m,2,2) tensors by closed form.
    # If Cov = v * [[1,-1],[-1,1]], then Cov_tru = v * (A[:,0]-A[:,1]) (A[:,0]-A[:,1])^T
    u = (A[:,0] - A[:,1]).reshape(2,1)  # 2x1
    # Cov_tru for each j: v_j * (u u^T)
    # Delta method: grad^T Cov_tru grad = v_j * (grad^T u)^2
    g1_0 = -1.0 / b_hat
    g1_1 =  1.0 / a_hat
    dot1 = g1_0 * u[0,0] + g1_1 * u[1,0]
    var1 = v1 * (dot1**2)

    g0_0 =  1.0 / d_hat
    g0_1 = -1.0 / c_hat
    dot0 = g0_0 * u[0,0] + g0_1 * u[1,0]
    var0 = v0 * (dot0**2)

    var_counts = var1 + var0

    # ---- Manuscript Var_extra for estimating (p,q) ----
    detK = (p + q - 1.0)
    detK2 = np.maximum(detK*detK, 1e-12)

    n_val = max(int(n_val), 1)
    var_extra = ((1.0 - q)/detK2)**2 * (p*(1.0-p))/n_val + ((1.0 - p)/detK2)**2 * (q*(1.0-q))/n_val

    var_total = var_counts + var_extra
    se = np.sqrt(var_total)
    z = logor / se
    pval = 2.0 * norm.sf(np.abs(z))
    return logor, se, pval, var_counts, var_extra

def volcano(x, p, c, out_png, title, c_label="|true logOR| (ref)"):
    y = -np.log10(np.clip(p, 1e-300, 1.0))
    plt.figure(figsize=(8.2, 6.2))
    sc = plt.scatter(x, y, c=c, s=4, alpha=0.6, cmap="RdBu_r")
    plt.colorbar(sc, label=c_label)
    plt.axvline(0.0, color="black", lw=1)
    plt.axhline(-np.log10(0.05), ls="--", lw=1)
    plt.xlim(-1, 1)          # force identical x scale
    plt.ylim(0, 40)          # force identical x scale
    plt.xlabel("log(OR)")
    plt.ylabel("-log10(p)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def bh_threshold(pvals, q=0.05):
    p = np.asarray(pvals)
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    thresh_line = q * (np.arange(1, m+1) / m)
    ok = p_sorted <= thresh_line
    if not np.any(ok):
        return None
    k = int(np.max(np.where(ok)[0]))
    return float(p_sorted[k])

def fdr_tpr(pvals, is_signal, alpha):
    called = (pvals < alpha)
    tp = int(np.sum(called & is_signal))
    fp = int(np.sum(called & (~is_signal)))
    disc = tp + fp
    fdr = (fp / disc) if disc > 0 else np.nan
    tpr = tp / max(int(np.sum(is_signal)), 1)
    return tp, fp, disc, fdr, tpr
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./icd_lambda_volcano_out")
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--m", type=int, default=30000)
    ap.add_argument("--N_cases", type=int, default=20000)
    ap.add_argument("--N_ctrl", type=int, default=20000)

    ap.add_argument("--alpha_mean", type=float, default=-6.5)
    ap.add_argument("--alpha_sd", type=float, default=2.2)
    ap.add_argument("--min_prev", type=float, default=1e-6)
    ap.add_argument("--max_prev", type=float, default=0.20)

    ap.add_argument("--beta_sd", type=float, default=0.05)
    ap.add_argument("--signal_frac", type=float, default=0.20)
    ap.add_argument("--signal_mult", type=float, default=6.0)

    ap.add_argument("--p", type=float, default=0.80)
    ap.add_argument("--q", type=float, default=0.70)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--n_val", type=int, default=4000, help="Validation size for estimating (p,q) in Var_extra (manuscript).")

    ap.add_argument("--N_cases_big", type=int, default=500000)
    ap.add_argument("--N_ctrl_big", type=int, default=500000)
    ap.add_argument("--n_exposed_big", type=int, default=20000)

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    m = args.m
    Nc = args.N_cases
    N0 = args.N_ctrl

    alpha = rng.normal(args.alpha_mean, args.alpha_sd, size=m)
    pi0 = 1.0/(1.0+np.exp(-alpha))
    pi0 = np.clip(pi0, args.min_prev, args.max_prev)

    beta = rng.normal(0.0, args.beta_sd, size=m)
    is_signal = rng.random(m) < args.signal_frac
    # Signals have larger variance effects (many high-|OR| points)
    beta[is_signal] = rng.normal(0.0, args.beta_sd * args.signal_mult, size=is_signal.sum())

    logit0 = np.log(pi0/(1.0-pi0))
    pi1 = 1.0/(1.0+np.exp(-(logit0 + beta)))
    pi1 = np.clip(pi1, args.min_prev, args.max_prev)

    a = rng.binomial(Nc, pi1).astype(np.int64)
    c = (Nc - a).astype(np.int64)
    b = rng.binomial(N0, pi0).astype(np.int64)
    d = (N0 - b).astype(np.int64)

    n_exposed = (a + b).astype(np.int64)

    logor_true, se_true, p_true = logor_wald_vec(a,b,c,d,ha=0.5)

    aT,bT,cT,dT = misclassify_tables(a,b,c,d,args.p,args.q,rng)
    logor_noisy, se_noisy, p_noisy = logor_wald_vec(aT,bT,cT,dT,ha=0.5)
    logor_lambda, se_lambda, p_lambda, var_counts_lam, var_extra_lam = lambda_correct_stats(aT,bT,cT,dT,args.p,args.q,args.lam,args.n_val)

    # Reference clean parabola panel with fixed exposed count
    Nc_big = args.N_cases_big
    N0_big = args.N_ctrl_big
    n1 = args.n_exposed_big
    OR = np.exp(beta)

    def solve_a_for_or(or_val, Nc, N0, n1):
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
        cand = [a for a in (a1,a2) if 0.0 <= a <= n1]
        if cand:
            return min(cand, key=lambda x: abs(x-target))
        return float(np.clip(a1, 0.0, n1))

    a_big = np.zeros(m, dtype=np.int64)
    b_big = np.zeros(m, dtype=np.int64)
    for j in range(m):
        a_est = solve_a_for_or(float(OR[j]), Nc_big, N0_big, n1)
        a_int = int(np.round(a_est))
        a_int = max(1, min(a_int, n1-1, Nc_big-1))
        b_int = n1 - a_int
        b_int = max(1, min(b_int, N0_big-1))
        a_int = n1 - b_int
        a_int = max(1, min(a_int, Nc_big-1))
        a_big[j] = a_int
        b_big[j] = b_int

    c_big = (Nc_big - a_big).astype(np.int64)
    d_big = (N0_big - b_big).astype(np.int64)
    logor_ref, se_ref, p_ref = logor_wald_vec(a_big, b_big, c_big, d_big, ha=0.5)

    c_eff = np.abs(logor_ref)

    volcano(logor_ref, p_ref, c_eff,
            os.path.join(args.out_dir, "volcano_true.png"),
            f"True volcano (equal-n_j ref; n_exposed={n1})")

    detK = args.p + args.q - 1.0
    volcano(logor_noisy, p_noisy, c_eff,
            os.path.join(args.out_dir, "volcano_noisy_naive.png"),
            f"Noisy naive volcano (ICD-like n_j; detK={detK:.3f})")

    volcano(logor_lambda, p_lambda, c_eff,
            os.path.join(args.out_dir, "volcano_lambda.png"),
            f"Lambda-OR volcano (x corrected; lam={args.lam})")

    # Performance summary: FDR/TPR at multiple alpha levels + BH(0.05)
    alpha_grid = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05])
    rows = []
    for alevel in alpha_grid:
        tp, fp, disc, fdr, tpr = fdr_tpr(p_noisy, is_signal, alevel)
        rows.append({"method":"noisy_naive", "alpha":float(alevel), "TP":tp, "FP":fp, "discoveries":disc, "FDR":fdr, "TPR":tpr})
        tp, fp, disc, fdr, tpr = fdr_tpr(p_lambda, is_signal, alevel)
        rows.append({"method":"lambda_corrected", "alpha":float(alevel), "TP":tp, "FP":fp, "discoveries":disc, "FDR":fdr, "TPR":tpr})

    bh_noisy = bh_threshold(p_noisy, q=0.05)
    bh_lam = bh_threshold(p_lambda, q=0.05)
    if bh_noisy is not None:
        tp, fp, disc, fdr, tpr = fdr_tpr(p_noisy, is_signal, bh_noisy + 1e-18)
        rows.append({"method":"noisy_naive_BH", "alpha":bh_noisy, "TP":tp, "FP":fp, "discoveries":disc, "FDR":fdr, "TPR":tpr})
    if bh_lam is not None:
        tp, fp, disc, fdr, tpr = fdr_tpr(p_lambda, is_signal, bh_lam + 1e-18)
        rows.append({"method":"lambda_BH", "alpha":bh_lam, "TP":tp, "FP":fp, "discoveries":disc, "FDR":fdr, "TPR":tpr})

    perf = pd.DataFrame(rows)
    perf.to_csv(os.path.join(args.out_dir, "perf_fdr_tpr.csv"), index=False)

    # Plot: TPR vs FDR across alpha_grid
    base = perf[perf["method"].isin(["noisy_naive","lambda_corrected"])].copy()
    plt.figure(figsize=(6.8,5.2))
    for meth in ["noisy_naive","lambda_corrected"]:
        sub = base[base["method"]==meth].sort_values("alpha")
        plt.plot(sub["FDR"].values, sub["TPR"].values, marker="o", label=meth)
    plt.xlabel("FDR")
    plt.ylabel("TPR")
    plt.title("TPR vs FDR across alpha thresholds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "tpr_vs_fdr.png"), dpi=220)
    plt.close()

    df = pd.DataFrame({
        "j": np.arange(m),
        "is_signal": is_signal.astype(int),
        "n_exposed": n_exposed,
        "pi0": pi0,
        "pi1": pi1,
        "beta": beta,
        "logOR_ref": logor_ref,
        "p_ref": p_ref,
        "logOR_true": logor_true,
        "p_true": p_true,
        "logOR_noisy": logor_noisy,
        "p_noisy": p_noisy,
        "logOR_lambda": logor_lambda,
        "p_lambda": p_lambda,
        "se_lambda": se_lambda,
        "var_counts_lambda": var_counts_lam,
        "var_extra_lambda": var_extra_lam,
    })
    df.to_csv(os.path.join(args.out_dir, "scan.csv"), index=False)

    sig = is_signal
    def corr(x,y):
        x = x[sig]; y = y[sig]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            return float("nan")
        return float(np.corrcoef(x[ok], y[ok])[0,1])

    print("Wrote:", os.path.abspath(args.out_dir))
    print(f"Signals: {sig.mean():.4f}")
    print("Corr(ref, noisy) on signals:", corr(logor_ref, logor_noisy))
    print("Corr(ref, lambda) on signals:", corr(logor_ref, logor_lambda))

if __name__ == "__main__":
    main()
