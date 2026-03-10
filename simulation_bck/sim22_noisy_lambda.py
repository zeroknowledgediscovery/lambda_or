#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def logor_wald(a,b,c,d,ha=0.5):
    a,b,c,d = a+ha,b+ha,c+ha,d+ha
    logor = np.log((a*d)/(b*c))
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = logor/se
    p = 2*norm.sf(abs(z))
    return logor,se,p


def misclassify_counts(a,b,c,d,p,q,rng):

    tp1 = rng.binomial(a,p)
    fp1 = rng.binomial(b,1-q)
    a_t = tp1 + fp1
    b_t = (a-tp1) + (b-fp1)

    tp0 = rng.binomial(c,p)
    fp0 = rng.binomial(d,1-q)
    c_t = tp0 + fp0
    d_t = (c-tp0) + (d-fp0)

    return a_t,b_t,c_t,d_t


def correction_matrix(p,q):
    return np.array([[q,1-p],[1-q,p]])


def lambda_logor(a,b,c,d,p,q,lam):

    K = correction_matrix(p,q)
    inv = np.linalg.inv(K + lam*np.eye(2))

    tru1 = inv @ np.array([b,a])
    tru0 = inv @ np.array([d,c])

    tru1 = np.clip(tru1,1e-6,None)
    tru0 = np.clip(tru0,1e-6,None)

    a2 = tru1[1]
    b2 = tru1[0]
    c2 = tru0[1]
    d2 = tru0[0]

    return np.log((a2*d2)/(b2*c2))


def volcano(x,p,out,title):

    y = -np.log10(np.clip(p,1e-300,1))

    plt.figure(figsize=(8,6))
    plt.scatter(x,y,s=8,alpha=.6)
    plt.axvline(0,color="black",lw=1)
    plt.axhline(-np.log10(0.05),ls="--",lw=1)
    plt.xlabel("log(OR)")
    plt.ylabel("-log10(p)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out,dpi=220)
    plt.close()


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir",default="./fresh_volcano_outputs")
    ap.add_argument("--seed",type=int,default=1)

    ap.add_argument("--m",type=int,default=4000)
    ap.add_argument("--n_per_feature",type=int,default=50000)

    ap.add_argument("--beta_sd",type=float,default=.4)

    ap.add_argument("--p",type=float,default=.80)
    ap.add_argument("--q",type=float,default=.70)
    ap.add_argument("--lam",type=float,default=.05)

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    m = args.m
    n = args.n_per_feature

    beta_true = rng.normal(0,args.beta_sd,size=m)

    logor_true = np.zeros(m)
    p_true = np.zeros(m)

    logor_naive = np.zeros(m)
    p_naive = np.zeros(m)

    logor_lambda = np.zeros(m)

    for j in range(m):

        or_true = np.exp(beta_true[j])

        p11 = or_true/(1+or_true) * 0.25
        p10 = 0.25 - p11
        p01 = 0.25 - p11
        p00 = 0.5 - p11 - p10 - p01

        probs = np.array([p00,p01,p10,p11])
        probs /= probs.sum()

        d,c,b,a = rng.multinomial(n,probs)

        lo,se,pv = logor_wald(a,b,c,d)
        logor_true[j] = lo
        p_true[j] = pv

        a_t,b_t,c_t,d_t = misclassify_counts(a,b,c,d,args.p,args.q,rng)

        lo2,se2,pv2 = logor_wald(a_t,b_t,c_t,d_t)
        logor_naive[j] = lo2
        p_naive[j] = pv2

        logor_lambda[j] = lambda_logor(a_t,b_t,c_t,d_t,args.p,args.q,args.lam)

    volcano(logor_true,p_true,
            os.path.join(args.out_dir,"volcano_true.png"),
            "True volcano")

    volcano(logor_naive,p_naive,
            os.path.join(args.out_dir,"volcano_naive_noisy.png"),
            "Naive volcano (noisy table)")

    volcano(logor_lambda,p_naive,
            os.path.join(args.out_dir,"volcano_lambda.png"),
            "Lambda OR corrected")

    plt.figure(figsize=(6,6))
    plt.scatter(logor_true,logor_naive,s=6,alpha=.5,label="naive")
    plt.scatter(logor_true,logor_lambda,s=6,alpha=.5,label="lambda")
    lim = np.max(np.abs(logor_true))*1.2
    plt.plot([-lim,lim],[-lim,lim],'k--')
    plt.xlabel("true logOR")
    plt.ylabel("estimated logOR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"recovery_scatter.png"),dpi=220)
    plt.close()

    print("Outputs written to:",os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
