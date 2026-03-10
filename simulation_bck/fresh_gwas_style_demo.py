#!/usr/bin/env python3
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def logor_wald(a,b,c,d,ha=0.5):
    a,b,c,d = a+ha,b+ha,c+ha,d+ha
    lo = np.log((a*d)/(b*c))
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    p  = 2*norm.sf(abs(lo/se))
    return lo,p

def misclassify(a,b,c,d,p,q,rng):
    tp1=rng.binomial(a,p);   fp1=rng.binomial(b,1-q)
    tp0=rng.binomial(c,p);   fp0=rng.binomial(d,1-q)
    aT=tp1+fp1;              bT=(a-tp1)+(b-fp1)
    cT=tp0+fp0;              dT=(c-tp0)+(d-fp0)
    return aT,bT,cT,dT

def Kmat(p,q): return np.array([[q,1-p],[1-q,p]],float)

def lambda_lo(aT,bT,cT,dT,p,q,lam):
    inv=np.linalg.inv(Kmat(p,q)+lam*np.eye(2))
    tru1=np.clip(inv@np.array([bT,aT],float),1e-6,None)
    tru0=np.clip(inv@np.array([dT,cT],float),1e-6,None)
    a,b,c,d=tru1[1],tru1[0],tru0[1],tru0[0]
    return np.log((a*d)/(b*c))

def volcano(x,p,out,title):
    y=-np.log10(np.clip(p,1e-300,1))
    plt.figure(figsize=(7,5))
    plt.scatter(x,y,s=7,alpha=.6)
    plt.axvline(0,color='k',lw=1); plt.axhline(-np.log10(0.05),ls='--',lw=1)
    plt.xlabel("log(OR)"); plt.ylabel("-log10(p)"); plt.title(title)
    plt.tight_layout(); plt.savefig(out,dpi=220); plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out_dir",default="./gwas_style_demo")
    ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--m",type=int,default=6000)
    ap.add_argument("--n",type=int,default=40000,help="per-feature sample size")
    ap.add_argument("--beta_sd",type=float,default=0.6,help="true logOR spread")
    ap.add_argument("--p",type=float,default=0.80); ap.add_argument("--q",type=float,default=0.70)
    ap.add_argument("--lam",type=float,default=0.05)
    ap.add_argument("--messy",action="store_true",
                    help="plot x=true but y=noisy p-values (parabola breaks)")
    args=ap.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    rng=np.random.default_rng(args.seed)

    beta=rng.normal(0,args.beta_sd,args.m)  # true logOR per feature
    OR=np.exp(beta)
    r=np.sqrt(OR)
    t=0.5*r/(1+r)                           # p11=p00=t, p10=p01=0.5-t
    probs=np.stack([t,0.5-t,0.5-t,t],1)     # 00,01,10,11

    lo_true=np.empty(args.m); p_true=np.empty(args.m)
    lo_naive=np.empty(args.m); p_naive=np.empty(args.m)
    lo_lam=np.empty(args.m)

    for j in range(args.m):
        d,c,b,a=rng.multinomial(args.n,probs[j])
        lo_true[j],p_true[j]=logor_wald(a,b,c,d)
        aT,bT,cT,dT=misclassify(a,b,c,d,args.p,args.q,rng)
        lo_naive[j],p_naive[j]=logor_wald(aT,bT,cT,dT)
        lo_lam[j]=lambda_lo(aT,bT,cT,dT,args.p,args.q,args.lam)

    volcano(lo_true,p_true,os.path.join(args.out_dir,"volcano_true.png"),"True volcano")
    volcano(lo_naive,p_naive,os.path.join(args.out_dir,"volcano_naive_noisy.png"),"Naive volcano (noisy table)")
    volcano(lo_lam,p_naive,os.path.join(args.out_dir,"volcano_lambda.png"),
            "Lambda-corrected x, noisy p-values")

    if args.messy:
        volcano(lo_true,p_naive,os.path.join(args.out_dir,"volcano_messy.png"),
                "Messy: x=true logOR, y=noisy p-values")

    plt.figure(figsize=(5.5,5.5))
    plt.scatter(lo_true,lo_naive,s=6,alpha=.4,label="naive")
    plt.scatter(lo_true,lo_lam,s=6,alpha=.4,label="lambda")
    lim=float(np.max(np.abs(lo_true))*1.05)
    plt.plot([-lim,lim],[-lim,lim],'k--',lw=1)
    plt.xlabel("true logOR"); plt.ylabel("estimated logOR"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir,"recovery_scatter.png"),dpi=220); plt.close()

    print("Wrote:", os.path.abspath(args.out_dir))

if __name__=="__main__": 
    main()
