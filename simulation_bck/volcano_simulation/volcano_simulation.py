#!/usr/bin/env python3
"""
naive_vs_lambda_yfloor_clean_updated.py

Simulation analogue of the attribution volcano plots for naive OR versus
lambda-OR-style inference with a validation-driven standard-error floor.

This script is intended to make the variance mechanism in the manuscript easy to
reproduce. It simulates many binary exposures with small nonzero effects,
constructs classical odds-ratio estimates from large case/control cohorts, and
then compares:

1. Naive inference, where the Wald standard error shrinks with the main cohort
   size.
2. Lambda-style inference, where an additional variance component is added to
   reflect uncertainty in estimating selection-conditional misclassification
   rates from a finite validation cohort.

The extra variance term is represented here by the leading dependence described
in the manuscript:

    Var_extra ~ 1 / (det(K)^2 * n_val),

where det(K) = p_sel + q_sel - 1 is the conditioning of the misclassification
operator and n_val is the size of the validation cohort used to estimate
(p_sel, q_sel). This produces a cohort-size-independent SE floor, which is the
mechanism illustrated by the volcano plots.

Outputs
-------
The script writes to --out_dir:

- volcano_naive.png
    Volcano plot using classical log-OR Wald p-values.
- volcano_lambda.png
    Volcano plot using the augmented SE including Var_extra.
- full_results.csv
    One row per simulated feature, with all simulated inputs and derived
    quantities needed to reproduce downstream summaries or plots.
- run_metadata.txt
    Human-readable run summary, including the full command used.

Replication command used as the default configuration
-----------------------------------------------------
python naive_vs_lambda_variance.py \
  --out_dir yfloor_demo \
  --seed 3 \
  --m 20000 \
  --n_case 600000 \
  --n_ctrl 600000 \
  --beta_mean 0.06 \
  --beta_sd 0.03 \
  --eps 0.5 \
  --n_val 1000 \
  --var_extra_scale 1.0
"""

import argparse
import os
import shlex
import sys
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


DEFAULT_REPRO_COMMAND = (
    "python naive_vs_lambda_variance.py "
    "--out_dir yfloor_demo "
    "--seed 3 "
    "--m 20000 "
    "--n_case 600000 "
    "--n_ctrl 600000 "
    "--beta_mean 0.06 "
    "--beta_sd 0.03 "
    "--eps 0.5 "
    "--n_val 1000 "
    "--var_extra_scale 1.0"
)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)



def expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform."""
    return 1.0 / (1.0 + np.exp(-x))



def logor_wald(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, ha: float = 0.5):
    """
    Compute classical log-odds-ratio Wald quantities from 2x2 table counts.

    Parameters
    ----------
    a, b, c, d : np.ndarray
        Cell counts for each simulated feature, using the standard notation
        a = exposed among cases,
        b = exposed among controls,
        c = unexposed among cases,
        d = unexposed among controls.
    ha : float, default=0.5
        Haldane-Anscombe continuity correction added to every cell to avoid
        undefined log-OR or infinite variance when a sampled count is zero.

    Returns
    -------
    logor : np.ndarray
        Classical log(OR) estimate.
    se : np.ndarray
        Classical Wald standard error sqrt(1/a + 1/b + 1/c + 1/d).
    z : np.ndarray
        Wald z-statistic.
    p : np.ndarray
        Two-sided normal-approximation p-value.
    """
    a = a.astype(float) + ha
    b = b.astype(float) + ha
    c = c.astype(float) + ha
    d = d.astype(float) + ha

    logor = np.log((a * d) / (b * c))
    se = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    z = logor / se
    p = 2.0 * norm.sf(np.abs(z))
    return logor, se, z, p



def safe_neglog10_p(p: np.ndarray, min_p: float = 1e-300) -> np.ndarray:
    """
    Convert p-values to -log10(p) with clipping to avoid underflow or inf.

    Clipping also prevents downstream warnings when plotting transformed values.
    """
    return -np.log10(np.clip(p, min_p, 1.0))



def volcano(x: np.ndarray, p: np.ndarray, out_png: str, title: str,
            xlim=(-0.5, 0.5), ylim=None) -> None:
    """
    Save a volcano plot.

    The manuscript plots significance on a -log10(p) axis. Here we store and plot
    that directly, rather than taking log10(-log10(p)), which can generate
    warnings if p = 1 for some points.
    """
    y = safe_neglog10_p(p)
    plt.figure(figsize=(8.0, 6.0))
    plt.scatter(x, y, s=6, alpha=0.55)
    plt.axvline(0.0, color="black", lw=1)
    plt.axhline(-np.log10(0.05), color="black", lw=1, ls="--")
    plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Effect size (log OR)")
    plt.ylabel(r"Significance ($-\log_{10} p$)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()



def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser with detailed help text."""
    description = dedent(
        """
        Simulate many binary exposures to compare naive OR inference against a
        lambda-OR-style variance floor. The main cohort controls the classical
        Wald SE, while the validation cohort contributes an extra variance term
        proportional to 1 / (det(K)^2 * n_val), where det(K) is supplied here as
        the proxy parameter --eps.
        """
    ).strip()

    epilog = dedent(
        f"""
        Default reproducibility command:
          {DEFAULT_REPRO_COMMAND}

        Parameter interpretation:
          --m                Number of simulated candidate exposures/features.
          --n_case           Number of cases in the main cohort.
          --n_ctrl           Number of controls in the main cohort.
          --prev0_mean       Mean of the control prevalence on the logit scale.
          --prev_sd          SD of the control prevalence on the logit scale.
          --beta_mean        Mean of the absolute normal effect-size distribution.
          --beta_sd          SD of the absolute normal effect-size distribution.
          --eps              Proxy for det(K) = p_sel + q_sel - 1. Smaller values
                             imply a more ill-conditioned misclassification
                             operator and therefore larger validation-driven
                             variance inflation. In the manuscript-style analogy,
                             this controls the SE floor through 1/(eps*sqrt(n_val)).
          --n_val            Validation cohort size used to estimate (p_sel, q_sel).
                             Larger values reduce the added variance term.
          --var_extra_scale  User multiplier applied to Var_extra. This is useful
                             for sensitivity analyses around the leading-order
                             theoretical scaling.
          --xlim             Half-width of the x-axis used in the volcano plots.
        """
    ).strip()

    ap = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    ap.add_argument(
        "--out_dir",
        type=str,
        default="yfloor_demo",
        help="Output directory for figures, CSV, and run metadata. Default: yfloor_demo",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Random number generator seed. Default: 3",
    )
    ap.add_argument(
        "--m",
        type=int,
        default=20000,
        help="Number of simulated exposures/features. Default: 20000",
    )
    ap.add_argument(
        "--n_case",
        type=int,
        default=600000,
        help="Number of cases in the main cohort. Default: 600000",
    )
    ap.add_argument(
        "--n_ctrl",
        type=int,
        default=600000,
        help="Number of controls in the main cohort. Default: 600000",
    )
    ap.add_argument(
        "--prev0_mean",
        type=float,
        default=-3.5,
        help=(
            "Mean of control exposure prevalence on the logit scale. "
            "If logit(prev0) ~ N(prev0_mean, prev_sd^2), this sets the average "
            "baseline rarity of features among controls. Default: -3.5"
        ),
    )
    ap.add_argument(
        "--prev_sd",
        type=float,
        default=1.0,
        help="Standard deviation of control prevalence on the logit scale. Default: 1.0",
    )
    ap.add_argument(
        "--beta_mean",
        type=float,
        default=0.06,
        help=(
            "Mean of the true log-OR effect distribution before taking absolute value. "
            "The script draws beta_j ~ |N(beta_mean, beta_sd^2)|. Default: 0.06"
        ),
    )
    ap.add_argument(
        "--beta_sd",
        type=float,
        default=0.03,
        help="SD of the true log-OR effect distribution. Default: 0.03",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help=(
            "Proxy for det(K)=p_sel+q_sel-1, the conditioning of the misclassification "
            "operator. Smaller eps implies stronger ill-conditioning and a larger "
            "validation-driven variance contribution. Default: 0.5"
        ),
    )
    ap.add_argument(
        "--n_val",
        type=int,
        default=1000,
        help=(
            "Validation cohort size governing uncertainty in the estimated selection-"
            "conditional misclassification rates. Default: 1000"
        ),
    )
    ap.add_argument(
        "--var_extra_scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the leading-order extra variance term. Use this for "
            "sensitivity analyses around the theoretical scaling. Default: 1.0"
        ),
    )
    ap.add_argument(
        "--xlim",
        type=float,
        default=0.35,
        help="Half-width of the x-axis used for both volcano plots. Default: 0.35",
    )
    return ap



def write_metadata(out_dir: str, args: argparse.Namespace, var_extra: float,
                   command_used: str, se_floor: float) -> None:
    """Write a plain-text metadata file summarizing the run and replication command."""
    txt = dedent(
        f"""
        Simulation: naive OR vs lambda-style variance floor

        Full command used:
        {command_used}

        Default manuscript replication command:
        {DEFAULT_REPRO_COMMAND}

        Resolved parameter values:
        out_dir={args.out_dir}
        seed={args.seed}
        m={args.m}
        n_case={args.n_case}
        n_ctrl={args.n_ctrl}
        prev0_mean={args.prev0_mean}
        prev_sd={args.prev_sd}
        beta_mean={args.beta_mean}
        beta_sd={args.beta_sd}
        eps={args.eps}
        n_val={args.n_val}
        var_extra_scale={args.var_extra_scale}
        xlim={args.xlim}

        Derived quantities:
        var_extra={var_extra:.12g}
        se_floor={se_floor:.12g}

        Interpretation:
        - var_extra is the added variance component used in the lambda-style
          inference analogue.
        - se_floor = sqrt(var_extra) is the sample-size-independent lower bound on
          the standard error induced by finite validation uncertainty.
        - In this simplified demonstration, the naive log-OR estimate itself is
          unchanged; only the inferential variance differs between the naive and
          lambda-style analyses.
        """
    ).strip() + "\n"

    with open(os.path.join(out_dir, "run_metadata.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)



def main() -> None:
    """Run the simulation, save plots, and write a full per-feature CSV."""
    parser = build_parser()
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    m = args.m
    n_case = args.n_case
    n_ctrl = args.n_ctrl

    # True nonzero effects, all positive in magnitude as in the manuscript sketch.
    beta_true = np.abs(rng.normal(args.beta_mean, args.beta_sd, size=m))

    # Baseline exposure prevalence among controls on the logit scale.
    logit_prev0 = rng.normal(args.prev0_mean, args.prev_sd, size=m)
    prev0 = expit(logit_prev0)
    prev1 = expit(logit_prev0 + beta_true)

    # Sample true 2x2 tables independently feature-wise.
    a_raw = rng.binomial(n_case, prev1)  # exposed among cases
    b_raw = rng.binomial(n_ctrl, prev0)  # exposed among controls
    c_raw = n_case - a_raw               # unexposed among cases
    d_raw = n_ctrl - b_raw               # unexposed among controls

    log_or_naive, se_naive, z_naive, p_naive = logor_wald(a_raw, b_raw, c_raw, d_raw)

    eps = max(float(args.eps), 1e-12)
    n_val = max(int(args.n_val), 1)
    var_extra = args.var_extra_scale * (1.0 / (eps * eps * n_val))
    se_floor = float(np.sqrt(var_extra))

    se_lambda = np.sqrt(se_naive ** 2 + var_extra)
    z_lambda = log_or_naive / se_lambda
    p_lambda = 2.0 * norm.sf(np.abs(z_lambda))

    y_naive = safe_neglog10_p(p_naive)
    y_lambda = safe_neglog10_p(p_lambda)

    xlim = (-args.xlim, args.xlim)

    volcano(
        log_or_naive,
        p_naive,
        os.path.join(args.out_dir, f"volcano_naive{n_case}.png"),
        f"Naive OR inference: median -log10(p) = {np.median(y_naive):.2f}",
        xlim=xlim,
        ylim=(0.0, max(5.0, float(np.quantile(y_naive, 0.995)))),
    )
    volcano(
        log_or_naive,
        p_lambda,
        os.path.join(args.out_dir, f"volcano_lambda{n_case}.png"),
        f"Lambda-style inference: eps = {eps:.3g}, n_val = {n_val}, SE floor = {se_floor:.4f}",
        xlim=xlim,
        ylim=(0.0, max(5.0, float(np.quantile(y_lambda, 0.995)))),
    )

    df = pd.DataFrame({
        "feature_id": np.arange(m, dtype=int),
        "seed": np.full(m, args.seed, dtype=int),
        "m": np.full(m, args.m, dtype=int),
        "n_case": np.full(m, args.n_case, dtype=int),
        "n_ctrl": np.full(m, args.n_ctrl, dtype=int),
        "prev0_mean": np.full(m, args.prev0_mean, dtype=float),
        "prev_sd": np.full(m, args.prev_sd, dtype=float),
        "beta_mean": np.full(m, args.beta_mean, dtype=float),
        "beta_sd": np.full(m, args.beta_sd, dtype=float),
        "eps": np.full(m, args.eps, dtype=float),
        "n_val": np.full(m, args.n_val, dtype=int),
        "var_extra_scale": np.full(m, args.var_extra_scale, dtype=float),
        "xlim": np.full(m, args.xlim, dtype=float),
        "beta_true": beta_true,
        "logit_prev0": logit_prev0,
        "prev0": prev0,
        "prev1": prev1,
        "a_raw": a_raw,
        "b_raw": b_raw,
        "c_raw": c_raw,
        "d_raw": d_raw,
        "log_or_naive": log_or_naive,
        "se_naive": se_naive,
        "z_naive": z_naive,
        "p_naive": -np.log10(p_naive),
        "y_naive": y_naive,
        "var_extra": np.full(m, var_extra, dtype=float),
        "se_floor": np.full(m, se_floor, dtype=float),
        "se_lambda": se_lambda,
        "z_lambda": z_lambda,
        "p_lambda": -np.log10(p_lambda),
        "y_lambda": y_lambda,
    })
    csv_path = os.path.join(args.out_dir, f"full_results_{n_case}.csv")
    df.to_csv(csv_path, index=False)

    command_used = " ".join(shlex.quote(tok) for tok in sys.argv)
    write_metadata(args.out_dir, args, var_extra, command_used, se_floor)

    print(f"var_extra={var_extra:.6g} (SE floor={se_floor:.5f}), eps={eps}, n_val={n_val}")
    print(f"Wrote CSV: {os.path.abspath(csv_path)}")
    print(f"Wrote: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
