#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAIVE_RE = re.compile(
    r"Naive OR FDR=([0-9eE+\-.]+)\s+\(discoveries=(\d+), TP=(\d+), FP=(\d+)\)"
)
LAMBDA_RE = re.compile(
    r"Lambda-style FDR=([0-9eE+\-.]+)\s+\(discoveries=(\d+), TP=(\d+), FP=(\d+)\)"
)
VAR_RE = re.compile(
    r"var_extra=([0-9eE+\-.]+)\s+\(SE floor=([0-9eE+\-.]+)\),\s+eps=([0-9eE+\-.]+),\s+n_val=(\d+)"
)


def default_n_case_grid():
    # 11 logarithmically spaced points from 1e2 to 1e6 inclusive.
    vals = np.logspace(2, 6, 11)
    vals = np.round(vals).astype(int)
    # guard against accidental duplicates after rounding
    vals = np.unique(vals)
    return vals.tolist()


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Sweep volcano_simulation_nulls.py over log-spaced n_case values and "
            "multiple alpha levels, with n_ctrl = 100 * n_case, collect summary "
            "metrics into CSV, and generate multi-curve performance plots."
        )
    )
    ap.add_argument(
        "--sim_script",
        type=str,
        default="./volcano_simulation_nulls.py",
        help="Path to volcano_simulation_nulls.py",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="ncase_alpha_sweep",
        help="Directory to write run outputs, summary CSV, and plots",
    )
    ap.add_argument(
        "--n_cases",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit n_case grid. If omitted, uses 11 log-spaced points from 100 to 1,000,000",
    )
    ap.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.05, 0.01],
        help="Alpha levels to sweep. Default: 0.05 0.01",
    )
    ap.add_argument(
        "--ctrl_case_ratio",
        type=float,
        default=100.0,
        help="n_ctrl / n_case ratio. Default: 100",
    )

    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--m", type=int, default=20000)
    ap.add_argument("--null_frac", type=float, default=0.8)
    ap.add_argument("--beta_mean", type=float, default=0.06)
    ap.add_argument("--beta_sd", type=float, default=0.03)
    ap.add_argument("--eps", type=float, default=0.8)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--var_extra_scale", type=float, default=1.0)

    ap.add_argument("--prev0_mean", type=float, default=None)
    ap.add_argument("--prev_sd", type=float, default=None)
    ap.add_argument("--xlim", type=float, default=None)

    return ap.parse_args()


def fmt_alpha(alpha: float) -> str:
    s = f"{alpha:g}"
    return s.replace(".", "p")


def run_one(sim_script, run_dir, args, n_case, alpha):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"

    n_ctrl = int(round(args.ctrl_case_ratio * n_case))

    cmd = [
        sys.executable,
        sim_script,
        "--out_dir", str(run_dir),
        "--seed", str(args.seed),
        "--m", str(args.m),
        "--null_frac", str(args.null_frac),
        "--n_case", str(n_case),
        "--n_ctrl", str(n_ctrl),
        "--beta_mean", str(args.beta_mean),
        "--beta_sd", str(args.beta_sd),
        "--eps", str(args.eps),
        "--n_val", str(args.n_val),
        "--var_extra_scale", str(args.var_extra_scale),
        "--alpha", str(alpha),
    ]

    if args.prev0_mean is not None:
        cmd.extend(["--prev0_mean", str(args.prev0_mean)])
    if args.prev_sd is not None:
        cmd.extend(["--prev_sd", str(args.prev_sd)])
    if args.xlim is not None:
        cmd.extend(["--xlim", str(args.xlim)])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = proc.stdout
    stderr = proc.stderr

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(stdout)
        if stderr:
            fh.write("\n[stderr]\n")
            fh.write(stderr)

    m_var = VAR_RE.search(stdout)
    m_naive = NAIVE_RE.search(stdout)
    m_lambda = LAMBDA_RE.search(stdout)

    if not (m_var and m_naive and m_lambda):
        raise RuntimeError(
            f"Could not parse expected metrics for n_case={n_case}, alpha={alpha}.\n"
            f"See log: {log_path}\n\n"
            f"Captured stdout:\n{stdout}"
        )

    row = {
        "alpha": float(alpha),
        "n_case": int(n_case),
        "n_ctrl": int(n_ctrl),
        "ctrl_case_ratio": float(args.ctrl_case_ratio),
        "seed": int(args.seed),
        "m": int(args.m),
        "null_frac": float(args.null_frac),
        "beta_mean": float(args.beta_mean),
        "beta_sd": float(args.beta_sd),
        "eps": float(m_var.group(3)),
        "n_val": int(m_var.group(4)),
        "var_extra_scale": float(args.var_extra_scale),
        "var_extra": float(m_var.group(1)),
        "se_floor": float(m_var.group(2)),
        "naive_fdr": float(m_naive.group(1)),
        "naive_discoveries": int(m_naive.group(2)),
        "naive_tp": int(m_naive.group(3)),
        "naive_fp": int(m_naive.group(4)),
        "lambda_fdr": float(m_lambda.group(1)),
        "lambda_discoveries": int(m_lambda.group(2)),
        "lambda_tp": int(m_lambda.group(3)),
        "lambda_fp": int(m_lambda.group(4)),
        "run_dir": str(run_dir),
    }

    print(
        f"alpha={alpha:g}, n_case={n_case}, n_ctrl={n_ctrl}: "
        f"naive_fdr={row['naive_fdr']:.4f}, lambda_fdr={row['lambda_fdr']:.4f}, "
        f"naive_disc={row['naive_discoveries']}, lambda_disc={row['lambda_discoveries']}"
    )
    return row


def plot_panel(ax, df, y_naive, y_lambda, ylabel, title, alphas):
    for alpha in alphas:
        sub = df[df["alpha"] == alpha].sort_values("n_case")
        ax.plot(
            sub["n_case"], sub[y_naive],
            marker="o", linewidth=2,
            label=f"Naive OR, α={alpha:g}"
        )
        ax.plot(
            sub["n_case"], sub[y_lambda],
            marker="o", linewidth=2,
            linestyle="--",
            label=f"Lambda-OR, α={alpha:g}"
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$n_{\mathrm{case}}$  (with $n_{\mathrm{ctrl}} = 100\,n_{\mathrm{case}}$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def make_plots(df, out_dir, alphas):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    plot_panel(
        axes[0], df,
        "naive_fdr", "lambda_fdr",
        "False discovery rate",
        "FDR",
        alphas,
    )
    axes[0].axhline(0.05, linewidth=1, linestyle=":", color="gray")
    axes[0].axhline(0.01, linewidth=1, linestyle=":", color="gray")

    plot_panel(
        axes[1], df,
        "naive_discoveries", "lambda_discoveries",
        "Number of discoveries",
        "Discoveries",
        alphas,
    )
    plot_panel(
        axes[2], df,
        "naive_tp", "lambda_tp",
        "True positives",
        "True positives",
        alphas,
    )
    plot_panel(
        axes[3], df,
        "naive_fp", "lambda_fp",
        "False positives",
        "False positives",
        alphas,
    )

    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "performance_summary_vs_ncase_alpha.png", dpi=220)
    plt.close(fig)

    for alpha in alphas:
        sub = df[df["alpha"] == alpha].sort_values("n_case")
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.ravel()

        pairs = [
            ("naive_fdr", "lambda_fdr", "False discovery rate", f"FDR (α={alpha:g})"),
            ("naive_discoveries", "lambda_discoveries", "Number of discoveries", f"Discoveries (α={alpha:g})"),
            ("naive_tp", "lambda_tp", "True positives", f"True positives (α={alpha:g})"),
            ("naive_fp", "lambda_fp", "False positives", f"False positives (α={alpha:g})"),
        ]

        for ax, (y1, y2, ylabel, title) in zip(axes, pairs):
            ax.plot(sub["n_case"], sub[y1], marker="o", linewidth=2, label="Naive OR")
            ax.plot(sub["n_case"], sub[y2], marker="o", linewidth=2, linestyle="--", label="Lambda-OR")
            ax.set_xscale("log")
            ax.set_xlabel(r"$n_{\mathrm{case}}$  (with $n_{\mathrm{ctrl}} = 100\,n_{\mathrm{case}}$)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)

        axes[0].axhline(alpha, linewidth=1, linestyle=":", color="gray")
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"performance_summary_alpha_{fmt_alpha(alpha)}.png", dpi=220)
        plt.close(fig)


def write_command_recap(args, out_dir, n_cases):
    txt = [
        "Sweep configuration",
        "",
        f"Simulation script: {args.sim_script}",
        f"Output directory: {out_dir}",
        "",
        "Fixed per-run parameters:",
        f"  seed={args.seed}",
        f"  m={args.m}",
        f"  null_frac={args.null_frac}",
        f"  beta_mean={args.beta_mean}",
        f"  beta_sd={args.beta_sd}",
        f"  eps={args.eps}",
        f"  n_val={args.n_val}",
        f"  var_extra_scale={args.var_extra_scale}",
        f"  ctrl_case_ratio={args.ctrl_case_ratio}",
        f"  alphas={' '.join(str(a) for a in args.alphas)}",
    ]
    if args.prev0_mean is not None:
        txt.append(f"  prev0_mean={args.prev0_mean}")
    if args.prev_sd is not None:
        txt.append(f"  prev_sd={args.prev_sd}")
    if args.xlim is not None:
        txt.append(f"  xlim={args.xlim}")

    txt.extend([
        "",
        "n_case grid:",
        "  " + " ".join(str(x) for x in n_cases),
        "",
        "Example per-run command template:",
        (
            f"{sys.executable} {args.sim_script} "
            f"--out_dir <run_dir> "
            f"--seed {args.seed} "
            f"--m {args.m} "
            f"--null_frac {args.null_frac} "
            f"--n_case <N> "
            f"--n_ctrl <100*N> "
            f"--beta_mean {args.beta_mean} "
            f"--beta_sd {args.beta_sd} "
            f"--eps {args.eps} "
            f"--n_val {args.n_val} "
            f"--var_extra_scale {args.var_extra_scale} "
            f"--alpha <ALPHA>"
        ),
        "",
    ])

    with open(out_dir / "sweep_metadata.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(txt))


def main():
    args = parse_args()
    sim_script = os.path.abspath(args.sim_script)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cases = args.n_cases if args.n_cases is not None else default_n_case_grid()
    alphas = list(args.alphas)

    rows = []
    for alpha in alphas:
        alpha_dir = out_dir / f"alpha_{fmt_alpha(alpha)}"
        alpha_dir.mkdir(parents=True, exist_ok=True)

        for n_case in n_cases:
            run_dir = alpha_dir / f"n_{n_case}"
            row = run_one(sim_script, run_dir, args, n_case, alpha)
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["alpha", "n_case"]).reset_index(drop=True)
    csv_path = out_dir / "sweep_summary.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    make_plots(df, out_dir, alphas)
    write_command_recap(args, out_dir, n_cases)

    print(f"\nWrote summary CSV: {csv_path}")
    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
