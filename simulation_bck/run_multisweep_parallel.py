#!/usr/bin/env python3
"""
run_multisweep_parallel.py

Multi-sweep driver for simulation_expanded.py that produces ONE long-format CSV.

Each output row corresponds to:
  (grid point p_sel, q_sel, n, theta) × (replicate) × (method in {naive, lambda-OR})

This script is compatible with the CURRENT simulation_expanded.py you attached:
  - expects sim module defines: Config dataclass, run_cell(Config) -> DataFrame
  - expects run_cell output columns: ln_true, ln_naive, se_naive, ln_lam, se_lam,
    lam, feasible, n, varphi, pi, theta, p_sel, q_sel, detK, a_hat,b_hat,c_hat,d_hat

It uses multiprocessing *processes* (safe for CPU-bound Python) and avoids pickling
module objects by importing the sim module inside each worker via Pool initializer.
It also includes the Python 3.12 dataclass import fix (register in sys.modules first).

Usage example:
  python run_multisweep_parallel.py \
    --sim_module /home/ishanu/Dropbox/ZED/Research/lambda_or/simulation/simulation_expanded.py \
    --out_csv multisweep_long.csv \
    --out_summary_csv multisweep_summary.csv \
    --procs 24 --R 250 \
    --p_grid 0.90 0.92 0.94 0.96 0.98 \
    --q_grid 0.60 0.70 0.80 0.90 0.95 \
    --n_grid 5000 10000 20000 50000 \
    --theta_grid 0.223143551 0.405465108 0.693147181 1.098612289

Thread mode (not recommended for CPU-bound, but supported):
  add --backend threads
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Robust dynamic import (Py3.12 dataclass fix)
# -------------------------
def load_module_from_path(path: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"sim_module not found: {p}")

    mod_name = f"sim_mod_{p.stem}_{abs(hash(str(p))) % (10**9)}"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {p}")

    mod = importlib.util.module_from_spec(spec)

    # Critical for Python 3.12 dataclasses: must exist in sys.modules before exec_module
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# -------------------------
# Long-format conversion
# -------------------------
def to_long_format(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert run_cell() output (wide) into long format with method column.
    """
    base_cols = [
        "replicate_id",
        "ln_true",
        "n",
        "varphi",
        "pi",
        "theta",
        "p_sel",
        "q_sel",
        "detK",
        "lam",
        "feasible",
        "a_hat",
        "b_hat",
        "c_hat",
        "d_hat",
    ]

    required = set(base_cols + ["ln_naive", "se_naive", "ln_lam", "se_lam"])
    missing = [c for c in required if c not in df_wide.columns]
    if missing:
        raise KeyError(
            "run_cell() output is missing expected columns: "
            + ", ".join(missing)
            + "\nUpdate to_long_format() to match simulation_expanded.py outputs."
        )

    df0 = df_wide[base_cols + ["ln_naive", "se_naive"]].copy()
    df0.rename(columns={"ln_naive": "ln_hat", "se_naive": "se_hat"}, inplace=True)
    df0["method"] = "naive"

    df1 = df_wide[base_cols + ["ln_lam", "se_lam"]].copy()
    df1.rename(columns={"ln_lam": "ln_hat", "se_lam": "se_hat"}, inplace=True)
    df1["method"] = "lambda-OR"

    out = pd.concat([df0, df1], ignore_index=True)
    out["bias"] = out["ln_hat"] - out["ln_true"]
    out["cover_95"] = (
        (out["ln_true"] >= out["ln_hat"] - 1.96 * out["se_hat"])
        & (out["ln_true"] <= out["ln_hat"] + 1.96 * out["se_hat"])
    ).astype(int)
    return out


# -------------------------
# Pool globals (avoid pickling module objects)
# -------------------------
_SIM = None
_BASE_CFG = None


def _init_pool(sim_module_path: str, R: int, varphi: float, pi: float, seed: int):
    """
    Runs once in each worker process/thread:
      - imports sim module
      - creates a base Config
      - seeds numpy RNG in this worker (run_cell uses module RNG; this sets local RNG only)
    """
    global _SIM, _BASE_CFG
    np.random.seed(seed + (os.getpid() % 1000000))

    _SIM = load_module_from_path(sim_module_path)

    for attr in ("Config", "run_cell"):
        if not hasattr(_SIM, attr):
            raise AttributeError(f"simulation module must define {attr}")

    # Build a base config and override per-task via dataclasses.replace
    _BASE_CFG = _SIM.Config(
        n=20000,
        varphi=float(varphi),
        theta=math.log(2),
        pi=float(pi),
        p_sel=0.90,
        q_sel=0.90,
        R=int(R),
    )


def _worker(task: Tuple[float, float, int, float]) -> pd.DataFrame:
    """
    One grid point -> run_cell -> long DF.
    """
    global _SIM, _BASE_CFG
    if _SIM is None or _BASE_CFG is None:
        raise RuntimeError("Worker not initialized. Pool initializer was not run.")

    p_sel, q_sel, n, theta = task
    #cfg = replace(_BASE_CFG, p_sel=float(p_sel), q_sel=float(q_sel), n=int(n), theta=float(theta))

    grid_seed = (
        hash((float(p_sel), float(q_sel), int(n), float(theta)))
        & 0x7FFFFFFF
    )

    cfg = replace(
        _BASE_CFG,
        p_sel=float(p_sel),
        q_sel=float(q_sel),
        n=int(n),
        theta=float(theta),
        seed=int(grid_seed),
    )


    
    df = _SIM.run_cell(cfg).reset_index(drop=True).copy()
    df["replicate_id"] = np.arange(len(df), dtype=int)

    long_df = to_long_format(df)
    long_df["grid_id"] = f"p{p_sel:.4f}_q{q_sel:.4f}_n{int(n)}_th{theta:.6f}"
    return long_df


# -------------------------
# Utilities
# -------------------------
def parse_float_list(xs: List[str]) -> List[float]:
    return [float(x) for x in xs]


def parse_int_list(xs: List[str]) -> List[int]:
    return [int(x) for x in xs]


def append_csv(path: Path, df: pd.DataFrame, header_written: bool) -> bool:
    df.to_csv(path, mode="a", index=False, header=(not header_written))
    return True


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_module", type=str, required=True, help="Path to simulation_expanded.py")
    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path (long format)")
    ap.add_argument("--out_summary_csv", type=str, default="", help="Optional summary CSV path")
    ap.add_argument("--backend", choices=["processes", "threads"], default="processes")
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--chunksize", type=int, default=1)
    ap.add_argument("--write_every", type=int, default=10)

    ap.add_argument("--R", type=int, default=250, help="Replicates per grid point")
    ap.add_argument("--varphi", type=float, default=0.5, help="P(X=1)")
    ap.add_argument("--pi", type=float, default=0.2, help="Target marginal P(Y=1)")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--p_grid", nargs="+", required=True)
    ap.add_argument("--q_grid", nargs="+", required=True)
    ap.add_argument("--n_grid", nargs="+", required=True)
    ap.add_argument("--theta_grid", nargs="+", required=True)

    args = ap.parse_args()

    p_grid = parse_float_list(args.p_grid)
    q_grid = parse_float_list(args.q_grid)
    n_grid = parse_int_list(args.n_grid)
    theta_grid = parse_float_list(args.theta_grid)

    tasks = list(itertools.product(p_grid, q_grid, n_grid, theta_grid))
    total = len(tasks)

    out_path = Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    header_written = False
    completed = 0

    # If summary requested, keep partials in memory (can be large). If you want
    # zero-memory summary, compute it later from the long CSV.
    parts_for_summary: List[pd.DataFrame] = []

    if args.backend == "threads":
        from multiprocessing.dummy import Pool  # thread Pool
    else:
        from multiprocessing import Pool  # process Pool

    with Pool(
        processes=int(args.procs),
        initializer=_init_pool,
        initargs=(args.sim_module, int(args.R), float(args.varphi), float(args.pi), int(args.seed)),
    ) as pool:
        for long_df in pool.imap_unordered(_worker, tasks, chunksize=int(args.chunksize)):
            header_written = append_csv(out_path, long_df, header_written)

            if args.out_summary_csv:
                parts_for_summary.append(long_df)

            completed += 1
            if completed % int(args.write_every) == 0 or completed == total:
                print(f"[progress] completed {completed}/{total} grid points")

    print(f"[done] wrote long-format CSV: {out_path}")

    if args.out_summary_csv:
        summ_path = Path(args.out_summary_csv).expanduser().resolve()
        summ_path.parent.mkdir(parents=True, exist_ok=True)

        df_all = pd.concat(parts_for_summary, ignore_index=True)

        grp_cols = ["method", "p_sel", "q_sel", "n", "theta", "detK"]
        summary = (
            df_all.groupby(grp_cols, as_index=False)
            .agg(
                bias_mean=("bias", "mean"),
                bias_sd=("bias", "std"),
                rmse=("bias", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
                cover_95=("cover_95", "mean"),
                se_mean=("se_hat", "mean"),
                feasible_rate=("feasible", "mean"),
                lam_median=("lam", "median"),
            )
        )
        summary.to_csv(summ_path, index=False)
        print(f"[done] wrote summary CSV: {summ_path}")


if __name__ == "__main__":
    main()
