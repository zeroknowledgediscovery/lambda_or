"""
lambda_or.py — Λ-OR (Lambda Odds Ratio) for misclassification-corrected attribution.

This module implements a ridge-stabilized inversion of a misclassified 2x2 contingency
table to recover corrected counts and compute the corrected log-odds ratio, its
variance (including uncertainty from selection-conditional sensitivity/specificity),
a Wald z-statistic, and a -log10 p-value with a large-|z| tail approximation.

Core API
--------
lambda_or(tilde_counts, p_sel, q_sel, n_val, lambda_start=1e-6, lambda_max=1e6,
          step=10.0, eps=1e-9) -> dict

Helpers
-------
pq_from_two_gates(pi_H1, pi_L1, pi_H0, pi_L0) -> (p_sel, q_sel)

Notation
--------
- tilde_counts: observed (misclassified) 2x2 counts as [[ã, b̃],[ĉ, d̃]]
- p_sel = P(tilde Y=1 | Y=1, selected), q_sel = P(tilde Y=0 | Y=0, selected)
- n_val: size of validation cohort used to estimate (p_sel, q_sel)

References
----------
See manuscript text for derivations and delta-method variance.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import log, sqrt, isfinite, erfc, pi, log10
from typing import Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class LambdaORResult:
    log_or: float
    neglog10_p: float
    z: float
    se: float
    counts: np.ndarray      # corrected 2x2 matrix [ [a,b], [c,d] ]
    lambda_used: float
    converged: bool
    meta: Dict[str, Any]


def _normal_two_sided_p(z_abs: float) -> float:
    """Two-sided p-value using complementary error function (numerically stable)."""
    # sf(|z|) = 0.5 * erfc(|z|/sqrt(2))
    return 2.0 * 0.5 * erfc(z_abs / sqrt(2.0))


def pq_from_two_gates(pi_H1: float, pi_L1: float, pi_H0: float, pi_L0: float) -> Tuple[float, float]:
    """
    Compute selection-conditional sensitivity/specificity from two-gate ROC probabilities:
    - pi_H1 = P(S in high-specificity gate | Y=1)
    - pi_L1 = P(S in low-specificity gate  | Y=1)
    - pi_H0 = P(S in high-specificity gate | Y=0)
    - pi_L0 = P(S in low-specificity gate  | Y=0)
    Returns (p_sel, q_sel).
    """
    denom1 = pi_H1 + pi_L1
    denom0 = pi_H0 + pi_L0
    if denom1 <= 0 or denom0 <= 0:
        raise ValueError("Invalid gate probabilities: denominators must be positive.")
    p_sel = pi_H1 / denom1
    q_sel = pi_L0 / denom0
    return p_sel, q_sel


def lambda_or(tilde_counts: np.ndarray,
              p_sel: float,
              q_sel: float,
              n_val: int,
              lambda_start: float = 1e-6,
              lambda_max: float = 1e6,
              step: float = 10.0,
              eps: float = 1e-9) -> LambdaORResult:
    """
    Compute Λ-OR given a misclassified 2x2 table and selection-conditional (p_sel, q_sel).

    Parameters
    ----------
    tilde_counts : array-like shape (2,2)
        Observed (misclassified) counts [[ã, b̃],[ĉ, d̃]].
    p_sel, q_sel : float
        Selection-conditional sensitivity and specificity (from ROC gates).
    n_val : int
        Validation cohort size used to estimate (p_sel, q_sel) for variance propagation.
    lambda_start, lambda_max : float
        Ridge path start and upper bound.
    step : float
        Multiplicative factor for ridge path (e.g., 10.0).
    eps : float
        Numerical/feasibility floor for corrected counts.

    Returns
    -------
    LambdaORResult
        Structured result with fields (log_or, neglog10_p, z, se, counts, lambda_used, converged, meta).
    """
    Ttil = np.asarray(tilde_counts, dtype=float)
    if Ttil.shape != (2,2):
        raise ValueError("tilde_counts must be 2x2.")

    if not (0.0 <= p_sel <= 1.0 and 0.0 <= q_sel <= 1.0):
        raise ValueError("p_sel and q_sel must be in [0,1].")
    if n_val <= 1:
        raise ValueError("n_val must be > 1.")

    # Misclassification matrix A
    A = np.array([[p_sel, 1.0 - p_sel],
                  [1.0 - q_sel, q_sel]], dtype=float)

    lam = float(lambda_start)
    converged = False
    counts = None

    while lam <= lambda_max:
        A_lam = A + lam * np.eye(2)
        try:
            # counts = tilde T * (A_lam^{-T})
            inv_AT = np.linalg.inv(A_lam).T
            M = Ttil @ inv_AT
        except np.linalg.LinAlgError:
            lam *= step
            continue

        if np.all(M > eps) and np.all(np.isfinite(M)):
            counts = M
            converged = True
            break
        lam *= step

    if not converged:
        return LambdaORResult(
            log_or=float('nan'),
            neglog10_p=float('nan'),
            z=float('nan'),
            se=float('nan'),
            counts=np.full((2,2), np.nan),
            lambda_used=float('nan'),
            converged=False,
            meta={
                "message": "Ridge path failed to find feasible counts.",
                "lambda_max": lambda_max,
                "p_sel": p_sel,
                "q_sel": q_sel
            }
        )

    a, b = counts[0,0], counts[0,1]
    c, d = counts[1,0], counts[1,1]
    # Corrected log odds ratio
    log_or = log((a * d) / (b * c))

    # Base variance
    var_base = 1.0/a + 1.0/b + 1.0/c + 1.0/d

    # Delta-method extra variance from (p_sel, q_sel)
    J = p_sel + q_sel - 1.0
    # Protect against divide by zero; if J ~ 0, variance will be huge anyway
    denom = max(J*J, 1e-18)
    ad = a * d
    bc = b * c
    dlog_dp = ((1.0 - q_sel) * (ad - bc)) / (denom * ad)
    dlog_dq = ((1.0 - p_sel) * (bc - ad)) / (denom * ad)

    var_p = p_sel * (1.0 - p_sel) / float(n_val)
    var_q = q_sel * (1.0 - q_sel) / float(n_val)
    var_extra = (dlog_dp ** 2) * var_p + (dlog_dq ** 2) * var_q

    se = sqrt(var_base + var_extra)
    z = log_or / se if se > 0 else float('inf')

    # Two-sided p-value and -log10 p with asymptotic tail
    z_abs = abs(z)
    if z_abs <= 7.0:
        p_two = _normal_two_sided_p(z_abs)
        # Guard for underflow
        p_two = max(min(p_two, 1.0), 1e-323)
        neglog10_p = -log10(p_two)
    else:
        # Asymptotic: -log10 p ≈ z^2/(2 ln 10) - log10( sqrt(2π)*|z| )
        neglog10_p = (z_abs*z_abs) / (2.0 * np.log(10.0)) - log10((2.0*pi)**0.5 * z_abs)

    meta = {
        "var_base": var_base,
        "var_extra": var_extra,
        "J": J,
        "lambda_path_start": lambda_start,
        "lambda_used": lam,
        "step": step,
        "eps": eps,
        "p_sel": p_sel,
        "q_sel": q_sel,
        "n_val": n_val
    }

    return LambdaORResult(
        log_or=log_or,
        neglog10_p=neglog10_p,
        z=z,
        se=se,
        counts=counts,
        lambda_used=lam,
        converged=converged,
        meta=meta
    )


__all__ = ["lambda_or", "pq_from_two_gates", "LambdaORResult"]
