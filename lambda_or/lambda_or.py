"""
lambda_or.py — λ-OR (Lambda Odds Ratio) for misclassification-corrected attribution.

This module implements a ridge-stabilized inversion of a misclassified 2x2 contingency
table to recover corrected counts and compute the corrected log-odds ratio, its
variance (including uncertainty from selection-conditional sensitivity/specificity),
a Wald z-statistic, and a -log10 p-value with a large-|z| tail approximation.

Core API
--------
lambda_or(tilde_counts, p_sel, q_sel, n_val, lambda_start=1e-6, lambda_max=1e6,
          step=10.0, eps=1e-9) -> LambdaORResult

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
from math import log, sqrt, erfc, pi, log10
from typing import Tuple, Dict, Any
import numpy as np


@dataclass
class LambdaORResult:
    log_or: float
    neglog10_p: float
    z: float
    se: float
    counts: np.ndarray      # corrected 2x2 matrix [[a, b], [c, d]]
    lambda_used: float
    converged: bool
    meta: Dict[str, Any]


def _normal_two_sided_p(z_abs: float) -> float:
    """Two-sided p-value using the complementary error function."""
    return erfc(z_abs / sqrt(2.0))


def _neglog10_p_from_z(z_abs: float) -> float:
    """
    Return -log10(two-sided p) from |z|.

    For moderate |z|, use erfc directly. For very large |z|, use the Mills-ratio
    approximation for the two-sided normal tail:

        p ≈ 2 * phi(z) / z

    which implies

        -log10 p ≈ z^2 / (2 ln 10) + log10(z) + 0.5 log10(2π) - log10(2).
    """
    if z_abs <= 7.0:
        p_two = max(min(_normal_two_sided_p(z_abs), 1.0), 1e-323)
        return -log10(p_two)

    return (
        (z_abs * z_abs) / (2.0 * np.log(10.0))
        + log10(sqrt(2.0 * pi) * z_abs)
        - log10(2.0)
    )


def neglog10_p_from_z(z_abs: float) -> float:
    return _neglog10_p_from_z(z_abs)


def pq_from_two_gates(pi_H1: float, pi_L1: float, pi_H0: float, pi_L0: float) -> Tuple[float, float]:
    """
    Compute selection-conditional sensitivity/specificity from two-gate ROC probabilities.

    Parameters
    ----------
    pi_H1 : float
        P(S in high-specificity gate | Y=1)
    pi_L1 : float
        P(S in low-score gate | Y=1)
    pi_H0 : float
        P(S in high-specificity gate | Y=0)
    pi_L0 : float
        P(S in low-score gate | Y=0)
    """
    denom1 = pi_H1 + pi_L1
    denom0 = pi_H0 + pi_L0
    if denom1 <= 0 or denom0 <= 0:
        raise ValueError("Invalid gate probabilities: denominators must be positive.")
    p_sel = pi_H1 / denom1
    q_sel = pi_L0 / denom0
    return p_sel, q_sel


def lambda_or(
    tilde_counts: np.ndarray,
    p_sel: float,
    q_sel: float,
    n_val: int,
    lambda_start: float = 1e-6,
    lambda_max: float = 1e6,
    step: float = 10.0,
    eps: float = 1e-9,
) -> LambdaORResult:
    """
    Compute λ-OR given a misclassified 2x2 table and selection-conditional (p_sel, q_sel).

    The variance calculation follows the manuscript algorithm: after obtaining the
    ridge-corrected table T^(λ) = \tilde T (K^T + λI)^(-1), propagate uncertainty in
    (p_sel, q_sel) using the ridge-aware gradient g_p, g_q derived from the corrected
    inverse rather than a reduced J-only approximation.
    """
    Ttil = np.asarray(tilde_counts, dtype=float)
    if Ttil.shape != (2, 2):
        raise ValueError("tilde_counts must be 2x2.")

    if not (0.0 <= p_sel <= 1.0 and 0.0 <= q_sel <= 1.0):
        raise ValueError("p_sel and q_sel must be in [0,1].")
    if n_val <= 1:
        raise ValueError("n_val must be > 1.")
    if lambda_start <= 0.0:
        raise ValueError("lambda_start must be positive.")
    if lambda_max < lambda_start:
        raise ValueError("lambda_max must be >= lambda_start.")
    if step <= 1.0:
        raise ValueError("step must be > 1.0 for a multiplicative ridge path.")

    # Misclassification matrix K.
    K = np.array(
        [[p_sel, 1.0 - p_sel],
         [1.0 - q_sel, q_sel]],
        dtype=float,
    )

    lam = float(lambda_start)
    converged = False
    counts = None
    A_inv = None  # A = (K^T + λI)^(-1) in the manuscript notation.

    while lam <= lambda_max:
        M_lam = K.T + lam * np.eye(2)
        try:
            A_inv = np.linalg.inv(M_lam)
            M = Ttil @ A_inv
        except np.linalg.LinAlgError:
            lam *= step
            continue

        if np.all(M >= eps) and np.all(np.isfinite(M)):
            counts = M
            converged = True
            break
        lam *= step

    if not converged or counts is None or A_inv is None:
        return LambdaORResult(
            log_or=float("nan"),
            neglog10_p=float("nan"),
            z=float("nan"),
            se=float("nan"),
            counts=np.full((2, 2), np.nan),
            lambda_used=float("nan"),
            converged=False,
            meta={
                "message": "Ridge path failed to find feasible counts.",
                "lambda_max": lambda_max,
                "p_sel": p_sel,
                "q_sel": q_sel,
            },
        )

    a, b = counts[0, 0], counts[0, 1]
    c, d = counts[1, 0], counts[1, 1]

    # Corrected log odds ratio.
    log_or = log((a * d) / (b * c))

    # Base variance.
    var_base = 1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d

    # Ridge-aware delta-method gradient from the manuscript algorithm.
    # Here A_inv = (K^T + λI)^(-1), with entries A_ij in the paper's notation.
    A11, A12 = float(A_inv[0, 0]), float(A_inv[0, 1])
    A21, A22 = float(A_inv[1, 0]), float(A_inv[1, 1])

    g_p = (a - b) * (A12 / b - A11 / a) + (c - d) * (A11 / c - A12 / d)
    g_q = (a - b) * (A21 / a - A22 / b) + (c - d) * (A22 / d - A21 / c)

    var_p = p_sel * (1.0 - p_sel) / float(n_val)
    var_q = q_sel * (1.0 - q_sel) / float(n_val)
    var_extra = (g_p * g_p) * var_p + (g_q * g_q) * var_q

    se = sqrt(var_base + var_extra)
    z = log_or / se if se > 0 else float("inf")
    z_abs = abs(z)
    neglog10_p = _neglog10_p_from_z(z_abs)

    meta = {
        "var_base": var_base,
        "var_extra": var_extra,
        "var_p": var_p,
        "var_q": var_q,
        "g_p": g_p,
        "g_q": g_q,
        "K": K,
        "A_inv": A_inv,
        "J": p_sel + q_sel - 1.0,
        "lambda_path_start": lambda_start,
        "lambda_used": lam,
        "step": step,
        "eps": eps,
        "p_sel": p_sel,
        "q_sel": q_sel,
        "n_val": n_val,
    }

    return LambdaORResult(
        log_or=log_or,
        neglog10_p=neglog10_p,
        z=z,
        se=se,
        counts=counts,
        lambda_used=lam,
        converged=converged,
        meta=meta,
    )


__all__ = ["lambda_or", "pq_from_two_gates", "LambdaORResult",    "neglog10_p_from_z"
]
