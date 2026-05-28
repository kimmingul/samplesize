"""Sample size for establishing reference intervals in clinical/lab medicine.

Implements:

* "Reference Intervals for Clinical and Lab Medicine"
  -> :func:`reference_intervals_clinical_lab`

Background
----------
A reference interval is the range [L, U] within which a specified fraction
(commonly 95%) of values from a healthy reference population are expected to
fall.  The standard parametric approach assumes normality and estimates the
interval from a reference sample of size n.

The limits are estimated as:
    L = x_bar - k * s,   U = x_bar + k * s

where k is the two-sided normal tolerance factor such that the interval
covers at least proportion ``coverage`` of the population with confidence
``confidence``.

Normal tolerance factor
-----------------------
For a two-sided normal tolerance interval covering proportion p with
confidence 1-gamma, the factor k(n, p, gamma) is defined by:

    P(interval covers >= p of pop) = 1 - gamma

Using the noncentral-t relationship (Hahn & Meeker 1991, eq. 4.13):

    k = t_{confidence, df=n-1, ncp=z_p * sqrt(n)} / sqrt(n)

where z_p = Phi^{-1}((1+p)/2).

At n=120 with p=0.95, confidence=0.90:  k ≈ 2.179 (slightly above z_{0.975}=1.960).

CLSI EP28-A3c minimum n
-----------------------
The CLSI EP28-A3c guideline recommends n >= 120 for a 95% reference interval
with 90% confidence.  This is derived from tolerance-interval theory and is
the standard recommendation in clinical laboratories.

Solving for n (precision-driven)
---------------------------------
The precision criterion can be stated as: find the smallest n such that the
tolerance factor k(n, p, confidence) <= target_k.

Since k decreases monotonically toward z_p as n -> infinity, the target must
satisfy target_k > z_p.  Typical choice: target_k = 2.0 for 95% interval.

References
----------
CLSI EP28-A3c (2010). Defining, Establishing, and Verifying Reference
Intervals in the Clinical Laboratory; Approved Guideline, 3rd ed.
Clinical and Laboratory Standards Institute.

Bland, M. (2000). An Introduction to Medical Statistics, 3rd ed.
Oxford University Press.  Chapter 8.

Hahn, G. J. and Meeker, W. Q. (1991). Statistical Intervals: A Guide for
Practitioners. Wiley.  Chapter 4.

Reed, A. H., Henry, R. J. and Mason, W. B. (2002). Influence of statistical
method used on the resulting estimate of normal range.
Clinical Chemistry, 17, 275-284.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import nct as _nct
from scipy.stats import norm as _norm


# ---------------------------------------------------------------------------
# Core computation: two-sided normal tolerance factor
# ---------------------------------------------------------------------------

def _z_p(coverage: float) -> float:
    """Normal quantile for two-sided coverage: Phi^{-1}((1+coverage)/2)."""
    return float(_norm.ppf((1.0 + coverage) / 2.0))


def _tolerance_factor(n: int, coverage: float, confidence: float) -> float:
    """Two-sided normal tolerance factor k(n, coverage, confidence).

    The factor k is such that [x_bar - k*s, x_bar + k*s] covers at least
    proportion ``coverage`` of a normal population with probability >=
    ``confidence``.

    Uses the noncentral-t inversion:
        k = nct.ppf(confidence, df=n-1, nc=z_p*sqrt(n)) / sqrt(n)

    Reference: Hahn & Meeker (1991), Statistical Intervals. Wiley.
    """
    zp = _z_p(coverage)
    ncp = zp * math.sqrt(n)
    df = n - 1
    t_quantile = float(_nct.ppf(confidence, df=df, nc=ncp))
    return t_quantile / math.sqrt(n)


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def reference_intervals_clinical_lab(
    *,
    coverage: float = 0.95,
    confidence: float = 0.90,
    target_k: float | None = None,
    n: int | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Sample size or tolerance factor for a normal reference interval.

    A reference interval [x_bar - k*s, x_bar + k*s] covers the central
    ``coverage`` fraction of the population with probability ``confidence``.
    The tolerance factor k depends on n, coverage, and confidence.

    **Two solve modes:**

    1. ``solve_for='n'`` (default when ``target_k`` given and ``n`` omitted):
       Find the smallest n such that k(n, coverage, confidence) <= ``target_k``.
       Requires ``target_k`` > z_{(1+coverage)/2}  (the limiting factor as n→∞).

    2. ``solve_for='power'`` (when ``n`` given):
       Compute the tolerance factor k achieved at sample size ``n``.
       ``achieved_power`` is set to ``confidence`` (the coverage probability
       is guaranteed by construction at any n; what changes is precision of k).

    **CLSI EP28-A3c mode** (when neither ``target_k`` nor ``n`` given):
       Returns the CLSI minimum recommendation: n=120 for a 95/90 reference
       interval, together with k(120, 0.95, 0.90) ≈ 2.179.

    Parameters
    ----------
    coverage
        Proportion of the reference population covered by the interval
        (default 0.95 for a 95% reference interval).
    confidence
        Required confidence that the tolerance limits are valid
        (default 0.90, as per CLSI EP28-A3c).
    target_k
        Maximum acceptable tolerance factor.  Must be > z_{(1+coverage)/2}.
        For a 95% interval z_p = 1.960; typical target_k = 2.0 to 2.5.
    n
        Sample size (required when solve_for='power').
    solve_for
        ``'n'`` or ``'power'``.  Inferred from provided arguments when omitted.

    Returns
    -------
    dict
        Standard envelope: method_id, solve_for, n, tolerance_factor_k,
        achieved_power, inputs_echo, citations.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be in (0, 1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    inputs_echo = {
        "coverage": coverage,
        "confidence": confidence,
        "target_k": target_k,
        "n": n,
        "solve_for": solve_for,
    }

    citations = [
        "CLSI EP28-A3c (2010). Defining, Establishing, and Verifying Reference "
        "Intervals in the Clinical Laboratory; Approved Guideline, 3rd ed. "
        "Clinical and Laboratory Standards Institute.",
        "Hahn, G. J. and Meeker, W. Q. (1991). Statistical Intervals: A Guide "
        "for Practitioners. Wiley.",
        "Bland, M. (2000). An Introduction to Medical Statistics, 3rd ed. "
        "Oxford University Press.",
        "Reed, A. H., Henry, R. J. and Mason, W. B. (2002). Influence of "
        "statistical method used on the resulting estimate of normal range. "
        "Clinical Chemistry, 17, 275-284.",
    ]

    # --- CLSI fixed-n mode ---
    if target_k is None and n is None:
        clsi_n = 120
        k = _tolerance_factor(clsi_n, coverage, confidence)
        return {
            "method_id": "reference_intervals_clinical_lab",
            "solve_for": "n",
            "n": clsi_n,
            "tolerance_factor_k": round(k, 4),
            "achieved_power": confidence,
            "inputs_echo": inputs_echo,
            "citations": citations,
            "notes": (
                "CLSI EP28-A3c minimum recommendation: n=120 for a 95% "
                "reference interval with 90% confidence.  Provide target_k "
                "for a precision-driven sample size."
            ),
        }

    # --- Infer solve_for ---
    if solve_for is None:
        if n is None:
            solve_for = "n"
        else:
            solve_for = "power"

    if solve_for == "n":
        if target_k is None:
            raise ValueError("target_k is required when solve_for='n'")
        zp = _z_p(coverage)
        if target_k <= zp:
            raise ValueError(
                f"target_k ({target_k}) must be > z_p ({zp:.4f}) for coverage={coverage}; "
                "the tolerance factor approaches z_p asymptotically as n -> infinity."
            )
        # Binary search for smallest n such that k(n) <= target_k.
        # k(n) is monotone-decreasing in n (achieved coverage tightens with N).
        n_max = 1_000_000
        # Verify the upper bound satisfies the condition; otherwise infeasible.
        if _tolerance_factor(n_max, coverage, confidence) > target_k:
            raise RuntimeError("failed to find n within limit")
        # Quick check at the lower bound.
        if _tolerance_factor(3, coverage, confidence) <= target_k:
            n_try = 3
        else:
            lo, hi = 3, n_max
            while lo < hi:
                mid = (lo + hi) // 2
                k_mid = _tolerance_factor(mid, coverage, confidence)
                if k_mid <= target_k:
                    hi = mid
                else:
                    lo = mid + 1
            n_try = lo

        achieved_k = _tolerance_factor(n_try, coverage, confidence)
        return {
            "method_id": "reference_intervals_clinical_lab",
            "solve_for": "n",
            "n": n_try,
            "tolerance_factor_k": round(achieved_k, 4),
            "achieved_power": confidence,
            "inputs_echo": inputs_echo,
            "citations": citations,
        }

    elif solve_for == "power":
        if n is None:
            raise ValueError("n is required when solve_for='power'")
        if n < 3:
            raise ValueError("n must be >= 3")
        k = _tolerance_factor(n, coverage, confidence)
        return {
            "method_id": "reference_intervals_clinical_lab",
            "solve_for": "power",
            "n": n,
            "tolerance_factor_k": round(k, 4),
            "achieved_power": confidence,
            "inputs_echo": inputs_echo,
            "citations": citations,
        }

    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")
