"""Acceptance sampling for attributes (OC-curve / sample size).


Finds the sample size n and acceptance number c for a single-sampling
plan based on two operating-characteristic (OC) points:

  * Acceptable Quality Level (AQL = p0): the producer wants lots at
    this defect rate accepted with probability >= 1-alpha.
  * Limiting Quality Level (LQL = p1): the consumer wants lots at
    this defect rate accepted with probability <= beta.

For a **finite** lot of size N the hypergeometric distribution is used:

    H(c; N, M, n) = sum_{j=0}^{c} h(j; N, M, n)
    h(j; N, M, n) = C(M,j) * C(N-M, n-j) / C(N, n)

For an **infinite** lot the binomial distribution is used:

    B(c; n, p) = sum_{j=0}^{c} C(n,j) * p^j * (1-p)^(n-j)

The algorithm searches over (n, c) pairs — increasing n from 1,
and for each n finding the smallest c such that the binomial/hypergeometric
CDF at p0 >= 1-alpha.  The pair is accepted when the CDF at p1 <= beta.

References
----------
* Kenett, R.S. and Zacks, S. (2014). Modern Industrial Statistics,
  2nd Ed. John Wiley & Sons.
* Ryan, T.P. (2013). Sample Size Determination and Power.
  John Wiley & Sons.
"""
from __future__ import annotations

from typing import Any

from scipy.stats import binom, hypergeom


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------


def _binom_cdf(c: int, n: int, p: float) -> float:
    """P(X <= c) where X ~ Binomial(n, p)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if c >= n else 0.0
    return float(binom.cdf(c, n, p))


def _hypgeom_cdf(c: int, lot_size: int, m: int, n: int) -> float:
    """P(X <= c) where X ~ Hypergeometric(lot_size, m, n).

    m = number of defectives in lot, n = sample size.
    """
    if m <= 0:
        return 1.0
    if m >= lot_size:
        return 1.0 if c >= n else 0.0
    return float(hypergeom.cdf(c, lot_size, m, n))


def _pa_at_p(
    c: int, n: int, p: float, lot_size: int | None
) -> float:
    """Probability of acceptance P(X <= c) for defect rate p."""
    if lot_size is None:
        # Infinite lot — binomial
        return _binom_cdf(c, n, p)
    else:
        m = int(round(lot_size * p))
        return _hypgeom_cdf(c, lot_size, m, n)


# ---------------------------------------------------------------------------
# Search algorithm
# ---------------------------------------------------------------------------


def _find_n_c(
    *,
    p0: float,
    alpha: float,
    p1: float,
    beta: float,
    lot_size: int | None,
    n_max: int = 10_000,
) -> tuple[int, int, float, float]:
    """Search for smallest n and corresponding c satisfying OC constraints.

    Returns (n, c, actual_alpha, actual_beta).
    """
    for n in range(1, n_max + 1):
        # Find smallest c such that Pa(p0) >= 1-alpha
        # (producer's constraint: accept good lots with high probability)
        c_candidate = -1
        for c in range(0, n + 1):
            pa_p0 = _pa_at_p(c, n, p0, lot_size)
            if pa_p0 >= 1.0 - alpha:
                c_candidate = c
                break
        if c_candidate < 0:
            continue

        # Check consumer constraint: Pa(p1) <= beta
        pa_p1 = _pa_at_p(c_candidate, n, p1, lot_size)
        if pa_p1 <= beta:
            # Found a valid plan
            actual_alpha = 1.0 - _pa_at_p(c_candidate, n, p0, lot_size)
            actual_beta = pa_p1
            return n, c_candidate, actual_alpha, actual_beta

    raise RuntimeError(
        f"Could not find valid (n, c) within n_max={n_max}. "
        "Try relaxing alpha, beta, or adjusting p0/p1."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def acceptance_sampling_attributes(
    *,
    p0: float,
    alpha: float = 0.05,
    p1: float,
    beta: float = 0.10,
    lot_size: int | None = None,
    n_max: int = 10_000,
) -> dict[str, Any]:
    """Sample size and acceptance number for lot acceptance sampling.


    Finds the minimum sample size n and acceptance number c such that:
      - P(accept | defect rate = p0) >= 1 - alpha  (producer's risk)
      - P(accept | defect rate = p1) <= beta        (consumer's risk)

    Parameters
    ----------
    p0
        Acceptable Quality Level (AQL): highest defect proportion for
        which the lot should still be accepted with high probability.
    alpha
        Producer's risk: probability of rejecting a lot at quality p0.
        Default 0.05.
    p1
        Limiting Quality Level (LQL / LTPD): defect proportion above
        which the lot should be rejected with high probability.
    beta
        Consumer's risk: probability of accepting a lot at quality p1.
        Default 0.10.
    lot_size
        Finite lot size N.  If None (default), the lot is treated as
        infinite and the binomial distribution is used.  For finite
        lots, the hypergeometric distribution is used.
    n_max
        Maximum sample size to search up to.  Default 10 000.
    """
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be in (0, 1)")
    if not 0.0 < p1 < 1.0:
        raise ValueError("p1 must be in (0, 1)")
    if p1 <= p0:
        raise ValueError("p1 (LQL) must be greater than p0 (AQL)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    if lot_size is not None and lot_size < 2:
        raise ValueError("lot_size must be >= 2 if specified")

    n, c, actual_alpha, actual_beta = _find_n_c(
        p0=p0, alpha=alpha, p1=p1, beta=beta,
        lot_size=lot_size, n_max=n_max,
    )

    inputs_echo: dict[str, Any] = {
        "p0": p0, "alpha": alpha, "p1": p1, "beta": beta,
        "lot_size": lot_size, "n_max": n_max,
    }

    dist_type = "hypergeometric" if lot_size is not None else "binomial"

    return {
        "method_id": "acceptance_sampling_attributes",
        "solve_for": "n",
        "n": n,
        "achieved_power": 1.0 - actual_beta,
        "acceptance_number": c,
        "rejection_number": c + 1,
        "actual_producer_risk": actual_alpha,
        "actual_consumer_risk": actual_beta,
        "distribution": dist_type,
        "inputs_echo": inputs_echo,
        "citations": [
            "Kenett, R.S. and Zacks, S. (2014). Modern Industrial Statistics, "
            "2nd Ed. John Wiley & Sons.",
            "Ryan, T.P. (2013). Sample Size Determination and Power. "
            "John Wiley & Sons.",
        ],
    }
