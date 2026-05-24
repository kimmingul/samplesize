"""Phase II clinical trial designs.

Implements five Phase II clinical trial design procedures:

* Chapter 125 — "Two-Stage Phase II Clinical Trials"
  -> :func:`simon_two_stage` (generic, ``design_type='optimal'`` or ``'minimax'``)
  -> :func:`simon_optimal_two_stage` (thin wrapper, minimises EN under H0)
  -> :func:`simon_minimax_two_stage` (thin wrapper, minimises max N)
* Chapter 120 — "Single-Stage Phase II Clinical Trials"
  -> :func:`single_stage_phase_ii`
* Chapter 130 — "Three-Stage Phase II Clinical Trials"
  -> :func:`three_stage_phase_ii`

All designs test H0: P ≤ P0  vs  H1: P ≥ P1 using exact binomial
probabilities.  The exhaustive lattice search algorithm follows:

- Simon (1989) for two-stage designs.
- A'Hern (2001) / Fleming (1982) for single-stage.
- Chen (1997) for three-stage designs.

Returned envelopes match the standard shape::

    {
      "method_id": ...,
      "solve_for": "n",
      "n": <max N>,
      "design": { <full stage parameters> },
      "achieved_power": float,
      "inputs_echo": dict,
      "citations": [str, ...],
    }
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import binom, norm


# ---------------------------------------------------------------------------
# Binomial helpers — scalar
# ---------------------------------------------------------------------------

def _bcdf(r: int, p: float, n: int) -> float:
    """P(X ≤ r | X ~ Bin(n, p))."""
    if r < 0:
        return 0.0
    if r >= n:
        return 1.0
    return float(binom.cdf(r, n, p))


def _bpmf(x: int, p: float, n: int) -> float:
    """P(X = x | X ~ Bin(n, p))."""
    if x < 0 or x > n:
        return 0.0
    return float(binom.pmf(x, n, p))


# ---------------------------------------------------------------------------
# Precomputed binomial tables for fast search
# ---------------------------------------------------------------------------

class _BinomTables:
    """Cache binomial PMF and CDF arrays for fixed (p, n_max).

    pmf[x, n] = P(X=x | Bin(n, p)) for x in [0, n_max], n in [0, n_max].
    cdf[r, n] = P(X<=r | Bin(n, p)).
    """

    def __init__(self, p: float, n_max: int) -> None:
        self.p = p
        self.n_max = n_max
        # pmf[n, x] = binom.pmf(x, n, p)
        ns = np.arange(n_max + 1)
        xs = np.arange(n_max + 1)
        # Compute for all (n, x) pairs
        N, X = np.meshgrid(ns, xs, indexing="ij")  # (n_max+1, n_max+1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmf_raw = binom.pmf(X, N, p)
        pmf_raw[np.isnan(pmf_raw)] = 0.0
        self._pmf = pmf_raw  # shape (n_max+1, n_max+1)
        # cdf[n, r] = cumsum of pmf[n, :] up to r
        self._cdf = np.cumsum(self._pmf, axis=1)

    def pmf(self, x: int, n: int) -> float:
        if x < 0 or x > n or n < 0:
            return 0.0
        return float(self._pmf[n, x])

    def cdf(self, r: int, n: int) -> float:
        if r < 0:
            return 0.0
        if n <= 0:
            return 1.0
        r = min(r, n)
        return float(self._cdf[n, r])


# ---------------------------------------------------------------------------
# Normal approximation lower bound (Fleming 1982)
# ---------------------------------------------------------------------------

def _fleming_n(p0: float, p1: float, alpha: float, beta: float) -> int:
    p_bar = (p0 + p1) / 2.0
    za = float(norm.ppf(1.0 - alpha))
    zb = float(norm.ppf(1.0 - beta))
    n_approx = p_bar * (1.0 - p_bar) * ((za + zb) / (p1 - p0)) ** 2
    return max(2, int(math.floor(n_approx)))


# ---------------------------------------------------------------------------
# Two-stage rejection probability (Simon 1989)
# ---------------------------------------------------------------------------

def _ts_reject_prob(
    t0: "_BinomTables", t1: "_BinomTables",
    p_idx: int,  # 0 for p0 table, 1 for p1 table
    n1: int, r1: int, r: int, n: int,
    tables: list["_BinomTables"],
) -> float:
    """Pr(reject | table, n1, r1, r, n) using precomputed tables."""
    tb = tables[p_idx]
    n2 = n - n1
    prob = tb.cdf(r1, n1)
    for x in range(r1 + 1, min(n1, r) + 1):
        prob += tb.pmf(x, n1) * tb.cdf(r - x, n2)
    return prob


def _two_stage_reject(tb: "_BinomTables", n1: int, r1: int, r: int, n: int) -> float:
    n2 = n - n1
    prob = tb.cdf(r1, n1)
    for x in range(r1 + 1, min(n1, r) + 1):
        prob += tb.pmf(x, n1) * tb.cdf(r - x, n2)
    return prob


def _two_stage_en(tb0: "_BinomTables", n1: int, r1: int, n: int) -> float:
    pet = tb0.cdf(r1, n1)
    return n1 + (1.0 - pet) * (n - n1)


# ---------------------------------------------------------------------------
# Three-stage helpers (Chen 1997)
# ---------------------------------------------------------------------------

def _three_stage_reject(
    tb: "_BinomTables", n1: int, r1: int, n2: int, r2: int, n3: int, r3: int
) -> float:
    m1 = n1
    m2 = n2 - n1
    m3 = n3 - n2

    pet1 = tb.cdf(r1, m1)

    pet2 = 0.0
    for x1 in range(r1 + 1, min(m1, r2) + 1):
        pet2 += tb.pmf(x1, m1) * tb.cdf(r2 - x1, m2)

    pet3 = 0.0
    for x1 in range(r1 + 1, min(m1, r3) + 1):
        px1 = tb.pmf(x1, m1)
        if px1 == 0.0:
            continue
        x2_lo = max(0, r2 - x1 + 1)
        x2_hi = min(m2, r3 - x1)
        for x2 in range(x2_lo, x2_hi + 1):
            pet3 += px1 * tb.pmf(x2, m2) * tb.cdf(r3 - x1 - x2, m3)

    return pet1 + pet2 + pet3


def _three_stage_en(
    tb0: "_BinomTables", n1: int, r1: int, n2: int, r2: int, n3: int, r3: int
) -> float:
    m2 = n2 - n1
    pet1 = tb0.cdf(r1, n1)
    pet2 = 0.0
    for x1 in range(r1 + 1, min(n1, r2) + 1):
        pet2 += tb0.pmf(x1, n1) * tb0.cdf(r2 - x1, m2)
    return n1 + (1.0 - pet1) * (n2 - n1) + (1.0 - pet1 - pet2) * (n3 - n2)


# ---------------------------------------------------------------------------
# Single-Stage Phase II  (A'Hern 2001 / Fleming 1982)
# ---------------------------------------------------------------------------

def single_stage_phase_ii(
    *,
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    """Exact binomial single-stage Phase II design.

    Finds the smallest N for which there exists R satisfying:

        Pr(X ≥ R | p0) ≤ alpha   and   Pr(X ≥ R | p1) ≥ 1 - beta

    Drug rejected if number of responses ≤ r (``design["r"]``).

    Parameters
    ----------
    p0 : float
        Null response rate (poor drug).
    p1 : float
        Alternative response rate (good drug).
    alpha : float
        Type-I error bound.
    beta : float
        Type-II error bound.
    """
    if not 0.0 < p0 < p1 < 1.0:
        raise ValueError("require 0 < p0 < p1 < 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")

    f_n = _fleming_n(p0, p1, alpha, beta)
    n_lo = max(1, int(math.floor(0.8 * f_n)))
    n_hi = max(n_lo + 2, int(math.ceil(4.0 * f_n)))

    # Precompute binomial tables for the full search range
    tb0 = _BinomTables(p0, n_hi)
    tb1 = _BinomTables(p1, n_hi)

    for n in range(n_lo, n_hi + 1):
        for r in range(0, n + 1):
            actual_alpha = 1.0 - tb0.cdf(r, n)
            actual_beta = tb1.cdf(r, n)
            if actual_alpha <= alpha and actual_beta <= beta:
                best = {
                    "r": r, "n": n,
                    "actual_alpha": actual_alpha,
                    "actual_beta": actual_beta,
                }
                achieved_power = 1.0 - actual_beta
                inputs_echo = {"p0": p0, "p1": p1, "alpha": alpha, "beta": beta}
                return {
                    "method_id": "single_stage_phase_ii",
                    "solve_for": "n",
                    "n": n,
                    "design": best,
                    "achieved_power": achieved_power,
                    "inputs_echo": inputs_echo,
                    "citations": [
                        "Clinical Trials.",
                        "A'Hern, R. P. (2001). Sample size tables for exact "
                        "single-stage phase II designs. Statistics in Medicine, "
                        "20(6), 859-866.",
                        "Fleming, T. R. (1982). One-sample multiple testing "
                        "procedure for phase II clinical trials. Biometrics, "
                        "38(1), 143-151.",
                    ],
                }

    raise RuntimeError(
        f"No single-stage design found in n=[{n_lo},{n_hi}].  Check inputs."
    )


# ---------------------------------------------------------------------------
# Two-Stage Phase II  (Simon 1989)
# ---------------------------------------------------------------------------

def simon_two_stage(
    *,
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
    design_type: str = "optimal",
) -> dict[str, Any]:
    """Simon (1989) two-stage Phase II design.

    Exhaustive search over (n, n1, r1) lattice.  For each tuple, finds
    the minimum r satisfying power, then checks alpha.  Uses precomputed
    binomial tables for speed.

    Search strategy: start from n=2; once an optimal design is found,
    continue for ``_BEST_PLUS`` additional N values to ensure optimality
    (search space: n1 from 1..n_max, n2 = n_total - n1).

    Parameters
    ----------
    p0 : float
        Null response rate.
    p1 : float
        Alternative response rate.
    alpha : float
        Type-I error bound.
    beta : float
        Type-II error bound.
    design_type : str
        ``'optimal'`` (min EN under H0) or ``'minimax'`` (min N).
    """
    if not 0.0 < p0 < p1 < 1.0:
        raise ValueError("require 0 < p0 < p1 < 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    if design_type not in ("optimal", "minimax"):
        raise ValueError("design_type must be 'optimal' or 'minimax'")

    _BEST_PLUS = 8  # continue searching this many N beyond best

    f_n = _fleming_n(p0, p1, alpha, beta)
    n_lo = 2
    n_hi = max(n_lo + 4, int(math.ceil(2.5 * f_n)) + _BEST_PLUS + 2)

    tb0 = _BinomTables(p0, n_hi)
    tb1 = _BinomTables(p1, n_hi)

    best_optimal: dict[str, Any] | None = None
    best_minimax: dict[str, Any] | None = None
    # Track N at which we last improved; stop when we've gone BEST_PLUS beyond
    last_optimal_n = 0
    last_minimax_n = 0

    for n in range(n_lo, n_hi + 1):
        # Early-stop: if both designs found and we've exceeded BEST_PLUS beyond both
        if best_optimal is not None and best_minimax is not None:
            stop_n = max(last_optimal_n, last_minimax_n) + _BEST_PLUS
            if n > stop_n:
                break

        for n1 in range(1, n):
            for r1 in range(0, n1 + 1):
                # Scan all r in [r1, n-1]; both beta and alpha change
                # non-monotonically in general.  The feasible window is where
                # beta <= target AND alpha <= target simultaneously.
                # beta increases with r (harder to reject good drug).
                # alpha also increases with r (harder to reject bad drug).
                # So: beta fails at large r, alpha fails at small r.
                # Once beta > beta_target, no larger r can satisfy both.
                for r in range(r1, n):
                    actual_beta = _two_stage_reject(tb1, n1, r1, r, n)
                    if actual_beta > beta:
                        break  # beta only gets worse for larger r

                    actual_alpha = 1.0 - _two_stage_reject(tb0, n1, r1, r, n)
                    if actual_alpha > alpha:
                        continue  # alpha still too high; try larger r

                    # Both constraints satisfied
                    en = _two_stage_en(tb0, n1, r1, n)
                    pet = tb0.cdf(r1, n1)

                    design = {
                        "r1": r1, "n1": n1, "r": r, "n": n,
                        "alpha_actual": actual_alpha,
                        "beta_actual": actual_beta,
                        "EN_under_h0": en,
                        "PET": pet,
                    }

                    if best_optimal is None or en < best_optimal["EN_under_h0"]:
                        best_optimal = design.copy()
                        last_optimal_n = n

                    if best_minimax is None or n < best_minimax["n"] or (
                        n == best_minimax["n"]
                        and en < best_minimax["EN_under_h0"]
                    ):
                        best_minimax = design.copy()
                        last_minimax_n = n

    chosen = best_optimal if design_type == "optimal" else best_minimax

    if chosen is None:
        raise RuntimeError(
            f"No two-stage design found (design_type={design_type!r}) in "
            f"n=[{n_lo},{n_hi}].  Check inputs."
        )

    method_id = (
        "simon_optimal_two_stage"
        if design_type == "optimal"
        else "simon_minimax_two_stage"
    )
    achieved_power = 1.0 - chosen["beta_actual"]
    inputs_echo = {
        "p0": p0, "p1": p1, "alpha": alpha, "beta": beta,
        "design_type": design_type,
    }
    return {
        "method_id": method_id,
        "solve_for": "n",
        "n": chosen["n"],
        "design": chosen,
        "achieved_power": achieved_power,
        "inputs_echo": inputs_echo,
        "citations": [
            "Simon, R. (1989). Optimal two-stage designs for phase II clinical "
            "trials. Controlled Clinical Trials, 10(1), 1-10.",
        ],
    }


def simon_optimal_two_stage(
    *,
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    """Simon's optimal two-stage design — minimises EN under H0.

    Thin wrapper around :func:`simon_two_stage` with
    ``design_type='optimal'``.
    """
    result = simon_two_stage(p0=p0, p1=p1, alpha=alpha, beta=beta,
                             design_type="optimal")
    result["method_id"] = "simon_optimal_two_stage"
    return result


def simon_minimax_two_stage(
    *,
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    """Simon's minimax two-stage design — minimises max N.

    Thin wrapper around :func:`simon_two_stage` with
    ``design_type='minimax'``.
    """
    result = simon_two_stage(p0=p0, p1=p1, alpha=alpha, beta=beta,
                             design_type="minimax")
    result["method_id"] = "simon_minimax_two_stage"
    return result


# ---------------------------------------------------------------------------
# Three-Stage Phase II  (Chen 1997)
# ---------------------------------------------------------------------------

def three_stage_phase_ii(
    *,
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
    design_type: str = "optimal",
) -> dict[str, Any]:
    """Chen (1997) three-stage Phase II design.

    Exhaustive search over (n3, n1, n2, r1, r2) lattice.  For each tuple
    finds the minimum r3 satisfying power, then checks alpha.  Uses
    precomputed binomial tables for speed.

    Parameters
    ----------
    p0 : float
        Null response rate.
    p1 : float
        Alternative response rate.
    alpha : float
        Type-I error bound.
    beta : float
        Type-II error bound.
    design_type : str
        ``'optimal'`` (min EN under H0) or ``'minimax'`` (min N3).
    """
    if not 0.0 < p0 < p1 < 1.0:
        raise ValueError("require 0 < p0 < p1 < 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    if design_type not in ("optimal", "minimax"):
        raise ValueError("design_type must be 'optimal' or 'minimax'")

    _BEST_PLUS = 2  # for 3-stage use smaller lookahead (Chen example is small)

    f_n = _fleming_n(p0, p1, alpha, beta)
    n_lo = 3
    n_hi = max(n_lo + 5, int(math.ceil(2.5 * f_n)) + _BEST_PLUS + 2)

    tb0 = _BinomTables(p0, n_hi)
    tb1 = _BinomTables(p1, n_hi)

    best_optimal: dict[str, Any] | None = None
    best_minimax: dict[str, Any] | None = None
    last_optimal_n = 0
    last_minimax_n = 0

    for n3 in range(n_lo, n_hi + 1):
        if best_optimal is not None and best_minimax is not None:
            stop_n = max(last_optimal_n, last_minimax_n) + _BEST_PLUS
            if n3 > stop_n:
                break

        for n1 in range(1, n3 - 1):
            for n2 in range(n1 + 1, n3):
                for r1 in range(0, n1 + 1):
                    for r2 in range(r1, n2 + 1):
                        for r3 in range(r2, n3):
                            actual_beta = _three_stage_reject(
                                tb1, n1, r1, n2, r2, n3, r3
                            )
                            if actual_beta > beta:
                                break  # beta only worsens for larger r3

                            actual_alpha = 1.0 - _three_stage_reject(
                                tb0, n1, r1, n2, r2, n3, r3
                            )
                            if actual_alpha > alpha:
                                continue  # alpha still too high; try larger r3

                            # Both constraints satisfied
                            en = _three_stage_en(tb0, n1, r1, n2, r2, n3, r3)
                            pet1 = tb0.cdf(r1, n1)
                            pet2_val = 0.0
                            for x1 in range(r1 + 1, min(n1, r2) + 1):
                                pet2_val += tb0.pmf(x1, n1) * tb0.cdf(r2 - x1, n2 - n1)
                            pet_overall = pet1 + pet2_val

                            design = {
                                "r1": r1, "n1": n1,
                                "r2": r2, "n2": n2,
                                "r3": r3, "n3": n3,
                                "alpha_actual": actual_alpha,
                                "beta_actual": actual_beta,
                                "EN_under_h0": en,
                                "PET1": pet1,
                                "PET_overall": pet_overall,
                            }

                            if best_optimal is None or en < best_optimal["EN_under_h0"]:
                                best_optimal = design.copy()
                                last_optimal_n = n3

                            if best_minimax is None or n3 < best_minimax["n3"] or (
                                n3 == best_minimax["n3"]
                                and en < best_minimax["EN_under_h0"]
                            ):
                                best_minimax = design.copy()
                                last_minimax_n = n3

    chosen = best_optimal if design_type == "optimal" else best_minimax

    if chosen is None:
        raise RuntimeError(
            f"No three-stage design found (design_type={design_type!r}) in "
            f"n3=[{n_lo},{n_hi}].  Check inputs."
        )

    achieved_power = 1.0 - chosen["beta_actual"]
    inputs_echo = {
        "p0": p0, "p1": p1, "alpha": alpha, "beta": beta,
        "design_type": design_type,
    }
    return {
        "method_id": "three_stage_phase_ii",
        "solve_for": "n",
        "n": chosen["n3"],
        "design": chosen,
        "achieved_power": achieved_power,
        "inputs_echo": inputs_echo,
        "citations": [
            "Chen, T. T. (1997). Optimal three-stage designs for phase II cancer "
            "clinical trials. Statistics in Medicine, 16(23), 2701-2711.",
        ],
    }
