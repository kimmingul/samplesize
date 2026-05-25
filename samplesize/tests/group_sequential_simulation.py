"""Group-sequential simulation-based power calculators.

Implements five simulation-based group-sequential power procedures via Monte-Carlo
testing.  All procedures share the same simulation engine: draw ``n_sims``
independent trial replicates (each of size n1+n2), apply O'Brien-Fleming (or
Pocock) alpha-spending boundaries at each equally-spaced look using the
cumulative data from that trial, and record whether any look crossed the
boundary.

Group-sequential simulation procedure:
1. Draw a full trial of n1 subjects in group 1 and n2 in group 2 upfront.
2. At each look k (information fraction tau_k), compute the test statistic
   using only the first ceil(tau_k * n1) subjects from each arm.
3. Stop and count an exit if the statistic crosses the boundary.

This approach yields power estimates consistent
with the analytic Lan-DeMets alpha-spending formulation.

Supported chapters
------------------
* ``group_sequential_two_means_simulation``
  Group-Sequential Tests for Two Means (Simulation)

* ``group_sequential_two_proportions_simulation``
  Group-Sequential Tests for Two Proportions (Simulation)

* ``group_sequential_logrank_simulation``
  Group-Sequential Logrank Tests (Simulation)

* ``group_sequential_ni_two_means_simulation``
  Group-Sequential Non-Inferiority Tests for Two Means (Simulation)

* ``group_sequential_ni_two_proportions_difference_simulation``
  Group-Sequential Non-Inferiority Tests for the Difference of Two
  Proportions (Simulation)"

Tolerance notes
---------------
Simulation power values have inherent Monte-Carlo variance.  With
``n_sims=10000`` and ``seed=42``, precision is approximately ±0.010 on the
power estimate (95% CI half-width ≈ 0.010 at power=0.80).

Validation tolerance for fixtures: ``achieved_power`` ±0.015.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from samplesize.tests.group_sequential import (
    _SPENDING,
    _solve_boundaries,
    _phi,
    _exit_probability,
)


# ---------------------------------------------------------------------------
# One-sided boundary solver for NI simulation procedures.
# The existing _solve_boundaries / _alpha_spent_obf was designed for the
# analytic two-sided case.  For one-sided group-sequential NI tests the convention uses
# the one-sided OBF spending:
#   alpha(tau) = 1 - Phi(z_{1-alpha} / sqrt(tau))
# and Pocock spending unchanged (it is already one-sided-correct).
# We provide a standalone solver here so that validated analytic code is
# not modified.
# ---------------------------------------------------------------------------

def _alpha_spent_obf_1sided(tau: float, alpha: float) -> float:
    """One-sided O'Brien-Fleming Lan-DeMets spending function.

    alpha(tau) = 1 - Phi(z_{1-alpha} / sqrt(tau))
    which equals alpha at tau=1 and 0 at tau=0.
    """
    import math as _m
    from samplesize.core import distributions as _D
    if tau <= 0.0:
        return 0.0
    if tau >= 1.0:
        return alpha
    z = _D.norm_ppf(1.0 - alpha)
    return 1.0 - _phi(z / _m.sqrt(tau))


def _alpha_spent_pocock_1sided(tau: float, alpha: float) -> float:
    """One-sided Pocock Lan-DeMets spending (same formula as two-sided)."""
    import math as _m
    if tau <= 0.0:
        return 0.0
    if tau >= 1.0:
        return alpha
    return alpha * _m.log(1.0 + (_m.e - 1.0) * tau)


_SPENDING_1SIDED = {
    "obrien-fleming": _alpha_spent_obf_1sided,
    "obf": _alpha_spent_obf_1sided,
    "pocock": _alpha_spent_pocock_1sided,
}


def _solve_boundaries_1sided(
    info_frac: list[float],
    alpha: float,
    spending: str,
) -> list[float]:
    """Solve one-sided GS boundaries using the corrected 1-sided spending."""
    spend_fn = _SPENDING_1SIDED[spending]
    K = len(info_frac)
    boundaries: list[float] = []
    for k in range(K):
        tau_k = info_frac[k]
        cum_alpha_k = spend_fn(tau_k, alpha)

        def f(bk: float, k=k, boundaries=boundaries,
              cum_alpha_k=cum_alpha_k) -> float:
            trial = boundaries + [bk]
            return _exit_probability(
                trial, info_frac[: k + 1], drift=0.0, sides=1,
            ) - cum_alpha_k

        lo, hi = 0.5, 12.0
        tries = 0
        while f(lo) < 0.0 and tries < 30:
            lo *= 0.5
            tries += 1
        tries = 0
        while f(hi) > 0.0 and tries < 30:
            hi *= 1.3
            tries += 1
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-7:
                break
        boundaries.append(0.5 * (lo + hi))
    return boundaries


# ---------------------------------------------------------------------------
# Shared simulation engine
# ---------------------------------------------------------------------------

def _gs_sim_power(
    *,
    n1: int,
    n2: int,
    boundaries: list[float],
    info_frac: list[float],
    draw_fn,   # callable(rng, n1, n2) -> (data1, data2): draw full trial
    stat_fn,   # callable(data1[:n1k], data2[:n2k]) -> float (z-statistic)
    sides: int,
    n_sims: int,
    rng: np.random.Generator,
) -> float:
    """Fraction of simulated trials that cross a boundary under H1.

    Each simulated trial draws the full sample upfront, then evaluates
    the test statistic using cumulative data at each look.  This matches
    the standard GS simulation approach (Brownian motion increments are correlated).

    Parameters
    ----------
    n1, n2
        Per-arm sample sizes at the *final* look.
    boundaries
        Upper boundary values (one per look).  Lower = -upper for two-sided.
    info_frac
        Information fractions (length = n_looks, last = 1.0).
    draw_fn
        Callable ``(rng, n1, n2) -> (array1, array2)``.  Draws the full
        trial sample under H1.
    stat_fn
        Callable ``(sub1, sub2) -> float``.  Computes the z/t statistic
        from the sub-arrays at a given look.
    sides
        1 or 2.
    n_sims
        Number of simulated trials.
    rng
        NumPy random Generator.
    """
    n_looks = len(boundaries)
    exits = 0
    for _ in range(n_sims):
        data1, data2 = draw_fn(rng, n1, n2)
        stopped = False
        for k in range(n_looks):
            n1k = max(2, math.ceil(info_frac[k] * n1))
            n2k = max(2, math.ceil(info_frac[k] * n2))
            z = stat_fn(data1[:n1k], data2[:n2k])
            bk = boundaries[k]
            if sides == 2:
                if abs(z) >= bk:
                    stopped = True
                    break
            else:
                if z >= bk:
                    stopped = True
                    break
        if stopped:
            exits += 1
    return exits / n_sims


# ---------------------------------------------------------------------------
# 1. Group-Sequential Tests for Two Means (Simulation)
# ---------------------------------------------------------------------------

def group_sequential_two_means_simulation(
    *,
    mean1: float,
    mean2: float,
    sd1: float,
    sd2: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n_looks: int = 5,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-sequential power for two-means comparison (simulation).

    (Simulation)" by Monte-Carlo simulation of the two-sample t-statistic
    at each look.  Supports ``obrien-fleming`` and ``pocock`` spending
    functions.

    Parameters
    ----------
    mean1, mean2
        Population means under H1.
    sd1
        Standard deviation of group 1.
    sd2
        Standard deviation of group 2 (defaults to ``sd1``).
    alpha
        Overall type-I error rate.
    power
        Not used for simulation (always solves for power given n1).
        Provided for API consistency; raises if both ``power`` and ``n1``
        are None.
    n1
        Per-group sample size at the final look (group 1).
    n_looks
        Number of looks.  Default 5.
    boundary
        Spending function: ``obrien-fleming`` (default) or ``pocock``.
    info_frac
        Custom information fractions.  If None, equally spaced.
    sides
        1 or 2 (default 2).
    allocation
        N2/N1 ratio.  Default 1.0.
    n_sims
        Number of Monte-Carlo trials.  Default 10 000.
    seed
        Random seed.  Default 42.
    """
    if n1 is None:
        raise ValueError("n1 (final per-group sample size) must be supplied")
    if sd2 is None:
        sd2 = sd1
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(f"unknown boundary {boundary!r}")

    n2 = max(2, math.ceil(allocation * n1))

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)

    boundaries = _solve_boundaries(info_frac, alpha, boundary_key, sides)
    rng = np.random.default_rng(seed)

    def _draw(rng_: np.random.Generator, nn1: int, nn2: int):
        return rng_.normal(mean1, sd1, nn1), rng_.normal(mean2, sd2, nn2)

    def _stat(sub1: np.ndarray, sub2: np.ndarray) -> float:
        n1k, n2k = len(sub1), len(sub2)
        s1 = sub1.std(ddof=1)
        s2 = sub2.std(ddof=1)
        se = math.sqrt(s1 * s1 / n1k + s2 * s2 / n2k)
        if se <= 0:
            return 0.0
        return (sub1.mean() - sub2.mean()) / se

    achieved = _gs_sim_power(
        n1=n1, n2=n2, boundaries=boundaries,
        info_frac=info_frac, draw_fn=_draw, stat_fn=_stat, sides=sides,
        n_sims=n_sims, rng=rng,
    )

    return {
        "method_id": "group_sequential_two_means_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "boundary": boundary_key,
        "n_looks": n_looks,
        "n_sims": n_sims,
        "inputs_echo": {
            "mean1": mean1, "mean2": mean2, "sd1": sd1, "sd2": sd2,
            "alpha": alpha, "n1": n1, "n_looks": n_looks,
            "boundary": boundary, "sides": sides, "allocation": allocation,
            "n_sims": n_sims, "seed": seed,
        },
        "citations": [
            "(Simulation)",
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods "
            "with Applications to Clinical Trials. Chapman & Hall.",
        ],
    }


# ---------------------------------------------------------------------------
# 2. Group-Sequential Tests for Two Proportions (Simulation)
# ---------------------------------------------------------------------------

def group_sequential_two_proportions_simulation(
    *,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    n1: int | None = None,
    n_looks: int = 5,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-sequential power for two-proportion comparison (simulation).

    (Simulation)" using the unpooled z-test for proportions.

    Parameters
    ----------
    p1, p2
        Proportions under H1.
    alpha
        Overall type-I error rate.
    n1
        Per-group sample size at the final look.
    n_looks
        Number of looks.  Default 5.
    boundary
        Spending function.
    info_frac
        Custom information fractions.
    sides
        1 or 2 (default 2).
    allocation
        N2/N1 ratio.
    n_sims
        Monte-Carlo replications.
    seed
        Random seed.
    """
    if n1 is None:
        raise ValueError("n1 (final per-group sample size) must be supplied")
    if not (0.0 < p1 < 1.0):
        raise ValueError("p1 must be in (0, 1)")
    if not (0.0 < p2 < 1.0):
        raise ValueError("p2 must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(f"unknown boundary {boundary!r}")

    n2 = max(2, math.ceil(allocation * n1))

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)

    boundaries = _solve_boundaries(info_frac, alpha, boundary_key, sides)
    rng = np.random.default_rng(seed)

    def _draw(rng_: np.random.Generator, nn1: int, nn2: int):
        return rng_.binomial(1, p1, nn1), rng_.binomial(1, p2, nn2)

    def _stat(sub1: np.ndarray, sub2: np.ndarray) -> float:
        n1k, n2k = len(sub1), len(sub2)
        ph1 = sub1.mean()
        ph2 = sub2.mean()
        se = math.sqrt(
            ph1 * (1 - ph1) / n1k + ph2 * (1 - ph2) / n2k + 1e-15
        )
        return (ph1 - ph2) / se

    achieved = _gs_sim_power(
        n1=n1, n2=n2, boundaries=boundaries,
        info_frac=info_frac, draw_fn=_draw, stat_fn=_stat, sides=sides,
        n_sims=n_sims, rng=rng,
    )

    return {
        "method_id": "group_sequential_two_proportions_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "boundary": boundary_key,
        "n_looks": n_looks,
        "n_sims": n_sims,
        "inputs_echo": {
            "p1": p1, "p2": p2, "alpha": alpha, "n1": n1,
            "n_looks": n_looks, "boundary": boundary, "sides": sides,
            "allocation": allocation, "n_sims": n_sims, "seed": seed,
        },
        "citations": [
            "(Simulation)",
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods "
            "with Applications to Clinical Trials. Chapman & Hall.",
        ],
    }


# ---------------------------------------------------------------------------
# 3. Group-Sequential Logrank Tests (Simulation)
# ---------------------------------------------------------------------------

def _exponential_survival_time(rng: np.random.Generator,
                                hazard: float, n: int) -> np.ndarray:
    """Generate exponential survival times (no censoring)."""
    return rng.exponential(1.0 / hazard, n)


def _logrank_stat(t1: np.ndarray, t2: np.ndarray) -> float:
    """Logrank z-statistic for two groups (no censoring assumed here)."""
    n1, n2 = len(t1), len(t2)
    # Pool and sort event times
    all_times = np.concatenate([t1, t2])
    # Use the Wilcoxon-style logrank via score approach
    # Score for each group 1 obs: expected - observed rank-contribution
    # Simplified logrank for complete (uncensored) data:
    #   O1, E1 = observed and expected events in group 1
    # At each unique time t_j:
    #   n1j = subjects at risk in group 1, n2j in group 2, nj = n1j+n2j
    #   dj  = events (1 each, since continuous); d1j = 1 if from group 1
    #   E1j = d1j_contribution: n1j/nj
    # logrank statistic:
    #   (O1 - E1) / sqrt(Var)  where Var = sum( (n1j*n2j*dj*(nj-dj)) / (nj^2*(nj-1)) )
    # For continuous data dj=1 always, so simplifies.
    order = np.argsort(all_times)
    group = np.concatenate([np.ones(n1, dtype=int),
                            np.zeros(n2, dtype=int)])[order]
    # At each event time one event happens
    nt = n1 + n2
    O1 = group.sum()   # total events from group 1
    E1 = 0.0
    V = 0.0
    n_at_risk_1 = n1
    n_at_risk_2 = n2
    for i in range(nt):
        n_at_risk = n_at_risk_1 + n_at_risk_2
        if n_at_risk == 0:
            break
        g = group[i]
        e1 = n_at_risk_1 / n_at_risk
        E1 += e1
        if n_at_risk > 1:
            V += (n_at_risk_1 * n_at_risk_2) / (n_at_risk * n_at_risk)
        # Update at-risk counts
        if g == 1:
            n_at_risk_1 -= 1
        else:
            n_at_risk_2 -= 1
    if V <= 0:
        return 0.0
    return (O1 - E1) / math.sqrt(V)


def group_sequential_logrank_simulation(
    *,
    hazard1: float,
    hazard2: float,
    alpha: float = 0.05,
    n1: int | None = None,
    n_looks: int = 5,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-sequential power for logrank test (simulation).

    via Monte-Carlo simulation of exponential survival times and the logrank
    statistic at each look.

    Parameters
    ----------
    hazard1
        Hazard rate of group 1 (control).
    hazard2
        Hazard rate of group 2 (treatment).
    alpha
        Overall type-I error rate.
    n1
        Per-group sample size at the final look (group 1).
    n_looks
        Number of looks.  Default 5.
    boundary
        Spending function: ``obrien-fleming`` (default) or ``pocock``.
    info_frac
        Custom information fractions.
    sides
        1 or 2 (default 2).
    allocation
        N2/N1 ratio.
    n_sims
        Monte-Carlo replications.
    seed
        Random seed.
    """
    if n1 is None:
        raise ValueError("n1 (final per-group sample size) must be supplied")
    if hazard1 <= 0 or hazard2 <= 0:
        raise ValueError("hazard rates must be positive")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(f"unknown boundary {boundary!r}")

    n2 = max(2, math.ceil(allocation * n1))

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)

    boundaries = _solve_boundaries(info_frac, alpha, boundary_key, sides)
    rng = np.random.default_rng(seed)

    def _draw(rng_: np.random.Generator, nn1: int, nn2: int):
        return (
            _exponential_survival_time(rng_, hazard1, nn1),
            _exponential_survival_time(rng_, hazard2, nn2),
        )

    def _stat(sub1: np.ndarray, sub2: np.ndarray) -> float:
        return _logrank_stat(sub1, sub2)

    achieved = _gs_sim_power(
        n1=n1, n2=n2, boundaries=boundaries,
        info_frac=info_frac, draw_fn=_draw, stat_fn=_stat, sides=sides,
        n_sims=n_sims, rng=rng,
    )

    return {
        "method_id": "group_sequential_logrank_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "hazard_ratio": hazard2 / hazard1,
        "boundary": boundary_key,
        "n_looks": n_looks,
        "n_sims": n_sims,
        "inputs_echo": {
            "hazard1": hazard1, "hazard2": hazard2, "alpha": alpha, "n1": n1,
            "n_looks": n_looks, "boundary": boundary, "sides": sides,
            "allocation": allocation, "n_sims": n_sims, "seed": seed,
        },
        "citations": [
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods "
            "with Applications to Clinical Trials. Chapman & Hall.",
            "Klein, J.P. & Moeschberger, M.L. (1997). Survival Analysis. "
            "Springer-Verlag.",
        ],
    }


# ---------------------------------------------------------------------------
# 4. Group-Sequential NI Tests for Two Means (Simulation)
# ---------------------------------------------------------------------------

def group_sequential_ni_two_means_simulation(
    *,
    mean1: float,
    mean2: float,
    sd1: float,
    sd2: float | None = None,
    margin: float,
    alpha: float = 0.05,
    n1: int | None = None,
    n_looks: int = 5,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    higher_means_better: bool = True,
    allocation: float = 1.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-sequential NI power for two means (simulation).

    Two Means (Simulation)".  The one-sided NI t-statistic at each look is:

        z_k = (mean1_k - mean2_k - NIM_signed) / SE_k

    where ``NIM_signed = -margin`` when ``higher_means_better=True``
    (H1: mean1 - mean2 > -|NIM|).

    Parameters
    ----------
    mean1
        Population mean of group 1 (treatment) under H1.
    mean2
        Population mean of group 2 (reference) under H1.
    sd1
        Standard deviation of group 1.
    sd2
        Standard deviation of group 2 (defaults to ``sd1``).
    margin
        Non-inferiority margin magnitude (positive).
    alpha
        One-sided type-I error rate (e.g. 0.05).
    n1
        Per-group sample size at the final look.
    n_looks
        Number of looks.
    boundary
        Spending function.
    info_frac
        Custom information fractions.
    higher_means_better
        If True, H1: mean1 - mean2 > -|margin|.
    allocation
        N2/N1 ratio.
    n_sims
        Monte-Carlo replications.
    seed
        Random seed.
    """
    if n1 is None:
        raise ValueError("n1 must be supplied")
    if margin <= 0:
        raise ValueError("margin must be positive")
    if sd2 is None:
        sd2 = sd1
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(f"unknown boundary {boundary!r}")

    sides = 1  # NI tests are one-sided
    n2 = max(2, math.ceil(allocation * n1))
    nim_signed = -margin if higher_means_better else margin

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)

    boundaries = _solve_boundaries_1sided(info_frac, alpha, boundary_key)
    rng = np.random.default_rng(seed)

    def _draw(rng_: np.random.Generator, nn1: int, nn2: int):
        return rng_.normal(mean1, sd1, nn1), rng_.normal(mean2, sd2, nn2)

    def _stat(sub1: np.ndarray, sub2: np.ndarray) -> float:
        n1k, n2k = len(sub1), len(sub2)
        diff = sub1.mean() - sub2.mean()
        s1 = sub1.std(ddof=1)
        s2 = sub2.std(ddof=1)
        se = math.sqrt(s1 * s1 / n1k + s2 * s2 / n2k)
        if se <= 0:
            return 0.0
        # For NI "higher better": reject H0 when (diff - nim_signed)/se >= bk
        return (diff - nim_signed) / se

    achieved = _gs_sim_power(
        n1=n1, n2=n2, boundaries=boundaries,
        info_frac=info_frac, draw_fn=_draw, stat_fn=_stat, sides=1,
        n_sims=n_sims, rng=rng,
    )

    return {
        "method_id": "group_sequential_ni_two_means_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "boundary": boundary_key,
        "n_looks": n_looks,
        "n_sims": n_sims,
        "inputs_echo": {
            "mean1": mean1, "mean2": mean2, "sd1": sd1, "sd2": sd2,
            "margin": margin, "alpha": alpha, "n1": n1,
            "n_looks": n_looks, "boundary": boundary,
            "higher_means_better": higher_means_better,
            "allocation": allocation, "n_sims": n_sims, "seed": seed,
        },
        "citations": [
            "Two Means (Simulation)",
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods "
            "with Applications to Clinical Trials. Chapman & Hall.",
        ],
    }


# ---------------------------------------------------------------------------
# 5. Group-Sequential NI Tests for Difference of Two Proportions (Simulation)
# ---------------------------------------------------------------------------

def group_sequential_ni_two_proportions_difference_simulation(
    *,
    p1: float,
    p2: float,
    margin: float,
    alpha: float = 0.05,
    n1: int | None = None,
    n_looks: int = 5,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    higher_proportions_better: bool = True,
    allocation: float = 1.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-sequential NI power for two proportions — difference (simulation).

    the Difference of Two Proportions (Simulation)".  The unpooled z-statistic
    at each look is:

        z_k = (ph1_k - ph2_k - D0) / SE_k

    where ``D0 = -margin`` when ``higher_proportions_better=True``
    (H1: p1 - p2 > -|margin|).

    Parameters
    ----------
    p1
        Proportion of group 1 (treatment) under H1.
    p2
        Proportion of group 2 (reference) under H1.
    margin
        Non-inferiority margin magnitude on the difference scale (positive).
    alpha
        One-sided type-I error rate.
    n1
        Per-group sample size at the final look.
    n_looks
        Number of looks.
    boundary
        Spending function.
    info_frac
        Custom information fractions.
    higher_proportions_better
        If True, H1: p1 - p2 > -|margin|.
    allocation
        N2/N1 ratio.
    n_sims
        Monte-Carlo replications.
    seed
        Random seed.
    """
    if n1 is None:
        raise ValueError("n1 must be supplied")
    if not (0.0 < p1 < 1.0):
        raise ValueError("p1 must be in (0, 1)")
    if not (0.0 < p2 < 1.0):
        raise ValueError("p2 must be in (0, 1)")
    if margin <= 0:
        raise ValueError("margin must be positive")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(f"unknown boundary {boundary!r}")

    sides = 1
    n2 = max(2, math.ceil(allocation * n1))
    d0 = -margin if higher_proportions_better else margin

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)

    boundaries = _solve_boundaries_1sided(info_frac, alpha, boundary_key)
    rng = np.random.default_rng(seed)

    def _draw(rng_: np.random.Generator, nn1: int, nn2: int):
        return rng_.binomial(1, p1, nn1), rng_.binomial(1, p2, nn2)

    def _stat(sub1: np.ndarray, sub2: np.ndarray) -> float:
        n1k, n2k = len(sub1), len(sub2)
        ph1 = sub1.mean()
        ph2 = sub2.mean()
        se = math.sqrt(
            ph1 * (1 - ph1) / n1k + ph2 * (1 - ph2) / n2k + 1e-15
        )
        return (ph1 - ph2 - d0) / se

    achieved = _gs_sim_power(
        n1=n1, n2=n2, boundaries=boundaries,
        info_frac=info_frac, draw_fn=_draw, stat_fn=_stat, sides=1,
        n_sims=n_sims, rng=rng,
    )

    return {
        "method_id": "group_sequential_ni_two_proportions_difference_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "boundary": boundary_key,
        "n_looks": n_looks,
        "n_sims": n_sims,
        "inputs_echo": {
            "p1": p1, "p2": p2, "margin": margin, "alpha": alpha, "n1": n1,
            "n_looks": n_looks, "boundary": boundary,
            "higher_proportions_better": higher_proportions_better,
            "allocation": allocation, "n_sims": n_sims, "seed": seed,
        },
        "citations": [
            "the Difference of Two Proportions (Simulation)",
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods "
            "with Applications to Clinical Trials. Chapman & Hall.",
        ],
    }


# ---------------------------------------------------------------------------
# 6. Group-Sequential NI Tests for the Ratio of Two Proportions (Simulation)
# ---------------------------------------------------------------------------

def group_sequential_ni_two_proportions_ratio_simulation(
    *,
    p2: float,
    r0: float,
    r1: float,
    alpha: float = 0.05,
    n1: int | None = None,
    n_looks: int = 5,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    higher_proportions_better: bool = True,
    allocation: float = 1.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-sequential NI power for two proportions — ratio (simulation).

    the Ratio of Two Proportions (Simulation)".

    The NI null hypothesis is H0: p1/p2 <= R0 vs H1: p1/p2 > R0 (when
    higher proportions are better).  The test statistic at each look uses
    the unpooled z-test for the log-ratio.  Under H1 the true ratio is R1,
    giving p1 = R1 * p2.

    Parameters
    ----------
    p2
        Proportion of the reference (group 2) under H1.
    r0
        Non-inferiority ratio R0 = P1.0 / P2 (H0 boundary).  Should be < 1
        when higher proportions are better.
    r1
        Actual ratio R1 = P1.1 / P2 under H1 (e.g. 1.0 for equal proportions).
    alpha
        One-sided type-I error rate.
    n1
        Per-group sample size at the final look.
    n_looks
        Number of looks.
    boundary
        Spending function: ``obrien-fleming`` (default) or ``pocock``.
    info_frac
        Custom information fractions.
    higher_proportions_better
        If True, H1: p1/p2 > R0 (reject when z is large).
    allocation
        N2/N1 ratio.
    n_sims
        Monte-Carlo replications.
    seed
        Random seed.
    """
    if n1 is None:
        raise ValueError("n1 must be supplied")
    if not (0.0 < p2 < 1.0):
        raise ValueError("p2 must be in (0, 1)")
    if r0 <= 0:
        raise ValueError("r0 must be positive")
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(f"unknown boundary {boundary!r}")

    # Derive group-1 proportions
    p1_h1 = r1 * p2
    p1_h0 = r0 * p2
    if not (0.0 < p1_h1 < 1.0):
        raise ValueError(f"r1*p2={p1_h1:.4f} must be in (0, 1)")
    if not (0.0 < p1_h0 < 1.0):
        raise ValueError(f"r0*p2={p1_h0:.4f} must be in (0, 1)")

    n2 = max(2, math.ceil(allocation * n1))

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)

    boundaries = _solve_boundaries_1sided(info_frac, alpha, boundary_key)
    rng = np.random.default_rng(seed)

    def _draw(rng_: np.random.Generator, nn1: int, nn2: int):
        return rng_.binomial(1, p1_h1, nn1), rng_.binomial(1, p2, nn2)

    def _stat(sub1: np.ndarray, sub2: np.ndarray) -> float:
        n1k, n2k = len(sub1), len(sub2)
        ph1 = sub1.mean()
        ph2 = sub2.mean()
        # z-stat for H0: ratio = R0; using log-ratio score approximation
        # Standard NI ratio z = (ph1/ph2 - R0) / SE_ratio
        # SE via delta method: sqrt(ph1*(1-ph1)/n1 + (R0^2)*ph2*(1-ph2)/n2) / ph2
        # but simpler: use difference z with transformed null
        # z = (log(ph1/ph2) - log(R0)) / sqrt((1-ph1)/(n1k*ph1) + (1-ph2)/(n2k*ph2))
        if ph1 <= 0 or ph2 <= 0:
            return 0.0
        se = math.sqrt(
            (1 - ph1) / (n1k * ph1 + 1e-15)
            + (1 - ph2) / (n2k * ph2 + 1e-15)
        )
        if se <= 0:
            return 0.0
        if higher_proportions_better:
            return (math.log(ph1 / ph2) - math.log(r0)) / se
        else:
            return (math.log(r0) - math.log(ph1 / ph2)) / se

    achieved = _gs_sim_power(
        n1=n1, n2=n2, boundaries=boundaries,
        info_frac=info_frac, draw_fn=_draw, stat_fn=_stat, sides=1,
        n_sims=n_sims, rng=rng,
    )

    return {
        "method_id": "group_sequential_ni_two_proportions_ratio_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "boundary": boundary_key,
        "n_looks": n_looks,
        "n_sims": n_sims,
        "inputs_echo": {
            "p2": p2, "r0": r0, "r1": r1, "alpha": alpha, "n1": n1,
            "n_looks": n_looks, "boundary": boundary,
            "higher_proportions_better": higher_proportions_better,
            "allocation": allocation, "n_sims": n_sims, "seed": seed,
        },
        "citations": [
            "the Ratio of Two Proportions (Simulation)",
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods "
            "with Applications to Clinical Trials. Chapman & Hall.",
        ],
    }


# ---------------------------------------------------------------------------
# 7. Group-Sequential NI Tests for Odds Ratio of Two Proportions (Simulation)
# ---------------------------------------------------------------------------

def group_sequential_ni_two_proportions_odds_ratio_simulation(
    *,
    p2: float,
    or0: float,
    or1: float,
    alpha: float = 0.05,
    n1: int | None = None,
    n_looks: int = 5,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    higher_proportions_better: bool = True,
    allocation: float = 1.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-sequential NI power for two proportions — odds ratio (simulation).

    the Odds Ratio of Two Proportions (Simulation)".

    The NI null hypothesis H0: OR(p1/p2) <= OR0 vs H1: OR(p1/p2) > OR0
    (when higher proportions are better).  P1 under H1 is derived from OR1
    and P2; P1 under H0 from OR0 and P2.

    Parameters
    ----------
    p2
        Proportion of the reference group under H1.
    or0
        Non-inferiority odds ratio (H0 boundary).  < 1 when higher better.
    or1
        Actual odds ratio under H1 (often 1.0 for equal proportions).
    alpha
        One-sided type-I error rate.
    n1
        Per-group sample size at the final look.
    n_looks
        Number of looks.
    boundary
        Spending function.
    info_frac
        Custom information fractions.
    higher_proportions_better
        If True, H1: OR(p1, p2) > OR0.
    allocation
        N2/N1 ratio.
    n_sims
        Monte-Carlo replications.
    seed
        Random seed.
    """
    if n1 is None:
        raise ValueError("n1 must be supplied")
    if not (0.0 < p2 < 1.0):
        raise ValueError("p2 must be in (0, 1)")
    if or0 <= 0:
        raise ValueError("or0 must be positive")
    if or1 <= 0:
        raise ValueError("or1 must be positive")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(f"unknown boundary {boundary!r}")

    # Derive p1 from OR and p2: OR = (p1/(1-p1)) / (p2/(1-p2))
    # => p1 = OR * p2/(1-p2) / (1 + OR * p2/(1-p2))
    odds2 = p2 / (1.0 - p2)
    p1_h1 = (or1 * odds2) / (1.0 + or1 * odds2)
    p1_h0 = (or0 * odds2) / (1.0 + or0 * odds2)
    if not (0.0 < p1_h1 < 1.0):
        raise ValueError(f"Derived p1 under H1 ({p1_h1:.4f}) out of (0,1)")
    if not (0.0 < p1_h0 < 1.0):
        raise ValueError(f"Derived p1 under H0 ({p1_h0:.4f}) out of (0,1)")

    n2 = max(2, math.ceil(allocation * n1))

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)

    boundaries = _solve_boundaries_1sided(info_frac, alpha, boundary_key)
    rng = np.random.default_rng(seed)

    def _draw(rng_: np.random.Generator, nn1: int, nn2: int):
        return rng_.binomial(1, p1_h1, nn1), rng_.binomial(1, p2, nn2)

    def _stat(sub1: np.ndarray, sub2: np.ndarray) -> float:
        n1k, n2k = len(sub1), len(sub2)
        ph1 = sub1.mean()
        ph2 = sub2.mean()
        if ph1 <= 0 or ph2 <= 0 or ph1 >= 1 or ph2 >= 1:
            return 0.0
        log_or_hat = math.log(ph1 / (1 - ph1)) - math.log(ph2 / (1 - ph2))
        se = math.sqrt(
            1.0 / (n1k * ph1 * (1 - ph1) + 1e-15)
            + 1.0 / (n2k * ph2 * (1 - ph2) + 1e-15)
        )
        if se <= 0:
            return 0.0
        if higher_proportions_better:
            return (log_or_hat - math.log(or0)) / se
        else:
            return (math.log(or0) - log_or_hat) / se

    achieved = _gs_sim_power(
        n1=n1, n2=n2, boundaries=boundaries,
        info_frac=info_frac, draw_fn=_draw, stat_fn=_stat, sides=1,
        n_sims=n_sims, rng=rng,
    )

    return {
        "method_id": "group_sequential_ni_two_proportions_odds_ratio_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "boundary": boundary_key,
        "n_looks": n_looks,
        "n_sims": n_sims,
        "inputs_echo": {
            "p2": p2, "or0": or0, "or1": or1, "alpha": alpha, "n1": n1,
            "n_looks": n_looks, "boundary": boundary,
            "higher_proportions_better": higher_proportions_better,
            "allocation": allocation, "n_sims": n_sims, "seed": seed,
        },
        "citations": [
            "the Odds Ratio of Two Proportions (Simulation)",
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods "
            "with Applications to Clinical Trials. Chapman & Hall.",
        ],
    }


# ---------------------------------------------------------------------------
# 8. Group-Sequential Superiority-by-Margin Tests for the Difference of Two
#    Proportions (Simulation)
# ---------------------------------------------------------------------------

def group_sequential_sup_two_proportions_difference_simulation(
    *,
    p2: float,
    d0: float,
    d1: float,
    alpha: float = 0.05,
    n1: int | None = None,
    n_looks: int = 5,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    higher_proportions_better: bool = True,
    allocation: float = 1.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-sequential superiority-by-margin for proportions — difference.

    for the Difference of Two Proportions (Simulation)".

    H0: p1 - p2 <= D0  vs  H1: p1 - p2 > D0 (when higher proportions better).
    P1 under H1 = P2 + D1.  The test statistic is the unpooled z for difference
    with the NI/superiority margin shifted: z = (ph1 - ph2 - D0) / SE.

    Parameters
    ----------
    p2
        Proportion of group 2 (reference) under H0 and H1.
    d0
        Superiority margin (D0 = P1.0 - P2 under H0).  Should be > 0 when
        higher proportions are better.
    d1
        Actual difference (D1 = P1.1 - P2) under H1.  Should be > D0.
    alpha
        One-sided type-I error rate.
    n1
        Per-group sample size at the final look.
    n_looks
        Number of looks.
    boundary
        Spending function.
    info_frac
        Custom information fractions.
    higher_proportions_better
        If True, H1: p1 - p2 > D0.
    allocation
        N2/N1 ratio.
    n_sims
        Monte-Carlo replications.
    seed
        Random seed.
    """
    if n1 is None:
        raise ValueError("n1 must be supplied")
    if not (0.0 < p2 < 1.0):
        raise ValueError("p2 must be in (0, 1)")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(f"unknown boundary {boundary!r}")

    p1_h1 = p2 + d1
    if not (0.0 < p1_h1 < 1.0):
        raise ValueError(f"p2+d1={p1_h1:.4f} must be in (0,1)")

    n2 = max(2, math.ceil(allocation * n1))

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)

    boundaries = _solve_boundaries_1sided(info_frac, alpha, boundary_key)
    rng = np.random.default_rng(seed)

    def _draw(rng_: np.random.Generator, nn1: int, nn2: int):
        return rng_.binomial(1, p1_h1, nn1), rng_.binomial(1, p2, nn2)

    def _stat(sub1: np.ndarray, sub2: np.ndarray) -> float:
        n1k, n2k = len(sub1), len(sub2)
        ph1 = sub1.mean()
        ph2 = sub2.mean()
        se = math.sqrt(
            ph1 * (1 - ph1) / n1k + ph2 * (1 - ph2) / n2k + 1e-15
        )
        if higher_proportions_better:
            return (ph1 - ph2 - d0) / se
        else:
            return (d0 - (ph1 - ph2)) / se

    achieved = _gs_sim_power(
        n1=n1, n2=n2, boundaries=boundaries,
        info_frac=info_frac, draw_fn=_draw, stat_fn=_stat, sides=1,
        n_sims=n_sims, rng=rng,
    )

    return {
        "method_id": "group_sequential_sup_two_proportions_difference_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "boundary": boundary_key,
        "n_looks": n_looks,
        "n_sims": n_sims,
        "inputs_echo": {
            "p2": p2, "d0": d0, "d1": d1, "alpha": alpha, "n1": n1,
            "n_looks": n_looks, "boundary": boundary,
            "higher_proportions_better": higher_proportions_better,
            "allocation": allocation, "n_sims": n_sims, "seed": seed,
        },
        "citations": [
            "for the Difference of Two Proportions (Simulation)",
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods "
            "with Applications to Clinical Trials. Chapman & Hall.",
        ],
    }


# ---------------------------------------------------------------------------
# 9. Group-Sequential Superiority-by-Margin Tests for Ratio of Two
#    Proportions (Simulation)
# ---------------------------------------------------------------------------

def group_sequential_sup_two_proportions_ratio_simulation(
    *,
    p2: float,
    r0: float,
    r1: float,
    alpha: float = 0.05,
    n1: int | None = None,
    n_looks: int = 5,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    higher_proportions_better: bool = True,
    allocation: float = 1.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-sequential superiority-by-margin for proportions — ratio.

    for the Ratio of Two Proportions (Simulation)".

    H0: p1/p2 <= R0  vs  H1: p1/p2 > R0 (when higher proportions are better
    and R0 > 1 for superiority).  P1 under H1 = R1 * P2.

    Parameters
    ----------
    p2
        Proportion of group 2 (reference).
    r0
        Superiority ratio R0 = P1.0/P2 under H0.  > 1 when higher better.
    r1
        Actual ratio R1 = P1.1/P2 under H1.  Should be > R0.
    alpha
        One-sided type-I error rate.
    n1
        Per-group sample size at the final look.
    n_looks
        Number of looks.
    boundary
        Spending function.
    info_frac
        Custom information fractions.
    higher_proportions_better
        If True, H1: p1/p2 > R0.
    allocation
        N2/N1 ratio.
    n_sims
        Monte-Carlo replications.
    seed
        Random seed.
    """
    if n1 is None:
        raise ValueError("n1 must be supplied")
    if not (0.0 < p2 < 1.0):
        raise ValueError("p2 must be in (0, 1)")
    if r0 <= 0:
        raise ValueError("r0 must be positive")
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(f"unknown boundary {boundary!r}")

    p1_h1 = r1 * p2
    if not (0.0 < p1_h1 < 1.0):
        raise ValueError(f"r1*p2={p1_h1:.4f} must be in (0, 1)")

    n2 = max(2, math.ceil(allocation * n1))

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)

    boundaries = _solve_boundaries_1sided(info_frac, alpha, boundary_key)
    rng = np.random.default_rng(seed)

    def _draw(rng_: np.random.Generator, nn1: int, nn2: int):
        return rng_.binomial(1, p1_h1, nn1), rng_.binomial(1, p2, nn2)

    def _stat(sub1: np.ndarray, sub2: np.ndarray) -> float:
        n1k, n2k = len(sub1), len(sub2)
        ph1 = sub1.mean()
        ph2 = sub2.mean()
        if ph1 <= 0 or ph2 <= 0:
            return 0.0
        se = math.sqrt(
            (1 - ph1) / (n1k * ph1 + 1e-15)
            + (1 - ph2) / (n2k * ph2 + 1e-15)
        )
        if se <= 0:
            return 0.0
        if higher_proportions_better:
            return (math.log(ph1 / ph2) - math.log(r0)) / se
        else:
            return (math.log(r0) - math.log(ph1 / ph2)) / se

    achieved = _gs_sim_power(
        n1=n1, n2=n2, boundaries=boundaries,
        info_frac=info_frac, draw_fn=_draw, stat_fn=_stat, sides=1,
        n_sims=n_sims, rng=rng,
    )

    return {
        "method_id": "group_sequential_sup_two_proportions_ratio_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "boundary": boundary_key,
        "n_looks": n_looks,
        "n_sims": n_sims,
        "inputs_echo": {
            "p2": p2, "r0": r0, "r1": r1, "alpha": alpha, "n1": n1,
            "n_looks": n_looks, "boundary": boundary,
            "higher_proportions_better": higher_proportions_better,
            "allocation": allocation, "n_sims": n_sims, "seed": seed,
        },
        "citations": [
            "for the Ratio of Two Proportions (Simulation)",
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods "
            "with Applications to Clinical Trials. Chapman & Hall.",
        ],
    }


# ---------------------------------------------------------------------------
# 10. Group-Sequential Superiority-by-Margin Tests for Odds Ratio of Two
#     Proportions (Simulation)
# ---------------------------------------------------------------------------

def group_sequential_sup_two_proportions_odds_ratio_simulation(
    *,
    p2: float,
    or0: float,
    or1: float,
    alpha: float = 0.05,
    n1: int | None = None,
    n_looks: int = 5,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    higher_proportions_better: bool = True,
    allocation: float = 1.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-sequential superiority-by-margin for proportions — odds ratio.

    for the Odds Ratio of Two Proportions (Simulation)".

    H0: OR(p1,p2) <= OR0  vs  H1: OR(p1,p2) > OR0 (higher better, OR0 > 1).
    P1 under H1 is derived from OR1 and P2.

    Parameters
    ----------
    p2
        Proportion of group 2 (reference).
    or0
        Superiority odds ratio under H0.  > 1 when higher proportions better.
    or1
        Actual odds ratio under H1.  Should be > OR0.
    alpha
        One-sided type-I error rate.
    n1
        Per-group sample size at the final look.
    n_looks
        Number of looks.
    boundary
        Spending function.
    info_frac
        Custom information fractions.
    higher_proportions_better
        If True, H1: OR(p1,p2) > OR0.
    allocation
        N2/N1 ratio.
    n_sims
        Monte-Carlo replications.
    seed
        Random seed.
    """
    if n1 is None:
        raise ValueError("n1 must be supplied")
    if not (0.0 < p2 < 1.0):
        raise ValueError("p2 must be in (0, 1)")
    if or0 <= 0:
        raise ValueError("or0 must be positive")
    if or1 <= 0:
        raise ValueError("or1 must be positive")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(f"unknown boundary {boundary!r}")

    odds2 = p2 / (1.0 - p2)
    p1_h1 = (or1 * odds2) / (1.0 + or1 * odds2)
    if not (0.0 < p1_h1 < 1.0):
        raise ValueError(f"Derived p1 under H1 ({p1_h1:.4f}) out of (0,1)")

    n2 = max(2, math.ceil(allocation * n1))

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)

    boundaries = _solve_boundaries_1sided(info_frac, alpha, boundary_key)
    rng = np.random.default_rng(seed)

    def _draw(rng_: np.random.Generator, nn1: int, nn2: int):
        return rng_.binomial(1, p1_h1, nn1), rng_.binomial(1, p2, nn2)

    def _stat(sub1: np.ndarray, sub2: np.ndarray) -> float:
        n1k, n2k = len(sub1), len(sub2)
        ph1 = sub1.mean()
        ph2 = sub2.mean()
        if ph1 <= 0 or ph2 <= 0 or ph1 >= 1 or ph2 >= 1:
            return 0.0
        log_or_hat = math.log(ph1 / (1 - ph1)) - math.log(ph2 / (1 - ph2))
        se = math.sqrt(
            1.0 / (n1k * ph1 * (1 - ph1) + 1e-15)
            + 1.0 / (n2k * ph2 * (1 - ph2) + 1e-15)
        )
        if se <= 0:
            return 0.0
        if higher_proportions_better:
            return (log_or_hat - math.log(or0)) / se
        else:
            return (math.log(or0) - log_or_hat) / se

    achieved = _gs_sim_power(
        n1=n1, n2=n2, boundaries=boundaries,
        info_frac=info_frac, draw_fn=_draw, stat_fn=_stat, sides=1,
        n_sims=n_sims, rng=rng,
    )

    return {
        "method_id": "group_sequential_sup_two_proportions_odds_ratio_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "boundary": boundary_key,
        "n_looks": n_looks,
        "n_sims": n_sims,
        "inputs_echo": {
            "p2": p2, "or0": or0, "or1": or1, "alpha": alpha, "n1": n1,
            "n_looks": n_looks, "boundary": boundary,
            "higher_proportions_better": higher_proportions_better,
            "allocation": allocation, "n_sims": n_sims, "seed": seed,
        },
        "citations": [
            "for the Odds Ratio of Two Proportions (Simulation)",
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods "
            "with Applications to Clinical Trials. Chapman & Hall.",
        ],
    }


# ---------------------------------------------------------------------------
# 11. Group-Sequential Tests for Two Means Assuming Normality (Simulation)
# ---------------------------------------------------------------------------

def group_sequential_two_means_normality_simulation(
    *,
    mean1: float,
    mean2: float,
    sd: float,
    alpha: float = 0.05,
    n1: int | None = None,
    n_looks: int = 5,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-sequential power for two means assuming normality (simulation).

    Normality (Simulation)".  Unlike the general two-means simulation which
    uses the Welch t-statistic with separate SDs, this chapter assumes equal
    variances and uses a pooled-SD t-statistic (Lakatos-style normality
    assumption).

    Parameters
    ----------
    mean1
        Population mean of group 1 (control) under H1.
    mean2
        Population mean of group 2 (treatment) under H1.
    sd
        Common standard deviation for both groups.
    alpha
        Overall type-I error rate.
    n1
        Per-group sample size at the final look.
    n_looks
        Number of looks.
    boundary
        Spending function: ``obrien-fleming`` (default) or ``pocock``.
    info_frac
        Custom information fractions.
    sides
        1 or 2 (default 2).
    allocation
        N2/N1 ratio.
    n_sims
        Monte-Carlo replications.
    seed
        Random seed.
    """
    if n1 is None:
        raise ValueError("n1 must be supplied")
    if sd <= 0:
        raise ValueError("sd must be positive")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(f"unknown boundary {boundary!r}")

    n2 = max(2, math.ceil(allocation * n1))

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)

    boundaries = _solve_boundaries(info_frac, alpha, boundary_key, sides)
    rng = np.random.default_rng(seed)

    def _draw(rng_: np.random.Generator, nn1: int, nn2: int):
        return rng_.normal(mean1, sd, nn1), rng_.normal(mean2, sd, nn2)

    def _stat(sub1: np.ndarray, sub2: np.ndarray) -> float:
        n1k, n2k = len(sub1), len(sub2)
        # Pooled-variance t-statistic (equal-variance assumption)
        s1 = sub1.std(ddof=1)
        s2 = sub2.std(ddof=1)
        sp2 = ((n1k - 1) * s1 * s1 + (n2k - 1) * s2 * s2) / (n1k + n2k - 2)
        sp = math.sqrt(sp2)
        se = sp * math.sqrt(1.0 / n1k + 1.0 / n2k)
        if se <= 0:
            return 0.0
        return (sub1.mean() - sub2.mean()) / se

    achieved = _gs_sim_power(
        n1=n1, n2=n2, boundaries=boundaries,
        info_frac=info_frac, draw_fn=_draw, stat_fn=_stat, sides=sides,
        n_sims=n_sims, rng=rng,
    )

    return {
        "method_id": "group_sequential_two_means_normality_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "boundary": boundary_key,
        "n_looks": n_looks,
        "n_sims": n_sims,
        "inputs_echo": {
            "mean1": mean1, "mean2": mean2, "sd": sd,
            "alpha": alpha, "n1": n1, "n_looks": n_looks,
            "boundary": boundary, "sides": sides,
            "allocation": allocation, "n_sims": n_sims, "seed": seed,
        },
        "citations": [
            "Normality (Simulation)",
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods "
            "with Applications to Clinical Trials. Chapman & Hall.",
            "Lakatos, E. (1988). Sample sizes based on the log-rank statistic "
            "in complex clinical trials. Biometrics 44.",
        ],
    }
