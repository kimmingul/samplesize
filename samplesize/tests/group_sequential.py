"""Group-sequential designs for comparing two means.


The implementation follows the Lan & DeMets (1983) alpha-spending
formulation with O'Brien-Fleming / Pocock / alpha-spending boundaries.  Supported spending
functions:

* ``obrien-fleming``  two-sided: alpha(tau) = 4 * (1 - Phi(z_{alpha/4}/sqrt(tau)))
                      one-sided: alpha(tau) = 2 * (1 - Phi(z_{alpha/2}/sqrt(tau)))
* ``pocock``           alpha(tau) = alpha * ln(1 + (e - 1) * tau)

Under H0 the sequence of standardized test statistics ``Z_k`` follows a
multivariate normal with ``Cov(Z_j, Z_k) = sqrt(tau_j / tau_k)`` for
``j <= k``.  Under H1 with drift ``theta`` the means are
``E[Z_k] = theta * sqrt(tau_k)``.

Rectangle (continuation) probabilities are evaluated via SciPy's
``multivariate_normal.cdf`` with 2^K inclusion-exclusion over the two
sides at each look.  For ``n_looks <= 10`` this is fast and produces six
correct decimal digits.

Limitations
-----------
* Only the two spending functions above are supported.  Hwang-Shih-DeCani
  and user-supplied boundaries are not implemented yet.
* Two-sided symmetric boundaries are assumed for ``sides=2``; the implementation does
  support asymmetric boundaries but this module follows the simpler
  symmetric case.
* Information times are equally spaced unless an explicit ``info_frac``
  vector is supplied.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import multivariate_normal, norm

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Spending functions
# ---------------------------------------------------------------------------
def _alpha_spent_obf(tau: float, alpha: float, sides: int) -> float:
    """O'Brien-Fleming Lan-DeMets spending function."""
    if tau <= 0.0:
        return 0.0
    if tau >= 1.0:
        return alpha
    if sides == 2:
        z = D.norm_ppf(1.0 - alpha / 4.0)
        return 4.0 * (1.0 - _phi(z / math.sqrt(tau)))
    z = D.norm_ppf(1.0 - alpha)
    return 2.0 * (1.0 - _phi(z / math.sqrt(tau)))


def _alpha_spent_pocock(tau: float, alpha: float, sides: int) -> float:
    """Pocock-type Lan-DeMets spending function."""
    if tau <= 0.0:
        return 0.0
    if tau >= 1.0:
        return alpha
    return alpha * math.log(1.0 + (math.e - 1.0) * tau)


_SPENDING = {
    "obrien-fleming": _alpha_spent_obf,
    "obf": _alpha_spent_obf,
    "pocock": _alpha_spent_pocock,
}


# ---------------------------------------------------------------------------
# Normal helpers
# ---------------------------------------------------------------------------
_SQRT2 = math.sqrt(2.0)


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


# ---------------------------------------------------------------------------
# Joint exit probability via SciPy multivariate-normal rectangle.
# ---------------------------------------------------------------------------
def _correlation(info_frac: list[float]) -> np.ndarray:
    K = len(info_frac)
    cov = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            ti, tj = info_frac[i], info_frac[j]
            cov[i, j] = math.sqrt(min(ti, tj) / max(ti, tj))
    return cov


def _no_reject_prob(
    boundaries: list[float],
    info_frac: list[float],
    drift: float,
    sides: int,
) -> float:
    """P(|Z_k| < b_k for all k) under drift theta (two-sided)
    or P(Z_k < b_k for all k) (one-sided).
    """
    K = len(boundaries)
    if K == 0:
        return 1.0
    mean = np.array([drift * math.sqrt(t) for t in info_frac])
    cov = _correlation(info_frac)
    rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)

    if sides == 1:
        return float(rv.cdf(np.array(boundaries)))

    # Two-sided rectangle via inclusion-exclusion.
    b = np.array(boundaries)
    total = 0.0
    for mask in range(1 << K):
        sign = 1
        upper = b.copy()
        for i in range(K):
            if (mask >> i) & 1:
                upper[i] = -b[i]
                sign = -sign
        total += sign * float(rv.cdf(upper))
    return total


def _exit_probability(
    boundaries: list[float],
    info_frac: list[float],
    drift: float,
    sides: int,
) -> float:
    """1 - P(no exit through all looks)."""
    return min(max(1.0 - _no_reject_prob(boundaries, info_frac, drift, sides),
                   0.0), 1.0)


# ---------------------------------------------------------------------------
# Boundary solver: find b_k such that the cumulative exit probability under
# H0 equals the cumulative alpha spent at look k.
# ---------------------------------------------------------------------------
def _solve_boundaries(
    info_frac: list[float],
    alpha: float,
    spending: str,
    sides: int,
) -> list[float]:
    spend_fn = _SPENDING[spending]
    K = len(info_frac)
    boundaries: list[float] = []
    for k in range(K):
        tau_k = info_frac[k]
        cum_alpha_k = spend_fn(tau_k, alpha, sides)

        # f(bk) = cum_exit(boundaries+[bk]) - cum_alpha_k
        # Decreasing in bk.
        def f(bk: float, k=k, boundaries=boundaries,
              cum_alpha_k=cum_alpha_k) -> float:
            trial = boundaries + [bk]
            return _exit_probability(
                trial, info_frac[: k + 1], drift=0.0, sides=sides,
            ) - cum_alpha_k

        lo, hi = 0.5, 12.0
        f_lo = f(lo)
        f_hi = f(hi)
        # f is monotone decreasing in bk; ensure bracket
        tries = 0
        while f_lo < 0.0 and tries < 30:
            lo *= 0.5
            f_lo = f(lo)
            tries += 1
        tries = 0
        while f_hi > 0.0 and tries < 30:
            hi *= 1.3
            f_hi = f(hi)
            tries += 1
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if f_mid > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-7:
                break
        boundaries.append(0.5 * (lo + hi))
    return boundaries


# ---------------------------------------------------------------------------
# Drift solver: find theta such that overall power = target.
# ---------------------------------------------------------------------------
def _solve_drift(
    boundaries: list[float],
    info_frac: list[float],
    power: float,
    sides: int,
) -> float:
    def f(theta: float) -> float:
        return _exit_probability(
            boundaries, info_frac, drift=theta, sides=sides,
        ) - power

    lo, hi = 0.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Expected sample size under H1.
# ---------------------------------------------------------------------------
def _expected_n(
    boundaries: list[float],
    info_frac: list[float],
    drift: float,
    n_max: int,
    sides: int,
) -> float:
    cum_exit_prev = 0.0
    expected = 0.0
    for k in range(len(boundaries)):
        cum_exit_k = _exit_probability(
            boundaries[: k + 1],
            info_frac[: k + 1],
            drift=drift,
            sides=sides,
        )
        inc = max(cum_exit_k - cum_exit_prev, 0.0)
        cum_exit_prev = cum_exit_k
        expected += inc * info_frac[k] * n_max
    expected += max(1.0 - cum_exit_prev, 0.0) * n_max
    return expected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def group_sequential_two_means(
    *,
    mean1: float,
    mean2: float,
    sd: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n_looks: int = 2,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    sides: int = 2,
    allocation: float = 1.0,
) -> dict[str, Any]:
    """Group-sequential design for two-mean superiority tests.

    Supply either ``power`` (solve for N) or ``n1`` (solve for power).
    """
    if mean1 == mean2:
        raise ValueError("mean1 and mean2 must differ")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(
            f"unknown boundary {boundary!r}; supported: "
            f"{sorted(_SPENDING.keys())}"
        )
    if n_looks < 1:
        raise ValueError("n_looks must be >= 1")

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)
    if abs(info_frac[-1] - 1.0) > 1e-9:
        raise ValueError("info_frac must end at 1.0")
    for i in range(1, len(info_frac)):
        if info_frac[i] <= info_frac[i - 1]:
            raise ValueError("info_frac must be strictly increasing")

    boundaries = _solve_boundaries(info_frac, alpha, boundary_key, sides)

    delta = mean1 - mean2
    if n1 is not None and power is None:
        n2 = max(2, math.ceil(allocation * n1))
        var_term = sd * sd * (1.0 / n1 + 1.0 / n2)
        theta = delta / math.sqrt(var_term)
        achieved = _exit_probability(
            boundaries, info_frac, drift=abs(theta), sides=sides,
        )
        n_max = n1 + n2
        expected = _expected_n(
            boundaries, info_frac, abs(theta), n_max, sides,
        )
        solve_for = "power"
        out_n1, out_n2 = n1, n2
        drift = theta
    elif power is not None:
        theta = _solve_drift(boundaries, info_frac, power, sides)
        # theta = delta / sqrt(sd^2 (1/n1 + 1/n2)); with n2 = r * n1:
        # n1 = (sd^2 (1 + 1/r) ) * (theta/delta)^2
        ratio = allocation
        n1_float = (sd * sd * (1.0 + 1.0 / ratio)) * (theta / delta) ** 2
        out_n1 = max(2, math.ceil(n1_float))
        out_n2 = max(2, math.ceil(ratio * out_n1))
        var_term = sd * sd * (1.0 / out_n1 + 1.0 / out_n2)
        theta_actual = abs(delta) / math.sqrt(var_term)
        achieved = _exit_probability(
            boundaries, info_frac, drift=theta_actual, sides=sides,
        )
        n_max = out_n1 + out_n2
        expected = _expected_n(
            boundaries, info_frac, theta_actual, n_max, sides,
        )
        solve_for = "n"
        drift = theta_actual
    else:
        raise ValueError("supply exactly one of (power, n1)")

    z_boundaries = list(boundaries)

    return {
        "method_id": "group_sequential_two_means",
        "solve_for": solve_for,
        "n": n_max,
        "n1": out_n1,
        "n2": out_n2,
        "n_max": n_max,
        "achieved_power": achieved,
        "n_expected_under_h1": expected,
        "z_boundaries": z_boundaries,
        "info_frac": list(info_frac),
        "drift": drift,
        "boundary": boundary_key,
        "n_looks": n_looks,
        "alpha": alpha,
        "sides": sides,
        "inputs_echo": {
            "mean1": mean1, "mean2": mean2, "sd": sd, "alpha": alpha,
            "power": power, "n1": n1, "n_looks": n_looks,
            "boundary": boundary, "sides": sides, "allocation": allocation,
        },
        "citations": [
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential boundaries for clinical trials. Biometrika 70.",
            "Reboussin, D.M., DeMets, D.L., Kim, K. & Lan, K.K.G. (1992). Programs for computing group sequential boundaries.",
            "O'Brien, P.C. & Fleming, T.R. (1979). A multiple testing procedure for clinical trials. Biometrics 35.",
            "Pocock, S.J. (1977). Group sequential methods in the design and analysis of clinical trials. Biometrika 64.",
        ],
    }


# ---------------------------------------------------------------------------
# Group-Sequential Tests for Two Proportions
# ---------------------------------------------------------------------------
# Reference: Wang & Tsiatis (1987); Lan & DeMets (1983).
#
# The drift parameter is:
#   theta = (p1 - p2) / sqrt(p_bar*(1-p_bar)/n1 + p_bar*(1-p_bar)/n2)
# where p_bar = (n1*p1 + n2*p2) / (n1 + n2).
#
# For equal allocation (n1 = n2 = n) this simplifies to:
#   p_bar = (p1 + p2) / 2
#   theta = (p1 - p2) / sqrt(2 * p_bar*(1-p_bar) / n)
# => theta = (p1 - p2) * sqrt(n) / sqrt(2 * p_bar*(1-p_bar))
# => n = (theta / (p1 - p2))^2 * 2 * p_bar*(1-p_bar)
# ---------------------------------------------------------------------------


def group_sequential_two_proportions(
    *,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n_looks: int = 4,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    sides: int = 2,
    allocation: float = 1.0,
) -> dict[str, Any]:
    """Group-sequential design for two-proportion superiority tests.

    using the Lan-DeMets alpha-spending formulation.  The test statistic is
    the standard two-sample z-test for proportions (no continuity correction).
    Supported spending functions: ``obrien-fleming`` and ``pocock``.

    Supply either ``power`` (solve for N) or ``n1`` (solve for power).

    Parameters
    ----------
    p1 : float
        Proportion in group 1 (null hypothesis value, also used under H1).
    p2 : float
        Proportion in group 2 under the alternative.
    alpha : float
        Overall two-sided (or one-sided) type-I error rate.
    power : float or None
        Target power (solve for N when supplied).
    n1 : int or None
        Per-group sample size at the final look (solve for power when
        supplied).
    n_looks : int
        Number of looks (interim + final).  Default 4.
    boundary : str
        Spending function: ``obrien-fleming`` (default) or ``pocock``.
    info_frac : list[float] or None
        Custom information fractions (must end at 1.0).
    sides : int
        1 or 2 (default 2).
    allocation : float
        N2 / N1 ratio.  Default 1.0 (equal allocation).
    """
    if not (0.0 < p1 < 1.0):
        raise ValueError("p1 must be in (0, 1)")
    if not (0.0 < p2 < 1.0):
        raise ValueError("p2 must be in (0, 1)")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    boundary_key = boundary.lower().replace("_", "-")
    if boundary_key not in _SPENDING:
        raise ValueError(
            f"unknown boundary {boundary!r}; supported: "
            f"{sorted(_SPENDING.keys())}"
        )
    if n_looks < 1:
        raise ValueError("n_looks must be >= 1")

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)
    if abs(info_frac[-1] - 1.0) > 1e-9:
        raise ValueError("info_frac must end at 1.0")
    for i in range(1, len(info_frac)):
        if info_frac[i] <= info_frac[i - 1]:
            raise ValueError("info_frac must be strictly increasing")

    boundaries = _solve_boundaries(info_frac, alpha, boundary_key, sides)

    ratio = allocation
    delta = p2 - p1  # signed difference

    def _drift_from_n1(nn1: int) -> float:
        nn2 = max(2, math.ceil(ratio * nn1))
        p_bar = (nn1 * p1 + nn2 * p2) / (nn1 + nn2)
        se = math.sqrt(p_bar * (1.0 - p_bar) * (1.0 / nn1 + 1.0 / nn2))
        if se <= 0:
            return 0.0
        return abs(delta) / se

    if n1 is not None and power is None:
        out_n1 = n1
        out_n2 = max(2, math.ceil(ratio * n1))
        theta = _drift_from_n1(n1)
        achieved = _exit_probability(
            boundaries, info_frac, drift=theta, sides=sides,
        )
        n_max = out_n1 + out_n2
        expected = _expected_n(boundaries, info_frac, theta, n_max, sides)
        solve_for = "power"

    elif power is not None:
        # Solve theta from power, then invert drift-to-n relation.
        theta = _solve_drift(boundaries, info_frac, power, sides)

        # theta = |delta| / sqrt(p_bar*(1-p_bar)*(1/n1 + 1/n2))
        # With equal allocation p_bar = (p1+p2)/2, n2 = ratio*n1:
        #   var_term = p_bar*(1-p_bar)*(1 + 1/ratio) / n1
        # => n1_float = p_bar*(1-p_bar)*(1+1/ratio) * (theta/delta)^2
        p_bar_avg = (p1 * ratio + p2) / (1.0 + ratio)
        var_factor = p_bar_avg * (1.0 - p_bar_avg) * (1.0 + 1.0 / ratio)
        n1_float = var_factor * (theta / abs(delta)) ** 2
        out_n1 = max(2, math.ceil(n1_float))
        out_n2 = max(2, math.ceil(ratio * out_n1))

        # Recalculate achieved power at ceiling N.
        theta_actual = _drift_from_n1(out_n1)
        achieved = _exit_probability(
            boundaries, info_frac, drift=theta_actual, sides=sides,
        )
        n_max = out_n1 + out_n2
        expected = _expected_n(
            boundaries, info_frac, theta_actual, n_max, sides,
        )
        solve_for = "n"
        theta = theta_actual

    else:
        raise ValueError("supply exactly one of (power, n1)")

    return {
        "method_id": "group_sequential_two_proportions",
        "solve_for": solve_for,
        "n": n_max,
        "n1": out_n1,
        "n2": out_n2,
        "n_max": n_max,
        "achieved_power": achieved,
        "n_expected_under_h1": expected,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "drift": theta,
        "boundary": boundary_key,
        "n_looks": n_looks,
        "alpha": alpha,
        "sides": sides,
        "inputs_echo": {
            "p1": p1, "p2": p2, "alpha": alpha,
            "power": power, "n1": n1, "n_looks": n_looks,
            "boundary": boundary, "sides": sides, "allocation": allocation,
        },
        "citations": [
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Reboussin, D.M., DeMets, D.L., Kim, K. & Lan, K.K.G. (1992). "
            "Programs for computing group sequential boundaries.",
            "O'Brien, P.C. & Fleming, T.R. (1979). A multiple testing "
            "procedure for clinical trials. Biometrics 35.",
            "Pocock, S.J. (1977). Group sequential methods in the design "
            "and analysis of clinical trials. Biometrika 64.",
        ],
    }
