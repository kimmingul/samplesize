"""Tests for Two Means in a Multicenter Randomized Design.

A two-arm trial stratified across :math:`Q` centers.  The data are
analysed with a mixed-effects model that has a fixed treatment effect
and a *random* center effect (no treatment-by-center interaction in the
primary test):

.. math::
    Y_{ijk} = \\mu + \\delta_i + C_j + \\varepsilon_{ijk}

with :math:`C_j \\sim N(0, \\sigma_C^2)` and
:math:`\\varepsilon_{ijk} \\sim N(0, \\sigma_\\varepsilon^2)`.  The
intraclass correlation is :math:`\\rho =
\\sigma_C^2 / (\\sigma_C^2 + \\sigma_\\varepsilon^2)` and the response
standard deviation is :math:`\\sigma = \\sqrt{\\sigma_C^2 +
\\sigma_\\varepsilon^2}`.

Power formula
-------------
Vierron and Giraudeau (2007) give the F-test of the treatment effect as

.. math::
    \\text{Power} = \\Phi\\!\\left(\\frac{\\delta\\sqrt{N}}{2\\,\\sigma\\sqrt{1-\\rho}}
                                 - z_{1-\\alpha/2}\\right)

where :math:`N` is the *total* sample size pooled across centers and
arms (some references drop the factor of 2; the
example only matches with the factor included).  This formula does not
depend on either :math:`Q` (number of centers) or the cluster size,
provided centers are roughly balanced.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


def _multicenter_power(
    *,
    delta: float,
    sd: float,
    icc: float,
    n_total: int,
    alpha: float,
    sides: int,
) -> float:
    if sd <= 0:
        raise ValueError("sd must be positive")
    if not 0.0 <= icc < 1.0:
        raise ValueError("icc must lie in [0, 1)")
    if n_total < 4:
        return 0.0

    from scipy.stats import norm
    se = 2.0 * sd * math.sqrt(1.0 - icc) / math.sqrt(n_total)
    ncp = abs(delta) / se
    if sides == 2:
        z_crit = D.norm_ppf(1.0 - alpha / 2.0)
        upper = 1.0 - norm.cdf(z_crit - ncp)
        lower = norm.cdf(-z_crit - ncp)
        return float(upper + lower)
    if sides == 1:
        z_crit = D.norm_ppf(1.0 - alpha)
        return float(1.0 - norm.cdf(z_crit - ncp))
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def _n_for_multicenter(
    *,
    delta: float,
    sd: float,
    icc: float,
    alpha: float,
    power: float,
    sides: int,
    n_min: int = 4,
    n_max: int = 10_000_000,
) -> tuple[int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if delta == 0:
        raise ValueError("delta must be non-zero to solve for N")

    def p_at(n: int) -> float:
        return _multicenter_power(
            delta=delta, sd=sd, icc=icc, n_total=n,
            alpha=alpha, sides=sides,
        )

    lo, hi = n_min, n_min
    while hi <= n_max:
        if p_at(hi) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if p_at(mid) >= power:
            hi = mid
        else:
            lo = mid

    return hi, p_at(hi)


def tests_two_means_multicenter(
    *,
    delta: float,
    sd: float,
    icc: float,
    alpha: float = 0.05,
    power: float | None = None,
    n_total: int | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two means in a multicenter randomized design.

    Parameters
    ----------
    delta : float
        Treatment effect :math:`\\mu_1 - \\mu_2`.
    sd : float
        Response standard deviation
        :math:`\\sigma = \\sqrt{\\sigma_C^2 + \\sigma_\\varepsilon^2}`.
    icc : float
        Intraclass (center) correlation :math:`\\rho = \\sigma_C^2 /
        \\sigma^2`, between 0 (inclusive) and 1 (exclusive).
    alpha : float
        Type-I error rate.
    power : float, optional
        Target power.  Supply this OR ``n_total``.
    n_total : int, optional
        Total sample size pooled across centers and arms.
    sides : int
        2 (default) or 1.
    solve_for : {"n", "power"}, optional
        Override the auto-detected target.
    """
    inputs_echo = {
        "delta": delta, "sd": sd, "icc": icc, "alpha": alpha,
        "power": power, "n_total": n_total, "sides": sides,
    }

    have_n = n_total is not None
    have_power = power is not None
    if have_n == have_power:
        raise ValueError("supply exactly one of (power, n_total)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n_total is not None
        n_out = int(n_total)
        achieved = _multicenter_power(
            delta=delta, sd=sd, icc=icc, n_total=n_out,
            alpha=alpha, sides=sides,
        )
    elif solve_for == "n":
        assert power is not None
        n_out, achieved = _n_for_multicenter(
            delta=delta, sd=sd, icc=icc, alpha=alpha,
            power=power, sides=sides,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    sd_center = sd * math.sqrt(icc)
    sd_error = sd * math.sqrt(1.0 - icc)
    return {
        "method_id": "tests_two_means_multicenter",
        "solve_for": solve_for,
        "n": n_out,
        "n_total": n_out,
        "achieved_power": achieved,
        "delta": delta,
        "sd_center": sd_center,
        "sd_error": sd_error,
        "inputs_echo": inputs_echo,
        "citations": [
            "Randomized Design (chapter 481).",
            "Vierron, E. and Giraudeau, B. (2007). Sample size "
            "calculation for multicenter randomized trial: Taking the "
            "center effect into account. Contemp Clin Trials 28:451-458.",
            "Vierron, E. and Giraudeau, B. (2009). Design effect in "
            "multicenter studies: gain or loss of power? BMC Med Res "
            "Methodol 9:39.",
        ],
    }
