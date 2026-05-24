"""Two-sample z-test power / sample-size (known variances).

  * Two-Sample Z-Tests Assuming Equal Variance (Enter Means)        [426]
  * Two-Sample Z-Tests Assuming Equal Variance (Enter Difference)   [427]
  * Two-Sample Z-Tests Allowing Unequal Variance (Enter Means)      [428]
  * Two-Sample Z-Tests Allowing Unequal Variance (Enter Difference) [429]

The z-test assumes the population variances are *known* (not estimated
from the sample), so power uses the standard normal distribution -- no
t / Welch-Satterthwaite degrees of freedom.

  equal var  : SE = sigma * sqrt(1/n1 + 1/n2)
  unequal var: SE = sqrt(sigma1**2/n1 + sigma2**2/n2)
  ncp        = (mu1 - mu2) / SE = delta / SE

Two-sided power:
  Power = Phi(ncp - z_{1-alpha/2}) + Phi(-ncp - z_{1-alpha/2})

One-sided power (Ha: mu1 > mu2):
  Power = Phi(ncp - z_{1-alpha})

One-sided power (Ha: mu1 < mu2):
  Power = Phi(-ncp - z_{1-alpha})    (= 1 - Phi(z_{1-alpha} + ncp))
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D
from samplesize.core import effect_sizes as E


# ---------------------------------------------------------------------------
# Core power formula (shared helper)
# ---------------------------------------------------------------------------

def _z_power(delta: float, se: float, alpha: float, sides: int) -> float:
    """Power of a two-sample z-test with known variances.

    Parameters
    ----------
    delta : assumed mu1 - mu2 (signed).
    se    : standard error of (xbar1 - xbar2) under the design.
    alpha : significance level.
    sides : 1 or 2.

    For sides==1, the direction is inferred from sign(delta): delta>0
    corresponds to Ha: mu1 > mu2; delta<0 to Ha: mu1 < mu2.  delta==0
    gives power == alpha.
    """
    if se <= 0:
        return 0.0
    ncp = delta / se
    if sides == 2:
        z_crit = D.norm_ppf(1.0 - alpha / 2.0)
        # P(Z > z_crit - ncp) + P(Z < -z_crit - ncp)
        from scipy.stats import norm
        return float(norm.sf(z_crit - ncp) + norm.cdf(-z_crit - ncp))
    if sides == 1:
        z_crit = D.norm_ppf(1.0 - alpha)
        from scipy.stats import norm
        if delta >= 0:
            return float(norm.sf(z_crit - ncp))
        return float(norm.cdf(-z_crit - ncp))
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def _se_equal(sd: float, n1: int, n2: int) -> float:
    return sd * math.sqrt(1.0 / n1 + 1.0 / n2)


def _se_unequal(sd1: float, sd2: float, n1: int, n2: int) -> float:
    return math.sqrt(sd1 * sd1 / n1 + sd2 * sd2 / n2)


def _solve_n(
    *,
    delta: float,
    se_at: Any,  # callable: (n1, n2) -> se
    alpha: float,
    power: float,
    sides: int,
    allocation: float,
    n_min: int = 2,
    n_max: int = 10_000_000,
) -> tuple[int, int, float]:
    """Bracketed bisection for the smallest n1 (and n2 = ceil(allocation*n1))
    that achieves the requested power.  Mirrors the t-test sibling.
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if delta == 0.0:
        raise ValueError("delta (mu1 - mu2) must be non-zero to solve for N")
    if allocation <= 0:
        raise ValueError("allocation (n2/n1) must be > 0")

    def n2_for(n1: int) -> int:
        return max(2, math.ceil(allocation * n1))

    def p_at(n1: int) -> float:
        n2 = n2_for(n1)
        return _z_power(delta, se_at(n1, n2), alpha, sides)

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

    n1 = hi
    n2 = n2_for(n1)
    return n1, n2, _z_power(delta, se_at(n1, n2), alpha, sides)


# ---------------------------------------------------------------------------
# Shared dispatcher (equal vs unequal variance, means vs delta)
# ---------------------------------------------------------------------------

def _dispatch_equal_var(
    *,
    delta: float,
    sd: float,
    alpha: float,
    power: float | None,
    n1: int | None,
    n2: int | None,
    sides: int,
    allocation: float,
    solve_for: str | None,
) -> tuple[str, int, int, float]:
    """Return (solve_for, n1, n2, achieved_power) for equal-variance case."""
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    elif n2 is not None and n1 is None:
        n1 = max(2, math.ceil(n2 / allocation))

    have_n = n1 is not None and n2 is not None
    have_power = power is not None

    if not (have_n or have_power):
        raise ValueError("supply (n1 [and n2]) or power")

    if solve_for is None:
        if not have_n:
            solve_for = "n"
        elif not have_power:
            solve_for = "power"
        else:
            raise ValueError("supply exactly one of (n, power) as missing")

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        se = _se_equal(sd, n1, n2)
        achieved = _z_power(delta, se, alpha, sides)
        return "power", n1, n2, achieved
    if solve_for == "n":
        assert power is not None
        n1r, n2r, achieved = _solve_n(
            delta=delta,
            se_at=lambda a, b: _se_equal(sd, a, b),
            alpha=alpha,
            power=power,
            sides=sides,
            allocation=allocation,
        )
        return "n", n1r, n2r, achieved
    raise ValueError(f"unknown solve_for: {solve_for!r}")


def _dispatch_unequal_var(
    *,
    delta: float,
    sd1: float,
    sd2: float,
    alpha: float,
    power: float | None,
    n1: int | None,
    n2: int | None,
    sides: int,
    allocation: float,
    solve_for: str | None,
) -> tuple[str, int, int, float]:
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    elif n2 is not None and n1 is None:
        n1 = max(2, math.ceil(n2 / allocation))

    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply (n1 [and n2]) or power")

    if solve_for is None:
        if not have_n:
            solve_for = "n"
        elif not have_power:
            solve_for = "power"
        else:
            raise ValueError("supply exactly one of (n, power) as missing")

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        se = _se_unequal(sd1, sd2, n1, n2)
        achieved = _z_power(delta, se, alpha, sides)
        return "power", n1, n2, achieved
    if solve_for == "n":
        assert power is not None
        n1r, n2r, achieved = _solve_n(
            delta=delta,
            se_at=lambda a, b: _se_unequal(sd1, sd2, a, b),
            alpha=alpha,
            power=power,
            sides=sides,
            allocation=allocation,
        )
        return "n", n1r, n2r, achieved
    raise ValueError(f"unknown solve_for: {solve_for!r}")


# ---------------------------------------------------------------------------
# Public API: 4 two-sample z-test variants
# ---------------------------------------------------------------------------

_CITATIONS_EQUAL = [
    "Julious, S.A. (2010). Sample Sizes for Clinical Trials.",
    "Chow, Shao & Wang (2008). Sample Size Calculations in Clinical Research, 2nd ed.",
    "Machin, Campbell, Fayers & Pinol (1997). Sample Size Tables.",
]

_CITATIONS_UNEQUAL = [
    "Julious, S.A. (2010). Sample Sizes for Clinical Trials.",
    "Chow, Shao & Wang (2008). Sample Size Calculations in Clinical Research, 2nd ed.",
]


def two_sample_z_equal_var_means(
    *,
    mean1: float,
    mean2: float,
    sd: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-sample z-test, equal known variance, entered as means.

    Supply exactly two of (`mean2`, `power`, `n1`-with-allocation); the
    missing one is solved for.  `mean2` is required here so the missing
    quantity is either `power` or `n`.
    """
    inputs_echo = {
        "mean1": mean1, "mean2": mean2, "sd": sd, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2, "sides": sides,
        "allocation": allocation,
    }
    delta = mean1 - mean2

    solved, n1r, n2r, achieved = _dispatch_equal_var(
        delta=delta, sd=sd, alpha=alpha, power=power,
        n1=n1, n2=n2, sides=sides, allocation=allocation,
        solve_for=solve_for,
    )

    return {
        "method_id": "two_sample_z_equal_var_means",
        "solve_for": solved,
        "n1": n1r, "n2": n2r, "n": n1r + n2r,
        "achieved_power": achieved,
        "effect_d": E.cohens_d(mean1, mean2, sd),
        "delta": delta,
        "inputs_echo": inputs_echo,
        "citations": _CITATIONS_EQUAL,
    }


def two_sample_z_equal_var_diff(
    *,
    delta: float,
    sd: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-sample z-test, equal known variance, entered as difference.

    `delta = mu1 - mu2` is supplied directly (signed; non-zero).
    """
    inputs_echo = {
        "delta": delta, "sd": sd, "alpha": alpha, "power": power,
        "n1": n1, "n2": n2, "sides": sides, "allocation": allocation,
    }

    solved, n1r, n2r, achieved = _dispatch_equal_var(
        delta=delta, sd=sd, alpha=alpha, power=power,
        n1=n1, n2=n2, sides=sides, allocation=allocation,
        solve_for=solve_for,
    )

    return {
        "method_id": "two_sample_z_equal_var_diff",
        "solve_for": solved,
        "n1": n1r, "n2": n2r, "n": n1r + n2r,
        "achieved_power": achieved,
        "effect_d": delta / sd if sd > 0 else float("nan"),
        "delta": delta,
        "inputs_echo": inputs_echo,
        "citations": _CITATIONS_EQUAL,
    }


def two_sample_z_unequal_var_means(
    *,
    mean1: float,
    mean2: float,
    sd1: float,
    sd2: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-sample z-test, unequal known variances, entered as means."""
    inputs_echo = {
        "mean1": mean1, "mean2": mean2, "sd1": sd1, "sd2": sd2,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2,
        "sides": sides, "allocation": allocation,
    }
    delta = mean1 - mean2

    solved, n1r, n2r, achieved = _dispatch_unequal_var(
        delta=delta, sd1=sd1, sd2=sd2, alpha=alpha, power=power,
        n1=n1, n2=n2, sides=sides, allocation=allocation,
        solve_for=solve_for,
    )

    # Effect size: standardised mean difference vs pooled (equal-n) SD.
    pooled_sd = math.sqrt(0.5 * (sd1 * sd1 + sd2 * sd2))
    return {
        "method_id": "two_sample_z_unequal_var_means",
        "solve_for": solved,
        "n1": n1r, "n2": n2r, "n": n1r + n2r,
        "achieved_power": achieved,
        "effect_d": (mean1 - mean2) / pooled_sd if pooled_sd > 0 else float("nan"),
        "delta": delta,
        "inputs_echo": inputs_echo,
        "citations": _CITATIONS_UNEQUAL,
    }


def two_sample_z_unequal_var_diff(
    *,
    delta: float,
    sd1: float,
    sd2: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-sample z-test, unequal known variances, entered as difference."""
    inputs_echo = {
        "delta": delta, "sd1": sd1, "sd2": sd2, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2, "sides": sides,
        "allocation": allocation,
    }

    solved, n1r, n2r, achieved = _dispatch_unequal_var(
        delta=delta, sd1=sd1, sd2=sd2, alpha=alpha, power=power,
        n1=n1, n2=n2, sides=sides, allocation=allocation,
        solve_for=solve_for,
    )

    pooled_sd = math.sqrt(0.5 * (sd1 * sd1 + sd2 * sd2))
    return {
        "method_id": "two_sample_z_unequal_var_diff",
        "solve_for": solved,
        "n1": n1r, "n2": n2r, "n": n1r + n2r,
        "achieved_power": achieved,
        "effect_d": delta / pooled_sd if pooled_sd > 0 else float("nan"),
        "delta": delta,
        "inputs_echo": inputs_echo,
        "citations": _CITATIONS_UNEQUAL,
    }
