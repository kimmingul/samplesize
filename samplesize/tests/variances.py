"""Tests for variances (chi-square one-variance and F-ratio two-variance tests).

Implements:
  * ``tests_one_variance`` — chi-square test of H0: sigma^2 = sigma_0^2.
  * ``tests_two_variances`` — F-test of H0: sigma_1^2 = sigma_2^2.

Power formulas (Ostle & Malone, 1988, p. 130).  Inputs ``v0``/``v1`` /
``v1`` / ``v2`` may be variances or standard deviations; pass
``scale="sd"`` to interpret as SD.  ``sides=1`` selects a one-tailed
test with the direction implied by the supplied effect (``v1 < v0`` =>
lower tail; ``v1 > v0`` => upper tail); ``sides=2`` is two-tailed.

One-variance critical region (variance scale):
  test statistic X = (n-1) * s^2 / v0  ~ chi2(n-1) under H0
  reject when X is in the rejection region; under the alternative
  X1 = (n-1) * s^2 / v1 ~ chi2(n-1) so X = X1 * v1/v0.

Two-variance critical region:
  F = s1^2 / s2^2  ~  F(n1-1, n2-1) under H0
  under the alternative F* = F_central * (v1/v2) so we work in the
  central F by rescaling the rejection cut-points by (v2/v1).
"""
from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Internal: variance/SD scale helpers


def _to_variance(value: float, scale: str) -> float:
    if scale == "variance":
        return float(value)
    if scale == "sd":
        return float(value) * float(value)
    raise ValueError(f"scale must be 'variance' or 'sd', got {scale!r}")


def _direction(v0: float, v1: float, sides: int) -> int:
    """Return -1 (lower), +1 (upper), or 0 (two-sided)."""
    if sides == 2:
        return 0
    if sides == 1:
        if v1 == v0:
            raise ValueError("v1 must differ from v0 for a one-sided test")
        return -1 if v1 < v0 else +1
    raise ValueError(f"sides must be 1 or 2, got {sides}")


# ---------------------------------------------------------------------------
# Tests for One Variance


def _one_var_power(v0: float, v1: float, n: int, alpha: float,
                   sides: int, known_mean: bool) -> float:
    if n < 2:
        return 0.0
    from scipy.stats import chi2
    df = n if known_mean else n - 1
    if df <= 0:
        return 0.0
    ratio = v0 / v1
    direction = _direction(v0, v1, sides)
    if direction == 0:
        # Two-sided: reject when X < chi2_{alpha/2} or X > chi2_{1-alpha/2}.
        c_lo = chi2.ppf(alpha / 2.0, df)
        c_hi = chi2.ppf(1.0 - alpha / 2.0, df)
        # Under alternative, X = X1 * v1/v0, so the cut-points in X1-space
        # are c * v0/v1 = c * ratio.
        lower_tail = chi2.cdf(c_lo * ratio, df)
        upper_tail = 1.0 - chi2.cdf(c_hi * ratio, df)
        return float(lower_tail + upper_tail)
    if direction == -1:
        # v1 < v0: reject when X is small, i.e. X < chi2_{alpha}.
        c = chi2.ppf(alpha, df)
        return float(chi2.cdf(c * ratio, df))
    # direction == +1: v1 > v0; reject when X is large.
    c = chi2.ppf(1.0 - alpha, df)
    return float(1.0 - chi2.cdf(c * ratio, df))


def _one_var_n(v0: float, v1: float, alpha: float, power: float,
               sides: int, known_mean: bool,
               n_min: int = 2, n_max: int = 1_000_000) -> tuple[int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if v0 == v1:
        raise ValueError("v0 and v1 must differ to solve for N")
    lo = max(n_min, 2)
    hi = lo
    while hi <= n_max:
        if _one_var_power(v0, v1, hi, alpha, sides, known_mean) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _one_var_power(v0, v1, mid, alpha, sides, known_mean) >= power:
            hi = mid
        else:
            lo = mid
    return hi, _one_var_power(v0, v1, hi, alpha, sides, known_mean)


def tests_one_variance(
    *,
    v0: float,
    v1: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    scale: str = "variance",
    known_mean: bool = False,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Chi-square test of a single variance.

    Provide exactly one of ``power`` or ``n``; the other is solved for.
    ``v0`` is the null (baseline) value, ``v1`` is the alternative
    value at which power is evaluated.  When ``scale="sd"`` both are
    treated as standard deviations.

    When ``known_mean`` is True the chi-square degrees of freedom is
    ``n`` (rather than ``n - 1``), increasing power slightly.
    """
    inputs_echo = {
        "v0": v0, "v1": v1, "alpha": alpha, "power": power, "n": n,
        "sides": sides, "scale": scale, "known_mean": known_mean,
    }
    if v0 <= 0 or v1 <= 0:
        raise ValueError("v0 and v1 must be positive")

    var0 = _to_variance(v0, scale)
    var1 = _to_variance(v1, scale)

    if solve_for is None:
        if n is None and power is not None:
            solve_for = "n"
        elif power is None and n is not None:
            solve_for = "power"
        else:
            raise ValueError("supply exactly one of (n, power)")

    if solve_for == "power":
        assert n is not None
        achieved = _one_var_power(var0, var1, n, alpha, sides, known_mean)
        result = {"n": n, "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n_req, achieved = _one_var_n(var0, var1, alpha, power, sides, known_mean)
        result = {"n": n_req, "achieved_power": achieved}
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "tests_one_variance",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Ostle, B. & Malone, L.C. (1988). Statistics in Research, 4th ed., p. 130.",
        ],
    }


# ---------------------------------------------------------------------------
# Tests for Two Variances


def _two_var_power(v1: float, v2: float, n1: int, n2: int,
                   alpha: float, sides: int) -> float:
    if n1 < 2 or n2 < 2:
        return 0.0
    from scipy.stats import f as fdist
    df1, df2 = n1 - 1, n2 - 1
    # Under H1 the central F statistic equals (s1^2/v1) / (s2^2/v2)
    # ~ F(df1, df2).  The observed ratio F* = s1^2/s2^2 = F_central * (v1/v2).
    direction = _direction(v1, v2, sides)
    # NOTE: _direction tests "v1 ≠ v0".  Two-variance cases:
    #   Case 2 (Ha: V1 > V2): upper-tail reject in F* > F_{1-alpha}.
    #   Case 3 (Ha: V1 < V2): lower-tail reject in F* < F_{alpha}.
    # _direction(v1, v2) returns +1 when v2 > v1 (i.e. V1 < V2) — that is
    # the lower-tail direction.  So adapt:
    scale = v1 / v2  # multiplier converting central F to observed F*

    if sides == 2:
        c_lo = fdist.ppf(alpha / 2.0, df1, df2)
        c_hi = fdist.ppf(1.0 - alpha / 2.0, df1, df2)
        # observed F* > c_hi  <=>  F_central > c_hi / scale
        upper = 1.0 - fdist.cdf(c_hi / scale, df1, df2)
        lower = fdist.cdf(c_lo / scale, df1, df2)
        return float(upper + lower)
    if v1 > v2:
        # Case 2: reject when F* > F_{1-alpha}.
        c = fdist.ppf(1.0 - alpha, df1, df2)
        return float(1.0 - fdist.cdf(c / scale, df1, df2))
    # v1 < v2: reject when F* < F_{alpha}.
    c = fdist.ppf(alpha, df1, df2)
    return float(fdist.cdf(c / scale, df1, df2))


def _two_var_n(v1: float, v2: float, alpha: float, power: float,
               sides: int, allocation: float = 1.0,
               n_min: int = 2, n_max: int = 10_000_000) -> tuple[int, int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if v1 == v2:
        raise ValueError("v1 and v2 must differ to solve for N")
    if allocation <= 0:
        raise ValueError("allocation (n2/n1) must be > 0")

    def n2_for(n1: int) -> int:
        return max(2, math.ceil(allocation * n1))

    def p_at(n1: int) -> float:
        n2 = n2_for(n1)
        return _two_var_power(v1, v2, n1, n2, alpha, sides)

    lo = max(n_min, 2)
    hi = lo
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
    return n1, n2, _two_var_power(v1, v2, n1, n2, alpha, sides)


def tests_two_variances(
    *,
    v1: float,
    v2: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    scale: str = "variance",
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """F-ratio test comparing two variances.

    Provide either (n1, n2) for "power", or ``power`` to solve for the
    per-group sample size with the given ``allocation`` (n2 = ceil(allocation·n1)).
    """
    inputs_echo = {
        "v1": v1, "v2": v2, "alpha": alpha, "power": power,
        "n1": n1, "n2": n2, "sides": sides, "scale": scale,
        "allocation": allocation,
    }
    if v1 <= 0 or v2 <= 0:
        raise ValueError("v1 and v2 must be positive")

    var1 = _to_variance(v1, scale)
    var2 = _to_variance(v2, scale)

    # Promote n1 ↔ n2 from allocation when one is missing.
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    elif n2 is not None and n1 is None:
        n1 = max(2, math.ceil(n2 / allocation))

    have_n = n1 is not None and n2 is not None
    have_power = power is not None

    if solve_for is None:
        if have_n and not have_power:
            solve_for = "power"
        elif have_power and not have_n:
            solve_for = "n"
        else:
            raise ValueError("supply exactly one of (n, power)")

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _two_var_power(var1, var2, n1, n2, alpha, sides)
        result = {
            "n1": n1, "n2": n2, "n": n1 + n2,
            "achieved_power": achieved,
        }
    elif solve_for == "n":
        assert power is not None
        n1r, n2r, achieved = _two_var_n(var1, var2, alpha, power, sides,
                                        allocation=allocation)
        result = {
            "n1": n1r, "n2": n2r, "n": n1r + n2r,
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "tests_two_variances",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Davies, O.L. (1971). Statistical Methods in Research and Production, p. 41.",
        ],
    }
