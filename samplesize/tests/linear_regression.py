"""Simple linear regression slope test — power & sample-size.


Tests H0: B = B0 vs H1: B ≠ B0 (or one-sided) for the slope of a
simple linear regression Y = A + B·X using the noncentral-F formulation.

The power function is

    Power = Pr(F > F_crit)

where F_crit is from the central F(1, N-2) distribution and F is
distributed as noncentral F(1, N-2, λ) with noncentrality parameter

    λ = N · (SX · (B - B0) / σ)²

and σ is the standard deviation of the residuals.

The residual SD σ can be specified:
  - Directly via ``sd_residuals`` (method "S").
  - Via the SD of Y (``sd_y``, method "SY"):
        σ = sqrt(sy² - B1² · SX²)
  - Via the correlation (``correlation``, method "R"):
        σ = B1 · SX · sqrt(1/R² - 1)

For a one-sided test, use ``2·alpha`` in place of ``alpha`` as
documented in Dupont & Plummer (1998).

References
----------
* Neter, J., Wasserman, W., and Kutner, M.H. (1983). Applied Linear
  Statistical Models, 2nd Ed. Richard D. Irwin, Inc.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import f as fdist
from scipy.stats import ncf

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Core power computation
# ---------------------------------------------------------------------------


def _compute_sigma(
    *,
    slope: float,
    b0: float,
    sx: float,
    sd_residuals: float | None,
    sd_y: float | None,
    correlation: float | None,
) -> float:
    """Compute residual SD from whichever specification is provided."""
    n_provided = sum(x is not None for x in (sd_residuals, sd_y, correlation))
    if n_provided != 1:
        raise ValueError(
            "Provide exactly one of: sd_residuals, sd_y, or correlation"
        )
    if sd_residuals is not None:
        if sd_residuals <= 0:
            raise ValueError("sd_residuals must be positive")
        return sd_residuals
    if sd_y is not None:
        if sd_y <= 0:
            raise ValueError("sd_y must be positive")
        val = sd_y ** 2 - slope ** 2 * sx ** 2
        if val <= 0:
            raise ValueError(
                "sd_y is too small relative to slope and sx: "
                "sd_y² - slope²·sx² must be > 0"
            )
        return math.sqrt(val)
    # correlation
    assert correlation is not None
    if not (-1.0 < correlation < 1.0) or correlation == 0.0:
        raise ValueError("correlation must be in (-1, 1) and non-zero")
    return abs(slope) * sx * math.sqrt(1.0 / correlation ** 2 - 1.0)


def _ncp(
    *,
    n: int,
    slope: float,
    b0: float,
    sx: float,
    sigma: float,
) -> float:
    """Noncentrality parameter λ = N · (SX·(B-B0)/σ)²."""
    return n * (sx * (slope - b0) / sigma) ** 2


def _power_at_n(
    *,
    n: int,
    slope: float,
    b0: float,
    sx: float,
    sigma: float,
    alpha: float,
    sides: int,
) -> float:
    """Power of the F-test for the slope."""
    if n <= 2:
        return 0.0
    df1, df2 = 1, n - 2
    # For one-sided test use 2·alpha (Dupont & Plummer 1998)
    alpha_eff = alpha if sides == 2 else 2.0 * alpha
    f_crit = fdist.ppf(1.0 - alpha_eff, df1, df2)
    lam = _ncp(n=n, slope=slope, b0=b0, sx=sx, sigma=sigma)
    return float(1.0 - ncf.cdf(f_crit, df1, df2, lam))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def linear_regression_test_slope(
    *,
    slope: float,
    sx: float,
    b0: float = 0.0,
    sd_residuals: float | None = None,
    sd_y: float | None = None,
    correlation: float | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    n: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power and sample size for testing the slope in simple linear regression.

    Tests H0: B = B0 vs H1: B ≠ B0 (two-sided) or H1: B > B0 / B < B0
    (one-sided) using a noncentral-F approach.


    Parameters
    ----------
    slope
        Slope (B1) under the alternative hypothesis.
    sx
        Standard deviation of the X values in the sample.
        Computed as sqrt(Σ(xi-x̄)²/N) — using N not N-1.
    b0
        Slope under the null hypothesis.  Default 0.
    sd_residuals
        Standard deviation of the residuals σ.  Provide exactly one of
        (sd_residuals, sd_y, correlation).
    sd_y
        Standard deviation of Y (ignoring X).  σ = sqrt(sy²-B1²·sx²).
    correlation
        Pearson correlation between Y and X.  σ = |B1|·sx·sqrt(1/R²-1).
    alpha
        Type-I error rate.  Default 0.05.
    sides
        1 or 2 (default 2).  For one-sided tests 2·alpha is used in
        the F-distribution.
    n
        Total sample size (provide when solve_for="power").
    power
        Target power (provide when solve_for="n").
    solve_for
        ``"n"`` or ``"power"``.  Inferred if only one of (n, power) given.
    """
    if sx <= 0:
        raise ValueError("sx must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    sigma = _compute_sigma(
        slope=slope, b0=b0, sx=sx,
        sd_residuals=sd_residuals, sd_y=sd_y, correlation=correlation,
    )

    inputs_echo: dict[str, Any] = {
        "slope": slope, "sx": sx, "b0": b0,
        "sd_residuals": sd_residuals, "sd_y": sd_y, "correlation": correlation,
        "alpha": alpha, "sides": sides, "n": n, "power": power,
        "sigma_used": sigma,
    }

    given = sum(x is not None for x in (n, power))
    if given == 0:
        raise ValueError("supply exactly one of (n, power)")
    if given == 2 and solve_for is None:
        raise ValueError("both n and power given; specify solve_for explicitly")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _power_at_n(
            n=n, slope=slope, b0=b0, sx=sx, sigma=sigma,
            alpha=alpha, sides=sides,
        )
        n_out = n

    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        if slope == b0:
            raise ValueError("slope must differ from b0 to solve for n")

        # Binary search
        lo, hi = 3, 10
        while hi <= 10_000_000:
            if _power_at_n(n=hi, slope=slope, b0=b0, sx=sx, sigma=sigma,
                           alpha=alpha, sides=sides) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 10,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _power_at_n(n=mid, slope=slope, b0=b0, sx=sx, sigma=sigma,
                           alpha=alpha, sides=sides) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        achieved = _power_at_n(
            n=n_out, slope=slope, b0=b0, sx=sx, sigma=sigma,
            alpha=alpha, sides=sides,
        )
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "linear_regression_test_slope",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "sigma_residuals": sigma,
        "ncp": _ncp(n=n_out, slope=slope, b0=b0, sx=sx, sigma=sigma),
        "inputs_echo": inputs_echo,
        "citations": [
            "Neter, J., Wasserman, W., and Kutner, M.H. (1983). Applied "
            "Linear Statistical Models, 2nd Ed. Richard D. Irwin.",
        ],
    }


# ---------------------------------------------------------------------------
# Tests for the Difference Between Two Linear Regression Slopes  (Ch 854)
# ---------------------------------------------------------------------------
# Dupont & Plummer (1998) t-based power formula.
#
# Power = T_v[δ√n2 - t_{v,α/2}] + T_v[-δ√n2 - t_{v,α/2}]
# where v = n1+n2-4,  m = n1/n2,  δ = (β2-β1)/σ_R,
#       σ_R² = σ²·[1/(m·σX1²) + 1/σX2²]
# ---------------------------------------------------------------------------


def _slope_diff_power(
    *,
    n1: int,
    n2: int,
    delta: float,
    sigma: float,
    sx1: float,
    sx2: float,
    alpha: float,
    sides: int,
) -> float:
    """Power for the two-slope difference test (Dupont & Plummer 1998)."""
    if n1 + n2 <= 4:
        return 0.0
    from scipy.stats import t as tdist

    v = n1 + n2 - 4
    m = n1 / n2
    sigma_r2 = sigma ** 2 * (1.0 / (m * sx1 ** 2) + 1.0 / sx2 ** 2)
    sigma_r = math.sqrt(sigma_r2)
    if sigma_r == 0.0:
        return 0.0
    delta_std = delta / sigma_r
    alpha_eff = alpha / 2.0 if sides == 2 else alpha
    t_crit = tdist.ppf(1.0 - alpha_eff, v)
    power = (
        tdist.cdf(delta_std * math.sqrt(n2) - t_crit, v)
        + tdist.cdf(-delta_std * math.sqrt(n2) - t_crit, v)
    )
    return float(power)


def tests_difference_two_regression_slopes(
    *,
    slope_diff: float,
    sigma: float,
    sx1: float,
    sx2: float,
    alpha: float = 0.05,
    sides: int = 2,
    n1: int | None = None,
    n2: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power / sample size for testing the difference between two regression slopes.

    Tests H0: β1=β2 using the Dupont & Plummer (1998) t-test formulation.

    Parameters
    ----------
    slope_diff
        Hypothesised difference β2 - β1 under the alternative.
    sigma
        Common residual standard deviation.
    sx1, sx2
        Population SD of X in groups 1 and 2 (divide by n, not n-1).
    alpha
        Type-I error rate.
    sides
        1 or 2.
    n1, n2
        Group sample sizes (supply when solve_for="power", or fix one when
        solving for sample size with equal groups via n1=n2=None).
    power
        Target power (supply when solve_for="n").
    solve_for
        ``"n"`` or ``"power"``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if sx1 <= 0 or sx2 <= 0:
        raise ValueError("sx1 and sx2 must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    inputs_echo: dict[str, Any] = {
        "slope_diff": slope_diff,
        "sigma": sigma,
        "sx1": sx1,
        "sx2": sx2,
        "alpha": alpha,
        "sides": sides,
        "n1": n1,
        "n2": n2,
        "power": power,
    }

    # Infer solve_for
    if solve_for is None:
        if power is None and (n1 is not None or n2 is not None):
            solve_for = "power"
        elif power is not None and n1 is None and n2 is None:
            solve_for = "n"
        else:
            raise ValueError(
                "specify solve_for='n' or solve_for='power' explicitly"
            )

    if solve_for == "power":
        if n1 is None or n2 is None:
            raise ValueError("both n1 and n2 required when solve_for='power'")
        achieved = _slope_diff_power(
            n1=n1, n2=n2, delta=slope_diff,
            sigma=sigma, sx1=sx1, sx2=sx2, alpha=alpha, sides=sides,
        )
        n_out = n1 + n2

    elif solve_for == "n":
        if power is None:
            raise ValueError("power required when solve_for='n'")
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        if slope_diff == 0.0:
            raise ValueError("slope_diff must be non-zero to solve for n")
        # Equal allocation: n1 = n2
        lo, hi = 4, 10
        while hi <= 10_000_000:
            _n = hi // 2
            if _slope_diff_power(
                n1=_n, n2=_n, delta=slope_diff,
                sigma=sigma, sx1=sx1, sx2=sx2, alpha=alpha, sides=sides,
            ) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 10,000,000")
        while lo + 2 < hi:
            mid = ((lo + hi) // 2) & ~1  # keep even so n1=n2
            _n = mid // 2
            if _slope_diff_power(
                n1=_n, n2=_n, delta=slope_diff,
                sigma=sigma, sx1=sx1, sx2=sx2, alpha=alpha, sides=sides,
            ) >= power:
                hi = mid
            else:
                lo = mid
        # hi is the smallest even total N achieving power
        _n = max(hi // 2, 2)
        n1 = n2 = _n
        achieved = _slope_diff_power(
            n1=n1, n2=n2, delta=slope_diff,
            sigma=sigma, sx1=sx1, sx2=sx2, alpha=alpha, sides=sides,
        )
        n_out = n1 + n2

    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "tests_difference_two_regression_slopes",
        "solve_for": solve_for,
        "n": n_out,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Two Linear Regression Slopes.",
            "Dupont, W.D. and Plummer, W.D. Jr. (1998). Power and Sample "
            "Size Calculations for Studies Involving Linear Regression. "
            "Controlled Clinical Trials, 19, 589-601.",
        ],
    }


# ---------------------------------------------------------------------------
# Tests for the Difference Between Two Linear Regression Intercepts (Ch 853)
# ---------------------------------------------------------------------------
# Same Dupont & Plummer (1998) framework but σ_R involves the X means too:
#
# σ_R² = (σ²/m)·[1 + μX1²/σX1² + m·(1 + μX2²/σX2²)]
# Power = T_v[δ√n2 - t_{v,α/2}] + T_v[-δ√n2 - t_{v,α/2}]
# δ = (α2-α1)/σ_R
# ---------------------------------------------------------------------------


def _intercept_diff_power(
    *,
    n1: int,
    n2: int,
    delta: float,
    sigma: float,
    sx1: float,
    sx2: float,
    mx1: float,
    mx2: float,
    alpha: float,
    sides: int,
) -> float:
    """Power for the two-intercept difference test (Dupont & Plummer 1998)."""
    if n1 + n2 <= 4:
        return 0.0
    from scipy.stats import t as tdist

    v = n1 + n2 - 4
    m = n1 / n2
    sigma_r2 = (sigma ** 2 / m) * (
        1.0 + mx1 ** 2 / sx1 ** 2 + m * (1.0 + mx2 ** 2 / sx2 ** 2)
    )
    sigma_r = math.sqrt(sigma_r2)
    if sigma_r == 0.0:
        return 0.0
    delta_std = delta / sigma_r
    alpha_eff = alpha / 2.0 if sides == 2 else alpha
    t_crit = tdist.ppf(1.0 - alpha_eff, v)
    power = (
        tdist.cdf(delta_std * math.sqrt(n2) - t_crit, v)
        + tdist.cdf(-delta_std * math.sqrt(n2) - t_crit, v)
    )
    return float(power)


def tests_difference_two_regression_intercepts(
    *,
    intercept_diff: float,
    sigma: float,
    sx1: float,
    sx2: float,
    mx1: float = 0.0,
    mx2: float = 0.0,
    alpha: float = 0.05,
    sides: int = 2,
    n1: int | None = None,
    n2: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power / sample size for testing the difference between two regression intercepts.

    Tests H0: α1=α2 using Dupont & Plummer (1998).

    Parameters
    ----------
    intercept_diff
        Hypothesised difference α2 - α1 under the alternative.
    sigma
        Common residual standard deviation.
    sx1, sx2
        Population SD of X in groups 1 and 2 (divide by n, not n-1).
    mx1, mx2
        Mean of X in groups 1 and 2.
    alpha
        Type-I error rate.
    sides
        1 or 2.
    n1, n2
        Group sample sizes (supply when solve_for="power").
    power
        Target power (supply when solve_for="n").
    solve_for
        ``"n"`` or ``"power"``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if sx1 <= 0 or sx2 <= 0:
        raise ValueError("sx1 and sx2 must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    inputs_echo: dict[str, Any] = {
        "intercept_diff": intercept_diff,
        "sigma": sigma,
        "sx1": sx1,
        "sx2": sx2,
        "mx1": mx1,
        "mx2": mx2,
        "alpha": alpha,
        "sides": sides,
        "n1": n1,
        "n2": n2,
        "power": power,
    }

    if solve_for is None:
        if power is None and (n1 is not None or n2 is not None):
            solve_for = "power"
        elif power is not None and n1 is None and n2 is None:
            solve_for = "n"
        else:
            raise ValueError(
                "specify solve_for='n' or solve_for='power' explicitly"
            )

    if solve_for == "power":
        if n1 is None or n2 is None:
            raise ValueError("both n1 and n2 required when solve_for='power'")
        achieved = _intercept_diff_power(
            n1=n1, n2=n2, delta=intercept_diff,
            sigma=sigma, sx1=sx1, sx2=sx2, mx1=mx1, mx2=mx2,
            alpha=alpha, sides=sides,
        )
        n_out = n1 + n2

    elif solve_for == "n":
        if power is None:
            raise ValueError("power required when solve_for='n'")
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        if intercept_diff == 0.0:
            raise ValueError("intercept_diff must be non-zero to solve for n")
        lo, hi = 4, 10
        while hi <= 10_000_000:
            _n = hi // 2
            if _intercept_diff_power(
                n1=_n, n2=_n, delta=intercept_diff,
                sigma=sigma, sx1=sx1, sx2=sx2, mx1=mx1, mx2=mx2,
                alpha=alpha, sides=sides,
            ) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 10,000,000")
        while lo + 2 < hi:
            mid = ((lo + hi) // 2) & ~1
            _n = mid // 2
            if _intercept_diff_power(
                n1=_n, n2=_n, delta=intercept_diff,
                sigma=sigma, sx1=sx1, sx2=sx2, mx1=mx1, mx2=mx2,
                alpha=alpha, sides=sides,
            ) >= power:
                hi = mid
            else:
                lo = mid
        _n = max(hi // 2, 2)
        n1 = n2 = _n
        achieved = _intercept_diff_power(
            n1=n1, n2=n2, delta=intercept_diff,
            sigma=sigma, sx1=sx1, sx2=sx2, mx1=mx1, mx2=mx2,
            alpha=alpha, sides=sides,
        )
        n_out = n1 + n2

    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "tests_difference_two_regression_intercepts",
        "solve_for": solve_for,
        "n": n_out,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Two Linear Regression Intercepts.",
            "Dupont, W.D. and Plummer, W.D. Jr. (1998). Power and Sample "
            "Size Calculations for Studies Involving Linear Regression. "
            "Controlled Clinical Trials, 19, 589-601.",
        ],
    }
