"""Confidence intervals for process-capability indices.

  - Chapter 296: Confidence Intervals for Cp   (`ci_cp`)
  - Chapter 297: Confidence Intervals for Cpk  (`ci_cpk`)

Cp is a process-potential capability index defined as
``Cp = (USL - LSL) / (6 * sigma)``.  Because Cp is a monotone transformation
of sigma, the 100(1 - alpha)% two-sided CI inverts the chi-square sampling
distribution of the variance:

    Cp_hat * sqrt( chi2_{n-1, alpha/2}   / (n - 1) )  <=  Cp
    Cp_hat * sqrt( chi2_{n-1, 1-alpha/2} / (n - 1) )  >=  Cp

The two-sided width is therefore

    Width = Cp_hat * ( sqrt(chi2_{n-1, 1-alpha/2}/(n-1))
                     - sqrt(chi2_{n-1, alpha/2}  /(n-1)) ).

Cpk is the centred capability index
``Cpk = min(USL - mu, mu - LSL) / (3 * sigma)``.  It depends on both the
process mean and standard deviation, so its sampling distribution does not
admit a closed-form CI.  Bissell (1990) / Kushler & Hurley
(1992) large-sample normal approximation reported by Mathews (2010):

    Cpk_hat * [ 1 +/- z_{1-alpha/2} * sqrt( (1/n) * (1/(9*Cpk_hat^2) + 1/2) ) ].

The two-sided width is

    Width = 2 * Cpk_hat * z_{1-alpha/2}
            * sqrt( (1/n) * (1/(9*Cpk_hat^2) + 1/2) ).

Both routines solve for the smallest integer ``n`` such that the achieved
width (or one-sided distance) is no larger than the requested target.  An
optional ``relative_error`` parameter expresses the target as a fraction of
the planning capability estimate.

References
----------
Mathews, P. (2010). Sample Size Calculations: Practical Methods for
    Engineers and Scientists. Mathews Malnar and Bailey.
Bissell, A. F. (1990). 'How Reliable is Your Capability Index?' Applied
    Statistics 39, 331-340.
Kushler, R. H. and Hurley, P. (1992). 'Confidence Bounds for Capability
    Indices.' Journal of Quality Technology 24, 188-195.
Kotz, S. and Johnson, N. L. (1993). Process Capability Indices.
    Chapman & Hall.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import chi2, norm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _solve_n_monotone(predicate, n_min: int = 2, n_max: int = 50_000_000) -> int:
    """Smallest integer n in [n_min, n_max] for which ``predicate(n)`` is True.

    The predicate is assumed monotone: once it becomes True it stays True for
    every larger n.  Width-style predicates (achieved width <= target) satisfy
    this for both Cp (chi-square multipliers shrink toward 1) and Cpk
    (1/sqrt(n) shrinkage).
    """
    if n_min < 2:
        n_min = 2
    lo = n_min
    hi = n_min
    while hi <= n_max:
        if predicate(hi):
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket n within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _resolve_target(width: float | None, relative_error: float | None,
                    estimate: float, sides: int, distance: float | None) -> float:
    """Return the absolute precision target on the index scale."""
    if sides == 2:
        if width is not None and width > 0:
            return float(width)
        if relative_error is not None and relative_error > 0:
            return float(relative_error) * abs(estimate)
        raise ValueError(
            "supply positive `width` or `relative_error` for sides=2"
        )
    # sides == 1
    if distance is not None and distance > 0:
        return float(distance)
    if relative_error is not None and relative_error > 0:
        return float(relative_error) * abs(estimate)
    raise ValueError(
        "supply positive `distance` or `relative_error` for sides=1"
    )


# ---------------------------------------------------------------------------
# Chapter 296: Confidence Intervals for Cp
# ---------------------------------------------------------------------------


def _cp_limits(n: int, cp: float, alpha: float, sides: int,
               interval_side: str) -> tuple[float | None, float | None]:
    df = n - 1
    if sides == 2:
        lo = cp * math.sqrt(chi2.ppf(alpha / 2.0, df) / df)
        hi = cp * math.sqrt(chi2.ppf(1.0 - alpha / 2.0, df) / df)
        return lo, hi
    if interval_side == "upper":
        return None, cp * math.sqrt(chi2.ppf(1.0 - alpha, df) / df)
    return cp * math.sqrt(chi2.ppf(alpha, df) / df), None


def _cp_precision(n: int, cp: float, alpha: float, sides: int,
                  interval_side: str) -> float:
    lo, hi = _cp_limits(n, cp, alpha, sides, interval_side)
    if sides == 2:
        return hi - lo  # type: ignore[operator]
    if interval_side == "upper":
        return hi - cp  # type: ignore[operator]
    return cp - lo  # type: ignore[operator]


def ci_cp(
    *,
    cp: float,
    alpha: float = 0.05,
    sides: int = 2,
    width: float | None = None,
    distance: float | None = None,
    relative_error: float | None = None,
    n: int | None = None,
    interval_side: str = "upper",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Sample size or achieved width for a Cp confidence interval.

    Parameters
    ----------
    cp : float
        Planning estimate of Cp = (USL - LSL) / (6 sigma).  Assumed equal to
        the realised sample Cp_hat for the design calculation.
    alpha : float
        Confidence level is ``1 - alpha``.
    sides : int
        ``2`` for a two-sided interval (specify ``width`` or
        ``relative_error``), ``1`` for a one-sided bound (specify
        ``distance`` or ``relative_error`` plus ``interval_side``).
    width : float, optional
        Two-sided target width (upper - lower).
    distance : float, optional
        One-sided target distance from Cp_hat to the bound.
    relative_error : float, optional
        Alternative precision target expressed as a fraction of ``cp``
        (target = relative_error * cp).
    n : int, optional
        Sample size; supply when solving for ``width``.
    interval_side : {'upper', 'lower'}
        Side of the one-sided bound (ignored when ``sides == 2``).
    solve_for : {'n', 'width'}
        Defaults to ``'n'`` when ``n`` is None, else ``'width'``.
    """
    if cp <= 0:
        raise ValueError("cp must be > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if sides == 1 and interval_side not in ("upper", "lower"):
        raise ValueError("interval_side must be 'upper' or 'lower'")

    inputs_echo = {
        "cp": cp, "alpha": alpha, "sides": sides,
        "width": width, "distance": distance,
        "relative_error": relative_error,
        "n": n, "interval_side": interval_side,
    }

    if solve_for is None:
        solve_for = "n" if n is None else "width"

    if solve_for == "n":
        target = _resolve_target(width, relative_error, cp, sides, distance)

        def predicate(nn: int) -> bool:
            return _cp_precision(nn, cp, alpha, sides, interval_side) <= target

        n_req = _solve_n_monotone(predicate, n_min=2)
        target_value = target
    elif solve_for == "width":
        if n is None or n < 2:
            raise ValueError("supply n >= 2 when solving for width")
        n_req = int(n)
        target_value = None
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    lo, hi = _cp_limits(n_req, cp, alpha, sides, interval_side)
    if sides == 2:
        achieved_width = float(hi - lo)  # type: ignore[operator]
        achieved_distance = None
    else:
        achieved_width = None
        if interval_side == "upper":
            achieved_distance = float(hi - cp)  # type: ignore[operator]
        else:
            achieved_distance = float(cp - lo)  # type: ignore[operator]

    return {
        "method_id": "ci_cp",
        "solve_for": solve_for,
        "n": n_req,
        "target_width": float(target_value) if target_value is not None and sides == 2 else None,
        "target_distance": float(target_value) if target_value is not None and sides == 1 else None,
        "achieved_width": achieved_width,
        "achieved_distance": achieved_distance,
        "lower_limit": float(lo) if lo is not None else None,
        "upper_limit": float(hi) if hi is not None else None,
        "cp": float(cp),
        "inputs_echo": inputs_echo,
        "citations": [
            "Mathews, P. (2010). Sample Size Calculations: Practical Methods "
            "for Engineers and Scientists.",
            "Kotz, S. and Johnson, N. L. (1993). Process Capability Indices.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 297: Confidence Intervals for Cpk
# ---------------------------------------------------------------------------


def _cpk_se(n: int, cpk: float) -> float:
    """Bissell/Kushler-Hurley large-sample SE multiplier for Cpk."""
    return math.sqrt((1.0 / n) * (1.0 / (9.0 * cpk * cpk) + 0.5))


def _cpk_limits(n: int, cpk: float, alpha: float, sides: int,
                interval_side: str) -> tuple[float | None, float | None]:
    se = _cpk_se(n, cpk)
    if sides == 2:
        z = norm.ppf(1.0 - alpha / 2.0)
        return cpk * (1.0 - z * se), cpk * (1.0 + z * se)
    z = norm.ppf(1.0 - alpha)
    if interval_side == "upper":
        return None, cpk * (1.0 + z * se)
    return cpk * (1.0 - z * se), None


def _cpk_precision(n: int, cpk: float, alpha: float, sides: int,
                   interval_side: str) -> float:
    lo, hi = _cpk_limits(n, cpk, alpha, sides, interval_side)
    if sides == 2:
        return hi - lo  # type: ignore[operator]
    if interval_side == "upper":
        return hi - cpk  # type: ignore[operator]
    return cpk - lo  # type: ignore[operator]


def ci_cpk(
    *,
    cpk: float,
    alpha: float = 0.05,
    sides: int = 2,
    width: float | None = None,
    distance: float | None = None,
    relative_error: float | None = None,
    n: int | None = None,
    interval_side: str = "upper",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Sample size or achieved width for a Cpk confidence interval.

    Uses the Bissell (1990) / Kushler & Hurley (1992) large-sample normal
    approximation reported by Mathews (2010).

    Parameters
    ----------
    cpk : float
        Planning estimate of Cpk = min(USL - mu, mu - LSL) / (3 sigma).
        Must be > 0 for the approximation to apply.
    alpha : float
        Confidence level is ``1 - alpha``.
    sides : int
        ``2`` for a two-sided interval, ``1`` for a one-sided bound.
    width : float, optional
        Two-sided target width.
    distance : float, optional
        One-sided target distance from Cpk_hat to the bound.
    relative_error : float, optional
        Alternative precision target as a fraction of ``cpk``.
    n : int, optional
        Sample size; supply when solving for ``width``.
    interval_side : {'upper', 'lower'}
        Side of the one-sided bound.
    solve_for : {'n', 'width'}
        Defaults to ``'n'`` when ``n`` is None, else ``'width'``.
    """
    if cpk <= 0:
        raise ValueError("cpk must be > 0 (large-sample approximation)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if sides == 1 and interval_side not in ("upper", "lower"):
        raise ValueError("interval_side must be 'upper' or 'lower'")

    inputs_echo = {
        "cpk": cpk, "alpha": alpha, "sides": sides,
        "width": width, "distance": distance,
        "relative_error": relative_error,
        "n": n, "interval_side": interval_side,
    }

    if solve_for is None:
        solve_for = "n" if n is None else "width"

    if solve_for == "n":
        target = _resolve_target(width, relative_error, cpk, sides, distance)

        def predicate(nn: int) -> bool:
            return _cpk_precision(nn, cpk, alpha, sides, interval_side) <= target

        n_req = _solve_n_monotone(predicate, n_min=2)
        target_value = target
    elif solve_for == "width":
        if n is None or n < 2:
            raise ValueError("supply n >= 2 when solving for width")
        n_req = int(n)
        target_value = None
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    lo, hi = _cpk_limits(n_req, cpk, alpha, sides, interval_side)
    if sides == 2:
        achieved_width = float(hi - lo)  # type: ignore[operator]
        achieved_distance = None
    else:
        achieved_width = None
        if interval_side == "upper":
            achieved_distance = float(hi - cpk)  # type: ignore[operator]
        else:
            achieved_distance = float(cpk - lo)  # type: ignore[operator]

    return {
        "method_id": "ci_cpk",
        "solve_for": solve_for,
        "n": n_req,
        "target_width": float(target_value) if target_value is not None and sides == 2 else None,
        "target_distance": float(target_value) if target_value is not None and sides == 1 else None,
        "achieved_width": achieved_width,
        "achieved_distance": achieved_distance,
        "lower_limit": float(lo) if lo is not None else None,
        "upper_limit": float(hi) if hi is not None else None,
        "cpk": float(cpk),
        "inputs_echo": inputs_echo,
        "citations": [
            "Mathews, P. (2010). Sample Size Calculations: Practical Methods "
            "for Engineers and Scientists.",
            "Bissell, A. F. (1990). 'How Reliable is Your Capability Index?' "
            "Applied Statistics 39, 331-340.",
            "Kushler, R. H. and Hurley, P. (1992). 'Confidence Bounds for "
            "Capability Indices.' Journal of Quality Technology 24, 188-195.",
        ],
    }
