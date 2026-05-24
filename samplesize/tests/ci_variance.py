"""Confidence-interval sample-size routines for variances.


- Ch 651: Confidence Intervals for One Variance using Variance
- Ch 653: Confidence Intervals for One Variance using Relative Error
- Ch 652: Confidence Intervals for One Variance with Tolerance Probability
- Ch 656: Confidence Intervals for the Ratio of Two Variances using Variances
- Ch 657: Confidence Intervals for the Ratio of Two Variances using
  Relative Error

All five solve for the smallest sample size that produces a confidence
interval no wider than a specified target (or — for relative-error
methods — that achieves the requested confidence level for a given
relative error).  Each callable supports `solve_for in {"n", "power"}`,
where "power" here re-uses the registry option to mean "evaluate the
interval at a fixed N" (sometimes called the *Actual Width*
or *Actual Confidence Level*).

Formulas (Zar 1984, Desu & Raghavarao 1990, Greenwood & Sandomire 1950,
Hahn & Meeker 1991):

  one variance, two-sided width given variance s^2:
      W(n) = (n-1) s^2 / chi^2_{alpha/2, n-1}
           - (n-1) s^2 / chi^2_{1-alpha/2, n-1}

  one variance, one-sided distance to limit (upper / lower):
      D_upper(n) = (n-1) s^2 / chi^2_{alpha, n-1}        - s^2
      D_lower(n) = s^2 - (n-1) s^2 / chi^2_{1-alpha, n-1}

  one variance, relative error (two-sided):
      CL(n, r) = 1 - p1 - p2
        p1 = P(chi^2_{n-1} > (n-1)(1+r))
        p2 = P(chi^2_{n-1} <   (n-1)(1-r))

  one variance, tolerance probability — replace s^2 in the width
  formula with
      s^2 = sigma^2 * F_{1-gamma; n-1, m-1}                  (prev sample)
      s^2 = sigma^2 * chi^2_{1-gamma, n-1} / (n-1)           (population V)
  where 1 - gamma is the tolerance probability.

  ratio of variances, two-sided width:
      W(n1, n2) = (s1^2/s2^2) * F_{alpha/2, n2-1, n1-1}
                - (s1^2/s2^2) / F_{alpha/2, n1-1, n2-1}

  ratio of variances, relative error (two-sided):
      CL(n1, n2, r) = G_{n1-1, n2-1}(1+r) - G_{n1-1, n2-1}(1-r)

`F_{q, df1, df2}` denotes the upper-q percentile (scipy `ppf(1-q,...)`).
"""
from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CITATIONS_VARIANCE = [
    "Zar, J. H. (1984). Biostatistical Analysis (2nd ed.). Prentice-Hall.",
]
_CITATIONS_REL = [
    "Desu, M. M. & Raghavarao, D. (1990). Sample Size Methodology. Academic Press.",
    "Greenwood, J. A. & Sandomire, M. M. (1950). JASA 45(250), 257-260.",
]
_CITATIONS_TOL = [
    "Hahn, G. J. & Meeker, W. Q. (1991). Statistical Intervals. Wiley.",
    "Harris, M., Horvitz, D. J. & Mood, A. M. (1948). JASA 43(243), 391-402.",
    "Kupper, L. L. & Hafner, K. B. (1989). The American Statistician.",
]
_CITATIONS_RATIO = [
    "Ostle, B. & Malone, L. C. (1988). Statistics in Research. Iowa State University Press.",
    "Zar, J. H. (1984). Biostatistical Analysis (2nd ed.). Prentice-Hall.",
]
_CITATIONS_RATIO_REL = [
    "Desu, M. M. & Raghavarao, D. (1990). Sample Size Methodology. Academic Press.",
]


def _chi2_ppf(q: float, df: int) -> float:
    from scipy.stats import chi2
    return float(chi2.ppf(q, df))


def _chi2_cdf(x: float, df: int) -> float:
    from scipy.stats import chi2
    return float(chi2.cdf(x, df))


def _f_ppf(q: float, df1: int, df2: int) -> float:
    from scipy.stats import f as fdist
    return float(fdist.ppf(q, df1, df2))


def _f_cdf(x: float, df1: int, df2: int) -> float:
    from scipy.stats import f as fdist
    return float(fdist.cdf(x, df1, df2))


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")


def _check_sides(sides: int) -> None:
    if sides not in (1, 2):
        raise ValueError(f"sides must be 1 or 2; got {sides}")


# ---------------------------------------------------------------------------
# Ch 651 — Confidence Intervals for One Variance using Variance
# ---------------------------------------------------------------------------

def _ci_one_var_width(n: int, variance: float, alpha: float, sides: int,
                      tail: str) -> tuple[float, float, float]:
    """Return (width_or_distance, lower_limit, upper_limit) for given N.

    `tail` is "two", "upper" or "lower" — only used for sides==1 to pick the
    one-sided limit; ignored for sides==2.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    df = n - 1
    s2 = variance
    if sides == 2:
        chi_lo = _chi2_ppf(alpha / 2.0, df)
        chi_hi = _chi2_ppf(1.0 - alpha / 2.0, df)
        lower = df * s2 / chi_hi
        upper = df * s2 / chi_lo
        return upper - lower, lower, upper
    # one-sided
    if tail == "upper":
        chi = _chi2_ppf(alpha, df)
        upper = df * s2 / chi
        return upper - s2, s2, upper
    if tail == "lower":
        chi = _chi2_ppf(1.0 - alpha, df)
        lower = df * s2 / chi
        return s2 - lower, lower, s2
    raise ValueError(f"tail must be 'two', 'upper' or 'lower'; got {tail!r}")


def ci_one_variance_using_variance(
    *,
    variance: float,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    tail: str = "two",
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 1_000_000,
) -> dict[str, Any]:
    """Sample size / actual width for a chi-square-based variance CI.

    Provide either (`variance`, `alpha`, `width`/`distance`) to solve for `n`,
    or (`variance`, `alpha`, `n`) to evaluate the achieved width.
    """
    _check_alpha(alpha)
    _check_sides(sides)
    if variance <= 0:
        raise ValueError("variance must be positive")

    if sides == 2:
        target = width
        target_key = "width"
    else:
        target = distance
        target_key = "distance"
        if tail not in ("upper", "lower"):
            raise ValueError(
                "for sides=1 set tail='upper' or 'lower'"
            )

    if solve_for is None:
        solve_for = "n" if n is None else "power"

    inputs_echo = {
        "variance": variance, "alpha": alpha, "width": width,
        "distance": distance, "n": n, "sides": sides, "tail": tail,
    }

    if solve_for == "power":
        if n is None:
            raise ValueError("solve_for='power' requires n")
        actual, lower, upper = _ci_one_var_width(
            n, variance, alpha, sides, tail if sides == 1 else "two"
        )
        out = {
            "n": int(n),
            f"achieved_{target_key}": actual,
            "lower_limit": lower,
            "upper_limit": upper,
        }
    elif solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                f"solve_for='n' requires positive {target_key}"
            )
        # Width is monotonically decreasing in N. Bracket then bisect.
        lo, hi = n_min, n_min
        while hi <= n_max:
            w, _, _ = _ci_one_var_width(
                hi, variance, alpha, sides, tail if sides == 1 else "two"
            )
            if w <= target:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError(f"could not bracket N within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            w, _, _ = _ci_one_var_width(
                mid, variance, alpha, sides, tail if sides == 1 else "two"
            )
            if w <= target:
                hi = mid
            else:
                lo = mid
        n_req = hi
        actual, lower, upper = _ci_one_var_width(
            n_req, variance, alpha, sides, tail if sides == 1 else "two"
        )
        out = {
            "n": int(n_req),
            f"target_{target_key}": float(target),
            f"achieved_{target_key}": actual,
            "lower_limit": lower,
            "upper_limit": upper,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_one_variance_using_variance",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CITATIONS_VARIANCE,
    }


# ---------------------------------------------------------------------------
# Ch 653 — Confidence Intervals for One Variance using Relative Error
# ---------------------------------------------------------------------------

def _rel_one_var_cl(n: int, r: float, sides: int, tail: str) -> float:
    """Actual confidence level for the relative-error CI on a single variance."""
    if n < 2:
        return 0.0
    df = n - 1
    if sides == 2:
        p1 = 1.0 - _chi2_cdf(df * (1.0 + r), df)
        p2 = _chi2_cdf(df * (1.0 - r), df) if r < 1.0 else 0.0
        return 1.0 - p1 - p2
    if tail == "upper":
        # P(s^2 < sigma^2 (1+r)) — sample variance below the upper bound
        return _chi2_cdf(df * (1.0 + r), df)
    if tail == "lower":
        return 1.0 - _chi2_cdf(df * (1.0 - r), df) if r < 1.0 else 1.0
    raise ValueError(f"tail must be 'upper' or 'lower'; got {tail!r}")


def ci_one_variance_relative_error(
    *,
    relative_error: float,
    alpha: float = 0.05,
    confidence_level: float | None = None,
    n: int | None = None,
    sides: int = 2,
    tail: str = "two",
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 5_000_000,
) -> dict[str, Any]:
    """Sample size or achieved confidence for a relative-error variance CI.

    `confidence_level` overrides `alpha` when supplied (supply as a probability, e.g. 0.95).
    """
    if not 0.0 < relative_error < 1.0:
        raise ValueError("relative_error must be in (0, 1)")
    _check_sides(sides)
    if confidence_level is not None:
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        alpha = 1.0 - confidence_level
    _check_alpha(alpha)
    target_cl = 1.0 - alpha

    if solve_for is None:
        solve_for = "n" if n is None else "power"

    inputs_echo = {
        "relative_error": relative_error, "alpha": alpha,
        "confidence_level": confidence_level, "n": n,
        "sides": sides, "tail": tail,
    }

    if solve_for == "power":
        if n is None:
            raise ValueError("solve_for='power' requires n")
        cl = _rel_one_var_cl(int(n), relative_error, sides, tail)
        out = {
            "n": int(n),
            "achieved_confidence_level": cl,
            "target_confidence_level": target_cl,
        }
    elif solve_for == "n":
        # CL increases monotonically in N. Bracket then bisect.
        lo, hi = n_min, n_min
        while hi <= n_max:
            cl = _rel_one_var_cl(hi, relative_error, sides, tail)
            if cl >= target_cl:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError(f"could not bracket N within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            cl = _rel_one_var_cl(mid, relative_error, sides, tail)
            if cl >= target_cl:
                hi = mid
            else:
                lo = mid
        n_req = hi
        cl_actual = _rel_one_var_cl(n_req, relative_error, sides, tail)
        out = {
            "n": int(n_req),
            "target_confidence_level": target_cl,
            "achieved_confidence_level": cl_actual,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_one_variance_relative_error",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CITATIONS_REL,
    }


# ---------------------------------------------------------------------------
# Ch 652 — Confidence Intervals for One Variance with Tolerance Probability
# ---------------------------------------------------------------------------

def _adjusted_s2(variance: float, n: int, tolerance: float,
                 variance_source: str, m: int | None) -> float:
    """Inflated s^2 driving the width formula."""
    df = n - 1
    if variance_source == "previous_sample":
        if m is None or m < 2:
            raise ValueError(
                "variance_source='previous_sample' requires m_previous >= 2"
            )
        factor = _f_ppf(tolerance, df, m - 1)
    elif variance_source == "population":
        factor = _chi2_ppf(tolerance, df) / df
    else:
        raise ValueError(
            f"variance_source must be 'population' or 'previous_sample'; "
            f"got {variance_source!r}"
        )
    return variance * factor


def _ci_one_var_tol_width(
    n: int, variance: float, alpha: float, sides: int, tail: str,
    tolerance: float, variance_source: str, m_previous: int | None,
) -> tuple[float, float, float, float]:
    """Return (width_or_distance, lower, upper, inflated_s2)."""
    if n < 2:
        raise ValueError("n must be >= 2")
    s2 = _adjusted_s2(variance, n, tolerance, variance_source, m_previous)
    df = n - 1
    if sides == 2:
        chi_lo = _chi2_ppf(alpha / 2.0, df)
        chi_hi = _chi2_ppf(1.0 - alpha / 2.0, df)
        lower = df * s2 / chi_hi
        upper = df * s2 / chi_lo
        return upper - lower, lower, upper, s2
    if tail == "upper":
        chi = _chi2_ppf(alpha, df)
        upper = df * s2 / chi
        return upper - s2, s2, upper, s2
    if tail == "lower":
        chi = _chi2_ppf(1.0 - alpha, df)
        lower = df * s2 / chi
        return s2 - lower, lower, s2, s2
    raise ValueError(f"tail must be 'two', 'upper' or 'lower'; got {tail!r}")


def ci_one_variance_tolerance(
    *,
    variance: float,
    alpha: float = 0.05,
    tolerance_probability: float = 0.90,
    variance_source: str = "population",
    m_previous: int | None = None,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    tail: str = "two",
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 5_000_000,
) -> dict[str, Any]:
    """Sample size / achieved width for a variance CI with assurance.

    `variance_source` is "population" (V is sigma^2) or "previous_sample"
    (V is s^2 from a prior sample of size `m_previous`).
    """
    _check_alpha(alpha)
    _check_sides(sides)
    if variance <= 0:
        raise ValueError("variance must be positive")
    if not 0.0 < tolerance_probability < 1.0:
        raise ValueError("tolerance_probability must be in (0, 1)")

    if sides == 2:
        target = width
        target_key = "width"
    else:
        target = distance
        target_key = "distance"
        if tail not in ("upper", "lower"):
            raise ValueError("for sides=1 set tail='upper' or 'lower'")

    if solve_for is None:
        solve_for = "n" if n is None else "power"

    inputs_echo = {
        "variance": variance, "alpha": alpha,
        "tolerance_probability": tolerance_probability,
        "variance_source": variance_source, "m_previous": m_previous,
        "width": width, "distance": distance, "n": n,
        "sides": sides, "tail": tail,
    }

    if solve_for == "power":
        if n is None:
            raise ValueError("solve_for='power' requires n")
        actual, lower, upper, s2 = _ci_one_var_tol_width(
            int(n), variance, alpha, sides,
            tail if sides == 1 else "two",
            tolerance_probability, variance_source, m_previous,
        )
        out = {
            "n": int(n),
            f"achieved_{target_key}": actual,
            "lower_limit": lower,
            "upper_limit": upper,
            "inflated_variance": s2,
        }
    elif solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                f"solve_for='n' requires positive {target_key}"
            )
        lo, hi = n_min, n_min
        while hi <= n_max:
            w, _, _, _ = _ci_one_var_tol_width(
                hi, variance, alpha, sides,
                tail if sides == 1 else "two",
                tolerance_probability, variance_source, m_previous,
            )
            if w <= target:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError(f"could not bracket N within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            w, _, _, _ = _ci_one_var_tol_width(
                mid, variance, alpha, sides,
                tail if sides == 1 else "two",
                tolerance_probability, variance_source, m_previous,
            )
            if w <= target:
                hi = mid
            else:
                lo = mid
        n_req = hi
        actual, lower, upper, s2 = _ci_one_var_tol_width(
            n_req, variance, alpha, sides,
            tail if sides == 1 else "two",
            tolerance_probability, variance_source, m_previous,
        )
        out = {
            "n": int(n_req),
            f"target_{target_key}": float(target),
            f"achieved_{target_key}": actual,
            "lower_limit": lower,
            "upper_limit": upper,
            "inflated_variance": s2,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_one_variance_tolerance",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CITATIONS_TOL,
    }


# ---------------------------------------------------------------------------
# Ch 656 — Confidence Intervals for the Ratio of Two Variances using Variances
# ---------------------------------------------------------------------------

def _ci_ratio_width(n1: int, n2: int, v1: float, v2: float,
                    alpha: float, sides: int, tail: str
                    ) -> tuple[float, float, float]:
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must be >= 2")
    ratio = v1 / v2
    if sides == 2:
        # F_{alpha/2, df1, df2} = upper-(alpha/2) quantile = ppf(1 - alpha/2)
        F_upper = _f_ppf(1.0 - alpha / 2.0, n2 - 1, n1 - 1)
        F_lower = _f_ppf(1.0 - alpha / 2.0, n1 - 1, n2 - 1)
        upper = ratio * F_upper
        lower = ratio / F_lower
        return upper - lower, lower, upper
    if tail == "upper":
        F_upper = _f_ppf(1.0 - alpha, n2 - 1, n1 - 1)
        upper = ratio * F_upper
        return upper - ratio, ratio, upper
    if tail == "lower":
        F_lower = _f_ppf(1.0 - alpha, n1 - 1, n2 - 1)
        lower = ratio / F_lower
        return ratio - lower, lower, ratio
    raise ValueError(f"tail must be 'two', 'upper' or 'lower'; got {tail!r}")


def ci_ratio_two_variances_using_variances(
    *,
    variance1: float,
    variance2: float,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation_ratio: float = 1.0,
    sides: int = 2,
    tail: str = "two",
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 1_000_000,
) -> dict[str, Any]:
    """F-based CI for sigma1^2 / sigma2^2.

    For `solve_for='n'` the search assumes equal allocation (`n2 = ceil(R *
    n1)`, default R = 1).  Both `n1` and `n2` are returned.
    """
    _check_alpha(alpha)
    _check_sides(sides)
    if variance1 <= 0 or variance2 <= 0:
        raise ValueError("variances must be positive")
    if allocation_ratio <= 0:
        raise ValueError("allocation_ratio (R = n2/n1) must be positive")

    if sides == 2:
        target = width
        target_key = "width"
    else:
        target = distance
        target_key = "distance"
        if tail not in ("upper", "lower"):
            raise ValueError("for sides=1 set tail='upper' or 'lower'")

    if solve_for is None:
        solve_for = "n" if (n1 is None or n2 is None) else "power"

    inputs_echo = {
        "variance1": variance1, "variance2": variance2, "alpha": alpha,
        "width": width, "distance": distance, "n1": n1, "n2": n2,
        "allocation_ratio": allocation_ratio, "sides": sides, "tail": tail,
    }

    if solve_for == "power":
        if n1 is None or n2 is None:
            raise ValueError("solve_for='power' requires both n1 and n2")
        actual, lower, upper = _ci_ratio_width(
            int(n1), int(n2), variance1, variance2, alpha, sides,
            tail if sides == 1 else "two",
        )
        out = {
            "n1": int(n1),
            "n2": int(n2),
            "n": int(n1) + int(n2),
            f"achieved_{target_key}": actual,
            "lower_limit": lower,
            "upper_limit": upper,
        }
    elif solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                f"solve_for='n' requires positive {target_key}"
            )

        def width_at(k: int) -> float:
            n1k = k
            n2k = max(2, math.ceil(allocation_ratio * k))
            w, _, _ = _ci_ratio_width(
                n1k, n2k, variance1, variance2, alpha, sides,
                tail if sides == 1 else "two",
            )
            return w

        lo, hi = n_min, n_min
        while hi <= n_max:
            if width_at(hi) <= target:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError(f"could not bracket N within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if width_at(mid) <= target:
                hi = mid
            else:
                lo = mid
        n1_req = hi
        n2_req = max(2, math.ceil(allocation_ratio * hi))
        actual, lower, upper = _ci_ratio_width(
            n1_req, n2_req, variance1, variance2, alpha, sides,
            tail if sides == 1 else "two",
        )
        out = {
            "n1": int(n1_req),
            "n2": int(n2_req),
            "n": int(n1_req) + int(n2_req),
            f"target_{target_key}": float(target),
            f"achieved_{target_key}": actual,
            "lower_limit": lower,
            "upper_limit": upper,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_ratio_two_variances_using_variances",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CITATIONS_RATIO,
    }


# ---------------------------------------------------------------------------
# Ch 657 — Confidence Intervals for the Ratio of Two Variances using
# Relative Error
# ---------------------------------------------------------------------------

def _rel_ratio_cl(n1: int, n2: int, r: float, sides: int, tail: str) -> float:
    if n1 < 2 or n2 < 2:
        return 0.0
    df1 = n1 - 1
    df2 = n2 - 1
    if sides == 2:
        return _f_cdf(1.0 + r, df1, df2) - _f_cdf(1.0 - r, df1, df2)
    if tail == "upper":
        return _f_cdf(1.0 + r, df1, df2)
    if tail == "lower":
        return 1.0 - _f_cdf(1.0 - r, df1, df2) if r < 1.0 else 1.0
    raise ValueError(f"tail must be 'upper' or 'lower'; got {tail!r}")


def ci_ratio_two_variances_relative_error(
    *,
    relative_error: float,
    alpha: float = 0.05,
    confidence_level: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation_ratio: float = 1.0,
    sides: int = 2,
    tail: str = "two",
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 5_000_000,
) -> dict[str, Any]:
    """Sample sizes for a relative-error CI on sigma1^2 / sigma2^2."""
    if not 0.0 < relative_error < 1.0:
        raise ValueError("relative_error must be in (0, 1)")
    _check_sides(sides)
    if allocation_ratio <= 0:
        raise ValueError("allocation_ratio (R = n2/n1) must be positive")
    if confidence_level is not None:
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        alpha = 1.0 - confidence_level
    _check_alpha(alpha)
    target_cl = 1.0 - alpha

    if solve_for is None:
        solve_for = "n" if (n1 is None or n2 is None) else "power"

    inputs_echo = {
        "relative_error": relative_error, "alpha": alpha,
        "confidence_level": confidence_level,
        "n1": n1, "n2": n2, "allocation_ratio": allocation_ratio,
        "sides": sides, "tail": tail,
    }

    if solve_for == "power":
        if n1 is None or n2 is None:
            raise ValueError("solve_for='power' requires both n1 and n2")
        cl = _rel_ratio_cl(int(n1), int(n2), relative_error, sides, tail)
        out = {
            "n1": int(n1),
            "n2": int(n2),
            "n": int(n1) + int(n2),
            "achieved_confidence_level": cl,
            "target_confidence_level": target_cl,
        }
    elif solve_for == "n":
        def cl_at(k: int) -> float:
            n1k = k
            n2k = max(2, math.ceil(allocation_ratio * k))
            return _rel_ratio_cl(n1k, n2k, relative_error, sides, tail)

        lo, hi = n_min, n_min
        while hi <= n_max:
            if cl_at(hi) >= target_cl:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError(f"could not bracket N within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if cl_at(mid) >= target_cl:
                hi = mid
            else:
                lo = mid
        n1_req = hi
        n2_req = max(2, math.ceil(allocation_ratio * hi))
        cl_actual = _rel_ratio_cl(
            n1_req, n2_req, relative_error, sides, tail
        )
        out = {
            "n1": int(n1_req),
            "n2": int(n2_req),
            "n": int(n1_req) + int(n2_req),
            "target_confidence_level": target_cl,
            "achieved_confidence_level": cl_actual,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_ratio_two_variances_relative_error",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CITATIONS_RATIO_REL,
    }
