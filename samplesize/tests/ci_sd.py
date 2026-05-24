"""Confidence intervals for one standard deviation - sample size calculators.

  - Chapter 640: Confidence Intervals for One Standard Deviation using Standard
    Deviation                                  (`ci_one_sd_using_sd`)
  - Chapter 642: Confidence Intervals for One Standard Deviation using Relative
    Error                                      (`ci_one_sd_relative_error`)
  - Chapter 641: Confidence Intervals for One Standard Deviation with Tolerance
    Probability                                (`ci_one_sd_tolerance`)

All three procedures invert the chi-square sampling distribution of the sample
variance to obtain a confidence interval on the standard deviation scale
(s * sqrt((n - 1) / chi2_q)).  The relative-error formulation expresses the CI
half-width as a proportion of sigma, and the tolerance-probability formulation
inflates the standard deviation by either a chi-square (S is population) or
F (S from a previous sample) factor so the future interval width meets the
target with probability >= tolerance.

References
----------
Hahn, G. J. and Meeker, W. Q. (1991). Statistical Intervals. Wiley.
Greenwood, J. A. and Sandomire, M. M. (1950). 'Sample Size Required for
    Estimating the Standard Deviation as a Per Cent of its True Value.'
    JASA 45(250), 257-260.
Desu, M. M. and Raghavarao, D. (1990). Sample Size Methodology. Academic Press.
Harris, M., Horvitz, D. J., and Mood, A. M. (1948). 'On the Determination of
    Sample Sizes in Designing Experiments.' JASA 43(243), 391-402.
Kupper, L. L. and Hafner, K. B. (1989). 'How Appropriate are Popular Sample
    Size Formulas?' The American Statistician 43, 101-105.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import chi2, f as fdist


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _two_sided_sd_limits(n: int, s: float, alpha: float) -> tuple[float, float]:
    """Lower and upper two-sided limits of the SD CI at level (1 - alpha)."""
    df = n - 1
    lo = s * math.sqrt(df / chi2.ppf(1.0 - alpha / 2.0, df))
    hi = s * math.sqrt(df / chi2.ppf(alpha / 2.0, df))
    return lo, hi


def _one_sided_upper(n: int, s: float, alpha: float) -> float:
    """Upper one-sided 1 - alpha confidence limit."""
    df = n - 1
    return s * math.sqrt(df / chi2.ppf(alpha, df))


def _one_sided_lower(n: int, s: float, alpha: float) -> float:
    """Lower one-sided 1 - alpha confidence limit."""
    df = n - 1
    return s * math.sqrt(df / chi2.ppf(1.0 - alpha, df))


def _two_sided_width(n: int, s: float, alpha: float) -> float:
    lo, hi = _two_sided_sd_limits(n, s, alpha)
    return hi - lo


def _one_sided_distance(n: int, s: float, alpha: float, side: str) -> float:
    if side == "upper":
        return _one_sided_upper(n, s, alpha) - s
    return s - _one_sided_lower(n, s, alpha)


def _solve_n_monotone(predicate, n_min: int = 2, n_max: int = 10_000_000) -> int:
    """Smallest n in [n_min, n_max] with predicate(n) True.

    Predicate is assumed to be monotone non-decreasing in n once True (i.e.
    once a sample size is large enough it stays large enough as n grows).
    """
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


# ---------------------------------------------------------------------------
# Chapter 640: Confidence Intervals for One Standard Deviation using SD
# ---------------------------------------------------------------------------


def ci_one_sd_using_sd(
    *,
    sd: float,
    alpha: float = 0.05,
    sides: int = 2,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    interval_side: str = "upper",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Sample size / interval for a CI on a single SD when sigma = ``sd``.

    Parameters
    ----------
    sd : float
        Planning value of the sample standard deviation (S).  Assumed to equal
        the realised sample SD.
    alpha : float
        Confidence level is ``1 - alpha``.
    sides : int
        ``2`` for a two-sided interval (specify ``width``), ``1`` for a
        one-sided bound (specify ``distance`` plus ``interval_side``).
    width : float, optional
        Two-sided target width (UCL - LCL); required when ``sides == 2`` and
        solving for ``n``.
    distance : float, optional
        One-sided distance from the SD to the CI bound; required when
        ``sides == 1`` and solving for ``n``.
    n : int, optional
        Sample size; required when ``solve_for == 'width'``.
    interval_side : {'upper', 'lower'}
        Which one-sided bound (only used when ``sides == 1``).
    solve_for : {'n', 'width'}
        Defaults to ``'n'`` when a width/distance is given and ``n`` is None.
    """
    if sd <= 0:
        raise ValueError("sd must be > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if sides == 1 and interval_side not in ("upper", "lower"):
        raise ValueError("interval_side must be 'upper' or 'lower'")

    inputs_echo = {
        "sd": sd, "alpha": alpha, "sides": sides,
        "width": width, "distance": distance, "n": n,
        "interval_side": interval_side,
    }

    target = width if sides == 2 else distance
    if solve_for is None:
        solve_for = "n" if n is None else "width"

    if solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                "supply a positive `width` (sides=2) or `distance` (sides=1)"
            )

        def predicate(nn: int) -> bool:
            if sides == 2:
                return _two_sided_width(nn, sd, alpha) <= target
            return _one_sided_distance(nn, sd, alpha, interval_side) <= target

        n_req = _solve_n_monotone(predicate)
    elif solve_for == "width":
        if n is None or n < 2:
            raise ValueError("supply n >= 2 when solving for the width")
        n_req = int(n)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    if sides == 2:
        lo, hi = _two_sided_sd_limits(n_req, sd, alpha)
        achieved_width = hi - lo
        achieved_distance = None
    else:
        if interval_side == "upper":
            hi = _one_sided_upper(n_req, sd, alpha)
            lo = None
        else:
            lo = _one_sided_lower(n_req, sd, alpha)
            hi = None
        achieved_distance = _one_sided_distance(n_req, sd, alpha, interval_side)
        achieved_width = None

    return {
        "method_id": "ci_one_sd_using_sd",
        "solve_for": solve_for,
        "n": n_req,
        "achieved_width": achieved_width,
        "achieved_distance": achieved_distance,
        "lower_limit": lo,
        "upper_limit": hi,
        "sd": sd,
        "inputs_echo": inputs_echo,
        "citations": [
            "Standard Deviation using Standard Deviation.",
            "Hahn, G. J. and Meeker, W. Q. (1991). Statistical Intervals.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 642: Confidence Intervals for One SD using Relative Error
# ---------------------------------------------------------------------------


def _relative_error_confidence(n: int, r: float, sides: int) -> float:
    """Actual confidence level 1 - p1 - p2 for relative error r at size n.

    Uses (n - 1) s^2 / sigma^2 ~ chi2(n - 1):
      p1 = Pr(s > sigma (1 + r)) = 1 - F_{n-1}( (n-1)(1+r)^2 )
      p2 = Pr(s < sigma (1 - r)) =     F_{n-1}( (n-1)(1-r)^2 ) if r < 1 else 0
    For one-sided upper, drop p2; for one-sided lower, drop p1.
    """
    df = n - 1
    p1 = 1.0 - chi2.cdf(df * (1.0 + r) ** 2, df)
    p2 = chi2.cdf(df * (1.0 - r) ** 2, df) if r < 1.0 else 0.0
    if sides == 2:
        return 1.0 - p1 - p2
    if sides == 1:
        # Bound is one-sided; whichever tail you are estimating, the other tail
        # of relative error is not part of the assurance.
        return 1.0 - max(p1, p2)
    raise ValueError("sides must be 1 or 2")


def ci_one_sd_relative_error(
    *,
    relative_error: float | None = None,
    confidence: float | None = None,
    n: int | None = None,
    alpha: float | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """SD CI sample size when the precision is specified as a proportion of sigma.

    Parameters
    ----------
    relative_error : float
        r in (0, 1).  Probability the sample SD lies within r * sigma of sigma
        is the target confidence.
    confidence : float
        Target confidence level (1 - alpha).  If omitted, ``alpha`` is used.
    n : int, optional
        Sample size; supply when solving for the achieved confidence.
    alpha : float, optional
        Equivalent to ``1 - confidence``.
    sides : int
        ``2`` (default) or ``1`` for one-sided bounds.
    solve_for : {'n', 'confidence'}
        Defaults to ``'n'`` when ``n`` is None, else ``'confidence'``.
    """
    if confidence is None and alpha is None:
        raise ValueError("supply either confidence or alpha")
    if confidence is None:
        confidence = 1.0 - float(alpha)  # type: ignore[arg-type]
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    inputs_echo = {
        "relative_error": relative_error, "confidence": confidence,
        "alpha": 1.0 - confidence, "n": n, "sides": sides,
    }

    if solve_for is None:
        solve_for = "n" if n is None else "confidence"

    if solve_for == "n":
        if relative_error is None or not 0.0 < relative_error < 1.0:
            raise ValueError("relative_error must be in (0, 1)")

        def predicate(nn: int) -> bool:
            return _relative_error_confidence(nn, relative_error, sides) >= confidence

        n_req = _solve_n_monotone(predicate, n_min=2)
        achieved = _relative_error_confidence(n_req, relative_error, sides)
    elif solve_for == "confidence":
        if n is None or n < 2:
            raise ValueError("supply n >= 2 when solving for confidence")
        if relative_error is None or not 0.0 < relative_error < 1.0:
            raise ValueError("relative_error must be in (0, 1)")
        n_req = int(n)
        achieved = _relative_error_confidence(n_req, relative_error, sides)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_one_sd_relative_error",
        "solve_for": solve_for,
        "n": n_req,
        "target_confidence": confidence,
        "achieved_confidence": float(achieved),
        "relative_error": relative_error,
        "inputs_echo": inputs_echo,
        "citations": [
            "Standard Deviation using Relative Error.",
            "Greenwood, J. A. and Sandomire, M. M. (1950). JASA 45(250), "
            "257-260.",
            "Desu, M. M. and Raghavarao, D. (1990). Sample Size Methodology.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 641: Confidence Intervals for One SD with Tolerance Probability
# ---------------------------------------------------------------------------


def _adjusted_sd(n: int, sigma: float, tolerance: float, *,
                 sd_source: str, prior_n: int | None) -> float:
    """Adjusted planning SD used to inflate the chi-square CI width.

    sd_source == 'population': s = sigma * sqrt(chi2_{1-gamma, n-1} / (n - 1))
        (Kupper & Hafner 1989 / Hahn & Meeker 1991).
    sd_source == 'previous_sample': s = sigma * sqrt(F_{1-gamma; n-1, m-1})
        (Harris, Horvitz & Mood 1948).
    Here gamma = 1 - tolerance is the upper-tail probability for the future SD.
    """
    df = n - 1
    gamma = 1.0 - tolerance
    if sd_source == "population":
        return sigma * math.sqrt(chi2.ppf(1.0 - gamma, df) / df)
    if sd_source == "previous_sample":
        if prior_n is None or prior_n < 2:
            raise ValueError("prior_n >= 2 required when sd_source='previous_sample'")
        return sigma * math.sqrt(fdist.ppf(1.0 - gamma, df, prior_n - 1))
    raise ValueError("sd_source must be 'population' or 'previous_sample'")


def _tolerance_width(n: int, sigma: float, tolerance: float, alpha: float,
                     sides: int, interval_side: str,
                     sd_source: str, prior_n: int | None) -> tuple[float, float]:
    """Return (width_or_distance, adjusted_sd) for a tolerance-CI design."""
    s = _adjusted_sd(n, sigma, tolerance, sd_source=sd_source, prior_n=prior_n)
    if sides == 2:
        return _two_sided_width(n, s, alpha), s
    return _one_sided_distance(n, s, alpha, interval_side), s


def ci_one_sd_tolerance(
    *,
    sd: float,
    alpha: float = 0.05,
    tolerance_probability: float | None = None,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    interval_side: str = "upper",
    sd_source: str = "population",
    prior_n: int | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Sample size / tolerance probability for a single-SD CI with assurance.

    Parameters
    ----------
    sd : float
        Planning value of sigma (the population SD, or the SD estimated from a
        previous sample of size ``prior_n``).
    alpha : float
        Confidence level is ``1 - alpha``.
    tolerance_probability : float
        Probability the realised interval width / distance is <= target.
        Required when ``solve_for == 'n'``.
    width, distance : float
        Two-sided target width, or one-sided target distance.
    n : int
        Sample size; required for solving for ``tolerance_probability``.
    sides : int
        ``1`` or ``2``.
    interval_side : {'upper', 'lower'}
        Side of the one-sided bound (ignored for two-sided).
    sd_source : {'population', 'previous_sample'}
        Whether ``sd`` is treated as the population SD (Kupper & Hafner) or
        comes from a previous sample of size ``prior_n`` (Harris/Horvitz/Mood).
    prior_n : int, optional
        Required when ``sd_source == 'previous_sample'``.
    solve_for : {'n', 'tolerance_probability'}
    """
    if sd <= 0:
        raise ValueError("sd must be > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if sd_source not in ("population", "previous_sample"):
        raise ValueError("sd_source must be 'population' or 'previous_sample'")

    target = width if sides == 2 else distance
    inputs_echo = {
        "sd": sd, "alpha": alpha, "tolerance_probability": tolerance_probability,
        "width": width, "distance": distance, "n": n, "sides": sides,
        "interval_side": interval_side, "sd_source": sd_source,
        "prior_n": prior_n,
    }

    if solve_for is None:
        if n is None:
            solve_for = "n"
        elif tolerance_probability is None:
            solve_for = "tolerance_probability"
        else:
            solve_for = "tolerance_probability"

    if solve_for == "n":
        if tolerance_probability is None or not 0.0 < tolerance_probability < 1.0:
            raise ValueError("tolerance_probability must be in (0, 1)")
        if target is None or target <= 0:
            raise ValueError("supply a positive `width` (sides=2) or `distance`")

        def predicate(nn: int) -> bool:
            w, _ = _tolerance_width(nn, sd, tolerance_probability, alpha,
                                    sides, interval_side, sd_source, prior_n)
            return w <= target

        n_req = _solve_n_monotone(predicate, n_min=2)
        achieved, s_adj = _tolerance_width(
            n_req, sd, tolerance_probability, alpha, sides, interval_side,
            sd_source, prior_n,
        )
        achieved_tol = float(tolerance_probability)
    elif solve_for == "tolerance_probability":
        if n is None or n < 2:
            raise ValueError("supply n >= 2 when solving for tolerance_probability")
        if target is None or target <= 0:
            raise ValueError("supply a positive `width` (sides=2) or `distance`")
        n_req = int(n)

        # Find the largest tolerance whose adjusted width still meets target.
        lo, hi = 1e-6, 1.0 - 1e-6
        # If target unattainable even at tol -> 0 (gamma=1), give up.
        w_lo, _ = _tolerance_width(n_req, sd, lo, alpha, sides, interval_side,
                                   sd_source, prior_n)
        if w_lo > target:
            achieved_tol = 0.0
            achieved = w_lo
            s_adj = _adjusted_sd(n_req, sd, lo, sd_source=sd_source,
                                 prior_n=prior_n)
        else:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                w_mid, _ = _tolerance_width(n_req, sd, mid, alpha, sides,
                                            interval_side, sd_source, prior_n)
                if w_mid <= target:
                    lo = mid
                else:
                    hi = mid
            achieved_tol = float(lo)
            achieved, s_adj = _tolerance_width(
                n_req, sd, achieved_tol, alpha, sides, interval_side,
                sd_source, prior_n,
            )
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    achieved_width = achieved if sides == 2 else None
    achieved_distance = achieved if sides == 1 else None

    return {
        "method_id": "ci_one_sd_tolerance",
        "solve_for": solve_for,
        "n": n_req,
        "achieved_width": achieved_width,
        "achieved_distance": achieved_distance,
        "achieved_tolerance_probability": achieved_tol,
        "adjusted_sd": float(s_adj),
        "sd": sd,
        "inputs_echo": inputs_echo,
        "citations": [
            "Standard Deviation with Tolerance Probability.",
            "Hahn, G. J. and Meeker, W. Q. (1991). Statistical Intervals.",
            "Harris, M., Horvitz, D. J., and Mood, A. M. (1948). JASA 43, "
            "391-402.",
            "Kupper, L. L. and Hafner, K. B. (1989). The American "
            "Statistician 43, 101-105.",
        ],
    }
