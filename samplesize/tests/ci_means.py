"""Confidence-interval sample-size routines for means.


- Ch 420: Confidence Intervals for One Mean
- Ch 496: Confidence Intervals for Paired Means
- Ch 471: Confidence Intervals for the Difference Between Two Means
- Ch 421: Confidence Intervals for One Mean with Tolerance Probability
- Ch 497: Confidence Intervals for Paired Means with Tolerance Probability
- Ch 472: Confidence Intervals for the Difference Between Two Means
         with Tolerance Probability

All methods solve for the smallest sample size that produces a CI half-width
(distance from mean to limit) no greater than a target value D, or evaluate
the achieved half-width at a fixed N.

For one mean and paired means:
  Known sigma:   D = z_{1-alpha/sides} * sigma / sqrt(n)
  Unknown sigma: D = t_{1-alpha/sides, n-1} * sigma / sqrt(n)  (iterative)

For difference of two means (Welch-Satterthwaite, unequal variance):
  D = t_{1-alpha/sides, nu} * sqrt(s1^2/n1 + s2^2/n2)

  where nu = (s1^2/n1 + s2^2/n2)^2 /
             (s1^4/(n1^2*(n1-1)) + s2^4/(n2^2*(n2-1)))

For equal variance (pooled):
  D = t_{1-alpha/sides, n1+n2-2} * sp * sqrt(1/n1 + 1/n2)

  where sp = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import chi2 as _chi2
from scipy.stats import f as _fdist

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Chapter 420 / 496: CI for One Mean (and Paired Means — same mechanics)
# ---------------------------------------------------------------------------

def _ci_one_mean_width(*, n: int, sigma: float, alpha: float,
                       sides: int, known_sigma: bool) -> float:
    """Achieved half-width at given n."""
    if n < 2:
        return float("inf")
    k = 1 if sides == 1 else 2
    if known_sigma:
        z = D.norm_ppf(1.0 - alpha / k)
        return z * sigma / math.sqrt(n)
    else:
        t = D.t_ppf(1.0 - alpha / k, df=n - 1)
        return t * sigma / math.sqrt(n)


def _ci_one_mean_n(*, distance: float, sigma: float, alpha: float,
                   sides: int, known_sigma: bool) -> tuple[int, float]:
    """Smallest n such that achieved half-width <= distance."""
    if distance <= 0:
        raise ValueError("distance must be > 0")
    # Known sigma: closed-form starting point, then verify
    k = 1 if sides == 1 else 2
    if known_sigma:
        z = D.norm_ppf(1.0 - alpha / k)
        n = math.ceil((z * sigma / distance) ** 2)
    else:
        # Start with z approximation, then iterate upward with t
        z = D.norm_ppf(1.0 - alpha / k)
        n = max(2, math.ceil((z * sigma / distance) ** 2))
    # Iterate upward until constraint satisfied (t-dist needs this)
    while _ci_one_mean_width(n=n, sigma=sigma, alpha=alpha,
                             sides=sides, known_sigma=known_sigma) > distance:
        n += 1
    achieved = _ci_one_mean_width(n=n, sigma=sigma, alpha=alpha,
                                  sides=sides, known_sigma=known_sigma)
    return n, achieved


def ci_one_mean(
    *,
    sigma: float,
    distance: float | None = None,
    n: int | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    known_sigma: bool = False,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for one mean.

    Parameters
    ----------
    sigma
        Standard deviation (known or estimated).
    distance
        Target half-width D (distance from mean to limit). Required when
        solve_for='n'.
    n
        Sample size. Required when solve_for='width'.
    alpha
        Type I error rate.
    sides
        1 (one-sided) or 2 (two-sided).
    known_sigma
        If True, use z-distribution; otherwise use t-distribution.
    solve_for
        'n' (default when distance supplied) or 'width'.
    """
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    have_distance = distance is not None
    have_n = n is not None
    if not (have_distance or have_n):
        raise ValueError("supply at least one of (distance, n)")
    if solve_for is None:
        solve_for = "n" if not have_n else "width"

    inputs_echo = dict(sigma=sigma, distance=distance, n=n, alpha=alpha,
                       sides=sides, known_sigma=known_sigma)

    if solve_for == "n":
        if distance is None:
            raise ValueError("distance is required when solve_for='n'")
        n_out, achieved_width = _ci_one_mean_n(
            distance=distance, sigma=sigma, alpha=alpha,
            sides=sides, known_sigma=known_sigma,
        )
    elif solve_for == "width":
        if n is None:
            raise ValueError("n is required when solve_for='width'")
        n_out = n
        achieved_width = _ci_one_mean_width(
            n=n, sigma=sigma, alpha=alpha, sides=sides, known_sigma=known_sigma,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "ci_one_mean",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": None,
        "achieved_width": achieved_width,
        "inputs_echo": inputs_echo,
        "citations": [
            "Hahn, G.J. and Meeker, W.Q. (1991). Statistical Intervals. "
            "John Wiley & Sons. New York.",
        ],
    }


def ci_paired_means(
    *,
    sigma_diff: float,
    distance: float | None = None,
    n: int | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    known_sigma: bool = False,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for paired means.

    Uses the same mechanics as ci_one_mean but operates on the standard
    deviation of paired differences.

    Parameters
    ----------
    sigma_diff
        Standard deviation of paired differences.
    distance
        Target half-width D.
    n
        Number of pairs. Required when solve_for='width'.
    alpha
        Type I error rate.
    sides
        1 or 2.
    known_sigma
        If True, use z-distribution; otherwise use t-distribution.
    solve_for
        'n' or 'width'.
    """
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if sigma_diff <= 0:
        raise ValueError("sigma_diff must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    have_distance = distance is not None
    have_n = n is not None
    if not (have_distance or have_n):
        raise ValueError("supply at least one of (distance, n)")
    if solve_for is None:
        solve_for = "n" if not have_n else "width"

    inputs_echo = dict(sigma_diff=sigma_diff, distance=distance, n=n,
                       alpha=alpha, sides=sides, known_sigma=known_sigma)

    if solve_for == "n":
        if distance is None:
            raise ValueError("distance is required when solve_for='n'")
        n_out, achieved_width = _ci_one_mean_n(
            distance=distance, sigma=sigma_diff, alpha=alpha,
            sides=sides, known_sigma=known_sigma,
        )
    elif solve_for == "width":
        if n is None:
            raise ValueError("n is required when solve_for='width'")
        n_out = n
        achieved_width = _ci_one_mean_width(
            n=n, sigma=sigma_diff, alpha=alpha,
            sides=sides, known_sigma=known_sigma,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "ci_paired_means",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": None,
        "achieved_width": achieved_width,
        "inputs_echo": inputs_echo,
        "citations": [
            "Hahn, G.J. and Meeker, W.Q. (1991). Statistical Intervals. "
            "John Wiley & Sons. New York.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 471: CI for Difference Between Two Means
# ---------------------------------------------------------------------------

def _welch_df(s1: float, s2: float, n1: int, n2: int) -> float:
    """Welch-Satterthwaite degrees of freedom."""
    v1 = s1 * s1 / n1
    v2 = s2 * s2 / n2
    num = (v1 + v2) ** 2
    den = v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1)
    if den <= 0:
        return float("inf")
    return num / den


def _pooled_sd(s1: float, s2: float, n1: int, n2: int) -> float:
    return math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))


def _ci_two_means_width(
    *, n1: int, n2: int, s1: float, s2: float,
    alpha: float, sides: int, equal_var: bool,
) -> float:
    """Achieved half-width at given n1, n2."""
    if n1 < 2 or n2 < 2:
        return float("inf")
    k = 1 if sides == 1 else 2
    if equal_var:
        df = n1 + n2 - 2
        sp = _pooled_sd(s1, s2, n1, n2)
        t = D.t_ppf(1.0 - alpha / k, df=df)
        return t * sp * math.sqrt(1.0 / n1 + 1.0 / n2)
    else:
        nu = _welch_df(s1, s2, n1, n2)
        t = D.t_ppf(1.0 - alpha / k, df=nu)
        return t * math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)


def _ci_two_means_n_equal(
    *, distance: float, s1: float, s2: float,
    alpha: float, sides: int, equal_var: bool,
) -> tuple[int, int, float]:
    """Find equal n1 = n2 such that achieved half-width <= distance."""
    k = 1 if sides == 1 else 2
    # Starting z approximation
    z = D.norm_ppf(1.0 - alpha / k)
    se2 = (s1 ** 2 + s2 ** 2)  # for equal n: sqrt((s1^2+s2^2)/n)
    n = max(2, math.ceil((z / distance) ** 2 * se2))
    while _ci_two_means_width(n1=n, n2=n, s1=s1, s2=s2,
                               alpha=alpha, sides=sides,
                               equal_var=equal_var) > distance:
        n += 1
    achieved = _ci_two_means_width(n1=n, n2=n, s1=s1, s2=s2,
                                    alpha=alpha, sides=sides,
                                    equal_var=equal_var)
    return n, n, achieved


def ci_difference_two_means(
    *,
    s1: float,
    s2: float | None = None,
    distance: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    n: int | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    equal_var: bool = False,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for difference of two means.

    Parameters
    ----------
    s1
        Standard deviation of group 1.
    s2
        Standard deviation of group 2. Defaults to s1 (equal variances case).
    distance
        Target half-width D. Required when solve_for='n'.
    n1, n2
        Per-group sample sizes. When neither is supplied, equal allocation
        is used. When one is fixed, the other is solved.
    n
        Total sample size (used when solve_for='width' with equal n1=n2).
    alpha
        Type I error rate.
    sides
        1 or 2.
    equal_var
        If True, use pooled-variance t-interval; otherwise Welch-Satterthwaite.
    solve_for
        'n' (default) or 'width'.
    """
    if s2 is None:
        s2 = s1
    if s1 <= 0 or s2 <= 0:
        raise ValueError("s1, s2 must be > 0")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    inputs_echo = dict(s1=s1, s2=s2, distance=distance, n1=n1, n2=n2,
                       alpha=alpha, sides=sides, equal_var=equal_var)

    # Determine solve mode
    have_distance = distance is not None
    have_n1 = n1 is not None
    have_n2 = n2 is not None
    have_n = n is not None
    if solve_for is None:
        # If both n1 and n2 are given (or total n given) and no distance, solve for width
        if (have_n1 and have_n2) or have_n:
            solve_for = "width"
        elif have_distance:
            solve_for = "n"
        else:
            raise ValueError("supply either (distance) to solve for n, "
                             "or (n1,n2 or n) to solve for width")

    if solve_for == "n":
        if distance is None:
            raise ValueError("distance is required when solve_for='n'")
        if have_n1 and not have_n2:
            # Fix n1, solve for n2
            _n1 = n1
            _n2 = 2
            while _ci_two_means_width(n1=_n1, n2=_n2, s1=s1, s2=s2,
                                       alpha=alpha, sides=sides,
                                       equal_var=equal_var) > distance:
                _n2 += 1
            achieved = _ci_two_means_width(n1=_n1, n2=_n2, s1=s1, s2=s2,
                                            alpha=alpha, sides=sides,
                                            equal_var=equal_var)
        elif have_n2 and not have_n1:
            # Fix n2, solve for n1
            _n2 = n2
            _n1 = 2
            while _ci_two_means_width(n1=_n1, n2=_n2, s1=s1, s2=s2,
                                       alpha=alpha, sides=sides,
                                       equal_var=equal_var) > distance:
                _n1 += 1
            achieved = _ci_two_means_width(n1=_n1, n2=_n2, s1=s1, s2=s2,
                                            alpha=alpha, sides=sides,
                                            equal_var=equal_var)
        else:
            # Equal allocation
            _n1, _n2, achieved = _ci_two_means_n_equal(
                distance=distance, s1=s1, s2=s2,
                alpha=alpha, sides=sides, equal_var=equal_var,
            )
        n_total = _n1 + _n2

    elif solve_for == "width":
        # Evaluate achieved width
        if have_n1 and have_n2:
            _n1, _n2 = n1, n2
        elif have_n:
            _n1 = n // 2
            _n2 = n - _n1
        else:
            raise ValueError("supply n1+n2 or total n when solve_for='width'")
        achieved = _ci_two_means_width(n1=_n1, n2=_n2, s1=s1, s2=s2,
                                        alpha=alpha, sides=sides,
                                        equal_var=equal_var)
        n_total = _n1 + _n2
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "ci_difference_two_means",
        "solve_for": solve_for,
        "n": n_total,
        "n1": _n1,
        "n2": _n2,
        "achieved_power": None,
        "achieved_width": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Between Two Means",
            "Ostle, B. and Malone, L.C. (1988). Statistics in Research. "
            "Iowa State University Press.",
            "Zar, J.H. (1984). Biostatistical Analysis, 2nd ed. Prentice-Hall.",
        ],
    }


# ---------------------------------------------------------------------------
# Tolerance-probability helpers (shared by Ch 421, 497, 472)
# ---------------------------------------------------------------------------

def _tol_adjusted_half_width_one_mean(
    n: int, sigma: float, alpha: float, sides: int,
    tolerance_probability: float,
    sd_source: str, prior_n: int | None,
) -> float:
    """Half-width of the CI after adjusting for tolerance probability.

    For population SD (Kupper & Hafner 1989):
        D_adj = t_{1-α/k, n-1} * sigma * sqrt(chi2_{tol, n-1} / (n-1)) / sqrt(n)

    For SD from a previous sample of size m (Harris, Horvitz & Mood 1948):
        D_adj = t_{1-α/k, n-1} * sigma * sqrt(F_{tol; n-1, m-1}) / sqrt(n)
    """
    if n < 2:
        return float("inf")
    k = 1 if sides == 1 else 2
    df = n - 1
    t_val = D.t_ppf(1.0 - alpha / k, df=df)
    if sd_source == "population":
        adj = math.sqrt(_chi2.ppf(tolerance_probability, df) / df)
    else:
        if prior_n is None or prior_n < 2:
            raise ValueError("prior_n >= 2 required when sd_source='previous_sample'")
        adj = math.sqrt(_fdist.ppf(tolerance_probability, df, prior_n - 1))
    return t_val * sigma * adj / math.sqrt(n)


def _tol_adjusted_half_width_two_means(
    n: int, sigma: float, alpha: float, sides: int,
    tolerance_probability: float,
    sd_source: str, prior_n_total: int | None,
) -> float:
    """Adjusted half-width for equal-n two-sample case (pooled SD).

    df = 2n - 2.  For population SD:
        D_adj = t_{1-α/k, 2n-2} * sigma * sqrt(2/n) * sqrt(chi2_{tol, 2n-2}/(2n-2))

    For SD from previous samples with total size m (Harris, Horvitz & Mood 1948):
        D_adj = t_{1-α/k, 2n-2} * sigma * sqrt(2/n) * sqrt(F_{tol; 2n-2, m-2})
    """
    if n < 2:
        return float("inf")
    k = 1 if sides == 1 else 2
    df = 2 * n - 2
    t_val = D.t_ppf(1.0 - alpha / k, df=df)
    if sd_source == "population":
        adj = math.sqrt(_chi2.ppf(tolerance_probability, df) / df)
    else:
        if prior_n_total is None or prior_n_total < 3:
            raise ValueError(
                "prior_n_total >= 3 (m1+m2) required when sd_source='previous_sample'"
            )
        adj = math.sqrt(_fdist.ppf(tolerance_probability, df, prior_n_total - 2))
    return t_val * sigma * math.sqrt(2.0 / n) * adj


# ---------------------------------------------------------------------------
# Chapter 421: CI for One Mean with Tolerance Probability
# ---------------------------------------------------------------------------


def ci_one_mean_tolerance(
    *,
    sigma: float,
    distance: float | None = None,
    n: int | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    tolerance_probability: float | None = None,
    sd_source: str = "population",
    prior_n: int | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for one mean with tolerance probability.

    Parameters
    ----------
    sigma
        Standard deviation (population or estimated from a previous sample).
    distance
        Target half-width D. Required when solve_for='n'.
    n
        Sample size. Required when solve_for='width'.
    alpha
        Type I error rate (confidence level = 1 - alpha).
    sides
        1 (one-sided) or 2 (two-sided).
    tolerance_probability
        Probability gamma that the future CI half-width <= distance. In (0, 1).
    sd_source
        'population' — sigma is treated as the true population SD
        (Kupper & Hafner 1989 / Hahn & Meeker 1991).
        'previous_sample' — sigma is estimated from a previous sample of
        size prior_n (Harris, Horvitz & Mood 1948).
    prior_n
        Size of the previous sample; required when sd_source='previous_sample'.
    solve_for
        'n' (default when distance supplied) or 'width'.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if sd_source not in ("population", "previous_sample"):
        raise ValueError("sd_source must be 'population' or 'previous_sample'")

    have_distance = distance is not None
    have_n = n is not None
    if not (have_distance or have_n):
        raise ValueError("supply at least one of (distance, n)")
    if solve_for is None:
        solve_for = "n" if not have_n else "width"

    inputs_echo = dict(
        sigma=sigma, distance=distance, n=n, alpha=alpha, sides=sides,
        tolerance_probability=tolerance_probability,
        sd_source=sd_source, prior_n=prior_n,
    )

    if solve_for == "n":
        if distance is None:
            raise ValueError("distance is required when solve_for='n'")
        if tolerance_probability is None or not 0 < tolerance_probability < 1:
            raise ValueError("tolerance_probability must be in (0, 1)")
        # Start at z-based estimate, iterate upward
        k = 1 if sides == 1 else 2
        z = D.norm_ppf(1.0 - alpha / k)
        n_out = max(2, math.ceil((z * sigma / distance) ** 2))
        while _tol_adjusted_half_width_one_mean(
            n_out, sigma, alpha, sides, tolerance_probability, sd_source, prior_n
        ) > distance:
            n_out += 1
        achieved_width = _tol_adjusted_half_width_one_mean(
            n_out, sigma, alpha, sides, tolerance_probability, sd_source, prior_n
        )
        achieved_tol = float(tolerance_probability)
    elif solve_for == "width":
        if n is None:
            raise ValueError("n is required when solve_for='width'")
        if tolerance_probability is None or not 0 < tolerance_probability < 1:
            raise ValueError("tolerance_probability must be in (0, 1)")
        n_out = n
        achieved_width = _tol_adjusted_half_width_one_mean(
            n_out, sigma, alpha, sides, tolerance_probability, sd_source, prior_n
        )
        achieved_tol = float(tolerance_probability)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "ci_one_mean_tolerance",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": None,
        "achieved_width": achieved_width,
        "tolerance_probability": float(tolerance_probability),
        "achieved_tolerance_probability": achieved_tol,
        "inputs_echo": inputs_echo,
        "citations": [
            "with Tolerance Probability",
            "Hahn, G. J. and Meeker, W. Q. (1991). Statistical Intervals. "
            "John Wiley & Sons. New York.",
            "Harris, M., Horvitz, D. J., and Mood, A. M. (1948). JASA 43, "
            "391-402.",
            "Kupper, L. L. and Hafner, K. B. (1989). The American "
            "Statistician 43, 101-105.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 497: CI for Paired Means with Tolerance Probability
# ---------------------------------------------------------------------------


def ci_paired_means_tolerance(
    *,
    sigma_diff: float,
    distance: float | None = None,
    n: int | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    tolerance_probability: float | None = None,
    sd_source: str = "population",
    prior_n: int | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for paired means with tolerance probability.

    Uses the same mechanics as ci_one_mean_tolerance but operates on the
    standard deviation of paired differences.

    Parameters
    ----------
    sigma_diff
        Standard deviation of paired differences.
    distance
        Target half-width D. Required when solve_for='n'.
    n
        Number of pairs. Required when solve_for='width'.
    alpha
        Type I error rate.
    sides
        1 or 2.
    tolerance_probability
        Probability that the future CI half-width <= distance.
    sd_source
        'population' or 'previous_sample'.
    prior_n
        Size of the previous paired sample (required when sd_source='previous_sample').
    solve_for
        'n' or 'width'.
    """
    if sigma_diff <= 0:
        raise ValueError("sigma_diff must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if sd_source not in ("population", "previous_sample"):
        raise ValueError("sd_source must be 'population' or 'previous_sample'")

    have_distance = distance is not None
    have_n = n is not None
    if not (have_distance or have_n):
        raise ValueError("supply at least one of (distance, n)")
    if solve_for is None:
        solve_for = "n" if not have_n else "width"

    inputs_echo = dict(
        sigma_diff=sigma_diff, distance=distance, n=n, alpha=alpha, sides=sides,
        tolerance_probability=tolerance_probability,
        sd_source=sd_source, prior_n=prior_n,
    )

    if solve_for == "n":
        if distance is None:
            raise ValueError("distance is required when solve_for='n'")
        if tolerance_probability is None or not 0 < tolerance_probability < 1:
            raise ValueError("tolerance_probability must be in (0, 1)")
        k = 1 if sides == 1 else 2
        z = D.norm_ppf(1.0 - alpha / k)
        n_out = max(2, math.ceil((z * sigma_diff / distance) ** 2))
        while _tol_adjusted_half_width_one_mean(
            n_out, sigma_diff, alpha, sides, tolerance_probability, sd_source, prior_n
        ) > distance:
            n_out += 1
        achieved_width = _tol_adjusted_half_width_one_mean(
            n_out, sigma_diff, alpha, sides, tolerance_probability, sd_source, prior_n
        )
        achieved_tol = float(tolerance_probability)
    elif solve_for == "width":
        if n is None:
            raise ValueError("n is required when solve_for='width'")
        if tolerance_probability is None or not 0 < tolerance_probability < 1:
            raise ValueError("tolerance_probability must be in (0, 1)")
        n_out = n
        achieved_width = _tol_adjusted_half_width_one_mean(
            n_out, sigma_diff, alpha, sides, tolerance_probability, sd_source, prior_n
        )
        achieved_tol = float(tolerance_probability)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "ci_paired_means_tolerance",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": None,
        "achieved_width": achieved_width,
        "tolerance_probability": float(tolerance_probability),
        "achieved_tolerance_probability": achieved_tol,
        "inputs_echo": inputs_echo,
        "citations": [
            "with Tolerance Probability",
            "Hahn, G. J. and Meeker, W. Q. (1991). Statistical Intervals. "
            "John Wiley & Sons. New York.",
            "Harris, M., Horvitz, D. J., and Mood, A. M. (1948). JASA 43, "
            "391-402.",
            "Kupper, L. L. and Hafner, K. B. (1989). The American "
            "Statistician 43, 101-105.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 472: CI for Difference Between Two Means with Tolerance Probability
# ---------------------------------------------------------------------------


def ci_difference_two_means_tolerance(
    *,
    sigma: float,
    distance: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    n: int | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    tolerance_probability: float | None = None,
    sd_source: str = "population",
    prior_n_total: int | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for difference of two means with tolerance probability.

    Assumes equal group sizes (n1 = n2) and equal population variances (pooled
    SD). Only the population-SD and previous-sample-SD adjustment modes are
    supported.

    Parameters
    ----------
    sigma
        Pooled standard deviation (population value or estimated from previous
        samples).
    distance
        Target half-width D. Required when solve_for='n'.
    n1, n2
        Per-group sample sizes. When neither is given, equal allocation is used.
        When n is given (total), equal per-group sizes are inferred.
    n
        Total sample size (for solve_for='width' with equal n1=n2).
    alpha
        Type I error rate.
    sides
        1 or 2.
    tolerance_probability
        Probability that future CI half-width <= distance.
    sd_source
        'population' or 'previous_sample'.
    prior_n_total
        Total previous-sample size m1+m2; required when
        sd_source='previous_sample'.
    solve_for
        'n' (default) or 'width'.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if sd_source not in ("population", "previous_sample"):
        raise ValueError("sd_source must be 'population' or 'previous_sample'")

    have_distance = distance is not None
    have_n1 = n1 is not None
    have_n2 = n2 is not None
    have_n = n is not None

    if solve_for is None:
        if (have_n1 and have_n2) or have_n:
            solve_for = "width"
        elif have_distance:
            solve_for = "n"
        else:
            raise ValueError(
                "supply either distance to solve for n, or n1+n2/n to solve for width"
            )

    inputs_echo = dict(
        sigma=sigma, distance=distance, n1=n1, n2=n2, n=n, alpha=alpha,
        sides=sides, tolerance_probability=tolerance_probability,
        sd_source=sd_source, prior_n_total=prior_n_total,
    )

    if solve_for == "n":
        if distance is None:
            raise ValueError("distance is required when solve_for='n'")
        if tolerance_probability is None or not 0 < tolerance_probability < 1:
            raise ValueError("tolerance_probability must be in (0, 1)")
        k = 1 if sides == 1 else 2
        z = D.norm_ppf(1.0 - alpha / k)
        # z-approximation starting point for equal n
        n_per = max(2, math.ceil(2 * (z * sigma / distance) ** 2))
        while _tol_adjusted_half_width_two_means(
            n_per, sigma, alpha, sides, tolerance_probability, sd_source, prior_n_total
        ) > distance:
            n_per += 1
        achieved_width = _tol_adjusted_half_width_two_means(
            n_per, sigma, alpha, sides, tolerance_probability, sd_source, prior_n_total
        )
        _n1 = _n2 = n_per
        achieved_tol = float(tolerance_probability)
    elif solve_for == "width":
        if have_n1 and have_n2:
            _n1, _n2 = n1, n2
        elif have_n:
            _n1 = n // 2
            _n2 = n - _n1
        else:
            raise ValueError("supply n1+n2 or total n when solve_for='width'")
        if tolerance_probability is None or not 0 < tolerance_probability < 1:
            raise ValueError("tolerance_probability must be in (0, 1)")
        # Use average per-group n for symmetric formula
        n_per = (_n1 + _n2) // 2
        achieved_width = _tol_adjusted_half_width_two_means(
            n_per, sigma, alpha, sides, tolerance_probability, sd_source, prior_n_total
        )
        achieved_tol = float(tolerance_probability)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "ci_difference_two_means_tolerance",
        "solve_for": solve_for,
        "n": _n1 + _n2,
        "n1": _n1,
        "n2": _n2,
        "achieved_power": None,
        "achieved_width": achieved_width,
        "tolerance_probability": float(tolerance_probability),
        "achieved_tolerance_probability": achieved_tol,
        "inputs_echo": inputs_echo,
        "citations": [
            "Between Two Means with Tolerance Probability",
            "Hahn, G. J. and Meeker, W. Q. (1991). Statistical Intervals. "
            "John Wiley & Sons. New York.",
            "Harris, M., Horvitz, D. J., and Mood, A. M. (1948). JASA 43, "
            "391-402.",
            "Kupper, L. L. and Hafner, K. B. (1989). The American "
            "Statistician 43, 101-105.",
            "Zar, J. H. (1984). Biostatistical Analysis, 2nd ed. "
            "Prentice-Hall.",
        ],
    }
