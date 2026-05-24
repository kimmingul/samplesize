"""Confidence-interval sample-size routines for exponential distribution parameters.


- Ch 406: Confidence Intervals for the Exponential Lifetime Mean
- Ch 409: Confidence Intervals for the Exponential Hazard Rate
- Ch 408: Confidence Intervals for Exponential Reliability
- Ch 407: Confidence Intervals for an Exponential Lifetime Percentile

All methods solve for the number of events E (Type-II censoring) that produces
a CI of the specified width, or evaluate the achieved width at a fixed E.

The key relationships (exact, chi-square pivot on 2E degrees of freedom):

Lifetime mean θ (Ch 406):
    LCL = 2E*θ / chi2_{1-α/2, 2E}
    UCL = 2E*θ / chi2_{α/2, 2E}
    Width = UCL - LCL

Hazard rate λ = 1/θ (Ch 409):
    LCL = λ * chi2_{α/2, 2E} / (2E)
    UCL = λ * chi2_{1-α/2, 2E} / (2E)
    Width = UCL - LCL

Reliability R(t) = exp(-t/θ) (Ch 408):
    LCL = exp(-t * chi2_{1-α/2, 2E} / (2E*θ))
    UCL = exp(-t * chi2_{α/2, 2E} / (2E*θ))
    Width = UCL - LCL

Lifetime percentile t_p = θ * (-ln(1-p)) (Ch 407):
    LCL = 2E*t_p / chi2_{1-α/2, 2E}
    UCL = 2E*t_p / chi2_{α/2, 2E}
    Width = UCL - LCL

For one-sided bounds, replace α/2 with α (or equivalently, one limit disappears).

Expected total subjects: N = E / (1 - pct_censored/100).

References
----------
Mathews, P. (2010). Sample Size Calculations. Mathews Malnar and Bailey.
Lawless, J. F. (2003). Statistical Models and Methods for Lifetime Data, 2nd ed. Wiley.
Nelson, W. (1982). Applied Life Data Analysis. Wiley.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import chi2 as _chi2


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _expected_n(events: int, pct_censored: float) -> int:
    """Expected total subjects for a Type-II censored study."""
    if pct_censored < 0 or pct_censored >= 100:
        raise ValueError("pct_censored must be in [0, 100)")
    fraction_observed = 1.0 - pct_censored / 100.0
    return math.ceil(events / fraction_observed)


def _solve_events_monotone(width_fn, target: float,
                           e_min: int = 1, e_max: int = 10_000_000) -> int:
    """Smallest integer E >= e_min such that width_fn(E) <= target.

    Uses exponential doubling to bracket, then bisection.
    """
    lo = e_min
    hi = e_min
    while hi <= e_max:
        if width_fn(hi) <= target:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"could not achieve target width within E={e_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if width_fn(mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# Chapter 406: CI for Exponential Lifetime Mean θ
# ---------------------------------------------------------------------------


def _lifetime_mean_width(events: int, theta: float, alpha: float,
                         sides: int) -> float:
    """Achieved CI width (or distance) for the exponential mean at E events."""
    if events < 1:
        return float("inf")
    df = 2 * events
    k = 1 if sides == 1 else 2
    ucl = 2 * events * theta / _chi2.ppf(alpha / k, df)
    lcl = 2 * events * theta / _chi2.ppf(1.0 - alpha / k, df)
    return ucl - lcl


def ci_exponential_lifetime_mean(
    *,
    theta: float,
    width: float | None = None,
    events: int | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    pct_censored: float = 0.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for the exponential mean lifetime θ.

    Parameters
    ----------
    theta
        Planning estimate of the mean lifetime (1/λ).
    width
        Target CI width (UCL - LCL). Required when solve_for='n'.
    events
        Number of events E. Required when solve_for='width'.
    alpha
        Type I error rate; confidence level = 1 - alpha.
    sides
        2 for two-sided, 1 for one-sided (distance from θ to bound).
    pct_censored
        Expected percent of subjects censored (not observed to fail).
        Used to compute expected total sample size N = E/(1 - pct/100).
    solve_for
        'n' (solve for events; default when width supplied) or 'width'.
    """
    if theta <= 0:
        raise ValueError("theta must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    have_width = width is not None
    have_events = events is not None
    if not (have_width or have_events):
        raise ValueError("supply at least one of (width, events)")
    if solve_for is None:
        solve_for = "n" if not have_events else "width"

    inputs_echo = dict(theta=theta, width=width, events=events, alpha=alpha,
                       sides=sides, pct_censored=pct_censored)

    if solve_for == "n":
        if width is None or width <= 0:
            raise ValueError("width > 0 is required when solve_for='n'")
        e_out = _solve_events_monotone(
            lambda e: _lifetime_mean_width(e, theta, alpha, sides), width
        )
    elif solve_for == "width":
        if events is None or events < 1:
            raise ValueError("events >= 1 is required when solve_for='width'")
        e_out = int(events)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    achieved_width = _lifetime_mean_width(e_out, theta, alpha, sides)
    n_subjects = _expected_n(e_out, pct_censored)

    df = 2 * e_out
    k = 1 if sides == 1 else 2
    lcl = 2 * e_out * theta / _chi2.ppf(1.0 - alpha / k, df)
    ucl = 2 * e_out * theta / _chi2.ppf(alpha / k, df)

    return {
        "method_id": "ci_exponential_lifetime_mean",
        "solve_for": solve_for,
        "n": n_subjects,
        "events": e_out,
        "achieved_power": None,
        "achieved_width": achieved_width,
        "ci_lower": lcl,
        "ci_upper": ucl,
        "inputs_echo": inputs_echo,
        "citations": [
            "Lifetime Mean",
            "Mathews, P. (2010). Sample Size Calculations: Practical Methods "
            "for Engineers and Scientists. Mathews Malnar and Bailey.",
            "Lawless, J. F. (2003). Statistical Models and Methods for "
            "Lifetime Data, 2nd ed. Wiley.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 409: CI for Exponential Hazard Rate λ
# ---------------------------------------------------------------------------


def _hazard_rate_width(events: int, lam: float, alpha: float,
                       sides: int) -> float:
    """Achieved CI width (or distance) for the exponential hazard rate λ."""
    if events < 1:
        return float("inf")
    df = 2 * events
    k = 1 if sides == 1 else 2
    ucl = lam * _chi2.ppf(1.0 - alpha / k, df) / (2 * events)
    lcl = lam * _chi2.ppf(alpha / k, df) / (2 * events)
    return ucl - lcl


def ci_exponential_hazard_rate(
    *,
    lam: float,
    width: float | None = None,
    events: int | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    pct_censored: float = 0.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for the exponential hazard rate λ.

    Parameters
    ----------
    lam
        Planning estimate of the hazard (failure) rate λ = 1/θ.
    width
        Target CI width. Required when solve_for='n'.
    events
        Number of events E. Required when solve_for='width'.
    alpha
        Type I error rate.
    sides
        2 for two-sided, 1 for one-sided.
    pct_censored
        Expected percent censored.
    solve_for
        'n' or 'width'.
    """
    if lam <= 0:
        raise ValueError("lam must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    have_width = width is not None
    have_events = events is not None
    if not (have_width or have_events):
        raise ValueError("supply at least one of (width, events)")
    if solve_for is None:
        solve_for = "n" if not have_events else "width"

    inputs_echo = dict(lam=lam, width=width, events=events, alpha=alpha,
                       sides=sides, pct_censored=pct_censored)

    if solve_for == "n":
        if width is None or width <= 0:
            raise ValueError("width > 0 is required when solve_for='n'")
        e_out = _solve_events_monotone(
            lambda e: _hazard_rate_width(e, lam, alpha, sides), width
        )
    elif solve_for == "width":
        if events is None or events < 1:
            raise ValueError("events >= 1 is required when solve_for='width'")
        e_out = int(events)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    achieved_width = _hazard_rate_width(e_out, lam, alpha, sides)
    n_subjects = _expected_n(e_out, pct_censored)

    df = 2 * e_out
    k = 1 if sides == 1 else 2
    lcl = lam * _chi2.ppf(alpha / k, df) / (2 * e_out)
    ucl = lam * _chi2.ppf(1.0 - alpha / k, df) / (2 * e_out)

    return {
        "method_id": "ci_exponential_hazard_rate",
        "solve_for": solve_for,
        "n": n_subjects,
        "events": e_out,
        "achieved_power": None,
        "achieved_width": achieved_width,
        "ci_lower": lcl,
        "ci_upper": ucl,
        "inputs_echo": inputs_echo,
        "citations": [
            "Hazard Rate",
            "Nelson, W. (1982). Applied Life Data Analysis. Wiley.",
            "Mathews, P. (2010). Sample Size Calculations: Practical Methods "
            "for Engineers and Scientists. Mathews Malnar and Bailey.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 408: CI for Exponential Reliability R(t) = exp(-t/θ)
# ---------------------------------------------------------------------------


def _reliability_width(events: int, theta: float, t: float,
                       alpha: float, sides: int) -> float:
    """Achieved CI width for the reliability R(t) = exp(-t/θ)."""
    if events < 1:
        return float("inf")
    df = 2 * events
    k = 1 if sides == 1 else 2
    lcl = math.exp(-t * _chi2.ppf(1.0 - alpha / k, df) / (2 * events * theta))
    ucl = math.exp(-t * _chi2.ppf(alpha / k, df) / (2 * events * theta))
    return ucl - lcl


def ci_exponential_reliability(
    *,
    theta: float,
    t: float,
    width: float | None = None,
    events: int | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    pct_censored: float = 0.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for exponential reliability R(t).

    Parameters
    ----------
    theta
        Planning estimate of the mean lifetime θ.
    t
        Time point at which reliability is evaluated.
    width
        Target CI width (UCL - LCL on the probability scale). Required when
        solve_for='n'.
    events
        Number of events E. Required when solve_for='width'.
    alpha
        Type I error rate.
    sides
        2 for two-sided, 1 for one-sided.
    pct_censored
        Expected percent censored.
    solve_for
        'n' or 'width'.
    """
    if theta <= 0:
        raise ValueError("theta must be > 0")
    if t <= 0:
        raise ValueError("t must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    have_width = width is not None
    have_events = events is not None
    if not (have_width or have_events):
        raise ValueError("supply at least one of (width, events)")
    if solve_for is None:
        solve_for = "n" if not have_events else "width"

    inputs_echo = dict(theta=theta, t=t, width=width, events=events,
                       alpha=alpha, sides=sides, pct_censored=pct_censored)

    if solve_for == "n":
        if width is None or width <= 0:
            raise ValueError("width > 0 is required when solve_for='n'")
        e_out = _solve_events_monotone(
            lambda e: _reliability_width(e, theta, t, alpha, sides), width
        )
    elif solve_for == "width":
        if events is None or events < 1:
            raise ValueError("events >= 1 is required when solve_for='width'")
        e_out = int(events)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    achieved_width = _reliability_width(e_out, theta, t, alpha, sides)
    n_subjects = _expected_n(e_out, pct_censored)

    rt = math.exp(-t / theta)
    df = 2 * e_out
    k = 1 if sides == 1 else 2
    lcl = math.exp(-t * _chi2.ppf(1.0 - alpha / k, df) / (2 * e_out * theta))
    ucl = math.exp(-t * _chi2.ppf(alpha / k, df) / (2 * e_out * theta))

    return {
        "method_id": "ci_exponential_reliability",
        "solve_for": solve_for,
        "n": n_subjects,
        "events": e_out,
        "achieved_power": None,
        "achieved_width": achieved_width,
        "reliability": rt,
        "ci_lower": lcl,
        "ci_upper": ucl,
        "inputs_echo": inputs_echo,
        "citations": [
            "Reliability",
            "Nelson, W. (1982). Applied Life Data Analysis. Wiley.",
            "Mathews, P. (2010). Sample Size Calculations: Practical Methods "
            "for Engineers and Scientists. Mathews Malnar and Bailey.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 407: CI for Exponential Lifetime Percentile t_p
# ---------------------------------------------------------------------------


def _percentile_value(theta: float, p: float) -> float:
    """100p-th percentile of Exp(θ): t_p = θ * (-ln(1 - p))."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    return theta * (-math.log(1.0 - p))


def _percentile_width(events: int, theta: float, p: float,
                      alpha: float, sides: int) -> float:
    """Achieved CI width for the exponential lifetime percentile t_p."""
    if events < 1:
        return float("inf")
    tp = _percentile_value(theta, p)
    df = 2 * events
    k = 1 if sides == 1 else 2
    ucl = 2 * events * tp / _chi2.ppf(alpha / k, df)
    lcl = 2 * events * tp / _chi2.ppf(1.0 - alpha / k, df)
    return ucl - lcl


def ci_exponential_lifetime_percentile(
    *,
    theta: float,
    p: float,
    width: float | None = None,
    events: int | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    pct_censored: float = 0.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for an exponential lifetime percentile t_p.

    Parameters
    ----------
    theta
        Planning estimate of the mean lifetime θ.
    p
        Percentile proportion in (0, 1). E.g. p=0.20 for the 20th percentile.
    width
        Target CI width. Required when solve_for='n'.
    events
        Number of events E. Required when solve_for='width'.
    alpha
        Type I error rate.
    sides
        2 for two-sided, 1 for one-sided.
    pct_censored
        Expected percent censored.
    solve_for
        'n' or 'width'.
    """
    if theta <= 0:
        raise ValueError("theta must be > 0")
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    have_width = width is not None
    have_events = events is not None
    if not (have_width or have_events):
        raise ValueError("supply at least one of (width, events)")
    if solve_for is None:
        solve_for = "n" if not have_events else "width"

    inputs_echo = dict(theta=theta, p=p, width=width, events=events,
                       alpha=alpha, sides=sides, pct_censored=pct_censored)

    if solve_for == "n":
        if width is None or width <= 0:
            raise ValueError("width > 0 is required when solve_for='n'")
        e_out = _solve_events_monotone(
            lambda e: _percentile_width(e, theta, p, alpha, sides), width
        )
    elif solve_for == "width":
        if events is None or events < 1:
            raise ValueError("events >= 1 is required when solve_for='width'")
        e_out = int(events)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    tp = _percentile_value(theta, p)
    achieved_width = _percentile_width(e_out, theta, p, alpha, sides)
    n_subjects = _expected_n(e_out, pct_censored)

    df = 2 * e_out
    k = 1 if sides == 1 else 2
    lcl = 2 * e_out * tp / _chi2.ppf(1.0 - alpha / k, df)
    ucl = 2 * e_out * tp / _chi2.ppf(alpha / k, df)

    return {
        "method_id": "ci_exponential_lifetime_percentile",
        "solve_for": solve_for,
        "n": n_subjects,
        "events": e_out,
        "achieved_power": None,
        "achieved_width": achieved_width,
        "lifetime_percentile": tp,
        "ci_lower": lcl,
        "ci_upper": ucl,
        "inputs_echo": inputs_echo,
        "citations": [
            "Lifetime Percentile",
            "Mathews, P. (2010). Sample Size Calculations: Practical Methods "
            "for Engineers and Scientists. Mathews Malnar and Bailey.",
            "Lawless, J. F. (2003). Statistical Models and Methods for "
            "Lifetime Data, 2nd ed. Wiley.",
        ],
    }
