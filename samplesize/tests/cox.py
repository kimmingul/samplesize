"""Cox proportional-hazards regression sample size — Hsieh & Lavori (2000).


Tests H0: beta_1 = 0 vs. H1: beta_1 = B in a Cox PH model.  The Hsieh
and Lavori (2000) formula relates required events D and total sample
size N to alpha, beta, the log-hazard ratio B per unit covariate, the
covariate's standard deviation sigma, the R^2 from regressing X_1 on
the remaining covariates, and the overall event rate P:

    D = (z_{1-alpha/k} + z_{1-beta})^2 / [(1 - R^2) * sigma^2 * B^2]
    N = D / P

where k = 1 (one-sided) or 2 (two-sided).  The formula extends
Schoenfeld (1983) and applies to discrete or continuous X_1.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


def _required_D(B: float, sd_x: float, r_squared: float,
                alpha: float, power: float, sides: int) -> float:
    if B == 0:
        raise ValueError("B (log hazard ratio) must be non-zero")
    if sd_x <= 0:
        raise ValueError("sd_x must be positive")
    if not 0.0 <= r_squared < 1.0:
        raise ValueError("r_squared must be in [0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    k = 1 if sides == 1 else 2
    z_alpha = D.norm_ppf(1.0 - alpha / k)
    z_beta = D.norm_ppf(power)
    num = (z_alpha + z_beta) ** 2
    den = (1.0 - r_squared) * sd_x * sd_x * B * B
    return num / den


def _power_at_n(*, B: float, sd_x: float, r_squared: float, event_rate: float,
                n: int, alpha: float, sides: int) -> float:
    if not 0.0 < event_rate <= 1.0:
        raise ValueError("event_rate must be in (0, 1]")
    if sd_x <= 0:
        raise ValueError("sd_x must be positive")
    if not 0.0 <= r_squared < 1.0:
        raise ValueError("r_squared must be in [0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n < 1:
        return 0.0
    k = 1 if sides == 1 else 2
    z_alpha = D.norm_ppf(1.0 - alpha / k)
    events = n * event_rate
    inside = events * (1.0 - r_squared) * sd_x * sd_x * B * B
    if inside <= 0:
        return 0.0
    z_beta = math.sqrt(inside) - z_alpha
    from scipy.stats import norm
    return float(norm.cdf(z_beta))


def power_at_n(*, B: float, sd_x: float, event_rate: float, n: int,
               alpha: float, sides: int = 2, r_squared: float = 0.0) -> float:
    return _power_at_n(B=B, sd_x=sd_x, r_squared=r_squared,
                       event_rate=event_rate, n=n, alpha=alpha, sides=sides)


def n_for_power(*, B: float, sd_x: float, event_rate: float, alpha: float,
                power: float, sides: int = 2,
                r_squared: float = 0.0) -> tuple[int, float]:
    if not 0.0 < event_rate <= 1.0:
        raise ValueError("event_rate must be in (0, 1]")
    d_required = _required_D(B, sd_x, r_squared, alpha, power, sides)
    n_real = d_required / event_rate
    n_total = max(1, math.ceil(n_real - 1e-9))
    achieved = _power_at_n(B=B, sd_x=sd_x, r_squared=r_squared,
                           event_rate=event_rate, n=n_total,
                           alpha=alpha, sides=sides)
    # Guard: ensure achieved >= power despite floating drift.
    while achieved < power and n_total < 10_000_000:
        n_total += 1
        achieved = _power_at_n(B=B, sd_x=sd_x, r_squared=r_squared,
                               event_rate=event_rate, n=n_total,
                               alpha=alpha, sides=sides)
    return n_total, achieved


def _schoenfeld_events(
    log_hr: float, log_hr_margin: float, pev1: float, pev2: float,
    p1: float, alpha: float, power: float, sides: int,
) -> float:
    """Required events D for a Cox / logrank NI or superiority test.

    The logrank statistic has mean (log(HR) - log(margin)) * sqrt(P1*P2*d*N)
    and unit variance.  Rearranging for events:

        D = ((z_alpha + z_beta) / (log(HR) - log(margin)))^2 / (P1 * P2)

    where D = d * N (total events), d = Pev1*P1 + Pev2*P2.
    """
    effect = abs(log_hr - log_hr_margin)
    if effect == 0.0:
        raise ValueError("log(HR) must differ from log(margin)")
    k = 1 if sides == 1 else 2
    z_a = D.norm_ppf(1.0 - alpha / k)
    z_b = D.norm_ppf(power)
    return (z_a + z_b) ** 2 / (p1 * (1.0 - p1) * effect ** 2)


def _schoenfeld_power(
    log_hr: float, log_hr_margin: float, pev1: float, pev2: float,
    p1: float, alpha: float, n: int, sides: int,
) -> float:
    d = pev1 * p1 + pev2 * (1.0 - p1)
    events = n * d
    # Use absolute separation: power is always Φ(|logHR - logMargin|*√(...) - z_α)
    effect = abs(log_hr - log_hr_margin)
    k = 1 if sides == 1 else 2
    z_a = D.norm_ppf(1.0 - alpha / k)
    z_beta = effect * math.sqrt(p1 * (1.0 - p1) * events) - z_a
    from scipy.stats import norm
    return float(norm.cdf(z_beta))


def _cox_ni_n_for_power(
    *, hr: float, hr_margin: float, pev1: float, pev2: float,
    p1: float, alpha: float, power: float, sides: int,
) -> tuple[int, float]:
    log_hr = math.log(hr)
    log_margin = math.log(hr_margin)
    d = pev1 * p1 + pev2 * (1.0 - p1)
    events_req = _schoenfeld_events(log_hr, log_margin, pev1, pev2,
                                    p1, alpha, power, sides)
    n_float = events_req / d
    n_total = max(2, math.ceil(n_float - 1e-9))
    # Ensure achieved >= power
    achieved = _schoenfeld_power(log_hr, log_margin, pev1, pev2,
                                 p1, alpha, n_total, sides)
    while achieved < power and n_total < 10_000_000:
        n_total += 1
        achieved = _schoenfeld_power(log_hr, log_margin, pev1, pev2,
                                     p1, alpha, n_total, sides)
    return n_total, achieved


def non_inferiority_cox_regression(
    *,
    hr: float,
    hr_ni: float,
    pev1: float,
    pev2: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    p1: float = 0.5,
    sides: int = 1,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for the hazard ratio using Cox PH (Schoenfeld).

    Non-Inferiority test for two survival curves using
    Cox's Proportional Hazards Model".

    Parameters
    ----------
    hr
        Assumed (actual) hazard ratio h2/h1.
    hr_ni
        Non-inferiority margin hazard ratio (> 1 when lower is better).
    pev1, pev2
        Probability of event in control / treatment group.
    p1
        Proportion in the control (group 1) arm.
    """
    inputs_echo = dict(hr=hr, hr_ni=hr_ni, pev1=pev1, pev2=pev2,
                       alpha=alpha, power=power, n=n, p1=p1, sides=sides)
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    log_hr = math.log(hr)
    log_ni = math.log(hr_ni)
    d = pev1 * p1 + pev2 * (1.0 - p1)

    if solve_for == "power":
        assert n is not None
        achieved = _schoenfeld_power(log_hr, log_ni, pev1, pev2,
                                     p1, alpha, n, sides)
        n_total = n
    elif solve_for == "n":
        assert power is not None
        n_total, achieved = _cox_ni_n_for_power(
            hr=hr, hr_margin=hr_ni, pev1=pev1, pev2=pev2,
            p1=p1, alpha=alpha, power=power, sides=sides,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    n1 = round(n_total * p1)
    n2 = n_total - n1
    events = int(round(n_total * d))
    return {
        "method_id": "non_inferiority_cox_regression",
        "solve_for": solve_for,
        "n": n_total,
        "n1": n1,
        "n2": n2,
        "events": events,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Using Cox's Proportional Hazards Model",
            "Schoenfeld, D.A. (1983). Sample-size formula for the "
            "proportional-hazards regression model. Biometrics 39:499-503.",
            "Chow, S.C., Shao, J., Wang, H. (2008). Sample Size Calculations "
            "in Clinical Research, 2nd Ed. Chapman & Hall/CRC.",
        ],
    }


def equivalence_cox_regression(
    *,
    hr: float,
    hr_eq: float,
    pev1: float,
    pev2: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    p1: float = 0.5,
    sides: int = 1,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Equivalence test for the hazard ratio using Cox PH (TOST, Schoenfeld).

    Equivalence test for two survival curves using
    Cox's Proportional Hazards Model".

    Power = Φ((log(HReq) - log(HR)) * sqrt(P1*P2*d*N) - z_α)
          + Φ((log(HReq) + log(HR)) * sqrt(P1*P2*d*N) - z_α) - 1

    Parameters
    ----------
    hr
        Assumed hazard ratio (1.0 for worst-case/equal).
    hr_eq
        Equivalence margin ratio (> 1); equivalent iff 1/hr_eq < HR < hr_eq.
    """
    inputs_echo = dict(hr=hr, hr_eq=hr_eq, pev1=pev1, pev2=pev2,
                       alpha=alpha, power=power, n=n, p1=p1, sides=sides)
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    log_hr = math.log(hr)
    log_eq = math.log(hr_eq)
    d = pev1 * p1 + pev2 * (1.0 - p1)

    from scipy.stats import norm as _norm

    def _tost_power(n_val: int) -> float:
        events = n_val * d
        kappa = math.sqrt(p1 * (1.0 - p1) * events)
        z_a = D.norm_ppf(1.0 - alpha)
        term1 = (log_eq - log_hr) * kappa - z_a
        term2 = (log_eq + log_hr) * kappa - z_a
        return float(_norm.cdf(term1) + _norm.cdf(term2) - 1.0)

    if solve_for == "power":
        assert n is not None
        achieved = _tost_power(n)
        n_total = n
    else:
        assert power is not None
        # Bisect for N
        lo, hi = 2, 2
        while _tost_power(hi) < power and hi < 10_000_000:
            lo = hi
            hi = max(hi + 1, hi * 2)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _tost_power(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_total = hi
        achieved = _tost_power(n_total)

    n1 = round(n_total * p1)
    n2 = n_total - n1
    events = int(round(n_total * d))
    return {
        "method_id": "equivalence_cox_regression",
        "solve_for": solve_for,
        "n": n_total,
        "n1": n1,
        "n2": n2,
        "events": events,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Using Cox's Proportional Hazards Model",
            "Schoenfeld, D.A. (1983). Sample-size formula for the "
            "proportional-hazards regression model. Biometrics 39:499-503.",
            "Chow, S.C., Shao, J., Wang, H. (2008). Sample Size Calculations "
            "in Clinical Research, 2nd Ed. Chapman & Hall/CRC.",
        ],
    }


def superiority_by_margin_cox_regression(
    *,
    hr: float,
    hr_su: float,
    pev1: float,
    pev2: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    p1: float = 0.5,
    sides: int = 1,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Superiority-by-margin test for the hazard ratio using Cox PH.

    Superiority-by-margin test for two survival curves using
    Using Cox's Proportional Hazards Model".

    Same Schoenfeld formula as NI with the margin replaced by hr_su.
    For "lower hazard better": H0: HR >= hr_su vs Ha: HR < hr_su (hr_su < 1).
    For "higher hazard better": H0: HR <= hr_su vs Ha: HR > hr_su (hr_su > 1).

    Parameters
    ----------
    hr
        Assumed hazard ratio h2/h1.
    hr_su
        Clinical superiority margin hazard ratio.
    """
    inputs_echo = dict(hr=hr, hr_su=hr_su, pev1=pev1, pev2=pev2,
                       alpha=alpha, power=power, n=n, p1=p1, sides=sides)
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    log_hr = math.log(hr)
    log_su = math.log(hr_su)
    d = pev1 * p1 + pev2 * (1.0 - p1)

    if solve_for == "power":
        assert n is not None
        achieved = _schoenfeld_power(log_hr, log_su, pev1, pev2,
                                     p1, alpha, n, sides)
        n_total = n
    else:
        assert power is not None
        n_total, achieved = _cox_ni_n_for_power(
            hr=hr, hr_margin=hr_su, pev1=pev1, pev2=pev2,
            p1=p1, alpha=alpha, power=power, sides=sides,
        )

    n1 = round(n_total * p1)
    n2 = n_total - n1
    events = int(round(n_total * d))
    return {
        "method_id": "superiority_by_margin_cox_regression",
        "solve_for": solve_for,
        "n": n_total,
        "n1": n1,
        "n2": n2,
        "events": events,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Curves Using Cox's Proportional Hazards Model",
            "Schoenfeld, D.A. (1983). Sample-size formula for the "
            "proportional-hazards regression model. Biometrics 39:499-503.",
            "Chow, S.C., Shao, J., Wang, H. (2008). Sample Size Calculations "
            "in Clinical Research, 2nd Ed. Chapman & Hall/CRC.",
        ],
    }


def cox_regression(
    *,
    B: float,
    sd_x: float,
    event_rate: float,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    r_squared: float = 0.0,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Sample size / power for the Cox PH regression coefficient test.

    Parameters
    ----------
    B
        Predicted change in log hazard per one-unit change of X_1
        (e.g. ln(1.5) for HR=1.5).
    sd_x
        Standard deviation of X_1 (sigma).
    event_rate
        Overall proportion of subjects experiencing the event (P).
        Set to 1.0 to interpret N as the number of events.
    alpha
        Type-I error.  For two-sided tests alpha is halved internally.
    n, power
        Supply exactly one to set ``solve_for``.
    r_squared
        R^2 from regressing X_1 on the remaining covariates.
        Use 0 when there are no other covariates.
    sides
        2 (default) or 1.
    """
    inputs_echo = {
        "B": B, "sd_x": sd_x, "event_rate": event_rate, "alpha": alpha,
        "n": n, "power": power, "r_squared": r_squared, "sides": sides,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = power_at_n(B=B, sd_x=sd_x, event_rate=event_rate, n=n,
                              alpha=alpha, sides=sides, r_squared=r_squared)
        n_total = n
    elif solve_for == "n":
        assert power is not None
        n_total, achieved = n_for_power(
            B=B, sd_x=sd_x, event_rate=event_rate, alpha=alpha,
            power=power, sides=sides, r_squared=r_squared,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    events = int(round(n_total * event_rate))
    return {
        "method_id": "cox_regression",
        "solve_for": solve_for,
        "n": n_total,
        "events": events,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Hsieh, F.Y. and Lavori, P.W. (2000). Sample-size calculations "
            "for the Cox proportional hazards regression model with "
            "nonbinary covariates. Controlled Clinical Trials 21:552-560.",
            "Schoenfeld, D.A. (1983). Sample-size formula for the "
            "proportional-hazards regression model. Biometrics 39:499-503.",
        ],
    }
