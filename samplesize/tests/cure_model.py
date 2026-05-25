"""One-sample cure model test — Wu (2015) Weibull-mixture logrank.


The test statistic L = (O - E) / sqrt((O + E) / 2) is asymptotically
standard normal under H0.  Power is computed via Wu (2015) equations
using numerical integration of the integrands v0, v1, v00, v01 over
[0, tau] with the censoring weight function G(t).

Power formula (one-sided):
    power ≈ Φ(-σ̄/σ * z_{1-α} - ω*sqrt(n) / σ)

where:
    ω   = v1 - v0
    σ̄²  = (v1 + v0) / 2
    σ²  = v1 - v1² + 2*v00 - v0² - 2*v01 + 2*v0*v1

and the survival mixture model is:
    S*(t) = π + (1-π) * exp(-λ * t^k)

Reference: Wu, J. (2015). Single-arm phase II trial design under parametric
cure models. Pharmaceutical Statistics. DOI:10.1002/pst.1678.
"""
from __future__ import annotations

import math
from typing import Any

from scipy import integrate

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Cure model survival / hazard helpers
# ---------------------------------------------------------------------------

def _S_star(t: float, pi: float, lam: float, k: float) -> float:
    """Mixture cure model survival: S*(t) = π + (1-π)·exp(-λ·t^k)."""
    if t <= 0.0:
        return 1.0
    return pi + (1.0 - pi) * math.exp(-lam * t ** k)


def _Lambda0_star(t: float, pi0: float, lam0: float, k: float) -> float:
    """Cumulative hazard under H0: Λ0*(t) = -ln(S0*(t))."""
    s = _S_star(t, pi0, lam0, k)
    if s <= 0.0:
        return float("inf")
    return -math.log(s)


def _h(t: float, pi: float, lam: float, k: float) -> float:
    """Hazard of mixture cure model at time t."""
    if t <= 0.0:
        return 0.0
    num = (1.0 - pi) * lam * k * t ** (k - 1.0) * math.exp(-lam * t ** k)
    s = _S_star(t, pi, lam, k)
    if s <= 0.0:
        return 0.0
    return num / s


def _G(t: float, tf: float, ta: float, tau: float) -> float:
    """Censoring weight (proportion still under study at time t).

    G(t) = 1           if t <= tf
           (tau - t)/ta if tf < t <= tau
           0            otherwise
    """
    if t <= tf:
        return 1.0
    if t <= tau:
        return (tau - t) / ta
    return 0.0


# ---------------------------------------------------------------------------
# Compute v0, v1, v00, v01 by quadrature
# ---------------------------------------------------------------------------

def _compute_v_integrals(
    pi0: float, pi1: float, lam0: float, lam1: float,
    k: float, ta: float, tf: float,
) -> tuple[float, float, float, float]:
    tau = ta + tf
    eps = 1e-10
    upper = tau - eps

    def integrand_v0(t: float) -> float:
        g = _G(t, tf, ta, tau)
        s1 = _S_star(t, pi1, lam1, k)
        h0 = _h(t, pi0, lam0, k)
        return g * s1 * h0

    def integrand_v1(t: float) -> float:
        g = _G(t, tf, ta, tau)
        s1 = _S_star(t, pi1, lam1, k)
        h1 = _h(t, pi1, lam1, k)
        return g * s1 * h1

    def integrand_v00(t: float) -> float:
        g = _G(t, tf, ta, tau)
        s1 = _S_star(t, pi1, lam1, k)
        h0 = _h(t, pi0, lam0, k)
        L0 = _Lambda0_star(t, pi0, lam0, k)
        return g * s1 * h0 * L0

    def integrand_v01(t: float) -> float:
        g = _G(t, tf, ta, tau)
        s1 = _S_star(t, pi1, lam1, k)
        h1 = _h(t, pi1, lam1, k)
        L0 = _Lambda0_star(t, pi0, lam0, k)
        return g * s1 * h1 * L0

    opts = {"limit": 200, "epsabs": 1e-9, "epsrel": 1e-9}
    v0, _ = integrate.quad(integrand_v0, eps, upper, **opts)
    v1, _ = integrate.quad(integrand_v1, eps, upper, **opts)
    v00, _ = integrate.quad(integrand_v00, eps, upper, **opts)
    v01, _ = integrate.quad(integrand_v01, eps, upper, **opts)
    return v0, v1, v00, v01


def _cure_power(
    pi0: float, pi1: float, lam0: float, lam1: float,
    k: float, ta: float, tf: float,
    n: int, alpha: float, sides: int,
) -> float:
    v0, v1, v00, v01 = _compute_v_integrals(pi0, pi1, lam0, lam1, k, ta, tf)
    omega = v1 - v0
    sigma_bar2 = (v1 + v0) / 2.0
    sigma2 = (v1 - v1 * v1 + 2.0 * v00 - v0 * v0
              - 2.0 * v01 + 2.0 * v0 * v1)
    if sigma2 <= 0.0 or sigma_bar2 <= 0.0:
        return 0.0
    sigma_bar = math.sqrt(sigma_bar2)
    sigma = math.sqrt(sigma2)
    k_sides = 1 if sides == 1 else 2
    z_a = D.norm_ppf(1.0 - alpha / k_sides)
    from scipy.stats import norm as _n
    return float(_n.cdf(-sigma_bar / sigma * z_a
                        - omega * math.sqrt(n) / sigma))


def _cure_n_for_power(
    pi0: float, pi1: float, lam0: float, lam1: float,
    k: float, ta: float, tf: float,
    alpha: float, power: float, sides: int,
) -> tuple[int, float]:
    v0, v1, v00, v01 = _compute_v_integrals(pi0, pi1, lam0, lam1, k, ta, tf)
    omega = v1 - v0
    sigma_bar2 = (v1 + v0) / 2.0
    sigma2 = (v1 - v1 * v1 + 2.0 * v00 - v0 * v0
              - 2.0 * v01 + 2.0 * v0 * v1)
    if sigma2 <= 0.0 or sigma_bar2 <= 0.0:
        raise ValueError("invalid integral values")
    sigma_bar = math.sqrt(sigma_bar2)
    sigma = math.sqrt(sigma2)
    k_sides = 1 if sides == 1 else 2
    z_a = D.norm_ppf(1.0 - alpha / k_sides)
    z_p = D.norm_ppf(power)
    # n = (σ̄·z_α + σ·z_power)² / ω²
    n_float = (sigma_bar * z_a + sigma * z_p) ** 2 / omega ** 2
    n_total = max(3, math.ceil(n_float - 1e-9))
    # Guard for rounding
    achieved = _cure_power(pi0, pi1, lam0, lam1, k, ta, tf, n_total,
                           alpha, sides)
    while achieved < power and n_total < 10_000_000:
        n_total += 1
        achieved = _cure_power(pi0, pi1, lam0, lam1, k, ta, tf, n_total,
                               alpha, sides)
    return n_total, achieved


def _lam_from_median(median: float, k: float) -> float:
    """Weibull scale from median: λ = ln(2) / M^k."""
    return math.log(2.0) / (median ** k)


def _lam_from_survival(s: float, t0: float, k: float) -> float:
    """Weibull scale from S(t0): λ = -ln(S) / t0^k."""
    if s <= 0.0 or s >= 1.0:
        raise ValueError("survival proportion must be in (0, 1)")
    return -math.log(s) / (t0 ** k)


def one_sample_cure_model_logrank(
    *,
    pi0: float,
    pi1: float | None = None,
    lam0: float | None = None,
    lam1: float | None = None,
    median0: float | None = None,
    hr: float | None = None,
    s0: float | None = None,
    s1: float | None = None,
    t0: float | None = None,
    k: float = 1.0,
    t_accrual: float,
    t_followup: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """One-sample cure model test (Wu 2015).

    One-sample cure model test following Wu (2015).

    The test compares a new treatment to a historical control using the
    parametric cure model logrank statistic of Wu (2015).

    Parameters
    ----------
    pi0
        Cure rate (proportion cured) in the historical control.
    pi1
        Cure rate in the new group.  Default = pi0.
    lam0
        Weibull scale parameter λ for the control latency distribution.
        Exactly one of (lam0, median0, s0+t0) must be provided for control.
    lam1
        Weibull scale parameter λ for treatment.  Inferred from hr if None.
    median0
        Median survival time of the latency distribution (non-cured),
        control group.  Used when lam0 is None.
    hr
        Hazard ratio λ1/λ0.  Used to compute lam1 when lam1 is None.
    s0, s1
        Proportion surviving at time t0 in control/treatment latency.
        Alternative to (lam0, median0) / (lam1, hr).
    t0
        Time at which s0/s1 are specified.
    k
        Weibull shape parameter (k=1 for exponential).
    t_accrual
        Accrual period ta.
    t_followup
        Follow-up period tf.
    """
    if pi1 is None:
        pi1 = pi0

    # Resolve lam0
    if lam0 is not None:
        pass
    elif median0 is not None:
        lam0 = _lam_from_median(median0, k)
    elif s0 is not None and t0 is not None:
        lam0 = _lam_from_survival(s0, t0, k)
    else:
        raise ValueError("supply one of (lam0, median0, s0+t0) for control")

    # Resolve lam1
    if lam1 is not None:
        pass
    elif hr is not None:
        lam1 = hr * lam0
    elif s1 is not None and t0 is not None:
        lam1 = _lam_from_survival(s1, t0, k)
    else:
        raise ValueError("supply one of (lam1, hr, s1+t0) for treatment")

    inputs_echo = dict(pi0=pi0, pi1=pi1, lam0=lam0, lam1=lam1, k=k,
                       t_accrual=t_accrual, t_followup=t_followup,
                       alpha=alpha, power=power, n=n, sides=sides,
                       hr=lam1 / lam0)

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _cure_power(pi0, pi1, lam0, lam1, k, t_accrual,
                               t_followup, n, alpha, sides)
        n_total = n
    else:
        assert power is not None
        n_total, achieved = _cure_n_for_power(
            pi0, pi1, lam0, lam1, k, t_accrual, t_followup, alpha,
            power, sides,
        )

    # Compute expected events
    tau = t_accrual + t_followup
    v0, v1, _, _ = _compute_v_integrals(pi0, pi1, lam0, lam1, k,
                                         t_accrual, t_followup)
    events = int(round(n_total * v1))

    return {
        "method_id": "one_sample_cure_model_logrank",
        "solve_for": solve_for,
        "n": n_total,
        "events": events,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Wu, J. (2015). Single-arm phase II trial design under parametric "
            "cure models. Pharmaceutical Statistics, DOI:10.1002/pst.1678.",
        ],
    }
