"""Survival sample-size — Freedman's (1982) logrank formula.


  HR = log(S2) / log(S1)              hazard ratio (treatment vs. control)
  φ  = (1 - p1) / p1                  group-2 to group-1 ratio (p1 = proportion in group 1)
  w  = loss-to-follow-up proportion

  z_{1-β} = |HR - 1| · √(N(1-w)·φ·[(1-S1) + φ·(1-S2)] / (1+φ)) / (1 + φ·HR)
            - z_{1-α/k}

  k = 1 (one-sided) or 2 (two-sided).
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


def _power(s1: float, s2: float, n: int, alpha: float, sides: int,
           p1: float, loss_to_followup: float) -> float:
    if not (0 < s1 < 1 and 0 < s2 < 1 and s1 != s2):
        raise ValueError("s1, s2 must be in (0, 1) and differ")
    if not 0 < p1 < 1:
        raise ValueError("p1 must be in (0, 1)")
    if not 0.0 <= loss_to_followup < 1.0:
        raise ValueError("loss_to_followup must be in [0, 1)")
    if n < 2:
        return 0.0
    hr = math.log(s2) / math.log(s1)
    phi = (1.0 - p1) / p1
    k = 1 if sides == 1 else 2
    z_alpha = D.norm_ppf(1.0 - alpha / k)
    inside = n * (1 - loss_to_followup) * phi \
             * ((1 - s1) + phi * (1 - s2)) / (1 + phi)
    if inside <= 0:
        return 0.0
    z_beta = abs(hr - 1) * math.sqrt(inside) / (1 + phi * hr) - z_alpha
    from scipy.stats import norm
    return float(norm.cdf(z_beta))


def power_at_n(*, s1: float, s2: float, n: int, alpha: float,
               sides: int = 2, p1: float = 0.5,
               loss_to_followup: float = 0.0) -> float:
    return _power(s1, s2, n, alpha, sides, p1, loss_to_followup)


def n_for_power(*, s1: float, s2: float, alpha: float, power: float,
                sides: int = 2, p1: float = 0.5,
                loss_to_followup: float = 0.0,
                n_min: int = 4, n_max: int = 10_000_000) -> tuple[int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    lo, hi = n_min, n_min
    while hi <= n_max:
        if _power(s1, s2, hi, alpha, sides, p1, loss_to_followup) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _power(s1, s2, mid, alpha, sides, p1, loss_to_followup) >= power:
            hi = mid
        else:
            lo = mid
    return hi, _power(s1, s2, hi, alpha, sides, p1, loss_to_followup)


def logrank_freedman(
    *,
    s1: float,
    s2: float,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    p1: float | None = None,
    allocation: float | None = None,
    loss_to_followup: float = 0.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-group survival sample size (Freedman 1982).

    Group allocation accepts either `p1` (proportion in group 1) or
    `allocation` (n2 / n1 ratio, matching the other two-arm methods).
    Default is balanced 1:1 (p1 = 0.5, allocation = 1.0).
    """
    if p1 is not None and allocation is not None:
        raise ValueError("supply only one of (p1, allocation), not both")
    if allocation is not None:
        if allocation <= 0:
            raise ValueError("allocation (n2/n1) must be > 0")
        p1 = 1.0 / (1.0 + allocation)
    elif p1 is None:
        p1 = 0.5
    inputs_echo = {
        "s1": s1, "s2": s2, "n": n, "alpha": alpha, "power": power,
        "sides": sides, "p1": p1, "loss_to_followup": loss_to_followup,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    hr = math.log(s2) / math.log(s1)

    if solve_for == "power":
        assert n is not None
        achieved = power_at_n(s1=s1, s2=s2, n=n, alpha=alpha, sides=sides,
                              p1=p1, loss_to_followup=loss_to_followup)
        n_total = n
    elif solve_for == "n":
        assert power is not None
        n_total, achieved = n_for_power(
            s1=s1, s2=s2, alpha=alpha, power=power, sides=sides,
            p1=p1, loss_to_followup=loss_to_followup,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    n1 = round(n_total * p1)
    n2 = n_total - n1
    return {
        "method_id": "logrank_freedman",
        "solve_for": solve_for,
        "n": n_total,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "hazard_ratio": hr,
        "inputs_echo": inputs_echo,
        "citations": [
            "Freedman, L.S. (1982). Tables of the number of patients required "
            "in clinical trials using the logrank test. Stat Med 1:121-129.",
        ],
    }


# =========================================================================
# Logrank Tests (Lachin and Foulkes)
# =========================================================================
#
# Extends the Freedman logrank by integrating the exponential hazards over
# an accrual window R and follow-up window T-R (Lachin & Foulkes 1986,
# Stat Med 5:65-72).  With uniform patient entry the per-group event
# probability is
#
#     P_i = (λ_i / (λ_i + η_i)) ·
#           [1 - (e^{-(λ_i+η_i)(T-R)} - e^{-(λ_i+η_i)T}) /
#                ((λ_i + η_i) · R)]
#
# where η_i is the (exponential) loss-to-follow-up hazard.  The sample
# size equation is (Eq. 2.1):
#
#     √N · |λ1 - λ2| =
#         z_α · √[ φ(λ̄) · (1/Q1 + 1/Q2) ]
#         + z_β · √[ φ(λ1)/Q1 + φ(λ2)/Q2 ]
#
# with φ(λ_i) = λ_i² / P_i and λ̄ = Q1·λ1 + Q2·λ2 (using P_bar evaluated
# at λ̄ with η_bar = Q1·η1 + Q2·η2).


def _prob_event_lf(lam: float, eta: float, R: float, T: float) -> float:
    """Probability of observing an event under uniform accrual on [0,R],
    exponential survival hazard ``lam`` and exponential loss-to-follow-up
    hazard ``eta``, with total study time ``T``."""
    s = lam + eta
    if s <= 0:
        return 0.0
    # Uniform-entry integral; numerically stable via expm1.
    follow = T - R
    if R <= 0:
        # All subjects enrolled at t=0; P = (λ/s)·(1 - e^{-sT})
        return (lam / s) * (-math.expm1(-s * T))
    # bracket = e^{-s(T-R)} - e^{-sT}
    bracket = math.exp(-s * follow) - math.exp(-s * T)
    return (lam / s) * (1.0 - bracket / (s * R))


def _lf_power(*, lam1: float, lam2: float, eta1: float, eta2: float,
              R: float, T: float, n: int, alpha: float, sides: int,
              p1: float) -> float:
    if n < 2:
        return 0.0
    q1, q2 = p1, 1.0 - p1
    p_e1 = _prob_event_lf(lam1, eta1, R, T)
    p_e2 = _prob_event_lf(lam2, eta2, R, T)
    if p_e1 <= 0 or p_e2 <= 0:
        return 0.0
    lam_bar = q1 * lam1 + q2 * lam2
    eta_bar = q1 * eta1 + q2 * eta2
    p_e_bar = _prob_event_lf(lam_bar, eta_bar, R, T)
    if p_e_bar <= 0:
        return 0.0
    phi1 = lam1 * lam1 / p_e1
    phi2 = lam2 * lam2 / p_e2
    phi_bar = lam_bar * lam_bar / p_e_bar

    k = 1 if sides == 1 else 2
    z_alpha = D.norm_ppf(1.0 - alpha / k)

    var_null = phi_bar * (1.0 / q1 + 1.0 / q2)
    var_alt = phi1 / q1 + phi2 / q2
    if var_null <= 0 or var_alt <= 0:
        return 0.0
    lhs = math.sqrt(n) * abs(lam1 - lam2)
    z_beta = (lhs - z_alpha * math.sqrt(var_null)) / math.sqrt(var_alt)
    from scipy.stats import norm
    return float(norm.cdf(z_beta))


def _lf_n_for_power(*, lam1: float, lam2: float, eta1: float, eta2: float,
                    R: float, T: float, alpha: float, power: float,
                    sides: int, p1: float, n_min: int = 4,
                    n_max: int = 10_000_000) -> tuple[int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    lo, hi = n_min, n_min
    while hi <= n_max:
        if _lf_power(lam1=lam1, lam2=lam2, eta1=eta1, eta2=eta2,
                     R=R, T=T, n=hi, alpha=alpha, sides=sides,
                     p1=p1) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _lf_power(lam1=lam1, lam2=lam2, eta1=eta1, eta2=eta2,
                     R=R, T=T, n=mid, alpha=alpha, sides=sides,
                     p1=p1) >= power:
            hi = mid
        else:
            lo = mid
    return hi, _lf_power(lam1=lam1, lam2=lam2, eta1=eta1, eta2=eta2,
                        R=R, T=T, n=hi, alpha=alpha, sides=sides, p1=p1)


def logrank_lachin_foulkes(
    *,
    s1: float | None = None,
    s2: float | None = None,
    lambda1: float | None = None,
    lambda2: float | None = None,
    t_accrual: float,
    t_followup: float,
    t_study: float | None = None,
    t0: float = 1.0,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    p1: float | None = None,
    allocation: float | None = None,
    loss_to_followup: float = 0.0,
    loss_to_followup_2: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-group survival sample size (Lachin & Foulkes 1986).

    Specify the survival distributions either via proportions surviving
    past ``t0`` (``s1``, ``s2``) or directly via exponential hazards
    (``lambda1``, ``lambda2``).  ``t_accrual`` is the recruitment window
    R; ``t_followup`` is the additional follow-up after accrual closes
    (T - R).  ``t_study`` may be supplied as a cross-check (T = R +
    follow-up).  Allocation may be given as ``p1`` (fraction in group 1)
    or ``allocation`` (n2/n1).  Loss-to-follow-up is an exponential
    dropout rate calibrated as a proportion lost over ``t0`` time units
    When ``loss_to_followup_2`` is
    omitted both groups share the rate.
    """
    # ---- Effect size ----------------------------------------------------
    if (s1 is None) != (s2 is None):
        raise ValueError("supply both s1 and s2 (or neither)")
    if (lambda1 is None) != (lambda2 is None):
        raise ValueError("supply both lambda1 and lambda2 (or neither)")
    if s1 is not None and lambda1 is not None:
        raise ValueError("supply survival proportions OR hazards, not both")
    if s1 is None and lambda1 is None:
        raise ValueError("supply (s1, s2) or (lambda1, lambda2)")

    if t0 <= 0:
        raise ValueError("t0 must be > 0")
    if s1 is not None:
        if not (0 < s1 < 1 and 0 < s2 < 1 and s1 != s2):
            raise ValueError("s1, s2 must be in (0, 1) and differ")
        lam1 = -math.log(s1) / t0
        lam2 = -math.log(s2) / t0
    else:
        if lambda1 <= 0 or lambda2 <= 0 or lambda1 == lambda2:
            raise ValueError("lambda1, lambda2 must be > 0 and differ")
        lam1, lam2 = float(lambda1), float(lambda2)

    # ---- Duration -------------------------------------------------------
    if t_accrual < 0 or t_followup < 0:
        raise ValueError("t_accrual and t_followup must be >= 0")
    if t_accrual == 0 and t_followup == 0:
        raise ValueError("t_accrual + t_followup must be > 0")
    T_total = t_accrual + t_followup
    if t_study is not None and not math.isclose(t_study, T_total, abs_tol=1e-9):
        raise ValueError(
            f"t_study={t_study!r} disagrees with t_accrual+t_followup="
            f"{T_total!r}"
        )

    # ---- Allocation -----------------------------------------------------
    if p1 is not None and allocation is not None:
        raise ValueError("supply only one of (p1, allocation), not both")
    if allocation is not None:
        if allocation <= 0:
            raise ValueError("allocation (n2/n1) must be > 0")
        p1 = 1.0 / (1.0 + allocation)
    elif p1 is None:
        p1 = 0.5
    if not 0 < p1 < 1:
        raise ValueError("p1 must be in (0, 1)")

    # ---- Loss-to-follow-up ---------------------------------------------
    if not 0.0 <= loss_to_followup < 1.0:
        raise ValueError("loss_to_followup must be in [0, 1)")
    if loss_to_followup_2 is None:
        loss_to_followup_2 = loss_to_followup
    if not 0.0 <= loss_to_followup_2 < 1.0:
        raise ValueError("loss_to_followup_2 must be in [0, 1)")
    eta1 = -math.log(1.0 - loss_to_followup) / t0 if loss_to_followup > 0 else 0.0
    eta2 = -math.log(1.0 - loss_to_followup_2) / t0 if loss_to_followup_2 > 0 else 0.0

    inputs_echo = {
        "s1": s1, "s2": s2, "lambda1": lam1, "lambda2": lam2,
        "t_accrual": t_accrual, "t_followup": t_followup,
        "t_study": T_total, "t0": t0, "n": n, "alpha": alpha,
        "power": power, "sides": sides, "p1": p1,
        "loss_to_followup": loss_to_followup,
        "loss_to_followup_2": loss_to_followup_2,
    }

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    hr = lam2 / lam1  # treatment vs control on hazard scale

    if solve_for == "power":
        assert n is not None
        achieved = _lf_power(lam1=lam1, lam2=lam2, eta1=eta1, eta2=eta2,
                             R=t_accrual, T=T_total, n=n, alpha=alpha,
                             sides=sides, p1=p1)
        n_total = n
    elif solve_for == "n":
        assert power is not None
        n_total, achieved = _lf_n_for_power(
            lam1=lam1, lam2=lam2, eta1=eta1, eta2=eta2,
            R=t_accrual, T=T_total, alpha=alpha, power=power,
            sides=sides, p1=p1,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    n1 = round(n_total * p1)
    n2 = n_total - n1
    return {
        "method_id": "logrank_lachin_foulkes",
        "solve_for": solve_for,
        "n": n_total,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "hazard_ratio": hr,
        "lambda1": lam1,
        "lambda2": lam2,
        "inputs_echo": inputs_echo,
        "citations": [
            "Lachin, J.M. & Foulkes, M.A. (1986). Evaluation of sample size "
            "and power for analyses of survival with allowance for "
            "nonuniform patient entry, losses to follow-up, noncompliance, "
            "and stratification. Biometrics 42:507-519.",
        ],
    }


# =========================================================================
# Non-Inferiority and Equivalence Tests for the Difference of Two Hazard
# Rates Assuming an Exponential Model (exponential survival)
# =========================================================================
#
# Uses the unconditional Chow-Shao-Wang (2008) / Lachin-Foulkes (1986)
# method.  The variance of the MLE of an exponential hazard rate with
# uniform accrual R, total study time T, and loss-to-followup hazard ω is:
#
#   σ²(h, ω, R, T) = h² / E[d | h, ω, R, T]
#
# where E[d | h, ω, R, T] = (h/(h+ω)) *
#     (1 + exp(-(h+ω)T) * (1 - exp((h+ω)R)) / ((h+ω)*R))
#
# The test statistic:
#   Z = ((h2 - h1) - Δ) / sqrt(σ²(h1)/N1 + σ²(h2)/N2)
#
# Under H_a:
#   z_{1-β} = ((h2 - h1) + Δ) / sqrt(σ²(h1)/N1 + σ²(h2)/N2) - z_{1-α}
#


def _sigma2_chow(h: float, omega: float, R: float, T: float) -> float:
    """Variance of MLE of exponential hazard h under uniform accrual.

    Chow, Shao, Wang (2008) eq, based on Lachin & Foulkes (1986).
    When ω = 0: σ²(h) = h² / E[d].
    """
    s = h + omega
    if s <= 0.0:
        return float("inf")
    if R <= 0.0:
        # instantaneous enrolment; E[d] = (h/s)*(1 - e^{-sT})
        e_d = (h / s) * (-math.expm1(-s * T))
    else:
        follow = T - R
        # E[d] = (h/s) * (1 + e^{-sT}*(1 - e^{sR}) / (s*R))
        e_d = (h / s) * (
            1.0 + math.exp(-s * T) * (-math.expm1(s * R)) / (s * R)
        )
    if e_d <= 0.0:
        return float("inf")
    return h * h / e_d


def _lf_hz_power(
    *,
    h1: float, h2: float, delta: float,
    omega1: float, omega2: float,
    R: float, T: float,
    n: int, alpha: float, sides: int, p1: float,
) -> float:
    """Power of a one-sided NI test: Ha: h2 - h1 < Δ."""
    n1 = max(1, round(n * p1))
    n2 = n - n1
    if n1 < 1 or n2 < 1:
        return 0.0
    s1 = _sigma2_chow(h1, omega1, R, T)
    s2 = _sigma2_chow(h2, omega2, R, T)
    denom = math.sqrt(s1 / n1 + s2 / n2)
    if denom <= 0.0:
        return 0.0
    k = 1 if sides == 1 else 2
    z_a = D.norm_ppf(1.0 - alpha / k)
    # Under Ha: h2 < h1 + Δ  =>  power = Φ((Δ - (h2-h1)) / σ - z_α)
    effect = delta - (h2 - h1)          # positive when H_a holds
    z_beta = effect / denom - z_a
    from scipy.stats import norm as _n
    return float(_n.cdf(z_beta))


def _lf_hz_n_for_power(
    *,
    h1: float, h2: float, delta: float,
    omega1: float, omega2: float,
    R: float, T: float,
    alpha: float, power: float, sides: int, p1: float,
    n_min: int = 4, n_max: int = 10_000_000,
) -> tuple[int, float]:
    lo, hi = n_min, n_min
    while hi <= n_max:
        if _lf_hz_power(h1=h1, h2=h2, delta=delta, omega1=omega1,
                        omega2=omega2, R=R, T=T, n=hi, alpha=alpha,
                        sides=sides, p1=p1) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _lf_hz_power(h1=h1, h2=h2, delta=delta, omega1=omega1,
                        omega2=omega2, R=R, T=T, n=mid, alpha=alpha,
                        sides=sides, p1=p1) >= power:
            hi = mid
        else:
            lo = mid
    # Ensure n1 + n2 is achievable with integer group sizes:
    # bump N until round(N*p1) + (N - round(N*p1)) gives sufficient power.
    n_total = hi
    while True:
        achieved = _lf_hz_power(h1=h1, h2=h2, delta=delta, omega1=omega1,
                                 omega2=omega2, R=R, T=T, n=n_total,
                                 alpha=alpha, sides=sides, p1=p1)
        if achieved >= power:
            break
        n_total += 1
    return n_total, _lf_hz_power(h1=h1, h2=h2, delta=delta, omega1=omega1,
                                  omega2=omega2, R=R, T=T, n=n_total,
                                  alpha=alpha, sides=sides, p1=p1)


def non_inferiority_two_hazard_rates(
    *,
    h1: float,
    h2: float | None = None,
    diff: float = 0.0,
    delta: float,
    t_accrual: float,
    t_followup: float,
    omega1: float = 0.0,
    omega2: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    p1: float = 0.5,
    sides: int = 1,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test comparing two exponential hazard rates.

    Non-inferiority test for the difference of two hazard rates
    assuming an exponential model.

    H0: h2 - h1 >= Δ  vs  Ha: h2 - h1 < Δ  (lower hazard better).

    Parameters
    ----------
    h1
        Control group hazard rate.
    h2
        Treatment hazard rate. If None, ``diff`` (= h2 - h1) is used.
    diff
        Difference h2 - h1 (used when h2 is None).
    delta
        Non-inferiority margin (Δ > 0; boundary = h1 + Δ).
    t_accrual
        Recruitment period R.
    t_followup
        Follow-up period (T - R).
    omega1, omega2
        Loss-to-follow-up hazard rates.  Default 0.
    """
    if omega2 is None:
        omega2 = omega1
    if h2 is None:
        h2 = h1 + diff
    T_total = t_accrual + t_followup
    inputs_echo = dict(h1=h1, h2=h2, diff=h2 - h1, delta=delta,
                       t_accrual=t_accrual, t_followup=t_followup,
                       T_total=T_total, omega1=omega1, omega2=omega2,
                       alpha=alpha, power=power, n=n, p1=p1, sides=sides)

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _lf_hz_power(h1=h1, h2=h2, delta=delta, omega1=omega1,
                                 omega2=omega2, R=t_accrual, T=T_total,
                                 n=n, alpha=alpha, sides=sides, p1=p1)
        n_total = n
    else:
        assert power is not None
        n_total, achieved = _lf_hz_n_for_power(
            h1=h1, h2=h2, delta=delta, omega1=omega1, omega2=omega2,
            R=t_accrual, T=T_total, alpha=alpha, power=power,
            sides=sides, p1=p1,
        )

    n1 = round(n_total * p1)
    n2 = n_total - n1
    return {
        "method_id": "non_inferiority_two_hazard_rates",
        "solve_for": solve_for,
        "n": n_total,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Two Hazard Rates Assuming an Exponential Model",
            "Chow, S.C., Shao, J., Wang, H. (2008). Sample Size Calculations "
            "in Clinical Research, 2nd Ed. Chapman & Hall/CRC.",
            "Lachin, J.M. & Foulkes, M.A. (1986). Evaluation of sample size "
            "and power for analyses of survival. Biometrics 42:507-519.",
        ],
    }


def _lf_eq_power(
    *,
    h1: float, h2: float, delta: float,
    omega1: float, omega2: float,
    R: float, T: float,
    n: int, alpha: float, p1: float,
) -> float:
    """TOST power for equivalence of two exponential hazard rates."""
    from scipy.stats import norm as _n
    n1 = max(1, round(n * p1))
    n2 = n - n1
    if n1 < 1 or n2 < 1:
        return 0.0
    s1 = _sigma2_chow(h1, omega1, R, T)
    s2 = _sigma2_chow(h2, omega2, R, T)
    denom = math.sqrt(s1 / n1 + s2 / n2)
    if denom <= 0.0:
        return 0.0
    z_a = D.norm_ppf(1.0 - alpha)
    diff = h2 - h1
    # Power of TOST: Φ(z_{β,lower}) + Φ(z_{β,upper}) - 1
    z1 = (delta - diff) / denom - z_a    # upper side
    z2 = (delta + diff) / denom - z_a    # lower side
    return max(0.0, float(_n.cdf(z1) + _n.cdf(z2) - 1.0))


def equivalence_two_hazard_rates(
    *,
    h1: float,
    h2: float | None = None,
    diff: float = 0.0,
    delta: float,
    t_accrual: float,
    t_followup: float,
    omega1: float = 0.0,
    omega2: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    p1: float = 0.5,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Equivalence test comparing two exponential hazard rates (TOST).

    Equivalence test for the difference of two hazard rates
    assuming an exponential model (TOST).

    H0: |h2 - h1| >= Δ  vs  Ha: |h2 - h1| < Δ.

    Parameters
    ----------
    h1
        Control group hazard rate.
    h2
        Treatment hazard rate. If None, ``diff`` (= h2 - h1) is used.
    diff
        Difference h2 - h1 (usually 0).
    delta
        Equivalence margin (Δ > 0).
    """
    if omega2 is None:
        omega2 = omega1
    if h2 is None:
        h2 = h1 + diff
    T_total = t_accrual + t_followup
    inputs_echo = dict(h1=h1, h2=h2, diff=h2 - h1, delta=delta,
                       t_accrual=t_accrual, t_followup=t_followup,
                       T_total=T_total, omega1=omega1, omega2=omega2,
                       alpha=alpha, power=power, n=n, p1=p1)

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _lf_eq_power(h1=h1, h2=h2, delta=delta, omega1=omega1,
                                 omega2=omega2, R=t_accrual, T=T_total,
                                 n=n, alpha=alpha, p1=p1)
        n_total = n
    else:
        assert power is not None
        lo, hi = 4, 4
        while hi <= 10_000_000:
            if _lf_eq_power(h1=h1, h2=h2, delta=delta, omega1=omega1,
                            omega2=omega2, R=t_accrual, T=T_total,
                            n=hi, alpha=alpha, p1=p1) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _lf_eq_power(h1=h1, h2=h2, delta=delta, omega1=omega1,
                            omega2=omega2, R=t_accrual, T=T_total,
                            n=mid, alpha=alpha, p1=p1) >= power:
                hi = mid
            else:
                lo = mid
        n_total = hi
        achieved = _lf_eq_power(h1=h1, h2=h2, delta=delta, omega1=omega1,
                                 omega2=omega2, R=t_accrual, T=T_total,
                                 n=n_total, alpha=alpha, p1=p1)

    n1 = round(n_total * p1)
    n2 = n_total - n1
    return {
        "method_id": "equivalence_two_hazard_rates",
        "solve_for": solve_for,
        "n": n_total,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Hazard Rates Assuming an Exponential Model",
            "Chow, S.C., Shao, J., Wang, H. (2008). Sample Size Calculations "
            "in Clinical Research, 2nd Ed. Chapman & Hall/CRC.",
            "Lachin, J.M. & Foulkes, M.A. (1986). Evaluation of sample size "
            "and power for analyses of survival. Biometrics 42:507-519.",
        ],
    }


# =========================================================================
# Logrank Tests Accounting for Competing Risks (Pintilie 2003)
# =========================================================================
#
# Based on Pintilie (2002/2006).  With competing risks the probability of
# observing the event of interest for subject in group i is:
#
#   Pr_ev,i = (h_ev,i / (h_ev,i + h_cr,i)) *
#             (1 - exp(-(h_ev,i + h_cr,i)*(T-R))*(e^{...} - 1)/((h_ev,i+h_cr,i)*R))
#
# where the second factor is the LF event probability formula.
# The overall event probability Pr_ev = p1*Pr_ev1 + (1-p1)*Pr_ev2.
# Required events: E = ((z_α + z_β) / log(HR))² / (p1*(1-p1)).
# Total N = E / Pr_ev, adjusted for loss W:  N_adj = N / (1 - W).
# Hazard rates are inferred from cumulative incidences at time T0.


def _hev_hcr_from_fev_fcr(
    fev: float, fcr: float, t0: float,
) -> tuple[float, float]:
    """Compute cause-specific hazard rates from cumulative incidences.

    Pintilie (2003) formula for competing risks:
        h_ev = Fev * (-ln(1 - Fev - Fcr)) / (t0 * (Fev + Fcr))
        h_cr = Fcr * (-ln(1 - Fev - Fcr)) / (t0 * (Fev + Fcr))
    """
    f_tot = fev + fcr
    if f_tot <= 0.0 or f_tot >= 1.0:
        raise ValueError("Fev + Fcr must be in (0, 1)")
    if fev < 0.0 or fcr < 0.0:
        raise ValueError("Fev and Fcr must be >= 0")
    factor = -math.log(1.0 - f_tot) / (t0 * f_tot)
    h_ev = fev * factor
    h_cr = fcr * factor
    return h_ev, h_cr


def _pr_event_competing(
    h_ev: float, h_cr: float, R: float, T: float,
) -> float:
    """Pr(event of interest) under competing risks with uniform accrual.

    Pintilie (2003) formula:
        Pr_ev,i = (h_ev/(h_ev+h_cr)) *
                  (1 - (exp(-(T-R)*(h_ev+h_cr)) - exp(-T*(h_ev+h_cr)))
                       / (R*(h_ev+h_cr)))
    """
    h_tot = h_ev + h_cr
    if h_tot <= 0.0:
        return 0.0
    follow = T - R
    if R <= 0.0:
        # All subjects enrolled at t=0
        return (h_ev / h_tot) * (-math.expm1(-h_tot * T))
    num = math.exp(-follow * h_tot) - math.exp(-T * h_tot)
    return (h_ev / h_tot) * (1.0 - num / (R * h_tot))


def _competing_risks_power(
    *,
    h_ev1: float, h_ev2: float, h_cr1: float, h_cr2: float,
    R: float, T_followup: float, T: float,
    n: int, p1: float, alpha: float, sides: int,
    loss: float = 0.0,
) -> float:
    """Power of logrank test for event of interest with competing risks."""
    pr1 = _pr_event_competing(h_ev1, h_cr1, R, T)
    pr2 = _pr_event_competing(h_ev2, h_cr2, R, T)
    pr_ev = p1 * pr1 + (1.0 - p1) * pr2
    if pr_ev <= 0.0:
        return 0.0
    n_eff = n * (1.0 - loss)
    events = n_eff * pr_ev
    hr = h_ev2 / h_ev1
    if hr <= 0.0 or hr == 1.0:
        return 0.0
    k = 1 if sides == 1 else 2
    z_a = D.norm_ppf(1.0 - alpha / k)
    z_beta = (math.sqrt(events * p1 * (1.0 - p1)) * abs(math.log(hr))
              - z_a)
    from scipy.stats import norm as _n
    return float(_n.cdf(z_beta))


def competing_risks_logrank(
    *,
    fev1: float,
    fev2: float | None = None,
    fcr1: float,
    fcr2: float | None = None,
    hr: float | None = None,
    t0: float,
    t_accrual: float,
    t_followup: float,
    loss: float = 0.0,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    p1: float = 0.5,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Logrank test for event of interest accounting for competing risks.

    Logrank test accounting for competing risks (Pintilie 2003).

    Parameters
    ----------
    fev1
        Cumulative incidence of event of interest in control at t0.
    fev2
        Cumulative incidence in treatment at t0. Supply either ``fev2``
        or ``hr``; if ``hr`` is given then hev2 = hr * hev1.
    fcr1
        Cumulative incidence of competing risks in control at t0.
    fcr2
        Cumulative incidence of competing risks in treatment at t0.
        Defaults to fcr1 (equal competing risks).
    hr
        Hazard ratio hev2 / hev1 (used when fev2 is None).
    t0
        Time point for cumulative incidence specification.
    t_accrual
        Accrual period R.
    t_followup
        Follow-up period T - R.
    loss
        Proportion lost to follow-up W.
    """
    if fcr2 is None:
        fcr2 = fcr1
    T_total = t_accrual + t_followup

    h_ev1, h_cr1 = _hev_hcr_from_fev_fcr(fev1, fcr1, t0)

    if fev2 is not None:
        h_ev2, h_cr2 = _hev_hcr_from_fev_fcr(fev2, fcr2, t0)
        hr_actual = h_ev2 / h_ev1
    elif hr is not None:
        h_ev2 = hr * h_ev1
        # competing risk hazard for treatment from fcr2 and the same total rate
        _, h_cr2 = _hev_hcr_from_fev_fcr(fcr2, fcr2, t0) if False else (0.0, 0.0)
        # Use marginal: h_cr2 comes from fcr2 alone (no competing event available)
        # but factor requires fev2 too.  Use: solve s.t. h_ev2+h_cr2 gives Fcr2.
        # Simplest: assume h_ev2/h_cr2 ratio same as control (equal competing risk
        # hazard rate assumption when HR input used).
        h_cr2 = h_cr1
        hr_actual = hr
    else:
        raise ValueError("supply either fev2 or hr")

    inputs_echo = dict(fev1=fev1, fev2=fev2, fcr1=fcr1, fcr2=fcr2,
                       hr=hr_actual, t0=t0, t_accrual=t_accrual,
                       t_followup=t_followup, loss=loss,
                       alpha=alpha, power=power, n=n, p1=p1, sides=sides)

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def _pw(n_val: int) -> float:
        return _competing_risks_power(
            h_ev1=h_ev1, h_ev2=h_ev2, h_cr1=h_cr1, h_cr2=h_cr2,
            R=t_accrual, T_followup=t_followup, T=T_total,
            n=n_val, p1=p1, alpha=alpha, sides=sides, loss=loss,
        )

    if solve_for == "power":
        assert n is not None
        achieved = _pw(n)
        n_total = n
    else:
        assert power is not None
        lo, hi = 4, 4
        while hi <= 100_000_000:
            if _pw(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _pw(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_total = hi
        achieved = _pw(n_total)

    n1 = round(n_total * p1)
    n2 = n_total - n1
    pr1 = _pr_event_competing(h_ev1, h_cr1, t_accrual, T_total)
    pr2 = _pr_event_competing(h_ev2, h_cr2, t_accrual, T_total)
    pr_ev = p1 * pr1 + (1.0 - p1) * pr2
    events = int(round(n_total * (1.0 - loss) * pr_ev))
    return {
        "method_id": "competing_risks_logrank",
        "solve_for": solve_for,
        "n": n_total,
        "n1": n1,
        "n2": n2,
        "events": events,
        "achieved_power": achieved,
        "hazard_ratio": hr_actual,
        "inputs_echo": inputs_echo,
        "citations": [
            "Pintilie, M. (2006). Competing Risks: A Practical Perspective. "
            "Wiley.",
            "Pintilie, M. (2002). Dealing with competing risks: testing "
            "hypotheses. Stat Med 21:3317-3324.",
        ],
    }


# =========================================================================
# Non-Inferiority Logrank Tests
# =========================================================================
#
# Jung et al. (2005) power formula:
#
#   1 - β = Φ( ((HR0-1)*D*Q1*Q2 - z_{1-α}*sqrt(HR0)) / (Q1 + Q2*HR0) )
#
# where D is total observed events.  D is obtained from N via the Lakatos
# Markov model (implemented here using the Lachin-Foulkes uniform-accrual
# formula for E[d]):
#
#   E[d]_i = (h_i / (h_i + ω_i)) *
#            (1 - (exp(-(h_i+ω_i)*(T-R)) - exp(-(h_i+ω_i)*T)) /
#                 ((h_i+ω_i)*R))
#
#   E[d] = Q1*E[d]_1 + Q2*E[d]_2
#   D = N * E[d]


def _jung_power(
    *, hr0: float, hr: float, h1: float,
    omega1: float, omega2: float,
    R: float, T: float,
    n: int, alpha: float, q1: float,
) -> float:
    """Power of the non-inferiority logrank test (Jung et al. 2005).

    Uses Jung (2005) approximate formula (Eq. 3.2):
        1 - beta = Phi( (HR0-1)*sqrt(D*Q1*Q2/HR0) - z_alpha )

    where D = N * E[d], E[d] computed via Lachin-Foulkes with per-group
    loss-to-follow-up hazards omega1, omega2.
    """
    if n < 2:
        return 0.0
    q2 = 1.0 - q1
    h2 = hr * h1
    # Expected event rates per group (Lachin-Foulkes formula)
    ed1 = _prob_event_lf(h1, omega1, R, T)
    ed2 = _prob_event_lf(h2, omega2, R, T)
    ed = q1 * ed1 + q2 * ed2
    if ed <= 0:
        return 0.0
    d_events = n * ed
    z_alpha = D.norm_ppf(1.0 - alpha)
    # Jung (2005) Eq: power = Phi( (HR0-1)*sqrt(D*Q1*Q2/HR0) - z_alpha )
    inner = d_events * q1 * q2 / hr0
    if inner <= 0:
        return 0.0
    z_beta = (hr0 - 1.0) * math.sqrt(inner) - z_alpha
    from scipy.stats import norm as _n
    return float(_n.cdf(z_beta))


def _jung_n(
    *, power: float, hr0: float, hr: float, h1: float,
    omega1: float, omega2: float,
    R: float, T: float,
    alpha: float, q1: float,
    n_max: int = 10_000_000,
) -> tuple[int, float]:
    lo, hi = 4, 4
    while hi <= n_max:
        if _jung_power(hr0=hr0, hr=hr, h1=h1, omega1=omega1, omega2=omega2,
                       R=R, T=T, n=hi, alpha=alpha, q1=q1) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _jung_power(hr0=hr0, hr=hr, h1=h1, omega1=omega1, omega2=omega2,
                       R=R, T=T, n=mid, alpha=alpha, q1=q1) >= power:
            hi = mid
        else:
            lo = mid
    achieved = _jung_power(hr0=hr0, hr=hr, h1=h1, omega1=omega1, omega2=omega2,
                            R=R, T=T, n=hi, alpha=alpha, q1=q1)
    return hi, achieved


def non_inferiority_logrank(
    *,
    h1: float,
    hr0: float,
    hr: float = 1.0,
    t_accrual: float,
    t_total: float,
    omega1: float = 0.0,
    omega2: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    q1: float = 0.5,
    sides: int = 1,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority two-sample logrank test.

    Uses Jung et al. (2005) power formula with Lakatos Markov / Lachin-Foulkes
    event-rate computation.

    H0: HR >= HR0  vs  H1: HR < HR0  (HR0 > 1, events are failures)

    Parameters
    ----------
    h1
        Hazard rate of the reference (control) group.
    hr0
        Non-inferiority HR bound (> 1).
    hr
        Actual (true) hazard ratio h2/h1.  Default 1.0 (equal groups).
    t_accrual
        Accrual period R (in the same time units as h1).
    t_total
        Total study duration T (T > R).
    omega1, omega2
        Loss-to-follow-up hazard rates. If omega2 is None, uses omega1.
    alpha
        One-sided significance level.
    power
        Target power. Required when solve_for='n'.
    n
        Total sample size. Required when solve_for='power'.
    q1
        Proportion of total N in the reference group. Default 0.5.
    sides
        Must be 1 (one-sided NI test).
    solve_for
        'n' or 'power'.
    """
    if sides != 1:
        raise ValueError("Non-inferiority logrank is always one-sided (sides=1)")
    if hr0 <= 1.0:
        raise ValueError("hr0 must be > 1 for non-inferiority")
    if h1 <= 0:
        raise ValueError("h1 must be > 0")
    if t_total <= t_accrual:
        raise ValueError("t_total must be > t_accrual")
    if not 0 < q1 < 1:
        raise ValueError("q1 must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if omega2 is None:
        omega2 = omega1

    inputs_echo = dict(h1=h1, hr0=hr0, hr=hr, t_accrual=t_accrual,
                       t_total=t_total, omega1=omega1, omega2=omega2,
                       alpha=alpha, power=power, n=n, q1=q1, sides=sides)

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _jung_power(hr0=hr0, hr=hr, h1=h1, omega1=omega1,
                                omega2=omega2, R=t_accrual, T=t_total,
                                n=n, alpha=alpha, q1=q1)
        n_total = n
    elif solve_for == "n":
        assert power is not None
        n_total, achieved = _jung_n(power=power, hr0=hr0, hr=hr, h1=h1,
                                     omega1=omega1, omega2=omega2,
                                     R=t_accrual, T=t_total,
                                     alpha=alpha, q1=q1)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    n1 = round(n_total * q1)
    n2 = n_total - n1
    return {
        "method_id": "non_inferiority_logrank",
        "solve_for": solve_for,
        "n": n_total,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Jung, S.H., Kang, S.J., McCall, L.M., Blumenstein, B. (2005). "
            "Sample sizes computation for two-sample noninferiority log-rank "
            "test. J Biopharmaceut Stat 15:969-979.",
            "Lakatos, E. (1988). Sample sizes based on the log-rank statistic "
            "in complex clinical trials. Biometrics 44:229-241.",
        ],
    }


# =========================================================================
# One-Sample Logrank Tests
# =========================================================================
#
# Wu (2015) formulas for a one-sample logrank test against a historical
# control.  Survival times follow a Weibull distribution with shape k.
#
#   L = (O - E) / sqrt(E)   ~ N(0,1) under H0
#
#   Power ≈ Φ( -σ0/σ * z_{1-α} - ω*sqrt(n)/σ )
#
#   where (integrals computed numerically):
#     ω  = σ1² - σ0²
#     σ² = p1 - p1² + 2*p00 - p0² - 2*p01 + 2*p0*p1
#     σ0² = p0,  σ1² = p1
#     p0  = ∫ G(t) S1(t) λ0(t) dt
#     p1  = ∫ G(t) S1(t) λ1(t) dt
#     p00 = ∫ G(t) S1(t) Λ0(t) λ0(t) dt
#     p01 = ∫ G(t) S1(t) Λ0(t) λ1(t) dt
#
# Uniform accrual censoring:
#   G(t) = 1              if t <= tf
#         (ta+tf-t)/ta    if tf < t <= ta+tf
#         0               otherwise
#
# Weibull S(t) = exp(-λ t^k)


def _wu_G(t: float, ta: float, tf: float) -> float:
    """Censoring survival function G(t) under uniform accrual."""
    if t <= tf:
        return 1.0
    if t <= ta + tf:
        return (ta + tf - t) / ta
    return 0.0


def _wu_integrals(
    lam0: float, lam1: float, k: float, ta: float, tf: float,
    n_steps: int = 2000,
) -> tuple[float, float, float, float]:
    """Compute (p0, p1, p00, p01) by numerical integration (trapezoidal)."""
    t_max = ta + tf
    dt = t_max / n_steps
    p0 = p1 = p00 = p01 = 0.0
    # Trapezoidal: integrate from 0 to t_max
    for i in range(n_steps + 1):
        t = i * dt
        g = _wu_G(t, ta, tf)
        if g <= 0:
            continue
        tk = t ** k
        s1 = math.exp(-lam1 * tk)
        lam0_t = k * lam0 * (t ** (k - 1)) if t > 0 else 0.0
        lam1_t = k * lam1 * (t ** (k - 1)) if t > 0 else 0.0
        Lambda0_t = lam0 * tk
        w = dt if (0 < i < n_steps) else dt / 2.0
        gS1 = g * s1
        p0 += w * gS1 * lam0_t
        p1 += w * gS1 * lam1_t
        p00 += w * gS1 * Lambda0_t * lam0_t
        p01 += w * gS1 * Lambda0_t * lam1_t
    return p0, p1, p00, p01


def _wu_power(
    *, lam0: float, lam1: float, k: float,
    ta: float, tf: float, n: int,
    alpha: float, sides: int,
) -> float:
    """Wu (2015) power for the one-sample logrank test."""
    if n < 2:
        return 0.0
    p0, p1, p00, p01 = _wu_integrals(lam0, lam1, k, ta, tf)
    sigma0_sq = p0
    sigma1_sq = p1
    omega = sigma1_sq - sigma0_sq
    sigma_sq = p1 - p1 ** 2 + 2.0 * p00 - p0 ** 2 - 2.0 * p01 + 2.0 * p0 * p1
    if sigma_sq <= 0:
        return 0.0
    sigma = math.sqrt(sigma_sq)
    sigma0 = math.sqrt(max(0.0, sigma0_sq))
    k_sides = 1 if sides == 1 else 2
    z_alpha = D.norm_ppf(1.0 - alpha / k_sides)
    from scipy.stats import norm as _n
    power_val = float(_n.cdf(-sigma0 / sigma * z_alpha - omega * math.sqrt(n) / sigma))
    return power_val


def _wu_n(
    *, lam0: float, lam1: float, k: float,
    ta: float, tf: float, alpha: float, power: float, sides: int,
    n_max: int = 1_000_000,
) -> tuple[int, float]:
    lo, hi = 2, 2
    while hi <= n_max:
        if _wu_power(lam0=lam0, lam1=lam1, k=k, ta=ta, tf=tf,
                     n=hi, alpha=alpha, sides=sides) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _wu_power(lam0=lam0, lam1=lam1, k=k, ta=ta, tf=tf,
                     n=mid, alpha=alpha, sides=sides) >= power:
            hi = mid
        else:
            lo = mid
    achieved = _wu_power(lam0=lam0, lam1=lam1, k=k, ta=ta, tf=tf,
                          n=hi, alpha=alpha, sides=sides)
    return hi, achieved


def one_sample_logrank(
    *,
    lambda0: float | None = None,
    lambda1: float | None = None,
    hr: float | None = None,
    m0: float | None = None,
    m1: float | None = None,
    k: float = 1.0,
    ta: float,
    tf: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """One-sample logrank test against a historical control.

    Uses Wu (2015) power/sample-size formula for Weibull survival with
    uniform accrual.

    Parameters
    ----------
    lambda0
        Hazard rate of the historical control (Weibull scale parameter).
    lambda1
        Hazard rate of the new treatment group.  Supply either lambda1 or hr.
    hr
        Hazard ratio lambda1/lambda0.  Used when lambda1 is None.
    m0
        Median survival time of the historical control.  Alternative to
        lambda0 (requires k).  lambda0 = log(2) / m0^k.
    m1
        Median survival time of the new group.  Alternative to lambda1.
    k
        Weibull shape parameter.  Default 1 (exponential).
    ta
        Accrual time.
    tf
        Follow-up time (after last subject enrolled).
    alpha
        Significance level.
    power
        Target power. Required when solve_for='n'.
    n
        Sample size. Required when solve_for='power'.
    sides
        1 or 2.
    solve_for
        'n' or 'power'.
    """
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if k <= 0:
        raise ValueError("k must be > 0")
    if ta < 0 or tf < 0:
        raise ValueError("ta and tf must be >= 0")
    if ta + tf <= 0:
        raise ValueError("ta + tf must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    # Resolve lambda0
    if m0 is not None:
        if m0 <= 0:
            raise ValueError("m0 must be > 0")
        lam0 = math.log(2.0) / (m0 ** k)
    elif lambda0 is not None:
        if lambda0 <= 0:
            raise ValueError("lambda0 must be > 0")
        lam0 = float(lambda0)
    else:
        raise ValueError("supply lambda0 or m0")

    # Resolve lambda1
    if m1 is not None:
        if m1 <= 0:
            raise ValueError("m1 must be > 0")
        lam1 = math.log(2.0) / (m1 ** k)
    elif lambda1 is not None:
        if lambda1 <= 0:
            raise ValueError("lambda1 must be > 0")
        lam1 = float(lambda1)
    elif hr is not None:
        if hr <= 0 or hr == 1.0:
            raise ValueError("hr must be > 0 and != 1")
        lam1 = hr * lam0
    else:
        raise ValueError("supply lambda1, m1, or hr")

    hr_actual = lam1 / lam0

    inputs_echo = dict(lambda0=lam0, lambda1=lam1, hr=hr_actual,
                       k=k, ta=ta, tf=tf,
                       alpha=alpha, power=power, n=n, sides=sides)

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _wu_power(lam0=lam0, lam1=lam1, k=k, ta=ta, tf=tf,
                              n=n, alpha=alpha, sides=sides)
        n_out = n
    elif solve_for == "n":
        assert power is not None
        n_out, achieved = _wu_n(lam0=lam0, lam1=lam1, k=k, ta=ta, tf=tf,
                                 alpha=alpha, power=power, sides=sides)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    # Expected events at solved/given N
    p0, p1_val, _, _ = _wu_integrals(lam0, lam1, k, ta, tf)
    events = int(round(n_out * p1_val))

    return {
        "method_id": "one_sample_logrank",
        "solve_for": solve_for,
        "n": n_out,
        "events": events,
        "achieved_power": achieved,
        "hazard_ratio": hr_actual,
        "inputs_echo": inputs_echo,
        "citations": [
            "Wu, J. (2015). Sample size calculation for the one-sample "
            "log-rank test. Pharmaceutical Statistics 14:26-33.",
            "Wu, J. (2014). A new one-sample log-rank test. "
            "J Biomet Biostat 5:210.",
        ],
    }
