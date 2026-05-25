"""Sample-size routines for correlated (paired) proportions — NI and equivalence.


- Ch 160: Non-Inferiority Tests for the Difference Between Two Correlated
          Proportions (Liu et al. 2002 / Nam 1997 score method)
- Ch 165: Equivalence Tests for the Difference Between Two Correlated
          Proportions (TOST, Liu et al. 2002)
- Ch 161: Non-Inferiority Tests for the Ratio of Two Correlated Proportions
          (Nam & Blackwelder 2002)

The 2×2 paired table:
    Experimental\\ Standard |  Yes   |  No
    Yes                    | p11    | p10    -> PT
    No                     | p01    | p00    -> 1-PT
                             PS      1-PS

Power uses the normal approximation to the multinomial.

All three methods accept ``nuisance_type`` and ``nuisance_value`` to specify
the fourth table parameter (nuisance parameter). Supported types:
    "p01", "p10", "p11", "p00", "p01_p10", "p11_p00", "sensitivity"
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm as _norm

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Helpers to resolve table probabilities from nuisance specification
# ---------------------------------------------------------------------------

def _resolve_table(
    ps: float, pt: float,
    nuisance_type: str, nuisance_value: float,
) -> tuple[float, float, float, float]:
    """Return (p11, p10, p01, p00) from marginals and nuisance parameter.

    The 2×2 table satisfies:
        p11 + p10 = PT
        p01 + p11 = PS
        p11 + p10 + p01 + p00 = 1
    Given one of these seven nuisance types we resolve all four.
    """
    nt = nuisance_type.lower()
    if nt == "p01":
        p01 = nuisance_value
        p11 = ps - p01
        p10 = pt - p11
        p00 = 1.0 - p11 - p10 - p01
    elif nt == "p10":
        p10 = nuisance_value
        p11 = pt - p10
        p01 = ps - p11
        p00 = 1.0 - p11 - p10 - p01
    elif nt == "p11":
        p11 = nuisance_value
        p10 = pt - p11
        p01 = ps - p11
        p00 = 1.0 - p11 - p10 - p01
    elif nt == "p00":
        p00 = nuisance_value
        p11 = 1.0 - p00 - (1.0 - pt) - (1.0 - ps)
        # p11 = PS + PT - 1 + p00 is not right; use:
        # p11 + p10 = PT, p01 + p11 = PS, p00 = 1 - (p11+p10+p01)
        # p11 = 1 - p00 - p10 - p01; but we need another eq.
        # Assume p10 + p01 = 1 - p00 - p11 => circular
        # Use: p00 given => p11 = PS + PT - 1 + p00  (from summing constraints)
        p11 = ps + pt - 1.0 + p00
        p10 = pt - p11
        p01 = ps - p11
    elif nt in ("p01_p10", "p01+p10"):
        # p01 + p10 = nuisance_value; assume p01 = p10 = nuisance/2
        p10 = nuisance_value / 2.0
        p01 = nuisance_value / 2.0
        p11 = pt - p10
        p00 = 1.0 - p11 - p10 - p01
    elif nt in ("p11_p00", "p11+p00"):
        # concordant fraction; assume p11 proportional to PS, p00 to (1-PS)
        conc = nuisance_value
        disc = 1.0 - conc
        # p11/(p11+p00) ≈ PS => p11 ≈ conc * PS / (PS + 1 - PS) = conc * PS
        p11 = conc * ps
        p00 = conc - p11
        p10 = pt - p11
        p01 = ps - p11
    elif nt in ("sensitivity", "p11/ps"):
        sens = nuisance_value
        p11 = sens * ps
        p10 = pt - p11
        p01 = ps - p11
        p00 = 1.0 - p11 - p10 - p01
    else:
        raise ValueError(f"Unknown nuisance_type: {nuisance_type!r}")
    return p11, p10, p01, p00


# ---------------------------------------------------------------------------
# Chapter 160: Non-Inferiority for Difference  (Liu et al. 2002 / Nam 1997)
# ---------------------------------------------------------------------------

def _ni_diff_cL_parts(
    p01: float, p10: float, da: float, d_ni: float,
) -> tuple[float, float]:
    """Compute p_bar_{L,01} and w_L for the NI lower bound formula."""
    a_L = -da * (1.0 - d_ni) - 2.0 * (p01 + d_ni)
    b_L = d_ni * (1.0 + d_ni) * p01
    discriminant = a_L ** 2 - 8.0 * b_L
    if discriminant < 0:
        discriminant = 0.0
    p_bar_L01 = (-a_L + math.sqrt(discriminant)) / 4.0
    # w_L
    numerator_w = 2.0 * p01 + da - da ** 2
    denominator_w = 2.0 * p_bar_L01 - d_ni - d_ni ** 2
    if denominator_w <= 0:
        return p_bar_L01, 0.0
    w_L = math.sqrt(max(0.0, numerator_w / denominator_w))
    return p_bar_L01, w_L


def _ni_diff_power(
    *, n: int, da: float, d_ni: float,
    p01: float, p10: float, alpha: float,
) -> float:
    """Normal-approximation power for NI difference test (one-sided)."""
    if n < 2:
        return 0.0
    if da <= -d_ni:
        return 0.0
    sigma = math.sqrt((p01 + p10 - da ** 2) / n)
    if sigma <= 0:
        return 0.0
    z_alpha = D.norm_ppf(1.0 - alpha)
    _, w_L = _ni_diff_cL_parts(p01=p01, p10=p10, da=da, d_ni=d_ni)
    if w_L <= 0:
        return 0.0
    c_L = (-da / sigma - d_ni / sigma + z_alpha / w_L)
    return float(1.0 - _norm.cdf(c_L))


def _ni_diff_n(
    *, power: float, da: float, d_ni: float,
    p01: float, p10: float, alpha: float,
    n_max: int = 1_000_000,
) -> tuple[int, float]:
    lo, hi = 2, 2
    while hi <= n_max:
        if _ni_diff_power(n=hi, da=da, d_ni=d_ni,
                          p01=p01, p10=p10, alpha=alpha) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _ni_diff_power(n=mid, da=da, d_ni=d_ni,
                          p01=p01, p10=p10, alpha=alpha) >= power:
            hi = mid
        else:
            lo = mid
    achieved = _ni_diff_power(n=hi, da=da, d_ni=d_ni,
                               p01=p01, p10=p10, alpha=alpha)
    return hi, achieved


def non_inferiority_paired_proportions(
    *,
    ps: float,
    da: float = 0.0,
    d_ni: float,
    nuisance_type: str = "p01",
    nuisance_value: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 1,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for difference of two correlated proportions.

    Uses the Nam (1997) RMLE-based score statistic
    (normal approximation).

    H0: PT - PS <= -|Dni|  vs  H1: PT - PS > -|Dni|

    Parameters
    ----------
    ps
        Standard (reference) proportion.
    da
        Actual difference PT - PS. Often 0.
    d_ni
        Non-inferiority margin (positive).
    nuisance_type
        Which nuisance parameter is specified: 'p01', 'p10', 'p11', 'p00',
        'p01_p10', 'p11_p00', or 'sensitivity'.
    nuisance_value
        Value of the nuisance parameter.
    alpha
        One-sided significance level.
    power
        Target power. Required when solve_for='n'.
    n
        Sample size. Required when solve_for='power'.
    sides
        Must be 1 (one-sided NI test).
    solve_for
        'n' or 'power'.
    """
    if sides != 1:
        raise ValueError("Non-inferiority is always one-sided (sides=1)")
    if d_ni <= 0:
        raise ValueError("d_ni must be > 0")
    if not 0 < ps < 1:
        raise ValueError("ps must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    pt = ps + da
    _, p10, p01, _ = _resolve_table(ps=ps, pt=pt,
                                     nuisance_type=nuisance_type,
                                     nuisance_value=nuisance_value)

    inputs_echo = dict(ps=ps, pt=pt, da=da, d_ni=d_ni,
                       nuisance_type=nuisance_type,
                       nuisance_value=nuisance_value,
                       alpha=alpha, power=power, n=n, sides=sides)

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _ni_diff_power(n=n, da=da, d_ni=d_ni,
                                   p01=p01, p10=p10, alpha=alpha)
        n_out = n
    elif solve_for == "n":
        assert power is not None
        n_out, achieved = _ni_diff_n(power=power, da=da, d_ni=d_ni,
                                      p01=p01, p10=p10, alpha=alpha)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "non_inferiority_paired_proportions",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "achieved_width": None,
        "inputs_echo": inputs_echo,
        "citations": [
            "Between Two Correlated Proportions",
            "Liu, J.P., Hsueh, H.M., Hsieh, E., Chen, J.J. (2002). Tests for "
            "equivalence or non-inferiority for paired binary data. Stat Med "
            "21:231-245.",
            "Nam, J.M. (1997). Establishing equivalence of two treatments and "
            "sample size requirements in matched-pairs design. Biometrics "
            "53:1422-1430.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 165: Equivalence (TOST) for Difference of Correlated Proportions
# ---------------------------------------------------------------------------

def _eq_diff_cU_parts(
    p01: float, p10: float, da: float, d_e: float,
) -> tuple[float, float]:
    """Compute p_bar_{U,01} and w_U for the equivalence upper bound formula."""
    a_U = -da * (1.0 + d_e) - 2.0 * (p01 - d_e)
    b_U = -d_e * (1.0 - d_e) * p01
    discriminant = a_U ** 2 - 8.0 * b_U
    if discriminant < 0:
        discriminant = 0.0
    p_bar_U01 = (-a_U + math.sqrt(discriminant)) / 4.0
    numerator_w = 2.0 * p01 + da - da ** 2
    denominator_w = 2.0 * p_bar_U01 + d_e - d_e ** 2
    if denominator_w <= 0:
        return p_bar_U01, 0.0
    w_U = math.sqrt(max(0.0, numerator_w / denominator_w))
    return p_bar_U01, w_U


def _eq_diff_power(
    *, n: int, da: float, d_e: float,
    p01: float, p10: float, alpha: float,
) -> float:
    """Normal-approximation TOST power for equivalence difference test."""
    if n < 2:
        return 0.0
    if abs(da) >= d_e:
        return 0.0
    sigma = math.sqrt((p01 + p10 - da ** 2) / n)
    if sigma <= 0:
        return 0.0
    z_alpha = D.norm_ppf(1.0 - alpha)
    # Lower bound
    _, w_L = _ni_diff_cL_parts(p01=p01, p10=p10, da=da, d_ni=d_e)
    # Upper bound
    _, w_U = _eq_diff_cU_parts(p01=p01, p10=p10, da=da, d_e=d_e)
    if w_L <= 0 or w_U <= 0:
        return 0.0
    c_L = -da / sigma - d_e / sigma + z_alpha / w_L
    c_U = -da / sigma + d_e / sigma - z_alpha / w_U
    diff = c_U - c_L
    if diff <= 0:
        return 0.0
    return max(0.0, float(_norm.cdf(c_U) - _norm.cdf(c_L)))


def _eq_diff_n(
    *, power: float, da: float, d_e: float,
    p01: float, p10: float, alpha: float,
    n_max: int = 2_000_000,
) -> tuple[int, float]:
    lo, hi = 2, 2
    while hi <= n_max:
        if _eq_diff_power(n=hi, da=da, d_e=d_e,
                          p01=p01, p10=p10, alpha=alpha) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _eq_diff_power(n=mid, da=da, d_e=d_e,
                          p01=p01, p10=p10, alpha=alpha) >= power:
            hi = mid
        else:
            lo = mid
    achieved = _eq_diff_power(n=hi, da=da, d_e=d_e,
                               p01=p01, p10=p10, alpha=alpha)
    return hi, achieved


def equivalence_paired_proportions(
    *,
    ps: float,
    da: float = 0.0,
    d_e: float,
    nuisance_type: str = "p01",
    nuisance_value: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Equivalence (TOST) test for difference of two correlated proportions.

    Uses the Nam (1997) RMLE-based score statistic
    (normal approximation).

    H0: |PT - PS| >= De  vs  H1: |PT - PS| < De

    Parameters
    ----------
    ps
        Standard proportion.
    da
        Actual difference PT - PS. Often 0.
    d_e
        Equivalence margin (positive).
    nuisance_type
        'p01', 'p10', 'p11', 'p00', 'p01_p10', 'p11_p00', or 'sensitivity'.
    nuisance_value
        Value of the nuisance parameter.
    alpha
        Significance level (each one-sided test).
    power
        Target power. Required when solve_for='n'.
    n
        Sample size. Required when solve_for='power'.
    sides
        Must be 2 (two-sided equivalence).
    solve_for
        'n' or 'power'.
    """
    if sides != 2:
        raise ValueError("Equivalence is always two-sided (sides=2)")
    if d_e <= 0:
        raise ValueError("d_e must be > 0")
    if not 0 < ps < 1:
        raise ValueError("ps must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    pt = ps + da
    _, p10, p01, _ = _resolve_table(ps=ps, pt=pt,
                                     nuisance_type=nuisance_type,
                                     nuisance_value=nuisance_value)

    inputs_echo = dict(ps=ps, pt=pt, da=da, d_e=d_e,
                       nuisance_type=nuisance_type,
                       nuisance_value=nuisance_value,
                       alpha=alpha, power=power, n=n, sides=sides)

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _eq_diff_power(n=n, da=da, d_e=d_e,
                                   p01=p01, p10=p10, alpha=alpha)
        n_out = n
    elif solve_for == "n":
        assert power is not None
        n_out, achieved = _eq_diff_n(power=power, da=da, d_e=d_e,
                                      p01=p01, p10=p10, alpha=alpha)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "equivalence_paired_proportions",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "achieved_width": None,
        "inputs_echo": inputs_echo,
        "citations": [
            "Between Two Correlated Proportions",
            "Liu, J.P., Hsueh, H.M., Hsieh, E., Chen, J.J. (2002). Tests for "
            "equivalence or non-inferiority for paired binary data. Stat Med "
            "21:231-245.",
            "Nam, J.M. (1997). Establishing equivalence of two treatments and "
            "sample size requirements in matched-pairs design. Biometrics "
            "53:1422-1430.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 161: Non-Inferiority for Ratio of Correlated Proportions
#              Nam & Blackwelder (2002)
# ---------------------------------------------------------------------------

def _resolve_p11_p10_p01_from_sensitivity(
    ps: float, pt: float, sensitivity: float,
) -> tuple[float, float, float]:
    p11 = sensitivity * ps
    p10 = pt - p11
    p01 = ps - p11
    return p11, p10, p01


def _ni_ratio_power(
    *, n: int, ra: float, r_ni: float,
    ps: float, pt: float,
    p11: float, p10: float, p01: float, p00: float,
    alpha: float,
) -> float:
    """Normal-approximation power for NI ratio test (Nam & Blackwelder 2002).

    Uses the re-parameterised form of the Nam & Blackwelder (2002) formula:
        power = Phi( E1*sqrt(n/V1_num) - z_alpha*sqrt(V0_num/V1_num) )

    where V0_num, E1, V1_num are the n-free quantities (multiply by n for
    the actual variances/expectations).
    """
    if n < 2:
        return 0.0
    if ra <= r_ni:
        return 0.0
    # CMLE p_bar_{10} under H0 at R=r_ni (Nam & Blackwelder 2002, Eq 3)
    disc = (pt - r_ni ** 2 * ps) ** 2 + 4.0 * r_ni ** 2 * p10 * p01
    if disc < 0:
        return 0.0
    p_bar10 = (-pt + r_ni ** 2 * (ps + 2.0 * p10) + math.sqrt(disc)) / (
        2.0 * r_ni * (r_ni + 1.0))
    p_bar01 = r_ni * p_bar10 - (r_ni - 1.0) * (1.0 - p00)

    # Scale-free (n-free) quantities
    v0_num = r_ni * (p_bar10 + p_bar01)         # = n * V_0(T_0)
    e1 = (ra - r_ni) * ps                        # mean of numerator of Z
    v1_num = ((ra + r_ni ** 2) * ps - 2.0 * r_ni * p11
              - (ra - r_ni) ** 2 * ps ** 2)      # = n * V_1(T_0)

    if v0_num <= 0 or v1_num <= 0:
        return 0.0

    z_alpha = D.norm_ppf(1.0 - alpha)
    # power = Phi( E1*sqrt(n/V1_num) - z_alpha*sqrt(V0_num/V1_num) )
    c_u = e1 * math.sqrt(n / v1_num) - z_alpha * math.sqrt(v0_num / v1_num)
    return float(_norm.cdf(c_u))


def _ni_ratio_n(
    *, power: float, ra: float, r_ni: float,
    ps: float, pt: float,
    p11: float, p10: float, p01: float, p00: float,
    alpha: float,
    n_max: int = 10_000_000,
) -> tuple[int, float]:
    lo, hi = 2, 2
    while hi <= n_max:
        if _ni_ratio_power(n=hi, ra=ra, r_ni=r_ni, ps=ps, pt=pt,
                           p11=p11, p10=p10, p01=p01, p00=p00,
                           alpha=alpha) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _ni_ratio_power(n=mid, ra=ra, r_ni=r_ni, ps=ps, pt=pt,
                           p11=p11, p10=p10, p01=p01, p00=p00,
                           alpha=alpha) >= power:
            hi = mid
        else:
            lo = mid
    achieved = _ni_ratio_power(n=hi, ra=ra, r_ni=r_ni, ps=ps, pt=pt,
                                p11=p11, p10=p10, p01=p01, p00=p00,
                                alpha=alpha)
    return hi, achieved


def non_inferiority_paired_proportions_ratio(
    *,
    ps: float,
    ra: float = 1.0,
    r_ni: float,
    nuisance_type: str = "p10",
    nuisance_value: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 1,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for ratio of two correlated proportions.

    Uses the Nam & Blackwelder (2002) score statistic
    (normal approximation).

    H0: PT/PS <= Rni  vs  H1: PT/PS > Rni  (Rni < 1)

    Parameters
    ----------
    ps
        Standard proportion.
    ra
        Actual ratio PT/PS. Often 1.0.
    r_ni
        Non-inferiority ratio (< 1).
    nuisance_type
        Which nuisance parameter is specified: 'p01', 'p10', 'p11', 'p00',
        'p01_p10', 'p11_p00', or 'sensitivity'.
    nuisance_value
        Value of the nuisance parameter.
    alpha
        One-sided significance level.
    power
        Target power. Required when solve_for='n'.
    n
        Sample size. Required when solve_for='power'.
    sides
        Must be 1 (one-sided NI test).
    solve_for
        'n' or 'power'.
    """
    if sides != 1:
        raise ValueError("Non-inferiority is always one-sided (sides=1)")
    if not (0 < r_ni < 1):
        raise ValueError("r_ni must be in (0, 1) for non-inferiority")
    if not 0 < ps < 1:
        raise ValueError("ps must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    pt = ps * ra
    p11, p10, p01, p00 = _resolve_table(ps=ps, pt=pt,
                                         nuisance_type=nuisance_type,
                                         nuisance_value=nuisance_value)

    inputs_echo = dict(ps=ps, pt=pt, ra=ra, r_ni=r_ni,
                       nuisance_type=nuisance_type,
                       nuisance_value=nuisance_value,
                       alpha=alpha, power=power, n=n, sides=sides)

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _ni_ratio_power(n=n, ra=ra, r_ni=r_ni, ps=ps, pt=pt,
                                    p11=p11, p10=p10, p01=p01, p00=p00,
                                    alpha=alpha)
        n_out = n
    elif solve_for == "n":
        assert power is not None
        n_out, achieved = _ni_ratio_n(power=power, ra=ra, r_ni=r_ni, ps=ps,
                                       pt=pt, p11=p11, p10=p10, p01=p01,
                                       p00=p00, alpha=alpha)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "non_inferiority_paired_proportions_ratio",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "achieved_width": None,
        "inputs_echo": inputs_echo,
        "citations": [
            "Two Correlated Proportions",
            "Nam, J.M. and Blackwelder, W.C. (2002). Analysis of the ratio "
            "of marginal probabilities in a matched-pair setting. Stat Med "
            "21:689-699.",
        ],
    }


# ---------------------------------------------------------------------------
# Equivalence (TOST) for Ratio of Two Correlated Proportions
# Nam & Blackwelder (2002) / Tango (1998) — two one-sided ratio tests
# ---------------------------------------------------------------------------
#
# TOST: H0: PT/PS <= Re  or  PT/PS >= 1/Re  (where Re in (0,1))
#       Ha: Re < PT/PS < 1/Re  (ratio equivalence)
#
# Power = Phi(E1*sqrt(n/V1) - z_alpha*sqrt(V0/V1))
#         evaluated separately for the lower (H1: R > Re) and upper
#         (H2: R < 1/Re) one-sided tests, then combined:
#
#   Power = max(0, P_lower + P_upper - 1)
#
# Each one-sided test uses the Nam & Blackwelder (2002) NI-ratio power
# formula:  _ni_ratio_power(...).  The lower test uses r_ni=Re (the
# equivalence ratio) and the upper test uses r_ni=1/Re inverted
# (i.e. Re < 1 for "upper test: PT < (1/Re)*PS" which is equivalent to
# "PS/PT > Re").

def _eq_ratio_power(
    *, n: int, ra: float, re: float,
    ps: float, pt: float,
    p11: float, p10: float, p01: float, p00: float,
    alpha: float,
) -> float:
    """TOST power for equivalence on ratio of two correlated proportions.

    Nam & Blackwelder (2002) score statistic applied as two one-sided tests.
    """
    if n < 2:
        return 0.0
    if not (re < ra < 1.0 / re):
        return 0.0
    # Lower one-sided test: H0: R <= Re  (re < 1)
    p_lower = _ni_ratio_power(
        n=n, ra=ra, r_ni=re,
        ps=ps, pt=pt,
        p11=p11, p10=p10, p01=p01, p00=p00,
        alpha=alpha,
    )
    # Upper one-sided test: H0: R >= 1/Re  equivalent to H0: PS/PT <= Re
    # Swap roles: PS_upper = pt, PT_upper = ps, R_upper = 1/ra, r_ni_upper = re
    re_upper = re  # 1/(1/re) = re, swapped direction
    ra_upper = 1.0 / ra
    # Rebuild table with swapped roles
    p11_u, p10_u, p01_u, p00_u = p11, p01, p10, p00  # swap p10 and p01
    p_upper = _ni_ratio_power(
        n=n, ra=ra_upper, r_ni=re_upper,
        ps=pt, pt=ps,
        p11=p11_u, p10=p10_u, p01=p01_u, p00=p00_u,
        alpha=alpha,
    )
    return max(0.0, p_lower + p_upper - 1.0)


def _eq_ratio_n(
    *, power: float, ra: float, re: float,
    ps: float, pt: float,
    p11: float, p10: float, p01: float, p00: float,
    alpha: float,
    n_max: int = 10_000_000,
) -> tuple[int, float]:
    lo, hi = 2, 2
    while hi <= n_max:
        if _eq_ratio_power(n=hi, ra=ra, re=re, ps=ps, pt=pt,
                           p11=p11, p10=p10, p01=p01, p00=p00,
                           alpha=alpha) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _eq_ratio_power(n=mid, ra=ra, re=re, ps=ps, pt=pt,
                           p11=p11, p10=p10, p01=p01, p00=p00,
                           alpha=alpha) >= power:
            hi = mid
        else:
            lo = mid
    achieved = _eq_ratio_power(n=hi, ra=ra, re=re, ps=ps, pt=pt,
                                p11=p11, p10=p10, p01=p01, p00=p00,
                                alpha=alpha)
    return hi, achieved


def equivalence_paired_proportions_ratio(
    *,
    ps: float,
    ra: float = 1.0,
    re: float,
    nuisance_type: str = "p10",
    nuisance_value: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Equivalence (TOST) test for ratio of two correlated proportions.

    Equivalence Tests for the Ratio of Two Correlated Proportions.
    Uses the Nam & Blackwelder (2002) score statistic applied as two
    one-sided ratio tests.

    H0: PT/PS <= Re  or  PT/PS >= 1/Re  vs  Ha: Re < PT/PS < 1/Re

    Parameters
    ----------
    ps
        Standard proportion.
    ra
        Actual ratio PT/PS.  Default 1.0 (PT = PS).
    re
        Equivalence ratio (0 < re < 1).  The equivalence region is
        (re, 1/re) on the ratio scale.
    nuisance_type
        Which nuisance parameter is specified: 'p01', 'p10', 'p11', 'p00',
        'p01_p10', 'p11_p00', or 'sensitivity'.
    nuisance_value
        Value of the nuisance parameter.
    alpha
        Significance level for each one-sided test.
    power
        Target power.  Required when solve_for='n'.
    n
        Sample size.  Required when solve_for='power'.
    sides
        Must be 2 (two-sided equivalence test).
    solve_for
        'n' or 'power'.
    """
    if sides != 2:
        raise ValueError("Equivalence ratio test is always two-sided (sides=2)")
    if not 0 < re < 1:
        raise ValueError("re must be in (0, 1)")
    if not 0 < ps < 1:
        raise ValueError("ps must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    pt = ps * ra
    p11, p10, p01, p00 = _resolve_table(ps=ps, pt=pt,
                                         nuisance_type=nuisance_type,
                                         nuisance_value=nuisance_value)

    inputs_echo = dict(ps=ps, pt=pt, ra=ra, re=re,
                       nuisance_type=nuisance_type,
                       nuisance_value=nuisance_value,
                       alpha=alpha, power=power, n=n, sides=sides)

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _eq_ratio_power(n=n, ra=ra, re=re, ps=ps, pt=pt,
                                    p11=p11, p10=p10, p01=p01, p00=p00,
                                    alpha=alpha)
        n_out = n
    elif solve_for == "n":
        assert power is not None
        n_out, achieved = _eq_ratio_n(power=power, ra=ra, re=re, ps=ps,
                                       pt=pt, p11=p11, p10=p10, p01=p01,
                                       p00=p00, alpha=alpha)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "equivalence_paired_proportions_ratio",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "achieved_width": None,
        "inputs_echo": inputs_echo,
        "citations": [
            "Nam, J.M. and Blackwelder, W.C. (2002). Analysis of the ratio "
            "of marginal probabilities in a matched-pair setting. "
            "Stat Med 21:689-699.",
            "Tango, T. (1998). Equivalence test and confidence interval for "
            "the difference in proportions for the paired-sample design. "
            "Stat Med 17:891-908.",
        ],
    }
