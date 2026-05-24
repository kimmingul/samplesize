"""Tests for Two Groups of Pre-Post Scores.

Two independent groups, each measured at two time points (pre and post).
The hypothesis of interest concerns the difference between groups in the
mean change from pre to post.

Reduction
---------
After differencing within each subject, the design collapses to a
two-sample, parallel-group test on the paired differences :math:`d_{ij}
= Y_{ij}^{T2} - Y_{ij}^{T1}`.  Let

.. math::
    \\sigma_{\\mathrm{Diff}}^{2}
        = \\sigma_{T1}^{2} + \\sigma_{T2}^{2}
          - 2\\rho\\,\\sigma_{T1}\\sigma_{T2}

be the variance of the within-subject difference.  Then the test of the
interaction (group-by-time) is equivalent to a two-sample test of the
mean difference :math:`\\delta = \\delta_2 - \\delta_1` (group 2 change
minus group 1 change) with common SD :math:`\\sigma_{\\mathrm{Diff}}`.

Test type
---------
* ``"t"`` (default) - noncentral Student-t with :math:`df = N_1 + N_2
  - 2` (Rosner 2011 Case 2).
* ``"z"`` - normal-approximation Z-test (Rosner 2011 Case 1).

Power formulas (one-sided, :math:`\\delta > 0`):

.. math::
    \\sigma_{\\bar X} = \\sigma_{\\mathrm{Diff}}\\sqrt{1/N_1 + 1/N_2}
    \\qquad \\lambda = \\delta / \\sigma_{\\bar X}

T-test: Power = :math:`1 - T'_{df,\\lambda}(t_{1-\\alpha,df})`.
Z-test: Power = :math:`1 - \\Phi(z_{1-\\alpha} - \\lambda)`.

Two-sided power adds the symmetric lower-tail contribution.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


VALID_TEST_TYPES = {"t", "z"}


def _sigma_diff(sd_t1: float, sd_t2: float, rho: float) -> float:
    var = sd_t1 * sd_t1 + sd_t2 * sd_t2 - 2.0 * rho * sd_t1 * sd_t2
    if var < 0:
        # numerical guard; clamp to zero rather than nan
        var = 0.0
    return math.sqrt(var)


def _pre_post_power(
    *,
    delta: float,
    sd_t1: float,
    sd_t2: float,
    rho: float,
    n1: int,
    n2: int,
    alpha: float,
    sides: int,
    test_type: str,
) -> float:
    if test_type not in VALID_TEST_TYPES:
        raise ValueError(
            f"test_type must be one of {sorted(VALID_TEST_TYPES)}"
        )
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    if sd_t1 <= 0 or sd_t2 <= 0:
        raise ValueError("sd_t1 and sd_t2 must be positive")
    if n1 < 2 or n2 < 2:
        return 0.0

    sigma_d = _sigma_diff(sd_t1, sd_t2, rho)
    if sigma_d <= 0:
        # If sigma_d collapses to 0 the test becomes degenerate; with any
        # non-zero delta the power is 1, otherwise 0.
        return 1.0 if delta != 0 else 0.0

    se = sigma_d * math.sqrt(1.0 / n1 + 1.0 / n2)
    ncp = delta / se

    if test_type == "t":
        df = n1 + n2 - 2.0
        if sides == 2:
            t_crit = D.t_ppf(1.0 - alpha / 2.0, df)
            upper = 1.0 - D.nct_cdf(t_crit, df, ncp)
            lower = D.nct_cdf(-t_crit, df, ncp)
            return upper + lower
        if sides == 1:
            t_crit = D.t_ppf(1.0 - alpha, df)
            if delta >= 0:
                return 1.0 - D.nct_cdf(t_crit, df, ncp)
            return D.nct_cdf(-t_crit, df, ncp)
        raise ValueError(f"sides must be 1 or 2, got {sides}")
    # Z-test branch
    from scipy.stats import norm
    if sides == 2:
        z_crit = D.norm_ppf(1.0 - alpha / 2.0)
        upper = 1.0 - norm.cdf(z_crit - ncp)
        lower = norm.cdf(-z_crit - ncp)
        return float(upper + lower)
    if sides == 1:
        z_crit = D.norm_ppf(1.0 - alpha)
        if delta >= 0:
            return float(1.0 - norm.cdf(z_crit - ncp))
        return float(norm.cdf(-z_crit - ncp))
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def _n_for_pre_post(
    *,
    delta: float,
    sd_t1: float,
    sd_t2: float,
    rho: float,
    alpha: float,
    power: float,
    sides: int,
    allocation: float,
    test_type: str,
    n_min: int = 2,
    n_max: int = 10_000_000,
) -> tuple[int, int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if delta == 0:
        raise ValueError("delta must be non-zero to solve for N")
    if allocation <= 0:
        raise ValueError("allocation (n2/n1) must be > 0")

    def n2_for(n1: int) -> int:
        return max(2, math.ceil(allocation * n1))

    def p_at(n1: int) -> float:
        return _pre_post_power(
            delta=delta, sd_t1=sd_t1, sd_t2=sd_t2, rho=rho,
            n1=n1, n2=n2_for(n1), alpha=alpha, sides=sides,
            test_type=test_type,
        )

    lo, hi = n_min, n_min
    while hi <= n_max:
        if p_at(hi) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if p_at(mid) >= power:
            hi = mid
        else:
            lo = mid

    n1 = hi
    n2 = n2_for(n1)
    return n1, n2, p_at(n1)


def tests_two_groups_pre_post(
    *,
    delta: float,
    sd_t1: float,
    sd_t2: float,
    rho: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    test_type: str = "t",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two groups of pre-post scores.

    Parameters
    ----------
    delta : float
        Difference between the group mean changes,
        :math:`\\delta = \\delta_2 - \\delta_1`.
    sd_t1 : float
        Common standard deviation of the time-1 (baseline) measurements.
    sd_t2 : float
        Common standard deviation of the time-2 (follow-up) measurements.
    rho : float
        Within-subject correlation between the two repeat measurements.
    alpha : float
        Type-I error rate.
    power : float, optional
        Target power.  Supply this OR (``n1``).
    n1, n2 : int, optional
        Group sample sizes.  If only ``n1`` is given, ``n2 = ceil(allocation*n1)``.
    sides : int
        2 (default) or 1.
    allocation : float
        ``n2 / n1`` allocation ratio (default 1.0).
    test_type : str
        ``"t"`` (default) for the noncentral-t formula or ``"z"`` for
        the normal approximation (Rosner 2011).
    solve_for : {"n", "power"}, optional
        Override the auto-detected target.
    """
    inputs_echo = {
        "delta": delta, "sd_t1": sd_t1, "sd_t2": sd_t2, "rho": rho,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2,
        "sides": sides, "allocation": allocation, "test_type": test_type,
    }

    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    elif n2 is not None and n1 is None:
        n1 = max(2, math.ceil(n2 / allocation))

    have_n = n1 is not None and n2 is not None
    have_power = power is not None

    if have_n == have_power:
        raise ValueError("supply exactly one of (power, n1)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _pre_post_power(
            delta=delta, sd_t1=sd_t1, sd_t2=sd_t2, rho=rho,
            n1=n1, n2=n2, alpha=alpha, sides=sides,
            test_type=test_type,
        )
        n1_out, n2_out = n1, n2
    elif solve_for == "n":
        assert power is not None
        n1_out, n2_out, achieved = _n_for_pre_post(
            delta=delta, sd_t1=sd_t1, sd_t2=sd_t2, rho=rho,
            alpha=alpha, power=power, sides=sides,
            allocation=allocation, test_type=test_type,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    sigma_d = _sigma_diff(sd_t1, sd_t2, rho)
    return {
        "method_id": "tests_two_groups_pre_post",
        "solve_for": solve_for,
        "n1": n1_out,
        "n2": n2_out,
        "n": n1_out + n2_out,
        "achieved_power": achieved,
        "sigma_diff": sigma_d,
        "delta": delta,
        "inputs_echo": inputs_echo,
        "citations": [
            "(chapter 432).",
            "Rosner, B. (2011). Fundamentals of Biostatistics, 7th ed. "
            "Brooks/Cole, Boston, MA.",
        ],
    }
