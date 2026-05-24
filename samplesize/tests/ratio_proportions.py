"""Tests for the ratio of two independent proportions (relative risk).


* "Non-Inferiority Tests for the Ratio of Two Proportions"
* "Equivalence Tests for the Ratio of Two Proportions"
* "Superiority by a Margin Tests for the Ratio of Two Proportions"

Note: "Tests for the Ratio
of Two Proportions" — a generic two-sided test on the relative risk is
implemented here as a thin wrapper around the superiority-by-margin
machinery with the null ratio fixed at 1.

All methods use the Farrington & Manning (1990) likelihood-score test
for the relative risk under the **normal-approximation** branch
("Approximate" power calculation):

    z_FMR(phi_0) = (p̂1/p̂2 - phi_0)
                   / sqrt(tilde-p1·tilde-q1/n1 + phi_0² · tilde-p2·tilde-q2/n2)

where (tilde-p1, tilde-p2) are the constrained MLEs that satisfy
tilde-p1 / tilde-p2 = phi_0.  The constrained MLE for tilde-p2 solves
the quadratic A·p² + B·p + C = 0 from Farrington & Manning (1990) /
Miettinen & Nurminen (1985):

    A = N · phi_0
    B = -[n1·phi_0 + x11 + n2 + x21·phi_0]
    C = x11 + x21
    tilde-p2 = (-B - sqrt(B² - 4AC)) / (2A)
    tilde-p1 = phi_0 · tilde-p2

For the "Approximate" power calculation, substitute the true
proportions p1.1 and p2 for the sample proportions, giving a closed-form
power expression

    Power = Φ((|sgn|·(p1 − phi_0·p2) − z_α · SE0) / SE1)

with

    SE0 = sqrt(tilde-p1·tilde-q1/n1 + phi_0² · tilde-p2·tilde-q2/n2)   (constrained, H0)
    SE1 = sqrt(p1·q1/n1 + phi_0² · p2·q2/n2)                          (true, H1)

This module reproduces the worked examples in chapters 211 (NI), 212
to
five decimal places (see ``tests/validation/fixtures``).
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Constrained MLEs and Farrington-Manning power core
# ---------------------------------------------------------------------------

def _constrained_mle_ratio(
    p1: float, p2: float, n1: int, n2: int, phi0: float,
) -> tuple[float, float]:
    """Constrained MLEs (tilde-p1, tilde-p2) under H0: p1/p2 = phi_0.

    Implements Farrington & Manning (1990) / Miettinen & Nurminen (1985):

        A = N · phi_0
        B = -(n1·phi_0 + x11 + n2 + x21·phi_0)
        C = x11 + x21
        tilde-p2 = (-B - sqrt(B² - 4AC)) / (2A)
        tilde-p1 = phi_0 · tilde-p2

    where ``x11 = n1 · p1`` and ``x21 = n2 · p2`` (replacing the observed
    counts by the true proportions for the asymptotic power formula).
    """
    if phi0 <= 0:
        raise ValueError("phi0 must be > 0")
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must be >= 2")
    N = n1 + n2
    x11 = n1 * p1
    x21 = n2 * p2
    m1 = x11 + x21
    A = N * phi0
    B = -(n1 * phi0 + x11 + n2 + x21 * phi0)
    C = m1
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        disc = 0.0
    p2t = (-B - math.sqrt(disc)) / (2.0 * A)
    # Numerical guard: clamp into (eps, 1-eps).
    eps = 1e-12
    p2t = min(1.0 - eps, max(eps, p2t))
    p1t = phi0 * p2t
    p1t = min(1.0 - eps, max(eps, p1t))
    return p1t, p2t


def _fm_se_h0(p1: float, p2: float, n1: int, n2: int, phi0: float) -> float:
    """Standard error of phi-hat under H0 using constrained MLEs."""
    p1t, p2t = _constrained_mle_ratio(p1, p2, n1, n2, phi0)
    return math.sqrt(
        p1t * (1.0 - p1t) / n1
        + phi0 * phi0 * p2t * (1.0 - p2t) / n2
    )


def _fm_se_h1(p1: float, p2: float, n1: int, n2: int, phi0: float) -> float:
    """Standard error of the F&M numerator under H1 (true p1, p2)."""
    return math.sqrt(
        p1 * (1.0 - p1) / n1
        + phi0 * phi0 * p2 * (1.0 - p2) / n2
    )


def _fm_one_sided_power(
    p1: float, p2: float, n1: int, n2: int,
    phi0: float, alpha: float, direction: str,
) -> float:
    """One-sided F&M likelihood-score power, normal-approximation branch.

    direction:
        "upper"  →  H1: phi > phi_0 (reject when z > +z_alpha)
        "lower"  →  H1: phi < phi_0 (reject when z < -z_alpha)
    """
    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("p1, p2 must be in (0, 1)")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if n1 < 2 or n2 < 2:
        return 0.0
    from scipy.stats import norm
    se0 = _fm_se_h0(p1, p2, n1, n2, phi0)
    se1 = _fm_se_h1(p1, p2, n1, n2, phi0)
    z_alpha = D.norm_ppf(1.0 - alpha)
    diff = p1 - phi0 * p2
    if direction == "upper":
        return float(norm.cdf((diff - z_alpha * se0) / se1))
    if direction == "lower":
        return float(norm.cdf((-diff - z_alpha * se0) / se1))
    raise ValueError(f"unsupported direction: {direction!r}")


def _fm_equivalence_power(
    p1: float, p2: float, n1: int, n2: int,
    phi_lower: float, phi_upper: float, alpha: float,
) -> float:
    """TOST F&M equivalence power on the relative-risk scale.

    Two one-sided F&M tests at significance ``alpha`` each:

        Test L:  H0: phi <= phi_lower  vs  H1: phi > phi_lower
        Test U:  H0: phi >= phi_upper  vs  H1: phi < phi_upper

    Equivalence is concluded when both reject.  Treating the two test
    statistics as comonotone (the usual large-sample TOST simplification
    for the "Approximate" branch), the joint power is

        P(both reject) ≈ Phi(z_L) + Phi(z_U) - 1

    where z_L and z_U are the two one-sided F&M statistics' shift terms.
    """
    if phi_lower <= 0 or phi_upper <= 0:
        raise ValueError("phi_lower and phi_upper must be > 0")
    if phi_upper <= phi_lower:
        raise ValueError("phi_upper must be > phi_lower")
    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("p1, p2 must be in (0, 1)")
    if n1 < 2 or n2 < 2:
        return 0.0
    from scipy.stats import norm
    z_alpha = D.norm_ppf(1.0 - alpha)
    se0_L = _fm_se_h0(p1, p2, n1, n2, phi_lower)
    se0_U = _fm_se_h0(p1, p2, n1, n2, phi_upper)
    se1_L = _fm_se_h1(p1, p2, n1, n2, phi_lower)
    se1_U = _fm_se_h1(p1, p2, n1, n2, phi_upper)
    diff_L = p1 - phi_lower * p2
    diff_U = p1 - phi_upper * p2
    pL = float(norm.cdf((diff_L - z_alpha * se0_L) / se1_L))
    pU = float(norm.cdf((-diff_U - z_alpha * se0_U) / se1_U))
    return max(0.0, pL + pU - 1.0)


# ---------------------------------------------------------------------------
# Generic sample-size bisection helper
# ---------------------------------------------------------------------------

def _solve_n_bisect(
    power_fn, target_power: float, allocation: float = 1.0,
    n_min: int = 4, n_max: int = 10_000_000,
) -> tuple[int, int, float]:
    """Bracket-and-bisect for the smallest n1 (with n2 = ceil(allocation·n1))
    that achieves at least ``target_power``."""
    if not 0.0 < target_power < 1.0:
        raise ValueError("power must be in (0, 1)")

    def n2_for(n1: int) -> int:
        return max(2, math.ceil(allocation * n1))

    def at(n1: int) -> float:
        return power_fn(n1, n2_for(n1))

    lo, hi = n_min, n_min
    while hi <= n_max:
        if at(hi) >= target_power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError("failed to bracket N within n_max")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if at(mid) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi, n2_for(hi), at(hi)


# ---------------------------------------------------------------------------
# Public methods
# ---------------------------------------------------------------------------

def _direction_for_ni(higher_is_better: bool) -> str:
    """Direction convention for the NI ratio test.

    When "Higher Proportions Are Better" the NI
    ratio R0 < 1 and the alternative is phi > R0 ("upper").  When
    "Higher Proportions Are Worse" R0 > 1 and the alternative is
    phi < R0 ("lower").
    """
    return "upper" if higher_is_better else "lower"


def _direction_for_sup_margin(higher_is_better: bool) -> str:
    """Direction convention for the Superiority-by-margin ratio test.

    When "Higher Proportions Are Better" R0 > 1
    and the alternative is phi > R0 ("upper").  When "Higher
    Proportions Are Worse" R0 < 1 and the alternative is phi < R0
    ("lower").
    """
    return "upper" if higher_is_better else "lower"


def non_inferiority_ratio_two_proportions(
    *,
    p2: float,
    margin: float,
    r1: float | None = None,
    p1: float | None = None,
    alpha: float = 0.025,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    higher_is_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for the ratio of two proportions (relative risk).

    Non-inferiority test for the ratio of two
    Proportions".  Implements the Farrington & Manning likelihood-score
    test, normal-approximation power.

    Parameters
    ----------
    p2
        Reference / control group proportion.  In (0, 1).
    margin
        Non-inferiority **ratio** ``R0 = phi_0`` on the relative-risk
        scale.  Must be positive.  When ``higher_is_better=True`` the
        chapter convention is ``R0 < 1`` (the new treatment is "good
        enough" if its success rate is at least ``R0·p2``).  When
        ``higher_is_better=False`` the chapter convention is ``R0 > 1``
        (the new treatment's failure rate is at most ``R0·p2``).
    r1
        Assumed actual ratio ``p1 / p2`` under H1.  Defaults to 1
        (typical "equal effect under H1" setup).
    p1
        Alternative way to specify the true p1 under H1.  Overrides
        ``r1`` when supplied.
    alpha
        One-sided significance level.  Default 0.025.
    power, n1, n2
        Supply either ``power`` (solve for n) or ``n1``/``n2`` (solve
        for power).
    allocation
        ``n2 = ceil(allocation · n1)`` when only ``n1`` is given.
    higher_is_better
        Direction toggle: ``"higher"`` (default) or ``"lower"``.
    solve_for
        One of ``{"n", "power"}`` — usually inferred.
    """
    if not 0.0 < p2 < 1.0:
        raise ValueError("p2 must be in (0, 1)")
    if margin <= 0.0:
        raise ValueError("margin (R0) must be > 0")
    # Note: some published examples sometimes
    # use R0 values that fall on the "wrong" side of 1 for the nominal
    # "Higher Proportions Are" toggle.  The Farrington-Manning power
    # calculation is well-defined for any positive R0 once a direction is
    # chosen via ``higher_is_better``, so we do NOT enforce a sign-of-R0
    # constraint here.
    phi0 = margin
    if p1 is None:
        r1_val = 1.0 if r1 is None else r1
        if r1_val <= 0:
            raise ValueError("r1 must be > 0")
        p1_val = r1_val * p2
    else:
        if not 0.0 < p1 < 1.0:
            raise ValueError("p1 must be in (0, 1)")
        p1_val = p1
        r1_val = p1_val / p2
    if not 0.0 < p1_val < 1.0:
        raise ValueError("Implied p1 = r1·p2 must lie in (0, 1)")
    direction = _direction_for_ni(higher_is_better)

    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    inputs_echo = {
        "p2": p2, "margin": margin, "r1": r1_val, "p1": p1_val,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2,
        "allocation": allocation, "higher_is_better": higher_is_better,
    }
    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def power_at(nn1: int, nn2: int) -> float:
        return _fm_one_sided_power(
            p1_val, p2, nn1, nn2, phi0, alpha, direction,
        )

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = power_at(n1, n2)
        result = {
            "n1": n1, "n2": n2, "n": n1 + n2,
            "achieved_power": achieved,
        }
    elif solve_for == "n":
        assert power is not None
        n1r, n2r, achieved = _solve_n_bisect(
            power_at, power, allocation=allocation,
        )
        result = {
            "n1": n1r, "n2": n2r, "n": n1r + n2r,
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "non_inferiority_ratio_two_proportions",
        "solve_for": solve_for,
        **result,
        "direction": direction,
        "inputs_echo": inputs_echo,
        "citations": [
            "Farrington, C.P. and Manning, G. (1990). 'Test statistics and "
            "sample size formulae for comparative binomial trials with null "
            "hypothesis of non-zero risk difference or non-unity relative "
            "risk.' Statistics in Medicine, 9, 1447-1454.",
            "Miettinen, O.S. and Nurminen, M. (1985). 'Comparative analysis "
            "of two rates.' Statistics in Medicine, 4, 213-226.",
            "Chow, Shao & Wang (2008). Sample Size Calculations in Clinical "
            "Research, 2nd ed.",
            "Blackwelder, W.C. (1993). 'Sample size and power for prospective "
            "analysis of relative risk.' Statistics in Medicine, 12, 691-698.",
        ],
    }


def equivalence_ratio_two_proportions(
    *,
    p2: float,
    lower_limit: float | None = None,
    upper_limit: float | None = None,
    r1: float = 1.0,
    p1: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Equivalence (TOST) test for the ratio of two proportions.

    Equivalence test for the ratio of two
    Proportions".  Implements two one-sided F&M likelihood-score tests
    in the normal-approximation branch.

    Equivalence limits (R0.L, R0.U) bracket the ratio:

        H0: phi <= R0.L  or  phi >= R0.U
        H1: R0.L < phi = R1 < R0.U

    Parameters
    ----------
    p2
        Reference / control group proportion.
    lower_limit, upper_limit
        Equivalence limits on the ratio scale.  Defaults to a symmetric
        ``(1/RU, RU)`` window when only one is supplied.  If both are
        omitted, defaults to (0.80, 1.25).
    r1
        Assumed actual ratio under H1.  Default 1.0.
    p1
        Alternative way to specify true p1 under H1; overrides ``r1``.
    alpha
        Per one-sided test (TOST) significance level.  Default 0.05.
    power, n1, n2, allocation, solve_for
        As for the NI version.
    """
    if not 0.0 < p2 < 1.0:
        raise ValueError("p2 must be in (0, 1)")
    if lower_limit is None and upper_limit is None:
        lower_limit, upper_limit = 0.80, 1.25
    elif lower_limit is None:
        if upper_limit is None or upper_limit <= 1.0:
            raise ValueError("upper_limit must be > 1")
        lower_limit = 1.0 / upper_limit
    elif upper_limit is None:
        if lower_limit <= 0.0 or lower_limit >= 1.0:
            raise ValueError("lower_limit must lie in (0, 1)")
        upper_limit = 1.0 / lower_limit
    if not (0.0 < lower_limit < 1.0):
        raise ValueError("lower_limit must lie in (0, 1)")
    if upper_limit <= 1.0:
        raise ValueError("upper_limit must be > 1")
    if r1 <= 0.0:
        raise ValueError("r1 must be > 0")
    if p1 is None:
        p1_val = r1 * p2
    else:
        if not 0.0 < p1 < 1.0:
            raise ValueError("p1 must be in (0, 1)")
        p1_val = p1
        r1 = p1_val / p2
    if not 0.0 < p1_val < 1.0:
        raise ValueError("Implied p1 must lie in (0, 1)")

    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    inputs_echo = {
        "p2": p2, "lower_limit": lower_limit, "upper_limit": upper_limit,
        "r1": r1, "p1": p1_val,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2,
        "allocation": allocation,
    }
    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def power_at(nn1: int, nn2: int) -> float:
        return _fm_equivalence_power(
            p1_val, p2, nn1, nn2, lower_limit, upper_limit, alpha,
        )

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = power_at(n1, n2)
        result = {
            "n1": n1, "n2": n2, "n": n1 + n2,
            "achieved_power": achieved,
        }
    elif solve_for == "n":
        assert power is not None
        n1r, n2r, achieved = _solve_n_bisect(
            power_at, power, allocation=allocation,
        )
        result = {
            "n1": n1r, "n2": n2r, "n": n1r + n2r,
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "equivalence_ratio_two_proportions",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Farrington, C.P. and Manning, G. (1990). 'Test statistics and "
            "sample size formulae for comparative binomial trials with null "
            "hypothesis of non-zero risk difference or non-unity relative "
            "risk.' Statistics in Medicine, 9, 1447-1454.",
            "Miettinen, O.S. and Nurminen, M. (1985). 'Comparative analysis "
            "of two rates.' Statistics in Medicine, 4, 213-226.",
            "Chow, S.C., Shao, J. and Wang, H. (2008). Sample Size "
            "Calculations in Clinical Research, 2nd ed.",
        ],
    }


def superiority_by_margin_ratio_two_proportions(
    *,
    p2: float,
    margin: float,
    r1: float | None = None,
    p1: float | None = None,
    alpha: float = 0.025,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    higher_is_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Superiority-by-margin test for the ratio of two proportions.

    Superiority-by-margin test for the ratio of two
    Proportions".  Uses the Farrington & Manning likelihood-score test
    (normal-approximation branch).

    Parameters
    ----------
    p2
        Reference proportion.
    margin
        Superiority-by-margin **ratio** ``R0``.  When
        ``higher_is_better=True`` the chapter convention is ``R0 > 1``
        and the alternative is ``H1: phi = R1 > R0``.  When
        ``higher_is_better=False`` the chapter convention is ``R0 < 1``
        and the alternative is ``H1: phi = R1 < R0``.
    r1
        Assumed actual ratio under H1 (must lie outside the margin in
        the appropriate direction).  Required (no default).
    p1
        Alternative to ``r1``.
    alpha
        One-sided significance level.  Default 0.025.
    power, n1, n2, allocation, higher_is_better, solve_for
        See the NI version.
    """
    if not 0.0 < p2 < 1.0:
        raise ValueError("p2 must be in (0, 1)")
    if margin <= 0.0:
        raise ValueError("margin (R0) must be > 0")
    # Better" and R0 < 1 when "Higher Proportions Are Worse"; we accept
    # any positive R0 and use ``higher_is_better`` purely to set the test
    # direction.
    phi0 = margin
    if p1 is None:
        if r1 is None:
            raise ValueError("supply r1 (or p1) for superiority-by-margin")
        if r1 <= 0:
            raise ValueError("r1 must be > 0")
        p1_val = r1 * p2
        r1_val = r1
    else:
        if not 0.0 < p1 < 1.0:
            raise ValueError("p1 must be in (0, 1)")
        p1_val = p1
        r1_val = p1_val / p2
    if not 0.0 < p1_val < 1.0:
        raise ValueError("Implied p1 = r1·p2 must lie in (0, 1)")
    direction = _direction_for_sup_margin(higher_is_better)

    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    inputs_echo = {
        "p2": p2, "margin": margin, "r1": r1_val, "p1": p1_val,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2,
        "allocation": allocation, "higher_is_better": higher_is_better,
    }
    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def power_at(nn1: int, nn2: int) -> float:
        return _fm_one_sided_power(
            p1_val, p2, nn1, nn2, phi0, alpha, direction,
        )

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = power_at(n1, n2)
        result = {
            "n1": n1, "n2": n2, "n": n1 + n2,
            "achieved_power": achieved,
        }
    elif solve_for == "n":
        assert power is not None
        n1r, n2r, achieved = _solve_n_bisect(
            power_at, power, allocation=allocation,
        )
        result = {
            "n1": n1r, "n2": n2r, "n": n1r + n2r,
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "superiority_by_margin_ratio_two_proportions",
        "solve_for": solve_for,
        **result,
        "direction": direction,
        "inputs_echo": inputs_echo,
        "citations": [
            "Two Proportions.",
            "Farrington, C.P. and Manning, G. (1990). 'Test statistics and "
            "sample size formulae for comparative binomial trials with null "
            "hypothesis of non-zero risk difference or non-unity relative "
            "risk.' Statistics in Medicine, 9, 1447-1454.",
            "Miettinen, O.S. and Nurminen, M. (1985). 'Comparative analysis "
            "of two rates.' Statistics in Medicine, 4, 213-226.",
            "Chow, S.C., Shao, J. and Wang, H. (2008). Sample Size "
            "Calculations in Clinical Research, 2nd ed.",
        ],
    }


def tests_ratio_two_proportions(
    *,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    sides: int = 2,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Generic test for the ratio of two proportions (H0: p1/p2 = 1).

    Note: there is no stand-alone "Tests for the Ratio of Two
    Proportions" chapter.  This wrapper exposes the Farrington & Manning
    likelihood-score test against the null phi_0 = 1, which is
    statistically equivalent to a two-sided test on the relative risk.

    Parameters
    ----------
    p1, p2
        Group proportions under H1.  Must lie in (0, 1).
    alpha
        Significance level (per side when ``sides=2``).
    sides
        1 or 2.  When ``sides=2`` two one-sided F&M
        tests at level alpha/2 and rejects if either does.
    power, n1, n2, allocation, solve_for
        As for the other methods.
    """
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("p1, p2 must be in (0, 1)")
    if p1 == p2:
        raise ValueError("p1 must differ from p2 to compute power")
    direction = "upper" if p1 > p2 else "lower"
    eff_alpha = alpha / 2.0 if sides == 2 else alpha

    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    inputs_echo = {
        "p1": p1, "p2": p2, "alpha": alpha, "sides": sides,
        "power": power, "n1": n1, "n2": n2, "allocation": allocation,
    }
    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def power_at(nn1: int, nn2: int) -> float:
        return _fm_one_sided_power(
            p1, p2, nn1, nn2, 1.0, eff_alpha, direction,
        )

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = power_at(n1, n2)
        result = {
            "n1": n1, "n2": n2, "n": n1 + n2,
            "achieved_power": achieved,
        }
    elif solve_for == "n":
        assert power is not None
        n1r, n2r, achieved = _solve_n_bisect(
            power_at, power, allocation=allocation,
        )
        result = {
            "n1": n1r, "n2": n2r, "n": n1r + n2r,
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "tests_ratio_two_proportions",
        "solve_for": solve_for,
        **result,
        "direction": direction,
        "inputs_echo": inputs_echo,
        "citations": [
            "for the Ratio of Two Proportions (no stand-alone chapter for the "
            "two-sided test against phi=1).",
            "Farrington, C.P. and Manning, G. (1990). Statistics in Medicine, "
            "9, 1447-1454.",
            "Miettinen, O.S. and Nurminen, M. (1985). Statistics in Medicine, "
            "4, 213-226.",
        ],
    }
