"""Odds-ratio-scale parallel two-proportion tests.

- "Non-Inferiority Tests for the Odds Ratio of Two Proportions"
- "Equivalence Tests for the Odds Ratio of Two Proportions"
- "Superiority by a Margin Tests for the Odds Ratio of Two Proportions"

All three procedures use the Farrington & Manning (1990) likelihood
score statistic for the odds ratio (Farrington & Manning 1990), with
power evaluated via the normal-approximation method.

Mathematical setup
------------------
The odds ratio between the treatment proportion ``p1`` and the
reference proportion ``p2`` is

    psi = [p1 / (1 - p1)] / [p2 / (1 - p2)]

For a target odds ratio ``psi0`` and reference proportion ``p2``, the
implied treatment proportion is

    p1_at(psi, p2) = psi * p2 / (1 - p2 + psi * p2)

Farrington & Manning (1990) likelihood score test statistic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    z_FMO = [ (p1_hat - p1~) / (p1~ q1~) - (p2_hat - p2~) / (p2~ q2~) ]
            / sqrt(1/(n1 p1~ q1~) + 1/(n2 p2~ q2~))

The constrained MLEs ``p1~`` and ``p2~`` satisfy the constraint
``psi~ = psi0`` and are obtained from the quadratic

    A = n2 (psi0 - 1)
    B = n1 psi0 + n2 - m1 (psi0 - 1)
    C = -m1
    p2~ = (-B + sqrt(B^2 - 4AC)) / (2A)
    p1~ = p2~ psi0 / (1 + p2~ (psi0 - 1))

where ``m1 = n1 p1_hat + n2 p2_hat``.

Normal-approximation power formula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a one-sided test ``H0: psi <= psi0`` vs. ``H1: psi > psi0`` (or its
mirror image for lower-tailed), substitute the alternative-hypothesis
proportions ``p1.1 = p1_at(psi1, p2)`` and ``p2`` into the score
statistic.  Then ``E[z_FMO | H1]`` equals the score statistic evaluated
at those substituted values, and

    Var[z_FMO | H1] = (sigma_unconstrained / sigma_constrained)^2

where ``sigma_unconstrained^2 = 1/(n1 p1.1 q1.1) + 1/(n2 p2 q2)`` and
``sigma_constrained^2 = 1/(n1 p1~ q1~) + 1/(n2 p2~ q2~)``.

The asymptotic power for an upper-tailed test is therefore

    Power = 1 - Phi( (z_alpha - z_FMO_at_H1) / (sigma_u / sigma_c) ).

For an equivalence (TOST) test with limits ``psi_L < 1 < psi_U`` we
apply this construction twice and combine via the Schuirmann-style
TOST approximation

    Power = max(0, P_L + P_U - 1).

References
----------
- Farrington & Manning (1990). Statistics in Medicine.
  Proportions (Chapter 212).

  Proportions (Chapter 215).

  Two Proportions (Chapter 207).
- Farrington, C.P. & Manning, G. (1990). "Test Statistics and Sample
  Size Formulae for Comparative Binomial Trials with Null Hypothesis
  of Non-Zero Risk Difference or Non-Unity Relative Risk."
  Statistics in Medicine 9: 1447-1454.
- Miettinen, O.S. & Nurminen, M. (1985). "Comparative analysis of two
  rates." Statistics in Medicine 4: 213-226.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# -------------------- shared helpers ---------------------------------------

def _p1_from_or(psi: float, p2: float) -> float:
    """Treatment proportion implied by odds ratio ``psi`` and reference p2."""
    return psi * p2 / (1.0 - p2 + psi * p2)


def _fm_score_components(
    n1: int, n2: int, p1_hat: float, p2_hat: float, psi0: float,
) -> tuple[float, float, float]:
    """Return (z_FMO_at_phat, sigma_constrained, sigma_unconstrained).

    All quantities follow the Farrington & Manning (1990) score statistic
    for the odds ratio at constraint ``psi0``.  When ``psi0 == 1`` the
    coefficient ``A`` in the quadratic vanishes and the constrained MLEs
    collapse to the pooled-proportion MLE; the limiting forms are used
    in that degenerate case.
    """
    if psi0 <= 0:
        raise ValueError("psi0 must be > 0")
    m1 = n1 * p1_hat + n2 * p2_hat
    if abs(psi0 - 1.0) < 1e-12:
        # Pooled proportion: tilde p1 == tilde p2 == m1/N
        p_tilde = m1 / (n1 + n2)
        p1_tilde = p2_tilde = p_tilde
    else:
        A = n2 * (psi0 - 1.0)
        B = n1 * psi0 + n2 - m1 * (psi0 - 1.0)
        C = -m1
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            disc = 0.0
        p2_tilde = (-B + math.sqrt(disc)) / (2.0 * A)
        # Numerical safety
        p2_tilde = min(max(p2_tilde, 1e-12), 1.0 - 1e-12)
        p1_tilde = p2_tilde * psi0 / (1.0 + p2_tilde * (psi0 - 1.0))
        p1_tilde = min(max(p1_tilde, 1e-12), 1.0 - 1e-12)
    q1_tilde = 1.0 - p1_tilde
    q2_tilde = 1.0 - p2_tilde
    # Numerator of z_FMO
    num = ((p1_hat - p1_tilde) / (p1_tilde * q1_tilde)
           - (p2_hat - p2_tilde) / (p2_tilde * q2_tilde))
    # Constrained variance (denominator of z_FMO)
    var_c = (1.0 / (n1 * p1_tilde * q1_tilde)
             + 1.0 / (n2 * p2_tilde * q2_tilde))
    sigma_c = math.sqrt(var_c)
    z_fmo = num / sigma_c
    # Unconstrained variance at the (substituted) actual proportions
    var_u = (1.0 / (n1 * p1_hat * (1.0 - p1_hat))
             + 1.0 / (n2 * p2_hat * (1.0 - p2_hat)))
    sigma_u = math.sqrt(var_u)
    return z_fmo, sigma_c, sigma_u


def _or_one_sided_power(
    n1: int, n2: int, p2: float, psi0: float, psi1: float,
    alpha: float, direction: str,
) -> float:
    """Asymptotic power for the FM-Score OR test.

    direction == 'upper': H0: psi <= psi0, H1: psi > psi0  (reject z > +z_a)
    direction == 'lower': H0: psi >= psi0, H1: psi < psi0  (reject z < -z_a)
    """
    if not (0 < p2 < 1):
        raise ValueError("p2 must be in (0, 1)")
    if psi0 <= 0 or psi1 <= 0:
        raise ValueError("odds ratios must be > 0")
    if n1 < 2 or n2 < 2:
        return 0.0
    p1_1 = _p1_from_or(psi1, p2)
    if not (0 < p1_1 < 1):
        return 0.0
    z_h1, sigma_c, sigma_u = _fm_score_components(n1, n2, p1_1, p2, psi0)
    z_alpha = D.norm_ppf(1.0 - alpha)
    sd_z = sigma_u / sigma_c
    from scipy.stats import norm
    if direction == "upper":
        return float(1.0 - norm.cdf((z_alpha - z_h1) / sd_z))
    elif direction == "lower":
        return float(norm.cdf((-z_alpha - z_h1) / sd_z))
    else:
        raise ValueError(f"unsupported direction: {direction!r}")


def _solve_n_equal(
    target_power: float,
    power_fn,  # callable(n) -> achieved power at n1=n2=n
    n_min: int = 4,
    n_max: int = 10_000_000,
) -> tuple[int, float]:
    """Bisect for the smallest integer n with power_fn(n) >= target_power."""
    if not 0.0 < target_power < 1.0:
        raise ValueError("power must be in (0, 1)")
    lo, hi = n_min, n_min
    while hi <= n_max:
        if power_fn(hi) >= target_power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError("failed to bracket N")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if power_fn(mid) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi, power_fn(hi)


# -------------------- Non-Inferiority -------------------------------------

def non_inferiority_odds_ratio_two_proportions(
    *,
    p2: float,
    or0: float,
    or1: float = 1.0,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    higher_is_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for the odds ratio of two proportions.

    Power is calculated by the Farrington &
    Manning (1990) likelihood score statistic, normal-approximation
    method.

    Parameters
    ----------
    p2
        Reference (control) proportion.
    or0
        Non-inferiority margin on the odds-ratio scale.  When
        ``higher_is_better=True`` this should be ``< 1`` (the smallest
        OR still considered non-inferior); when ``False`` it should be
        ``> 1``.
    or1
        Actual treatment-vs-control odds ratio (default 1.0, the
        "non-inferior with no real difference" scenario).
    alpha
        One-sided significance level (default 0.05;
        uses 0.05 for NI on the OR scale).
    power, n1, n2
        Supply either ``power`` (solve for N) or sample sizes (solve
        for power).
    allocation
        ``n2 = ceil(allocation * n1)`` when only n1 is given.
    higher_is_better
        Direction convention.  ``True`` (default) for "successes are
        good"; ``False`` flips the inequalities.
    solve_for
        Either ``"n"`` or ``"power"``; inferred from inputs when
        omitted.
    """
    if or0 == 1.0:
        raise ValueError("or0 cannot equal 1 (no margin)")
    if higher_is_better and or0 >= 1.0:
        raise ValueError("higher_is_better=True requires or0 < 1")
    if (not higher_is_better) and or0 <= 1.0:
        raise ValueError("higher_is_better=False requires or0 > 1")
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    inputs_echo = {
        "p2": p2, "or0": or0, "or1": or1, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2,
        "allocation": allocation, "higher_is_better": higher_is_better,
    }
    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    direction = "upper" if higher_is_better else "lower"

    def p_at(n_val: int) -> tuple[float, int]:
        n2_val = max(2, math.ceil(allocation * n_val))
        return _or_one_sided_power(
            n_val, n2_val, p2, or0, or1, alpha, direction
        ), n2_val

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _or_one_sided_power(
            n1, n2, p2, or0, or1, alpha, direction
        )
        result = {"n1": n1, "n2": n2, "n": n1 + n2,
                  "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n1r, achieved = _solve_n_equal(
            power, lambda n: p_at(n)[0]
        )
        n2r = max(2, math.ceil(allocation * n1r))
        result = {"n1": n1r, "n2": n2r, "n": n1r + n2r,
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    # Implied proportions, useful for echo / audit
    p1_0 = _p1_from_or(or0, p2)
    p1_1 = _p1_from_or(or1, p2)
    return {
        "method_id": "non_inferiority_odds_ratio_two_proportions",
        "solve_for": solve_for,
        **result,
        "p1_0": p1_0,
        "p1_1": p1_1,
        "inputs_echo": inputs_echo,
        "citations": [
            "Two Proportions (Chapter 212).",
            "Farrington, C.P. & Manning, G. (1990). 'Test Statistics and "
            "Sample Size Formulae for Comparative Binomial Trials with "
            "Null Hypothesis of Non-Zero Risk Difference or Non-Unity "
            "Relative Risk.' Statistics in Medicine 9:1447-1454.",
            "Miettinen, O.S. & Nurminen, M. (1985). 'Comparative analysis "
            "of two rates.' Statistics in Medicine 4:213-226.",
        ],
    }


# -------------------- Equivalence (TOST) ----------------------------------

def equivalence_odds_ratio_two_proportions(
    *,
    p2: float,
    or0_lower: float,
    or0_upper: float,
    or1: float = 1.0,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Equivalence (TOST) test for the odds ratio of two proportions.

    Power is calculated by the Farrington &
    Manning (1990) likelihood score statistic, normal-approximation
    method, combined via the Schuirmann-style TOST approximation

        Power = max(0, P_lower + P_upper - 1)

    where ``P_lower`` is the power of the upper-tailed test against
    ``or0_lower`` and ``P_upper`` is the power of the lower-tailed test
    against ``or0_upper``.

    Parameters
    ----------
    p2
        Reference proportion.
    or0_lower
        Lower equivalence bound on the OR scale.  Must satisfy
        ``0 < or0_lower < 1``.
    or0_upper
        Upper equivalence bound on the OR scale.  Must satisfy
        ``or0_upper > 1``.  Typically ``or0_upper = 1 / or0_lower``.
    or1
        Actual treatment-vs-control odds ratio (default 1.0, the
        "fully equivalent" scenario).
    alpha
        Per one-sided test (TOST) significance level (default 0.05).
    power, n1, n2, allocation
        Sample-size / power knobs (see other methods).
    solve_for
        ``"n"`` or ``"power"``; inferred when omitted.
    """
    if not (0 < or0_lower < 1):
        raise ValueError("or0_lower must lie in (0, 1)")
    if or0_upper <= 1:
        raise ValueError("or0_upper must be > 1")
    if or1 <= 0:
        raise ValueError("or1 must be > 0")
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    inputs_echo = {
        "p2": p2, "or0_lower": or0_lower, "or0_upper": or0_upper,
        "or1": or1, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2,
        "allocation": allocation,
    }
    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def _power_at(n1v: int, n2v: int) -> float:
        p_lower = _or_one_sided_power(
            n1v, n2v, p2, or0_lower, or1, alpha, "upper"
        )
        p_upper = _or_one_sided_power(
            n1v, n2v, p2, or0_upper, or1, alpha, "lower"
        )
        return max(0.0, p_lower + p_upper - 1.0)

    def p_at(n_val: int) -> float:
        n2_val = max(2, math.ceil(allocation * n_val))
        return _power_at(n_val, n2_val)

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _power_at(n1, n2)
        result = {"n1": n1, "n2": n2, "n": n1 + n2,
                  "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n1r, achieved = _solve_n_equal(power, p_at)
        n2r = max(2, math.ceil(allocation * n1r))
        result = {"n1": n1r, "n2": n2r, "n": n1r + n2r,
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    p1_0L = _p1_from_or(or0_lower, p2)
    p1_0U = _p1_from_or(or0_upper, p2)
    p1_1 = _p1_from_or(or1, p2)
    return {
        "method_id": "equivalence_odds_ratio_two_proportions",
        "solve_for": solve_for,
        **result,
        "p1_0_lower": p1_0L,
        "p1_0_upper": p1_0U,
        "p1_1": p1_1,
        "inputs_echo": inputs_echo,
        "citations": [
            "Two Proportions (Chapter 215).",
            "Farrington, C.P. & Manning, G. (1990). 'Test Statistics and "
            "Sample Size Formulae for Comparative Binomial Trials with "
            "Null Hypothesis of Non-Zero Risk Difference or Non-Unity "
            "Relative Risk.' Statistics in Medicine 9:1447-1454.",
            "Schuirmann, D.J. (1987). 'A comparison of the two one-sided "
            "tests procedure and the power approach for assessing the "
            "equivalence of average bioavailability.' Journal of "
            "Pharmacokinetics and Biopharmaceutics 15(6):657-680.",
        ],
    }


# -------------------- Superiority by a Margin -----------------------------

def superiority_by_margin_odds_ratio_two_proportions(
    *,
    p2: float,
    or0: float,
    or1: float,
    alpha: float = 0.025,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    higher_is_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Superiority-by-margin test for the odds ratio of two proportions.

    Mathematically identical in mechanics to the
    non-inferiority test on the OR scale, with the margin moved to the
    other side of 1.  Power is calculated by the Farrington & Manning
    (1990) likelihood score statistic, normal-approximation method.

    For ``higher_is_better=True`` (default):
        H0: psi <= or0    vs.   H1: psi > or0       (or0 > 1)
    For ``higher_is_better=False``:
        H0: psi >= or0    vs.   H1: psi < or0       (or0 < 1)

    Parameters
    ----------
    p2
        Reference proportion.
    or0
        Superiority margin on the OR scale.  When
        ``higher_is_better=True`` this should be ``> 1``; when ``False``
        it should be ``< 1``.
    or1
        Actual odds ratio at which power is calculated.  For
        ``higher_is_better=True``, ``or1`` should exceed ``or0``; for
        ``False``, ``or1`` should be less than ``or0``.
    alpha
        One-sided significance level (default 0.025).
    power, n1, n2, allocation
        Sample-size / power knobs (see other methods).
    higher_is_better
        Direction convention.
    solve_for
        ``"n"`` or ``"power"``; inferred when omitted.
    """
    if or0 == 1.0:
        raise ValueError("or0 cannot equal 1 (no margin)")
    if higher_is_better and or0 <= 1.0:
        raise ValueError("higher_is_better=True requires or0 > 1")
    if (not higher_is_better) and or0 >= 1.0:
        raise ValueError("higher_is_better=False requires or0 < 1")
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    inputs_echo = {
        "p2": p2, "or0": or0, "or1": or1, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2,
        "allocation": allocation, "higher_is_better": higher_is_better,
    }
    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    direction = "upper" if higher_is_better else "lower"

    def p_at(n_val: int) -> tuple[float, int]:
        n2_val = max(2, math.ceil(allocation * n_val))
        return _or_one_sided_power(
            n_val, n2_val, p2, or0, or1, alpha, direction
        ), n2_val

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _or_one_sided_power(
            n1, n2, p2, or0, or1, alpha, direction
        )
        result = {"n1": n1, "n2": n2, "n": n1 + n2,
                  "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n1r, achieved = _solve_n_equal(
            power, lambda n: p_at(n)[0]
        )
        n2r = max(2, math.ceil(allocation * n1r))
        result = {"n1": n1r, "n2": n2r, "n": n1r + n2r,
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    p1_0 = _p1_from_or(or0, p2)
    p1_1 = _p1_from_or(or1, p2)
    return {
        "method_id": "superiority_by_margin_odds_ratio_two_proportions",
        "solve_for": solve_for,
        **result,
        "p1_0": p1_0,
        "p1_1": p1_1,
        "inputs_echo": inputs_echo,
        "citations": [
            "Ratio of Two Proportions (Chapter 207).",
            "Farrington, C.P. & Manning, G. (1990). 'Test Statistics and "
            "Sample Size Formulae for Comparative Binomial Trials with "
            "Null Hypothesis of Non-Zero Risk Difference or Non-Unity "
            "Relative Risk.' Statistics in Medicine 9:1447-1454.",
            "Miettinen, O.S. & Nurminen, M. (1985). 'Comparative analysis "
            "of two rates.' Statistics in Medicine 4:213-226.",
        ],
    }
