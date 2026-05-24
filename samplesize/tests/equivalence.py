"""Equivalence (TOST) tests, parallel two-group designs.

- "Equivalence Tests for Two Means using Differences"
- "Equivalence Tests for the Difference Between Two Proportions"

Power formula (Schuirmann/Phillips TOST, assuming sigma known for the
power integral but central-t critical value for the test):

  SE = σ · √(1/N1 + 1/N2)        (or analogous for proportions)
  Power = Φ((EU - D)/SE - t_α)  −  Φ((EL - D)/SE + t_α)

where (EL, EU) are the (lower, upper) equivalence limits (signed, with
EL < EU), D is the true mean difference, and t_α uses df = N1+N2-2
for the means test (or z_α for proportions).
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


def _norm_cdf(x: float) -> float:
    from scipy.stats import norm
    return float(norm.cdf(x))


# -------------------- Two-mean equivalence ----------------------------------

def _eq_means_power(mean1: float, mean2: float, sd: float,
                    lower_margin: float, upper_margin: float,
                    alpha: float, n1: int, n2: int) -> float:
    if upper_margin <= lower_margin:
        raise ValueError("upper_margin must be > lower_margin")
    if n1 < 2 or n2 < 2:
        return 0.0
    df = n1 + n2 - 2
    se = sd * math.sqrt(1.0 / n1 + 1.0 / n2)
    diff = mean1 - mean2
    t_alpha = D.t_ppf(1.0 - alpha, df)
    upper_term = _norm_cdf((upper_margin - diff) / se - t_alpha)
    lower_term = _norm_cdf((lower_margin - diff) / se + t_alpha)
    return max(0.0, upper_term - lower_term)


def eq_means_power_at_n(*, mean1: float, mean2: float, sd: float,
                        lower_margin: float, upper_margin: float,
                        alpha: float, n1: int, n2: int) -> float:
    return _eq_means_power(mean1, mean2, sd, lower_margin, upper_margin,
                           alpha, n1, n2)


def eq_means_n_for_power(*, mean1: float, mean2: float, sd: float,
                         lower_margin: float, upper_margin: float,
                         alpha: float, power: float,
                         allocation: float = 1.0,
                         n_min: int = 4,
                         n_max: int = 10_000_000) -> tuple[int, int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")

    def n2_for(n1):
        return max(2, math.ceil(allocation * n1))

    def p_at(n1):
        return _eq_means_power(mean1, mean2, sd, lower_margin, upper_margin,
                               alpha, n1, n2_for(n1))

    lo, hi = n_min, n_min
    while hi <= n_max:
        if p_at(hi) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError("failed to bracket N")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if p_at(mid) >= power:
            hi = mid
        else:
            lo = mid
    n1 = hi
    n2 = n2_for(n1)
    return n1, n2, _eq_means_power(mean1, mean2, sd, lower_margin,
                                   upper_margin, alpha, n1, n2)


def equivalence_two_means(
    *,
    mean1: float,
    mean2: float,
    sd: float,
    lower_margin: float | None = None,
    upper_margin: float | None = None,
    margin: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """TOST-style equivalence test on the mean difference.

    Supply either explicit (`lower_margin`, `upper_margin`) or symmetric
    `margin` (treated as (−margin, +margin)).
    """
    if margin is not None:
        if lower_margin is not None or upper_margin is not None:
            raise ValueError("supply either `margin` or (lower_margin, upper_margin)")
        lower_margin, upper_margin = -abs(margin), abs(margin)
    if lower_margin is None or upper_margin is None:
        raise ValueError("equivalence limits required")

    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    inputs_echo = {
        "mean1": mean1, "mean2": mean2, "sd": sd,
        "lower_margin": lower_margin, "upper_margin": upper_margin,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2,
        "allocation": allocation,
    }
    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = eq_means_power_at_n(
            mean1=mean1, mean2=mean2, sd=sd,
            lower_margin=lower_margin, upper_margin=upper_margin,
            alpha=alpha, n1=n1, n2=n2,
        )
        result = {"n1": n1, "n2": n2, "n": n1 + n2,
                  "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n1r, n2r, achieved = eq_means_n_for_power(
            mean1=mean1, mean2=mean2, sd=sd,
            lower_margin=lower_margin, upper_margin=upper_margin,
            alpha=alpha, power=power, allocation=allocation,
        )
        result = {"n1": n1r, "n2": n2r, "n": n1r + n2r,
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "equivalence_two_means",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Schuirmann, D.J. (1987). A comparison of the two one-sided tests procedure.",
            "Julious, S.A. (2010). Sample Sizes for Clinical Trials.",
        ],
    }


# -------------------- Two-proportion equivalence ----------------------------

def _eq_props_power(p1: float, p2: float, lower_margin: float,
                    upper_margin: float, alpha: float,
                    n1: int, n2: int) -> float:
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("p1, p2 must be in (0, 1)")
    if upper_margin <= lower_margin:
        raise ValueError("upper_margin must be > lower_margin")
    if n1 < 2 or n2 < 2:
        return 0.0
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    z_alpha = D.norm_ppf(1.0 - alpha)
    upper_term = _norm_cdf((upper_margin - diff) / se - z_alpha)
    lower_term = _norm_cdf((lower_margin - diff) / se + z_alpha)
    return max(0.0, upper_term - lower_term)


def equivalence_two_proportions(
    *,
    p1: float,
    p2: float,
    lower_margin: float | None = None,
    upper_margin: float | None = None,
    margin: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """TOST equivalence test on the difference of two proportions."""
    if margin is not None:
        if lower_margin is not None or upper_margin is not None:
            raise ValueError("supply either `margin` or (lower_margin, upper_margin)")
        lower_margin, upper_margin = -abs(margin), abs(margin)
    if lower_margin is None or upper_margin is None:
        raise ValueError("equivalence limits required")

    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    inputs_echo = {
        "p1": p1, "p2": p2,
        "lower_margin": lower_margin, "upper_margin": upper_margin,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2,
        "allocation": allocation,
    }
    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def p_at(n1_val):
        n2_val = max(2, math.ceil(allocation * n1_val))
        return _eq_props_power(p1, p2, lower_margin, upper_margin, alpha,
                               n1_val, n2_val), n2_val

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _eq_props_power(p1, p2, lower_margin, upper_margin,
                                    alpha, n1, n2)
        result = {"n1": n1, "n2": n2, "n": n1 + n2,
                  "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        lo, hi = 4, 4
        while hi <= 10_000_000:
            ach, _ = p_at(hi)
            if ach >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            ach, _ = p_at(mid)
            if ach >= power:
                hi = mid
            else:
                lo = mid
        n1r = hi
        achieved, n2r = p_at(n1r)
        result = {"n1": n1r, "n2": n2r, "n": n1r + n2r,
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "equivalence_two_proportions",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Schuirmann, D.J. (1987). A comparison of the two one-sided tests procedure.",
        ],
    }


# -------------------- One-mean equivalence ----------------------------------

def _eq_one_mean_power(mean: float, reference: float, sd: float,
                       lower_margin: float, upper_margin: float,
                       alpha: float, n: int) -> float:
    """TOST power for the one-sample equivalence t-test.

    sigma-known / central-t critical value approximation:

      SE = σ/√n,  df = n − 1
      Power = Φ((EU - μ)/SE - t_α)  −  Φ((EL - μ)/SE + t_α)

    where (EL, EU) are interpreted relative to ``reference``
    (defaults to 0 — the chapter compares the population mean
    directly to absolute equivalence limits, with `reference` shifting
    the limits when non-zero).
    """
    if upper_margin <= lower_margin:
        raise ValueError("upper_margin must be > lower_margin")
    if n < 2:
        return 0.0
    df = n - 1
    se = sd / math.sqrt(n)
    diff = mean - reference
    t_alpha = D.t_ppf(1.0 - alpha, df)
    upper_term = _norm_cdf((upper_margin - diff) / se - t_alpha)
    lower_term = _norm_cdf((lower_margin - diff) / se + t_alpha)
    return max(0.0, upper_term - lower_term)


def eq_one_mean_power_at_n(*, mean: float, reference: float, sd: float,
                           lower_margin: float, upper_margin: float,
                           alpha: float, n: int) -> float:
    return _eq_one_mean_power(mean, reference, sd, lower_margin,
                              upper_margin, alpha, n)


def eq_one_mean_n_for_power(*, mean: float, reference: float, sd: float,
                            lower_margin: float, upper_margin: float,
                            alpha: float, power: float,
                            n_min: int = 2,
                            n_max: int = 10_000_000) -> tuple[int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")

    def p_at(n):
        return _eq_one_mean_power(mean, reference, sd, lower_margin,
                                  upper_margin, alpha, n)

    lo, hi = n_min, n_min
    while hi <= n_max:
        if p_at(hi) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError("failed to bracket N")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if p_at(mid) >= power:
            hi = mid
        else:
            lo = mid
    return hi, p_at(hi)


def equivalence_one_mean(
    *,
    mean: float,
    reference: float = 0.0,
    sd: float,
    lower_margin: float | None = None,
    upper_margin: float | None = None,
    margin: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """TOST-style equivalence test on one mean.

    Supply either
    explicit ``lower_margin`` and ``upper_margin`` (the absolute
    equivalence limits EL and EU) or a symmetric ``margin``
    (treated as (−margin, +margin) about ``reference``).
    """
    if margin is not None:
        if lower_margin is not None or upper_margin is not None:
            raise ValueError("supply either `margin` or (lower_margin, upper_margin)")
        # Symmetric limits about the reference.
        lower_margin = reference - abs(margin)
        upper_margin = reference + abs(margin)
    if lower_margin is None or upper_margin is None:
        raise ValueError("equivalence limits required")

    inputs_echo = {
        "mean": mean, "reference": reference, "sd": sd,
        "lower_margin": lower_margin, "upper_margin": upper_margin,
        "alpha": alpha, "power": power, "n": n,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = eq_one_mean_power_at_n(
            mean=mean, reference=reference, sd=sd,
            lower_margin=lower_margin, upper_margin=upper_margin,
            alpha=alpha, n=n,
        )
        result = {"n": n, "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n_r, achieved = eq_one_mean_n_for_power(
            mean=mean, reference=reference, sd=sd,
            lower_margin=lower_margin, upper_margin=upper_margin,
            alpha=alpha, power=power,
        )
        result = {"n": n_r, "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "equivalence_one_mean",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Schuirmann, D.J. (1987). A comparison of the two one-sided tests procedure.",
            "Phillips, K.F. (1990). Power of the two one-sided tests procedure in bioequivalence.",
        ],
    }


# -------------------- One-proportion equivalence ---------------------------

def _eq_one_prop_power(p: float, p0l: float, p0u: float, alpha: float,
                       n: int, se_method: str) -> float:
    """Normal-approximation TOST power for one-sample equivalence on P.

    Z-Test using S(P0) (default) and Z-Test using S(Phat) variants
    are supported.

      Power = Phi((sqrt(n)(P0U - P1) - z_alpha * sd_U) / sd1)
            - Phi((sqrt(n)(P0L - P1) + z_alpha * sd_L) / sd1)

    where for se_method=="s_p0", sd_U=sqrt(P0U(1-P0U)) and
    sd_L=sqrt(P0L(1-P0L)); for se_method=="s_phat",
    sd_U=sd_L=sqrt(P1(1-P1)).  The P1 denominator (sd1) is always
    sqrt(P1(1-P1)).
    """
    if not (0 < p < 1 and 0 < p0l < 1 and 0 < p0u < 1):
        raise ValueError("p, p0l, p0u must be in (0, 1)")
    if p0u <= p0l:
        raise ValueError("p0u must be > p0l")
    if n < 2:
        return 0.0
    from scipy.stats import norm
    if se_method == "s_p0":
        sd_lower = math.sqrt(p0l * (1 - p0l))
        sd_upper = math.sqrt(p0u * (1 - p0u))
    elif se_method == "s_phat":
        sd_lower = sd_upper = math.sqrt(p * (1 - p))
    else:
        raise ValueError(f"unsupported se_method: {se_method!r}")
    sd1 = math.sqrt(p * (1 - p))
    z_alpha = D.norm_ppf(1.0 - alpha)
    upper_term = (math.sqrt(n) * (p0u - p) - z_alpha * sd_upper) / sd1
    lower_term = (math.sqrt(n) * (p0l - p) + z_alpha * sd_lower) / sd1
    return max(0.0, float(norm.cdf(upper_term) - norm.cdf(lower_term)))


def equivalence_one_proportion(
    *,
    p: float,
    p0: float | None = None,
    lower_margin: float | None = None,
    upper_margin: float | None = None,
    margin: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    se_method: str = "s_p0",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """TOST-style equivalence test for a single proportion.

    The
    equivalence cutoffs (P0L, P0U) are constructed around a baseline
    reference value.  ``p0`` may be supplied as the baseline (typical
    PB = the response rate of the standard
    treatment), defaulting to ``p`` when omitted (so the limits are
    symmetric about the value at which power is evaluated).  Use
    either a symmetric ``margin`` (treated as ±margin about the
    baseline) or explicit ``lower_margin``/``upper_margin`` (absolute
    deviations from the baseline; both must be > 0).
    """
    if margin is not None:
        if lower_margin is not None or upper_margin is not None:
            raise ValueError(
                "supply either `margin` or (lower_margin, upper_margin)"
            )
        if margin <= 0:
            raise ValueError("margin must be > 0")
        lower_margin = upper_margin = abs(margin)
    if lower_margin is None or upper_margin is None:
        raise ValueError("equivalence margins required")
    if lower_margin <= 0 or upper_margin <= 0:
        raise ValueError("lower_margin and upper_margin must be > 0")

    baseline = p0 if p0 is not None else p
    if not 0 < baseline < 1:
        raise ValueError("baseline (p0 or p) must be in (0, 1)")
    p0l = baseline - lower_margin
    p0u = baseline + upper_margin
    if not 0 < p0l < 1 or not 0 < p0u < 1:
        raise ValueError("equivalence limits must lie in (0, 1)")

    inputs_echo = {
        "p": p, "p0": p0, "lower_margin": lower_margin,
        "upper_margin": upper_margin, "p0l": p0l, "p0u": p0u,
        "alpha": alpha, "power": power, "n": n,
        "se_method": se_method,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def p_at(n_val):
        return _eq_one_prop_power(p, p0l, p0u, alpha, n_val, se_method)

    if solve_for == "power":
        assert n is not None
        achieved = p_at(n)
        result = {"n": n, "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 2, 2
        while hi <= 10_000_000:
            if p_at(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if p_at(mid) >= power:
                hi = mid
            else:
                lo = mid
        result = {"n": hi, "achieved_power": p_at(hi)}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "equivalence_one_proportion",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Chow, Shao & Wang (2008). Sample Size Calculations in Clinical Research, p.86.",
            "Schuirmann, D.J. (1987). A comparison of the two one-sided tests procedure.",
        ],
    }


# -------------------- Two-mean equivalence on ratio scale -------------------

def equivalence_two_means_ratios(
    *,
    r1: float = 1.0,
    lower_limit: float | None = None,
    upper_limit: float | None = None,
    cv: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """TOST equivalence test for two means on the ratio scale.

    Equivalence test for two means using ratios
    (Julious 2004).  Power and sample size are computed by
    log-transforming the inputs and dispatching to the difference-scale
    equivalence two-mean t-test:

    * SD on log scale: ``sigma_X = sqrt(ln(CV^2 + 1))``
    * Equivalence limits on log scale: ``(ln(lower_limit), ln(upper_limit))``
      (e.g. RL=0.80, RU=1.25 -> (-0.223144, +0.223144), the symmetric
      80/125% FDA/EMA window).
    * True difference on log scale: D = ln(r1)

    Parameters
    ----------
    r1
        True ratio mu_T / mu_R at which power is calculated.  Default
        1.0.  Must be positive.
    lower_limit, upper_limit
        Equivalence limits on the ratio scale.  Must satisfy
        ``0 < lower_limit < 1 < upper_limit``.  Defaults to the
        80%/125% bioequivalence window when both are omitted.
    cv
        Coefficient of variation on the original (unlogged) scale,
        as a decimal.  Must be positive.
    alpha
        Per one-sided test (TOST) significance level (default 0.05).
    power, n1, n2
        Supply either power (solve for n) or sample sizes (solve for
        power).
    allocation
        ``n2 = ceil(allocation * n1)`` when only n1 is given.
    """
    if cv <= 0:
        raise ValueError("cv must be positive")
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    if lower_limit is None and upper_limit is None:
        lower_limit, upper_limit = 0.80, 1.25
    elif lower_limit is None:
        # Allow specifying only RU and using RL = 1/RU (symmetric margins).
        if upper_limit is None or upper_limit <= 1:
            raise ValueError("upper_limit must be > 1")
        lower_limit = 1.0 / upper_limit
    elif upper_limit is None:
        if lower_limit >= 1 or lower_limit <= 0:
            raise ValueError("lower_limit must lie in (0, 1)")
        upper_limit = 1.0 / lower_limit
    if not (0 < lower_limit < 1):
        raise ValueError("lower_limit must lie in (0, 1)")
    if upper_limit <= 1:
        raise ValueError("upper_limit must be > 1")

    sigma_log = math.sqrt(math.log(cv * cv + 1.0))
    lower_log = math.log(lower_limit)
    upper_log = math.log(upper_limit)
    diff_log = math.log(r1)

    inner = equivalence_two_means(
        mean1=diff_log,
        mean2=0.0,
        sd=sigma_log,
        lower_margin=lower_log,
        upper_margin=upper_log,
        alpha=alpha,
        power=power,
        n1=n1,
        n2=n2,
        allocation=allocation,
        solve_for=solve_for,
    )

    inputs_echo = {
        "r1": r1, "lower_limit": lower_limit, "upper_limit": upper_limit,
        "cv": cv, "alpha": alpha, "power": power, "n1": n1, "n2": n2,
        "allocation": allocation,
    }
    return {
        "method_id": "equivalence_two_means_ratios",
        "solve_for": inner["solve_for"],
        "n1": inner["n1"],
        "n2": inner["n2"],
        "n": inner["n"],
        "achieved_power": inner["achieved_power"],
        "sigma_log": sigma_log,
        "lower_log": lower_log,
        "upper_log": upper_log,
        "diff_log": diff_log,
        "inputs_echo": inputs_echo,
        "citations": [
            "Julious, S.A. (2004). Tutorial in biostatistics: sample sizes "
            "for clinical trials with normal data. Statistics in Medicine, "
            "23:1921-1986.",
            "Schuirmann, D.J. (1987). A comparison of the two one-sided "
            "tests procedure.",
        ],
    }
