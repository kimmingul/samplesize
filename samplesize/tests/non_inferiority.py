"""Non-inferiority tests, parallel two-group designs.

- "Non-Inferiority Tests for Two Means using Differences"
- "Non-Inferiority Tests for the Difference Between Two Proportions"

Both share the same logical structure: a one-sided test that the
treatment is not worse than the control by more than a pre-specified
margin (NIM > 0).  Two conventions for which direction is "better":

  higher_is_better=True  → H1: μ1 - μ2 > -NIM  (treatment ≥ reference - NIM)
  higher_is_better=False → H1: μ1 - μ2 <  +NIM (treatment ≤ reference + NIM)

  Test direction is one-sided.  Pooled-variance t (means) and z-pooled
  (proportions).
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# -------------------- Two-mean NI ------------------------------------------

def _ni_means_power(mean1: float, mean2: float, sd: float,
                    margin: float, alpha: float,
                    n1: int, n2: int, higher_is_better: bool) -> float:
    if margin <= 0:
        raise ValueError("margin must be > 0 (magnitude of NI margin)")
    if n1 < 2 or n2 < 2:
        return 0.0
    df = n1 + n2 - 2
    se = sd * math.sqrt(1.0 / n1 + 1.0 / n2)
    diff = mean1 - mean2
    if higher_is_better:
        ncp = (diff + margin) / se
    else:
        ncp = (margin - diff) / se
    t_crit = D.t_ppf(1.0 - alpha, df)
    return 1.0 - D.nct_cdf(t_crit, df, ncp)


def ni_means_power_at_n(*, mean1: float, mean2: float, sd: float,
                        margin: float, alpha: float,
                        n1: int, n2: int,
                        higher_is_better: bool = True) -> float:
    return _ni_means_power(mean1, mean2, sd, margin, alpha,
                           n1, n2, higher_is_better)


def ni_means_n_for_power(*, mean1: float, mean2: float, sd: float,
                         margin: float, alpha: float, power: float,
                         higher_is_better: bool = True,
                         allocation: float = 1.0,
                         n_min: int = 4,
                         n_max: int = 10_000_000) -> tuple[int, int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")

    def n2_for(n1):
        return max(2, math.ceil(allocation * n1))

    def p_at(n1):
        return _ni_means_power(mean1, mean2, sd, margin, alpha,
                               n1, n2_for(n1), higher_is_better)

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
    return n1, n2, _ni_means_power(mean1, mean2, sd, margin, alpha,
                                   n1, n2, higher_is_better)


def non_inferiority_two_means(
    *,
    mean1: float,
    mean2: float,
    sd: float,
    margin: float,
    alpha: float = 0.025,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    higher_is_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for two means, pooled-variance t."""
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    inputs_echo = {
        "mean1": mean1, "mean2": mean2, "sd": sd, "margin": margin,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2,
        "allocation": allocation, "higher_is_better": higher_is_better,
    }
    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = ni_means_power_at_n(
            mean1=mean1, mean2=mean2, sd=sd, margin=margin, alpha=alpha,
            n1=n1, n2=n2, higher_is_better=higher_is_better,
        )
        result = {"n1": n1, "n2": n2, "n": n1 + n2,
                  "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n1r, n2r, achieved = ni_means_n_for_power(
            mean1=mean1, mean2=mean2, sd=sd, margin=margin,
            alpha=alpha, power=power,
            higher_is_better=higher_is_better, allocation=allocation,
        )
        result = {"n1": n1r, "n2": n2r, "n": n1r + n2r,
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "non_inferiority_two_means",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Chow, Shao & Wang (2008). Sample Size Calculations in Clinical Research.",
            "Julious, S.A. (2004). Tutorial in biostatistics: sample sizes for clinical trials with normal data.",
        ],
    }


# -------------------- Two-proportion NI ------------------------------------

def _ni_props_power(p1: float, p2: float, margin: float, alpha: float,
                    n1: int, n2: int, higher_is_better: bool) -> float:
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("p1, p2 must be in (0, 1)")
    if margin <= 0:
        raise ValueError("margin must be > 0")
    if n1 < 2 or n2 < 2:
        return 0.0
    from scipy.stats import norm
    diff = p1 - p2
    if higher_is_better:
        delta = diff + margin
    else:
        delta = margin - diff
    # Unpooled variance under H1 (standard for NI z-test).
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_alpha = D.norm_ppf(1 - alpha)
    return float(1.0 - norm.cdf(z_alpha - delta / se))


def non_inferiority_two_proportions(
    *,
    p1: float,
    p2: float,
    margin: float,
    alpha: float = 0.025,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    higher_is_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for two proportions, z-test on the difference."""
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    inputs_echo = {
        "p1": p1, "p2": p2, "margin": margin, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2, "allocation": allocation,
        "higher_is_better": higher_is_better,
    }
    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def p_at(n1_val):
        n2_val = max(2, math.ceil(allocation * n1_val))
        return _ni_props_power(p1, p2, margin, alpha, n1_val, n2_val,
                               higher_is_better), n2_val

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _ni_props_power(p1, p2, margin, alpha, n1, n2,
                                   higher_is_better)
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
        "method_id": "non_inferiority_two_proportions",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Chow, Shao & Wang (2008). Sample Size Calculations in Clinical Research.",
        ],
    }


# -------------------- One-mean NI ------------------------------------------

def _ni_one_mean_power(mean: float, reference: float, sd: float,
                       margin: float, alpha: float,
                       n: int, higher_is_better: bool) -> float:
    """Noncentral-t power for the one-sample non-inferiority t-test.

    σ/√n with df = n − 1.  The shift used for the NCP follows the
    same convention as the two-sample version:

      higher_is_better=True  → ncp = (D + |NIM|) / (σ/√n)
      higher_is_better=False → ncp = (|NIM| - D) / (σ/√n)

    where D = mean − reference.
    """
    if margin <= 0:
        raise ValueError("margin must be > 0 (magnitude of NI margin)")
    if n < 2:
        return 0.0
    df = n - 1
    se = sd / math.sqrt(n)
    diff = mean - reference
    if higher_is_better:
        ncp = (diff + margin) / se
    else:
        ncp = (margin - diff) / se
    t_crit = D.t_ppf(1.0 - alpha, df)
    return 1.0 - D.nct_cdf(t_crit, df, ncp)


def ni_one_mean_power_at_n(*, mean: float, reference: float, sd: float,
                           margin: float, alpha: float, n: int,
                           higher_is_better: bool = True) -> float:
    return _ni_one_mean_power(mean, reference, sd, margin, alpha, n,
                              higher_is_better)


def ni_one_mean_n_for_power(*, mean: float, reference: float, sd: float,
                            margin: float, alpha: float, power: float,
                            higher_is_better: bool = True,
                            n_min: int = 2,
                            n_max: int = 10_000_000) -> tuple[int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")

    def p_at(n):
        return _ni_one_mean_power(mean, reference, sd, margin, alpha, n,
                                  higher_is_better)

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


def non_inferiority_one_mean(
    *,
    mean: float,
    reference: float = 0.0,
    sd: float,
    margin: float,
    alpha: float = 0.025,
    power: float | None = None,
    n: int | None = None,
    higher_is_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for one mean, one-sample t.

    Power
    formula uses the noncentral t with df = n − 1 and ncp built from
    the (signed) NI margin and the true mean difference D = mean -
    reference.  Set ``reference`` to 0 for paired-difference designs
    where ``mean`` is the mean of the paired differences themselves.
    """
    inputs_echo = {
        "mean": mean, "reference": reference, "sd": sd, "margin": margin,
        "alpha": alpha, "power": power, "n": n,
        "higher_is_better": higher_is_better,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = ni_one_mean_power_at_n(
            mean=mean, reference=reference, sd=sd, margin=margin,
            alpha=alpha, n=n, higher_is_better=higher_is_better,
        )
        result = {"n": n, "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n_r, achieved = ni_one_mean_n_for_power(
            mean=mean, reference=reference, sd=sd, margin=margin,
            alpha=alpha, power=power,
            higher_is_better=higher_is_better,
        )
        result = {"n": n_r, "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "non_inferiority_one_mean",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Chow, Shao & Wang (2003). Sample Size Calculations in Clinical Research, p.50, 54-55.",
            "Julious, S.A. (2004). Tutorial in biostatistics: sample sizes for clinical trials with normal data.",
        ],
    }


# -------------------- One-proportion NI ------------------------------------

def _ni_one_prop_power(p: float, p0: float, alpha: float, n: int,
                       higher_is_better: bool, se_method: str) -> float:
    """Normal-approximation power for one-sample non-inferiority test.

    Implements the Z-Test using S(P0) (default) and Z-Test using
    S(P1)/S(Phat) variants for the one-sided NI test.

      Upper one-sided (higher is better), H0: P <= P0 vs H1: P > P0
        Power = 1 - Phi((sqrt(n)(P0 - P1) + z_alpha * sd0) / sd1)

      Lower one-sided (higher is worse), H0: P >= P0 vs H1: P < P0
        Power = Phi((sqrt(n)(P0 - P1) - z_alpha * sd0) / sd1)

    where
      sd0 = sqrt(P0(1-P0))  if se_method == "s_p0"
      sd0 = sqrt(P1(1-P1))  if se_method == "s_phat"
      sd1 = sqrt(P1(1-P1))    (denominator is always P1 std)
    """
    if not (0 < p < 1 and 0 < p0 < 1):
        raise ValueError("p, p0 must be in (0, 1)")
    if n < 2:
        return 0.0
    from scipy.stats import norm
    if se_method == "s_p0":
        sd0 = math.sqrt(p0 * (1 - p0))
    elif se_method == "s_phat":
        sd0 = math.sqrt(p * (1 - p))
    else:
        raise ValueError(f"unsupported se_method: {se_method!r}")
    sd1 = math.sqrt(p * (1 - p))
    z_alpha = D.norm_ppf(1 - alpha)
    if higher_is_better:
        # P0 < P (treatment proportion at least P0)
        arg = (math.sqrt(n) * (p0 - p) + z_alpha * sd0) / sd1
        return float(1.0 - norm.cdf(arg))
    else:
        # higher proportions worse: P0 > P
        arg = (math.sqrt(n) * (p0 - p) - z_alpha * sd0) / sd1
        return float(norm.cdf(arg))


def non_inferiority_one_proportion(
    *,
    p: float,
    p0: float | None = None,
    margin: float | None = None,
    alpha: float = 0.025,
    power: float | None = None,
    n: int | None = None,
    sides: int = 1,
    higher_is_better: bool = True,
    se_method: str = "s_p0",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for one proportion.

    Uses
    the normal-approximation z-test with either S(P0) (default) or
    S(Phat) for the standard error.

    Specify the NI cutoff either directly as ``p0`` (the smallest
    proportion still deemed non-inferior) or implicitly via
    ``margin`` together with a baseline.  When ``margin`` is supplied,
    the cutoff is computed as

      higher_is_better=True  -> p0 = p_baseline - |margin|
      higher_is_better=False -> p0 = p_baseline + |margin|

    where ``p_baseline`` defaults to ``p`` (the value at which power
    is evaluated (P1=PB convention).  If
    you have a separate baseline different from p, supply ``p0``
    directly.

    ``sides`` is fixed at 1 (NI is inherently one-sided); the kwarg
    is accepted for API parity with other proportion methods.
    """
    if sides != 1:
        raise ValueError("non-inferiority is one-sided; sides must be 1")
    if p0 is None:
        if margin is None:
            raise ValueError("supply either p0 or margin")
        if margin <= 0:
            raise ValueError("margin must be > 0 (magnitude)")
        # Use p itself as the baseline (P1=PB convention).
        if higher_is_better:
            p0 = p - margin
        else:
            p0 = p + margin
    if not 0 < p0 < 1:
        raise ValueError("p0 must be in (0, 1)")

    inputs_echo = {
        "p": p, "p0": p0, "margin": margin, "alpha": alpha,
        "power": power, "n": n, "sides": sides,
        "higher_is_better": higher_is_better, "se_method": se_method,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def p_at(n_val):
        return _ni_one_prop_power(p, p0, alpha, n_val,
                                  higher_is_better, se_method)

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
        "method_id": "non_inferiority_one_proportion",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Chow, Shao & Wang (2008). Sample Size Calculations in Clinical Research, p.85.",
            "Blackwelder, W.C. (1998). Equivalence Trials. In Encyclopedia of Biostatistics.",
        ],
    }


# -------------------- Two-mean NI on ratio scale ----------------------------

def non_inferiority_two_means_ratios(
    *,
    r1: float = 1.0,
    margin: float,
    cv: float,
    alpha: float = 0.025,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    higher_is_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for two means on the ratio scale.

    Non-inferiority test for two means using ratios
    (Julious 2004).  Power and sample size are computed by
    log-transforming the inputs and dispatching to the difference-scale
    NI two-mean t-test:

    * SD on log scale: ``sigma_X = sqrt(ln(CV^2 + 1))``
    * Margin on log scale:
        - higher_is_better=True  -> NIM' = -ln(1 - |margin|)  (positive)
        - higher_is_better=False -> NIM' =  ln(1 + |margin|)  (positive)
    * True difference on log scale: D = ln(r1)

    Parameters
    ----------
    r1
        True ratio mu_T / mu_R at which the power is calculated.
        Default 1.0.  Must be positive.
    margin
        Magnitude of the non-inferiority margin (NIM) on the ratio
        scale.  Strictly positive.  When higher_is_better=True, the
        non-inferiority bound is 1 - margin (e.g. margin=0.20 ->
        bound=0.80); when higher_is_better=False, the bound is
        1 + margin.
    cv
        Coefficient of variation on the original (unlogged) scale, as
        a decimal.  Must be positive.
    alpha
        One-sided significance level (default 0.025).
    power, n1, n2
        Supply either power (solve for n) or sample sizes (solve for
        power).
    allocation
        ``n2 = ceil(allocation * n1)`` when only n1 is given.
    higher_is_better
        Direction of the non-inferiority test on the ratio scale.

    Returns
    -------
    Dict with ``method_id``, ``solve_for``, ``n1``, ``n2``, ``n``,
    ``achieved_power``, ``sigma_log`` (log-scale SD), ``inputs_echo``,
    and ``citations``.
    """
    if cv <= 0:
        raise ValueError("cv must be positive")
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    if margin <= 0:
        raise ValueError("margin must be > 0 (magnitude)")
    if higher_is_better:
        if margin >= 1.0:
            raise ValueError(
                "margin must be < 1 when higher_is_better=True "
                "(NI bound = 1 - margin must remain positive)"
            )

    sigma_log = math.sqrt(math.log(cv * cv + 1.0))
    # Magnitude of the NI margin on the log scale.
    if higher_is_better:
        # Bound on ratio scale = 1 - margin; on log scale = ln(1 - margin) < 0.
        margin_log = -math.log(1.0 - margin)
    else:
        # Bound on ratio scale = 1 + margin; on log scale = ln(1 + margin) > 0.
        margin_log = math.log(1.0 + margin)
    diff_log = math.log(r1)

    inner = non_inferiority_two_means(
        mean1=diff_log,
        mean2=0.0,
        sd=sigma_log,
        margin=margin_log,
        alpha=alpha,
        power=power,
        n1=n1,
        n2=n2,
        allocation=allocation,
        higher_is_better=higher_is_better,
        solve_for=solve_for,
    )

    inputs_echo = {
        "r1": r1, "margin": margin, "cv": cv, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2,
        "allocation": allocation, "higher_is_better": higher_is_better,
    }
    return {
        "method_id": "non_inferiority_two_means_ratios",
        "solve_for": inner["solve_for"],
        "n1": inner["n1"],
        "n2": inner["n2"],
        "n": inner["n"],
        "achieved_power": inner["achieved_power"],
        "sigma_log": sigma_log,
        "margin_log": margin_log,
        "diff_log": diff_log,
        "inputs_echo": inputs_echo,
        "citations": [
            "Julious, S.A. (2004). Tutorial in biostatistics: sample sizes "
            "for clinical trials with normal data. Statistics in Medicine, "
            "23:1921-1986.",
            "Chow, Shao & Wang (2003). Sample Size Calculations in Clinical Research.",
        ],
    }
