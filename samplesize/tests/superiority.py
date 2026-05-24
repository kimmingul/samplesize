"""Superiority-by-a-margin tests, parallel two-group designs.

- "Superiority by a Margin Tests for Two Means using Differences"
- "Superiority by a Margin Tests for the Difference Between Two Proportions"

Both share the same logical structure as Non-Inferiority but with the
margin SIGN flipped: instead of testing "treatment >= reference - margin"
(NI), they test "treatment >= reference + margin" (superiority by a
margin).  Two conventions for which direction is "better":

  higher_is_better=True  -> H1: mu1 - mu2 >  +SM (treatment > reference + SM)
  higher_is_better=False -> H1: mu1 - mu2 <  -SM (treatment < reference - SM)

Test direction is one-sided.  Pooled-variance t (means) and z-unpooled
(proportions).  This procedure "uses the
same mechanics as the Non-Inferiority Tests" with the margin sign
flipped (Chapter 448 references Chapter 450 for validation).
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# -------------------- Two-mean superiority-by-margin -----------------------

def _sup_means_power(mean1: float, mean2: float, sd: float,
                     margin: float, alpha: float,
                     n1: int, n2: int, higher_is_better: bool) -> float:
    """Power for one-sided pooled-variance t-test for superiority by margin.

    For higher_is_better:
        H0: mu1 - mu2 <= +|SM|  vs  H1: mu1 - mu2 > +|SM|
        Test statistic: ((X1bar - X2bar) - |SM|) / se
        NCP = (D - |SM|) / se  where D = mean1 - mean2
        Power = 1 - T'_{df}(t_alpha; ncp)

    For lower_is_better (higher_is_better=False):
        H0: mu1 - mu2 >= -|SM|  vs  H1: mu1 - mu2 < -|SM|
        NCP = -(D + |SM|) / se
        Power = T'_{df}(-t_alpha; ncp)  equivalently 1 - T'_{df}(t_alpha; ncp')
    """
    if margin <= 0:
        raise ValueError("margin must be > 0 (magnitude of superiority margin)")
    if n1 < 2 or n2 < 2:
        return 0.0
    df = n1 + n2 - 2
    se = sd * math.sqrt(1.0 / n1 + 1.0 / n2)
    diff = mean1 - mean2
    if higher_is_better:
        ncp = (diff - margin) / se
    else:
        ncp = -(diff + margin) / se
    t_crit = D.t_ppf(1.0 - alpha, df)
    return 1.0 - D.nct_cdf(t_crit, df, ncp)


def sup_means_power_at_n(*, mean1: float, mean2: float, sd: float,
                         margin: float, alpha: float,
                         n1: int, n2: int,
                         higher_is_better: bool = True) -> float:
    return _sup_means_power(mean1, mean2, sd, margin, alpha,
                            n1, n2, higher_is_better)


def sup_means_n_for_power(*, mean1: float, mean2: float, sd: float,
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
        return _sup_means_power(mean1, mean2, sd, margin, alpha,
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
    return n1, n2, _sup_means_power(mean1, mean2, sd, margin, alpha,
                                    n1, n2, higher_is_better)


def superiority_by_margin_two_means(
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
    """Superiority-by-margin test for two means, pooled-variance t.

    For higher_is_better=True (default), tests
        H0: mean1 - mean2 <= +|margin|  vs  H1: mean1 - mean2 > +|margin|.
    For higher_is_better=False, tests
        H0: mean1 - mean2 >= -|margin|  vs  H1: mean1 - mean2 < -|margin|.

    Provide one of (power, (n1,n2)); the other is solved for.
    """
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
        achieved = sup_means_power_at_n(
            mean1=mean1, mean2=mean2, sd=sd, margin=margin, alpha=alpha,
            n1=n1, n2=n2, higher_is_better=higher_is_better,
        )
        result = {"n1": n1, "n2": n2, "n": n1 + n2,
                  "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n1r, n2r, achieved = sup_means_n_for_power(
            mean1=mean1, mean2=mean2, sd=sd, margin=margin,
            alpha=alpha, power=power,
            higher_is_better=higher_is_better, allocation=allocation,
        )
        result = {"n1": n1r, "n2": n2r, "n": n1r + n2r,
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "superiority_by_margin_two_means",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Chow, Shao & Wang (2008). Sample Size Calculations in Clinical Research.",
            "Julious, S.A. (2010). Sample Sizes for Clinical Trials.",
        ],
    }


# -------------------- Two-proportion superiority-by-margin -----------------

def _sup_props_power(p1: float, p2: float, margin: float, alpha: float,
                     n1: int, n2: int, higher_is_better: bool) -> float:
    """Power for one-sided unpooled-variance z-test for superiority by margin.

    For higher_is_better:
        H0: p1 - p2 <= +|margin|  vs  H1: p1 - p2 > +|margin|
        z = ((p1hat - p2hat) - |margin|) / se
        Under H1 (true p1, p2): power = 1 - Phi(z_alpha - (D - |margin|)/se)
    For lower_is_better:
        H0: p1 - p2 >= -|margin|  vs  H1: p1 - p2 < -|margin|
        power = 1 - Phi(z_alpha - (-|margin| - D)/se)
    """
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("p1, p2 must be in (0, 1)")
    if margin <= 0:
        raise ValueError("margin must be > 0")
    if n1 < 2 or n2 < 2:
        return 0.0
    from scipy.stats import norm
    diff = p1 - p2
    if higher_is_better:
        delta = diff - margin
    else:
        delta = -(diff + margin)
    # Unpooled variance under H1 (standard for z-test on the difference).
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_alpha = D.norm_ppf(1 - alpha)
    return float(1.0 - norm.cdf(z_alpha - delta / se))


def superiority_by_margin_two_proportions(
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
    """Superiority-by-margin test for two proportions, unpooled z-test.

    For higher_is_better=True (default), tests
        H0: p1 - p2 <= +|margin|  vs  H1: p1 - p2 > +|margin|.
    For higher_is_better=False, tests
        H0: p1 - p2 >= -|margin|  vs  H1: p1 - p2 < -|margin|.
    """
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
        return _sup_props_power(p1, p2, margin, alpha, n1_val, n2_val,
                                higher_is_better), n2_val

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _sup_props_power(p1, p2, margin, alpha, n1, n2,
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
        "method_id": "superiority_by_margin_two_proportions",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Chow, Shao & Wang (2008). Sample Size Calculations in Clinical Research.",
            "Farrington, C.P. & Manning, G. (1990). Statistics in Medicine.",
        ],
    }


# -------------------- One-proportion superiority-by-margin -----------------

def _sup_one_prop_power(p: float, p0: float, alpha: float, n: int,
                        higher_is_better: bool, se_method: str) -> float:
    """Normal-approximation power for one-sample superiority-by-margin.

    Mathematically identical to the NI one-proportion power formula --
    the only difference is the sign / placement of the cutoff P0
    relative to the baseline.  This test shares the same
    Z-test formulas as Chapter 105 (Non-Inferiority).

      Upper one-sided (higher is better), H0: P <= P0 vs H1: P > P0
        Power = 1 - Phi((sqrt(n)(P0 - P1) + z_alpha * sd0) / sd1)

      Lower one-sided (higher is worse), H0: P >= P0 vs H1: P < P0
        Power = Phi((sqrt(n)(P0 - P1) - z_alpha * sd0) / sd1)
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
        arg = (math.sqrt(n) * (p0 - p) + z_alpha * sd0) / sd1
        return float(1.0 - norm.cdf(arg))
    else:
        arg = (math.sqrt(n) * (p0 - p) - z_alpha * sd0) / sd1
        return float(norm.cdf(arg))


def superiority_by_margin_one_proportion(
    *,
    p: float,
    p0: float | None = None,
    margin: float | None = None,
    alpha: float = 0.025,
    power: float | None = None,
    n: int | None = None,
    higher_is_better: bool = True,
    se_method: str = "s_p0",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Superiority-by-margin test for a single proportion.

    Uses the normal-approximation z-test with either
    S(P0) (default) or S(Phat) for the standard error.

    Specify the cutoff either directly as ``p0`` (the smallest
    proportion still considered "superior by the margin") or
    implicitly via ``margin``.  When ``margin`` is supplied the
    cutoff is computed as

      higher_is_better=True  -> p0 = p_baseline + |margin|
      higher_is_better=False -> p0 = p_baseline - |margin|

    with ``p_baseline`` defaulting to (p - margin) when only margin
    and the actual proportion p are known.  Typically,
    however, p0 and the actual proportion p (P1) are independent
    inputs, so prefer supplying ``p0`` explicitly.
    """
    if p0 is None:
        if margin is None:
            raise ValueError("supply either p0 or margin")
        if margin <= 0:
            raise ValueError("margin must be > 0 (magnitude)")
        # When only margin is given, assume p is the actual proportion
        # and the baseline equals p - margin (higher_is_better) so
        # that p0 = baseline + margin == p.  This is a degenerate
        # default; p0 supplied separately.
        baseline = p - margin if higher_is_better else p + margin
        if higher_is_better:
            p0 = baseline + margin
        else:
            p0 = baseline - margin
    if not 0 < p0 < 1:
        raise ValueError("p0 must be in (0, 1)")

    inputs_echo = {
        "p": p, "p0": p0, "margin": margin, "alpha": alpha,
        "power": power, "n": n,
        "higher_is_better": higher_is_better, "se_method": se_method,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def p_at(n_val):
        return _sup_one_prop_power(p, p0, alpha, n_val,
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
        "method_id": "superiority_by_margin_one_proportion",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Chow, Shao & Wang (2008). Sample Size Calculations in Clinical Research, p.85.",
            "Fleiss, Levin & Paik (2003). Statistical Methods for Rates and Proportions, 3rd ed.",
        ],
    }


# -------------------- One-mean superiority-by-margin -----------------------

def _sup_one_mean_power(mean: float, ref: float, margin: float, sd: float,
                         alpha: float, n: int,
                         higher_is_better: bool) -> float:
    """One-sample superiority-by-margin t-test power.

    For higher_is_better (Case 1 — high values good):
        H0: mu <= ref + |margin|  vs  H1: mu > ref + |margin|
        ncp = (D - |SM|) / (sd / sqrt(n))  where D = mean - ref
        Power = 1 - T'_{n-1}(t_alpha; ncp)

    For lower_is_better (Case 2 — high values bad):
        H0: mu >= ref - |margin|  vs  H1: mu < ref - |margin|
        ncp = -(D + |margin|) / (sd / sqrt(n))
        Power = T'_{n-1}(t_alpha; ncp)
    """
    if n < 2:
        return 0.0
    if sd <= 0:
        raise ValueError("sd must be positive")
    se = sd / math.sqrt(n)
    diff = mean - ref
    df = n - 1
    t_crit = D.t_ppf(1.0 - alpha, df)
    if higher_is_better:
        ncp = (diff - abs(margin)) / se
    else:
        ncp = -(diff + abs(margin)) / se
    return 1.0 - D.nct_cdf(t_crit, df, ncp)


def superiority_by_margin_one_mean(
    *,
    mean: float,
    ref: float = 0.0,
    margin: float,
    sd: float,
    alpha: float = 0.025,
    power: float | None = None,
    n: int | None = None,
    higher_is_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Superiority-by-margin test for one mean.

    One-sample t-test with a superiority margin SM.

    For higher_is_better=True (default):
        H0: μ ≤ ref + |SM|  vs  H1: μ > ref + |SM|
    For higher_is_better=False:
        H0: μ ≥ ref - |SM|  vs  H1: μ < ref - |SM|

    Parameterisation:
        D  = mean - ref  (true difference)
        SM = margin      (magnitude of superiority margin, > 0)

    Parameters
    ----------
    mean : float
        True mean at which power is computed.
    ref : float
        Reference value (baseline; default 0).
    margin : float
        Superiority margin (positive magnitude |SM|).
    sd : float
        Population standard deviation.
    alpha : float
        One-sided significance level (default 0.025).
    power, n : float/int or None
        Supply one; the other is solved.
    higher_is_better : bool
        Determines direction of the test (default True).
    """
    if margin <= 0:
        raise ValueError("margin must be > 0 (magnitude of superiority margin)")
    if sd <= 0:
        raise ValueError("sd must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    inputs_echo = {
        "mean": mean, "ref": ref, "margin": margin, "sd": sd,
        "alpha": alpha, "power": power, "n": n,
        "higher_is_better": higher_is_better,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def p_at(n_val: int) -> float:
        return _sup_one_mean_power(mean, ref, margin, sd, alpha,
                                   n_val, higher_is_better)

    if solve_for == "power":
        assert n is not None
        achieved = p_at(n)
        result: dict[str, Any] = {"n": n, "achieved_power": achieved}
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
        "method_id": "superiority_by_margin_one_mean",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Chow, S.C.; Shao, J. & Wang, H. (2003). Sample Size Calculations "
            "in Clinical Research, p.50. Marcel Dekker.",
        ],
    }


# -------------------- Two-mean superiority-by-margin on ratio scale --------

def superiority_by_margin_two_means_ratios(
    *,
    r1: float,
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
    """Superiority-by-margin test for two means on the ratio scale.

    Superiority-by-margin test for two means using
    Ratios" (Julious 2004).  Power and sample size are computed by
    log-transforming the inputs and dispatching to the
    difference-scale superiority-by-margin two-mean t-test:

    * SD on log scale: ``sigma_X = sqrt(ln(CV^2 + 1))``
    * Margin on log scale (magnitude):
        - higher_is_better=True  -> SM' = ln(1 + |margin|)
        - higher_is_better=False -> SM' = -ln(1 - |margin|)
    * True difference on log scale: D = ln(r1)

    Parameters
    ----------
    r1
        True ratio mu_T / mu_R.  Must be positive.  When
        higher_is_better=True it should exceed 1 + margin for non-zero
        power; when False, below 1 - margin.
    margin
        Magnitude of the superiority margin (SM) on the ratio scale.
        Strictly positive.  When higher_is_better=True the superiority
        bound is 1 + margin; when False the bound is 1 - margin
        (must be > 0).
    cv
        Coefficient of variation on the original (unlogged) scale,
        as a decimal.  Must be positive.
    alpha, power, n1, n2, allocation, higher_is_better, solve_for
        As for the difference-scale variant.
    """
    if cv <= 0:
        raise ValueError("cv must be positive")
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    if margin <= 0:
        raise ValueError("margin must be > 0 (magnitude)")
    if not higher_is_better:
        if margin >= 1.0:
            raise ValueError(
                "margin must be < 1 when higher_is_better=False "
                "(superiority bound = 1 - margin must remain positive)"
            )

    sigma_log = math.sqrt(math.log(cv * cv + 1.0))
    if higher_is_better:
        margin_log = math.log(1.0 + margin)
    else:
        margin_log = -math.log(1.0 - margin)
    diff_log = math.log(r1)

    inner = superiority_by_margin_two_means(
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
        "method_id": "superiority_by_margin_two_means_ratios",
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
            "using Ratios",
            "Julious, S.A. (2004). Tutorial in biostatistics: sample sizes "
            "for clinical trials with normal data. Statistics in Medicine, "
            "23:1921-1986.",
            "Chow, Shao & Wang (2003). Sample Size Calculations in Clinical Research.",
        ],
    }
