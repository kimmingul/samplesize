"""Two-sample t-test power / sample-size.

(Enter Means)".  Pooled-variance Student t.

  df = n1 + n2 - 2
  λ  = (μ1 - μ2) / (σ · √(1/n1 + 1/n2))

  two-sided power = [1 - T'(t_{α/2}; df, λ)] + T'(-t_{α/2}; df, λ)
  one-sided power = 1 - T'(t_α; df, λ)              (Ha: μ1 > μ2)
                  = T'(-t_α; df, λ)                  (Ha: μ1 < μ2)
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D
from samplesize.core import effect_sizes as E


def _power(mean1: float, mean2: float, sd: float,
           n1: int, n2: int, alpha: float, sides: int) -> float:
    if n1 < 2 or n2 < 2:
        return 0.0
    df = n1 + n2 - 2
    se = sd * math.sqrt(1.0 / n1 + 1.0 / n2)
    ncp = (mean1 - mean2) / se
    if sides == 2:
        t_crit = D.t_ppf(1 - alpha / 2.0, df)
        upper = 1.0 - D.nct_cdf(t_crit, df, ncp)
        lower = D.nct_cdf(-t_crit, df, ncp)
        return upper + lower
    if sides == 1:
        t_crit = D.t_ppf(1 - alpha, df)
        if mean1 >= mean2:
            return 1.0 - D.nct_cdf(t_crit, df, ncp)
        return D.nct_cdf(-t_crit, df, ncp)
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def power_at_n(*, mean1: float, mean2: float, sd: float,
               n1: int, n2: int, alpha: float, sides: int = 2) -> float:
    return _power(mean1, mean2, sd, n1, n2, alpha, sides)


def n_for_power(*, mean1: float, mean2: float, sd: float, alpha: float,
                power: float, sides: int = 2, allocation: float = 1.0,
                n_min: int = 2, n_max: int = 10_000_000) -> tuple[int, int, float]:
    """Solve for (n1, n2, achieved_power) where n2 = ceil(allocation · n1)."""
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if mean1 == mean2:
        raise ValueError("mean1 and mean2 must differ to solve for N")
    if allocation <= 0:
        raise ValueError("allocation (n2/n1) must be > 0")

    def n2_for(n1):
        return max(2, math.ceil(allocation * n1))

    def p_at(n1):
        n2 = n2_for(n1)
        return _power(mean1, mean2, sd, n1, n2, alpha, sides)

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
    return n1, n2, _power(mean1, mean2, sd, n1, n2, alpha, sides)


def effect_for_power(*, mean1: float, sd: float, n1: int, n2: int,
                     alpha: float, power: float, sides: int = 2,
                     direction: str = "above",
                     tol: float = 1e-6) -> float:
    """Minimum detectable mean2 in the requested direction relative to mean1."""
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    sign = +1.0 if direction == "above" else -1.0
    delta_lo, delta_hi = 0.0, max(sd, 1.0)
    for _ in range(60):
        m2 = mean1 + sign * delta_hi
        if _power(mean1, m2, sd, n1, n2, alpha, sides) >= power:
            break
        delta_hi *= 2.0
    else:
        raise RuntimeError("failed to bracket detectable effect")
    for _ in range(200):
        mid = 0.5 * (delta_lo + delta_hi)
        m2 = mean1 + sign * mid
        if _power(mean1, m2, sd, n1, n2, alpha, sides) >= power:
            delta_hi = mid
        else:
            delta_lo = mid
        if delta_hi - delta_lo < tol:
            break
    return mean1 + sign * delta_hi


def two_sample_t_equal_var(
    *,
    mean1: float,
    mean2: float | None = None,
    sd: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    solve_for: str | None = None,
    direction: str = "above",
) -> dict[str, Any]:
    """Pooled-variance two-sample t-test solver.

    Provide exactly two of (`mean2`, `power`, `n1`-with-allocation); the
    missing one is solved for.  `n2` defaults to `n1 * allocation`.
    """
    inputs_echo = {
        "mean1": mean1, "mean2": mean2, "sd": sd, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2, "sides": sides,
        "allocation": allocation,
    }

    # Determine effective n1, n2 if both given.
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    elif n2 is not None and n1 is None:
        n1 = max(2, math.ceil(n2 / allocation))

    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    have_effect = mean2 is not None

    given = sum((have_n, have_power, have_effect))
    if given < 2:
        raise ValueError("supply exactly two of (mean2, power, n)")

    if solve_for is None:
        if not have_n:
            solve_for = "n"
        elif not have_power:
            solve_for = "power"
        elif not have_effect:
            solve_for = "effect"
        else:
            raise ValueError("all three of (mean2, power, n) provided")

    if solve_for == "power":
        assert mean2 is not None and n1 is not None and n2 is not None
        achieved = power_at_n(
            mean1=mean1, mean2=mean2, sd=sd, n1=n1, n2=n2,
            alpha=alpha, sides=sides,
        )
        result = {
            "n1": n1, "n2": n2, "n": n1 + n2,
            "achieved_power": achieved,
            "effect_d": E.cohens_d(mean1, mean2, sd),
        }
    elif solve_for == "n":
        assert mean2 is not None and power is not None
        n1r, n2r, achieved = n_for_power(
            mean1=mean1, mean2=mean2, sd=sd, alpha=alpha, power=power,
            sides=sides, allocation=allocation,
        )
        result = {
            "n1": n1r, "n2": n2r, "n": n1r + n2r,
            "achieved_power": achieved,
            "effect_d": E.cohens_d(mean1, mean2, sd),
        }
    elif solve_for == "effect":
        assert n1 is not None and n2 is not None and power is not None
        m2 = effect_for_power(
            mean1=mean1, sd=sd, n1=n1, n2=n2, alpha=alpha, power=power,
            sides=sides, direction=direction,
        )
        achieved = power_at_n(
            mean1=mean1, mean2=m2, sd=sd, n1=n1, n2=n2,
            alpha=alpha, sides=sides,
        )
        result = {
            "n1": n1, "n2": n2, "n": n1 + n2,
            "mean2": m2,
            "achieved_power": achieved,
            "effect_d": E.cohens_d(mean1, m2, sd),
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "two_sample_t_equal_var",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Julious, S.A. (2010). Sample Sizes for Clinical Trials.",
            "Machin, Campbell, Fayers & Pinol (1997). Sample Size Tables.",
        ],
    }


# ---------------------------------------------------------------------------
# Welch / Aspin-Welch-Satterthwaite t-test (unequal variances).
#
# Two-sample t-test variants
#   "Two-Sample T-Tests Allowing Unequal Variance (Enter Means)"      (424)
#   "Two-Sample T-Tests Allowing Unequal Variance (Enter Difference)" (425)
#
# Test statistic (population analogue):
#   SE = sqrt(σ1² / n1 + σ2² / n2)
#   λ  = (μ1 - μ2) / SE       (non-centrality)
#   df = (σ1²/n1 + σ2²/n2)²
#        ----------------------------------------------
#        (σ1²/n1)² / (n1 - 1)  +  (σ2²/n2)² / (n2 - 1)   (Welch-Satterthwaite)
#
# Power is computed from the non-central t with that df and ncp, mirroring
# `_power` above.  All three Welch entry points reuse the same kernel.
# ---------------------------------------------------------------------------


def _power_welch(delta: float, sd1: float, sd2: float,
                 n1: int, n2: int, alpha: float, sides: int) -> float:
    """Aspin-Welch-Satterthwaite t-test power (population formula)."""
    if n1 < 2 or n2 < 2:
        return 0.0
    if sd1 <= 0.0 or sd2 <= 0.0:
        raise ValueError("sd1 and sd2 must be > 0")
    v1 = (sd1 * sd1) / n1
    v2 = (sd2 * sd2) / n2
    se = math.sqrt(v1 + v2)
    # Welch-Satterthwaite degrees of freedom
    num = (v1 + v2) ** 2
    den = (v1 * v1) / (n1 - 1) + (v2 * v2) / (n2 - 1)
    df = num / den
    ncp = delta / se
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


def _n_for_power_welch(*, delta: float, sd1: float, sd2: float, alpha: float,
                       power: float, sides: int, allocation: float,
                       n_min: int = 2,
                       n_max: int = 10_000_000) -> tuple[int, int, float]:
    """Solve for (n1, n2, achieved_power) with n2 = ceil(allocation · n1)."""
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if delta == 0:
        raise ValueError("delta must be non-zero to solve for N")
    if allocation <= 0:
        raise ValueError("allocation (n2/n1) must be > 0")

    def n2_for(n1: int) -> int:
        return max(2, math.ceil(allocation * n1))

    def p_at(n1: int) -> float:
        return _power_welch(delta, sd1, sd2, n1, n2_for(n1), alpha, sides)

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
    return n1, n2, _power_welch(delta, sd1, sd2, n1, n2, alpha, sides)


def _effect_for_power_welch(*, sd1: float, sd2: float, n1: int, n2: int,
                            alpha: float, power: float, sides: int,
                            direction: str = "above",
                            tol: float = 1e-6) -> float:
    """Minimum detectable |δ| in the requested direction (signed)."""
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    sign = +1.0 if direction == "above" else -1.0
    lo, hi = 0.0, max(sd1, sd2, 1.0)
    for _ in range(60):
        if _power_welch(sign * hi, sd1, sd2, n1, n2, alpha, sides) >= power:
            break
        hi *= 2.0
    else:
        raise RuntimeError("failed to bracket detectable effect")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _power_welch(sign * mid, sd1, sd2, n1, n2, alpha, sides) >= power:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return sign * hi


def _welch_solve(*,
                 delta: float,
                 sd1: float,
                 sd2: float,
                 alpha: float,
                 power: float | None,
                 n1: int | None,
                 n2: int | None,
                 sides: int,
                 allocation: float,
                 solve_for: str | None,
                 direction: str,
                 method_id: str,
                 chapter_label: str,
                 inputs_echo: dict[str, Any],
                 ) -> dict[str, Any]:
    """Shared dispatcher for both Welch (means and diff) variants."""
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    elif n2 is not None and n1 is None:
        n1 = max(2, math.ceil(n2 / allocation))

    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    have_effect = delta is not None and not (
        # `delta is None` flag is encoded as math.nan from the caller
        isinstance(delta, float) and math.isnan(delta)
    )

    given = sum((have_n, have_power, have_effect))
    if given < 2:
        raise ValueError("supply exactly two of (effect, power, n)")

    if solve_for is None:
        if not have_n:
            solve_for = "n"
        elif not have_power:
            solve_for = "power"
        elif not have_effect:
            solve_for = "effect"
        else:
            raise ValueError("all three of (effect, power, n) provided")

    if solve_for == "power":
        assert n1 is not None and n2 is not None and have_effect
        achieved = _power_welch(delta, sd1, sd2, n1, n2, alpha, sides)
        result = {
            "n1": n1, "n2": n2, "n": n1 + n2,
            "achieved_power": achieved,
            "delta": delta,
        }
    elif solve_for == "n":
        assert have_effect and power is not None
        n1r, n2r, achieved = _n_for_power_welch(
            delta=delta, sd1=sd1, sd2=sd2, alpha=alpha, power=power,
            sides=sides, allocation=allocation,
        )
        result = {
            "n1": n1r, "n2": n2r, "n": n1r + n2r,
            "achieved_power": achieved,
            "delta": delta,
        }
    elif solve_for == "effect":
        assert n1 is not None and n2 is not None and power is not None
        d = _effect_for_power_welch(
            sd1=sd1, sd2=sd2, n1=n1, n2=n2, alpha=alpha, power=power,
            sides=sides, direction=direction,
        )
        achieved = _power_welch(d, sd1, sd2, n1, n2, alpha, sides)
        result = {
            "n1": n1, "n2": n2, "n": n1 + n2,
            "delta": d,
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": method_id,
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            f"Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. ({chapter_label})",
            "Welch, B.L. (1938/1947). The significance of the difference "
            "between two means when the population variances are unequal.",
            "Julious, S.A. (2010). Sample Sizes for Clinical Trials.",
            "Chow, Shao & Wang (2008). Sample Size Calculations in "
            "Clinical Research, 2nd ed.",
        ],
    }


def two_sample_t_welch_means(
    *,
    mean1: float,
    mean2: float | None = None,
    sd1: float,
    sd2: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    solve_for: str | None = None,
    direction: str = "above",
) -> dict[str, Any]:
    """Welch (unequal-variance) two-sample t-test, enter-means form.

    Provide exactly two of (`mean2`, `power`, `n1`/`n2`).  When solving for
    the effect, the result is reported both as the absolute mean difference
    (`delta`) and as the implied `mean2`.
    """
    inputs_echo = {
        "mean1": mean1, "mean2": mean2,
        "sd1": sd1, "sd2": sd2,
        "alpha": alpha, "power": power,
        "n1": n1, "n2": n2,
        "sides": sides, "allocation": allocation,
    }
    delta = (mean1 - mean2) if mean2 is not None else float("nan")
    res = _welch_solve(
        delta=delta, sd1=sd1, sd2=sd2,
        alpha=alpha, power=power, n1=n1, n2=n2,
        sides=sides, allocation=allocation,
        solve_for=solve_for, direction=direction,
        method_id="two_sample_t_welch_means",
        chapter_label="Two-Sample T-Tests Allowing Unequal Variance "
                      "(Enter Means)",
        inputs_echo=inputs_echo,
    )
    if res["solve_for"] == "effect":
        # Convert solved Δ back to mean2 using mean1 + (-Δ)?  Chapter convention:
        # δ = μ1 - μ2, so μ2 = μ1 - δ.
        res["mean2"] = mean1 - res["delta"]
        res["effect_d"] = E.cohens_d(
            mean1, res["mean2"], math.sqrt(0.5 * (sd1 * sd1 + sd2 * sd2)),
        )
    else:
        if mean2 is not None:
            res["mean2"] = mean2
            res["effect_d"] = E.cohens_d(
                mean1, mean2, math.sqrt(0.5 * (sd1 * sd1 + sd2 * sd2)),
            )
    return res


def two_sample_t_welch_diff(
    *,
    delta: float | None = None,
    sd1: float,
    sd2: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    solve_for: str | None = None,
    direction: str = "above",
) -> dict[str, Any]:
    """Welch (unequal-variance) two-sample t-test, enter-difference form.

    Numerically identical to `two_sample_t_welch_means`; the difference
    `δ = μ1 - μ2` is supplied directly instead of via two means.
    """
    inputs_echo = {
        "delta": delta,
        "sd1": sd1, "sd2": sd2,
        "alpha": alpha, "power": power,
        "n1": n1, "n2": n2,
        "sides": sides, "allocation": allocation,
    }
    d = delta if delta is not None else float("nan")
    res = _welch_solve(
        delta=d, sd1=sd1, sd2=sd2,
        alpha=alpha, power=power, n1=n1, n2=n2,
        sides=sides, allocation=allocation,
        solve_for=solve_for, direction=direction,
        method_id="two_sample_t_welch_diff",
        chapter_label="Two-Sample T-Tests Allowing Unequal Variance "
                      "(Enter Difference)",
        inputs_echo=inputs_echo,
    )
    return res


# ---------------------------------------------------------------------------
# Equal-variance two-sample t-test, enter-difference parameterisation.
#
# Two-sample t-test with effect-size input
#   "Two-Sample T-Tests Assuming Equal Variance (Enter Difference)" (423)
#
# Identical kernel to `two_sample_t_equal_var` (pooled SD, df = n1 + n2 - 2)
# but parametrised directly on δ = μ1 - μ2 instead of (μ1, μ2).
# ---------------------------------------------------------------------------


def two_sample_t_equal_var_diff(
    *,
    delta: float | None = None,
    sd: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    solve_for: str | None = None,
    direction: str = "above",
) -> dict[str, Any]:
    """Pooled-variance two-sample t-test, enter-difference form.

    Internally delegates to `two_sample_t_equal_var` with `mean1 = 0` and
    `mean2 = -delta` so that `mean1 - mean2 = delta`.  When solving for the
    effect, returns the solved `delta`.
    """
    inputs_echo = {
        "delta": delta, "sd": sd,
        "alpha": alpha, "power": power,
        "n1": n1, "n2": n2,
        "sides": sides, "allocation": allocation,
    }

    # Translate to enter-means call (μ1 = 0, μ2 = -δ → μ1 - μ2 = δ).
    mean1 = 0.0
    mean2 = -delta if delta is not None else None

    inner = two_sample_t_equal_var(
        mean1=mean1, mean2=mean2, sd=sd, alpha=alpha,
        power=power, n1=n1, n2=n2, sides=sides,
        allocation=allocation, solve_for=solve_for,
        direction=direction,
    )

    # Repackage with delta and the diff-form method id.
    if inner["solve_for"] == "effect":
        solved_mean2 = inner["mean2"]
        resolved_delta = mean1 - solved_mean2
    else:
        resolved_delta = delta

    out: dict[str, Any] = {
        "method_id": "two_sample_t_equal_var_diff",
        "solve_for": inner["solve_for"],
        "n1": inner["n1"],
        "n2": inner["n2"],
        "n": inner["n"],
        "achieved_power": inner["achieved_power"],
        "delta": resolved_delta,
        "effect_d": (resolved_delta / sd) if resolved_delta is not None else None,
        "inputs_echo": inputs_echo,
        "citations": [
            "(Enter Difference)",
            "Julious, S.A. (2010). Sample Sizes for Clinical Trials.",
            "Machin, Campbell, Fayers & Pinol (1997). Sample Size Tables.",
        ],
    }
    return out


# ---------------------------------------------------------------------------
# Log-scale two-sample t-test parametrised on a ratio / fold change.
#
# Two-sample t-test variants
#   "Tests for Fold Change of Two Means"           (446)  -> fold change FC
#   "Tests for Two Means using Ratios"             (445)  -> mean ratio  phi
#
# Both procedures assume :math:`Y_i = \exp(X_i)` is lognormal with the
# logged response :math:`X_i = \ln(Y_i)` normally distributed with
# pooled within-group SD :math:`\sigma_X = \sqrt{\ln(\mathrm{COV}^2 + 1)}`.
# A hypothesis about the ratio on the original (Y) scale translates
# directly into a hypothesis about the difference of means on the log
# (X) scale, so power and sample size reduce to the equal-variance
# two-sample t-test on the logged data with mean difference
# :math:`\ln(R_1) - \ln(R_0) = \ln(R_1 / R_0)`.
# ---------------------------------------------------------------------------


def _sigma_from_cov(cov: float) -> float:
    if cov <= 0:
        raise ValueError("cov (coefficient of variation) must be > 0")
    return math.sqrt(math.log(cov * cov + 1.0))


def _ratio_solver(
    *,
    ratio_h0: float,
    ratio_h1: float,
    cov: float,
    alpha: float,
    power: float | None,
    n1: int | None,
    n2: int | None,
    sides: int,
    allocation: float,
    direction: str,
    solve_for: str | None,
    method_id: str,
    chapter_label: str,
    inputs_echo: dict[str, Any],
    ratio_h0_key: str,
    ratio_h1_key: str,
) -> dict[str, Any]:
    """Shared dispatcher for fold-change and ratio chapters."""
    if ratio_h0 <= 0 or ratio_h1 <= 0:
        raise ValueError("ratios under H0 and H1 must both be > 0")
    if ratio_h0 == ratio_h1 and (power is not None and n1 is None):
        raise ValueError("ratio under H0 and H1 must differ to solve for N")

    sigma = _sigma_from_cov(cov)
    log_h0 = math.log(ratio_h0)
    log_h1 = math.log(ratio_h1)
    # On the log scale we view group 2 (treatment) minus group 1 (reference)
    # so mean_diff = ln(R1) - ln(R0).  Then mean1 = 0, mean2 = -mean_diff
    # gives mean1 - mean2 = mean_diff, matching the equal-var solver's
    # convention.
    mean_diff = log_h1 - log_h0
    inner = two_sample_t_equal_var(
        mean1=0.0,
        mean2=(-mean_diff) if mean_diff != 0 else None,
        sd=sigma,
        alpha=alpha,
        power=power,
        n1=n1,
        n2=n2,
        sides=sides,
        allocation=allocation,
        solve_for=solve_for,
        direction=direction,
    )
    # Translate inner result back into ratio-scale outputs.
    if inner["solve_for"] == "effect":
        solved_mean2 = inner["mean2"]
        resolved_log_diff = 0.0 - solved_mean2
        resolved_ratio_h1 = math.exp(log_h0 + resolved_log_diff)
    else:
        resolved_log_diff = mean_diff
        resolved_ratio_h1 = ratio_h1

    effect_size = abs(resolved_log_diff) / sigma
    out: dict[str, Any] = {
        "method_id": method_id,
        "solve_for": inner["solve_for"],
        "n1": inner["n1"],
        "n2": inner["n2"],
        "n": inner["n"],
        "achieved_power": inner["achieved_power"],
        ratio_h0_key: ratio_h0,
        ratio_h1_key: resolved_ratio_h1,
        "log_diff": resolved_log_diff,
        "sigma_log": sigma,
        "effect_size": effect_size,
        "inputs_echo": inputs_echo,
        "citations": [
            f"Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. ({chapter_label})",
            "Julious, S.A. (2004). Tutorial in Biostatistics: Sample "
            "sizes for clinical trials with Normal data. Statistics in "
            "Medicine 23:1921-1986.",
            "Chow, S.C., Shao, J., and Wang, H. (2003). Sample Size "
            "Calculations in Clinical Research. Marcel Dekker, New York.",
        ],
    }
    return out


def tests_fold_change_two_means(
    *,
    fc0: float = 1.0,
    fc1: float,
    cov: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    direction: str = "above",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Tests for Fold Change of Two Means.

    The fold change :math:`FC = \\mu_T / \\mu_R` of two lognormal means
    is analysed via the two-sample equal-variance t-test on the logged
    data with pooled within-group SD
    :math:`\\sigma_X = \\sqrt{\\ln(\\mathrm{COV}^2 + 1)}` and mean
    difference :math:`\\ln(FC_1) - \\ln(FC_0)`.

    Parameters
    ----------
    fc0 : float
        Fold change under :math:`H_0` (commonly 1.0).
    fc1 : float
        Fold change under :math:`H_1` at which power is to be evaluated.
    cov : float
        Coefficient of variation of :math:`Y` (original scale).
    alpha : float
        Type-I error rate.
    power : float, optional
        Target power; supply this OR ``n1``.
    n1, n2 : int, optional
        Per-arm sample sizes; if only ``n1`` is given, ``n2 = ceil(allocation*n1)``.
    sides : int
        2 (default) or 1.
    allocation : float
        ``n2/n1`` allocation ratio (default 1.0).
    direction : str
        For ``solve_for='effect'`` only: ``"above"`` or ``"below"``
        (interpreted on the log scale).
    solve_for : {"n", "power", "effect"}, optional
        Override the auto-detected target.
    """
    inputs_echo = {
        "fc0": fc0, "fc1": fc1, "cov": cov, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2, "sides": sides,
        "allocation": allocation, "direction": direction,
    }
    return _ratio_solver(
        ratio_h0=fc0, ratio_h1=fc1, cov=cov, alpha=alpha,
        power=power, n1=n1, n2=n2, sides=sides,
        allocation=allocation, direction=direction,
        solve_for=solve_for,
        method_id="tests_fold_change_two_means",
        chapter_label="Tests for Fold Change of Two Means (chapter 446)",
        inputs_echo=inputs_echo,
        ratio_h0_key="fc0",
        ratio_h1_key="fc1",
    )


def tests_two_means_ratios(
    *,
    r0: float = 1.0,
    r1: float,
    cov: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    direction: str = "above",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Tests for Two Means using Ratios.

    Identical numerics to :func:`tests_fold_change_two_means`: the
    ratio :math:`\\phi = \\mu_T / \\mu_R` of two lognormal means is
    analysed via the two-sample equal-variance t-test on the logged
    data.

    Parameters
    ----------
    r0 : float
        Ratio under :math:`H_0` (commonly 1.0).
    r1 : float
        True ratio under :math:`H_1` at which power is evaluated.
    cov : float
        Coefficient of variation of :math:`Y` (original scale).
    alpha, power, n1, n2, sides, allocation, direction, solve_for
        Same semantics as :func:`tests_fold_change_two_means`.
    """
    inputs_echo = {
        "r0": r0, "r1": r1, "cov": cov, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2, "sides": sides,
        "allocation": allocation, "direction": direction,
    }
    return _ratio_solver(
        ratio_h0=r0, ratio_h1=r1, cov=cov, alpha=alpha,
        power=power, n1=n1, n2=n2, sides=sides,
        allocation=allocation, direction=direction,
        solve_for=solve_for,
        method_id="tests_two_means_ratios",
        chapter_label="Tests for Two Means using Ratios (chapter 445)",
        inputs_echo=inputs_echo,
        ratio_h0_key="r0",
        ratio_h1_key="r1",
    )
