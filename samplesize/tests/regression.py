"""Multiple regression power & sample-size calculators.


Implements three methods:

1. multiple_regression_omnibus
   F-test for ΔR² in nested regression (full vs. reduced model).
   Cohen's f² = R²_{T|C} / (1 − R²_C − R²_{T|C})
   λ = N · f²
   df1 = q   (predictors being tested, set T)
   df2 = N − k − 1   (k = total predictors in C + T)

2. multiple_regression_partial
   t-test (F with df1=1) for one predictor controlling for the others.
   Uses f² = ρ²_partial / (1 − ρ²_partial)
   which equals R²_{T|C}/(1−R²_{TC}) when q=1.
   Same noncentral-F machinery; df1 = 1.

3. multivariate_regression_wilks
   Multivariate regression: Wilks' Λ → Rao's F approximation.
   Tests q added predictors in the presence of p response variables.
   λ = N · η / df1   where η = 1 − Λ^{1/g}

   a  = q   (predictors tested)
   b  = p   (response variables)
   g  = sqrt((a²b² − 4) / (a² + b² − 5))   [or 1 if denominator ≤ 0]
   Λ  = (1 − f²)^{1/1} for q=1, or approximated via f² otherwise
   η  = 1 − Λ^{1/g}
   df1 = a · b
   df2 = g · [(N − k − 1) − (b − a + 1)/2] − (a·b − 2)/2

   For the sample-size search the Wilks Λ is mapped from the user-supplied
   partial η² equivalent (delta = R²_{T|C} / (1 − R²_C)):
     Λ_approx = (1 − delta)^a    [exact when a=1 or b=1]

References
----------
* Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences,
  2nd Ed. Lawrence Erlbaum.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _ncf_power(ncp: float, df1: float, df2: float, alpha: float) -> float:
    """Power via noncentral F."""
    if df1 <= 0 or df2 <= 0:
        return 0.0
    from scipy.stats import f as fdist
    f_crit = fdist.ppf(1.0 - alpha, df1, df2)
    return float(1.0 - D.ncf_cdf(f_crit, df1, df2, ncp))


# ---------------------------------------------------------------------------
# Method 6: multiple_regression_omnibus
# ---------------------------------------------------------------------------


def multiple_regression_omnibus(
    *,
    r2_t_given_c: float,
    n_tested: int,
    r2_c: float = 0.0,
    n_controlled: int = 0,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """F-test for ΔR² in nested multiple regression.

    Tests whether the set T of ``n_tested`` predictors adds significant
    explanatory power (R²_{T|C}) after controlling for the set C of
    ``n_controlled`` predictors (with R²_C).

    Parameters
    ----------
    r2_t_given_c
        Incremental R² due to the predictors being tested (R²_{T|C}).
        Must be in (0, 1) and r2_c + r2_t_given_c < 1.
    n_tested
        Number of predictors being tested (q ≥ 1).
    r2_c
        R² of the control predictors alone (default 0 → no controls).
    n_controlled
        Number of control predictors (default 0).
    n
        Total sample size (for solve_for='power').
    alpha
        Significance level (default 0.05).
    power
        Target power (for solve_for='n').
    solve_for
        ``'power'`` or ``'n'``.
    """
    if not 0.0 < r2_t_given_c < 1.0:
        raise ValueError("r2_t_given_c must be in (0, 1)")
    if not 0.0 <= r2_c < 1.0:
        raise ValueError("r2_c must be in [0, 1)")
    if r2_c + r2_t_given_c >= 1.0:
        raise ValueError("r2_c + r2_t_given_c must be < 1")
    if n_tested < 1:
        raise ValueError("n_tested must be >= 1")
    if n_controlled < 0:
        raise ValueError("n_controlled must be >= 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    k_total = n_controlled + n_tested  # total predictors in model
    f2 = r2_t_given_c / (1.0 - r2_c - r2_t_given_c)
    df1 = n_tested

    inputs_echo = {
        "r2_t_given_c": r2_t_given_c, "n_tested": n_tested,
        "r2_c": r2_c, "n_controlled": n_controlled,
        "n": n, "alpha": alpha, "power": power,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def pwr(n_val: int) -> float:
        df2 = n_val - k_total - 1
        if df2 <= 0:
            return 0.0
        ncp = n_val * f2
        return _ncf_power(ncp, df1, df2, alpha)

    if solve_for == "power":
        assert n is not None
        achieved = pwr(n)
        n_out = n
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        # Need n > k_total + 1 for df2 > 0
        n_min = k_total + 2
        lo, hi = n_min, max(n_min, 10)
        while hi <= 10_000_000:
            if pwr(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 10,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pwr(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        achieved = pwr(n_out)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "multiple_regression_omnibus",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "f_squared": f2,
        "df1": df1,
        "df2": n_out - k_total - 1,
        "ncp": n_out * f2,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cohen, J. (1988). Statistical Power Analysis for the Behavioral "
            "Sciences, 2nd Ed. Lawrence Erlbaum.",
        ],
    }


# ---------------------------------------------------------------------------
# Method 7: multiple_regression_partial
# ---------------------------------------------------------------------------


def multiple_regression_partial(
    *,
    rho_partial_squared: float,
    n_other: int = 0,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """t-test (F, df1=1) for one partial regression coefficient.

    Tests whether a single predictor X_j is significant after controlling
    for the remaining ``n_other`` predictors.

    Parameters
    ----------
    rho_partial_squared
        Squared partial correlation of X_j with Y, partialling out the
        other predictors.  ρ²_partial = R²_{T|C} / (1 − R²_C) when q=1.
        Must be in (0, 1).
    n_other
        Number of other predictors in the model (k − 1).
    n
        Total sample size (for solve_for='power').
    alpha
        Significance level (default 0.05).
    power
        Target power (for solve_for='n').
    solve_for
        ``'power'`` or ``'n'``.
    """
    if not 0.0 < rho_partial_squared < 1.0:
        raise ValueError("rho_partial_squared must be in (0, 1)")
    if n_other < 0:
        raise ValueError("n_other must be >= 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    # f² = ρ²_p / (1 − ρ²_p)
    f2 = rho_partial_squared / (1.0 - rho_partial_squared)
    df1 = 1
    k_total = n_other + 1  # total predictors including tested one

    inputs_echo = {
        "rho_partial_squared": rho_partial_squared, "n_other": n_other,
        "n": n, "alpha": alpha, "power": power,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def pwr(n_val: int) -> float:
        df2 = n_val - k_total - 1
        if df2 <= 0:
            return 0.0
        ncp = n_val * f2
        return _ncf_power(ncp, df1, df2, alpha)

    if solve_for == "power":
        assert n is not None
        achieved = pwr(n)
        n_out = n
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        n_min = k_total + 2
        lo, hi = n_min, max(n_min, 10)
        while hi <= 10_000_000:
            if pwr(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 10,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pwr(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        achieved = pwr(n_out)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "multiple_regression_partial",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "f_squared": f2,
        "df1": df1,
        "df2": n_out - k_total - 1,
        "ncp": n_out * f2,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cohen, J. (1988). Statistical Power Analysis for the Behavioral "
            "Sciences, 2nd Ed. Lawrence Erlbaum.",
        ],
    }


# ---------------------------------------------------------------------------
# Method 8: multivariate_regression_wilks
# ---------------------------------------------------------------------------


def _wilks_g(a: float, b: float) -> float:
    """Rao's g factor for Wilks' Λ F approximation."""
    denom = a ** 2 + b ** 2 - 5.0
    if denom <= 0:
        return 1.0
    return math.sqrt((a ** 2 * b ** 2 - 4.0) / denom)


def multivariate_regression_wilks(
    *,
    r2_t_given_c: float,
    n_responses: int,
    n_tested: int,
    n_controlled: int = 0,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Multivariate regression Wilks' Λ F-approximation.

    Tests q predictors (n_tested) for their joint effect on p response
    variables (n_responses), after controlling for n_controlled other
    predictors.  Uses Rao's F approximation to Wilks' Λ.

    Parameters
    ----------
    r2_t_given_c
        Multivariate R² increment attributable to the tested predictors.
        Scalar summary: δ = R²_{T|C}  (0 < δ < 1).
    n_responses
        Number of response variables (p ≥ 2).
    n_tested
        Number of predictors being tested (q ≥ 1).
    n_controlled
        Number of control predictors (default 0).
    n
        Total sample size (for solve_for='power').
    alpha
        Significance level (default 0.05).
    power
        Target power (for solve_for='n').
    solve_for
        ``'power'`` or ``'n'``.
    """
    if not 0.0 < r2_t_given_c < 1.0:
        raise ValueError("r2_t_given_c must be in (0, 1)")
    if n_responses < 2:
        raise ValueError("n_responses must be >= 2")
    if n_tested < 1:
        raise ValueError("n_tested must be >= 1")
    if n_controlled < 0:
        raise ValueError("n_controlled must be >= 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    a = float(n_tested)
    b = float(n_responses)
    k_total = n_controlled + n_tested

    # Approximate Wilks' Λ from the multivariate R²:
    #   δ = R²_{T|C}  →  Λ ≈ (1 − δ)^a   [exact for a=1 or b=1]
    delta = r2_t_given_c
    g = _wilks_g(a, b)

    inputs_echo = {
        "r2_t_given_c": r2_t_given_c, "n_responses": n_responses,
        "n_tested": n_tested, "n_controlled": n_controlled,
        "n": n, "alpha": alpha, "power": power,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def pwr(n_val: int) -> float:
        r = k_total  # rank of X (model rank including intercept offset)
        df_error = n_val - r - 1  # = N - k_total - 1 for regression
        if df_error <= 0:
            return 0.0
        wilks_lambda = (1.0 - delta) ** a
        eta = 1.0 - wilks_lambda ** (1.0 / g)
        df1 = a * b
        df2 = g * (df_error - (b - a + 1.0) / 2.0) - (a * b - 2.0) / 2.0
        if df1 <= 0 or df2 <= 0:
            return 0.0
        F_stat = (eta / df1) / ((1.0 - eta) / df2)
        ncp = df1 * F_stat
        return _ncf_power(ncp, df1, df2, alpha)

    if solve_for == "power":
        assert n is not None
        achieved = pwr(n)
        n_out = n
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        n_min = k_total + 2
        lo, hi = n_min, max(n_min, 10)
        while hi <= 10_000_000:
            if pwr(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 10,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pwr(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        achieved = pwr(n_out)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    # Compute final statistics
    df_error = n_out - k_total - 1
    wilks_lambda = (1.0 - delta) ** a
    eta = 1.0 - wilks_lambda ** (1.0 / g)
    df1 = a * b
    df2 = g * (df_error - (b - a + 1.0) / 2.0) - (a * b - 2.0) / 2.0
    F_stat = (eta / df1) / ((1.0 - eta) / df2) if df2 > 0 else 0.0

    return {
        "method_id": "multivariate_regression_wilks",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "wilks_lambda": wilks_lambda,
        "approx_f": F_stat,
        "df1": df1,
        "df2": df2,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cohen, J. (1988). Statistical Power Analysis for the Behavioral "
            "Sciences, 2nd Ed. Lawrence Erlbaum.",
            "Rao, C.R. (1951). 'An Asymptotic Expansion of the Distribution "
            "of Wilks' Criterion.' Bull. Inst. Internat. Statist. 33, 177-180.",
        ],
    }
