"""One-way ANOVA F-test power & sample-size.


  σ_m² = Σᵢ nᵢ·(μᵢ - μ̄_W)² / N        (weighted between-mean variance)
  μ̄_W = Σᵢ (nᵢ/N) μᵢ                   (weighted mean)
  λ   = N · σ_m² / σ²                  (NCP for noncentral F)
  df₁ = k - 1,  df₂ = N - k
  Power = 1 - F'(F_{α, df₁, df₂}; df₁, df₂, λ)

Equivalent effect-size parameterisation (Cohen's f):
  f²  = σ_m² / σ²,  λ = N · f²
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


def _check_means(means: list[float], allocation: list[int]):
    if len(means) < 2:
        raise ValueError("need at least 2 group means")
    if len(allocation) != len(means):
        raise ValueError("len(allocation) must equal len(means)")


def _sigma_m_squared(means: list[float], ni: list[int]) -> float:
    n = sum(ni)
    mu_bar = sum(ni[i] * means[i] for i in range(len(means))) / n
    return sum(ni[i] * (means[i] - mu_bar) ** 2 for i in range(len(means))) / n


def _power_anova(means: list[float], ni: list[int], sigma: float,
                 alpha: float) -> float:
    k = len(means)
    n_total = sum(ni)
    if n_total <= k:
        return 0.0
    df1 = k - 1
    df2 = n_total - k
    sm2 = _sigma_m_squared(means, ni)
    ncp = n_total * sm2 / (sigma ** 2)
    from scipy.stats import f as fdist
    f_crit = fdist.ppf(1.0 - alpha, df1, df2)
    return float(1.0 - D.ncf_cdf(f_crit, df1, df2, ncp))


def power_at_n(*, means: list[float], n: int, sigma: float, alpha: float,
               allocation: list[float] | None = None) -> float:
    """Power for per-group base n with equal/unequal allocation."""
    if allocation is None:
        allocation = [1.0] * len(means)
    if len(allocation) != len(means):
        raise ValueError("allocation length must match means length")
    ni = [max(2, math.ceil(n * a)) for a in allocation]
    return _power_anova(means, ni, sigma, alpha)


def n_for_power(*, means: list[float], sigma: float, alpha: float,
                power: float, allocation: list[float] | None = None,
                n_min: int = 2, n_max: int = 1_000_000) -> tuple[int, list[int], float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if allocation is None:
        allocation = [1.0] * len(means)

    def ni_for(n_base):
        return [max(2, math.ceil(n_base * a)) for a in allocation]

    def p_at(n_base):
        return _power_anova(means, ni_for(n_base), sigma, alpha)

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
    ni = ni_for(hi)
    return hi, ni, _power_anova(means, ni, sigma, alpha)


def one_way_f(
    *,
    means: list[float],
    sigma: float,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    allocation: list[float] | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """One-way ANOVA F-test solver. `n` is the per-group base size."""
    inputs_echo = {
        "means": means, "sigma": sigma, "n": n, "alpha": alpha,
        "power": power, "allocation": allocation,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        achieved = power_at_n(means=means, n=n, sigma=sigma, alpha=alpha,
                              allocation=allocation)
        ni = ([max(2, math.ceil(n * a)) for a in (allocation or [1.0] * len(means))])
        result = {"n_per_group": n, "n_per_group_list": ni, "n_total": sum(ni),
                  "achieved_power": achieved,
                  "effect_f": math.sqrt(_sigma_m_squared(means, ni)) / sigma}
    elif solve_for == "n":
        assert power is not None
        n_base, ni, achieved = n_for_power(means=means, sigma=sigma,
                                            alpha=alpha, power=power,
                                            allocation=allocation)
        result = {"n_per_group": n_base, "n_per_group_list": ni,
                  "n_total": sum(ni), "achieved_power": achieved,
                  "effect_f": math.sqrt(_sigma_m_squared(means, ni)) / sigma}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "one_way_anova_f",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.",
            "Fleiss, J. (1986). The Design and Analysis of Clinical Experiments.",
        ],
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#
# Tests a SINGLE linear contrast L = Σ cᵢ μᵢ (with Σ cᵢ = 0) versus L = 0 in
# a one-way design via the noncentral F(1, N-G, λ_C) distribution.
#
#   σ_C  = |Σ cᵢ μᵢ| / √( N · Σ cᵢ² / nᵢ )
#   λ_C  = N · σ_C² / σ²
#         = (Σ cᵢ μᵢ)² / ( σ² · Σ cᵢ² / nᵢ )
#   F_crit = F_{α, 1, N-G}
#   Power  = 1 − F'(F_crit; 1, N-G, λ_C)
#
# Reference: Desu & Raghavarao (1990). Cross-checked via hand calculation
# (means=[1,2,3], c=[-2,1,1], σ=5, n=5 → σ_C²=0.5, λ_C=0.3,
# F_{0.95,1,12}=4.7472, power=0.0797).


def _contrast_sigma_c_squared(
    means: list[float], coefs: list[float], ni: list[int]
) -> float:
    """σ_C² = (Σ cᵢ μᵢ)² / ( N · Σ cᵢ² / nᵢ ).  Returned squared form."""
    n_total = sum(ni)
    num = sum(coefs[i] * means[i] for i in range(len(means))) ** 2
    den = n_total * sum(coefs[i] ** 2 / ni[i] for i in range(len(coefs)))
    if den <= 0:
        return 0.0
    return num / den


def _power_contrast(
    means: list[float],
    coefs: list[float],
    ni: list[int],
    sigma: float,
    alpha: float,
) -> float:
    g = len(means)
    n_total = sum(ni)
    if n_total <= g:
        return 0.0
    sigma_c2 = _contrast_sigma_c_squared(means, coefs, ni)
    ncp = n_total * sigma_c2 / (sigma ** 2)
    df1 = 1
    df2 = n_total - g
    from scipy.stats import f as fdist
    f_crit = fdist.ppf(1.0 - alpha, df1, df2)
    return float(1.0 - D.ncf_cdf(f_crit, df1, df2, ncp))


def _check_contrast(coefs: list[float], means: list[float]) -> None:
    if len(coefs) != len(means):
        raise ValueError("len(contrast) must equal len(means)")
    s = sum(coefs)
    # is "Contrasts" and the technical-details section assumes Σcᵢ = 0.
    # Tolerate small float noise.
    if abs(s) > 1e-6 * max(1.0, max(abs(c) for c in coefs)):
        raise ValueError(
            f"contrast coefficients must sum to zero (sum={s})"
        )


def one_way_anova_contrasts(
    *,
    means: list[float],
    contrast: list[float],
    sigma: float,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    allocation: list[float] | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """One-way ANOVA single-contrast solver.

    Tests whether the linear combination ``L = Σ cᵢ μᵢ`` differs from zero
    using a noncentral F(1, N−G, λ_C) test, where

        σ_C² = (Σ cᵢ μᵢ)² / ( N · Σ cᵢ² / nᵢ )
        λ_C  = N · σ_C² / σ²
        F_crit = F_{α, 1, N-G}
        Power  = 1 − F'(F_crit; 1, N-G, λ_C)

    Parameters
    ----------
    means
        Group means (length G ≥ 2).
    contrast
        Contrast coefficients (length G; must sum to zero).
    sigma
        Common within-group standard deviation (> 0).
    n
        Per-group base sample size.  Combined with ``allocation`` (default
        equal) to derive per-group nᵢ via ``ceil(n · aᵢ)``.
        Required when ``solve_for='power'``.
    alpha
        Significance level (default 0.05).
    power
        Target power (used when ``solve_for='n'``).
    allocation
        Per-group allocation ratios (default ``[1,…,1]`` -> equal nᵢ).
    solve_for
        ``'power'`` or ``'n'``.  Defaults to ``'n'`` if ``power`` is given,
        else ``'power'``.
    """
    if len(means) < 2:
        raise ValueError("need at least 2 group means")
    _check_contrast(contrast, means)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if allocation is None:
        allocation = [1.0] * len(means)
    if len(allocation) != len(means):
        raise ValueError("allocation length must match means length")

    inputs_echo = {
        "means": means, "contrast": contrast, "sigma": sigma, "n": n,
        "alpha": alpha, "power": power, "allocation": allocation,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def ni_for(n_base: int) -> list[int]:
        return [max(2, math.ceil(n_base * a)) for a in allocation]

    if solve_for == "power":
        assert n is not None
        ni = ni_for(n)
        achieved = _power_contrast(means, contrast, ni, sigma, alpha)
        sigma_c = math.sqrt(_contrast_sigma_c_squared(means, contrast, ni))
        n_out = int(n)
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 2, 2
        # geometric bracket
        while hi <= 1_000_000:
            if _power_contrast(means, contrast, ni_for(hi), sigma, alpha) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket n within 1,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _power_contrast(means, contrast, ni_for(mid), sigma, alpha) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        ni = ni_for(hi)
        achieved = _power_contrast(means, contrast, ni, sigma, alpha)
        sigma_c = math.sqrt(_contrast_sigma_c_squared(means, contrast, ni))
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    contrast_value = sum(contrast[i] * means[i] for i in range(len(means)))
    effect_size = sigma_c / sigma
    return {
        "method_id": "one_way_anova_contrasts",
        "solve_for": solve_for,
        "n_per_group": n_out,
        "n_per_group_list": ni,
        "n_total": sum(ni),
        "achieved_power": achieved,
        "contrast_value": contrast_value,
        "sigma_c": sigma_c,
        "effect_size": effect_size,
        "inputs_echo": inputs_echo,
        "citations": [
            "Desu, M. M. & Raghavarao, D. (1990). Sample Size Methodology. "
            "Academic Press.",
            "Fleiss, J. (1986). The Design and Analysis of Clinical Experiments. "
            "Wiley.",
            "Kirk, R. E. (1982). Experimental Design: Procedures for the "
            "Behavioral Sciences. Brooks/Cole.",
        ],
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#
# Two-way fixed-effects ANOVA F-tests for main effects A, B and interaction AxB.
#
# Effect specification: each term's σ_m is the SD of its effects vector.
#   σ_m(A)  = sqrt( Σᵢ aᵢ² / a )       where aᵢ = μᵢ• - μ̄   (row marginal - grand mean)
#   σ_m(B)  = sqrt( Σⱼ bⱼ² / b )
#   σ_m(AB) = sqrt( Σᵢⱼ (ab)ᵢⱼ² / (ab) )
#
# Degrees of freedom:
#   df₁(A)  = a - 1,  df₂ = N - a*b = a*b*(n-1)
#   df₁(B)  = b - 1
#   df₁(AB) = (a-1)*(b-1)
#   N       = a * b * n
#
# NCP: λ = N · σ_m² / σ²


def _sigma_m_from_marginals(marginals: list[float]) -> float:
    """SD of a list of effects (mean already zero by construction)."""
    k = len(marginals)
    mu_bar = sum(marginals) / k
    return math.sqrt(sum((e - mu_bar) ** 2 for e in marginals) / k)


def _interaction_sigma_m(cell_means: list[list[float]]) -> float:
    """SD of the interaction effects from a 2-D table of cell means.

    cell_means[i][j] = μᵢⱼ  (row i, col j).
    σ_m(AB) = sqrt( Σᵢⱼ (μᵢⱼ - μᵢ• - μ•ⱼ + μ̄)² / (I*J) )
    """
    a = len(cell_means)
    b = len(cell_means[0])
    grand = sum(cell_means[i][j] for i in range(a) for j in range(b)) / (a * b)
    row_means = [sum(cell_means[i]) / b for i in range(a)]
    col_means = [sum(cell_means[i][j] for i in range(a)) / a for j in range(b)]
    sse = sum(
        (cell_means[i][j] - row_means[i] - col_means[j] + grand) ** 2
        for i in range(a) for j in range(b)
    )
    return math.sqrt(sse / (a * b))


def _power_two_way(sigma_m: float, sigma: float, n_total: int,
                   df1: int, df2: int, alpha: float) -> float:
    if df1 <= 0 or df2 <= 0 or sigma <= 0:
        return 0.0
    ncp = n_total * (sigma_m ** 2) / (sigma ** 2)
    from scipy.stats import f as fdist
    f_crit = fdist.ppf(1.0 - alpha, df1, df2)
    return float(1.0 - D.ncf_cdf(f_crit, df1, df2, ncp))


def factorial_anova_two_way(
    *,
    cell_means: list[list[float]],
    sigma: float,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    effect: str = "A",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-way fixed-effects ANOVA F-test solver.

    Tests one of: main effect A, main effect B, or AxB interaction.

    Parameters
    ----------
    cell_means
        2-D list of shape [a][b] giving cell means under H1.
        Rows = levels of factor A, columns = levels of factor B.
    sigma
        Common within-cell standard deviation (> 0).
    n
        Number of observations *per cell* (solve for power when given).
    alpha
        Significance level (default 0.05).
    power
        Target power (solve for n when given).
    effect
        Which term to test: ``'A'`` (default), ``'B'``, or ``'AxB'``.
    solve_for
        ``'power'`` or ``'n'``. Auto-detected from which of (n, power) is
        supplied when not given explicitly.
    """
    if not cell_means or not cell_means[0]:
        raise ValueError("cell_means must be a non-empty 2-D list")
    a = len(cell_means)
    b = len(cell_means[0])
    if a < 2 or b < 2:
        raise ValueError("each factor must have at least 2 levels")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if effect not in ("A", "B", "AxB"):
        raise ValueError("effect must be 'A', 'B', or 'AxB'")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    inputs_echo = {
        "cell_means": cell_means, "sigma": sigma, "n": n,
        "alpha": alpha, "power": power, "effect": effect,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    # Compute effect-specific σ_m and df1
    grand = sum(cell_means[i][j] for i in range(a) for j in range(b)) / (a * b)
    row_means = [sum(cell_means[i]) / b for i in range(a)]
    col_means = [sum(cell_means[i][j] for i in range(a)) / a for j in range(b)]

    if effect == "A":
        sigma_m = _sigma_m_from_marginals(row_means)
        df1 = a - 1
    elif effect == "B":
        sigma_m = _sigma_m_from_marginals(col_means)
        df1 = b - 1
    else:  # AxB
        sigma_m = _interaction_sigma_m(cell_means)
        df1 = (a - 1) * (b - 1)

    def n_total_from_n(n_per_cell):
        return a * b * n_per_cell

    def df2_from_n(n_per_cell):
        return a * b * (n_per_cell - 1)

    def power_at(n_per_cell):
        n_tot = n_total_from_n(n_per_cell)
        df2 = df2_from_n(n_per_cell)
        return _power_two_way(sigma_m, sigma, n_tot, df1, df2, alpha)

    if solve_for == "power":
        assert n is not None
        if n < 2:
            raise ValueError("n (per cell) must be >= 2")
        achieved = power_at(n)
        n_out = n
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 2, 2
        while hi <= 1_000_000:
            if power_at(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket n within 1,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if power_at(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        achieved = power_at(n_out)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    n_total = n_total_from_n(n_out)
    return {
        "method_id": "factorial_anova_two_way",
        "solve_for": solve_for,
        "n_per_cell": n_out,
        "n_total": n_total,
        "achieved_power": achieved,
        "effect": effect,
        "sigma_m": sigma_m,
        "effect_size_f": sigma_m / sigma,
        "df1": df1,
        "df2": df2_from_n(n_out),
        "inputs_echo": inputs_echo,
        "citations": [
            "Neter, J., Kutner, M., Nachtsheim, C., Wasserman, W. (1996). "
            "Applied Linear Statistical Models. Richard D. Irwin.",
            "Winer, B.J. (1991). Statistical Principles in Experimental Design, "
            "3rd Ed. McGraw-Hill.",
        ],
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#
# ANCOVA: within-group residual variance is reduced by the covariate(s).
#   σ_ε² = (1 - ρ²) σ²      where ρ² = R² of covariate(s) with Y
#   λ    = N · σ_m² / σ_ε²
#   df₁  = k - 1
#   df₂  = N - k - p          (p = number of covariates; default 1)
#
# Reference: Keppel (1991).


def ancova_one_way(
    *,
    means: list[float],
    sigma: float,
    r_squared: float,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n_covariates: int = 1,
    allocation: list[float] | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """One-way ANCOVA F-test solver.

    Reduces residual variance by a covariate whose R² with Y is *r_squared*.

        σ_ε² = (1 - r_squared) · σ²
        λ    = N · σ_m² / σ_ε²
        df₁  = k - 1
        df₂  = N - k - n_covariates

    Parameters
    ----------
    means
        Group means under H1 (length k ≥ 2).
    sigma
        Common within-group standard deviation *ignoring* the covariate.
    r_squared
        Multiple R² of the covariate(s) with the response within groups
        (0 ≤ r_squared < 1).
    n
        Per-group base sample size.
    alpha
        Significance level (default 0.05).
    power
        Target power (solve for n when given).
    n_covariates
        Number of covariates (default 1); adjusts df₂.
    allocation
        Per-group allocation ratios (default equal).
    solve_for
        ``'power'`` or ``'n'``.
    """
    if len(means) < 2:
        raise ValueError("need at least 2 group means")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if not 0.0 <= r_squared < 1.0:
        raise ValueError("r_squared must be in [0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_covariates < 1:
        raise ValueError("n_covariates must be >= 1")
    if allocation is None:
        allocation = [1.0] * len(means)
    if len(allocation) != len(means):
        raise ValueError("allocation length must match means length")

    inputs_echo = {
        "means": means, "sigma": sigma, "r_squared": r_squared, "n": n,
        "alpha": alpha, "power": power,
        "n_covariates": n_covariates, "allocation": allocation,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    k = len(means)
    sigma_e2 = (1.0 - r_squared) * (sigma ** 2)
    sigma_e = math.sqrt(sigma_e2)

    def ni_for(n_base: int) -> list[int]:
        return [max(2, math.ceil(n_base * a)) for a in allocation]

    def _power_ancova(n_base: int) -> float:
        ni = ni_for(n_base)
        n_total = sum(ni)
        df1 = k - 1
        df2 = n_total - k - n_covariates
        if df2 <= 0:
            return 0.0
        mu_bar = sum(ni[i] * means[i] for i in range(k)) / n_total
        sigma_m2 = sum(ni[i] * (means[i] - mu_bar) ** 2 for i in range(k)) / n_total
        ncp = n_total * sigma_m2 / sigma_e2
        from scipy.stats import f as fdist
        f_crit = fdist.ppf(1.0 - alpha, df1, df2)
        return float(1.0 - D.ncf_cdf(f_crit, df1, df2, ncp))

    if solve_for == "power":
        assert n is not None
        achieved = _power_ancova(n)
        ni = ni_for(n)
        n_out = n
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 2, 2
        while hi <= 1_000_000:
            if _power_ancova(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket n within 1,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _power_ancova(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        ni = ni_for(n_out)
        achieved = _power_ancova(n_out)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    n_total = sum(ni)
    mu_bar = sum(ni[i] * means[i] for i in range(k)) / n_total
    sigma_m2 = sum(ni[i] * (means[i] - mu_bar) ** 2 for i in range(k)) / n_total
    sigma_m = math.sqrt(sigma_m2)

    return {
        "method_id": "ancova_one_way",
        "solve_for": solve_for,
        "n_per_group": n_out,
        "n_per_group_list": ni,
        "n_total": n_total,
        "achieved_power": achieved,
        "sigma_m": sigma_m,
        "sigma_epsilon": sigma_e,
        "effect_size_f": sigma_m / sigma_e,
        "inputs_echo": inputs_echo,
        "citations": [
            "Keppel, G. (1991). Design and Analysis: A Researcher's Handbook, "
            "3rd Ed. Prentice-Hall.",
            "Borm, G.F., Fransen, J., Lemmens, W.A. (2007). 'A simple sample "
            "size formula for analysis of covariance in randomized clinical "
            "trials.' J Clinical Epidemiology 60, 1234-1238.",
        ],
    }
