"""Repeated-measures and mixed ANOVA power & sample-size.

Implements:
  * anova_repeated_measures_within  — one-way within-subject F-test
    under sphericity (or Greenhouse-Geisser ε correction).
    One-Way Repeated Measures.

  * anova_mixed_between_within  — split-plot / mixed ANOVA:
    between-group F-test and within-subject F-test for groups × time.
    Repeated Measures Analysis.

Both methods use the compound-symmetry / univariate F approximation.

Within-subject F-test formulas (compound symmetry, Geisser-Greenhouse ε)
----------------------------------------------------------------------
  σ_m²  = Σ(μᵢ - μ̄)² / M          (variance of M condition means)
  σ_e²  = σ_w² · (1 − ρ)           (within-subject residual variance)
  λ_0   = N · M · σ_m² / σ_e²      (NCP without ε)
  df₁   = (M − 1) · ε
  df₂   = (N − 1) · (M − 1) · ε
  λ     = λ_0 · ε

Between-group F-test formula (split-plot / mixed design)
---------------------------------------------------------
  Groups define k levels with n subjects per group (N = k·n).
  σ_m_B²  = Σᵢ nᵢ·(μ̄ᵢ − μ̄)² / N  (weighted between-group variance
                                     of group-marginal means)
  σ_B²    = σ_w² · (1 + (M−1)·ρ)   (between-subject error variance)
  λ_B     = N · σ_m_B² / σ_B²
  df₁_B   = k − 1
  df₂_B   = N − k

References
----------
* Muller, K.E. & Barton, C.N. (1989). JASA 84, 549-555.
* Muller, K.E., LaVange, L.E., Ramey, S.L., Ramey, C.T. (1992). JASA 87, 1209-1226.
* Maxwell, S.E. & Delaney, H.D. (2003). Designing Experiments and Analyzing Data.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sigma_m_squared(means: list[float]) -> float:
    """Population variance of a list of means (divisor = len(means))."""
    k = len(means)
    mu_bar = sum(means) / k
    return sum((m - mu_bar) ** 2 for m in means) / k


def _power_within_f(
    means: list[float],
    sigma_w: float,
    rho: float,
    epsilon: float,
    n: int,
    alpha: float,
) -> tuple[float, float, float, float]:
    """Power, df1, df2, ncp for the within-subject F-test."""
    m = len(means)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    df1 = (m - 1) * epsilon
    df2 = (n - 1) * (m - 1) * epsilon
    if df1 <= 0 or df2 <= 0:
        return 0.0, df1, df2, 0.0
    sigma_m2 = _sigma_m_squared(means)
    sigma_e2 = sigma_w ** 2 * (1.0 - rho)
    if sigma_e2 <= 0:
        return 1.0, df1, df2, float("inf")
    ncp_base = n * m * sigma_m2 / sigma_e2
    ncp = ncp_base * epsilon
    from scipy.stats import f as fdist
    f_crit = fdist.ppf(1.0 - alpha, df1, df2)
    power = float(1.0 - D.ncf_cdf(f_crit, df1, df2, ncp))
    return power, df1, df2, ncp


# ---------------------------------------------------------------------------
# Method 2: anova_repeated_measures_within
# ---------------------------------------------------------------------------


def anova_repeated_measures_within(
    *,
    means: list[float],
    sigma_w: float,
    rho: float = 0.0,
    epsilon: float = 1.0,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """One-way within-subject (repeated measures) F-test.

    Tests equality of M repeated-measures condition means using the
    univariate F-test under compound symmetry, optionally with the
    Greenhouse-Geisser ε correction.

    Parameters
    ----------
    means
        List of M (≥ 2) condition means under H1.
    sigma_w
        Between-subject standard deviation at a single time point
        (assumed equal across time).
    rho
        Compound-symmetry correlation between repeated measurements
        (0 ≤ ρ < 1; default 0).
    epsilon
        Greenhouse-Geisser sphericity correction
        (1/(M-1) ≤ ε ≤ 1; default 1 → no correction).
        # Greenhouse-Geisser correction not yet wired for automatic
        # computation; supply the known ε directly.
    n
        Number of subjects (for solve_for='power').
    alpha
        Significance level (default 0.05).
    power
        Target power (for solve_for='n').
    solve_for
        ``'power'`` or ``'n'``.
    """
    if len(means) < 2:
        raise ValueError("need at least 2 condition means")
    if sigma_w <= 0:
        raise ValueError("sigma_w must be > 0")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    m = len(means)
    eps_min = 1.0 / (m - 1) if m > 2 else 1.0
    if not (eps_min - 1e-9 <= epsilon <= 1.0 + 1e-9):
        raise ValueError(f"epsilon must be in [{eps_min:.4f}, 1.0]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    inputs_echo = {
        "means": means, "sigma_w": sigma_w, "rho": rho,
        "epsilon": epsilon, "n": n, "alpha": alpha, "power": power,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def pwr(n_val):
        p, _, _, _ = _power_within_f(means, sigma_w, rho, epsilon, n_val, alpha)
        return p

    if solve_for == "power":
        assert n is not None
        pw, df1, df2, ncp = _power_within_f(means, sigma_w, rho, epsilon, n, alpha)
        achieved = pw
        n_out = n
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 2, 2
        while hi <= 1_000_000:
            if pwr(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 1,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pwr(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        _, df1, df2, ncp = _power_within_f(means, sigma_w, rho, epsilon, n_out, alpha)
        achieved = pwr(n_out)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    sigma_m = math.sqrt(_sigma_m_squared(means))
    sigma_e = math.sqrt(sigma_w ** 2 * (1.0 - rho))

    return {
        "method_id": "anova_repeated_measures_within",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "sigma_m": sigma_m,
        "effect_f": sigma_m / sigma_e if sigma_e > 0 else float("inf"),
        "df1": df1,
        "df2": df2,
        "ncp": ncp,
        "inputs_echo": inputs_echo,
        "citations": [
            "Muller, K.E. & Barton, C.N. (1989). 'Approximate Power for "
            "Repeated-Measures ANOVA Lacking Sphericity.' JASA 84, 549-555.",
            "Maxwell, S.E. & Delaney, H.D. (2003). Designing Experiments "
            "and Analyzing Data, 2nd Ed. Psychology Press.",
        ],
    }


# ---------------------------------------------------------------------------
# Method 3: anova_mixed_between_within
# ---------------------------------------------------------------------------


def anova_mixed_between_within(
    *,
    group_means: list[list[float]],
    sigma_w: float,
    rho: float = 0.0,
    epsilon: float = 1.0,
    n_per_group: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    test: str = "within",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Mixed ANOVA (split-plot) power calculator.

    Tests either the between-group main effect or the within-subject
    main effect in a k-group × M-measurement mixed design.

    Parameters
    ----------
    group_means
        2-D list of shape [k][M]: group_means[i][j] = mean for group i
        at time/condition j.
    sigma_w
        Between-subject standard deviation (single time point, equal
        across groups and times under compound symmetry).
    rho
        Compound-symmetry within-subject correlation (0 ≤ ρ < 1;
        default 0).
    epsilon
        Greenhouse-Geisser ε (applied to the within-subject test only;
        default 1 → no correction).
        # Greenhouse-Geisser correction not yet wired for automatic
        # computation; supply the known ε directly.
    n_per_group
        Number of subjects per group.
    alpha
        Significance level (default 0.05).
    power
        Target power.
    test
        Which effect to power: ``'within'`` (default) — within-subject
        main effect (time/condition); ``'between'`` — between-group main
        effect.
    solve_for
        ``'power'`` or ``'n'``.
    """
    if not group_means or not group_means[0]:
        raise ValueError("group_means must be a non-empty 2-D list")
    k = len(group_means)
    m = len(group_means[0])
    if k < 2:
        raise ValueError("need at least 2 groups (k >= 2)")
    if m < 2:
        raise ValueError("need at least 2 repeated measurements (M >= 2)")
    if sigma_w <= 0:
        raise ValueError("sigma_w must be > 0")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    eps_min = 1.0 / (m - 1) if m > 2 else 1.0
    if not (eps_min - 1e-9 <= epsilon <= 1.0 + 1e-9):
        raise ValueError(f"epsilon must be in [{eps_min:.4f}, 1.0]")
    if test not in ("within", "between"):
        raise ValueError("test must be 'within' or 'between'")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    inputs_echo = {
        "group_means": group_means, "sigma_w": sigma_w, "rho": rho,
        "epsilon": epsilon, "n_per_group": n_per_group, "alpha": alpha,
        "power": power, "test": test,
    }
    have_n = n_per_group is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n_per_group, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    # Marginal means for each test
    # Between-group: average each group's means over time → shape [k]
    group_marginal = [sum(group_means[i]) / m for i in range(k)]
    # Within-subject: average each time point over groups → shape [m]
    time_marginal = [sum(group_means[i][j] for i in range(k)) / k for j in range(m)]

    def _power_between(n_g: int) -> float:
        """Between-group F-test power."""
        n_total = k * n_g
        df1 = k - 1
        df2 = n_total - k
        if df2 <= 0:
            return 0.0
        mu_bar = sum(group_marginal) / k
        sigma_m2 = sum((gm - mu_bar) ** 2 for gm in group_marginal) / k
        # Between-subject error variance under compound symmetry
        sigma_b2 = sigma_w ** 2 * (1.0 + (m - 1) * rho)
        if sigma_b2 <= 0:
            return 1.0
        ncp = n_total * sigma_m2 / sigma_b2
        from scipy.stats import f as fdist
        f_crit = fdist.ppf(1.0 - alpha, df1, df2)
        return float(1.0 - D.ncf_cdf(f_crit, df1, df2, ncp))

    def _power_within(n_g: int) -> float:
        """Within-subject F-test power (pooled across groups)."""
        n_total = k * n_g
        pw, _, _, _ = _power_within_f(
            time_marginal, sigma_w, rho, epsilon, n_total, alpha
        )
        return pw

    pwr_fn = _power_between if test == "between" else _power_within

    if solve_for == "power":
        assert n_per_group is not None
        achieved = pwr_fn(n_per_group)
        n_out = n_per_group
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 2, 2
        while hi <= 1_000_000:
            if pwr_fn(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket n within 1,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pwr_fn(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        achieved = pwr_fn(n_out)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    # Compute descriptive statistics for the chosen test
    if test == "between":
        mu_bar = sum(group_marginal) / k
        sigma_m = math.sqrt(sum((gm - mu_bar) ** 2 for gm in group_marginal) / k)
        sigma_err = math.sqrt(sigma_w ** 2 * (1.0 + (m - 1) * rho))
        df1_out = k - 1
        df2_out = k * n_out - k
    else:
        sigma_m = math.sqrt(_sigma_m_squared(time_marginal))
        sigma_e2 = sigma_w ** 2 * (1.0 - rho)
        sigma_err = math.sqrt(sigma_e2) if sigma_e2 > 0 else 0.0
        df1_out = (m - 1) * epsilon
        df2_out = (k * n_out - 1) * (m - 1) * epsilon

    return {
        "method_id": "anova_mixed_between_within",
        "solve_for": solve_for,
        "n_per_group": n_out,
        "n_total": k * n_out,
        "achieved_power": achieved,
        "test": test,
        "sigma_m": sigma_m,
        "sigma_error": sigma_err,
        "effect_f": sigma_m / sigma_err if sigma_err > 0 else float("inf"),
        "df1": df1_out,
        "df2": df2_out,
        "inputs_echo": inputs_echo,
        "citations": [
            "Muller, K.E., LaVange, L.E., Ramey, S.L., Ramey, C.T. (1992). "
            "'Power Calculations for General Linear Multivariate Models "
            "Including Repeated Measures Applications.' JASA 87, 1209-1226.",
            "Maxwell, S.E. & Delaney, H.D. (2003). Designing Experiments "
            "and Analyzing Data, 2nd Ed. Psychology Press.",
        ],
    }
