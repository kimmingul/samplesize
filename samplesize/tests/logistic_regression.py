"""Simple logistic regression power & sample-size calculator.


Implements power calculations for testing H0: β1 = 0 vs H1: β1 ≠ 0
in a simple logistic regression model using the Hsieh, Block & Larsen
(1998) approximations.

Two covariate types are supported:

1. **Normal (continuous) covariate** – Hsieh et al. (1998) formula:

       N = (z_{1-α/2} + z_{1-β})² / [P*(1-P*) · B²]

   where P* is the event probability at the mean of X, and B is the
   logistic regression slope (one-unit or one-SD increment depending
   on parameterisation).  For power calculations the formula is
   inverted using the standard normal approximation.

   The slope B is recovered from (P0, OR) as:
       B = ln(OR)   [since OR = exp(B · 1 SD)]

2. **Binary (binomial) covariate** – Hsieh et al. (1998) formula:

       N = [z_{1-α/2}·√(P̄(1-P̄)/R) + z_{1-β}·√(P0(1-P0) + P1(1-P1)(1-R)/R)]²
           / [(P0-P1)²(1-R)]

   where P0 = Pr(Y=1|X=0), P1 = Pr(Y=1|X=1), R = Pr(X=1),
   P̄ = (1-R)·P0 + R·P1.

For multiple covariates the adjustment N_m = N / (1-ρ²) is applied,
where ρ² is the R-squared of X1 on the remaining covariates.

References
----------
* Hsieh, F.Y., Block, D.A., and Larsen, M.D. (1998). A Simple Method
  of Sample Size Calculation for Linear and Logistic Regression.
  Statistics in Medicine, 17(4):1623-1634.
* Whittemore, A.S. (1981). Sample Size for Logistic Regression with
  Small Response Probability. J. Am. Stat. Assoc. 76(373):27-32.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _p1_from_or(p0: float, odds_ratio: float) -> float:
    """Compute P1 given P0 and the odds ratio OR = (P1/(1-P1))/(P0/(1-P0))."""
    odds0 = p0 / (1.0 - p0)
    odds1 = odds_ratio * odds0
    return odds1 / (1.0 + odds1)


def _or_from_p0_p1(p0: float, p1: float) -> float:
    """Compute odds ratio from P0 and P1."""
    return (p1 / (1.0 - p1)) / (p0 / (1.0 - p0))


def _power_normal_covariate(
    *,
    n: int,
    p0: float,
    odds_ratio: float,
    alpha: float,
    sides: int,
    r_squared: float,
) -> float:
    """Power for continuous (normal) covariate using Hsieh (1998)."""
    if n <= 0:
        return 0.0
    z_alpha = D.norm_ppf(1.0 - alpha / sides)
    b_squared = math.log(odds_ratio) ** 2
    p_star = p0  # event rate at mean of X
    # From N = (z_alpha + z_beta)^2 / [P*(1-P*) * B^2 * (1-rho^2)]
    # => (z_alpha + z_beta)^2 = N * P*(1-P*) * B^2 * (1-rho^2)
    # => z_beta = sqrt(N * P*(1-P*) * B^2 * (1-rho^2)) - z_alpha
    variance_term = p_star * (1.0 - p_star) * b_squared * (1.0 - r_squared)
    if variance_term <= 0:
        return 0.0
    z_beta = math.sqrt(n * variance_term) - z_alpha
    return float(D.norm_ppf.__self__.cdf(z_beta)) if hasattr(D.norm_ppf, '__self__') else _norm_cdf(z_beta)


def _norm_cdf(x: float) -> float:
    from scipy.stats import norm
    return float(norm.cdf(x))


def _power_normal_cov(
    *,
    n: int,
    p0: float,
    odds_ratio: float,
    alpha: float,
    sides: int,
    r_squared: float,
) -> float:
    """Power for continuous covariate (Hsieh 1998)."""
    if n <= 0:
        return 0.0
    z_alpha = D.norm_ppf(1.0 - alpha / sides)
    b_squared = math.log(odds_ratio) ** 2
    p_star = p0
    variance_term = p_star * (1.0 - p_star) * b_squared * (1.0 - r_squared)
    if variance_term <= 0:
        return 0.0
    # z_beta = sqrt(N * var_term) - z_alpha
    # power = Phi(z_beta)
    z_beta = math.sqrt(n * variance_term) - z_alpha
    return _norm_cdf(z_beta)


def _n_normal_cov(
    *,
    power: float,
    p0: float,
    odds_ratio: float,
    alpha: float,
    sides: int,
    r_squared: float,
) -> int:
    """Sample size for continuous covariate (Hsieh 1998)."""
    z_alpha = D.norm_ppf(1.0 - alpha / sides)
    z_beta = D.norm_ppf(power)
    b_squared = math.log(odds_ratio) ** 2
    p_star = p0
    variance_term = p_star * (1.0 - p_star) * b_squared * (1.0 - r_squared)
    if variance_term <= 0:
        raise ValueError("variance_term is zero; check p0 and odds_ratio")
    n_raw = ((z_alpha + z_beta) ** 2) / variance_term
    return math.ceil(n_raw)


def _power_binary_cov(
    *,
    n: int,
    p0: float,
    p1: float,
    r: float,
    alpha: float,
    sides: int,
    r_squared: float,
) -> float:
    """Power for binary covariate (Hsieh 1998)."""
    if n <= 0:
        return 0.0
    p_bar = (1.0 - r) * p0 + r * p1
    denom = (p0 - p1) ** 2 * (1.0 - r)
    if denom == 0:
        return 0.0
    # Invert: N * denom = [z_alpha * sqrt(P_bar*(1-P_bar)/r) + z_beta * sqrt(...)]^2
    # => sqrt(N * denom) = z_alpha * A + z_beta * B  → solve for power
    z_alpha = D.norm_ppf(1.0 - alpha / sides)
    var_null = p_bar * (1.0 - p_bar) / r
    var_alt = p0 * (1.0 - p0) + p1 * (1.0 - p1) * (1.0 - r) / r
    A = math.sqrt(var_null) if var_null > 0 else 0.0
    B = math.sqrt(var_alt) if var_alt > 0 else 0.0
    if B == 0:
        return 0.0
    # sqrt(N * denom) = z_alpha * A + z_beta * B
    # z_beta = (sqrt(N * denom) - z_alpha * A) / B
    z_beta = (math.sqrt(n * denom) - z_alpha * A) / B
    # Apply r_squared adjustment: effective N_eff = N * (1-rho^2)
    # Re-derive with adjusted N
    n_eff = n * (1.0 - r_squared)
    z_beta = (math.sqrt(n_eff * denom) - z_alpha * A) / B
    return _norm_cdf(z_beta)


def _n_binary_cov(
    *,
    power: float,
    p0: float,
    p1: float,
    r: float,
    alpha: float,
    sides: int,
    r_squared: float,
) -> int:
    """Sample size for binary covariate (Hsieh 1998)."""
    p_bar = (1.0 - r) * p0 + r * p1
    denom = (p0 - p1) ** 2 * (1.0 - r)
    if denom == 0:
        raise ValueError("p0 must differ from p1 for binary covariate")
    z_alpha = D.norm_ppf(1.0 - alpha / sides)
    z_beta = D.norm_ppf(power)
    var_null = p_bar * (1.0 - p_bar) / r
    var_alt = p0 * (1.0 - p0) + p1 * (1.0 - p1) * (1.0 - r) / r
    numerator = (z_alpha * math.sqrt(var_null) + z_beta * math.sqrt(var_alt)) ** 2
    n_raw = numerator / (denom * (1.0 - r_squared))
    return math.ceil(n_raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def logistic_regression_simple(
    *,
    p0: float,
    odds_ratio: float | None = None,
    p1: float | None = None,
    covariate_type: str = "normal",
    r_binary: float = 0.5,
    r_squared: float = 0.0,
    alpha: float = 0.05,
    sides: int = 2,
    n: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power and sample size for simple logistic regression (Hsieh 1998).

    Tests H0: β1 = 0 vs H1: β1 ≠ 0 (or one-sided) in a logistic regression
    of a binary Y on a single covariate X.

    Parameters
    ----------
    p0
        Baseline event probability Pr(Y=1).
        - Normal covariate: probability at the mean of X.
        - Binary covariate: probability when X=0.
    odds_ratio
        Odds ratio to detect.  Exactly one of (odds_ratio, p1) must be given.
    p1
        Alternative event probability.
        - Normal covariate: probability when X is 1 SD above mean.
        - Binary covariate: probability when X=1.
        Converted to an odds ratio internally.
    covariate_type
        ``"normal"`` (continuous) or ``"binary"`` (binomial X=0/1).
    r_binary
        Proportion of sample with X=1 when covariate_type=="binary".  Default 0.5.
    r_squared
        R² of X1 on the other covariates (for multiple logistic adjustment).
        Use 0.0 (default) for simple logistic regression.
    alpha
        Type-I error rate.  Default 0.05.
    sides
        1 or 2 (default 2).
    n
        Total sample size (provide when solve_for="power").
    power
        Target power (provide when solve_for="n").
    solve_for
        ``"n"`` or ``"power"``.  Inferred if only one of (n, power) is given.
    """
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if r_squared < 0.0 or r_squared >= 1.0:
        raise ValueError("r_squared must be in [0, 1)")

    # Resolve odds_ratio / p1
    if odds_ratio is None and p1 is None:
        raise ValueError("supply exactly one of (odds_ratio, p1)")
    if odds_ratio is not None and p1 is not None:
        raise ValueError("supply exactly one of (odds_ratio, p1), not both")
    if p1 is not None:
        if not 0.0 < p1 < 1.0:
            raise ValueError("p1 must be in (0, 1)")
        odds_ratio = _or_from_p0_p1(p0, p1)
    else:
        p1 = _p1_from_or(p0, odds_ratio)  # type: ignore[arg-type]

    if odds_ratio <= 0:
        raise ValueError("odds_ratio must be positive")
    if odds_ratio == 1.0:
        raise ValueError("odds_ratio must differ from 1.0 to have detectable effect")

    ctype = covariate_type.strip().lower()
    if ctype not in ("normal", "binary", "continuous", "binomial"):
        raise ValueError("covariate_type must be 'normal' or 'binary'")
    is_binary = ctype in ("binary", "binomial")

    if is_binary:
        if not 0.0 < r_binary < 1.0:
            raise ValueError("r_binary must be in (0, 1)")

    inputs_echo: dict[str, Any] = {
        "p0": p0, "odds_ratio": odds_ratio, "p1": p1,
        "covariate_type": covariate_type, "r_binary": r_binary,
        "r_squared": r_squared, "alpha": alpha, "sides": sides,
        "n": n, "power": power,
    }

    # Determine solve_for
    given = sum(x is not None for x in (n, power))
    if given == 0:
        raise ValueError("supply exactly one of (n, power)")
    if given == 2 and solve_for is None:
        raise ValueError("both n and power given; specify solve_for explicitly")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        if is_binary:
            achieved = _power_binary_cov(
                n=n, p0=p0, p1=p1, r=r_binary,
                alpha=alpha, sides=sides, r_squared=r_squared,
            )
        else:
            achieved = _power_normal_cov(
                n=n, p0=p0, odds_ratio=odds_ratio,
                alpha=alpha, sides=sides, r_squared=r_squared,
            )
        n_out = n

    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        if is_binary:
            n_out = _n_binary_cov(
                power=power, p0=p0, p1=p1, r=r_binary,
                alpha=alpha, sides=sides, r_squared=r_squared,
            )
            achieved = _power_binary_cov(
                n=n_out, p0=p0, p1=p1, r=r_binary,
                alpha=alpha, sides=sides, r_squared=r_squared,
            )
        else:
            n_out = _n_normal_cov(
                power=power, p0=p0, odds_ratio=odds_ratio,
                alpha=alpha, sides=sides, r_squared=r_squared,
            )
            achieved = _power_normal_cov(
                n=n_out, p0=p0, odds_ratio=odds_ratio,
                alpha=alpha, sides=sides, r_squared=r_squared,
            )
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "logistic_regression_simple",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "p1": p1,
        "odds_ratio": odds_ratio,
        "inputs_echo": inputs_echo,
        "citations": [
            "Hsieh, F.Y., Block, D.A., and Larsen, M.D. (1998). A Simple "
            "Method of Sample Size Calculation for Linear and Logistic "
            "Regression. Statistics in Medicine, 17(4):1623-1634.",
            "Whittemore, A.S. (1981). Sample Size for Logistic Regression "
            "with Small Response Probability. J. Am. Stat. Assoc. "
            "76(373):27-32.",
        ],
    }
