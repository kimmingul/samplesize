"""One-sample tests for a mean.

One-sample mean tests:
- One-sample / paired t-test (SD unknown, default)
- z-test branch when SD is known
- Optional Wilcoxon adjustment for nonparametric alternatives

Power formulae (SD unknown, df = n - 1, λ = (mean1 - mean0)/(σ/√n)):

  one-tailed Ha: mean1 > mean0:
      power = 1 - T'(t_α; df, λ)

  one-tailed Ha: mean1 < mean0:
      power = T'(-t_α; df, -λ)

  two-tailed Ha: mean1 ≠ mean0:
      power = [1 - T'(t_{α/2}; df, λ)] + T'(-t_{α/2}; df, λ)

For SD known the same formulae apply with the standard normal in place
of the central-t and the standard normal in place of the noncentral-t
(λ unchanged).
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D
from samplesize.core import effect_sizes as E

# Wilcoxon adjustment factors per Al-Sunduqchi & Guenther (1990).
WILCOXON_FACTORS = {
    "ignore":            1.0,
    "uniform":           1.0,
    "double_exponential": 2.0 / 3.0,
    "logistic":          9.0 / math.pi ** 2,
    "normal":            math.pi / 3.0,
}

# ---- power given fixed N ---------------------------------------------------


def _power_t(mean0: float, mean1: float, sd: float, n: int,
             alpha: float, sides: int) -> float:
    """Noncentral-t power for the one-sample t-test."""
    if n < 2:
        return 0.0
    df = n - 1
    se = sd / math.sqrt(n)
    ncp = (mean1 - mean0) / se
    if sides == 2:
        t_crit = D.t_ppf(1 - alpha / 2.0, df)
        upper = 1.0 - D.nct_cdf(t_crit, df, ncp)
        lower = D.nct_cdf(-t_crit, df, ncp)
        return upper + lower
    elif sides == 1:
        t_crit = D.t_ppf(1 - alpha, df)
        if mean1 >= mean0:
            return 1.0 - D.nct_cdf(t_crit, df, ncp)
        else:
            return D.nct_cdf(-t_crit, df, ncp)
    else:
        raise ValueError(f"sides must be 1 or 2, got {sides}")


def _power_z(mean0: float, mean1: float, sd: float, n: int,
             alpha: float, sides: int) -> float:
    """Normal power for the one-sample z-test (SD known)."""
    if n < 1:
        return 0.0
    from scipy.stats import norm
    se = sd / math.sqrt(n)
    ncp = (mean1 - mean0) / se
    if sides == 2:
        z_crit = D.norm_ppf(1 - alpha / 2.0)
        upper = 1.0 - norm.cdf(z_crit - ncp)
        lower = norm.cdf(-z_crit - ncp)
        return float(upper + lower)
    elif sides == 1:
        z_crit = D.norm_ppf(1 - alpha)
        if mean1 >= mean0:
            return float(1.0 - norm.cdf(z_crit - ncp))
        else:
            return float(norm.cdf(-z_crit - ncp))
    else:
        raise ValueError(f"sides must be 1 or 2, got {sides}")


def power_at_n(*, mean0: float, mean1: float, sd: float, n: int,
               alpha: float, sides: int = 2,
               sd_known: bool = False,
               nonparametric: str = "ignore") -> float:
    """Compute power for a fixed sample size.

    `nonparametric` applies a Wilcoxon adjustment by scaling the effective
    sample size before invoking the t/z formula.
    """
    factor = WILCOXON_FACTORS[nonparametric]
    n_eff = max(2 if not sd_known else 1, int(round(n * factor)))
    if sd_known:
        return _power_z(mean0, mean1, sd, n_eff, alpha, sides)
    return _power_t(mean0, mean1, sd, n_eff, alpha, sides)


# ---- solve for N -----------------------------------------------------------


def n_for_power(*, mean0: float, mean1: float, sd: float, alpha: float,
                power: float, sides: int = 2, sd_known: bool = False,
                nonparametric: str = "ignore",
                n_min: int = 2, n_max: int = 1_000_000) -> tuple[int, float]:
    """Smallest N achieving ≥ `power`.  Returns (n, achieved_power)."""
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if mean0 == mean1:
        raise ValueError("mean0 and mean1 must differ to solve for N")

    # Bracket then bisect.
    lo, hi = n_min, n_min
    while hi <= n_max:
        p = power_at_n(mean0=mean0, mean1=mean1, sd=sd, n=hi, alpha=alpha,
                       sides=sides, sd_known=sd_known,
                       nonparametric=nonparametric)
        if p >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        p = power_at_n(mean0=mean0, mean1=mean1, sd=sd, n=mid, alpha=alpha,
                       sides=sides, sd_known=sd_known,
                       nonparametric=nonparametric)
        if p >= power:
            hi = mid
        else:
            lo = mid

    achieved = power_at_n(mean0=mean0, mean1=mean1, sd=sd, n=hi, alpha=alpha,
                          sides=sides, sd_known=sd_known,
                          nonparametric=nonparametric)
    return hi, achieved


# ---- solve for minimum detectable mean1 ------------------------------------


def effect_for_power(*, mean0: float, sd: float, n: int, alpha: float,
                     power: float, sides: int = 2, sd_known: bool = False,
                     nonparametric: str = "ignore",
                     direction: str = "either",
                     tol: float = 1e-6) -> float:
    """Minimum detectable mean1 (closer of the two roots for two-sided).

    `direction`:
      - "above": mean1 > mean0
      - "below": mean1 < mean0
      - "either": choose the smaller |mean1 - mean0|; for two-sided this
        is symmetric (mean1 = mean0 + Δ for some Δ > 0).
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")

    sign = +1.0 if direction in ("above", "either") else -1.0
    delta_lo, delta_hi = 0.0, max(sd, 1.0)
    # Expand until power at delta_hi exceeds target.
    for _ in range(60):
        m1 = mean0 + sign * delta_hi
        p_hi = power_at_n(mean0=mean0, mean1=m1, sd=sd, n=n, alpha=alpha,
                          sides=sides, sd_known=sd_known,
                          nonparametric=nonparametric)
        if p_hi >= power:
            break
        delta_hi *= 2.0
    else:
        raise RuntimeError("failed to bracket detectable effect")

    # Bisect.
    for _ in range(200):
        mid = 0.5 * (delta_lo + delta_hi)
        m1 = mean0 + sign * mid
        p = power_at_n(mean0=mean0, mean1=m1, sd=sd, n=n, alpha=alpha,
                       sides=sides, sd_known=sd_known,
                       nonparametric=nonparametric)
        if p >= power:
            delta_hi = mid
        else:
            delta_lo = mid
        if delta_hi - delta_lo < tol:
            break
    return mean0 + sign * delta_hi


# ---- top-level entry point used by the registry ---------------------------


def one_sample_t(
    *,
    mean0: float,
    mean1: float | None = None,
    sd: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    sd_known: bool = False,
    nonparametric: str = "ignore",
    solve_for: str | None = None,
    direction: str = "above",
) -> dict[str, Any]:
    """Single-mean test power / sample-size / detectable-effect solver.

    Provide exactly two of (`mean1`, `power`, `n`); the missing one is
    solved for.  `solve_for` overrides auto-detection if both are
    supplied.
    """
    nonparametric = nonparametric.lower()
    if nonparametric not in WILCOXON_FACTORS:
        raise ValueError(
            f"nonparametric must be one of {sorted(WILCOXON_FACTORS)}: "
            f"got {nonparametric!r}"
        )

    inputs_echo = {
        "mean0": mean0, "mean1": mean1, "sd": sd, "alpha": alpha,
        "power": power, "n": n, "sides": sides, "sd_known": sd_known,
        "nonparametric": nonparametric,
    }

    given = sum(x is not None for x in (mean1, power, n))
    if given < 2:
        raise ValueError(
            "supply exactly two of (mean1, power, n); leave the third None"
        )

    if solve_for is None:
        if n is None:
            solve_for = "n"
        elif power is None:
            solve_for = "power"
        elif mean1 is None:
            solve_for = "effect"
        else:
            raise ValueError(
                "all three of (mean1, power, n) provided; set one to None "
                "or pass solve_for explicitly"
            )

    if solve_for == "power":
        assert mean1 is not None and n is not None
        achieved = power_at_n(
            mean0=mean0, mean1=mean1, sd=sd, n=n, alpha=alpha,
            sides=sides, sd_known=sd_known, nonparametric=nonparametric,
        )
        effect = E.cohens_d(mean1, mean0, sd)
        result = {"n": n, "achieved_power": achieved, "effect_d": effect}

    elif solve_for == "n":
        assert mean1 is not None and power is not None
        n_req, achieved = n_for_power(
            mean0=mean0, mean1=mean1, sd=sd, alpha=alpha, power=power,
            sides=sides, sd_known=sd_known, nonparametric=nonparametric,
        )
        effect = E.cohens_d(mean1, mean0, sd)
        result = {"n": n_req, "achieved_power": achieved, "effect_d": effect}

    elif solve_for == "effect":
        assert n is not None and power is not None
        m1 = effect_for_power(
            mean0=mean0, sd=sd, n=n, alpha=alpha, power=power,
            sides=sides, sd_known=sd_known, nonparametric=nonparametric,
            direction=direction,
        )
        achieved = power_at_n(
            mean0=mean0, mean1=m1, sd=sd, n=n, alpha=alpha,
            sides=sides, sd_known=sd_known, nonparametric=nonparametric,
        )
        result = {
            "n": n,
            "mean1": m1,
            "achieved_power": achieved,
            "effect_d": E.cohens_d(m1, mean0, sd),
        }

    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "one_sample_t",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.",
        ],
    }
