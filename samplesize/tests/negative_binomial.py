"""Negative binomial two-sample rate ratio test.

Covers:
  negative_binomial_two_rates  — negative binomial two-sample rate ratio test
    Wald/LR test for H0: RR=1 vs Ha: RR≠1 (or one-sided)
    using Zhu & Lakkis (2014) formula with Method 3 null-variance estimation
    (maximum likelihood).

  Ch 438: Tests for the Ratio of Two Negative Binomial Rates
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm as _norm


def _z(p: float) -> float:
    return float(_norm.ppf(p))


def _norm_cdf(x: float) -> float:
    return float(_norm.cdf(x))


# ---------------------------------------------------------------------------
# Variance components (Zhu & Lakkis 2014)
# ---------------------------------------------------------------------------

def _va(lam1: float, lam2: float, R: float, mu_t: float, kappa: float) -> float:
    """V_A: variance component under the alternative hypothesis."""
    return (1.0 / mu_t) * (1.0 / lam1 + 1.0 / (R * lam2)) + (1.0 + R) * kappa / R


def _v0_method3(lam1: float, lam2: float, R: float, mu_t: float, kappa: float) -> float:
    """V_0|M3: null variance via MLE (weighted average rates), Method 3.

    V_0|3 = (1+R)^2 / (mu_t * R * (lam1 + R*lam2)) + (1+R)*kappa/R
    """
    return (1.0 + R) ** 2 / (mu_t * R * (lam1 + R * lam2)) + (1.0 + R) * kappa / R


# ---------------------------------------------------------------------------
# Power and sample size
# ---------------------------------------------------------------------------

def _nb_power(
    n1: int, lam1: float, lam2: float, R: float, mu_t: float,
    kappa: float, alpha: float, sides: int
) -> float:
    """Power of the NB Wald/LR test at N1 (Method 3 null variance)."""
    if n1 < 1:
        return 0.0
    va = _va(lam1, lam2, R, mu_t, kappa)
    v0 = _v0_method3(lam1, lam2, R, mu_t, kappa)
    log_rr = math.log(lam2 / lam1)
    if sides == 2:
        z_a = _z(1.0 - alpha / 2.0)
    else:
        z_a = _z(1.0 - alpha)
    return _norm_cdf(
        (math.sqrt(n1) * abs(log_rr) - z_a * math.sqrt(v0)) / math.sqrt(va)
    )


def _nb_n1_formula(
    lam1: float, lam2: float, R: float, mu_t: float,
    kappa: float, alpha: float, power: float, sides: int
) -> int:
    """Closed-form N1 from Zhu & Lakkis (2014) p.378 (Method 3)."""
    va = _va(lam1, lam2, R, mu_t, kappa)
    v0 = _v0_method3(lam1, lam2, R, mu_t, kappa)
    log_rr = math.log(lam2 / lam1)
    if sides == 2:
        z_a = _z(1.0 - alpha / 2.0)
    else:
        z_a = _z(1.0 - alpha)
    z_b = _z(power)
    n1_raw = (z_a * math.sqrt(v0) + z_b * math.sqrt(va)) ** 2 / log_rr ** 2
    return max(2, math.ceil(n1_raw))


def negative_binomial_two_rates(
    *,
    lam1: float,
    lam2: float,
    kappa: float,
    mu_t: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    null_var_method: int = 3,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Negative binomial two-sample rate ratio test.

    Tests H0: RR=1 vs Ha: RR≠1 (sides=2) or one-sided (sides=1)
    using the Wald/LR test and Zhu & Lakkis (2014) asymptotic formula.

    Parameters
    ----------
    lam1 : float
        Event rate per time unit in group 1 (control).
    lam2 : float
        Event rate per time unit in group 2 (treatment).
    kappa : float
        Negative binomial dispersion parameter (κ ≥ 0). κ=0 → Poisson.
    mu_t : float
        Average subject exposure time (same unit as lam1/lam2).
    alpha : float
        Type I error rate.
    power : float or None
        Target power (supply with lam1/lam2 to solve for n).
    n1 : int or None
        Group 1 sample size (supply with lam1/lam2 to solve for power).
    sides : int
        1 = one-sided, 2 = two-sided (default).
    allocation : float
        R = n2/n1 ratio (default 1.0 = equal groups).
    null_var_method : int
        1, 2, or 3.  3 = MLE (default, recommended by Zhu & Lakkis 2014).
    solve_for : str or None
        'n' or 'power'. Inferred if None.
    """
    inputs_echo = {
        "lam1": lam1, "lam2": lam2, "kappa": kappa, "mu_t": mu_t,
        "alpha": alpha, "power": power, "n1": n1,
        "sides": sides, "allocation": allocation,
        "null_var_method": null_var_method,
    }

    if lam1 <= 0 or lam2 <= 0:
        raise ValueError("lam1 and lam2 must be > 0")
    if lam1 == lam2:
        raise ValueError("lam1 and lam2 must differ")
    if kappa < 0:
        raise ValueError("kappa must be >= 0")
    if mu_t <= 0:
        raise ValueError("mu_t must be > 0")
    if null_var_method not in (1, 2, 3):
        raise ValueError("null_var_method must be 1, 2, or 3")
    if null_var_method != 3:
        raise NotImplementedError(
            "Only null_var_method=3 (MLE) is implemented. "
            "Methods 1 and 2 are not yet supported."
        )

    R = allocation  # n2/n1

    if solve_for is None:
        if n1 is None and power is not None:
            solve_for = "n"
        elif n1 is not None and power is None:
            solve_for = "power"
        else:
            raise ValueError("supply exactly one of (n1, power)")

    if solve_for == "n":
        assert power is not None
        # Closed-form seed
        n1_seed = _nb_n1_formula(lam1, lam2, R, mu_t, kappa, alpha, power, sides)
        # Walk forward to ensure discrete ceiling is correct
        n_try = max(2, n1_seed - 2)
        while True:
            achieved = _nb_power(n_try, lam1, lam2, R, mu_t, kappa, alpha, sides)
            if achieved >= power:
                break
            n_try += 1
            if n_try > 10_000_000:
                raise RuntimeError("failed to bracket N1")
        n1_out = n_try
        n2_out = max(2, math.ceil(R * n1_out))
        achieved_out = achieved
    else:  # power
        assert n1 is not None
        n1_out = n1
        n2_out = max(2, math.ceil(R * n1))
        achieved_out = _nb_power(n1, lam1, lam2, R, mu_t, kappa, alpha, sides)

    return {
        "method_id": "negative_binomial_two_rates",
        "solve_for": solve_for,
        "n1": n1_out,
        "n2": n2_out,
        "n": n1_out + n2_out,
        "achieved_power": achieved_out,
        "inputs_echo": inputs_echo,
        "citations": [
            "Zhu, H. and Lakkis, H. (2014). Sample Size Calculation for Comparing "
            "Two Negative Binomial Rates. Statistics in Medicine, 33, 376-387.",
        ],
    }
