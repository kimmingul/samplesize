"""Poisson regression power and sample size — Signorini (1991).


Tests H0: β1=0 vs H1: β1=B1 using the Signorini (1991) score-statistic
formula extended by Hsieh, Block & Larsen (1998) for multiple covariates.

The sample size formula is:

    N = φ · (z_{1-α/k}·√V0 + z_{1-β}·√V1)² / (μT · exp(β0) · B1² · (1-R²))

where:
  - φ  = over-dispersion parameter (1 for pure Poisson)
  - μT = mean exposure time
  - R² = squared multiple correlation of X1 with other covariates
  - V0 = Var(b1 | β1=0) = 1/Var(X1)
  - V1 = Var(b1 | β1=B1) — depends on the distribution of X1

Supported X1 distributions: Normal, Binomial, Exponential, Uniform.

Power is computed by inverting the N formula:
  z_{1-β} = sqrt(N · μT · exp(β0) · B1² · (1-R²) / φ) / √V1
             - z_{1-α/k} · √V0 / √V1
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm as normdist

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Variance functions for b1 under the null and alternative
# ---------------------------------------------------------------------------


def _var_null(*, x_dist: str, x_params: dict[str, float]) -> float:
    """V(b1 | β1=0) = 1/Var(X1)."""
    var_x = _var_x(x_dist=x_dist, x_params=x_params, b1=0.0)
    if var_x <= 0:
        raise ValueError("Var(X1) must be positive")
    return 1.0 / var_x


def _var_alt(*, b1: float, x_dist: str, x_params: dict[str, float]) -> float:
    """V(b1 | β1=B1) — distribution-specific formula."""
    if x_dist == "normal":
        mu = x_params["mu"]
        sigma = x_params["sigma"]
        return (1.0 / sigma ** 2) * math.exp(-(b1 * mu + b1 ** 2 * sigma ** 2 / 2.0))
    elif x_dist == "binomial":
        pi = x_params["pi"]
        return 1.0 / (1.0 - pi) + 1.0 / (pi * math.exp(b1))
    elif x_dist == "exponential":
        lam = x_params["lambda"]
        return (lam - b1) ** 3 / lam
    elif x_dist == "uniform":
        c = x_params["c"]
        d = x_params["d"]
        if abs(b1) < 1e-15:
            # Limit as b1→0: Var(X) on uniform [c,d]
            return 12.0 / (d - c) ** 2
        m = (math.exp(b1 * d) - math.exp(b1 * c)) / ((d - c) * b1)
        m1 = (
            math.exp(b1 * d) * (b1 * d - 1) - math.exp(b1 * c) * (b1 * c - 1)
        ) / ((d - c) * b1 ** 2)
        m11 = (
            math.exp(b1 * d) * (2 - 2 * b1 * d + b1 ** 2 * d ** 2)
            - math.exp(b1 * c) * (2 - 2 * b1 * c + b1 ** 2 * c ** 2)
        ) / ((d - c) * b1 ** 3)
        denom = m * m11 - m1 ** 2
        if denom <= 0:
            raise ValueError("Uniform V1 denominator ≤ 0; check parameters")
        return m / denom
    else:
        raise ValueError(f"Unknown x_dist: {x_dist!r}; use 'normal', 'binomial', 'exponential', or 'uniform'")


def _var_x(*, x_dist: str, x_params: dict[str, float], b1: float) -> float:
    """Var(X1) for each supported distribution."""
    if x_dist == "normal":
        return x_params["sigma"] ** 2
    elif x_dist == "binomial":
        pi = x_params["pi"]
        return pi * (1.0 - pi)
    elif x_dist == "exponential":
        lam = x_params["lambda"]
        return lam ** (-2)
    elif x_dist == "uniform":
        c = x_params["c"]
        d = x_params["d"]
        return (d - c) ** 2 / 12.0
    else:
        raise ValueError(f"Unknown x_dist: {x_dist!r}")


# ---------------------------------------------------------------------------
# Core power / N calculation
# ---------------------------------------------------------------------------


def _power_at_n(
    *,
    n: int,
    b1: float,
    exp_b0: float,
    mu_t: float,
    phi: float,
    r_squared: float,
    x_dist: str,
    x_params: dict[str, float],
    alpha: float,
    sides: int,
) -> float:
    """Power at a given N (Signorini 1991 / Hsieh et al. 1998)."""
    k = 2 if sides == 2 else 1
    z_alpha = D.norm_ppf(1.0 - alpha / k)
    v0 = _var_null(x_dist=x_dist, x_params=x_params)
    v1 = _var_alt(b1=b1, x_dist=x_dist, x_params=x_params)
    # z_{1-β} = sqrt(N·μT·exp(β0)·B1²·(1-R²)/φ) / √V1
    #            - z_alpha · √V0/√V1
    denom = phi * v1
    numer_core = n * mu_t * exp_b0 * b1 ** 2 * (1.0 - r_squared)
    if denom <= 0 or numer_core <= 0:
        return 0.0
    z_beta = math.sqrt(numer_core / denom) - z_alpha * math.sqrt(v0 / v1)
    return float(normdist.cdf(z_beta))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def poisson_regression(
    *,
    exp_b0: float,
    rate_ratio: float,
    mu_t: float = 1.0,
    phi: float = 1.0,
    r_squared: float = 0.0,
    x_dist: str = "normal",
    x_params: dict[str, float] | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    n: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power and sample size for a Poisson regression coefficient test.

    Tests H0: β1=0 vs H1: β1=log(rate_ratio) following Signorini (1991).

    Parameters
    ----------
    exp_b0
        Baseline response rate exp(β0) when all covariates are zero.
    rate_ratio
        exp(β1)/exp(β0) — the response-rate ratio per unit increase in X1.
    mu_t
        Mean exposure time.
    phi
        Over-dispersion parameter (1 = no over-dispersion).
    r_squared
        R² when X1 is regressed on other covariates.  Use 0 when X1 is
        the only covariate.
    x_dist
        Distribution of X1: ``'normal'``, ``'binomial'``,
        ``'exponential'``, or ``'uniform'``.
    x_params
        Parameters for the X1 distribution.

        - normal:      ``{"mu": ..., "sigma": ...}``
        - binomial:    ``{"pi": ...}``
        - exponential: ``{"lambda": ...}``
        - uniform:     ``{"c": ..., "d": ...}``
    alpha
        Type-I error rate.
    sides
        1 or 2.
    n
        Sample size (supply when solve_for="power").
    power
        Target power (supply when solve_for="n").
    solve_for
        ``"n"`` or ``"power"``.
    """
    if exp_b0 <= 0:
        raise ValueError("exp_b0 must be positive")
    if rate_ratio <= 0:
        raise ValueError("rate_ratio must be positive")
    if mu_t <= 0:
        raise ValueError("mu_t must be positive")
    if phi <= 0:
        raise ValueError("phi must be positive")
    if not 0.0 <= r_squared < 1.0:
        raise ValueError("r_squared must be in [0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    if x_params is None:
        x_params = {}

    b1 = math.log(rate_ratio)

    inputs_echo: dict[str, Any] = {
        "exp_b0": exp_b0,
        "rate_ratio": rate_ratio,
        "b1": b1,
        "mu_t": mu_t,
        "phi": phi,
        "r_squared": r_squared,
        "x_dist": x_dist,
        "x_params": x_params,
        "alpha": alpha,
        "sides": sides,
        "n": n,
        "power": power,
    }

    given = sum(x is not None for x in (n, power))
    if given == 0:
        raise ValueError("supply exactly one of (n, power)")
    if given == 2 and solve_for is None:
        raise ValueError("both n and power given; specify solve_for explicitly")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _power_at_n(
            n=n, b1=b1, exp_b0=exp_b0, mu_t=mu_t, phi=phi,
            r_squared=r_squared, x_dist=x_dist, x_params=x_params,
            alpha=alpha, sides=sides,
        )
        n_out = n

    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        if b1 == 0.0:
            raise ValueError("rate_ratio must differ from 1 to solve for n")
        lo, hi = 2, 10
        while hi <= 10_000_000:
            if _power_at_n(
                n=hi, b1=b1, exp_b0=exp_b0, mu_t=mu_t, phi=phi,
                r_squared=r_squared, x_dist=x_dist, x_params=x_params,
                alpha=alpha, sides=sides,
            ) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 10,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _power_at_n(
                n=mid, b1=b1, exp_b0=exp_b0, mu_t=mu_t, phi=phi,
                r_squared=r_squared, x_dist=x_dist, x_params=x_params,
                alpha=alpha, sides=sides,
            ) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        achieved = _power_at_n(
            n=n_out, b1=b1, exp_b0=exp_b0, mu_t=mu_t, phi=phi,
            r_squared=r_squared, x_dist=x_dist, x_params=x_params,
            alpha=alpha, sides=sides,
        )

    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "poisson_regression",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Signorini, D.F. (1991). Sample size for Poisson regression. "
            "Biometrika, 78(2), 446-450.",
            "Hsieh, F.Y., Block, D.A., and Larsen, M.D. (1998). A simple "
            "method of sample size calculation for linear and logistic "
            "regression. Statistics in Medicine, 17, 1623-1634.",
        ],
    }
