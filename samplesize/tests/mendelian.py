"""Mendelian randomization — continuous and binary outcome.

Continuous outcome (Brion et al. 2013):
    mendelian_randomization_continuous

Binary outcome (Burgess 2014; Brion et al. 2013):
    mendelian_randomization_binary

For the binary outcome the instrumental-variable Wald estimator on the
log-odds scale is used.  The non-centrality parameter (NCP) of the
underlying chi-square(1) test is:

    NCP = N · ρ²(XG) · [log(OR)]² · Pc · (1 - Pc)

where OR is the causal odds ratio per SD change in X, Pc is the
proportion of Y = 1 in the population, and ρ²(XG) is the variance
of X explained by the instrument G.  This is the score-test
approximation given in Burgess (2014) eq. 2.

Continuous outcome detail (Brion et al. 2013):

Models the causal effect β_YX of exposure X on outcome Y using a genetic
instrumental variable G.

Power is based on a non-central chi-square with df=1:

    Power = Pr(χ²_{1,NCP} > χ²_{1,1-α})

where the NCP is:

    NCP = E(b_2SLS)² / σ²_{b_2SLS}

and

    σ²_{b_2SLS} = σ²_{eY} / (N · ρ²_{XG} · σ²_X)

    E(b_2SLS) = β_YX + cov(eY, eX) / (N · ρ²_{XG})   [biased version]

For sample-size search we use the simplified large-sample approximation:

    NCP ≈ N · ρ²_{XG} · σ²_X · β²_{YX} / σ²_{eY}

which is obtained by noting that for large N the confounding bias term
is negligible and σ²_{eY} ≈ σ²_Y - σ²_X · β_YX · (2β_OLS - β_YX).

Reference: Brion, M.J.A., Shakhbazov, K., Visscher, P.M. (2013).
Calculating statistical power in Mendelian randomization studies.
International Journal of Epidemiology, 42, 1497-1501.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import chi2
from scipy.stats import ncx2


# ---------------------------------------------------------------------------
# Core NCP and power
# ---------------------------------------------------------------------------


def _sigma_ey_sq(
    *,
    beta_yx: float,
    beta_ols: float,
    sigma_x: float,
    sigma_y: float,
) -> float:
    """σ²_{eY} = σ²_Y - σ²_X · β_YX · (2β_OLS - β_YX)."""
    return sigma_y ** 2 - sigma_x ** 2 * beta_yx * (2.0 * beta_ols - beta_yx)


def _ncp(
    *,
    n: int,
    beta_yx: float,
    beta_ols: float,
    rho2_xg: float,
    sigma_x: float,
    sigma_y: float,
) -> float:
    """Non-centrality parameter for testing β_YX."""
    cov_ey_ex = (beta_ols - beta_yx) * sigma_x ** 2
    sigma_ey2 = _sigma_ey_sq(
        beta_yx=beta_yx, beta_ols=beta_ols,
        sigma_x=sigma_x, sigma_y=sigma_y,
    )
    if sigma_ey2 <= 0:
        raise ValueError(
            "σ²_{eY} ≤ 0; check that σ_Y and the β parameters are consistent"
        )
    # Variance of b_2SLS
    sigma_b2 = sigma_ey2 / (n * rho2_xg * sigma_x ** 2)
    # Expected value of b_2SLS
    e_b2sls = beta_yx + cov_ey_ex / (n * rho2_xg)
    return e_b2sls ** 2 / sigma_b2


def _power_at_n(
    *,
    n: int,
    beta_yx: float,
    beta_ols: float,
    rho2_xg: float,
    sigma_x: float,
    sigma_y: float,
    alpha: float,
    sides: int,
) -> float:
    """Power at a given N."""
    alpha_eff = alpha if sides == 2 else 2.0 * alpha  # chi2 is always 2-sided
    chi2_crit = chi2.ppf(1.0 - alpha_eff, df=1)
    ncp = _ncp(
        n=n, beta_yx=beta_yx, beta_ols=beta_ols,
        rho2_xg=rho2_xg, sigma_x=sigma_x, sigma_y=sigma_y,
    )
    return float(1.0 - ncx2.cdf(chi2_crit, df=1, nc=ncp))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mendelian_randomization_continuous(
    *,
    beta_yx: float,
    beta_ols: float,
    rho2_xg: float,
    sigma_x: float,
    sigma_y: float,
    alpha: float = 0.05,
    sides: int = 2,
    n: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power and sample size for Mendelian randomization with continuous outcome.


    Parameters
    ----------
    beta_yx
        Causal effect of X on Y (β_YX, the 2SLS target parameter).
    beta_ols
        Expected OLS regression coefficient of Y on X (β_OLS).
    rho2_xg
        Proportion of variance in X explained by G (R² of X~G regression).
    sigma_x
        Standard deviation of X.
    sigma_y
        Standard deviation of Y.
    alpha
        Type-I error rate.
    sides
        1 or 2.  For one-sided tests α is used in place of α/2 (same chi²
        quantile substitution as logistic/Cox).
    n
        Sample size (supply when solve_for="power").
    power
        Target power (supply when solve_for="n").
    solve_for
        ``"n"`` or ``"power"``.
    """
    if not 0.0 < rho2_xg < 1.0:
        raise ValueError("rho2_xg must be in (0, 1)")
    if sigma_x <= 0:
        raise ValueError("sigma_x must be positive")
    if sigma_y <= 0:
        raise ValueError("sigma_y must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    inputs_echo: dict[str, Any] = {
        "beta_yx": beta_yx,
        "beta_ols": beta_ols,
        "rho2_xg": rho2_xg,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
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
            n=n, beta_yx=beta_yx, beta_ols=beta_ols,
            rho2_xg=rho2_xg, sigma_x=sigma_x, sigma_y=sigma_y,
            alpha=alpha, sides=sides,
        )
        n_out = n

    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        if beta_yx == 0.0:
            raise ValueError("beta_yx must be non-zero to solve for n")
        lo, hi = 3, 100
        while hi <= 100_000_000:
            if _power_at_n(
                n=hi, beta_yx=beta_yx, beta_ols=beta_ols,
                rho2_xg=rho2_xg, sigma_x=sigma_x, sigma_y=sigma_y,
                alpha=alpha, sides=sides,
            ) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 100,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _power_at_n(
                n=mid, beta_yx=beta_yx, beta_ols=beta_ols,
                rho2_xg=rho2_xg, sigma_x=sigma_x, sigma_y=sigma_y,
                alpha=alpha, sides=sides,
            ) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        achieved = _power_at_n(
            n=n_out, beta_yx=beta_yx, beta_ols=beta_ols,
            rho2_xg=rho2_xg, sigma_x=sigma_x, sigma_y=sigma_y,
            alpha=alpha, sides=sides,
        )

    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "mendelian_randomization_continuous",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Continuous Outcome.",
            "Brion, M.J.A., Shakhbazov, K., Visscher, P.M. (2013). "
            "Calculating statistical power in Mendelian randomization "
            "studies. International Journal of Epidemiology, 42, 1497-1501.",
            "Burgess, S. and Thompson, S.G. (2015). Mendelian Randomization "
            "Methods for Using Genetic Variants in Causal Estimation. "
            "Chapman & Hall/CRC Press.",
        ],
    }


# ---------------------------------------------------------------------------
# Mendelian Randomization — Binary Outcome (Burgess 2014)
# ---------------------------------------------------------------------------


def _mr_binary_ncp(*, n: int, rho2_xg: float, log_or: float, pc: float) -> float:
    """NCP for the MR binary outcome score test.

    Burgess (2014) Int J Epidemiol 43:922-929, equation 2.
    NCP = N · ρ²(XG) · (log OR)² · Pc · (1 - Pc)
    """
    return n * rho2_xg * log_or ** 2 * pc * (1.0 - pc)


def _mr_binary_power(
    *,
    n: int,
    rho2_xg: float,
    odds_ratio: float,
    pc: float,
    alpha: float,
    sides: int,
) -> float:
    """Power of the MR binary outcome test at a given N."""
    alpha_eff = alpha if sides == 2 else 2.0 * alpha
    chi2_crit = chi2.ppf(1.0 - alpha_eff, df=1)
    log_or = math.log(odds_ratio)
    ncp = _mr_binary_ncp(n=n, rho2_xg=rho2_xg, log_or=log_or, pc=pc)
    return float(1.0 - ncx2.cdf(chi2_crit, df=1, nc=ncp))


def mendelian_randomization_binary(
    *,
    odds_ratio: float,
    rho2_xg: float,
    pc: float,
    alpha: float = 0.05,
    sides: int = 2,
    n: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power and sample size for Mendelian randomization with binary outcome.

    Mendelian Randomization with a Binary Outcome.

    Uses the score-test NCP approximation from Burgess (2014):

        NCP = N · ρ²(XG) · (log OR)² · Pc · (1 - Pc)

    Parameters
    ----------
    odds_ratio
        Causal odds ratio of Y per one-SD change in X (OR > 0, OR ≠ 1).
    rho2_xg
        Proportion of variance in X explained by G (R² of X ~ G regression).
    pc
        Proportion of Y = 1 in the study population.
    alpha
        Type-I error rate.
    sides
        1 or 2.  For one-sided tests α is used in place of α/2 (same
        chi² quantile substitution as the continuous MR method).
    n
        Sample size (supply when solve_for='power').
    power
        Target power (supply when solve_for='n').
    solve_for
        'n' or 'power'.
    """
    if not 0.0 < rho2_xg < 1.0:
        raise ValueError("rho2_xg must be in (0, 1)")
    if not 0.0 < pc < 1.0:
        raise ValueError("pc must be in (0, 1)")
    if odds_ratio <= 0.0 or odds_ratio == 1.0:
        raise ValueError("odds_ratio must be > 0 and != 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    inputs_echo: dict[str, Any] = {
        "odds_ratio": odds_ratio,
        "rho2_xg": rho2_xg,
        "pc": pc,
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
        achieved = _mr_binary_power(
            n=n, rho2_xg=rho2_xg, odds_ratio=odds_ratio,
            pc=pc, alpha=alpha, sides=sides,
        )
        n_out = n

    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 3, 100
        while hi <= 1_000_000_000:
            if _mr_binary_power(
                n=hi, rho2_xg=rho2_xg, odds_ratio=odds_ratio,
                pc=pc, alpha=alpha, sides=sides,
            ) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 1,000,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _mr_binary_power(
                n=mid, rho2_xg=rho2_xg, odds_ratio=odds_ratio,
                pc=pc, alpha=alpha, sides=sides,
            ) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        achieved = _mr_binary_power(
            n=n_out, rho2_xg=rho2_xg, odds_ratio=odds_ratio,
            pc=pc, alpha=alpha, sides=sides,
        )

    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "mendelian_randomization_binary",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Burgess, S. (2014). Sample size and power calculations in "
            "Mendelian randomization with a single instrumental variable "
            "and a binary outcome. International Journal of Epidemiology "
            "43:922-929.",
            "Brion, M.J.A., Shakhbazov, K., Visscher, P.M. (2013). "
            "Calculating statistical power in Mendelian randomization "
            "studies. International Journal of Epidemiology 42:1497-1501.",
            "Burgess, S. and Thompson, S.G. (2015). Mendelian Randomization "
            "Methods for Using Genetic Variants in Causal Estimation. "
            "Chapman & Hall/CRC Press.",
        ],
    }
