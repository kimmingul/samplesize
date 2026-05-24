"""Pearson correlation power and sample size.

exact cumulative correlation density.  This module implements the
exact noncentral correlation density via scipy's gamma + hypergeometric
2F1 and integrates it numerically.  A Fisher-z approximation backend
is retained for callers that want the textbook (pwr-package style)
result.

Exact density (Hotelling 1953, simplified):

    f(r; ρ, n)
      = (n - 2)
        · Γ(n - 1) / [√(2π) · Γ(n - 1/2)]
        · (1 - ρ²)^((n-1)/2) · (1 - r²)^((n-4)/2)
        · (1 - ρr)^(-(n - 3/2))
        · ₂F₁(1/2, 1/2; n - 1/2; (1 + ρr)/2)
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# -------- Exact (Guenther/Hotelling) backend --------------------------------

def _exact_log_pdf(r: float, rho: float, n: int) -> float:
    """log f(r; ρ, n) (returns -inf when r is at the boundary)."""
    if abs(r) >= 1 or abs(rho) >= 1 or n < 4:
        return float("-inf")
    from scipy.special import gammaln, hyp2f1
    z = 0.5 * (1.0 + rho * r)
    f21 = hyp2f1(0.5, 0.5, n - 0.5, z)
    if f21 <= 0 or not math.isfinite(f21):
        return float("-inf")
    log_pdf = (
        math.log(n - 2)
        + gammaln(n - 1) - gammaln(n - 0.5)
        + 0.5 * (n - 1) * math.log(1.0 - rho * rho)
        + 0.5 * (n - 4) * math.log(1.0 - r * r)
        - (n - 1.5) * math.log(1.0 - rho * r)
        - 0.5 * math.log(2.0 * math.pi)
        + math.log(float(f21))
    )
    return log_pdf


def _exact_pdf(r: float, rho: float, n: int) -> float:
    lp = _exact_log_pdf(r, rho, n)
    return math.exp(lp) if lp > -700 else 0.0


def _exact_cdf(r_val: float, rho: float, n: int) -> float:
    """P(R ≤ r_val | ρ, n) via numerical integration of the exact PDF."""
    from scipy.integrate import quad
    if r_val <= -1:
        return 0.0
    if r_val >= 1:
        return 1.0
    val, _ = quad(_exact_pdf, -1.0 + 1e-9, r_val,
                  args=(rho, n), limit=200)
    return max(0.0, min(1.0, val))


def _exact_critical_r(alpha: float, rho0: float, n: int, sides: int) -> float:
    """Critical r for one-sided upper test of H0:ρ=ρ0 vs H1:ρ>ρ0
    at level alpha (two-sided uses alpha/2)."""
    from scipy.optimize import brentq
    target = 1.0 - (alpha / 2.0 if sides == 2 else alpha)
    return brentq(lambda r: _exact_cdf(r, rho0, n) - target,
                  rho0 + 1e-6, 1.0 - 1e-6, xtol=1e-9)


def _exact_power(r: float, rho0: float, n: int,
                 alpha: float, sides: int) -> float:
    if r == rho0:
        return float(alpha)
    if n <= 3:
        return 0.0
    if sides == 2:
        # Two critical values, symmetric about rho0 only when rho0=0.
        r_hi = _exact_critical_r(alpha, rho0, n, sides=2)
        # Lower critical via reflection of the alpha/2 tail of the
        # density at rho0.  Use brentq on the lower-tail CDF.
        from scipy.optimize import brentq
        r_lo = brentq(lambda r_: _exact_cdf(r_, rho0, n) - alpha / 2.0,
                      -1.0 + 1e-6, rho0 - 1e-6, xtol=1e-9)
        upper_tail = 1.0 - _exact_cdf(r_hi, r, n)
        lower_tail = _exact_cdf(r_lo, r, n)
        return upper_tail + lower_tail
    # one-sided
    if r >= rho0:
        r_crit = _exact_critical_r(alpha, rho0, n, sides=1)
        return 1.0 - _exact_cdf(r_crit, r, n)
    else:
        from scipy.optimize import brentq
        r_crit = brentq(lambda r_: _exact_cdf(r_, rho0, n) - alpha,
                        -1.0 + 1e-6, rho0 - 1e-6, xtol=1e-9)
        return _exact_cdf(r_crit, r, n)


# -------- Fisher-z (legacy / fast) backend ----------------------------------

def _fisher_z_power(rho0: float, rho1: float, n: int,
                    alpha: float, sides: int) -> float:
    if rho0 == rho1:
        return float(alpha)
    if n <= 3:
        return 0.0
    from scipy.stats import norm
    delta = math.atanh(rho1) - math.atanh(rho0)
    se = 1.0 / math.sqrt(n - 3)
    if sides == 2:
        z = D.norm_ppf(1 - alpha / 2.0)
        upper = 1.0 - norm.cdf(z - delta / se)
        lower = norm.cdf(-z - delta / se)
        return float(upper + lower)
    z = D.norm_ppf(1 - alpha)
    if rho1 > rho0:
        return float(1.0 - norm.cdf(z - delta / se))
    return float(norm.cdf(-z - delta / se))


# -------- public dispatch ---------------------------------------------------

def _power(rho0: float, rho1: float, n: int, alpha: float, sides: int,
           method: str) -> float:
    if not -1.0 < rho0 < 1.0 or not -1.0 < rho1 < 1.0:
        raise ValueError("rho values must lie in (-1, 1)")
    if method == "exact":
        return _exact_power(rho1, rho0, n, alpha, sides)
    if method == "fisher-z":
        return _fisher_z_power(rho0, rho1, n, alpha, sides)
    raise ValueError(f"method must be 'exact' or 'fisher-z', got {method!r}")


def power_at_n(*, r: float, n: int, alpha: float, sides: int = 2,
               rho0: float = 0.0, method: str = "exact") -> float:
    return _power(rho0, r, n, alpha, sides, method)


def n_for_power(*, r: float, alpha: float, power: float, sides: int = 2,
                rho0: float = 0.0, method: str = "exact",
                n_min: int = 4, n_max: int = 100_000) -> tuple[int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if r == rho0:
        raise ValueError("r and rho0 must differ to solve for N")

    lo, hi = n_min, n_min
    while hi <= n_max:
        if _power(rho0, r, hi, alpha, sides, method) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _power(rho0, r, mid, alpha, sides, method) >= power:
            hi = mid
        else:
            lo = mid
    return hi, _power(rho0, r, hi, alpha, sides, method)


def pearson(
    *,
    r: float,
    rho0: float = 0.0,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    method: str = "exact",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Pearson correlation significance test.

    `r` is the population correlation under H1; `rho0` is the null value
    (defaults to 0).  `method` is `"exact"` (Guenther/Hotelling exact
    density) or `"fisher-z"` (textbook
    Fisher z-transform; slightly less accurate at small N).
    """
    inputs_echo = {
        "r": r, "rho0": rho0, "n": n, "alpha": alpha,
        "power": power, "sides": sides, "method": method,
    }
    if (n is None) == (power is None):
        if not (n is not None and power is not None):
            raise ValueError("supply exactly one of (n, power)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = power_at_n(r=r, n=n, alpha=alpha, sides=sides,
                              rho0=rho0, method=method)
        result = {"n": n, "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n_req, achieved = n_for_power(r=r, alpha=alpha, power=power,
                                      sides=sides, rho0=rho0, method=method)
        result = {"n": n_req, "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "pearson_correlation",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Guenther, W.C. (1977). Desk calculation of probabilities for the "
            "distribution of the sample correlation coefficient.",
            "Hotelling, H. (1953). New light on the correlation coefficient and "
            "its transforms.",
        ],
    }
