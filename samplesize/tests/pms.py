"""Post-Marketing Surveillance — Machin et al. (1997).


Four design types are supported:

1. Cohort – No Background Incidence
   β = Σ_{i=0}^{A-1} [N^i · R0^i · exp(-N·R0) / i!]  (Poisson CDF)
   Power = 1 - β

2. Cohort – Known Background Incidence
   z_{1-β} = (D·√N - z_{1-α}·√R0) / √(R0+D)

3. Cohort – Unknown Background Incidence (control group)
   z_{1-β} = (D·√(MN) - z_{1-α}·√((M+1)·R̄·(1-R̄))) /
              √(M·R0·(1-R0) + (R0+D)·(1-R0-D))
   where R̄ = (R0 + M·(R0+D)) / (1+M)

4. Matched Case-Control Study
   z_{1-β} = (|R0-Ω|·√N - z_{1-α}·√((1+1/M)·Π·(1-Π))) /
              √(R0·(1-R0)/M + Ω·(1-Ω))
   where Ω = (R0+D)/(1+D),  Π = (R0/(1+M))·(M + Ω/R0)
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm as normdist
from scipy.stats import poisson as poisson_dist

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Design-specific power functions
# ---------------------------------------------------------------------------


def _power_type1(n: int, r0: float, a: int) -> float:
    """Cohort, no background incidence — Poisson CDF."""
    # β = P(Y < A) where Y ~ Poisson(N·R0)
    mu = n * r0
    beta = float(poisson_dist.cdf(a - 1, mu))
    return 1.0 - beta


def _power_type2(n: int, r0: float, d: float, alpha: float, sides: int) -> float:
    """Cohort, known background incidence."""
    k = 2 if sides == 2 else 1
    z_alpha = D.norm_ppf(1.0 - alpha / k)
    numer = d * math.sqrt(n) - z_alpha * math.sqrt(r0)
    denom = math.sqrt(r0 + d)
    if denom == 0:
        return 0.0
    z_beta = numer / denom
    return float(normdist.cdf(z_beta))


def _power_type3(
    n: int, r0: float, d: float, m: float, alpha: float, sides: int
) -> float:
    """Cohort, unknown background incidence (control group)."""
    k = 2 if sides == 2 else 1
    z_alpha = D.norm_ppf(1.0 - alpha / k)
    r_bar = (r0 + m * (r0 + d)) / (1.0 + m)
    numer = d * math.sqrt(m * n) - z_alpha * math.sqrt(
        (m + 1) * r_bar * (1.0 - r_bar)
    )
    denom = math.sqrt(
        m * r0 * (1.0 - r0) + (r0 + d) * (1.0 - r0 - d)
    )
    if denom == 0:
        return 0.0
    z_beta = numer / denom
    return float(normdist.cdf(z_beta))


def _power_type4(
    n: int, r0: float, d: float, m: float, alpha: float, sides: int
) -> float:
    """Matched case-control study."""
    k = 2 if sides == 2 else 1
    z_alpha = D.norm_ppf(1.0 - alpha / k)
    omega = (r0 + d) / (1.0 + d)
    pi = (r0 / (1.0 + m)) * (m + omega / r0)
    numer = abs(r0 - omega) * math.sqrt(n) - z_alpha * math.sqrt(
        (1.0 + 1.0 / m) * pi * (1.0 - pi)
    )
    denom = math.sqrt(r0 * (1.0 - r0) / m + omega * (1.0 - omega))
    if denom == 0:
        return 0.0
    z_beta = numer / denom
    return float(normdist.cdf(z_beta))


def _power_dispatch(
    *,
    design_type: int,
    n: int,
    r0: float,
    d: float,
    a: int,
    m: float,
    alpha: float,
    sides: int,
) -> float:
    if design_type == 1:
        return _power_type1(n=n, r0=r0, a=a)
    elif design_type == 2:
        return _power_type2(n=n, r0=r0, d=d, alpha=alpha, sides=sides)
    elif design_type == 3:
        return _power_type3(n=n, r0=r0, d=d, m=m, alpha=alpha, sides=sides)
    elif design_type == 4:
        return _power_type4(n=n, r0=r0, d=d, m=m, alpha=alpha, sides=sides)
    else:
        raise ValueError(f"design_type must be 1–4, got {design_type}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def post_marketing_surveillance(
    *,
    design_type: int = 1,
    r0: float,
    d: float = 0.0,
    a: int = 1,
    m: float = 1.0,
    alpha: float = 0.05,
    sides: int = 1,
    n: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power and sample size for post-marketing surveillance studies.


    1 – Cohort, no background incidence (Poisson model)
    2 – Cohort, known background incidence
    3 – Cohort, unknown background incidence (requires control group)
    4 – Matched case-control study

    Parameters
    ----------
    design_type
        1, 2, 3, or 4 (see module docstring).
    r0
        Background incidence rate (baseline proportion of adverse events).
    d
        Additional incidence rate attributable to the drug (types 2–4).
    a
        Number of occurrences threshold (type 1 only).
    m
        Number of control patients per case patient (types 3 and 4).
    alpha
        Type-I error probability.
    sides
        1 (one-sided) or 2 (two-sided).  For type 1, this parameter is
        unused (power is computed directly from the Poisson CDF).
    n
        Number of patients (supply when solve_for="power").
    power
        Target power (supply when solve_for="n").
    solve_for
        ``"n"`` or ``"power"``.
    """
    if not 0.0 < r0 < 1.0:
        raise ValueError("r0 must be in (0, 1)")
    if design_type in (2, 3, 4) and d <= 0:
        raise ValueError("d must be positive for design_type 2, 3, 4")
    if design_type in (2, 3, 4) and r0 + d >= 1.0:
        raise ValueError("r0 + d must be < 1")
    if design_type == 1 and a < 1:
        raise ValueError("a must be >= 1")
    if design_type in (3, 4) and m < 1:
        raise ValueError("m must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    inputs_echo: dict[str, Any] = {
        "design_type": design_type,
        "r0": r0,
        "d": d,
        "a": a,
        "m": m,
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
        achieved = _power_dispatch(
            design_type=design_type, n=n, r0=r0, d=d, a=a,
            m=m, alpha=alpha, sides=sides,
        )
        n_out = n

    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 1, 10
        while hi <= 100_000_000:
            if _power_dispatch(
                design_type=design_type, n=hi, r0=r0, d=d, a=a,
                m=m, alpha=alpha, sides=sides,
            ) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 100,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _power_dispatch(
                design_type=design_type, n=mid, r0=r0, d=d, a=a,
                m=m, alpha=alpha, sides=sides,
            ) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        achieved = _power_dispatch(
            design_type=design_type, n=n_out, r0=r0, d=d, a=a,
            m=m, alpha=alpha, sides=sides,
        )

    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "post_marketing_surveillance",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Machin, D., Campbell, M., Fayers, P., and Pinol, A. (1997). "
            "Sample Size Tables for Clinical Studies, 2nd Ed. Blackwell "
            "Science. Oxford.",
        ],
    }
