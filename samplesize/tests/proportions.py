"""Tests for one and two proportions.

statistics; this module covers the normal-approximation z-tests:

  z_s0       — z-test using S(P0)              (default; Ryan 2013)
  z_s0_cc    — z-test using S(P0) + continuity correction (Fleiss)
  z_s_phat   — z-test using S(Phat)             (Chow, Shao & Wang)
  z_s_phat_cc — z-test using S(Phat) + continuity correction

Exact binomial-enumeration tests are not in
the MVP and raise NotImplementedError if requested.

For each variant the two-sided power is

  Power = Φ((√n(P0-P1) - z_{α/2}·s ± c) / √(P1(1-P1)))
        + 1 - Φ((√n(P0-P1) + z_{α/2}·s ± c) / √(P1(1-P1)))

with s ∈ {√(P0(1-P0)), √(P1(1-P1))} and c ∈ {0, 1/(2√n)}.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D
from samplesize.core import effect_sizes as E

VALID_TYPES = {"z_s0", "z_s0_cc", "z_s_phat", "z_s_phat_cc"}


def _power_normal(p0: float, p1: float, n: int, alpha: float, sides: int,
                  test_type: str) -> float:
    if p1 <= 0 or p1 >= 1 or p0 <= 0 or p0 >= 1:
        raise ValueError("p0 and p1 must lie in (0, 1)")
    if n < 2:
        return 0.0
    sqrt_n = math.sqrt(n)
    s_num = (math.sqrt(p0 * (1 - p0))
             if test_type.startswith("z_s0") else math.sqrt(p1 * (1 - p1)))
    s_den = math.sqrt(p1 * (1 - p1))
    cc = (1.0 / (2.0 * sqrt_n)) if test_type.endswith("_cc") else 0.0

    base = sqrt_n * (p0 - p1)
    if sides == 2:
        z = D.norm_ppf(1 - alpha / 2.0)
        from scipy.stats import norm
        upper = norm.cdf((base - z * s_num - cc) / s_den)
        lower = 1.0 - norm.cdf((base + z * s_num + cc) / s_den)
        return float(upper + lower)
    if sides == 1:
        z = D.norm_ppf(1 - alpha)
        from scipy.stats import norm
        if p1 > p0:        # upper one-sided: H1 P > P0
            return float(1.0 - norm.cdf((base + z * s_num + cc) / s_den))
        else:              # lower one-sided: H1 P < P0
            return float(norm.cdf((base - z * s_num - cc) / s_den))
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def power_at_n(*, p0: float, p1: float, n: int, alpha: float,
               sides: int = 2, test_type: str = "z_s0") -> float:
    if test_type == "exact":
        raise NotImplementedError(
            "exact binomial-enumeration test is not in the MVP; "
            "use a z-test variant in VALID_TYPES"
        )
    if test_type not in VALID_TYPES:
        raise ValueError(f"test_type must be one of {sorted(VALID_TYPES)}")
    return _power_normal(p0, p1, n, alpha, sides, test_type)


def n_for_power(*, p0: float, p1: float, alpha: float, power: float,
                sides: int = 2, test_type: str = "z_s0",
                n_min: int = 2, n_max: int = 10_000_000) -> tuple[int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if p0 == p1:
        raise ValueError("p0 and p1 must differ to solve for N")

    lo, hi = n_min, n_min
    while hi <= n_max:
        if power_at_n(p0=p0, p1=p1, n=hi, alpha=alpha, sides=sides,
                      test_type=test_type) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if power_at_n(p0=p0, p1=p1, n=mid, alpha=alpha, sides=sides,
                      test_type=test_type) >= power:
            hi = mid
        else:
            lo = mid

    achieved = power_at_n(p0=p0, p1=p1, n=hi, alpha=alpha, sides=sides,
                          test_type=test_type)
    return hi, achieved


def effect_for_power(*, p0: float, n: int, alpha: float, power: float,
                     sides: int = 2, test_type: str = "z_s0",
                     direction: str = "above", tol: float = 1e-9) -> float:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if direction == "above":
        lo, hi = p0 + 1e-9, 1.0 - 1e-9
    elif direction == "below":
        lo, hi = 1e-9, p0 - 1e-9
    else:
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")

    # power at extreme is presumably high; bisect for the boundary.
    p_lo = power_at_n(p0=p0, p1=hi if direction == "above" else lo,
                      n=n, alpha=alpha, sides=sides, test_type=test_type)
    if p_lo < power:
        raise RuntimeError("requested power not achievable at the boundary")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if power_at_n(p0=p0, p1=mid, n=n, alpha=alpha, sides=sides,
                      test_type=test_type) >= power:
            if direction == "above":
                hi = mid
            else:
                lo = mid
        else:
            if direction == "above":
                lo = mid
            else:
                hi = mid
        if abs(hi - lo) < tol:
            break
    return hi if direction == "above" else lo


VALID_TWO_PROP_TYPES = {"z_pooled", "z_unpooled", "z_pooled_cc", "z_unpooled_cc"}


def _two_prop_power(p1: float, p2: float, n1: int, n2: int, alpha: float,
                    sides: int, test_type: str) -> float:
    """Chow et al. (2008) large-sample power for the two-proportion z-test."""
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("p1, p2 must be in (0, 1)")
    if n1 < 2 or n2 < 2:
        return 0.0

    p_bar = (n1 * p1 + n2 * p2) / (n1 + n2)
    sigma_p = math.sqrt(p_bar * (1 - p_bar) * (1.0 / n1 + 1.0 / n2))
    sigma_u = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    cc = 0.5 * (1.0 / n1 + 1.0 / n2) if test_type.endswith("_cc") else 0.0
    sigma_crit = sigma_p if test_type.startswith("z_pooled") else sigma_u

    from scipy.stats import norm
    if sides == 2:
        z = D.norm_ppf(1 - alpha / 2.0)
        # Upper tail (rejection because p̂1 - p̂2 large) and lower tail.
        upper = 1.0 - norm.cdf((z * sigma_crit + cc - diff) / sigma_u)
        lower = norm.cdf((-z * sigma_crit - cc - diff) / sigma_u)
        return float(upper + lower)
    if sides == 1:
        z = D.norm_ppf(1 - alpha)
        if diff > 0:
            return float(1.0 - norm.cdf((z * sigma_crit + cc - diff) / sigma_u))
        return float(norm.cdf((-z * sigma_crit - cc - diff) / sigma_u))
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def two_prop_power_at_n(*, p1: float, p2: float, n1: int, n2: int,
                        alpha: float, sides: int = 2,
                        test_type: str = "z_pooled") -> float:
    if test_type not in VALID_TWO_PROP_TYPES:
        raise ValueError(f"test_type must be one of {sorted(VALID_TWO_PROP_TYPES)}")
    return _two_prop_power(p1, p2, n1, n2, alpha, sides, test_type)


def two_prop_n_for_power(*, p1: float, p2: float, alpha: float, power: float,
                         sides: int = 2, allocation: float = 1.0,
                         test_type: str = "z_pooled",
                         n_min: int = 2, n_max: int = 10_000_000):
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ to solve for N")

    def n2_for(n1):
        return max(2, math.ceil(allocation * n1))

    def p_at(n1):
        return _two_prop_power(p1, p2, n1, n2_for(n1), alpha, sides, test_type)

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

    n1 = hi
    n2 = n2_for(n1)
    return n1, n2, _two_prop_power(p1, p2, n1, n2, alpha, sides, test_type)


def two_proportions(
    *,
    p1: float,
    p2: float | None = None,
    diff: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    test_type: str = "z_pooled",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Normal-approximation z-test for two independent proportions."""
    if diff is not None and p2 is None:
        p2 = p1 - diff
    elif diff is not None and p2 is not None:
        raise ValueError("supply only one of (p2, diff)")

    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    elif n2 is not None and n1 is None:
        n1 = max(2, math.ceil(n2 / allocation))

    inputs_echo = {
        "p1": p1, "p2": p2, "diff": diff, "n1": n1, "n2": n2,
        "alpha": alpha, "power": power, "sides": sides,
        "allocation": allocation, "test_type": test_type,
    }

    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    have_p2 = p2 is not None
    given = sum((have_n, have_power, have_p2))
    if given < 2:
        raise ValueError("supply exactly two of (p2/diff, n, power)")

    if solve_for is None:
        if not have_n:
            solve_for = "n"
        elif not have_power:
            solve_for = "power"
        else:
            raise ValueError("supply two of (p2/diff, n, power)")

    if solve_for == "power":
        assert p2 is not None and n1 is not None and n2 is not None
        achieved = two_prop_power_at_n(
            p1=p1, p2=p2, n1=n1, n2=n2, alpha=alpha, sides=sides,
            test_type=test_type,
        )
        result = {"n1": n1, "n2": n2, "n": n1 + n2, "p2": p2,
                  "achieved_power": achieved,
                  "effect_h": E.cohens_h(p1, p2)}
    elif solve_for == "n":
        assert p2 is not None and power is not None
        n1r, n2r, achieved = two_prop_n_for_power(
            p1=p1, p2=p2, alpha=alpha, power=power, sides=sides,
            allocation=allocation, test_type=test_type,
        )
        result = {"n1": n1r, "n2": n2r, "n": n1r + n2r, "p2": p2,
                  "achieved_power": achieved,
                  "effect_h": E.cohens_h(p1, p2)}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "two_proportions",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Chow, S.C., Shao, J., Wang, H. (2008). Sample Size Calculations in Clinical Research.",
            "Fleiss, Levin & Paik (2003). Statistical Methods for Rates and Proportions.",
        ],
    }


def one_proportion(
    *,
    p0: float,
    p1: float | None = None,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    test_type: str = "z_s0",
    solve_for: str | None = None,
    direction: str = "above",
) -> dict[str, Any]:
    """Normal-approximation z-test for a single proportion."""
    inputs_echo = {
        "p0": p0, "p1": p1, "n": n, "alpha": alpha, "power": power,
        "sides": sides, "test_type": test_type, "direction": direction,
    }
    given = sum(x is not None for x in (p1, n, power))
    if given < 2:
        raise ValueError("supply exactly two of (p1, n, power)")
    if solve_for is None:
        if n is None:
            solve_for = "n"
        elif power is None:
            solve_for = "power"
        elif p1 is None:
            solve_for = "effect"

    if solve_for == "power":
        assert p1 is not None and n is not None
        achieved = power_at_n(p0=p0, p1=p1, n=n, alpha=alpha, sides=sides,
                              test_type=test_type)
        result = {"n": n, "achieved_power": achieved,
                  "effect_h": E.cohens_h(p1, p0)}
    elif solve_for == "n":
        assert p1 is not None and power is not None
        n_req, achieved = n_for_power(p0=p0, p1=p1, alpha=alpha, power=power,
                                      sides=sides, test_type=test_type)
        result = {"n": n_req, "achieved_power": achieved,
                  "effect_h": E.cohens_h(p1, p0)}
    elif solve_for == "effect":
        assert n is not None and power is not None
        p1_det = effect_for_power(p0=p0, n=n, alpha=alpha, power=power,
                                  sides=sides, test_type=test_type,
                                  direction=direction)
        achieved = power_at_n(p0=p0, p1=p1_det, n=n, alpha=alpha,
                              sides=sides, test_type=test_type)
        result = {
            "n": n, "p1": p1_det, "achieved_power": achieved,
            "effect_h": E.cohens_h(p1_det, p0),
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "one_proportion",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Ryan, T.P. (2013). Sample Size Determination and Power.",
            "Fleiss, Levin & Paik (2003). Statistical Methods for Rates and Proportions.",
        ],
    }
