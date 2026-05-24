"""Tests for one and two exponential means.

Implements two exponential distribution test procedures:

  * Chapter 405 — Tests for One Exponential Mean
    -> :func:`tests_one_exponential_mean`

  * Chapter 435 — Tests for Two Exponential Means
    -> :func:`tests_two_exponential_means`

For one exponential mean the key result (Epstein 1960) is that
``2·r·θ̂/θ ~ χ²(2r)`` where ``r`` is the number of failures.  For the
fixed-failure (Type-II censored) plan the smallest ``r`` satisfying the
chi-square ratio criterion is found; ``n`` is then derived from the
expected study duration formula.

For two exponential means the ratio  ``θ̂₁/θ̂₂ ~ (θ₁/θ₂)·F(r₁, r₂)``
under the null (θ₁ = θ₂) gives a standard F-distribution test.  Power
and sample size follow directly from the F distribution.

------------------
Epstein, B. (1960). Tests for the validity of the assumption that the
underlying distribution of life is exponential. Technometrics, 2, 83-101,
439-468.

Bain, L.J. & Engelhardt, M. (1991). Statistical Analysis of Reliability and
Life-Testing Models, 2nd Ed., Marcel Dekker.

Desu, M.M. & Raghavarao, D. (1990). Sample Size Methodology. Academic Press.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import chi2 as _chi2
from scipy.stats import f as _f


# ---------------------------------------------------------------------------
# One exponential mean — fixed-failure (Type-II censored) plan
# ---------------------------------------------------------------------------

def _chi2_ratio_power(r: int, theta0: float, theta1: float,
                       alpha: float, sides: int) -> float:
    """Power of the chi-square ratio test for fixed-failures r.

    Under H₀: θ = θ₀, the statistic 2rθ̂/θ₀ ~ χ²(2r).
    Under H₁: θ = θ₁, the statistic 2rθ̂/θ₀ ~ (θ₁/θ₀)·χ²(2r).

    Epstein (1960, p.437) criterion:
      H₁: θ₀ > θ₁  (consumer/lower tail)
        Reject H₀ if θ̂ ≤ θ_C where θ_C is chosen so P(reject | H₀) = α.
        Critical chi-square: χ²_{α, 2r}  (lower tail).
        Power = P(χ²(2r) ≤ χ²_{α,2r} · θ₀/θ₁).
    """
    df = 2 * r
    if sides == 1:
        # H₁: θ₀ > θ₁ — one-sided lower
        crit_low = _chi2.ppf(alpha, df)
        power = float(_chi2.cdf(crit_low * theta0 / theta1, df))
        return power
    else:
        # two-sided — rare for this test but handle gracefully
        crit_low = _chi2.ppf(alpha / 2.0, df)
        crit_high = _chi2.ppf(1.0 - alpha / 2.0, df)
        ratio = theta0 / theta1
        power = float(_chi2.cdf(crit_low * ratio, df)
                      + (1.0 - _chi2.cdf(crit_high * ratio, df)))
        return power


def _find_r_for_power(theta0: float, theta1: float, alpha: float,
                       power: float, sides: int,
                       r_max: int = 5000) -> int:
    """Smallest r ≥ 1 achieving >= power."""
    for r in range(1, r_max + 1):
        if _chi2_ratio_power(r, theta0, theta1, alpha, sides) >= power:
            return r
    raise RuntimeError(
        f"Could not achieve power={power} with r ≤ {r_max}; "
        "check that theta0 > theta1 for the one-sided test."
    )


def _expected_duration_without_replacement(r: int, n: int,
                                            theta: float) -> float:
    """E(t₀) for without-replacement, failure-terminated experiment."""
    return theta * sum(1.0 / (n - i + 1) for i in range(1, r + 1))


def _expected_duration_with_replacement(r: int, n: int,
                                         theta: float) -> float:
    """E(t₀) for with-replacement, failure-terminated experiment."""
    return theta * r / n


def _n_from_expected_duration(r: int, theta: float, t0: float,
                               replacement: str) -> int:
    """Smallest n giving E(t₀) ≤ t0 for the given replacement method."""
    if replacement == "with":
        # n = ceil(theta * r / t0)
        return max(r, math.ceil(theta * r / t0))
    else:
        # without replacement: use the log approximation for large n,
        # exact harmonic for small n.
        # Solve Σ_{i=1}^{r} 1/(n-i+1) = t0 / theta numerically.
        target = t0 / theta
        for n in range(r, 100_000):
            if _expected_duration_without_replacement(r, n, 1.0) <= target:
                return n
        raise RuntimeError("n could not be found within 100,000; check inputs.")


def tests_one_exponential_mean(
    *,
    theta0: float,
    theta1: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    r: int | None = None,
    t0: float = 1.0,
    sides: int = 1,
    replacement: str = "without",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Tests for one exponential mean.

    Uses the chi-square pivot ``2·r·θ̂/θ ~ χ²(2r)`` from Epstein (1960).
    For a fixed-failure (Type-II censored) plan:

    * ``theta0`` — null (acceptable / producer) mean life
    * ``theta1`` — alternative (unacceptable / consumer) mean life
    * ``alpha`` — producer's risk (Type-I error)
    * ``power`` — 1 - beta (consumer's risk = beta)
    * ``r`` — number of failures to observe (if fixing r directly)
    * ``t0`` — study duration (used to derive n from r)
    * ``sides`` — 1 (default; H₁: θ₀ > θ₁) or 2
    * ``replacement`` — "with" or "without" (default)

    When ``solve_for="power"`` provide ``n`` (and optionally ``r``).
    When ``solve_for="n"`` provide ``power``.

    Returns
    -------
    dict
        Standard envelope plus ``r`` (number of failures) and
        ``expected_t0`` (expected study duration).
    """
    if theta0 <= 0 or theta1 <= 0:
        raise ValueError("theta0 and theta1 must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if replacement not in ("with", "without"):
        raise ValueError("replacement must be 'with' or 'without'")

    inputs_echo: dict[str, Any] = {
        "theta0": theta0, "theta1": theta1, "alpha": alpha,
        "power": power, "n": n, "r": r, "t0": t0,
        "sides": sides, "replacement": replacement,
    }

    if solve_for is None:
        if power is None and n is not None:
            solve_for = "power"
        elif power is not None:
            solve_for = "n"
        else:
            raise ValueError("supply one of (n, power) or set solve_for")

    if solve_for == "power":
        if n is None:
            raise ValueError("n required for solve_for='power'")
        # If r not supplied, derive from the chi-square criterion given n
        # and t0 (time-terminated mode).  For fixed-failure mode supply r.
        if r is None:
            # Approximate r ~ n * (1 - exp(-t0/theta1)) for without replacement
            p_fail = 1.0 - math.exp(-t0 / theta1)
            r_est = max(1, round(n * p_fail))
            r = r_est
        achieved = _chi2_ratio_power(r, theta0, theta1, alpha, sides)
        et0 = (_expected_duration_with_replacement(r, n, theta1)
               if replacement == "with"
               else _expected_duration_without_replacement(r, n, theta1))
        result: dict[str, Any] = {"n": n, "r": r,
                                   "achieved_power": achieved,
                                   "expected_t0": et0}

    elif solve_for == "n":
        if power is None:
            raise ValueError("power required for solve_for='n'")
        if not 0 < power < 1:
            raise ValueError("power must be in (0, 1)")
        r_found = _find_r_for_power(theta0, theta1, alpha, power, sides)
        achieved = _chi2_ratio_power(r_found, theta0, theta1, alpha, sides)
        n_found = _n_from_expected_duration(r_found, theta1, t0, replacement)
        et0 = (_expected_duration_with_replacement(r_found, n_found, theta1)
               if replacement == "with"
               else _expected_duration_without_replacement(r_found, n_found, theta1))
        result = {"n": n_found, "r": r_found,
                  "achieved_power": achieved,
                  "expected_t0": et0}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "tests_one_exponential_mean",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Epstein, B. (1960). Tests for the validity of the assumption that the "
            "underlying distribution of life is exponential. Technometrics, 2, 83-101.",
            "Bain, L.J. & Engelhardt, M. (1991). Statistical Analysis of Reliability "
            "and Life-Testing Models, 2nd Ed., Marcel Dekker.",
        ],
    }


# ---------------------------------------------------------------------------
# Two exponential means — F-ratio test
# ---------------------------------------------------------------------------

def _two_exp_power(n1: int, n2: int, theta1: float, theta2: float,
                   alpha: float, sides: int) -> float:
    """Power of the F-ratio test for two exponential means.

    θ̂₁/θ̂₂ ~ (θ₁/θ₂)·F(r₁, r₂) where r_i = n_i (run to all failures).

    Under H₁: true ratio ρ = θ₁/θ₂:
      Two-sided: reject if F ≤ F_{α/2, r₁, r₂} or F ≥ F_{1-α/2, r₁, r₂}
      where F = θ̂₁/θ̂₂.
      Power = P(F(r₁,r₂) ≤ F_{α/2,r₁,r₂}/ρ) + P(F(r₁,r₂) ≥ F_{1-α/2,r₁,r₂}/ρ)
    """
    r1, r2 = n1, n2   # failures = sample sizes (run to completion)
    rho = theta1 / theta2
    if sides == 2:
        f_lo = _f.ppf(alpha / 2.0, r1, r2)
        f_hi = _f.ppf(1.0 - alpha / 2.0, r1, r2)
        power = float(_f.cdf(f_lo / rho, r1, r2)
                      + (1.0 - _f.cdf(f_hi / rho, r1, r2)))
    else:
        if rho > 1.0:
            # H₁: θ₁ > θ₂ — upper tail
            f_crit = _f.ppf(1.0 - alpha, r1, r2)
            power = float(1.0 - _f.cdf(f_crit / rho, r1, r2))
        else:
            # H₁: θ₁ < θ₂ — lower tail
            f_crit = _f.ppf(alpha, r1, r2)
            power = float(_f.cdf(f_crit / rho, r1, r2))
    return power


def tests_two_exponential_means(
    *,
    theta1: float,
    theta2: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Tests for two exponential means.

    Uses the F-ratio ``θ̂₁/θ̂₂ ~ (θ₁/θ₂)·F(r₁,r₂)`` where the
    experiment is run until all items fail (r_i = n_i).

    Parameters
    ----------
    theta1, theta2 : positive floats
        Mean lifetimes of groups 1 and 2 under H₁.
    alpha : float
        Type-I error rate.
    power : float or None
        Target power (1 – β).
    n1, n2 : int or None
        Sample sizes.  Supply exactly one of (n1/n2) or power.
    allocation : float
        n2 = ceil(allocation * n1).  Ignored when n2 is supplied.
    sides : int
        1 or 2 (default 2).
    solve_for : str or None
        "n" or "power".
    """
    if theta1 <= 0 or theta2 <= 0:
        raise ValueError("theta1 and theta2 must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))

    inputs_echo: dict[str, Any] = {
        "theta1": theta1, "theta2": theta2, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2,
        "allocation": allocation, "sides": sides,
    }

    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply one of (n1/n2, power)")

    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _two_exp_power(n1, n2, theta1, theta2, alpha, sides)
        result: dict[str, Any] = {"n1": n1, "n2": n2, "n": n1 + n2,
                                   "achieved_power": achieved}

    elif solve_for == "n":
        assert power is not None
        if not 0 < power < 1:
            raise ValueError("power must be in (0, 1)")

        def n2_for(n1_val: int) -> int:
            return max(2, math.ceil(allocation * n1_val))

        def p_at(n1_val: int) -> float:
            return _two_exp_power(n1_val, n2_for(n1_val),
                                   theta1, theta2, alpha, sides)

        lo, hi = 2, 2
        while hi <= 10_000_000:
            if p_at(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if p_at(mid) >= power:
                hi = mid
            else:
                lo = mid
        n1r = hi
        n2r = n2_for(n1r)
        achieved = _two_exp_power(n1r, n2r, theta1, theta2, alpha, sides)
        result = {"n1": n1r, "n2": n2r, "n": n1r + n2r,
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "tests_two_exponential_means",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Bain, L.J. & Engelhardt, M. (1991). Statistical Analysis of Reliability "
            "and Life-Testing Models, 2nd Ed., Marcel Dekker.",
            "Desu, M.M. & Raghavarao, D. (1990). Sample Size Methodology. Academic Press.",
        ],
    }
