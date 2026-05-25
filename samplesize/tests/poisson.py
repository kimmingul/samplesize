"""Poisson rate sample-size / power calculators.

Covers:
  one_sample_poisson_rate       — one-sample Poisson rate test
  two_sample_poisson_rates      — two-sample Poisson rates (W5 variance-stabilised)
  incidence_rate_ratio_with_followup — incidence rate ratio (W5, variable t1/t2)

  Ch 412: Tests for One Poisson Rate
  Ch 437: Tests for the Ratio of Two Poisson Rates
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm as _norm, poisson as _poisson


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _z(p: float) -> float:
    """Inverse standard-normal CDF."""
    return float(_norm.ppf(p))


def _norm_cdf(x: float) -> float:
    return float(_norm.cdf(x))


# ---------------------------------------------------------------------------
# Method 1: one_sample_poisson_rate
# ---------------------------------------------------------------------------
# The one-sided test H0: λ ≤ λ0 vs Ha: λ > λ0 (or lower direction).
# Power is exact Poisson enumeration at the integer critical value X*.
# Sample size is found by searching n upward until power ≥ target.
#
# Critical value for upper test: smallest X* such that
#   sum_{x=X*}^∞ Poisson(n*λ0).pmf(x) ≤ α
# Power = sum_{x=X*}^∞ Poisson(n*λ1).pmf(x)
# ---------------------------------------------------------------------------

def _critical_value_upper(n: int, lam0: float, alpha: float) -> int:
    """Smallest X* such that P(X≥X* | nλ0) ≤ α (upper one-sided test)."""
    mu0 = n * lam0
    # Start from the mean and search upward
    x = max(0, int(mu0))
    while True:
        tail = float(1.0 - _poisson.cdf(x - 1, mu0))  # P(X >= x)
        if tail <= alpha:
            return x
        x += 1


def _critical_value_lower(n: int, lam0: float, alpha: float) -> int:
    """Largest X* such that P(X≤X* | nλ0) ≤ α (lower one-sided test)."""
    mu0 = n * lam0
    x = max(0, int(mu0))
    while x >= 0:
        tail = float(_poisson.cdf(x, mu0))  # P(X <= x)
        if tail <= alpha:
            return x
        if x == 0:
            return -1  # no valid critical value
        x -= 1
    return -1


def _one_sample_poisson_power(
    n: int, lam0: float, lam1: float, alpha: float, sides: int
) -> float:
    """Exact Poisson power for fixed n."""
    if n < 1:
        return 0.0
    mu1 = n * lam1
    if sides == 1:
        if lam1 > lam0:
            # upper test
            xstar = _critical_value_upper(n, lam0, alpha)
            return float(1.0 - _poisson.cdf(xstar - 1, mu1))
        else:
            # lower test
            xstar = _critical_value_lower(n, lam0, alpha)
            if xstar < 0:
                return 0.0
            return float(_poisson.cdf(xstar, mu1))
    else:
        # two-sided: upper and lower, each at alpha/2
        xstar_u = _critical_value_upper(n, lam0, alpha / 2.0)
        xstar_l = _critical_value_lower(n, lam0, alpha / 2.0)
        power = float(1.0 - _poisson.cdf(xstar_u - 1, mu1))
        if xstar_l >= 0:
            power += float(_poisson.cdf(xstar_l, mu1))
        return power


def one_sample_poisson_rate(
    *,
    lam0: float,
    lam1: float | None = None,
    n: int | None = None,
    alpha: float = 0.025,
    power: float | None = None,
    sides: int = 1,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """One-sample Poisson rate test.

    H0: λ = λ0  vs  Ha: λ ≠ λ0  (sides=2)  or  one-sided (sides=1).

    Provide exactly two of (lam1, n, power).
    """
    inputs_echo = {
        "lam0": lam0, "lam1": lam1, "n": n,
        "alpha": alpha, "power": power, "sides": sides,
    }

    given = sum(x is not None for x in (lam1, n, power))
    if given < 2:
        raise ValueError("supply exactly two of (lam1, n, power)")
    if lam0 <= 0:
        raise ValueError("lam0 must be > 0")
    if lam1 is not None and lam1 <= 0:
        raise ValueError("lam1 must be > 0")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    if solve_for is None:
        if n is None:
            solve_for = "n"
        elif power is None:
            solve_for = "power"
        else:
            raise ValueError("supply exactly two of (lam1, n, power)")

    if solve_for == "power":
        assert lam1 is not None and n is not None
        achieved = _one_sample_poisson_power(n, lam0, lam1, alpha, sides)
        result: dict[str, Any] = {"n": n, "achieved_power": achieved}

    elif solve_for == "n":
        assert lam1 is not None and power is not None
        if lam1 == lam0:
            raise ValueError("lam1 must differ from lam0 to solve for n")
        # Search n upward
        n_req = 1
        n_max = 10_000_000
        while n_req <= n_max:
            achieved = _one_sample_poisson_power(n_req, lam0, lam1, alpha, sides)
            if achieved >= power:
                break
            n_req += 1
        else:
            raise RuntimeError("failed to find n within limit")
        result = {"n": n_req, "achieved_power": achieved}

    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "one_sample_poisson_rate",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Guenther, W.C. (1977). Sampling Inspection in Statistical Quality Control. "
            "Griffin's Statistical Monographs. Pages 25-29.",
            "Ostle, B. and Malone, L. (1988). Statistics in Research, 4th Edition. "
            "Iowa State University Press. Pages 116-118.",
        ],
    }


# ---------------------------------------------------------------------------
# Method 2 & 7: two_sample_poisson_rates / incidence_rate_ratio_with_followup
#   Both use the W5 variance-stabilised statistic (Gu et al. 2008).
#   Method 7 is the same formula with t1 ≠ t2.
# ---------------------------------------------------------------------------
# W5 sample size formula (solving for N1):
#   N1 = [ (z_alpha * C + z_power * D) / A ]^2 / (lam1 * t1) - 3/8
# where
#   d   = t1*N1 / (t2*N2) → for equal N: d = t1/t2
#   A   = 2*(1 - sqrt(RR0/RRa))
#   C   = sqrt((RR0 + d) / RRa)
#   D   = sqrt((RRa + d) / RRa)
#   B   = lam1 * t1 * N1 + 3/8
#
# Power formula (one-sided, upper):
#   Power = Phi( (|A| * sqrt(B) - z_alpha * C) / D )
#
# For two-sided tests both directions are considered and power is summed
# (normally only one direction contributes for RRa ≠ RR0).
# ---------------------------------------------------------------------------

def _w5_power(
    n1: int, lam1: float, t1: float, t2: float,
    rr0: float, rra: float, alpha: float, sides: int
) -> float:
    """W5 variance-stabilised power for given N1."""
    if n1 < 1:
        return 0.0
    n2 = n1  # equal allocation (N2=N1)
    d = (t1 * n1) / (t2 * n2)  # = t1/t2 for equal N
    A = 2.0 * (1.0 - math.sqrt(rr0 / rra))
    B = lam1 * t1 * n1 + 3.0 / 8.0
    C = math.sqrt((rr0 + d) / rra)
    D = math.sqrt((rra + d) / rra)
    if sides == 1:
        z_a = _z(1.0 - alpha)
        return _norm_cdf((abs(A) * math.sqrt(B) - z_a * C) / D)
    else:
        z_a = _z(1.0 - alpha / 2.0)
        pw_upper = _norm_cdf((abs(A) * math.sqrt(B) - z_a * C) / D)
        # lower tail (swap rr0/rra)
        if rra != 0:
            A2 = 2.0 * (1.0 - math.sqrt(rra / rr0))
            C2 = math.sqrt((rra + d) / rr0)
            D2 = math.sqrt((rr0 + d) / rr0)
            B2 = lam1 * t1 * n1 + 3.0 / 8.0
            pw_lower = _norm_cdf((abs(A2) * math.sqrt(B2) - z_a * C2) / D2)
        else:
            pw_lower = 0.0
        return pw_upper + pw_lower


def _w5_n_for_power_formula(
    lam1: float, t1: float, t2: float,
    rr0: float, rra: float, alpha: float, power: float, sides: int
) -> int:
    """Closed-form N1 seed from W5 formula (Gu et al. 2008; equal N allocation).

    The function searches for the smallest N1 such that power (computed
    with B = lam1*t1*N1 + 3/8) meets the target.  This closed form inverts
    that equation:
        [(z_a*C + z_b*D)/A]^2 = B = lam1*t1*N1 + 3/8
    =>  N1 = ([(z_a*C + z_b*D)/A]^2 - 3/8) / (lam1*t1)
    """
    d = t1 / t2  # equal N → d = t1/t2
    A = 2.0 * (1.0 - math.sqrt(rr0 / rra))
    C = math.sqrt((rr0 + d) / rra)
    D = math.sqrt((rra + d) / rra)
    if sides == 1:
        z_a = _z(1.0 - alpha)
    else:
        z_a = _z(1.0 - alpha / 2.0)
    z_b = _z(power)
    # Solve: (z_a*C + z_b*D)^2 = A^2 * B  =>  B = val^2/A^2
    # B = lam1*t1*N1 + 3/8  =>  N1 = (B - 3/8) / (lam1*t1)
    val = z_a * C + z_b * D
    B_target = (val / A) ** 2
    n1_raw = (B_target - 3.0 / 8.0) / (lam1 * t1)
    return max(2, math.ceil(n1_raw))


def _two_poisson_w5(
    *,
    lam1: float,
    rra: float,
    t1: float,
    t2: float,
    rr0: float,
    alpha: float,
    power: float | None,
    n1: int | None,
    sides: int,
    solve_for: str,
    method_id: str,
    citation_chapter: str,
) -> dict[str, Any]:
    """Core W5 solver shared by two_sample_poisson_rates and
    incidence_rate_ratio_with_followup."""
    if solve_for == "n":
        assert power is not None
        # Use closed-form seed then verify/correct with exact power
        n1_seed = _w5_n_for_power_formula(
            lam1, t1, t2, rr0, rra, alpha, power, sides
        )
        # Walk forward until power achieved (discrete ceiling)
        n_try = max(2, n1_seed - 2)
        while True:
            achieved = _w5_power(n_try, lam1, t1, t2, rr0, rra, alpha, sides)
            if achieved >= power:
                break
            n_try += 1
            if n_try > 100_000_000:
                raise RuntimeError("failed to bracket N1")
        n1_out = n_try
        n2_out = n_try
        achieved_out = achieved
    else:  # power
        assert n1 is not None
        n1_out = n1
        n2_out = n1
        achieved_out = _w5_power(n1, lam1, t1, t2, rr0, rra, alpha, sides)

    return {
        "method_id": method_id,
        "solve_for": solve_for,
        "n1": n1_out,
        "n2": n2_out,
        "n": n1_out + n2_out,
        "achieved_power": achieved_out,
        "inputs_echo": {
            "lam1": lam1, "rra": rra, "t1": t1, "t2": t2,
            "rr0": rr0, "alpha": alpha, "power": power,
            "n1": n1, "sides": sides,
        },
        "citations": [
            "Gu, K., Ng, H.K.T., Tang, M.L., Schucany, W.R. (2008). Testing the ratio of two Poisson rates. Biometrical Journal.",
            "Gu, K., Ng, H.K.T., Tang, M.L., and Schucany, W. (2008). "
            "Testing the Ratio of Two Poisson Rates. "
            "Biometrical Journal, 50(2), 283-298.",
            "Huffman, M. (1984). An Improved Approximate Two-Sample Poisson Test. "
            "Applied Statistics, 33(2), 224-226.",
        ],
    }


def two_sample_poisson_rates(
    *,
    lam1: float,
    rra: float,
    rr0: float = 1.0,
    t1: float = 1.0,
    t2: float = 1.0,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    sides: int = 1,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-sample Poisson rate ratio test using W5 variance-stabilised statistic
    (W5 statistic, Gu et al. 2008; equal sample sizes N1=N2).

    RR = λ2/λ1, test H0: RR = RR0 vs Ha: RR ≠ RR0 (sides=2) or one-sided (sides=1).
    Supply exactly two of (rra, n1, power) — rra is always required.
    """
    if solve_for is None:
        if n1 is None and power is not None:
            solve_for = "n"
        elif n1 is not None and power is None:
            solve_for = "power"
        else:
            raise ValueError("supply exactly one of (n1, power) with rra")

    return _two_poisson_w5(
        lam1=lam1, rra=rra, t1=t1, t2=t2, rr0=rr0,
        alpha=alpha, power=power, n1=n1, sides=sides,
        solve_for=solve_for,
        method_id="two_sample_poisson_rates",
        citation_chapter="Chapter 437: Tests for the Ratio of Two Poisson Rates",
    )


# ---------------------------------------------------------------------------
# Method: tests_two_poisson_means
#   Tests whether two Poisson means (counts, not rates) are equal.
#   Uses the conditional score test (Krishnamoorthy & Thomson 2004) and
#   the C-test (Przyborowski & Wilenski 1940; Reiczigel et al. 2008).
#
#   H0: lambda1 = lambda2   vs   Ha: lambda1 != lambda2  (or one-sided)
#
#   The test statistic is based on observing x1 ~ Poisson(n1*lambda1) and
#   x2 ~ Poisson(n2*lambda2).  For equal sample sizes (n1=n2=n) and
#   H0: lambda1/lambda2 = ratio0 the power is found by exact enumeration
#   over the joint Poisson distribution.
#
#   For practical purposes (and matching standard textbook formulae) we use
#   the normal approximation:
#       Z = (x1 - n1*lam0) / sqrt(n1*lam0 + n2*lam0)  (under H0: both = lam0)
#   and the non-null distribution of Z under Ha: (lambda1=lam1, lambda2=lam2)
#       mu_Z  = (n1*lam1 - n1*lam0) / sqrt(n1*lam0 + n2*lam0)     [approx]
#       sd_Z  = sqrt((n1*lam1 + n2*lam2) / (n1*lam0 + n2*lam0))
#
#   This yields a closed-form approximation suitable for sample-size planning.
#   The approach follows Bain & Engelhardt (1992), and Krishnamoorthy &
#   Thomson (2004) who derive exact conditional tests and compare with this
#   normal approximation.
# ---------------------------------------------------------------------------

def _two_poisson_means_power(
    n1: int, n2: int, lam1: float, lam2: float,
    lam0: float, alpha: float, sides: int,
) -> float:
    """Normal-approximation power for the two-Poisson-means test.

    Under H0: lambda1 = lambda2 = lam0 (the common mean under H0).
    Under Ha: lambda1=lam1, lambda2=lam2.

    The variance-stabilised (square-root) approach gives:

        sqrt(x1) - sqrt(x2) ~ N(sqrt(n1*lam1) - sqrt(n2*lam2), 1/4+1/4)

    For equal-n allocation the NCP is:

        delta = 2 * (sqrt(n*lam1) - sqrt(n*lam2))

    which is used with a chi-square(1) or z-test reference.

    We use the simpler Wald-type approximation directly here:

        Z = (X1/n1 - X2/n2) / sqrt(lam0/n1 + lam0/n2)

    Under Ha: E[Z] = (lam1 - lam2) / sqrt(lam0/n1 + lam0/n2)

    Power = Phi(-z_crit + NCP) for one-sided upper test.
    """
    if n1 < 1 or n2 < 1:
        return 0.0
    se0 = math.sqrt(lam0 / n1 + lam0 / n2)
    if se0 == 0:
        return 1.0 if lam1 != lam2 else alpha
    ncp = (lam1 - lam2) / se0
    if sides == 1:
        z_crit = _z(1.0 - alpha)
        # Use |ncp| so power is the same regardless of direction
        return _norm_cdf(abs(ncp) - z_crit)
    else:
        z_crit = _z(1.0 - alpha / 2.0)
        return _norm_cdf(ncp - z_crit) + _norm_cdf(-ncp - z_crit)


def tests_two_poisson_means(
    *,
    lam1: float,
    lam2: float,
    lam0: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-sample test for equality of two Poisson means.

    Tests H0: lambda1 = lambda2 vs Ha: lambda1 != lambda2 (or one-sided).
    Uses the normal approximation to the Poisson for sample-size planning.

    Unlike the two-sample rate-ratio test, here both groups contribute equal
    observation windows (counts, not rates) and the parameter of interest is
    the absolute difference of means.

    Provide exactly two of (n1, power); n2 defaults to n1 (equal allocation).

    Parameters
    ----------
    lam1
        Poisson mean of group 1 under H1 (> 0).
    lam2
        Poisson mean of group 2 under H1 (> 0).
    lam0
        Hypothesised common mean under H0.  Defaults to (lam1+lam2)/2
        when omitted (pooled estimate approach).
    alpha
        Significance level (default 0.05).
    power
        Target power (required when solve_for='n').
    n1
        Sample size for group 1 (required when solve_for='power').
    n2
        Sample size for group 2.  Defaults to n1 (equal allocation).
    sides
        1 for one-sided, 2 for two-sided (default).
    solve_for
        ``'n'`` or ``'power'``.  Inferred when omitted.

    Returns
    -------
    dict
        Standard envelope: method_id, solve_for, n, n1, n2,
        achieved_power, inputs_echo, citations.
    """
    if lam1 <= 0:
        raise ValueError("lam1 must be > 0")
    if lam2 <= 0:
        raise ValueError("lam2 must be > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    # Default lam0 to midpoint of the two means
    if lam0 is None:
        lam0 = (lam1 + lam2) / 2.0
    if lam0 <= 0:
        raise ValueError("lam0 must be > 0")

    inputs_echo = {
        "lam1": lam1, "lam2": lam2, "lam0": lam0,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2, "sides": sides,
    }

    # Infer solve_for
    if solve_for is None:
        if n1 is None and power is not None:
            solve_for = "n"
        elif n1 is not None and power is None:
            solve_for = "power"
        else:
            raise ValueError("supply exactly one of (n1, power)")

    citations = [
        "Krishnamoorthy, K. and Thomson, J. (2004). A more powerful test for "
        "comparing two Poisson means. Journal of Statistical Planning and "
        "Inference, 119, 23-35.",
        "Bain, L. J. and Engelhardt, M. (1992). Introduction to Probability "
        "and Mathematical Statistics, 2nd ed. Duxbury Press.",
        "Przyborowski, J. and Wilenski, H. (1940). Homogeneity of results in "
        "testing samples from Poisson series. Biometrika, 31, 313-323.",
    ]

    if solve_for == "power":
        if n1 is None:
            raise ValueError("n1 is required when solve_for='power'")
        if n1 < 1:
            raise ValueError("n1 must be >= 1")
        n2_eff = n2 if n2 is not None else n1
        if n2_eff < 1:
            raise ValueError("n2 must be >= 1")
        achieved = _two_poisson_means_power(n1, n2_eff, lam1, lam2, lam0, alpha, sides)
        return {
            "method_id": "tests_two_poisson_means",
            "solve_for": "power",
            "n1": n1,
            "n2": n2_eff,
            "n": n1 + n2_eff,
            "achieved_power": achieved,
            "inputs_echo": inputs_echo,
            "citations": citations,
        }

    elif solve_for == "n":
        if power is None:
            raise ValueError("power is required when solve_for='n'")
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        if lam1 == lam2:
            raise ValueError("lam1 must differ from lam2 to solve for n")

        # Search for smallest n1 (equal allocation n2=n1)
        n_try = 1
        n_max = 10_000_000
        while n_try <= n_max:
            achieved = _two_poisson_means_power(n_try, n_try, lam1, lam2, lam0, alpha, sides)
            if achieved >= power:
                break
            n_try += 1
        else:
            raise RuntimeError("failed to find n within limit")

        return {
            "method_id": "tests_two_poisson_means",
            "solve_for": "n",
            "n1": n_try,
            "n2": n_try,
            "n": 2 * n_try,
            "achieved_power": achieved,
            "inputs_echo": inputs_echo,
            "citations": citations,
        }

    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")


def incidence_rate_ratio_with_followup(
    *,
    lam1: float,
    rra: float,
    rr0: float = 1.0,
    t1: float,
    t2: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    sides: int = 1,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Person-time incidence rate ratio with potentially unequal follow-up times
    per arm (W5 statistic, Gu et al. 2008; variable t1/t2).

    Identical formula to two_sample_poisson_rates but t1 and t2 are explicit
    required arguments (no default of 1.0).
    """
    if solve_for is None:
        if n1 is None and power is not None:
            solve_for = "n"
        elif n1 is not None and power is None:
            solve_for = "power"
        else:
            raise ValueError("supply exactly one of (n1, power) with rra")

    return _two_poisson_w5(
        lam1=lam1, rra=rra, t1=t1, t2=t2, rr0=rr0,
        alpha=alpha, power=power, n1=n1, sides=sides,
        solve_for=solve_for,
        method_id="incidence_rate_ratio_with_followup",
        citation_chapter="Chapter 437: Tests for the Ratio of Two Poisson Rates",
    )
