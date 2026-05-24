"""MxM (square) cross-over design power & sample-size.


An MxM cross-over design has M treatments and M periods.  Each subject
receives all M treatments in a sequence determined by a Latin square or
similar balanced arrangement.  The analysis treats the design as a
one-way repeated-measures ANOVA with M levels (one per treatment).

The power calculations follow the Muller, LaVange, Ramey & Ramey
(1992) / Muller & Barton (1989) general linear multivariate model
framework, the same machinery used for one-way repeated
measures (Chapter 569) and validated there.

For the Geisser-Greenhouse correction (the default test statistic in
For MxM designs, the epsilon parameter adjusts the F degrees of
freedom:

    df1 = (M-1) · ε
    df2 = (N-1) · (M-1) · ε
    λ   = N · M · σ_m² / [σ² · (1-ρ)] · ε

The compound-symmetry correlation ρ models equal correlations between
any two treatment measurements on the same subject.

This module is a thin wrapper around ``samplesize.tests.anova_rm``
(which implements the same algorithm) with MxM-specific metadata.

References
----------
* Muller, K.E., LaVange, L.E., Ramey, S.L., and Ramey, C.T. (1992).
  Power Calculations for General Linear Multivariate Models Including
  Repeated Measures Applications. JASA 87(420):1209-1226.
* Muller, K.E. and Barton, C.N. (1989). Approximate Power for
  Repeated-Measures ANOVA Lacking Sphericity. JASA 84(406):549-555.
* Jones, B. and Kenward, M.G. (2015). Design and Analysis of
  Cross-Over Trials, 3rd Ed. CRC Press.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.tests.anova_rm import _power_within_f, _sigma_m_squared
from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _geisser_greenhouse_epsilon_cs(m: int, rho: float) -> float:
    """GG epsilon for compound-symmetry (CS) covariance matrix.

    For a CS matrix with equal σ and constant off-diagonal ρ, the
    GG epsilon is exactly 1.0 when the matrix is spherical.  Under CS,
    the matrix is spherical (ε = 1) when all variances of pairwise
    differences are equal — which is satisfied by definition.  Hence
    ε = 1 for a true CS covariance.

    When the GG-corrected F-test is used with a CS
    covariance, ε = 1 applies (CS covariance → sphericity satisfied).
    We therefore return 1.0 here.
    """
    return 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mxm_crossover(
    *,
    means: list[float],
    sigma: float,
    rho: float,
    epsilon: float | None = None,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power and sample size for an MxM balanced cross-over design.

    Geisser-Greenhouse corrected F-test under a compound-symmetry
    covariance assumption.

    Parameters
    ----------
    means
        List of M (≥ 3) treatment means under the alternative hypothesis.
        Under H0 all means are equal.  The length of this list determines
        the number of treatments/periods M.
    sigma
        Between-subject standard deviation at a single time point
        (assumed equal across all M treatments).
    rho
        Compound-symmetry correlation between any two measurements on
        the same subject (0 ≤ ρ < 1).
    epsilon
        Geisser-Greenhouse sphericity correction (1/(M-1) ≤ ε ≤ 1).
        If ``None`` (default), epsilon is set to 1.0, which is exact
        for a true compound-symmetry covariance matrix and matches
        For compound-symmetry covariance epsilon=1 applies.
    alpha
        Significance level.  Default 0.05.
    n
        Number of subjects (for solve_for="power").
    power
        Target power (for solve_for="n").
    solve_for
        ``"n"`` or ``"power"``.  Inferred from whichever of (n, power)
        is provided.
    """
    m = len(means)
    if m < 3:
        raise ValueError(
            "MxM cross-over requires at least M=3 treatments; "
            "use the 2x2 cross-over procedures for M=2"
        )
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    # Resolve epsilon
    eps_min = 1.0 / (m - 1)
    if epsilon is None:
        epsilon = 1.0  # CS covariance → ε = 1 (sphericity satisfied)
    else:
        if not (eps_min - 1e-9 <= epsilon <= 1.0 + 1e-9):
            raise ValueError(
                f"epsilon must be in [{eps_min:.4f}, 1.0] for M={m}"
            )

    inputs_echo: dict[str, Any] = {
        "means": means, "sigma": sigma, "rho": rho,
        "epsilon": epsilon, "alpha": alpha, "n": n, "power": power,
    }

    given = sum(x is not None for x in (n, power))
    if given == 0:
        raise ValueError("supply exactly one of (n, power)")
    if given == 2 and solve_for is None:
        raise ValueError("both n and power given; specify solve_for explicitly")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    def pwr(n_val: int) -> float:
        p, _, _, _ = _power_within_f(means, sigma, rho, epsilon, n_val, alpha)
        return p

    if solve_for == "power":
        assert n is not None
        pw, df1, df2, ncp = _power_within_f(means, sigma, rho, epsilon, n, alpha)
        achieved = pw
        n_out = n

    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 2, m + 1
        while hi <= 1_000_000:
            if pwr(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 1,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pwr(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        pw, df1, df2, ncp = _power_within_f(means, sigma, rho, epsilon, n_out, alpha)
        achieved = pwr(n_out)

    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    sigma_m = math.sqrt(_sigma_m_squared(means))
    sigma_e = math.sqrt(sigma ** 2 * (1.0 - rho))

    return {
        "method_id": "mxm_crossover",
        "solve_for": solve_for,
        "n": n_out,
        "achieved_power": achieved,
        "m_periods_treatments": m,
        "sigma_m": sigma_m,
        "sigma_e": sigma_e,
        "epsilon": epsilon,
        "inputs_echo": inputs_echo,
        "citations": [
            "Muller, K.E., LaVange, L.E., Ramey, S.L., and Ramey, C.T. "
            "(1992). Power Calculations for General Linear Multivariate "
            "Models Including Repeated Measures Applications. "
            "JASA 87(420):1209-1226.",
            "Muller, K.E. and Barton, C.N. (1989). Approximate Power for "
            "Repeated-Measures ANOVA Lacking Sphericity. "
            "JASA 84(406):549-555.",
            "Jones, B. and Kenward, M.G. (2015). Design and Analysis of "
            "Cross-Over Trials, 3rd Ed. CRC Press.",
        ],
    }
