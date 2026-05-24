"""McNemar test for two correlated (paired) proportions.

covers the matched-pair binary outcome design.  The 2x2 table is

                 Control = Yes   Control = No
    Case = Yes        P11             P10
    Case = No         P01             P00

with discordant pairs counted in cells (1,0) and (0,1).  The hypothesis
P10 = P01 is equivalent to marginal homogeneity Pt = Ps; equivalently,
OR = P10/P01 = 1.

This module implements the exact "Binomial Enumeration" power
calculation (Schork & Williams 1980), which marginalises over the
number of discordant pairs R ~ Binomial(N, PD) and conditional on R
uses the exact one- or two-sided binomial test of n12 ~ Binomial(R,
0.5) vs Binomial(R, P10/PD).

The user supplies the discordant cell proportions directly:

  p10 : proportion of pairs (case=Yes, control=No)  -- "1 -> 0" shift
  p01 : proportion of pairs (case=No, control=Yes)  -- "0 -> 1" shift

Equivalently, ``proportion_discordant = p10 + p01`` and the odds ratio
``OR = p10 / p01`` can be reconstructed from the inputs.
"""
from __future__ import annotations

import math
from typing import Any


def _exact_power(N: int, p10: float, p01: float, alpha: float,
                 sides: int) -> float:
    """Exact (binomial-enumeration) McNemar power, per Schork-Williams 1980.

    Marginal: R = #discordant ~ Binomial(N, PD) where PD = p10 + p01.
    Conditional on R, the count n12 ~ Binomial(R, P10/PD) under H1 and
    Binomial(R, 0.5) under H0.  We use the exact binomial critical
    region; the two-sided test rejects if the count falls in either
    tail with cumulative null probability <= alpha/2.
    """
    if N < 1:
        return 0.0
    PD = p10 + p01
    if PD <= 0.0 or PD > 1.0:
        raise ValueError("p10 + p01 must lie in (0, 1]")
    if p10 < 0 or p01 < 0:
        raise ValueError("p10 and p01 must be non-negative")
    if p10 == p01:
        return alpha  # H0 holds; power equals the size

    from scipy.stats import binom

    p_alt = p10 / PD            # P(n12 | discordant) under H1
    one_minus_PD = 1.0 - PD
    log_PD = math.log(PD) if PD > 0 else float("-inf")
    log_qPD = math.log(one_minus_PD) if one_minus_PD > 0 else float("-inf")

    # Pre-compute log N choose R via lgamma.
    lgammaN1 = math.lgamma(N + 1)

    half_alpha = alpha / 2.0 if sides == 2 else alpha
    upper_tail = (p_alt > 0.5)  # direction of H1 in conditional space

    power = 0.0
    for R in range(1, N + 1):
        # P(R discordant) = C(N, R) PD^R (1-PD)^(N-R)
        log_pR = (lgammaN1 - math.lgamma(R + 1) - math.lgamma(N - R + 1)
                  + R * log_PD + (N - R) * log_qPD)
        prob_R = math.exp(log_pR)
        if prob_R < 1e-15 and R > N * PD * 2 + 20:
            # tail truncation: once we are far past the mode and the
            # probability has decayed to machine precision, stop.
            break
        if prob_R == 0.0:
            continue

        # Critical region under H0: n12 ~ Binomial(R, 0.5).
        # Use survival/cdf to find cl (largest c with CDF(c) <= half_alpha)
        # and cu (smallest c with SF(c-1) <= half_alpha).
        if sides == 2:
            # Lower critical point.
            cl = -1
            cum = 0.0
            for j in range(R + 1):
                cum += binom.pmf(j, R, 0.5)
                if cum <= half_alpha + 1e-15:
                    cl = j
                else:
                    break
            # Upper critical point.
            cu = R + 1
            cum = 0.0
            for j in range(R, -1, -1):
                cum += binom.pmf(j, R, 0.5)
                if cum <= half_alpha + 1e-15:
                    cu = j
                else:
                    break
            pwr_R = 0.0
            if cl >= 0:
                pwr_R += float(binom.cdf(cl, R, p_alt))
            if cu <= R:
                pwr_R += float(1.0 - binom.cdf(cu - 1, R, p_alt))
        elif sides == 1:
            # One-sided rejection region in the direction of H1.
            if upper_tail:
                cu = R + 1
                cum = 0.0
                for j in range(R, -1, -1):
                    cum += binom.pmf(j, R, 0.5)
                    if cum <= half_alpha + 1e-15:
                        cu = j
                    else:
                        break
                pwr_R = float(1.0 - binom.cdf(cu - 1, R, p_alt)) if cu <= R else 0.0
            else:
                cl = -1
                cum = 0.0
                for j in range(R + 1):
                    cum += binom.pmf(j, R, 0.5)
                    if cum <= half_alpha + 1e-15:
                        cl = j
                    else:
                        break
                pwr_R = float(binom.cdf(cl, R, p_alt)) if cl >= 0 else 0.0
        else:
            raise ValueError(f"sides must be 1 or 2, got {sides}")

        power += prob_R * pwr_R

    return float(power)


def power_at_n(*, p10: float, p01: float, n: int, alpha: float,
               sides: int = 2) -> float:
    return _exact_power(n, p10, p01, alpha, sides)


def n_for_power(*, p10: float, p01: float, alpha: float, power: float,
                sides: int = 2, n_min: int = 2,
                n_max: int = 100_000) -> tuple[int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if p10 == p01:
        raise ValueError("p10 and p01 must differ to solve for N")

    lo, hi = n_min, n_min
    while hi <= n_max:
        if _exact_power(hi, p10, p01, alpha, sides) >= power:
            break
        lo = hi
        hi = max(hi + 1, int(hi * 1.7))
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _exact_power(mid, p10, p01, alpha, sides) >= power:
            hi = mid
        else:
            lo = mid

    achieved = _exact_power(hi, p10, p01, alpha, sides)
    return hi, achieved


def mcnemar(
    *,
    p10: float | None = None,
    p01: float | None = None,
    proportion_discordant: float | None = None,
    odds_ratio: float | None = None,
    difference: float | None = None,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """McNemar paired-proportions sample-size solver.

    Accepts either the direct discordant cell proportions
    ``(p10, p01)`` or one of the convenience parameterisations:

      * ``proportion_discordant`` + ``odds_ratio``
      * ``proportion_discordant`` + ``difference``  (where D = p10-p01)

    Solves for the missing quantity in ``{n, power}``.
    """
    # ---- normalise effect-size parameterisations ----
    if p10 is None or p01 is None:
        if proportion_discordant is None:
            raise ValueError(
                "supply either (p10, p01) or (proportion_discordant, "
                "odds_ratio | difference)"
            )
        PD = proportion_discordant
        if odds_ratio is not None and difference is not None:
            raise ValueError("supply only one of (odds_ratio, difference)")
        if odds_ratio is not None:
            if odds_ratio <= 0:
                raise ValueError("odds_ratio must be positive")
            p01 = PD / (1.0 + odds_ratio)
            p10 = odds_ratio * p01
        elif difference is not None:
            p10 = (PD + difference) / 2.0
            p01 = (PD - difference) / 2.0
        else:
            raise ValueError(
                "with proportion_discordant supply odds_ratio or difference"
            )

    if p10 < 0 or p01 < 0:
        raise ValueError("p10 and p01 must be non-negative")
    PD = p10 + p01
    if not 0.0 < PD <= 1.0:
        raise ValueError("p10 + p01 must lie in (0, 1]")

    inputs_echo = {
        "p10": p10, "p01": p01,
        "proportion_discordant": PD,
        "odds_ratio": (p10 / p01) if p01 > 0 else None,
        "difference": p10 - p01,
        "n": n, "alpha": alpha, "power": power, "sides": sides,
    }

    have_n = n is not None
    have_power = power is not None
    if have_n == have_power:
        raise ValueError("supply exactly one of (n, power); the other is solved")

    if solve_for is None:
        solve_for = "power" if have_n else "n"

    if solve_for == "power":
        assert n is not None
        achieved = power_at_n(p10=p10, p01=p01, n=n, alpha=alpha, sides=sides)
        result = {"n": n, "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n_req, achieved = n_for_power(p10=p10, p01=p01, alpha=alpha,
                                      power=power, sides=sides)
        result = {"n": n_req, "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "mcnemar",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "(McNemar Test)",
            "Schork, M.A. and Williams, G.W. (1980). 'Number of "
            "Observations Required for the Comparison of Two Correlated "
            "Proportions.' Communications in Statistics-Simula. Computa., "
            "B9(4), 349-357.",
            "Machin, D., Campbell, M., Fayers, P., and Pinol, A. (1997). "
            "Sample Size Tables for Clinical Studies, 2nd ed. Blackwell.",
        ],
    }
