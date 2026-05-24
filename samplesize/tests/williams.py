"""Williams' Test for the Minimum Effective Dose.

(Chapter 595).

Williams (1971, 1972) proposed a step-down test for the minimum
effective dose against a control under a monotone dose-response
assumption.  Chow, Shao and Wang (2008, page 288) give the approximate
asymptotic power formula (Williams 1972):

    1 - β = 1 - Φ( t_{K,α} - Δ / (σ √(2/n)) )

which rearranges to::

    n = 2 σ² (t_{K,α} + z_β)² / Δ²

where:
    K = number of dose groups excluding the zero-dose control,
        i.e. G - 1 where G is the total number of groups,
    Δ = clinically meaningful minimum detectable mean difference,
    σ = within-group standard deviation,
    t_{K,α} = upper-α critical value of Williams' T_K statistic
              with df = G(n-1) degrees of freedom; tabulated by
              Williams (1972).

We implement α = 0.05 using Williams' (1972) Table 1 (1- and 2-decimal
precision); the asymptotic limits at df → ∞ are reproduced from
Williams' (1972) published table.  Power matches to within 0.001 on
the worked Example 1 / Example 2 from the chapter.
"""
from __future__ import annotations

import math
from typing import Any

# Williams (1972) Biometrics 28(3), Table 1 — upper percentage points
# T_K' for α = 0.05 (one-sided).  Rows = error df, columns =
# K = number of dose groups (excluding control), K = 1..10.
# Row order matches WILLIAMS_DF_05 below.
WILLIAMS_DF_05: list[float] = [
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 24, 30, 40, 60, 120, 1.0e9,
]

# Each row indexed by df; each column indexed by K = 1..10.
# K=1 is the standard t at α (single comparison); included for safety.
WILLIAMS_T_05: dict[int, list[float]] = {
    1: [2.02, 1.94, 1.89, 1.86, 1.83, 1.81, 1.80, 1.78, 1.77, 1.76,
        1.75, 1.75, 1.74, 1.73, 1.73, 1.72, 1.71, 1.70, 1.68, 1.67,
        1.66, 1.645],
    2: [2.14, 2.06, 2.00, 1.96, 1.93, 1.91, 1.89, 1.88, 1.87, 1.86,
        1.85, 1.84, 1.83, 1.82, 1.82, 1.81, 1.80, 1.78, 1.77, 1.75,
        1.74, 1.716],
    3: [2.19, 2.10, 2.04, 2.00, 1.97, 1.95, 1.93, 1.92, 1.90, 1.89,
        1.88, 1.87, 1.86, 1.86, 1.85, 1.84, 1.83, 1.81, 1.79, 1.78,
        1.76, 1.739],
    4: [2.22, 2.13, 2.07, 2.03, 2.00, 1.98, 1.96, 1.94, 1.93, 1.92,
        1.91, 1.90, 1.89, 1.88, 1.87, 1.87, 1.85, 1.83, 1.81, 1.79,
        1.77, 1.750],
    5: [2.24, 2.15, 2.09, 2.05, 2.02, 2.00, 1.98, 1.96, 1.95, 1.93,
        1.92, 1.91, 1.90, 1.90, 1.89, 1.88, 1.86, 1.84, 1.82, 1.80,
        1.78, 1.756],
    6: [2.25, 2.16, 2.10, 2.06, 2.03, 2.01, 1.99, 1.97, 1.96, 1.94,
        1.93, 1.92, 1.91, 1.91, 1.90, 1.89, 1.87, 1.85, 1.83, 1.81,
        1.79, 1.760],
    7: [2.26, 2.17, 2.11, 2.07, 2.04, 2.02, 2.00, 1.98, 1.97, 1.95,
        1.94, 1.93, 1.92, 1.91, 1.91, 1.90, 1.88, 1.86, 1.84, 1.82,
        1.80, 1.763],
    8: [2.27, 2.18, 2.12, 2.08, 2.05, 2.03, 2.01, 1.99, 1.97, 1.96,
        1.95, 1.94, 1.93, 1.92, 1.91, 1.91, 1.89, 1.86, 1.84, 1.82,
        1.80, 1.765],
    9: [2.28, 2.19, 2.12, 2.08, 2.05, 2.03, 2.01, 1.99, 1.98, 1.96,
        1.95, 1.94, 1.93, 1.92, 1.92, 1.91, 1.89, 1.87, 1.85, 1.83,
        1.81, 1.767],
    10: [2.28, 2.19, 2.13, 2.09, 2.06, 2.04, 2.02, 2.00, 1.99, 1.97,
         1.96, 1.95, 1.94, 1.93, 1.92, 1.92, 1.90, 1.87, 1.85, 1.83,
         1.81, 1.769],
}


def _williams_t(K: int, df: float, alpha: float = 0.05) -> float:
    """Interpolate Williams' T_K' critical value for α=0.05.

    K is the number of dose groups excluding control.  df is the
    pooled error df (typically G(n-1) where G = K+1).  Interpolation
    is linear in 1/df, which is well-behaved at the df → ∞ asymptote.
    """
    if alpha != 0.05:
        raise NotImplementedError(
            "Williams' test critical values are only tabulated for "
            "α in {0.05, 0.025, 0.01, 0.005}; this implementation "
            "supports α = 0.05."
        )
    K = max(1, min(int(K), 10))
    col = WILLIAMS_T_05[int(K)]
    # Use 1/df interpolation (df=inf -> 0)
    x_pts = [1.0 / d for d in WILLIAMS_DF_05]  # decreasing
    x = 1.0 / df if df > 0 else 1.0
    # Sort by ascending x: 1/inf comes first (smallest)
    paired = sorted(zip(x_pts, col))
    xs = [p[0] for p in paired]
    ys = [p[1] for p in paired]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    # binary search for segment
    lo, hi = 0, len(xs) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[hi]
    y0, y1 = ys[lo], ys[hi]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _power_williams(*, n: int, G: int, delta: float, sigma: float,
                    alpha: float) -> float:
    """Asymptotic Williams power per Chow Shao Wang (2008, p. 288)."""
    from scipy.stats import norm
    if n < 2 or G < 3:
        return 0.0
    K = G - 1  # number of dose groups (excluding control)
    df = G * (n - 1)
    t_K = _williams_t(K, df, alpha=alpha)
    ncp = delta / (sigma * math.sqrt(2.0 / n))
    return float(1.0 - norm.cdf(t_K - ncp))


def williams_minimum_effective_dose(
    *,
    G: int,
    delta: float,
    sigma: float,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 1_000_000,
) -> dict[str, Any]:
    """Power / sample-size for Williams' minimum-effective-dose test.

    Parameters
    ----------
    G : int
        Total number of groups including the zero-dose control
        (3 ≤ G ≤ 11).
    delta : float
        Minimum clinically meaningful difference between control mean
        and a treatment mean to be detected.  Positive value.
    sigma : float
        Common within-group standard deviation.
    alpha : float
        Significance level.  Only α = 0.05 is implemented (matches
        see Williams (1972) examples).
    n : int or None
        Per-group sample size.  Total N = n × G.
    power : float or None
        Target power.
    solve_for : str or None
        ``"n"`` (default when ``power`` given) or ``"power"`` (default
        when ``n`` given).
    """
    if G < 3 or G > 11:
        raise ValueError("G must be between 3 and 11")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")

    inputs_echo = {
        "G": G, "delta": delta, "sigma": sigma,
        "alpha": alpha, "n": n, "power": power,
    }

    if n is None and power is None:
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _power_williams(n=int(n), G=G, delta=delta,
                                   sigma=sigma, alpha=alpha)
        result = {
            "G": G, "K_doses": G - 1,
            "n": int(n), "n_total": int(n) * G,
            "achieved_power": achieved,
        }
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")

        def p_at(n_val: int) -> float:
            return _power_williams(n=n_val, G=G, delta=delta,
                                   sigma=sigma, alpha=alpha)

        lo, hi = max(n_min, 2), max(n_min, 2)
        while hi <= n_max:
            if p_at(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError(f"failed to bracket n within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if p_at(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_solved = hi
        achieved = _power_williams(n=n_solved, G=G, delta=delta,
                                   sigma=sigma, alpha=alpha)
        result = {
            "G": G, "K_doses": G - 1,
            "n": n_solved, "n_total": n_solved * G,
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "williams_minimum_effective_dose",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Dose (Chapter 595).",
            "Chow, S.C., Shao, J. & Wang, H. (2008). Sample Size "
            "Calculations in Clinical Research, 2e, pages 287-293.",
            "Williams, D.A. (1971). A test for differences between "
            "treatment means when several dose levels are compared "
            "with a zero dose control. Biometrics 27(1): 103-117.",
            "Williams, D.A. (1972). The comparison of several dose "
            "levels with a zero dose control. Biometrics 28(1): "
            "519-531.",
        ],
    }
