"""Coefficient alpha (Cronbach's alpha) sample-size routines.

Currently implements the confidence-interval procedure following
chapter 818 ("Confidence Intervals for Coefficient Alpha").  Companion
hypothesis-test routines (`tests_one_coefficient_alpha`,
`tests_two_coefficient_alphas`) live elsewhere or will be appended here
by their owners; do not refactor existing definitions when extending
this module.

Reference
---------
Feldt, L. S., Woodruff, D. J., and Salih, F. A. (1987). 'Statistical
    Inference for Coefficient Alpha.' Applied Psychological Measurement,
    11(1), 93-103.
Bonett, D. G. (2002). 'Sample Size Requirements for Testing and
    Estimating Coefficient Alpha.' Journal of Educational and Behavioral
    Statistics, 27(4), 335-340.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import f as fdist


def _alpha_ci_limits(ca: float, n: int, k: int, alpha: float, sides: int,
                     bound: str) -> tuple[float | None, float | None]:
    """Two-sided LCL/UCL or one-sided bound per Feldt et al. (1987)."""
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    if sides == 2:
        # CA_L uses F_{1 - alpha/2}; CA_U uses F_{alpha/2}.
        ca_l = 1.0 - (1.0 - ca) * fdist.ppf(1.0 - alpha / 2.0, df1, df2)
        ca_u = 1.0 - (1.0 - ca) * fdist.ppf(alpha / 2.0, df1, df2)
        return ca_l, ca_u
    if sides == 1:
        if bound == "upper":
            ca_u = 1.0 - (1.0 - ca) * fdist.ppf(alpha, df1, df2)
            return None, ca_u
        if bound == "lower":
            ca_l = 1.0 - (1.0 - ca) * fdist.ppf(1.0 - alpha, df1, df2)
            return ca_l, None
        raise ValueError("bound must be 'upper' or 'lower'")
    raise ValueError("sides must be 1 or 2")


def _alpha_width(ca: float, n: int, k: int, alpha: float, sides: int,
                 bound: str) -> float:
    lcl, ucl = _alpha_ci_limits(ca, n, k, alpha, sides, bound)
    if sides == 2:
        return float(ucl - lcl)  # type: ignore[operator]
    if bound == "upper":
        return float(ucl - ca)  # type: ignore[operator]
    return float(ca - lcl)  # type: ignore[operator]


def ci_coefficient_alpha(
    *,
    ca: float,
    k: int,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    interval_side: str = "upper",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Sample size / interval for a CI on Cronbach's coefficient alpha.

    Parameters
    ----------
    ca : float
        Planning value of the sample coefficient alpha (in (0, 1)).
    k : int
        Number of items per subject (must be >= 2; the procedure requires > 2 in
        practice, but k == 2 is well-defined and returned without error).
    alpha : float
        Confidence level is ``1 - alpha``.
    width : float, optional
        Target two-sided width (UCL - LCL).  Required when solving for ``n``
        with ``sides == 2``.
    distance : float, optional
        Target one-sided distance from CA to the limit.
    n : int, optional
        Number of subjects.  Required when solving for ``width``/``distance``.
    sides : int
        ``2`` (default) or ``1``.
    interval_side : {'upper', 'lower'}
        Bound side for one-sided intervals.
    solve_for : {'n', 'width'}
        Defaults to ``'n'`` if ``n`` is None.
    """
    if not 0.0 < ca < 1.0:
        raise ValueError("ca must be in (0, 1)")
    if k < 2:
        raise ValueError("k must be >= 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    target = width if sides == 2 else distance
    inputs_echo = {
        "ca": ca, "k": k, "alpha": alpha, "width": width,
        "distance": distance, "n": n, "sides": sides,
        "interval_side": interval_side,
    }

    if solve_for is None:
        solve_for = "n" if n is None else "width"

    if solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                "supply a positive `width` (sides=2) or `distance` (sides=1)"
            )

        def predicate(nn: int) -> bool:
            return _alpha_width(ca, nn, k, alpha, sides, interval_side) <= target

        lo, hi = 2, 2
        n_max = 10_000_000
        while hi <= n_max:
            if predicate(hi):
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError(f"failed to bracket n within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if predicate(mid):
                hi = mid
            else:
                lo = mid
        n_req = hi
    elif solve_for == "width":
        if n is None or n < 2:
            raise ValueError("supply n >= 2 when solving for width")
        n_req = int(n)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    lcl, ucl = _alpha_ci_limits(ca, n_req, k, alpha, sides, interval_side)
    if sides == 2:
        achieved_width = float(ucl - lcl)  # type: ignore[operator]
        achieved_distance = None
    else:
        achieved_width = None
        achieved_distance = (
            float(ucl - ca) if interval_side == "upper" else float(ca - lcl)  # type: ignore[operator]
        )

    return {
        "method_id": "ci_coefficient_alpha",
        "solve_for": solve_for,
        "n": n_req,
        "k": int(k),
        "ca": ca,
        "lower_limit": lcl,
        "upper_limit": ucl,
        "achieved_width": achieved_width,
        "achieved_distance": achieved_distance,
        "inputs_echo": inputs_echo,
        "citations": [
            "Coefficient Alpha.",
            "Feldt, L. S., Woodruff, D. J., and Salih, F. A. (1987). "
            "Applied Psychological Measurement, 11(1), 93-103.",
            "Bonett, D. G. (2002). Journal of Educational and Behavioral "
            "Statistics, 27(4), 335-340.",
        ],
    }


# ---------------------------------------------------------------------------
# Tests for One Coefficient Alpha


def _one_alpha_direction(ca0: float, ca1: float, sides: int) -> int:
    """Return -1 (CA1<CA0), +1 (CA1>CA0), or 0 (two-sided)."""
    if sides == 2:
        return 0
    if sides == 1:
        if ca1 == ca0:
            raise ValueError("ca1 must differ from ca0 for a one-sided test")
        return -1 if ca1 < ca0 else +1
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def _one_alpha_power(ca0: float, ca1: float, n: int, k: int,
                     alpha: float, sides: int) -> float:
    """Feldt (1965) F-test: W = (1 - rho0)/(1 - rho_hat) ~ F(N-1, (N-1)(K-1)).

    Under H1, the same statistic times (1-rho1)/(1-rho0) is central F.
    """
    df1 = n - 1
    df2 = (k - 1) * (n - 1)
    if df1 <= 0 or df2 <= 0:
        return 0.0

    direction = _one_alpha_direction(ca0, ca1, sides)

    if direction == 0:
        # Two-sided: reject when W <= F_{alpha/2} or W >= F_{1-alpha/2}.
        scale = (1.0 - ca1) / (1.0 - ca0)
        c_lo = fdist.ppf(alpha / 2.0, df1, df2)
        c_hi = fdist.ppf(1.0 - alpha / 2.0, df1, df2)
        lower = fdist.cdf(c_lo * scale, df1, df2)
        upper = 1.0 - fdist.cdf(c_hi * scale, df1, df2)
        return float(lower + upper)

    if direction == +1:
        # H1: CA1 > CA0 means 1 - rho_hat is small => W is small => reject when W <= F_alpha.
        scale = (1.0 - ca1) / (1.0 - ca0)
        c = fdist.ppf(alpha, df1, df2)
        return float(fdist.cdf(c / scale, df1, df2))

    # direction == -1: H1: CA1 < CA0 -> W large -> reject when W >= F_{1-alpha}.
    scale = (1.0 - ca1) / (1.0 - ca0)
    c = fdist.ppf(1.0 - alpha, df1, df2)
    return float(1.0 - fdist.cdf(c / scale, df1, df2))


def _one_alpha_n(ca0: float, ca1: float, k: int, alpha: float, power: float,
                 sides: int, n_min: int = 3, n_max: int = 10_000_000
                 ) -> tuple[int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if ca0 == ca1:
        raise ValueError("ca0 and ca1 must differ to solve for N")
    lo = max(n_min, 3)
    hi = lo
    while hi <= n_max:
        if _one_alpha_power(ca0, ca1, hi, k, alpha, sides) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _one_alpha_power(ca0, ca1, mid, k, alpha, sides) >= power:
            hi = mid
        else:
            lo = mid
    return hi, _one_alpha_power(ca0, ca1, hi, k, alpha, sides)


def tests_one_coefficient_alpha(
    *,
    ca0: float,
    ca1: float,
    k: int,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Feldt (1965) F-test for one Cronbach's coefficient alpha.

    Tests H0: rho = ca0 versus the supplied alternative.  Provide exactly
    one of ``n`` or ``power``; the other is solved for.

    Parameters
    ----------
    ca0 : float
        Coefficient alpha under the null hypothesis (in (-1, 1)).
    ca1 : float
        Coefficient alpha under the alternative — value at which power
        is evaluated (in (-1, 1)).
    k : int
        Number of items / raters in each subject (>= 2).
    alpha : float
        Significance level (in (0, 1)).
    sides : int
        ``2`` (default) or ``1``; one-sided direction is taken from the
        sign of ``ca1 - ca0``.

    Notes
    -----
    The F-statistic ``W = (1 - ca0)/(1 - alpha_hat)`` is distributed as
    ``F(N - 1, (N - 1)(K - 1))`` under H0 (Feldt 1965).
    """
    inputs_echo = {
        "ca0": ca0, "ca1": ca1, "k": k, "alpha": alpha,
        "power": power, "n": n, "sides": sides,
    }
    if not -1.0 < ca0 < 1.0:
        raise ValueError("ca0 must be in (-1, 1)")
    if not -1.0 < ca1 < 1.0:
        raise ValueError("ca1 must be in (-1, 1)")
    if k < 2:
        raise ValueError("k must be >= 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    if solve_for is None:
        if n is None and power is not None:
            solve_for = "n"
        elif power is None and n is not None:
            solve_for = "power"
        else:
            raise ValueError("supply exactly one of (n, power)")

    if solve_for == "power":
        assert n is not None
        achieved = _one_alpha_power(ca0, ca1, n, k, alpha, sides)
        result = {"n": int(n), "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        n_req, achieved = _one_alpha_n(ca0, ca1, k, alpha, power, sides)
        result = {"n": n_req, "achieved_power": achieved}
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "tests_one_coefficient_alpha",
        "solve_for": solve_for,
        **result,
        "k": int(k),
        "inputs_echo": inputs_echo,
        "citations": [
            "Feldt, L. S. (1965). 'The Approximate Sampling Distribution of "
            "Kuder-Richardson Reliability Coefficient Twenty.' Psychometrika, "
            "30(3), 357-370.",
            "Feldt, L. S., Woodruff, D. J., and Salih, F. A. (1987). "
            "'Statistical Inference for Coefficient Alpha.' Applied "
            "Psychological Measurement, 11(1), 93-103.",
            "Bonett, D. G. (2002). 'Sample Size Requirements for Testing and "
            "Estimating Coefficient Alpha.' Journal of Educational and "
            "Behavioral Statistics, 27(4), 335-340.",
        ],
    }


# ---------------------------------------------------------------------------
# Tests for Two Coefficient Alphas


def _two_alpha_df(n1: int, n2: int, k1: int, k2: int, phi: float
                  ) -> tuple[float, float]:
    """Feldt & Brennan (1989) / Feldt et al. (1999) degrees of freedom."""
    c1 = (n1 - 1) * (k1 - 1)
    c2 = (n2 - 1) * (k2 - 1)
    dependent = phi != 0.0
    # Large-sample shortcut (Feldt et al. 1999).
    if c1 > 1000 and c2 > 1000 and k1 > 25 and k2 > 25:
        if dependent:
            d = (n1 - 1 - 7.0 * phi * phi) / (1.0 - phi * phi)
            return float(d), float(d)
        return float(n1 - 1), float(n2 - 1)
    # General formula.
    A = (c1 * (n2 - 1)) / ((c1 - 2) * (n2 - 3))
    B = (
        (n1 + 1) * (n2 - 1) ** 2 * (c2 + 2) * c1 * c1
    ) / (
        (n2 - 3) * (n2 - 5) * (n1 - 1) * (c1 - 2) * (c1 - 4) * c2
    )
    if not dependent:
        nu1 = (2.0 * A * A) / (2.0 * B - A * B - A * A)
        nu2 = (2.0 * A) / (A - 1.0)
        return float(nu1), float(nu2)
    # Dependent case (phi != 0).
    M = A - (2.0 * phi * phi) / (n1 - 1)
    V = B - A * A - (4.0 * phi * phi) / (n1 - 1)
    nu1 = (2.0 * M * M) / (V * (2.0 - M) - M * M * (M - 1.0))
    nu2 = (2.0 * M) / (M - 1.0)
    return float(nu1), float(nu2)


def _two_alpha_power(ca1: float, ca20: float, ca21: float,
                     n1: int, n2: int, k1: int, k2: int,
                     alpha: float, sides: int, phi: float) -> float:
    """Feldt et al. (1999) F-test power for two coefficient alphas.

    Under H0 (rho_2 = rho_1 = ca20), the observed statistic
        F* = (1 - alpha_hat_2)/(1 - alpha_hat_1)
    is approximately central F with (nu1, nu2) d.f.; under H1 the scaled
    statistic ``F* * (1 - rho_1)/(1 - rho_2)`` is central F.
    """
    if phi != 0.0:
        n2 = n1  # dependent design forces equal N
    if n1 < 6 or n2 < 6:
        # General df formulas require (n-3) and (n-5) positive.
        return 0.0
    if k1 < 2 or k2 < 2:
        return 0.0

    nu1, nu2 = _two_alpha_df(n1, n2, k1, k2, phi)

    if sides == 1:
        # Direction implied by ca21 vs ca20.
        if ca21 == ca20:
            raise ValueError("ca21 must differ from ca20 for a one-sided test")
        if ca21 > ca20:
            # H1: rho_2 > rho_1 (numerator small) -> reject when F* <= F_alpha.
            c = fdist.ppf(alpha, nu1, nu2)
            scale = (1.0 - ca1) / (1.0 - ca21)
            return float(fdist.cdf(c * scale, nu1, nu2))
        # ca21 < ca20: reject when F* >= F_{1-alpha}.
        c = fdist.ppf(1.0 - alpha, nu1, nu2)
        scale = (1.0 - ca1) / (1.0 - ca21)
        return float(1.0 - fdist.cdf(c * scale, nu1, nu2))

    # Two-sided: reject when F* <= F_{alpha/2} or F* >= F_{1-alpha/2}.
    scale = (1.0 - ca21) / (1.0 - ca20)
    c_lo = fdist.ppf(alpha / 2.0, nu1, nu2)
    c_hi = fdist.ppf(1.0 - alpha / 2.0, nu1, nu2)
    lower = fdist.cdf(c_lo * scale, nu1, nu2)
    upper = 1.0 - fdist.cdf(c_hi * scale, nu1, nu2)
    return float(lower + upper)


def _two_alpha_n(ca1: float, ca20: float, ca21: float, k1: int, k2: int,
                 alpha: float, power: float, sides: int, phi: float,
                 allocation: float = 1.0,
                 n_min: int = 6, n_max: int = 10_000_000
                 ) -> tuple[int, int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if ca20 == ca21:
        raise ValueError("ca20 and ca21 must differ to solve for N")
    if allocation <= 0:
        raise ValueError("allocation (n2/n1) must be > 0")

    def n2_for(n1v: int) -> int:
        if phi != 0.0:
            return n1v
        return max(n_min, math.ceil(allocation * n1v))

    def p_at(n1v: int) -> float:
        return _two_alpha_power(ca1, ca20, ca21, n1v, n2_for(n1v),
                                k1, k2, alpha, sides, phi)

    lo = max(n_min, 6)
    hi = lo
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
    n1r = hi
    n2r = n2_for(n1r)
    return n1r, n2r, p_at(n1r)


def tests_two_coefficient_alphas(
    *,
    ca1: float,
    ca20: float,
    ca21: float,
    k1: int,
    k2: int,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    phi: float = 0.0,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Feldt et al. (1999) test comparing two Cronbach's alphas.

    Either supply ``power`` to solve for the per-set sample size, or
    supply ``n1`` (and optionally ``n2``) to obtain the achieved power.
    When ``phi != 0`` the design is dependent and ``n2`` is forced equal
    to ``n1``.

    Parameters
    ----------
    ca1 : float
        Coefficient alpha for set 1 (CA1).
    ca20 : float
        Coefficient alpha for set 2 under H0 (CA20).
    ca21 : float
        Coefficient alpha for set 2 under H1 — power is computed here.
    k1, k2 : int
        Number of items/raters in each scale (>= 2).
    sides : int
        ``2`` (default) or ``1``; one-sided direction taken from
        sign of ``ca21 - ca20``.
    phi : float
        Correlation between dataset average scores; 0 selects the
        independent case.
    allocation : float
        Target ``n2 / n1`` ratio when solving for N (independent case).
    """
    inputs_echo = {
        "ca1": ca1, "ca20": ca20, "ca21": ca21, "k1": k1, "k2": k2,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2,
        "sides": sides, "phi": phi, "allocation": allocation,
    }
    for v, name in ((ca1, "ca1"), (ca20, "ca20"), (ca21, "ca21")):
        if not -1.0 < v < 1.0:
            raise ValueError(f"{name} must be in (-1, 1)")
    if k1 < 2 or k2 < 2:
        raise ValueError("k1 and k2 must be >= 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if not -1.0 < phi < 1.0:
        raise ValueError("phi must be in (-1, 1)")

    if phi != 0.0:
        # Dependent: force n2 == n1.
        if n1 is not None:
            n2 = n1
        elif n2 is not None:
            n1 = n2
    else:
        if n1 is not None and n2 is None:
            n2 = max(6, math.ceil(allocation * n1))
        elif n2 is not None and n1 is None:
            n1 = max(6, math.ceil(n2 / allocation))

    have_n = n1 is not None and n2 is not None
    have_power = power is not None

    if solve_for is None:
        if have_n and not have_power:
            solve_for = "power"
        elif have_power and not have_n:
            solve_for = "n"
        else:
            raise ValueError("supply exactly one of (n, power)")

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _two_alpha_power(ca1, ca20, ca21, n1, n2, k1, k2,
                                    alpha, sides, phi)
        result = {
            "n1": int(n1), "n2": int(n2), "n": int(n1) + int(n2),
            "achieved_power": achieved,
        }
    elif solve_for == "n":
        assert power is not None
        n1r, n2r, achieved = _two_alpha_n(
            ca1, ca20, ca21, k1, k2, alpha, power, sides, phi,
            allocation=allocation,
        )
        result = {
            "n1": n1r, "n2": n2r, "n": n1r + n2r,
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "tests_two_coefficient_alphas",
        "solve_for": solve_for,
        **result,
        "k1": int(k1),
        "k2": int(k2),
        "inputs_echo": inputs_echo,
        "citations": [
            "Feldt, L. S., and Brennan, R. L. (1989). 'Reliability.' In "
            "R. L. Linn (Ed.), Educational Measurement (3rd ed.), 105-146.",
            "Feldt, L. S., Woodruff, D. J., and Salih, F. A. (1987). "
            "Applied Psychological Measurement, 11(1), 93-103.",
            "Feldt, L. S., and Ankenmann, R. D. (1999). 'Determining Sample "
            "Size for a Test of the Equality of Alpha Coefficients When the "
            "Number of Part-Tests is Small.' Psychological Methods, 4(4), "
            "366-377.",
        ],
    }
