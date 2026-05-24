"""Higher-Order Cross-Over Designs - Difference of Two Means.

Implements four higher-order cross-over design procedures:

  * Chapter 527 - Tests for the Difference of Two Means in a
    Higher-Order Cross-Over Design (inequality / "Tests"):
    ``tests_higher_order_cross_over_diff``
  * Chapter 530 - Non-Inferiority Tests for the Difference of Two Means
    in a Higher-Order Cross-Over Design:
    ``non_inferiority_higher_order_cross_over_diff``
  * Chapter 540 - Equivalence Tests for the Difference of Two Means in a
    Higher-Order Cross-Over Design (TOST):
    ``equivalence_higher_order_cross_over_diff``
  * Chapter 528 - Superiority by a Margin Tests for the Difference of
    Two Means in a Higher-Order Cross-Over Design:
    ``superiority_by_margin_higher_order_cross_over_diff``

All four chapters share the same higher-order cross-over machinery
described by Chen, Chow & Li (1997).  The key quantity is

    se = sigma_w * sqrt(b / n_avg)

where ``n_avg`` is the average number of subjects per sequence
(N / S), and ``(V, b)`` is a *design constant* pair:

    Balaam (2x4: AA|BB|AB|BA)              V = 4n - 3,  b = 2
    Two-Sequence Dual (3x2: ABB|BAA)       V = 4n - 4,  b = 3/4
    Four-Period Two-Sequence (4x2)         V = 6n - 5,  b = 11/20
    Four-Period Four-Sequence (4x4)        V = 12n - 5, b = 1/4

Power follows the shifted central-t formulation of Chen, Chow & Li
(1997) - this matches every
example used for validation to four decimals.

Two-sided inequality (Chapter 527):
    Power = T_V( arg - t_{V, 1-alpha/2} )
            with arg = |mu1 - mu2| / se

One-sided NI / Sup (Chapter 530, 528):
    higher better:   Power = T_V( (delta - eps)/se - t_{V, 1-alpha} )
    lower  better:   Power = 1 - T_V( t_{V, 1-alpha} - (eps - delta)/se )

TOST equivalence (Chapter 540):
    Power = T_V( (EU - delta)/se - tc ) - T_V( tc - (delta - EL)/se )
            with tc = t_{V, 1-alpha}
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# ---- design parameters -----------------------------------------------------

# Canonical design name -> (S, V_of_n, b, label)
# n is the average subjects per sequence (n_avg = N / S).
_DESIGNS: dict[str, tuple[int, "callable[[float], float]", float, str]] = {
    "balaam": (4, (lambda n: 4 * n - 3), 2.0,
               "2x4 Balaam: AA|BB|AB|BA"),
    "two_sequence_dual": (2, (lambda n: 4 * n - 4), 3.0 / 4.0,
                          "3x2 Two-Sequence Dual: ABB|BAA"),
    "four_period_two_sequence": (2, (lambda n: 6 * n - 5), 11.0 / 20.0,
                                 "4x2 Four-Period, Two-Sequence: ABBA|BAAB"),
    "four_period_four_sequence": (4, (lambda n: 12 * n - 5), 1.0 / 4.0,
                                  "4x4 Four-Period, Four-Sequence"),
}

# Friendly aliases the user may pass in.
_DESIGN_ALIASES: dict[str, str] = {
    # Balaam
    "balaam": "balaam",
    "2x4": "balaam",
    "2x4_balaam": "balaam",
    "aa|bb|ab|ba": "balaam",
    # Two-Sequence Dual
    "two_sequence_dual": "two_sequence_dual",
    "dual": "two_sequence_dual",
    "3x2": "two_sequence_dual",
    "3x2_dual": "two_sequence_dual",
    "abb|baa": "two_sequence_dual",
    # Four-Period, Two-Sequence
    "four_period_two_sequence": "four_period_two_sequence",
    "4x2": "four_period_two_sequence",
    "abba|baab": "four_period_two_sequence",
    # Four-Period, Four-Sequence
    "four_period_four_sequence": "four_period_four_sequence",
    "4x4": "four_period_four_sequence",
    "williams_4": "four_period_four_sequence",
}


def _resolve_design(
    design: str | None,
    n_sequences: int | None,
    n_periods: int | None,
    efficiency: float | None,
) -> tuple[int, "callable[[float], float]", float, str]:
    """Return (S, V_of_n, b, label) for the requested higher-order design.

    Either ``design`` is a recognised string keyword, or all three of
    ``n_sequences``, ``n_periods`` and ``efficiency`` are provided to
    describe a custom design.  In the custom case the degrees-of-freedom
    formula follows the convention

        V(n) = N * (P - 1) - (S - 1) - 1
             = n * S * (P - 1) - S

    where ``n`` is the average subjects per sequence.  ``efficiency``
    plays the role of ``b`` (so ``b = efficiency``).
    """
    if design is not None:
        key = design.strip().lower().replace(" ", "_").replace("-", "_")
        # also strip a possible "design" prefix
        if key in _DESIGN_ALIASES:
            canon = _DESIGN_ALIASES[key]
            return _DESIGNS[canon]
        # try one-pass strip of common decorations
        key2 = key.replace("design", "").strip("_")
        if key2 in _DESIGN_ALIASES:
            canon = _DESIGN_ALIASES[key2]
            return _DESIGNS[canon]
        raise ValueError(
            f"unknown design {design!r}; supported: "
            "balaam, two_sequence_dual, four_period_two_sequence, "
            "four_period_four_sequence (aliases 2x4, 3x2, 4x2, 4x4)"
        )

    # Custom design via (S, P, efficiency)
    if n_sequences is None or n_periods is None or efficiency is None:
        raise ValueError(
            "supply either `design` or all of `n_sequences`, `n_periods`, "
            "`efficiency`"
        )
    if n_sequences < 1:
        raise ValueError("n_sequences must be >= 1")
    if n_periods < 2:
        raise ValueError("n_periods must be >= 2")
    if efficiency <= 0:
        raise ValueError("efficiency must be positive")
    S = int(n_sequences)
    P = int(n_periods)
    b = float(efficiency)

    def _Vfn(n: float, _S: int = S, _P: int = P) -> float:
        return n * _S * (_P - 1) - _S

    label = f"custom S={S} periods={P} efficiency={b}"
    return (S, _Vfn, b, label)


# ---- core power expressions -----------------------------------------------


def _se_at_n(sd_w: float, b: float, n_avg: float) -> float:
    return sd_w * math.sqrt(b / n_avg)


def _power_inequality(
    *, diff: float, sd_w: float, n_total: int,
    S: int, Vfn, b: float, alpha: float, sides: int,
) -> float:
    """Shifted central-t power for Chapter 527 inequality tests."""
    if n_total < S + 1:
        return 0.0
    n_avg = n_total / S
    V = Vfn(n_avg)
    if V <= 0:
        return 0.0
    se = _se_at_n(sd_w, b, n_avg)
    if se <= 0:
        return 0.0
    arg = abs(diff) / se
    if sides == 2:
        tc = D.t_ppf(1.0 - alpha / 2.0, V)
        # matches every two-sided example to >=4 decimals (the opposite
        # tail is negligible for non-trivial alternatives).
        return _stdt(arg - tc, V)
    elif sides == 1:
        tc = D.t_ppf(1.0 - alpha, V)
        return _stdt(arg - tc, V)
    else:
        raise ValueError(f"sides must be 1 or 2, got {sides}")


def _stdt(x: float, df: float) -> float:
    """Central-t CDF via the noncentral-t with ncp=0 (scipy backend)."""
    return D.nct_cdf(x, df, 0.0)


def _power_directional(
    *, delta: float, eps: float, sd_w: float, n_total: int,
    S: int, Vfn, b: float, alpha: float, higher_better: bool,
) -> float:
    """Power for the NI / Sup directional tests (Chapters 528, 530).

    ``eps`` is the *signed* margin entering the hypothesis
    (Diff > eps when ``higher_better=True``, Diff < eps otherwise).
    For NI:  eps = -|NIM| (higher better) or +|NIM| (lower better).
    For Sup: eps = +|SM|  (higher better) or -|SM|  (lower better).
    """
    if n_total < S + 1:
        return 0.0
    n_avg = n_total / S
    V = Vfn(n_avg)
    if V <= 0:
        return 0.0
    se = _se_at_n(sd_w, b, n_avg)
    if se <= 0:
        return 0.0
    tc = D.t_ppf(1.0 - alpha, V)
    if higher_better:
        arg = (delta - eps) / se
        return _stdt(arg - tc, V)
    else:
        arg = (eps - delta) / se
        return 1.0 - _stdt(tc - arg, V)


def _power_equivalence(
    *, delta: float, EL: float, EU: float, sd_w: float, n_total: int,
    S: int, Vfn, b: float, alpha: float,
) -> float:
    """TOST power for the equivalence chapter (540)."""
    if n_total < S + 1:
        return 0.0
    if not (EL < delta < EU):
        # Outside the equivalence region the formula is mathematically
        # valid but power is essentially 0.
        pass
    n_avg = n_total / S
    V = Vfn(n_avg)
    if V <= 0:
        return 0.0
    se = _se_at_n(sd_w, b, n_avg)
    if se <= 0:
        return 0.0
    tc = D.t_ppf(1.0 - alpha, V)
    upper = _stdt((EU - delta) / se - tc, V)
    lower = _stdt(tc - (delta - EL) / se, V)
    return max(0.0, upper - lower)


# ---- N-search helpers ------------------------------------------------------


def _next_multiple_of(n: int, m: int) -> int:
    if m <= 1:
        return n
    return ((n + m - 1) // m) * m


def _solve_n(
    power_fn,  # callable(n_total) -> float
    *,
    S: int,
    target_power: float,
    equal_per_sequence: bool = True,
    n_min: int = 0,
    n_max: int = 1_000_000,
) -> tuple[int, float]:
    """Smallest N >= n_min achieving >= target_power.

    If ``equal_per_sequence`` is True ("Equal Per Sequence" mode),
    N is constrained to be a multiple of ``S``.
    Otherwise "Exact" mode is used, which searches one subject at a time.
    """
    if not 0.0 < target_power < 1.0:
        raise ValueError("power must be in (0, 1)")

    step = S if equal_per_sequence else 1
    lo = max(n_min, S + 1)
    lo = _next_multiple_of(lo, step)
    hi = lo
    while hi <= n_max:
        p = power_fn(hi)
        if p >= target_power:
            break
        lo = hi
        hi = max(hi + step, hi * 2)
        hi = _next_multiple_of(hi, step)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")

    while lo + step < hi:
        mid = (lo + hi) // 2
        mid = _next_multiple_of(mid, step)
        if mid <= lo:
            mid = lo + step
        if mid >= hi:
            break
        p = power_fn(mid)
        if p >= target_power:
            hi = mid
        else:
            lo = mid

    return hi, power_fn(hi)


# ===========================================================================
# 1) Tests for the Difference of Two Means (Chapter 527)
# ===========================================================================


def tests_higher_order_cross_over_diff(
    *,
    mean1: float | None = None,
    mean2: float | None = None,
    sd_w: float,
    design: str | None = None,
    n_sequences: int | None = None,
    n_periods: int | None = None,
    efficiency: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    equal_per_sequence: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Inequality test on the mean difference in a higher-order cross-over.

    Means in a Higher-Order Cross-Over Design" using the Chen, Chow &
    Li (1997) shifted central-t formulation.

    Inputs
    ------
    mean1, mean2
        Treatment means; the test concerns ``Diff1 = mean1 - mean2``
        (Diff0 = 0).  Provide both.
    sd_w
        Within-subject SD, ``sigma_w = sqrt(WMSE)``.
    design
        Higher-order design keyword - one of ``balaam``,
        ``two_sequence_dual`` (3x2 ABB|BAA), ``four_period_two_sequence``
        (4x2 ABBA|BAAB), ``four_period_four_sequence`` (4x4).  Common
        aliases (``2x4``, ``3x2``, ``4x2``, ``4x4``) are accepted.
    n_sequences, n_periods, efficiency
        Custom design specification.  ``efficiency`` is the ``b``
        constant entering ``se = sigma_w * sqrt(b/n_avg)``; ``V`` uses
        ``N*(P-1) - (S-1) - 1``.  Ignored when ``design`` is given.
    alpha
        Type-I error rate.
    power, n
        Provide exactly one; the other is solved for.  ``n`` is the
        total sample size across all sequences.
    sides
        1 or 2 (default 2).
    equal_per_sequence
        When solving for N, restrict to multiples of S ("Equal Per
        Sequence" mode); otherwise "Exact" mode.  Default True.
    """
    if mean1 is None or mean2 is None:
        raise ValueError("supply both mean1 and mean2")
    if sd_w <= 0:
        raise ValueError("sd_w must be positive")
    S, Vfn, b, label = _resolve_design(
        design, n_sequences, n_periods, efficiency
    )

    inputs_echo = {
        "mean1": mean1, "mean2": mean2, "sd_w": sd_w,
        "design": design, "n_sequences": n_sequences,
        "n_periods": n_periods, "efficiency": efficiency,
        "alpha": alpha, "power": power, "n": n, "sides": sides,
        "equal_per_sequence": equal_per_sequence,
    }

    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError(
            "supply exactly one of (power, n); leave the other None"
        )
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    diff = mean1 - mean2

    if solve_for == "power":
        assert n is not None
        achieved = _power_inequality(
            diff=diff, sd_w=sd_w, n_total=n,
            S=S, Vfn=Vfn, b=b, alpha=alpha, sides=sides,
        )
        n_used = n
    elif solve_for == "n":
        assert power is not None
        if diff == 0:
            raise ValueError("mean1 must differ from mean2 to solve for N")

        def _pwr(nt: int) -> float:
            return _power_inequality(
                diff=diff, sd_w=sd_w, n_total=nt,
                S=S, Vfn=Vfn, b=b, alpha=alpha, sides=sides,
            )

        n_used, achieved = _solve_n(
            _pwr, S=S, target_power=power,
            equal_per_sequence=equal_per_sequence,
        )
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "tests_higher_order_cross_over_diff",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // S if S > 0 else None,
        "n_sequences": S,
        "design_label": label,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Two Means in a Higher-Order Cross-Over Design.",
            "Chen, K.W.; Chow, S.C. & Li, G. (1997). A Note on Sample "
            "Size Determination for Bioequivalence Studies with "
            "Higher-Order Crossover Designs. J. Pharmacokin. Biopharm. "
            "25(6): 753-765.",
            "Chow, S.C. & Liu, J.P. (1999). Design and Analysis of "
            "Bioavailability and Bioequivalence Studies. Marcel Dekker.",
            "Chow, S.C.; Shao, J. & Wang, H. (2003). Sample Size "
            "Calculations in Clinical Research. Marcel Dekker.",
        ],
    }


# ===========================================================================
# 2) Non-Inferiority Tests (Chapter 530)
# ===========================================================================


def non_inferiority_higher_order_cross_over_diff(
    *,
    margin: float,
    diff: float = 0.0,
    sd_w: float,
    design: str | None = None,
    n_sequences: int | None = None,
    n_periods: int | None = None,
    efficiency: float | None = None,
    higher_means_better: bool = True,
    alpha: float = 0.025,
    power: float | None = None,
    n: int | None = None,
    equal_per_sequence: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for mean difference in a higher-order cross-over.


    Inputs
    ------
    margin
        Magnitude |NIM| of the non-inferiority margin (a positive
        number).  Internally signed by ``higher_means_better``.
    diff
        True mean difference ``D = mu_T - mu_R`` (default 0).
    sd_w
        Within-subject SD.
    design / n_sequences / n_periods / efficiency
        Design specification - see ``tests_higher_order_cross_over_diff``.
    higher_means_better
        If True, the alternative is ``D > -|NIM|`` (higher better,
        ``eps = -|NIM|``).  If False, ``D < +|NIM|`` (lower better).
    alpha
        One-sided type-I error rate (default 0.025).
    power, n
        Provide exactly one.
    """
    if margin <= 0:
        raise ValueError("margin (|NIM|) must be positive")
    if sd_w <= 0:
        raise ValueError("sd_w must be positive")
    S, Vfn, b, label = _resolve_design(
        design, n_sequences, n_periods, efficiency
    )

    eps = -margin if higher_means_better else margin

    inputs_echo = {
        "margin": margin, "diff": diff, "sd_w": sd_w,
        "design": design, "n_sequences": n_sequences,
        "n_periods": n_periods, "efficiency": efficiency,
        "higher_means_better": higher_means_better,
        "alpha": alpha, "power": power, "n": n,
        "equal_per_sequence": equal_per_sequence,
    }

    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError(
            "supply exactly one of (power, n); leave the other None"
        )
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _power_directional(
            delta=diff, eps=eps, sd_w=sd_w, n_total=n,
            S=S, Vfn=Vfn, b=b, alpha=alpha,
            higher_better=higher_means_better,
        )
        n_used = n
    elif solve_for == "n":
        assert power is not None

        def _pwr(nt: int) -> float:
            return _power_directional(
                delta=diff, eps=eps, sd_w=sd_w, n_total=nt,
                S=S, Vfn=Vfn, b=b, alpha=alpha,
                higher_better=higher_means_better,
            )

        n_used, achieved = _solve_n(
            _pwr, S=S, target_power=power,
            equal_per_sequence=equal_per_sequence,
        )
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "non_inferiority_higher_order_cross_over_diff",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // S if S > 0 else None,
        "n_sequences": S,
        "design_label": label,
        "signed_margin": eps,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Difference of Two Means in a Higher-Order Cross-Over Design.",
            "Chen, K.W.; Chow, S.C. & Li, G. (1997). A Note on Sample "
            "Size Determination for Bioequivalence Studies with "
            "Higher-Order Crossover Designs. J. Pharmacokin. Biopharm. "
            "25(6): 753-765.",
            "Chow, S.C. & Liu, J.P. (1999). Design and Analysis of "
            "Bioavailability and Bioequivalence Studies. Marcel Dekker.",
        ],
    }


# ===========================================================================
# 3) Equivalence Tests - TOST (Chapter 540)
# ===========================================================================


def equivalence_higher_order_cross_over_diff(
    *,
    lower_limit: float,
    upper_limit: float,
    diff: float = 0.0,
    sd_w: float,
    design: str | None = None,
    n_sequences: int | None = None,
    n_periods: int | None = None,
    efficiency: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    equal_per_sequence: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Equivalence (TOST) test on the difference of two means in a
    higher-order cross-over.


    Inputs
    ------
    lower_limit, upper_limit
        Equivalence limits ``(EL, EU)`` on the difference scale.  Must
        satisfy ``EL < 0 < EU`` and ``EL < diff < EU`` for non-zero power.
    diff
        True mean difference ``D = mu_T - mu_R`` (default 0).
    sd_w
        Within-subject SD.
    design / n_sequences / n_periods / efficiency
        Design specification.
    alpha
        Per one-sided test alpha (TOST).  Default 0.05.
    power, n
        Provide exactly one.
    """
    if not (lower_limit < 0):
        raise ValueError("lower_limit must be negative")
    if not (upper_limit > 0):
        raise ValueError("upper_limit must be positive")
    if sd_w <= 0:
        raise ValueError("sd_w must be positive")
    S, Vfn, b, label = _resolve_design(
        design, n_sequences, n_periods, efficiency
    )

    inputs_echo = {
        "lower_limit": lower_limit, "upper_limit": upper_limit,
        "diff": diff, "sd_w": sd_w,
        "design": design, "n_sequences": n_sequences,
        "n_periods": n_periods, "efficiency": efficiency,
        "alpha": alpha, "power": power, "n": n,
        "equal_per_sequence": equal_per_sequence,
    }

    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError(
            "supply exactly one of (power, n); leave the other None"
        )
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _power_equivalence(
            delta=diff, EL=lower_limit, EU=upper_limit,
            sd_w=sd_w, n_total=n,
            S=S, Vfn=Vfn, b=b, alpha=alpha,
        )
        n_used = n
    elif solve_for == "n":
        assert power is not None
        if not (lower_limit < diff < upper_limit):
            raise ValueError(
                "true diff must lie strictly inside (lower_limit, "
                "upper_limit) for sample-size calculation"
            )

        def _pwr(nt: int) -> float:
            return _power_equivalence(
                delta=diff, EL=lower_limit, EU=upper_limit,
                sd_w=sd_w, n_total=nt,
                S=S, Vfn=Vfn, b=b, alpha=alpha,
            )

        n_used, achieved = _solve_n(
            _pwr, S=S, target_power=power,
            equal_per_sequence=equal_per_sequence,
        )
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "equivalence_higher_order_cross_over_diff",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // S if S > 0 else None,
        "n_sequences": S,
        "design_label": label,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Difference of Two Means in a Higher-Order Cross-Over Design.",
            "Chen, K.W.; Chow, S.C. & Li, G. (1997). A Note on Sample "
            "Size Determination for Bioequivalence Studies with "
            "Higher-Order Crossover Designs. J. Pharmacokin. Biopharm. "
            "25(6): 753-765.",
            "Schuirmann, D.J. (1987). A comparison of the two one-sided "
            "tests procedure and the power approach for assessing the "
            "equivalence of average bioavailability. J. Pharmacokin. "
            "Biopharm. 15: 657-680.",
            "Chow, S.C. & Liu, J.P. (1999). Design and Analysis of "
            "Bioavailability and Bioequivalence Studies. Marcel Dekker.",
        ],
    }


# ===========================================================================
# 4) Superiority by a Margin (Chapter 528)
# ===========================================================================


def superiority_by_margin_higher_order_cross_over_diff(
    *,
    margin: float,
    diff: float,
    sd_w: float,
    design: str | None = None,
    n_sequences: int | None = None,
    n_periods: int | None = None,
    efficiency: float | None = None,
    higher_means_better: bool = True,
    alpha: float = 0.025,
    power: float | None = None,
    n: int | None = None,
    equal_per_sequence: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Superiority-by-a-margin test for mean difference in a higher-order
    cross-over.


    Inputs
    ------
    margin
        Magnitude |SM| of the superiority margin (a positive number).
    diff
        True mean difference ``D = mu_T - mu_R``.  Must exceed the
        margin in the appropriate direction.
    sd_w
        Within-subject SD.
    design / n_sequences / n_periods / efficiency
        Design specification.
    higher_means_better
        If True, alt: ``D > +|SM|`` (eps = +|SM|).  If False,
        alt: ``D < -|SM|`` (eps = -|SM|).
    alpha
        One-sided type-I error.  Default 0.025.
    power, n
        Provide exactly one.
    """
    if margin <= 0:
        raise ValueError("margin (|SM|) must be positive")
    if sd_w <= 0:
        raise ValueError("sd_w must be positive")
    S, Vfn, b, label = _resolve_design(
        design, n_sequences, n_periods, efficiency
    )

    eps = margin if higher_means_better else -margin

    inputs_echo = {
        "margin": margin, "diff": diff, "sd_w": sd_w,
        "design": design, "n_sequences": n_sequences,
        "n_periods": n_periods, "efficiency": efficiency,
        "higher_means_better": higher_means_better,
        "alpha": alpha, "power": power, "n": n,
        "equal_per_sequence": equal_per_sequence,
    }

    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError(
            "supply exactly one of (power, n); leave the other None"
        )
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _power_directional(
            delta=diff, eps=eps, sd_w=sd_w, n_total=n,
            S=S, Vfn=Vfn, b=b, alpha=alpha,
            higher_better=higher_means_better,
        )
        n_used = n
    elif solve_for == "n":
        assert power is not None

        def _pwr(nt: int) -> float:
            return _power_directional(
                delta=diff, eps=eps, sd_w=sd_w, n_total=nt,
                S=S, Vfn=Vfn, b=b, alpha=alpha,
                higher_better=higher_means_better,
            )

        n_used, achieved = _solve_n(
            _pwr, S=S, target_power=power,
            equal_per_sequence=equal_per_sequence,
        )
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "superiority_by_margin_higher_order_cross_over_diff",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // S if S > 0 else None,
        "n_sequences": S,
        "design_label": label,
        "signed_margin": eps,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "for the Difference of Two Means in a Higher-Order "
            "Cross-Over Design.",
            "Chen, K.W.; Chow, S.C. & Li, G. (1997). A Note on Sample "
            "Size Determination for Bioequivalence Studies with "
            "Higher-Order Crossover Designs. J. Pharmacokin. Biopharm. "
            "25(6): 753-765.",
            "Chow, S.C. & Liu, J.P. (1999). Design and Analysis of "
            "Bioavailability and Bioequivalence Studies. Marcel Dekker.",
        ],
    }


# ===========================================================================
# 5) Non-Inferiority Tests for the Ratio of Two Means in a Higher-Order
#    Cross-Over Design
# ===========================================================================
#
# Multiplicative model: log-transform data so X = ln(Y).  The within-subject
# SD on the log scale is sigma_w = sqrt(ln(COV^2 + 1)) where COV is the
# original-scale coefficient of variation.
#
# The test statistic SE is  se = sigma_w * sqrt(b / n_avg).
# Degrees of freedom V = V_fn(n_avg) (same as difference chapters).
#
# Power for "higher better" NI (H1: phi > 1 - NIM):
#   eps_log = ln(1 - NIM)   (negative)
#   power = T_V( (ln(R1) - eps_log)/se - t_{V,1-alpha} )
#
# Power for "higher worse" NI (H1: phi < 1 + NIM):
#   eps_log = ln(1 + NIM)   (positive)
#   power = 1 - T_V( t_{V,1-alpha} - (eps_log - ln(R1))/se )
# ===========================================================================

# ===========================================================================
# 5) Tests for the Ratio of Two Means (Chapter 526)
# ===========================================================================


def _power_ineq_ratio_ho(
    *, r1: float, cov: float, n_total: int,
    S: int, Vfn, b: float, alpha: float, sides: int,
) -> float:
    """Power for the inequality test on the ratio in a higher-order cross-over.

    Higher-Order Cross-Over Design".

    The formula is identical to the difference inequality (Chapter 527) but
    applied to the log-transformed data:

        sigma_X = sqrt(ln(COV² + 1))
        diff_log = |ln(r1)|
        se = sigma_X * sqrt(b / n_avg)
        Power = T_V( diff_log / se - t_{V, 1-alpha/2} )   [two-sided]
    """
    if n_total < S + 1:
        return 0.0
    n_avg = n_total / S
    V = Vfn(n_avg)
    if V <= 0:
        return 0.0
    sigma_w = _sigma_w_from_cov(cov)
    se = sigma_w * math.sqrt(b / n_avg)
    if se <= 0:
        return 0.0
    log_r1 = abs(math.log(r1))
    if sides == 2:
        tc = D.t_ppf(1.0 - alpha / 2.0, V)
        return _stdt(log_r1 / se - tc, V)
    else:
        tc = D.t_ppf(1.0 - alpha, V)
        return _stdt(log_r1 / se - tc, V)


def tests_ratio_two_means_higher_order_crossover(
    *,
    r1: float,
    cov: float,
    design: str | None = None,
    n_sequences: int | None = None,
    n_periods: int | None = None,
    efficiency: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    equal_per_sequence: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Tests for the ratio of two means in a higher-order cross-over design.

    Higher-Order Cross-Over Design" (Chen, Chow & Li 1997).

    Power is computed using the shifted central-t formula on the log-transformed
    data.  This is the inequality (two-sided or one-sided) counterpart to the
    NI and equivalence chapters for the ratio.

    Parameters
    ----------
    r1 : float
        True mean ratio μ_T / μ_R under H₁.  Must be positive and ≠ 1.
    cov : float
        Coefficient of variation (COV) on the original scale.  Used to
        derive the within-subject log-scale SD:
        ``sigma_X = sqrt(ln(COV² + 1))``.
    design : str or None
        Higher-order design keyword: ``balaam`` (2x4), ``two_sequence_dual``
        (3x2 ABB|BAA), ``four_period_two_sequence`` (4x2), or
        ``four_period_four_sequence`` (4x4).  Common aliases accepted.
    n_sequences, n_periods, efficiency : optional
        Custom design; ignored when ``design`` is given.
    alpha : float
        Type-I error rate (default 0.05 two-sided).
    power, n : float or int or None
        Supply exactly one; the other is solved for.  ``n`` is total subjects
        across all sequences.
    sides : int
        1 or 2 (default 2).
    equal_per_sequence : bool
        Constrain N to multiples of S ("Equal Per Sequence" mode).
    solve_for : str or None
        ``"n"`` or ``"power"``.

    Returns
    -------
    dict
        Standard envelope with ``n``, ``achieved_power``, ``n_per_sequence``,
        ``n_sequences``, ``design_label``, ``sigma_w`` (log-scale within SD).
    """
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    if r1 == 1.0:
        raise ValueError("r1 must differ from 1.0 (H₀ value)")
    if cov <= 0:
        raise ValueError("cov must be positive")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    S, Vfn, b, label = _resolve_design(design, n_sequences, n_periods, efficiency)

    inputs_echo: dict[str, Any] = {
        "r1": r1, "cov": cov,
        "design": design, "n_sequences": n_sequences,
        "n_periods": n_periods, "efficiency": efficiency,
        "alpha": alpha, "power": power, "n": n,
        "sides": sides, "equal_per_sequence": equal_per_sequence,
    }

    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError("supply exactly one of (power, n)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _power_ineq_ratio_ho(
            r1=r1, cov=cov, n_total=n,
            S=S, Vfn=Vfn, b=b, alpha=alpha, sides=sides,
        )
        n_used = n
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")

        def _pwr(nt: int) -> float:
            return _power_ineq_ratio_ho(
                r1=r1, cov=cov, n_total=nt,
                S=S, Vfn=Vfn, b=b, alpha=alpha, sides=sides,
            )

        n_used, achieved = _solve_n(
            _pwr, S=S, target_power=power,
            equal_per_sequence=equal_per_sequence,
        )
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    sigma_w = _sigma_w_from_cov(cov)

    return {
        "method_id": "tests_ratio_two_means_higher_order_crossover",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // S if S > 0 else None,
        "n_sequences": S,
        "design_label": label,
        "sigma_w": sigma_w,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "in a Higher-Order Cross-Over Design.",
            *_CITATIONS_RATIO_HO,
        ],
    }


_CITATIONS_RATIO_HO = [
    "Chen, K.W.; Chow, S.C. & Li, G. (1997). A Note on Sample "
    "Size Determination for Bioequivalence Studies with "
    "Higher-Order Crossover Designs. J. Pharmacokin. Biopharm. "
    "25(6): 753-765.",
    "Chow, S.C. & Liu, J.P. (1999). Design and Analysis of "
    "Bioavailability and Bioequivalence Studies. Marcel Dekker.",
    "Chow, S.C.; Shao, J. & Wang, H. (2003). Sample Size "
    "Calculations in Clinical Research. Marcel Dekker.",
    "Schuirmann, D.J. (1987). A comparison of the two one-sided "
    "tests procedure and the power approach for assessing the "
    "equivalence of average bioavailability. J. Pharmacokin. "
    "Biopharm. 15: 657-680.",
]


def _sigma_w_from_cov(cov: float) -> float:
    """Log-scale within-subject SD from original-scale COV."""
    return math.sqrt(math.log(cov * cov + 1.0))


def _power_ni_ratio_ho(
    *, r1: float, nim: float, cov: float, n_total: int,
    S: int, Vfn, b: float, alpha: float,
    higher_means_better: bool,
) -> float:
    """Power for NI test on ratio in higher-order cross-over.

    Non-Inferiority Tests for the
    Ratio of Two Means in a Higher-Order Cross-Over Design".
    """
    if n_total < S + 1:
        return 0.0
    n_avg = n_total / S
    V = Vfn(n_avg)
    if V <= 0:
        return 0.0
    sigma_w = _sigma_w_from_cov(cov)
    se = sigma_w * math.sqrt(b / n_avg)
    if se <= 0:
        return 0.0
    tc = D.t_ppf(1.0 - alpha, V)
    log_r1 = math.log(r1)
    if higher_means_better:
        # H0: phi <= 1-NIM, H1: phi > 1-NIM
        if nim >= 1:
            raise ValueError("nim must be < 1 when higher_means_better=True")
        eps_log = math.log(1.0 - nim)
        arg = (log_r1 - eps_log) / se
        return _stdt(arg - tc, V)
    else:
        # H0: phi >= 1+NIM, H1: phi < 1+NIM
        eps_log = math.log(1.0 + nim)
        arg = (eps_log - log_r1) / se
        return 1.0 - _stdt(tc - arg, V)


def non_inferiority_higher_order_cross_over_ratio(
    *,
    nim: float,
    r1: float = 1.0,
    cov: float,
    design: str | None = None,
    n_sequences: int | None = None,
    n_periods: int | None = None,
    efficiency: float | None = None,
    higher_means_better: bool = True,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    equal_per_sequence: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for ratio of two means in a higher-order
    cross-over design.

    Two Means in a Higher-Order Cross-Over Design".

    The analysis uses the multiplicative model on the log scale.
    ``sigma_w = sqrt(ln(COV^2 + 1))``.

    Inputs
    ------
    nim
        Non-inferiority margin magnitude (positive).  When
        ``higher_means_better=True``, the lower bound is ``1 - NIM``
        (so NIM must be < 1).  When False, the upper bound is ``1 + NIM``.
    r1
        True mean ratio mu_T / mu_R at which power is computed.
        Default 1.0.
    cov
        Coefficient of variation on the original scale (decimal, e.g.
        0.40 for 40%).
    design / n_sequences / n_periods / efficiency
        Higher-order design specification (see
        ``tests_higher_order_cross_over_diff``).
    higher_means_better
        If True (default), H1: phi > 1 - NIM.  If False, H1: phi < 1 + NIM.
    alpha
        One-sided type-I error rate.  Default 0.05.
    power, n
        Provide exactly one; the other is solved for.
    equal_per_sequence
        When solving for N, restrict to multiples of S.  Default True.
    """
    if nim <= 0:
        raise ValueError("nim must be positive")
    if higher_means_better and nim >= 1:
        raise ValueError("nim must be < 1 when higher_means_better=True")
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    if cov <= 0:
        raise ValueError("cov must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")

    S, Vfn, b, label = _resolve_design(
        design, n_sequences, n_periods, efficiency
    )

    inputs_echo = {
        "nim": nim, "r1": r1, "cov": cov,
        "design": design, "n_sequences": n_sequences,
        "n_periods": n_periods, "efficiency": efficiency,
        "higher_means_better": higher_means_better,
        "alpha": alpha, "power": power, "n": n,
        "equal_per_sequence": equal_per_sequence,
    }

    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError("supply exactly one of (power, n)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _power_ni_ratio_ho(
            r1=r1, nim=nim, cov=cov, n_total=n,
            S=S, Vfn=Vfn, b=b, alpha=alpha,
            higher_means_better=higher_means_better,
        )
        n_used = n
    elif solve_for == "n":
        assert power is not None

        def _pwr(nt: int) -> float:
            return _power_ni_ratio_ho(
                r1=r1, nim=nim, cov=cov, n_total=nt,
                S=S, Vfn=Vfn, b=b, alpha=alpha,
                higher_means_better=higher_means_better,
            )

        n_used, achieved = _solve_n(
            _pwr, S=S, target_power=power,
            equal_per_sequence=equal_per_sequence,
        )
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    sigma_w = _sigma_w_from_cov(cov)

    return {
        "method_id": "higher_order_crossover_ni_ratio",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // S if S > 0 else None,
        "n_sequences": S,
        "design_label": label,
        "sigma_w": sigma_w,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Two Means in a Higher-Order Cross-Over Design.",
            *_CITATIONS_RATIO_HO,
        ],
    }


# ===========================================================================
# 6) Equivalence Tests for the Ratio of Two Means in a Higher-Order
#    Cross-Over Design
# ===========================================================================
#
# TOST power:
#   Power(phi) = T_V( (ln(RU) - |ln(phi)|) / (CV_m*sqrt(b/n)) - t_{V,1-alpha} )
#              - T_V( t_{V,1-alpha} - (|ln(phi)| - ln(RL)) / (CV_m*sqrt(b/n)) )
#
# where CV_m = sigma_w = sqrt(ln(COV^2 + 1)).
# ===========================================================================


def _power_eq_ratio_ho(
    *, r1: float, rl: float, ru: float, cov: float, n_total: int,
    S: int, Vfn, b: float, alpha: float,
) -> float:
    """TOST power for equivalence on ratio in higher-order cross-over.

    Equivalence Tests for the
    Ratio of Two Means in a Higher-Order Cross-Over Design".
    """
    if n_total < S + 1:
        return 0.0
    if not (0 < rl < 1 < ru):
        raise ValueError("must have 0 < rl < 1 < ru")
    n_avg = n_total / S
    V = Vfn(n_avg)
    if V <= 0:
        return 0.0
    sigma_w = _sigma_w_from_cov(cov)
    se = sigma_w * math.sqrt(b / n_avg)
    if se <= 0:
        return 0.0
    tc = D.t_ppf(1.0 - alpha, V)
    abs_log_r1 = abs(math.log(r1))
    log_ru = math.log(ru)
    log_rl = math.log(rl)  # negative
    upper = _stdt((log_ru - abs_log_r1) / se - tc, V)
    lower = _stdt(tc - (abs_log_r1 - log_rl) / se, V)
    return max(0.0, upper - lower)


def equivalence_higher_order_cross_over_ratio(
    *,
    rl: float,
    ru: float,
    r1: float = 1.0,
    cov: float,
    design: str | None = None,
    n_sequences: int | None = None,
    n_periods: int | None = None,
    efficiency: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    equal_per_sequence: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Equivalence (TOST) test for ratio of two means in a higher-order
    cross-over design.

    Means in a Higher-Order Cross-Over Design".

    Inputs
    ------
    rl
        Lower equivalence limit on the ratio scale (0 < rl < 1).
        Standard bioequivalence choice: 0.80.
    ru
        Upper equivalence limit on the ratio scale (ru > 1).
        Standard bioequivalence choice: 1.25.
    r1
        True ratio mu_T / mu_R at which power is computed.  Default 1.0.
    cov
        Original-scale coefficient of variation (decimal).
    design / n_sequences / n_periods / efficiency
        Higher-order design specification.
    alpha
        Per one-sided test (TOST) significance level.  Default 0.05.
    power, n
        Provide exactly one.
    equal_per_sequence
        Restrict N to multiples of S when solving.  Default True.
    """
    if not (0 < rl < 1):
        raise ValueError("rl must be in (0, 1)")
    if ru <= 1:
        raise ValueError("ru must be > 1")
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    if cov <= 0:
        raise ValueError("cov must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")

    S, Vfn, b, label = _resolve_design(
        design, n_sequences, n_periods, efficiency
    )

    inputs_echo = {
        "rl": rl, "ru": ru, "r1": r1, "cov": cov,
        "design": design, "n_sequences": n_sequences,
        "n_periods": n_periods, "efficiency": efficiency,
        "alpha": alpha, "power": power, "n": n,
        "equal_per_sequence": equal_per_sequence,
    }

    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError("supply exactly one of (power, n)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _power_eq_ratio_ho(
            r1=r1, rl=rl, ru=ru, cov=cov, n_total=n,
            S=S, Vfn=Vfn, b=b, alpha=alpha,
        )
        n_used = n
    elif solve_for == "n":
        assert power is not None
        if not (rl < r1 < ru):
            raise ValueError(
                "r1 must lie strictly between rl and ru for power > 0"
            )

        def _pwr(nt: int) -> float:
            return _power_eq_ratio_ho(
                r1=r1, rl=rl, ru=ru, cov=cov, n_total=nt,
                S=S, Vfn=Vfn, b=b, alpha=alpha,
            )

        n_used, achieved = _solve_n(
            _pwr, S=S, target_power=power,
            equal_per_sequence=equal_per_sequence,
        )
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    sigma_w = _sigma_w_from_cov(cov)

    return {
        "method_id": "higher_order_crossover_eq_ratio",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // S if S > 0 else None,
        "n_sequences": S,
        "design_label": label,
        "sigma_w": sigma_w,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Means in a Higher-Order Cross-Over Design.",
            *_CITATIONS_RATIO_HO,
        ],
    }
