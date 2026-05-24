"""Specialty two-proportion tests.

Five chapters are implemented here:

* ``tests_two_proportions_stratified`` -- Cochran-Mantel-Haenszel
  stratified design (Woolson, Bean & Rojas 1986; Nam 1992).
* ``tests_two_proportions_repeated_measures`` -- time-averaged
  difference / mixed-model logit-link test for two proportions
  measured repeatedly (Liu & Wu 2005; Brown & Prescott 2006).
* ``tests_two_correlated_proportions_matched`` -- Dupont (1988)
  matched case-control design with M controls per case.
* ``tests_two_ordered_categorical`` -- Whitehead (1993) Wilcoxon /
  proportional-odds test for two ordered-categorical groups.
* ``cochran_armitage_trend`` -- Nam (1987) trend-in-proportions test
  across ordered groups (continuity-corrected or uncorrected).

Each routine follows the project convention:

* Keyword-only parameters.
* Solve for either ``n`` or ``power`` (the user supplies one).
* Returns a dict ``{method_id, solve_for, n, achieved_power,
  inputs_echo, citations}`` with extra method-specific fields where
  natural (e.g. ``n1, n2`` per stratum).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1. CMH stratified two-proportion test
# ---------------------------------------------------------------------------

def _cmh_power_at_M(
    M: float,
    R1: Sequence[float],
    R2: Sequence[float],
    p2: Sequence[float],
    or0: float,
    or1: float,
    alpha: float,
    sides: int,
    continuity: bool,
) -> tuple[float, int, int]:
    """Asymptotic power of the Cochran-Mantel-Haenszel z test.

    Implements the Woolson-Bean-Rojas / Nam (1992) closed-form
    approximation generalised to OR0 != 1 (see
    Proportions in a Stratified Design`` chapter).

    ``R1, R2, p2`` are per-stratum vectors; the per-stratum sample
    sizes are ``n1j = M*R1[j]``, ``n2j = M*R2[j]``.  ``M`` is the
    sample-size multiplier (need not be an integer for the formula).

    Returns ``(power, ceil(N1), ceil(N2))``.
    """
    J = len(R1)
    if len(R2) != J or len(p2) != J:
        raise ValueError("R1, R2, p2 must have the same length")

    def _p1(or_, p2j):
        return or_ * p2j / (1.0 - p2j + or_ * p2j)

    # Per-stratum integer sample sizes (the formula evaluates on the
    # real-valued n's so we keep the float versions for power).
    n1f = [M * r for r in R1]
    n2f = [M * r for r in R2]

    p1_h0 = [_p1(or0, q) for q in p2]
    p1_h1 = [_p1(or1, q) for q in p2]

    w = [n1 * n2 / (n1 + n2) if (n1 + n2) > 0 else 0.0
         for n1, n2 in zip(n1f, n2f)]

    EU = sum(wj * ((p11 - q) - (p10 - q))
             for wj, p11, p10, q in zip(w, p1_h1, p1_h0, p2))

    if or0 == 1.0:
        # Common-OR variance under H0 uses the average response per stratum.
        pbar = [(p11 * n1 + q * n2) / (n1 + n2) if (n1 + n2) > 0 else 0.0
                for p11, q, n1, n2 in zip(p1_h1, p2, n1f, n2f)]
        V0 = sum(wj * pb * (1 - pb) for wj, pb in zip(w, pbar))
    else:
        V0 = sum(
            (wj ** 2) * (p10 * (1 - p10) / n1 + q * (1 - q) / n2)
            for wj, p10, q, n1, n2 in zip(w, p1_h0, p2, n1f, n2f)
        )

    V1 = sum(
        (wj ** 2) * (p11 * (1 - p11) / n1 + q * (1 - q) / n2)
        for wj, p11, q, n1, n2 in zip(w, p1_h1, p2, n1f, n2f)
    )

    if V1 <= 0 or V0 <= 0:
        return 0.0, int(math.ceil(sum(n1f))), int(math.ceil(sum(n2f)))

    cc = 0.5 if continuity else 0.0

    z_alpha = norm.ppf(1 - (alpha / 2 if sides == 2 else alpha))
    arg = (z_alpha * math.sqrt(V0) - EU + cc) / math.sqrt(V1)
    power = 1.0 - norm.cdf(arg)
    if sides == 2:
        # Symmetric two-sided: add lower-tail contribution.
        arg_lo = (-z_alpha * math.sqrt(V0) - EU - cc) / math.sqrt(V1)
        power += norm.cdf(arg_lo)
    return (float(min(max(power, 0.0), 1.0)),
            int(math.ceil(sum(n1f))), int(math.ceil(sum(n2f))))


def tests_two_proportions_stratified(
    *,
    p2: Sequence[float],
    r1: Sequence[float],
    r2: Sequence[float] | None = None,
    or1: float,
    or0: float = 1.0,
    alpha: float = 0.05,
    m: float | None = None,
    power: float | None = None,
    sides: int = 1,
    continuity: bool = True,
    m_max: float = 1_000_000.0,
) -> dict[str, Any]:
    """Cochran-Mantel-Haenszel power / sample size in stratified 2x2 tables.

    Parameters
    ----------
    p2 : sequence
        Per-stratum control proportions :math:`p_{2j}`.
    r1, r2 : sequence
        Per-stratum group multipliers; ``n1j = M*r1[j]``,
        ``n2j = M*r2[j]``.  ``r2`` defaults to a copy of ``r1``.
    or1 : float
        Odds ratio under the alternative hypothesis.
    or0 : float, default 1
        Odds ratio under the null hypothesis (the usual CMH test fixes
        this at 1).
    alpha : float, default 0.05
    m : float, optional
        Sample-size multiplier; supply with ``power=None`` to solve
        for power.
    power : float, optional
        Target power; supply with ``m=None`` to solve for the
        smallest M giving at least this power.
    sides : int, default 1
        1 for the (typical) one-sided CMH test, 2 for two-sided.
    continuity : bool, default True
        Apply the Fleiss / Nam 1/2 continuity correction.
    m_max : float
        Upper search bound when solving for M.
    """
    if r2 is None:
        r2 = list(r1)
    if not (len(p2) == len(r1) == len(r2)):
        raise ValueError("p2, r1, r2 must have the same length")
    if or1 <= 0 or or0 <= 0:
        raise ValueError("odds ratios must be positive")
    if or1 == or0:
        raise ValueError("or1 must differ from or0")

    have_m = m is not None
    have_p = power is not None
    if have_m == have_p:
        raise ValueError("supply exactly one of (m, power)")

    inputs_echo = {
        "p2": list(p2), "r1": list(r1), "r2": list(r2),
        "or1": or1, "or0": or0, "alpha": alpha, "m": m,
        "power": power, "sides": sides, "continuity": continuity,
    }

    if have_m:
        pwr, n1, n2 = _cmh_power_at_M(
            float(m), r1, r2, p2, or0, or1, alpha, sides, continuity,
        )
        N = n1 + n2
        result = {
            "n": N, "n1": n1, "n2": n2, "m": float(m),
            "achieved_power": pwr,
        }
        solve_for = "power"
    else:
        target = float(power)
        if not 0.0 < target < 1.0:
            raise ValueError("power must lie in (0, 1)")

        # Bracket-then-bisect on M (real-valued).  The function reports a
        # fractional M; we report the ceiling integer N implied by it.
        def f(M):
            return _cmh_power_at_M(M, r1, r2, p2, or0, or1, alpha,
                                   sides, continuity)[0] - target

        lo = 2.0
        hi = max(lo + 1, 2.0)
        # Expand until power exceeds target or hi blows past m_max.
        while f(hi) < 0 and hi < m_max:
            lo = hi
            hi = hi * 2.0
        if f(hi) < 0:
            raise RuntimeError("failed to bracket M within m_max")

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) >= 0:
                hi = mid
            else:
                lo = mid

        M_star = hi
        pwr, n1, n2 = _cmh_power_at_M(M_star, r1, r2, p2, or0, or1,
                                      alpha, sides, continuity)
        N = n1 + n2
        # If after ceiling the power slipped below target, bump M_star
        # until ceil sample sizes give at least the target power.
        guard = 0
        while pwr < target and guard < 200:
            M_star += max(0.5, M_star * 0.01)
            pwr, n1, n2 = _cmh_power_at_M(M_star, r1, r2, p2, or0, or1,
                                          alpha, sides, continuity)
            N = n1 + n2
            guard += 1
        result = {
            "n": N, "n1": n1, "n2": n2, "m": float(M_star),
            "achieved_power": pwr,
        }
        solve_for = "n"

    return {
        "method_id": "tests_two_proportions_stratified",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Design (Cochran-Mantel-Haenszel Test)",
            "Woolson, R.F., Bean, J.A. and Rojas, P.B. (1986). 'Sample "
            "Size for Case-Control Studies Using Cochran's Statistic.' "
            "Biometrics 42, 927-932.",
            "Nam, J. (1992). 'Sample Size Determination for Case-Control "
            "Studies and the Comparison of Stratified and Unstratified "
            "Analyses.' Biometrics 48, 389-395.",
        ],
    }


# ---------------------------------------------------------------------------
# 2. Repeated measures two-proportion test (Liu & Wu / Brown & Prescott)
# ---------------------------------------------------------------------------

def _rm_cov_factor(m: int, rho: float, cov_type: str) -> float:
    """Sum of the elements of R^{-1} where R is the m-by-m correlation matrix
    for one of the supported covariance patterns.

    The variance of beta-hat-1 is proportional to ``1 / sum(R^{-1})``;
    equivalently, the effective sample size per subject is
    ``sum(R^{-1})``.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    cov_type = cov_type.lower()
    if cov_type in ("simple", "independent"):
        return float(m)
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie in (-1, 1)")

    if cov_type in ("cs", "compound", "compound_symmetry", "compound-symmetry"):
        # R = (1-rho) I + rho 1 1'.  Sum of R^{-1} entries =
        #   m / (1 + (m-1) rho).
        return m / (1.0 + (m - 1) * rho)
    if cov_type in ("ar1", "ar(1)"):
        # AR(1) inverse row-sums: only first/last rows have the larger
        # value 1/(1-rho).  Closed form: sum(R^{-1}) =
        #   [m - (m-2) rho] / (1 - rho) / (1 + rho) ... Actually the
        # well-known result is
        #   sum(R^{-1}) = (m (1 - rho) + 2 rho) / (1 - rho^2) ... wrong.
        # Use the explicit formula: for AR(1) the sum equals
        #   (2 + (m-2)(1 - rho)) / (1 + rho)
        # Derivation: 1'R^{-1}1 with the tridiagonal R^{-1}.
        return (2.0 + (m - 2) * (1.0 - rho)) / (1.0 + rho)
    if cov_type in ("banded", "banded1", "banded(1)"):
        # Build R explicitly and invert (small m so cheap).
        import numpy as np
        R = np.eye(m)
        for i in range(m - 1):
            R[i, i + 1] = rho
            R[i + 1, i] = rho
        try:
            inv = np.linalg.inv(R)
        except np.linalg.LinAlgError as e:
            raise ValueError("Banded(1) correlation is singular") from e
        return float(inv.sum())
    raise ValueError(f"unknown covariance type: {cov_type!r}")


def _rm_power(
    n1: int, n2: int, p1: float, p2: float, m: int, rho: float,
    cov_type: str, alpha: float, sides: int, scale: str,
) -> float:
    """Two-proportion repeated-measures power (Liu & Wu 2005).

    ``scale`` -- ``"diff"`` for the P1-P2 test, ``"logit"`` for the
    logit / log-OR test (two test-statistic options:
    radio buttons).
    """
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0):
        raise ValueError("p1 and p2 must lie strictly in (0, 1)")
    if n1 < 2 or n2 < 2:
        return 0.0

    q1, q2 = 1 - p1, 1 - p2

    eff = _rm_cov_factor(m, rho, cov_type)

    if scale == "diff":
        d = p1 - p2
        # Pooled variance under H0 uses pooled proportion.
        p_pool = (n1 * p1 + n2 * p2) / (n1 + n2)
        sigma0 = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2) / eff)
        sigma1 = math.sqrt((p1 * q1 / n1 + p2 * q2 / n2) / eff)
    elif scale == "logit":
        d = math.log(p1 / q1) - math.log(p2 / q2)
        # Null variance: variance of logit-difference
        # under H0 uses the pooled proportion via the same factor).
        p_pool = (n1 * p1 + n2 * p2) / (n1 + n2)
        q_pool = 1 - p_pool
        # var(logit(p_hat)) ~ 1 / (n p q); for the difference we get
        # sum of 1/(n_k p_pool q_pool) under H0.
        sigma0 = math.sqrt(
            (1.0 / (n1 * p_pool * q_pool) + 1.0 / (n2 * p_pool * q_pool)) / eff
        )
        sigma1 = math.sqrt(
            (1.0 / (n1 * p1 * q1) + 1.0 / (n2 * p2 * q2)) / eff
        )
    else:
        raise ValueError("scale must be 'diff' or 'logit'")

    if sigma1 <= 0:
        return 0.0
    z_alpha = norm.ppf(1 - (alpha / 2 if sides == 2 else alpha))
    # Power, two-sided form taken from chapter:
    #   Power = 1 - Phi( (sigma0/sigma1) z_{1-a/2} - d/sigma1 )
    #         + (if 2-sided) Phi( -(sigma0/sigma1) z_{1-a/2} - d/sigma1 )
    arg_up = (sigma0 / sigma1) * z_alpha - d / sigma1
    power = 1.0 - norm.cdf(arg_up)
    if sides == 2:
        arg_lo = -(sigma0 / sigma1) * z_alpha - d / sigma1
        power += norm.cdf(arg_lo)
    return float(min(max(power, 0.0), 1.0))


def tests_two_proportions_repeated_measures(
    *,
    p1: float | None = None,
    p2: float,
    odds_ratio: float | None = None,
    m: int,
    rho: float,
    cov_type: str = "compound_symmetry",
    alpha: float = 0.05,
    sides: int = 2,
    scale: str = "logit",
    n_per_group: int | None = None,
    n1: int | None = None,
    n2: int | None = None,
    power: float | None = None,
    n_max: int = 1_000_000,
) -> dict[str, Any]:
    """Two-proportion test in a repeated-measures design.

    Effect size: supply either ``p1`` or ``odds_ratio`` (with
    ``p2``).  Sample size: supply ``n_per_group`` or both
    ``n1`` and ``n2`` to solve for power, or ``power`` to solve for
    the equal per-group N.
    """
    if p1 is None:
        if odds_ratio is None:
            raise ValueError("supply p1 or odds_ratio")
        if odds_ratio <= 0:
            raise ValueError("odds_ratio must be positive")
        p1 = odds_ratio * p2 / (1.0 - p2 + odds_ratio * p2)
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0):
        raise ValueError("p1 and p2 must lie in (0, 1)")
    if m < 1:
        raise ValueError("m must be >= 1")

    have_n = (n_per_group is not None) or (n1 is not None and n2 is not None)
    have_p = power is not None
    if have_n == have_p:
        raise ValueError("supply exactly one of (n, power)")

    inputs_echo = {
        "p1": p1, "p2": p2, "odds_ratio": (p1 / (1 - p1)) / (p2 / (1 - p2)),
        "m": m, "rho": rho, "cov_type": cov_type,
        "alpha": alpha, "sides": sides, "scale": scale,
        "n_per_group": n_per_group, "n1": n1, "n2": n2, "power": power,
    }

    if have_n:
        if n_per_group is not None:
            n1_, n2_ = int(n_per_group), int(n_per_group)
        else:
            n1_, n2_ = int(n1), int(n2)
        pwr = _rm_power(n1_, n2_, p1, p2, m, rho, cov_type, alpha, sides, scale)
        result = {"n": n1_ + n2_, "n1": n1_, "n2": n2_, "achieved_power": pwr}
        solve_for = "power"
    else:
        target = float(power)
        if not 0.0 < target < 1.0:
            raise ValueError("power must lie in (0, 1)")

        lo, hi = 2, 4
        while hi <= n_max:
            if _rm_power(hi, hi, p1, p2, m, rho, cov_type, alpha,
                         sides, scale) >= target:
                break
            lo = hi
            hi = max(hi + 1, int(hi * 1.7))
        else:
            raise RuntimeError(f"failed to bracket N within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _rm_power(mid, mid, p1, p2, m, rho, cov_type, alpha,
                         sides, scale) >= target:
                hi = mid
            else:
                lo = mid
        n_star = hi
        pwr = _rm_power(n_star, n_star, p1, p2, m, rho, cov_type,
                        alpha, sides, scale)
        result = {"n": 2 * n_star, "n1": n_star, "n2": n_star,
                  "achieved_power": pwr}
        solve_for = "n"

    return {
        "method_id": "tests_two_proportions_repeated_measures",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Measures Design",
            "Liu, H. and Wu, T. (2005). 'Sample Size Calculation and "
            "Power Analysis of Time-Averaged Difference.' Journal of "
            "Modern Applied Statistical Methods, 4(2), 434-445.",
            "Brown, H. and Prescott, R. (2006). Applied Mixed Models in "
            "Medicine, 2nd ed. Wiley.",
            "Diggle, P.J., Liang, K.Y., and Zeger, S.L. (1994). Analysis "
            "of Longitudinal Data. Oxford University Press.",
        ],
    }


# ---------------------------------------------------------------------------
# 3. Matched case-control design (Dupont 1988)
# ---------------------------------------------------------------------------

def _dupont_t_vector(m: int, p0: float, or_: float,
                     phi: float) -> list[float]:
    """Return the marginal probabilities ``t_1, ..., t_M`` from Dupont (1988)
    for a given OR (also used at OR = 1 for the null reference)."""
    q0 = 1.0 - p0
    p1 = or_ * p0 / (q0 + or_ * p0)
    q1 = 1.0 - p1
    s = math.sqrt(p1 * q1 * p0 * q0)
    p11 = p1 * p0 + phi * s
    p01 = q1 * p0 - phi * s
    eps = -1e-9
    if min(p11, p01, p1 * q0 - phi * s, q1 * q0 + phi * s) < eps:
        raise ValueError(
            "Implied joint probabilities are negative; check (p0, phi, OR)"
        )
    p0p = p11 / p1 if p1 > 0 else 0.0
    p0m = p01 / q1 if q1 > 0 else 0.0
    q0p = 1.0 - p0p
    q0m = 1.0 - p0m
    from math import comb
    t = [0.0] * (m + 1)
    for k in range(1, m + 1):
        t[k] = (
            p1 * comb(m, k - 1) * (p0p ** (k - 1)) * (q0p ** (m - k + 1))
            + q1 * comb(m, k) * (p0m ** k) * (q0m ** (m - k))
        )
    return t


def _dupont_power(
    n: int, m: int, p0: float, or_: float, phi: float, alpha: float,
) -> float:
    """Power for Dupont's 2-by-(M+1) matched case-control test."""
    if n < 1:
        return 0.0

    # Per Dupont (1988): t_k is computed under H1
    # (the true OR/joint), and e_OR, v_OR use this t_k together with
    # the appropriate value of OR.  The null reference (e_1, v_1) uses
    # the same t_k but evaluated at OR = 1.
    t_alt = _dupont_t_vector(m, p0, or_, phi)

    def _sums(or_value):
        e = 0.0
        v = 0.0
        for k in range(1, m + 1):
            denom = k * or_value + (m - k + 1)
            e += k * t_alt[k] * or_value / denom
            v += k * t_alt[k] * or_value * (m - k + 1) / (denom * denom)
        return e, v

    e_or, v_or = _sums(or_)
    e_1, v_1 = _sums(1.0)

    if v_or <= 0 or v_1 <= 0:
        return 0.0

    z = norm.ppf(1 - alpha / 2)
    a_up = (math.sqrt(n) * (e_1 - e_or) - z * math.sqrt(v_1)) / math.sqrt(v_or)
    a_lo = (math.sqrt(n) * (e_1 - e_or) + z * math.sqrt(v_1)) / math.sqrt(v_or)
    power = norm.cdf(a_up) + (1.0 - norm.cdf(a_lo))
    return float(min(max(power, 0.0), 1.0))


def tests_two_correlated_proportions_matched(
    *,
    p0: float,
    odds_ratio: float,
    phi: float = 0.2,
    m: int = 1,
    n: int | None = None,
    power: float | None = None,
    alpha: float = 0.05,
    n_max: int = 100_000,
) -> dict[str, Any]:
    """Dupont (1988) matched case-control sample-size routine.

    Parameters
    ----------
    p0 : float
        Probability that a sampled control patient was exposed.
    odds_ratio : float
        Odds ratio of being exposed for cases relative to controls.
    phi : float
        Correlation of exposure between matched case and control.
        Dupont (1988) recommends 0.2 when no prior estimate is
        available.
    m : int
        Number of matched controls per case (>= 1).
    n : int, optional
        Number of cases; if supplied, solve for power.
    power : float, optional
        Target power; if supplied, solve for N (cases).
    alpha : float
        Two-sided significance level.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if not 0 < p0 < 1:
        raise ValueError("p0 must lie in (0, 1)")
    if odds_ratio <= 0:
        raise ValueError("odds_ratio must be positive")
    if not -1 < phi < 1:
        raise ValueError("phi must lie in (-1, 1)")

    have_n = n is not None
    have_p = power is not None
    if have_n == have_p:
        raise ValueError("supply exactly one of (n, power)")

    inputs_echo = {
        "p0": p0, "odds_ratio": odds_ratio, "phi": phi, "m": m,
        "n": n, "power": power, "alpha": alpha,
    }

    if have_n:
        pwr = _dupont_power(int(n), m, p0, odds_ratio, phi, alpha)
        result = {"n": int(n), "achieved_power": pwr, "m": m}
        solve_for = "power"
    else:
        target = float(power)
        if not 0.0 < target < 1.0:
            raise ValueError("power must lie in (0, 1)")
        lo, hi = 1, 2
        while hi <= n_max:
            if _dupont_power(hi, m, p0, odds_ratio, phi, alpha) >= target:
                break
            lo = hi
            hi = max(hi + 1, int(hi * 1.7))
        else:
            raise RuntimeError(f"failed to bracket N within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _dupont_power(mid, m, p0, odds_ratio, phi, alpha) >= target:
                hi = mid
            else:
                lo = mid
        n_star = hi
        pwr = _dupont_power(n_star, m, p0, odds_ratio, phi, alpha)
        result = {"n": n_star, "achieved_power": pwr, "m": m}
        solve_for = "n"

    return {
        "method_id": "tests_two_correlated_proportions_matched",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Matched Case-Control Design",
            "Dupont, W.D. (1988). 'Power Calculations for Matched "
            "Case-Control Studies.' Biometrics 44, 1157-1168.",
            "Breslow, N.E. and Day, N.E. (1980). Statistical Methods in "
            "Cancer Research, Volume I. The Analysis of Case-Control "
            "Studies. IARC Lyon.",
        ],
    }


# ---------------------------------------------------------------------------
# 4. Two ordered-categorical variables (Whitehead 1993)
# ---------------------------------------------------------------------------

def _whitehead_V(n1: int, n2: int, p_c: Sequence[float],
                 p_e: Sequence[float]) -> float:
    """The Whitehead (1993) variance factor V for the efficient score."""
    N = n1 + n2
    if N < 4:
        return 0.0
    cube = sum(((pc + pe) / 2.0) ** 3 for pc, pe in zip(p_c, p_e))
    return n1 * n2 * N / (3 * (N + 1) ** 2) * (1.0 - cube)


def _whitehead_pe(theta: float, p_c: Sequence[float]) -> list[float]:
    """Compute experimental-group category probabilities under proportional
    odds with log-OR theta.

    Q_iE / (1 - Q_iE) = exp(theta) * Q_iC / (1 - Q_iC)
    => Q_iE = (e^theta Q_iC) / (1 - Q_iC + e^theta Q_iC)
    """
    e = math.exp(theta)
    Q_c = []
    s = 0.0
    for p in p_c:
        s += p
        Q_c.append(s)
    # The last cumulative should be 1.0 exactly; clamp.
    Q_c[-1] = 1.0
    Q_e = []
    for q in Q_c[:-1]:
        Qe = e * q / (1.0 - q + e * q)
        Q_e.append(Qe)
    Q_e.append(1.0)
    p_e = [Q_e[0]] + [Q_e[i] - Q_e[i - 1] for i in range(1, len(Q_e))]
    return p_e


def _whitehead_power(
    n1: int, n2: int, theta: float, p_c: Sequence[float],
    alpha: float, sides: int,
) -> float:
    if n1 < 2 or n2 < 2:
        return 0.0
    if abs(theta) < 1e-15:
        return alpha
    p_e = _whitehead_pe(theta, p_c)
    V = _whitehead_V(n1, n2, p_c, p_e)
    if V <= 0:
        return 0.0
    z = norm.ppf(1 - (alpha / 2 if sides == 2 else alpha))
    arg = z - abs(theta) * math.sqrt(V)
    power = 1.0 - norm.cdf(arg)
    return float(min(max(power, 0.0), 1.0))


def tests_two_ordered_categorical(
    *,
    p_c: Sequence[float],
    theta: float,
    alpha: float = 0.05,
    sides: int = 2,
    n_per_group: int | None = None,
    n1: int | None = None,
    n2: int | None = None,
    power: float | None = None,
    n_max: int = 1_000_000,
) -> dict[str, Any]:
    """Whitehead (1993) ordered-categorical Wilcoxon test.

    ``p_c`` is the vector of category probabilities in the control
    group (must be strictly positive and sum to 1; values are
    normalised silently).  ``theta`` is the assumed common log-odds
    ratio under proportional odds.
    """
    p_c = list(p_c)
    if len(p_c) < 2:
        raise ValueError("need at least two ordered categories")
    if any(p <= 0 for p in p_c):
        raise ValueError("control proportions must be strictly positive")
    total = sum(p_c)
    p_c = [p / total for p in p_c]

    have_n = (n_per_group is not None) or (n1 is not None and n2 is not None)
    have_p = power is not None
    if have_n == have_p:
        raise ValueError("supply exactly one of (n, power)")

    inputs_echo = {
        "p_c": list(p_c), "theta": theta, "alpha": alpha, "sides": sides,
        "n_per_group": n_per_group, "n1": n1, "n2": n2, "power": power,
    }

    if have_n:
        if n_per_group is not None:
            n1_, n2_ = int(n_per_group), int(n_per_group)
        else:
            n1_, n2_ = int(n1), int(n2)
        pwr = _whitehead_power(n1_, n2_, theta, p_c, alpha, sides)
        result = {"n": n1_ + n2_, "n1": n1_, "n2": n2_,
                  "achieved_power": pwr}
        solve_for = "power"
    else:
        target = float(power)
        if not 0.0 < target < 1.0:
            raise ValueError("power must lie in (0, 1)")
        lo, hi = 2, 4
        while hi <= n_max:
            if _whitehead_power(hi, hi, theta, p_c, alpha, sides) >= target:
                break
            lo = hi
            hi = max(hi + 1, int(hi * 1.7))
        else:
            raise RuntimeError(f"failed to bracket N within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _whitehead_power(mid, mid, theta, p_c, alpha, sides) >= target:
                hi = mid
            else:
                lo = mid
        n_star = hi
        pwr = _whitehead_power(n_star, n_star, theta, p_c, alpha, sides)
        result = {"n": 2 * n_star, "n1": n_star, "n2": n_star,
                  "achieved_power": pwr}
        solve_for = "n"

    return {
        "method_id": "tests_two_ordered_categorical",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Whitehead, J. (1993). 'Sample Size Calculations for Ordered "
            "Categorical Data.' Statistics in Medicine 12, 2257-2271.",
            "Julious, S.A. (2010). Sample Sizes for Clinical Trials. "
            "Chapman & Hall/CRC.",
            "Machin, D., Campbell, M., Fayers, P. and Pinol, A. (1997). "
            "Sample Size Tables for Clinical Studies, 2nd ed. Blackwell.",
        ],
    }


# ---------------------------------------------------------------------------
# 5. Cochran-Armitage test for trend (Nam 1987)
# ---------------------------------------------------------------------------

def _ca_power(
    n: Sequence[int], p: Sequence[float], x: Sequence[float],
    alpha: float, sides: int, continuity: bool, direction: int,
) -> float:
    """Approximate (normal-theory) power of the Cochran-Armitage test.

    ``direction`` is +1 (increasing, upper-tailed), -1 (decreasing,
    lower-tailed), or 0 (two-sided).
    """
    k = len(n)
    if not (len(p) == len(x) == k):
        raise ValueError("n, p, x must have the same length")
    N = sum(n)
    if N <= 0:
        return 0.0
    xbar = sum(ni * xi for ni, xi in zip(n, x)) / N
    p_bar = sum(ni * pi for ni, pi in zip(n, p)) / N
    q_bar = 1.0 - p_bar
    Sxx = sum(ni * (xi - xbar) ** 2 for ni, xi in zip(n, x))
    if Sxx <= 0:
        return 0.0
    if continuity:
        # equally-spaced or average-gap fallback
        if k >= 2:
            Delta = sum(x[i + 1] - x[i] for i in range(k - 1)) / (k - 1)
        else:
            Delta = 0.0
    else:
        Delta = 0.0

    num_signed = sum(ni * pi * (xi - xbar) for ni, pi, xi in zip(n, p, x))
    denom = math.sqrt(sum(ni * pi * (1 - pi) * (xi - xbar) ** 2
                          for ni, pi, xi in zip(n, p, x)))
    if denom <= 0:
        return 0.0

    if sides == 1:
        if direction == 0:
            # infer from proportions
            direction = 1 if num_signed >= 0 else -1
        z_crit = norm.ppf(1 - alpha)
        if direction > 0:
            uU = (-(num_signed - Delta / 2.0)
                  + z_crit * math.sqrt(p_bar * q_bar * Sxx)) / denom
            return float(min(max(1.0 - norm.cdf(uU), 0.0), 1.0))
        else:
            uL = (-(num_signed + Delta / 2.0)
                  - z_crit * math.sqrt(p_bar * q_bar * Sxx)) / denom
            return float(min(max(norm.cdf(uL), 0.0), 1.0))
    elif sides == 2:
        z_crit = norm.ppf(1 - alpha / 2)
        uU = (-(num_signed - Delta / 2.0)
              + z_crit * math.sqrt(p_bar * q_bar * Sxx)) / denom
        uL = (-(num_signed + Delta / 2.0)
              - z_crit * math.sqrt(p_bar * q_bar * Sxx)) / denom
        return float(min(max(1.0 - norm.cdf(uU) + norm.cdf(uL), 0.0), 1.0))
    else:
        raise ValueError("sides must be 1 or 2")


def cochran_armitage_trend(
    *,
    p: Sequence[float],
    x: Sequence[float] | None = None,
    n: Sequence[int] | None = None,
    n_pattern: Sequence[float] | None = None,
    n_mult: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    continuity: bool = True,
    direction: int = 0,
    n_max: int = 1_000_000,
) -> dict[str, Any]:
    """Cochran-Armitage trend-in-proportions test (Nam 1987).

    Effect size: ``p`` is the strictly monotone vector of alternative
    proportions across ``k`` ordered groups.  ``x`` is the
    corresponding covariate (dose) vector; if omitted, equally-spaced
    integers ``0, 1, ..., k-1`` are used.

    Sample size: supply ``n`` directly, or ``n_mult`` with optional
    ``n_pattern`` (multipliers for unequal allocation), or ``power``
    to solve for the smallest equal per-group ``n_mult``.

    ``sides`` is 1 (one-sided) or 2 (two-sided).  ``direction`` is
    +1 / -1 / 0 (auto) for one-sided tests; ignored when sides==2.
    """
    k = len(p)
    if k < 2:
        raise ValueError("p must have at least two entries")
    if x is None:
        x = list(range(k))
    else:
        x = list(x)
    if len(x) != k:
        raise ValueError("len(x) must equal len(p)")
    if n_pattern is None:
        n_pattern = [1.0] * k
    if len(n_pattern) != k:
        raise ValueError("len(n_pattern) must equal len(p)")

    have_n_full = n is not None
    have_n_mult = n_mult is not None
    have_p = power is not None

    # Exactly one of the three sample-size paths must be supplied.
    n_sources = sum([have_n_full, have_n_mult, have_p])
    if n_sources != 1:
        raise ValueError(
            "supply exactly one of (n, n_mult, power)"
        )

    inputs_echo = {
        "p": list(p), "x": x, "n": list(n) if n is not None else None,
        "n_pattern": list(n_pattern), "n_mult": n_mult,
        "alpha": alpha, "power": power, "sides": sides,
        "continuity": continuity, "direction": direction,
    }

    if have_n_full:
        n_vec = list(n)
        pwr = _ca_power(n_vec, p, x, alpha, sides, continuity, direction)
        result = {"n": sum(n_vec), "n_vec": n_vec, "achieved_power": pwr}
        solve_for = "power"
    elif have_n_mult:
        n_vec = [int(math.ceil(n_mult * m)) for m in n_pattern]
        pwr = _ca_power(n_vec, p, x, alpha, sides, continuity, direction)
        result = {"n": sum(n_vec), "n_vec": n_vec, "n_mult": int(n_mult),
                  "achieved_power": pwr}
        solve_for = "power"
    else:
        target = float(power)
        if not 0 < target < 1:
            raise ValueError("power must lie in (0, 1)")

        def pwr_at(nm):
            n_vec = [int(math.ceil(nm * m)) for m in n_pattern]
            return _ca_power(n_vec, p, x, alpha, sides, continuity, direction)

        lo, hi = 2, 4
        while hi <= n_max:
            if pwr_at(hi) >= target:
                break
            lo = hi
            hi = max(hi + 1, int(hi * 1.7))
        else:
            raise RuntimeError(f"failed to bracket n_mult within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pwr_at(mid) >= target:
                hi = mid
            else:
                lo = mid
        nm = hi
        n_vec = [int(math.ceil(nm * m)) for m in n_pattern]
        pwr = _ca_power(n_vec, p, x, alpha, sides, continuity, direction)
        result = {"n": sum(n_vec), "n_vec": n_vec, "n_mult": nm,
                  "achieved_power": pwr}
        solve_for = "n"

    return {
        "method_id": "cochran_armitage_trend",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Nam, J. (1987). 'A Simple Approximation for Calculating Sample "
            "Sizes for Detecting Linear Trend in Proportions.' Biometrics "
            "43, 701-705.",
            "Armitage, P. (1955). 'Tests for Linear Trends in Proportions "
            "and Frequencies.' Biometrics 11, 375-386.",
        ],
    }
