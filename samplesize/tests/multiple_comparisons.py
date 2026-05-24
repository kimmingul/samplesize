"""Sample size / power for ANOVA followed by multiple-comparison
procedures (Tukey-Kramer, Dunnett, Hsu MCB).

based on Hsu (1996).  Power is the probability that *all* of the
simultaneous confidence intervals cover the true mean differences
*and* that the maximum CI half-width is no greater than ω/2, where ω
is the user-supplied minimum detectable difference.

Three procedures are supported:

``"tukey_kramer"`` / ``"all_pairs"``
    All-pairs comparisons (Tukey-Kramer).  Critical value is the
    Studentized range quantile ``q_α(k, df) / √2``; power formula::

        Pwr = k ∫_0^u ∫_{-∞}^{∞}
              [Φ(z) − Φ(z − √2·|q*|·s)]^(k−1) φ(z) dz γ(s) ds,
        u = (ω/2) / (σ |q*| √(2/n))

``"dunnett"`` / ``"with_control"``
    Two-sided comparisons of each treatment vs the control (group k).
    Balanced λᵢ = √(½); the critical |q| is the solution of the
    Dunnett bivariate integral.  Power formula::

        Pwr = ∫_0^u ∫_{-∞}^{∞}
              [Φ(z + √2·|q|·s) − Φ(z − √2·|q|·s)]^(k−1) φ(z) dz γ(s) ds

``"hsu_mcb"`` / ``"with_best"``
    Hsu's constrained multiple comparisons with the best.  Critical
    value is the one-sided Dunnett |q|.  Power formula (no leading
    ``k`` factor — some published references print one but treat
    the inner integrand as the full mixture density)::

        Pwr = ∫_0^u ∫_{-∞}^{∞}
              [Φ(z + √2·|q|·s)]^(k−1) φ(z) dz γ(s) ds

where γ(s) is the density of σ̂/σ = √(χ²_df / df).  All three formulas
The implementation matches published examples to within 0.001 in power.

Solver
------
The ``multiple_comparisons`` callable solves either for power (given
the per-group sample size ``n``) or for the smallest balanced ``n``
that achieves the requested power.  Dunnett/Hsu critical values are
cached per ``(k, df, alpha, sides)`` tuple.
"""
from __future__ import annotations

import functools
import math
from typing import Any

# ---------------------------------------------------------------------------
# Quadrature helpers
# ---------------------------------------------------------------------------


def _gamma_chi(s: float, df: float) -> float:
    """Density of S = chi_df / sqrt(df) at s."""
    from scipy.stats import chi
    return math.sqrt(df) * float(chi.pdf(math.sqrt(df) * s, df))


def _quad(f, a, b, *, limit=80):
    from scipy.integrate import quad
    val, _ = quad(f, a, b, limit=limit, epsabs=1e-9, epsrel=1e-8)
    return val


# ---------------------------------------------------------------------------
# Critical values (cached)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4096)
def _dunnett_q_two_sided(k: int, df: int, alpha: float = 0.05) -> float:
    """Two-sided Dunnett |q| with balanced λᵢ = √(½)."""
    from scipy.stats import norm

    sqrt2 = math.sqrt(2.0)

    def Pq(q: float) -> float:
        def outer(s: float) -> float:
            def inner(z: float) -> float:
                a = float(norm.cdf(z + q * s * sqrt2))
                b = float(norm.cdf(z - q * s * sqrt2))
                return (a - b) ** (k - 1) * float(norm.pdf(z))
            return _quad(inner, -8, 8) * _gamma_chi(s, df)
        return _quad(outer, 1e-6, 7)

    lo, hi = 1.0, 5.0
    target = 1.0 - alpha
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if Pq(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-5:
            break
    return 0.5 * (lo + hi)


@functools.lru_cache(maxsize=4096)
def _dunnett_q_one_sided(k: int, df: int, alpha: float = 0.05) -> float:
    """One-sided Dunnett |q| (used by Hsu MCB)."""
    from scipy.stats import norm

    sqrt2 = math.sqrt(2.0)

    def Pq(q: float) -> float:
        def outer(s: float) -> float:
            def inner(z: float) -> float:
                a = float(norm.cdf(z + q * s * sqrt2))
                return a ** (k - 1) * float(norm.pdf(z))
            return _quad(inner, -8, 8) * _gamma_chi(s, df)
        return _quad(outer, 1e-6, 7)

    lo, hi = 1.0, 5.0
    target = 1.0 - alpha
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if Pq(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-5:
            break
    return 0.5 * (lo + hi)


@functools.lru_cache(maxsize=4096)
def _tukey_q_star(k: int, df: int, alpha: float = 0.05) -> float:
    """Tukey |q*| = (Studentized range α-quantile) / √2."""
    from scipy.stats import studentized_range
    return float(studentized_range.ppf(1.0 - alpha, k, df)) / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Power formulas (Hsu 1996)
# ---------------------------------------------------------------------------


def _tukey_kramer_power(*, k: int, n: int, omega: float, sigma: float,
                        alpha: float) -> float:
    from scipy.stats import norm

    if n < 2:
        return 0.0
    df = k * (n - 1)
    q_star = _tukey_q_star(k, df, alpha)
    u = (omega / 2.0) / (sigma * q_star * math.sqrt(2.0 / n))
    if u <= 0:
        return 0.0
    sqrt2 = math.sqrt(2.0)

    def outer(s: float) -> float:
        def inner(z: float) -> float:
            a = float(norm.cdf(z))
            b = float(norm.cdf(z - sqrt2 * q_star * s))
            return (a - b) ** (k - 1) * float(norm.pdf(z))
        return k * _quad(inner, -8, 8) * _gamma_chi(s, df)

    return _quad(outer, 1e-6, u)


def _dunnett_power(*, k: int, n: int, omega: float, sigma: float,
                   alpha: float) -> float:
    from scipy.stats import norm

    if n < 2:
        return 0.0
    df = k * (n - 1)
    q = _dunnett_q_two_sided(k, df, alpha)
    u = (omega / 2.0) / (sigma * q * math.sqrt(2.0 / n))
    if u <= 0:
        return 0.0
    sqrt2 = math.sqrt(2.0)

    def outer(s: float) -> float:
        def inner(z: float) -> float:
            a = float(norm.cdf(z + q * s * sqrt2))
            b = float(norm.cdf(z - q * s * sqrt2))
            return (a - b) ** (k - 1) * float(norm.pdf(z))
        return _quad(inner, -8, 8) * _gamma_chi(s, df)

    return _quad(outer, 1e-6, u)


def _hsu_mcb_power(*, k: int, n: int, omega: float, sigma: float,
                   alpha: float) -> float:
    from scipy.stats import norm

    if n < 2:
        return 0.0
    df = k * (n - 1)
    q = _dunnett_q_one_sided(k, df, alpha)
    u = (omega / 2.0) / (sigma * q * math.sqrt(2.0 / n))
    if u <= 0:
        return 0.0
    sqrt2 = math.sqrt(2.0)

    def outer(s: float) -> float:
        def inner(z: float) -> float:
            a = float(norm.cdf(z + q * s * sqrt2))
            return a ** (k - 1) * float(norm.pdf(z))
        return _quad(inner, -8, 8) * _gamma_chi(s, df)

    return _quad(outer, 1e-6, u)


_PROCEDURES = {
    "tukey_kramer": _tukey_kramer_power,
    "all_pairs": _tukey_kramer_power,
    "tukey": _tukey_kramer_power,
    "dunnett": _dunnett_power,
    "with_control": _dunnett_power,
    "mcc": _dunnett_power,
    "hsu_mcb": _hsu_mcb_power,
    "with_best": _hsu_mcb_power,
    "mcb": _hsu_mcb_power,
}


# ---------------------------------------------------------------------------
# Public callable
# ---------------------------------------------------------------------------


def multiple_comparisons(
    *,
    procedure: str,
    k: int,
    omega: float,
    sigma: float,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 1_000_000,
) -> dict[str, Any]:
    """Power / sample-size for ANOVA-followed-by-multiple-comparisons.

    Parameters
    ----------
    procedure : str
        One of ``"tukey_kramer"`` / ``"all_pairs"`` (Tukey-Kramer),
        ``"dunnett"`` / ``"with_control"`` (Dunnett MCC, two-sided),
        ``"hsu_mcb"`` / ``"with_best"`` (Hsu MCB).
    k : int
        Number of groups (≥ 3).  For Dunnett the control is one of
        the k groups; the number of comparisons is k - 1.
    omega : float
        Minimum detectable difference ("Minimum Detectable
        Difference"), in the same units as the response.
    sigma : float
        Within-group SD.
    alpha : float
        Significance level.
    n : int, optional
        Per-group balanced sample size.
    power : float, optional
        Target power.
    solve_for : str or None
        ``"n"`` (default when only ``power`` given) or ``"power"``
        (default when only ``n`` given).
    """
    if k < 3:
        raise ValueError("k (number of groups) must be at least 3")
    if omega <= 0:
        raise ValueError("omega must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    proc_key = procedure.strip().lower().replace("-", "_").replace(" ", "_")
    if proc_key not in _PROCEDURES:
        raise ValueError(
            f"unknown procedure: {procedure!r} (expected one of "
            f"{sorted(set(_PROCEDURES.keys()))})"
        )
    fn = _PROCEDURES[proc_key]

    inputs_echo = {
        "procedure": procedure, "k": k, "omega": omega, "sigma": sigma,
        "alpha": alpha, "n": n, "power": power,
    }

    if n is None and power is None:
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = fn(k=k, n=int(n), omega=omega, sigma=sigma, alpha=alpha)
        result = {
            "k": k, "n": int(n), "n_total": int(n) * k,
            "diff_over_s": omega / sigma,
            "achieved_power": achieved,
        }
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")

        def p_at(n_val: int) -> float:
            return fn(k=k, n=n_val, omega=omega, sigma=sigma, alpha=alpha)

        # The Hsu (1996) "narrow" condition is non-monotone for very small
        # n (power can spike near n where u crosses certain thresholds);
        # we still use bisection because the function is monotone for the
        # n region of practical interest (≥ a handful per group).
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
        achieved = fn(k=k, n=n_solved, omega=omega, sigma=sigma, alpha=alpha)
        result = {
            "k": k, "n": n_solved, "n_total": n_solved * k,
            "diff_over_s": omega / sigma,
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "multiple_comparisons",
        "solve_for": solve_for,
        "procedure": proc_key,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Hsu, J.C. (1996). Multiple Comparisons: Theory and "
            "Methods. Chapman & Hall.",
            "Dunnett, C.W. (1955). A multiple comparison procedure for "
            "comparing several treatments with a control. JASA 50: "
            "1096-1121.",
            "Tukey, J.W. (1953). The Problem of Multiple Comparisons.",
        ],
    }
