"""Chi-Square tests power & sample-size.

chi-square with Cohen's w effect-size convention).

Cohen's effect size:
    w = sqrt( Σᵢ (p₁ᵢ - p₀ᵢ)² / p₀ᵢ )

Noncentral chi-square:
    λ        = N · w²
    χ²_crit  = χ²_{1-α, df}
    Power    = 1 - F'_{df, λ}(χ²_crit)

where F'_{df, λ} is the noncentral chi-square CDF.

Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
"""
from __future__ import annotations

from typing import Any

from scipy.stats import chi2, ncx2


def _validate(w: float, df: int, alpha: float) -> None:
    if w <= 0:
        raise ValueError("w (effect size) must be > 0")
    if df < 1:
        raise ValueError("df must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def _power_chi_square(*, w: float, n: int, df: int, alpha: float) -> float:
    if n < 1:
        return 0.0
    lam = float(n) * w * w
    crit = chi2.ppf(1.0 - alpha, df)
    return float(1.0 - ncx2.cdf(crit, df, lam))


def power_at_n(*, w: float, n: int, df: int, alpha: float) -> float:
    _validate(w, df, alpha)
    return _power_chi_square(w=w, n=n, df=df, alpha=alpha)


def n_for_power(*, w: float, df: int, alpha: float, power: float,
                n_min: int = 2, n_max: int = 10_000_000) -> tuple[int, float]:
    _validate(w, df, alpha)
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")

    def p_at(n: int) -> float:
        return _power_chi_square(w=w, n=n, df=df, alpha=alpha)

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
    return hi, p_at(hi)


def chi_square(
    *,
    w: float,
    df: int,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Chi-square test (general r×c / goodness-of-fit) power & sample size.

    Parameters
    ----------
    w : float
        Cohen's w effect size (>0).
    df : int
        Degrees of freedom of the chi-square test
        (e.g. (R-1)(C-1) for an R×C contingency table).
    alpha : float
        Type-I error.
    n : int, optional
        Total sample size (N).  Required when ``solve_for == 'power'``.
    power : float, optional
        Desired power.  Required when ``solve_for == 'n'``.
    solve_for : str, optional
        ``'n'`` or ``'power'``.  Inferred from which of ``n``/``power``
        is supplied if omitted.
    """
    inputs_echo = {
        "w": w, "df": df, "alpha": alpha, "n": n, "power": power,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    _validate(w, df, alpha)

    if solve_for == "power":
        assert n is not None
        achieved = power_at_n(w=w, n=int(n), df=df, alpha=alpha)
        out_n = int(n)
    elif solve_for == "n":
        assert power is not None
        out_n, achieved = n_for_power(w=w, df=df, alpha=alpha, power=power)
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    chi2_crit = float(chi2.ppf(1.0 - alpha, df))

    return {
        "method_id": "chi_square",
        "solve_for": solve_for,
        "n": out_n,
        "achieved_power": achieved,
        "ncp": float(out_n) * w * w,
        "chi2_crit": chi2_crit,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cohen, J. (1988). Statistical Power Analysis for the Behavioral "
            "Sciences (2nd ed.). Hillsdale, NJ: Lawrence Erlbaum.",
        ],
    }
