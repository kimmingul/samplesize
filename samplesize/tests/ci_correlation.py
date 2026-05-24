"""Confidence-interval sample-size routines for correlation coefficients.

This module implements six CI procedures for correlation coefficients, all framed as "what
sample size N is needed to obtain a target confidence-interval width
(two-sided) or distance (one-sided) for a planning estimate of the
correlation?":

* :func:`ci_pearson_correlation` — Fisher z-transform
  with variance ``1 / (n - 3)``.
* :func:`ci_spearman_correlation` — Fisher z-transform
  with Bonett-Wright variance ``(1 + r**2/2) / (n - 3)``.
* :func:`ci_kendall_tau_b` — Fisher z-transform with
  Bonett-Wright variance ``0.437 / (n - 4)``.
* :func:`ci_intraclass_correlation` — F-distribution
  limits per Shrout & Fleiss (1979) and Bonett (2002), for one-way and
  two-way ANOVA designs.
* :func:`ci_point_biserial_correlation` — Tate
  (1954, 1955) closed-form normal-approximation variance.
* :func:`ci_kappa` — Cohen (1960) large-sample SE for
  Cohen's kappa using the Cohen or Fleiss-Cohen-Everitt formula.

All Bonett-Wright procedures share a common Fisher-z-style
sample-size search; ICC, point-biserial, and kappa use bespoke variance
formulas. Every public function:

* Accepts keyword-only arguments.
* Requires explicit ``sides`` (``1`` or ``2``).
* Uses ``solve_for`` in ``{"n", "width"}`` (and ``"distance"`` for
  one-sided), defaulting to ``"n"`` when ``n`` is not supplied.
* Searches an ascending integer N until the achieved CI width (or
  one-sided distance) is ``<= target``;
  "smallest N satisfying the precision goal".

References
----------
Bonett, D. G. and Wright, T. A. (2000). 'Sample Size Requirements for
    Estimating Pearson, Kendall and Spearman Correlations.'
    Psychometrika, 65(1), 23-28.
Bonett, D. G. (2002). 'Sample size requirements for estimating
    intraclass correlations with desired precision.' Statistics in
    Medicine, 21, 1331-1335.
Shrout, P. E. and Fleiss, J. L. (1979). 'Intraclass Correlations: Uses
    in Assessing Rater Reliability.' Psychological Bulletin, 86(2),
    420-428.
Tate, R. F. (1954). 'Correlation Between a Discrete and Continuous
    Variable. Point-Biserial Correlation.' Annals of Mathematical
    Statistics, 25(3), 603-607.
Tate, R. F. (1955). 'Applications of Correlation Models for Biserial
    Data.' Journal of the American Statistical Association, 50(272),
    1078-1095.
Fisher, R. A. (1921). 'On the probable error of a coefficient of
    correlation deduced from a small sample.' Metron, i(4), 1-32.
Cohen, J. (1960). 'A Coefficient of Agreement for Nominal Scales.'
    Educational and Psychological Measurement, 20(1), 37-46.
Fleiss, J. L., Cohen, J., and Everitt, B. S. (1969). 'Large Sample
    Standard Errors of Kappa and Weighted Kappa.' Psychological
    Bulletin, 72(5), 323-327.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import f as fdist
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_sides(sides: int) -> None:
    if sides not in (1, 2):
        raise ValueError(f"sides must be 1 or 2, got {sides!r}")


def _validate_interval_side(interval_side: str) -> None:
    if interval_side not in ("upper", "lower"):
        raise ValueError(
            f"interval_side must be 'upper' or 'lower', got {interval_side!r}"
        )


def _z_critical(alpha: float, sides: int) -> float:
    """Two-sided uses 1 - alpha/2; one-sided uses 1 - alpha."""
    return float(norm.ppf(1.0 - alpha / (2.0 if sides == 2 else 1.0)))


def _fisher_ci(r: float, se: float, alpha: float, sides: int,
               interval_side: str) -> tuple[float | None, float | None]:
    """Fisher z-transform back-transformed CI for correlation r."""
    z = _z_critical(alpha, sides)
    zr = math.atanh(r)
    if sides == 2:
        return math.tanh(zr - z * se), math.tanh(zr + z * se)
    if interval_side == "upper":
        return None, math.tanh(zr + z * se)
    return math.tanh(zr - z * se), None


def _measure(lcl: float | None, ucl: float | None, r: float,
             sides: int, interval_side: str) -> float:
    """Two-sided width or one-sided distance from r to the limit."""
    if sides == 2:
        assert lcl is not None and ucl is not None
        return float(ucl - lcl)
    if interval_side == "upper":
        assert ucl is not None
        return float(ucl - r)
    assert lcl is not None
    return float(r - lcl)


def _bsearch_n(predicate, n_min: int, n_max: int = 10_000_000) -> int:
    """Smallest integer N >= n_min with ``predicate(N)`` True (ascending)."""
    lo = n_min
    hi = lo
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
    return hi


# ---------------------------------------------------------------------------
# Shared Bonett-Wright / Fisher-z driver
# ---------------------------------------------------------------------------

def _bw_solve(
    *,
    r: float,
    alpha: float,
    width: float | None,
    distance: float | None,
    n: int | None,
    sides: int,
    interval_side: str,
    solve_for: str | None,
    se_fn,
    n_floor: int,
    method_id: str,
    citations: list[str],
) -> dict[str, Any]:
    if not -1.0 < r < 1.0:
        raise ValueError("r must be in (-1, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    _validate_sides(sides)
    _validate_interval_side(interval_side)

    target = width if sides == 2 else distance
    inputs_echo = {
        "r": r, "alpha": alpha, "width": width, "distance": distance,
        "n": n, "sides": sides, "interval_side": interval_side,
    }

    if solve_for is None:
        solve_for = "n" if n is None else ("width" if sides == 2 else "distance")

    if solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                "supply a positive `width` (sides=2) or `distance` (sides=1)"
            )

        def ok(nn: int) -> bool:
            se = se_fn(r, nn)
            if se is None:
                return False
            lcl, ucl = _fisher_ci(r, se, alpha, sides, interval_side)
            return _measure(lcl, ucl, r, sides, interval_side) <= target

        n_req = _bsearch_n(ok, n_min=n_floor)
    elif solve_for in ("width", "distance"):
        if n is None or n < n_floor:
            raise ValueError(f"supply n >= {n_floor} when solving for width/distance")
        n_req = int(n)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    se = se_fn(r, n_req)
    if se is None:
        raise ValueError(
            f"variance is undefined at n={n_req} (need n > {n_floor - 1})"
        )
    lcl, ucl = _fisher_ci(r, se, alpha, sides, interval_side)
    measure = _measure(lcl, ucl, r, sides, interval_side)
    achieved_width = float(measure) if sides == 2 else None
    achieved_distance = None if sides == 2 else float(measure)

    return {
        "method_id": method_id,
        "solve_for": solve_for,
        "n": int(n_req),
        "r": float(r),
        "lower_limit": (None if lcl is None else float(lcl)),
        "upper_limit": (None if ucl is None else float(ucl)),
        "achieved_width": achieved_width,
        "achieved_distance": achieved_distance,
        "inputs_echo": inputs_echo,
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# 1) Pearson (Fisher z-transform)
# ---------------------------------------------------------------------------

def ci_pearson_correlation(
    *,
    r: float,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    interval_side: str = "upper",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for Pearson's product-moment correlation.

    Implements the Fisher z-transform back-transformation (Fisher 1915;
    chapter 801 (Bonett & Wright, 2000). Variance of ``z_r`` is
    ``1 / (n - 3)``.

    Parameters
    ----------
    r : float
        Planning estimate of the sample Pearson correlation, in ``(-1, 1)``.
    alpha : float
        Confidence level is ``1 - alpha``. Defaults to 0.05.
    width : float, optional
        Target two-sided CI width ``r_U - r_L`` (required when
        ``sides=2`` and solving for N).
    distance : float, optional
        Target one-sided distance from ``r`` to the limit (required when
        ``sides=1`` and solving for N).
    n : int, optional
        Sample size; pass this to compute the achieved width/distance
        rather than to solve for N.
    sides : int
        ``2`` for a two-sided interval or ``1`` for an upper/lower bound.
    interval_side : {"upper", "lower"}
        Side of the one-sided bound; ignored when ``sides == 2``.
    solve_for : {"n", "width", "distance"}, optional
        Forced solve target. Defaults to ``"n"`` if ``n is None`` else
        ``"width"`` (sides=2) or ``"distance"`` (sides=1).
    """
    def se_fn(_r: float, nn: int) -> float | None:
        if nn <= 3:
            return None
        return math.sqrt(1.0 / (nn - 3))

    return _bw_solve(
        r=r, alpha=alpha, width=width, distance=distance, n=n,
        sides=sides, interval_side=interval_side, solve_for=solve_for,
        se_fn=se_fn, n_floor=4,
        method_id="ci_pearson_correlation",
        citations=[
            "Pearson's Correlation.",
            "Bonett, D. G. and Wright, T. A. (2000). 'Sample Size "
            "Requirements for Estimating Pearson, Kendall and Spearman "
            "Correlations.' Psychometrika, 65(1), 23-28.",
            "Fisher, R. A. (1921). 'On the probable error of a coefficient "
            "of correlation deduced from a small sample.' Metron, i(4), 1-32.",
        ],
    )


# ---------------------------------------------------------------------------
# 2) Spearman
# ---------------------------------------------------------------------------

def ci_spearman_correlation(
    *,
    r: float,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    interval_side: str = "upper",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for Spearman's rank correlation (Bonett-Wright).

    variance of ``z_r`` equal to ``(1 + r**2 / 2) / (n - 3)``.
    """
    def se_fn(rr: float, nn: int) -> float | None:
        if nn <= 3:
            return None
        return math.sqrt((1.0 + rr * rr / 2.0) / (nn - 3))

    return _bw_solve(
        r=r, alpha=alpha, width=width, distance=distance, n=n,
        sides=sides, interval_side=interval_side, solve_for=solve_for,
        se_fn=se_fn, n_floor=4,
        method_id="ci_spearman_correlation",
        citations=[
            "Spearman's Rank Correlation.",
            "Bonett, D. G. and Wright, T. A. (2000). Psychometrika, "
            "65(1), 23-28.",
            "Fisher, R. A. (1921). Metron, i(4), 1-32.",
        ],
    )


# ---------------------------------------------------------------------------
# 3) Kendall tau-b
# ---------------------------------------------------------------------------

def ci_kendall_tau_b(
    *,
    r: float,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    interval_side: str = "upper",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for Kendall's tau-b correlation (Bonett-Wright).

    variance of ``z_r`` equal to ``0.437 / (n - 4)``.
    """
    def se_fn(_r: float, nn: int) -> float | None:
        if nn <= 4:
            return None
        return math.sqrt(0.437 / (nn - 4))

    return _bw_solve(
        r=r, alpha=alpha, width=width, distance=distance, n=n,
        sides=sides, interval_side=interval_side, solve_for=solve_for,
        se_fn=se_fn, n_floor=5,
        method_id="ci_kendall_tau_b",
        citations=[
            "Kendall's Tau-b Correlation.",
            "Bonett, D. G. and Wright, T. A. (2000). Psychometrika, "
            "65(1), 23-28.",
            "Fisher, R. A. (1921). Metron, i(4), 1-32.",
        ],
    )


# ---------------------------------------------------------------------------
# 4) Intraclass correlation
# ---------------------------------------------------------------------------

_ICC_MODELS = {
    "one_way_random": 0,            # b = 0
    "two_way_random": 1,            # b = 1
    "two_way_mixed": 1,             # b = 1
}


def _icc_limits(r: float, n: int, k: int, alpha: float, sides: int,
                interval_side: str, model: str
                ) -> tuple[float | None, float | None]:
    """Shrout & Fleiss (1979) / Bonett (2002) F-distribution limits."""
    if model not in _ICC_MODELS:
        raise ValueError(
            f"model must be one of {sorted(_ICC_MODELS)}, got {model!r}"
        )
    b = _ICC_MODELS[model]
    v1 = (n - b) * (k - 1)
    v2 = n - 1
    if v1 <= 0 or v2 <= 0:
        return None, None
    # r = (MSB - MSE) / (MSB + (K-1)*MSE)  =>  F0 = MSB/MSE
    if r >= 1.0 - 1e-15:
        return None, None
    f0 = (1.0 + r * (k - 1)) / (1.0 - r)

    if sides == 2:
        fl = f0 / fdist.ppf(1.0 - alpha / 2.0, v2, v1)
        fu = f0 * fdist.ppf(1.0 - alpha / 2.0, v1, v2)
        rl = (fl - 1.0) / (fl + k - 1)
        ru = (fu - 1.0) / (fu + k - 1)
        return float(rl), float(ru)
    # one-sided
    if interval_side == "upper":
        fu = f0 * fdist.ppf(1.0 - alpha, v1, v2)
        ru = (fu - 1.0) / (fu + k - 1)
        return None, float(ru)
    fl = f0 / fdist.ppf(1.0 - alpha, v2, v1)
    rl = (fl - 1.0) / (fl + k - 1)
    return float(rl), None


def ci_intraclass_correlation(
    *,
    r: float,
    k: int,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    interval_side: str = "upper",
    model: str = "two_way_mixed",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for an intraclass correlation coefficient.

    constructs the F-distribution interval

        F_L = F_0 / F_{1 - alpha/2, V2, V1},  F_U = F_0 * F_{1 - alpha/2, V1, V2}
        r_L = (F_L - 1) / (F_L + K - 1),     r_U = (F_U - 1) / (F_U + K - 1)

    where ``F_0 = (1 + r (K - 1)) / (1 - r)``, ``V1 = (N - b)(K - 1)``
    with ``b = 0`` for the one-way random-effects model and ``b = 1``
    for the two-way (random or mixed) effects models, and ``V2 = N - 1``.

    Parameters
    ----------
    r : float
        Planning estimate of the sample ICC, in ``[0, 1)``.
    k : int
        Number of observations (raters / measurements) per subject.
        Must be >= 2.
    model : {"one_way_random", "two_way_random", "two_way_mixed"}
        ANOVA model that governs the degrees of freedom. The two two-way
        choices are statistically equivalent in this formulation
        (``b = 1``).
    sides : int
        ``2`` for a two-sided interval, ``1`` for a one-sided bound.
    """
    if not 0.0 <= r < 1.0:
        raise ValueError("r must be in [0, 1)")
    if k < 2:
        raise ValueError("k must be >= 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    _validate_sides(sides)
    _validate_interval_side(interval_side)
    if model not in _ICC_MODELS:
        raise ValueError(
            f"model must be one of {sorted(_ICC_MODELS)}, got {model!r}"
        )

    target = width if sides == 2 else distance
    inputs_echo = {
        "r": r, "k": k, "alpha": alpha, "width": width, "distance": distance,
        "n": n, "sides": sides, "interval_side": interval_side, "model": model,
    }

    if solve_for is None:
        solve_for = "n" if n is None else ("width" if sides == 2 else "distance")

    # Floor: V1, V2 must be positive => N >= 2 in any model.
    n_floor = 2

    if solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                "supply a positive `width` (sides=2) or `distance` (sides=1)"
            )

        def ok(nn: int) -> bool:
            lcl, ucl = _icc_limits(r, nn, k, alpha, sides, interval_side, model)
            if (sides == 2 and (lcl is None or ucl is None)) or \
               (sides == 1 and interval_side == "upper" and ucl is None) or \
               (sides == 1 and interval_side == "lower" and lcl is None):
                return False
            return _measure(lcl, ucl, r, sides, interval_side) <= target

        n_req = _bsearch_n(ok, n_min=n_floor)
    elif solve_for in ("width", "distance"):
        if n is None or n < n_floor:
            raise ValueError(f"supply n >= {n_floor} when solving for width/distance")
        n_req = int(n)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    lcl, ucl = _icc_limits(r, n_req, k, alpha, sides, interval_side, model)
    measure = _measure(lcl, ucl, r, sides, interval_side)
    achieved_width = float(measure) if sides == 2 else None
    achieved_distance = None if sides == 2 else float(measure)

    return {
        "method_id": "ci_intraclass_correlation",
        "solve_for": solve_for,
        "n": int(n_req),
        "k": int(k),
        "r": float(r),
        "lower_limit": (None if lcl is None else float(lcl)),
        "upper_limit": (None if ucl is None else float(ucl)),
        "achieved_width": achieved_width,
        "achieved_distance": achieved_distance,
        "inputs_echo": inputs_echo,
        "citations": [
            "Intraclass Correlation.",
            "Shrout, P. E. and Fleiss, J. L. (1979). 'Intraclass "
            "Correlations: Uses in Assessing Rater Reliability.' "
            "Psychological Bulletin, 86(2), 420-428.",
            "Bonett, D. G. (2002). 'Sample size requirements for "
            "estimating intraclass correlations with desired precision.' "
            "Statistics in Medicine, 21, 1331-1335.",
        ],
    }


# ---------------------------------------------------------------------------
# 5) Point-biserial
# ---------------------------------------------------------------------------

def _pb_variance(r: float, n: int, p: float) -> float:
    """Tate (1954) approximate variance of the sample point-biserial r."""
    num = r * r + 2.0 * p * (1.0 - p) * (2.0 - 3.0 * r * r)
    return num / (4.0 * n * p * (1.0 - p)) * (1.0 - r * r) ** 2


def ci_point_biserial_correlation(
    *,
    r: float,
    p: float,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    interval_side: str = "upper",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for a point-biserial correlation (random-design).

    approximation. The variance of the sample point-biserial ``r`` is

        sigma_r^2 = [rho^2 + 2 P (1 - P) (2 - 3 rho^2)] / [4 n P (1 - P)]
                    * (1 - rho^2)^2

    and the symmetric two-sided interval has limits
    ``r ± z_{1 - alpha/2} * sigma_r``. One-sided limits replace ``alpha/2``
    by ``alpha``. Only the random design (X ~ Bernoulli(P)) is supported,
    as described in Tate (1954).

    Parameters
    ----------
    r : float
        Planning estimate of the sample point-biserial correlation, in
        ``(-1, 1)``.
    p : float
        Probability that the dichotomous variable X equals 1; ``P`` in
        Must be in ``(0, 1)``.
    """
    if not -1.0 < r < 1.0:
        raise ValueError("r must be in (-1, 1)")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    _validate_sides(sides)
    _validate_interval_side(interval_side)

    target = width if sides == 2 else distance
    inputs_echo = {
        "r": r, "p": p, "alpha": alpha, "width": width, "distance": distance,
        "n": n, "sides": sides, "interval_side": interval_side,
    }

    if solve_for is None:
        solve_for = "n" if n is None else ("width" if sides == 2 else "distance")

    z = _z_critical(alpha, sides)

    def limits(nn: int) -> tuple[float | None, float | None]:
        if nn < 1:
            return None, None
        sigma = math.sqrt(_pb_variance(r, nn, p))
        if sides == 2:
            return r - z * sigma, r + z * sigma
        if interval_side == "upper":
            return None, r + z * sigma
        return r - z * sigma, None

    n_floor = 4  # Tate notes "n > 25" for the approximation; this is a
                 # mathematical, not practical, floor.

    if solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                "supply a positive `width` (sides=2) or `distance` (sides=1)"
            )

        def ok(nn: int) -> bool:
            lcl, ucl = limits(nn)
            return _measure(lcl, ucl, r, sides, interval_side) <= target

        n_req = _bsearch_n(ok, n_min=n_floor)
    elif solve_for in ("width", "distance"):
        if n is None or n < n_floor:
            raise ValueError(f"supply n >= {n_floor} when solving for width/distance")
        n_req = int(n)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    lcl, ucl = limits(n_req)
    measure = _measure(lcl, ucl, r, sides, interval_side)
    achieved_width = float(measure) if sides == 2 else None
    achieved_distance = None if sides == 2 else float(measure)

    return {
        "method_id": "ci_point_biserial_correlation",
        "solve_for": solve_for,
        "n": int(n_req),
        "r": float(r),
        "p": float(p),
        "lower_limit": (None if lcl is None else float(lcl)),
        "upper_limit": (None if ucl is None else float(ucl)),
        "achieved_width": achieved_width,
        "achieved_distance": achieved_distance,
        "inputs_echo": inputs_echo,
        "citations": [
            "Biserial Correlation.",
            "Tate, R. F. (1954). 'Correlation Between a Discrete and "
            "Continuous Variable. Point-Biserial Correlation.' Annals of "
            "Mathematical Statistics, 25(3), 603-607.",
            "Tate, R. F. (1955). 'Applications of Correlation Models for "
            "Biserial Data.' Journal of the American Statistical "
            "Association, 50(272), 1078-1095.",
        ],
    }


# ---------------------------------------------------------------------------
# 6) Cohen's Kappa
# ---------------------------------------------------------------------------

def _kappa_sd_cohen(kappa: float, po: float) -> float:
    """Cohen (1960) SD(kappa) from kappa and PO.

    PE is derived from kappa = (PO - PE) / (1 - PE)  =>  PE = (PO - kappa) / (1 - kappa).
    SD(kappa) = sqrt(PO * (1 - PO)) / (1 - PE).
    """
    if kappa >= 1.0:
        raise ValueError("kappa must be < 1")
    pe = (po - kappa) / (1.0 - kappa)
    denom = 1.0 - pe
    if denom <= 0:
        raise ValueError("1 - PE must be > 0")
    return math.sqrt(po * (1.0 - po)) / denom


def ci_kappa(
    *,
    kappa: float,
    po: float,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    interval_side: str = "upper",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for Cohen's kappa coefficient of agreement.

    normal approximation.  The standard error of the estimated kappa is

        SE(kappa) = SD(kappa) / sqrt(N)

    where (using Cohen's formula)

        SD(kappa) = sqrt(PO * (1 - PO)) / (1 - PE),
        PE        = (PO - kappa) / (1 - kappa).

    The two-sided CI is ``kappa ± z_{1 - alpha/2} * SE``, giving width
    ``2 * z * SD(kappa) / sqrt(N)``.  We search for the smallest N where
    the achieved width ≤ target width (or distance for one-sided).

    Parameters
    ----------
    kappa : float
        Planning estimate of Cohen's kappa, in ``(-1, 1)``.
    po : float
        Proportion of subjects on which the two raters agree (PO), in
        ``(0, 1)``.  Used with ``kappa`` to derive SD(kappa) via Cohen's
        formula.
    alpha : float
        Significance level; confidence level is ``1 - alpha``.
    width : float, optional
        Target two-sided CI width (required when ``sides=2`` and solving
        for N).
    distance : float, optional
        Target one-sided distance from kappa to limit (required when
        ``sides=1`` and solving for N).
    n : int, optional
        Fixed sample size; pass to compute achieved width/distance rather
        than to solve for N.
    sides : int
        ``2`` for two-sided, ``1`` for one-sided.
    interval_side : {"upper", "lower"}
        Side of the one-sided bound; ignored when ``sides == 2``.
    solve_for : {"n", "width", "distance"}, optional
        Forced solve target.  Defaults to ``"n"`` when ``n is None``.
    """
    if not -1.0 < kappa < 1.0:
        raise ValueError("kappa must be in (-1, 1)")
    if not 0.0 < po < 1.0:
        raise ValueError("po must be in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    _validate_sides(sides)
    _validate_interval_side(interval_side)

    sd_kappa = _kappa_sd_cohen(kappa, po)
    pe = (po - kappa) / (1.0 - kappa)

    target = width if sides == 2 else distance
    inputs_echo = {
        "kappa": kappa, "po": po, "alpha": alpha, "width": width,
        "distance": distance, "n": n, "sides": sides,
        "interval_side": interval_side,
    }

    if solve_for is None:
        solve_for = "n" if n is None else ("width" if sides == 2 else "distance")

    z = _z_critical(alpha, sides)

    def _achieved(nn: int) -> tuple[float, float, float]:
        """Return (achieved_width_or_dist, lcl, ucl) at sample size nn."""
        se = sd_kappa / math.sqrt(nn)
        if sides == 2:
            lcl = kappa - z * se
            ucl = kappa + z * se
            return ucl - lcl, lcl, ucl
        if interval_side == "upper":
            ucl = kappa + z * se
            return ucl - kappa, kappa, ucl
        lcl = kappa - z * se
        return kappa - lcl, lcl, kappa

    n_floor = 1

    if solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                "supply a positive `width` (sides=2) or `distance` (sides=1)"
            )

        def ok(nn: int) -> bool:
            achieved, _, _ = _achieved(nn)
            return achieved <= target

        n_req = _bsearch_n(ok, n_min=n_floor)
    elif solve_for in ("width", "distance"):
        if n is None or n < n_floor:
            raise ValueError(f"supply n >= {n_floor} when solving for width/distance")
        n_req = int(n)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    measure, lcl, ucl = _achieved(n_req)
    achieved_width = float(measure) if sides == 2 else None
    achieved_distance = None if sides == 2 else float(measure)

    return {
        "method_id": "ci_kappa",
        "solve_for": solve_for,
        "n": int(n_req),
        "kappa": float(kappa),
        "po": float(po),
        "pe": float(pe),
        "sd_kappa": float(sd_kappa),
        "lower_limit": float(lcl),
        "upper_limit": float(ucl),
        "achieved_width": achieved_width,
        "achieved_distance": achieved_distance,
        "achieved_power": None,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cohen, J. (1960). 'A Coefficient of Agreement for Nominal "
            "Scales.' Educational and Psychological Measurement, 20(1), "
            "37-46.",
            "Fleiss, J. L., Cohen, J., and Everitt, B. S. (1969). 'Large "
            "Sample Standard Errors of Kappa and Weighted Kappa.' "
            "Psychological Bulletin, 72(5), 323-327.",
        ],
    }
