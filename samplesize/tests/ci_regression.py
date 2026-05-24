"""Confidence-interval sample-size routines for regression parameters.


- Ch 856: Confidence Intervals for Linear Regression Slope
- Ch 857: Confidence Intervals for Michaelis-Menten Parameters

Both solve for the smallest sample size that produces a confidence
interval no farther from the point estimate to the limit than a target
distance (two-sided: half-width = target distance; one-sided: signed
distance). Each callable also supports ``solve_for="power"``, which
*evaluates the achieved half-width at fixed N* (sometimes called
*Actual Distance from Slope to Limit* or *CI Width*).
"""
from __future__ import annotations

import math
from typing import Any, Sequence


_CIT_SLOPE = [
    "Ostle, B. and Malone, L.C. (1988). Statistics in Research. "
    "Iowa State University Press, Ames, Iowa.",
]
_CIT_MM = [
    "Parameters",
    "Raaijmakers, J. G. W. (1987). Statistical Analysis of the "
    "Michaelis-Menten Equation. Biometrics, 43(4), 793-803.",
]


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")


def _check_sides(sides: int) -> None:
    if sides not in (1, 2):
        raise ValueError(f"sides must be 1 or 2; got {sides}")


def _norm_ppf(q: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(q))


def _t_ppf(q: float, df: float) -> float:
    from scipy.stats import t
    return float(t.ppf(q, df))


def _t_for(alpha: float, sides: int, df: float) -> float:
    return _t_ppf(1.0 - (alpha / 2.0 if sides == 2 else alpha), df)


def _z_for(alpha: float, sides: int) -> float:
    return _norm_ppf(1.0 - (alpha / 2.0 if sides == 2 else alpha))


# ===========================================================================
# Ch 856 - Confidence Intervals for Linear Regression Slope
# ===========================================================================

_SLOPE_RESID_METHODS = {"s", "sy", "r"}


def _resid_sd(method: str, b: float, sx: float,
              s: float | None, sy: float | None,
              r: float | None, n: int) -> float:
    """Return s (SD of residuals, divisor n-2) given the specified input mode."""
    if n <= 2:
        raise ValueError("n must be >= 3 for slope CI (need n-2 dof)")
    if method == "s":
        if s is None or s <= 0:
            raise ValueError("residual_method='s' requires positive s")
        return float(s)
    if method == "sy":
        if sy is None or sy <= 0:
            raise ValueError("residual_method='sy' requires positive sy")
        inner = sy * sy - (b * sx) ** 2
        if inner <= 0:
            raise ValueError(
                "sy^2 - (b*sx)^2 must be > 0 (sy too small relative to b*sx)"
            )
        return math.sqrt(inner * (n - 1) / (n - 2))
    if method == "r":
        if r is None or not (0.0 < r < 1.0):
            raise ValueError("residual_method='r' requires 0 < r < 1")
        inner = 1.0 / (r * r) - 1.0
        return abs(b) * sx * math.sqrt(inner * (n - 1) / (n - 2))
    raise ValueError(
        f"residual_method must be one of {sorted(_SLOPE_RESID_METHODS)}; "
        f"got {method!r}"
    )


def _slope_distance(b: float, sx: float, n: int, alpha: float,
                    sides: int, residual_method: str,
                    s: float | None, sy: float | None,
                    r: float | None) -> float:
    """Half-width D = t_{1-alpha/2, n-2} * s / sqrt((n-1)*sx^2)."""
    if n <= 2:
        return math.inf
    s_eff = _resid_sd(residual_method, b, sx, s, sy, r, n)
    t_val = _t_for(alpha, sides, n - 2)
    denom = math.sqrt((n - 1) * sx * sx)
    return t_val * s_eff / denom


def ci_linear_regression_slope(
    *,
    b: float,
    sx: float,
    alpha: float = 0.05,
    distance: float | None = None,
    width: float | None = None,
    n: int | None = None,
    sides: int = 2,
    residual_method: str = "s",
    s: float | None = None,
    sy: float | None = None,
    r: float | None = None,
    solve_for: str | None = None,
    n_min: int = 3,
    n_max: int = 5_000_000,
) -> dict[str, Any]:
    """Sample size / achieved distance for a simple-linear-regression
    slope confidence interval.
    """
    _check_alpha(alpha)
    _check_sides(sides)
    if sx <= 0:
        raise ValueError("sx must be > 0")
    if residual_method not in _SLOPE_RESID_METHODS:
        raise ValueError(
            f"residual_method must be one of "
            f"{sorted(_SLOPE_RESID_METHODS)}"
        )

    target = distance
    if target is None and width is not None:
        if sides != 2:
            raise ValueError("width can only be supplied when sides=2")
        target = width / 2.0

    if solve_for is None:
        solve_for = "n" if n is None else "power"

    inputs_echo = {
        "b": b, "sx": sx, "alpha": alpha, "distance": distance,
        "width": width, "n": n, "sides": sides,
        "residual_method": residual_method,
        "s": s, "sy": sy, "r": r,
    }

    def D_at(k: int) -> float:
        return _slope_distance(b, sx, k, alpha, sides,
                               residual_method, s, sy, r)

    if solve_for == "power":
        if n is None:
            raise ValueError("solve_for='power' requires n")
        D = D_at(int(n))
        out = {
            "n": int(n),
            "achieved_distance": D,
            "achieved_width": 2.0 * D,
        }
    elif solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                "solve_for='n' requires positive distance or width"
            )
        lo, hi = n_min, n_min
        while hi <= n_max:
            if D_at(hi) <= target:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError(
                f"could not bracket N within {n_max} for distance={target}"
            )
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if D_at(mid) <= target:
                hi = mid
            else:
                lo = mid
        n_req = hi
        D = D_at(n_req)
        out = {
            "n": int(n_req),
            "target_distance": float(target),
            "achieved_distance": D,
            "achieved_width": 2.0 * D,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_linear_regression_slope",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CIT_SLOPE,
    }


# ===========================================================================
# Ch 857 - Confidence Intervals for Michaelis-Menten Parameters
# ===========================================================================

def _mm_design_arrays(
    c_values: Sequence[float],
    allocation: Sequence[int] | None,
    n_total: int | None,
    n_per: int | None,
    multipliers: Sequence[float] | None,
    percentages: Sequence[float] | None,
    base_n: int | None,
) -> tuple[list[float], list[int]]:
    """Resolve the C-values and per-C sample sizes."""
    c_list = [float(c) for c in c_values]
    if any(c <= 0 for c in c_list):
        raise ValueError("all C values must be > 0")
    k = len(c_list)
    if k < 3:
        raise ValueError("Michaelis-Menten requires at least 3 C values")

    if allocation is not None:
        sizes = [int(round(x)) for x in allocation]
    elif n_per is not None:
        sizes = [int(round(n_per))] * k
    elif multipliers is not None and base_n is not None:
        sizes = [int(math.ceil(base_n * m)) for m in multipliers]
    elif percentages is not None and n_total is not None:
        sizes = [int(math.ceil(n_total * p / 100.0)) for p in percentages]
    elif n_total is not None:
        # equal allocation
        each = n_total // k
        sizes = [each] * k
    else:
        raise ValueError(
            "must supply one of: allocation, n_per, "
            "(multipliers + base_n), (percentages + n_total), n_total"
        )

    if len(sizes) != k:
        raise ValueError(
            f"allocation length {len(sizes)} does not match number of C "
            f"values {k}"
        )
    if any(x < 1 for x in sizes):
        raise ValueError("each per-C sample size must be >= 1")
    return c_list, sizes


def _mm_variances(c_values: Sequence[float], sizes: Sequence[int],
                  vmax: float, km: float,
                  sigma: float) -> tuple[float, float, float, float]:
    """Return (var_Vmax, var_Km, SE_Vmax, SE_Km) per Raaijmakers (1987).

    U_i = Vmax / (C_i + Km);
    U_bar = sum(n_i * U_i) / N;
    S_UU  = sum(n_i * (U_i - U_bar)^2);
    var(Km)   = sigma^2 / [(1 + 2 sigma^2 / Vmax^2) * S_UU];
    var(Vmax) = sigma^2 / N + U_bar^2 * var(Km).
    """
    if vmax <= 0 or km <= 0 or sigma <= 0:
        raise ValueError("vmax, km, sigma must all be > 0")
    n_total = sum(sizes)
    if n_total < 2:
        raise ValueError("total sample size must be >= 2")
    u = [vmax / (c + km) for c in c_values]
    u_bar = sum(n_i * u_i for n_i, u_i in zip(sizes, u)) / n_total
    s_uu = sum(n_i * (u_i - u_bar) ** 2 for n_i, u_i in zip(sizes, u))
    if s_uu <= 0:
        raise ValueError(
            "design is degenerate: S_UU = 0 (C values do not vary)"
        )
    sigma2 = sigma * sigma
    var_km = sigma2 / ((1.0 + 2.0 * sigma2 / (vmax * vmax)) * s_uu)
    var_vmax = sigma2 / n_total + (u_bar ** 2) * var_km
    se_km = math.sqrt(max(var_km, 0.0))
    se_vmax = math.sqrt(max(var_vmax, 0.0))
    return var_vmax, var_km, se_vmax, se_km


def ci_michaelis_menten(
    *,
    c_values: Sequence[float],
    vmax: float,
    km: float,
    sigma: float,
    alpha: float = 0.05,
    n: int | None = None,
    n_per: int | None = None,
    allocation: Sequence[int] | None = None,
    n_total: int | None = None,
    multipliers: Sequence[float] | None = None,
    percentages: Sequence[float] | None = None,
    base_n: int | None = None,
    width_vmax: float | None = None,
    width_km: float | None = None,
    sides: int = 2,
    target: str = "vmax",
    solve_for: str | None = None,
    n_min: int = 1,
    n_max: int = 5_000_000,
) -> dict[str, Any]:
    """Sample size / achieved widths for the Michaelis-Menten CI."""
    _check_alpha(alpha)
    _check_sides(sides)
    if target not in ("vmax", "km"):
        raise ValueError("target must be 'vmax' or 'km'")
    c_list = [float(c) for c in c_values]
    if any(c <= 0 for c in c_list):
        raise ValueError("all C values must be > 0")
    if len(c_list) < 3:
        raise ValueError("Michaelis-Menten requires at least 3 C values")
    k = len(c_list)

    if solve_for is None:
        if (allocation is not None or n_per is not None
                or n is not None
                or (multipliers is not None and base_n is not None)
                or (percentages is not None and n_total is not None)
                or n_total is not None):
            solve_for = "power"
        else:
            solve_for = "n"

    inputs_echo = {
        "c_values": c_list, "vmax": vmax, "km": km, "sigma": sigma,
        "alpha": alpha, "n": n, "n_per": n_per,
        "allocation": list(allocation) if allocation else None,
        "n_total": n_total,
        "multipliers": list(multipliers) if multipliers else None,
        "percentages": list(percentages) if percentages else None,
        "base_n": base_n, "width_vmax": width_vmax,
        "width_km": width_km, "sides": sides, "target": target,
    }

    z = _z_for(alpha, sides)

    def _compute(sizes: list[int]) -> dict[str, Any]:
        var_vmax, var_km, se_vmax, se_km = _mm_variances(
            c_list, sizes, vmax, km, sigma
        )
        if sides == 2:
            w_vmax = 2.0 * z * se_vmax
            w_km = 2.0 * z * se_km
        else:
            w_vmax = z * se_vmax
            w_km = z * se_km
        return {
            "n": int(sum(sizes)),
            "allocation": list(sizes),
            "se_vmax": se_vmax,
            "se_km": se_km,
            "achieved_width_vmax": w_vmax,
            "achieved_width_km": w_km,
            "achieved_distance_vmax": z * se_vmax,
            "achieved_distance_km": z * se_km,
        }

    if solve_for == "power":
        _, sizes = _mm_design_arrays(
            c_list, allocation, n_total, n_per,
            multipliers, percentages, base_n,
        )
        out = _compute(sizes)
    elif solve_for == "n":
        target_width = width_vmax if target == "vmax" else width_km
        if target_width is None or target_width <= 0:
            raise ValueError(
                f"solve_for='n' with target={target!r} requires positive "
                f"width_{target}"
            )

        def width_at(k_per: int) -> float:
            sizes = [k_per] * k
            r = _compute(sizes)
            return (r["achieved_width_vmax"] if target == "vmax"
                    else r["achieved_width_km"])

        lo, hi = n_min, n_min
        while hi <= n_max:
            if width_at(hi) <= target_width:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError(
                f"could not bracket N within {n_max} for {target} width "
                f"{target_width}"
            )
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if width_at(mid) <= target_width:
                hi = mid
            else:
                lo = mid
        sizes = [hi] * k
        out = _compute(sizes)
        out[f"target_width_{target}"] = float(target_width)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_michaelis_menten",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CIT_MM,
    }


__all__ = [
    "ci_linear_regression_slope",
    "ci_michaelis_menten",
]
