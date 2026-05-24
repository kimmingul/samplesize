"""Confidence-interval sample-size routines for proportions.


- Ch 115: Confidence Intervals for One Proportion
- Ch 116: Confidence Intervals for One Proportion from a Finite Population
- Ch 216: Confidence Intervals for the Difference Between Two Proportions
- Ch 217: Confidence Intervals for the Ratio of Two Proportions
- Ch 218: Confidence Intervals for the Odds Ratio of Two Proportions

All five solve for the smallest sample size that produces a confidence
interval no wider than a target width (two-sided) or no farther from the
estimate to the limit than a target distance (one-sided). Each callable
also supports ``solve_for="power"``, which here means *evaluate the
achieved width / distance at fixed N* (sometimes called *Actual Width* or
*Actual Distance from P to Limit*).

For each chapter the implemented closed-form / inversion-based methods
were chosen to cover standard worked examples:

* one-proportion:    Wald (simple asymptotic), Wald-CC, Wilson (score),
                     Wilson-CC, Clopper-Pearson exact.
* finite population: normal approximation to the hypergeometric with
                     the finite-population correction (Machin et al.).
* difference:        Pearson chi-square (Wald), Yates CC, Wilson score
                     (Newcombe Method 10), Wilson score CC (Newcombe
                     Method 11).
* ratio:             Katz log, Walters log+1/2.
* odds ratio:        log-Wald (Simple+1/2 / Fleiss-style continuity).

The score-with-skewness (Gart & Nam) and the iterated Fleiss interval
are not implemented in the MVP and raise ``NotImplementedError``.
"""
from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CIT_ONE_PROP = [
    "Newcombe, R. G. (1998). Two-Sided Confidence Intervals for the "
    "Single Proportion: Comparison of Seven Methods. Stat. Med., 17, "
    "857-872.",
    "Fleiss, J. L., Levin, B., Paik, M. C. (2003). Statistical Methods "
    "for Rates and Proportions, 3rd ed. Wiley.",
]
_CIT_ONE_PROP_FIN = [
    "from a Finite Population",
    "Machin, D., Campbell, M., Tan, S. B., Tan, S. H. (2009). Sample "
    "Size Tables for Clinical Studies, 3rd ed. Wiley-Blackwell.",
]
_CIT_DIFF = [
    "Between Two Proportions",
    "Newcombe, R. G. (1998). Interval Estimation for the Difference "
    "Between Independent Proportions: Comparison of Eleven Methods. "
    "Stat. Med., 17, 873-890.",
]
_CIT_RATIO = [
    "Proportions",
    "Katz, D., Baptista, J., Azen, S. P., Pike, M. C. (1978). Obtaining "
    "Confidence Intervals for the Risk Ratio in Cohort Studies. "
    "Biometrics, 34, 469-474.",
    "Gart, J. J. & Nam, J. (1988). Approximate Interval Estimation of "
    "the Ratio of Binomial Parameters. Biometrics, 44, 323-338.",
]
_CIT_OR = [
    "Two Proportions",
    "Fleiss, J. L., Levin, B., Paik, M. C. (2003). Statistical Methods "
    "for Rates and Proportions, 3rd ed. Wiley.",
]


def _norm_ppf(q: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(q))


def _beta_ppf(q: float, a: float, b: float) -> float:
    from scipy.stats import beta
    return float(beta.ppf(q, a, b))


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")


def _check_sides(sides: int) -> None:
    if sides not in (1, 2):
        raise ValueError(f"sides must be 1 or 2; got {sides}")


def _z_for(alpha: float, sides: int) -> float:
    return _norm_ppf(1.0 - (alpha / 2.0 if sides == 2 else alpha))


def _bisect_n(width_at, target: float, n_min: int = 2,
              n_max: int = 5_000_000) -> int:
    """Find smallest N (integer) such that width_at(N) <= target.

    Width is assumed monotonically decreasing in N.
    """
    lo, hi = n_min, n_min
    while hi <= n_max:
        if width_at(hi) <= target:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"could not bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if width_at(mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi


# ===========================================================================
# Ch 115 - Confidence Intervals for One Proportion
# ===========================================================================

_ONE_PROP_METHODS = {
    "wald",                # Simple Asymptotic
    "wald_cc",             # Simple Asymptotic w/ continuity correction
    "wilson",              # Score (Wilson)
    "wilson_cc",           # Score with continuity correction
    "clopper_pearson",     # Exact (Clopper-Pearson)
}


def _wald_limits(p: float, n: int, alpha: float, sides: int,
                 tail: str, cc: bool) -> tuple[float, float]:
    z = _z_for(alpha, sides)
    se = math.sqrt(p * (1.0 - p) / n)
    half = z * se + (1.0 / (2.0 * n) if cc else 0.0)
    L = max(0.0, p - half)
    U = min(1.0, p + half)
    if sides == 1:
        if tail == "upper":
            L = 0.0
        elif tail == "lower":
            U = 1.0
        else:
            raise ValueError("tail must be 'upper' or 'lower' for sides=1")
    return L, U


def _wilson_limits(p: float, n: int, alpha: float, sides: int,
                   tail: str, cc: bool) -> tuple[float, float]:
    z = _z_for(alpha, sides)
    if not cc:
        a = 1.0 + z * z / n
        b = -(2.0 * p + z * z / n)
        c = p * p
        disc = max(b * b - 4.0 * a * c, 0.0)
        L = (-b - math.sqrt(disc)) / (2.0 * a)
        U = (-b + math.sqrt(disc)) / (2.0 * a)
    else:
        # Newcombe / Score with continuity correction
        inner_lo = z * z - 2.0 - 1.0 / n + 4.0 * p * (n * (1.0 - p) + 1.0)
        inner_up = z * z + 2.0 - 1.0 / n + 4.0 * p * (n * (1.0 - p) - 1.0)
        denom = 2.0 * (n + z * z)
        L = ((2.0 * n * p + z * z - 1.0) - z * math.sqrt(max(inner_lo, 0.0))) / denom
        U = ((2.0 * n * p + z * z + 1.0) + z * math.sqrt(max(inner_up, 0.0))) / denom
    L = max(0.0, L)
    U = min(1.0, U)
    if sides == 1:
        if tail == "upper":
            L = 0.0
        elif tail == "lower":
            U = 1.0
        else:
            raise ValueError("tail must be 'upper' or 'lower' for sides=1")
    return L, U


def _clopper_pearson_limits(p: float, n: int, alpha: float,
                            sides: int, tail: str) -> tuple[float, float]:
    """Exact (Clopper-Pearson) limits at p_hat = p (so r = round(p*n)).

    Uses the beta-quantile representation.
    """
    r = int(round(p * n))
    if sides == 2:
        a = alpha / 2.0
    else:
        a = alpha
    if r == 0:
        L = 0.0
    else:
        L = _beta_ppf(a, r, n - r + 1)
    if r == n:
        U = 1.0
    else:
        U = _beta_ppf(1.0 - a, r + 1, n - r)
    if sides == 1:
        if tail == "upper":
            L = 0.0
        elif tail == "lower":
            U = 1.0
        else:
            raise ValueError("tail must be 'upper' or 'lower' for sides=1")
    return L, U


def _one_prop_limits(method: str, p: float, n: int, alpha: float,
                     sides: int, tail: str) -> tuple[float, float]:
    if method == "wald":
        return _wald_limits(p, n, alpha, sides, tail, cc=False)
    if method == "wald_cc":
        return _wald_limits(p, n, alpha, sides, tail, cc=True)
    if method == "wilson":
        return _wilson_limits(p, n, alpha, sides, tail, cc=False)
    if method == "wilson_cc":
        return _wilson_limits(p, n, alpha, sides, tail, cc=True)
    if method == "clopper_pearson":
        return _clopper_pearson_limits(p, n, alpha, sides, tail)
    raise ValueError(
        f"method must be one of {sorted(_ONE_PROP_METHODS)}; got {method!r}"
    )


def _one_prop_metric(method: str, p: float, n: int, alpha: float,
                     sides: int, tail: str) -> tuple[float, float, float]:
    """Return (width-or-distance, lower, upper) at fixed N."""
    L, U = _one_prop_limits(method, p, n, alpha, sides, tail)
    if sides == 2:
        return U - L, L, U
    if tail == "upper":
        return U - p, L, U
    if tail == "lower":
        return p - L, L, U
    raise ValueError("unreachable")


def ci_one_proportion(
    *,
    p: float,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    tail: str = "two",
    method: str = "wald",
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 5_000_000,
) -> dict[str, Any]:
    """Sample size / achieved width for a one-proportion CI.

    Parameters
    ----------
    p : float
        Assumed sample proportion.
    alpha : float
        One minus the confidence level.
    width : float, optional
        Target two-sided interval width (required for sides=2 when solving
        for N).
    distance : float, optional
        Target one-sided distance from P to limit (required for sides=1
        when solving for N).
    n : int, optional
        Fixed sample size for ``solve_for='power'``.
    sides : {1, 2}
        Two-sided width or one-sided distance.
    tail : {'two', 'upper', 'lower'}
        For ``sides=1`` choose the upper or lower limit. Ignored when
        ``sides=2``.
    method : str
        One of ``{"wald", "wald_cc", "wilson", "wilson_cc",
        "clopper_pearson"}``.
    solve_for : {'n', 'power'}, optional
        Solve for sample size (default) or evaluate the interval at a
        fixed N.
    """
    _check_alpha(alpha)
    _check_sides(sides)
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if method not in _ONE_PROP_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_ONE_PROP_METHODS)}"
        )

    if sides == 2:
        target = width
        target_key = "width"
    else:
        target = distance
        target_key = "distance"
        if tail not in ("upper", "lower"):
            raise ValueError("for sides=1 set tail='upper' or 'lower'")

    if solve_for is None:
        solve_for = "n" if n is None else "power"

    inputs_echo = {
        "p": p, "alpha": alpha, "width": width, "distance": distance,
        "n": n, "sides": sides, "tail": tail, "method": method,
    }

    if solve_for == "power":
        if n is None:
            raise ValueError("solve_for='power' requires n")
        actual, L, U = _one_prop_metric(method, p, int(n), alpha,
                                        sides, tail)
        out = {
            "n": int(n),
            f"achieved_{target_key}": actual,
            "lower_limit": L,
            "upper_limit": U,
        }
    elif solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                f"solve_for='n' requires positive {target_key}"
            )

        def width_at(k: int) -> float:
            w, _, _ = _one_prop_metric(method, p, k, alpha, sides, tail)
            return w

        n_req = _bisect_n(width_at, float(target),
                          n_min=n_min, n_max=n_max)
        actual, L, U = _one_prop_metric(method, p, n_req, alpha,
                                        sides, tail)
        out = {
            "n": int(n_req),
            f"target_{target_key}": float(target),
            f"achieved_{target_key}": actual,
            "lower_limit": L,
            "upper_limit": U,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_one_proportion",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CIT_ONE_PROP,
    }


# ===========================================================================
# Ch 116 - Confidence Intervals for One Proportion from a Finite Population
# ===========================================================================

def _finite_d(p: float, n: int, N: int, alpha: float, sides: int) -> float:
    if n >= N:
        return 0.0
    z = _z_for(alpha, sides)
    return z * math.sqrt(p * (1.0 - p) * (N - n) / (n * N))


def ci_one_proportion_finite(
    *,
    p: float,
    N: int,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n: int | None = None,
    sides: int = 2,
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int | None = None,
) -> dict[str, Any]:
    """One-proportion CI with finite-population correction.

    Implements Machin et al. (2009): the normal approximation to the
    hypergeometric with FPC = sqrt((N - n) / N). The function reports the
    half-width as the *precision*, ``d``. For ``sides=2`` users may
    equivalently supply ``width = 2*d``.
    """
    _check_alpha(alpha)
    _check_sides(sides)
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if N < 2:
        raise ValueError("N (population size) must be >= 2")
    if n_max is None:
        n_max = N

    #   width: full two-sided width (W = 2*d).
    #   distance: half-width d (one-sided precision).
    if width is not None and distance is not None:
        raise ValueError("supply only one of (width, distance)")
    if width is not None:
        d_target = width / 2.0
    else:
        d_target = distance

    if solve_for is None:
        solve_for = "n" if n is None else "power"

    inputs_echo = {
        "p": p, "N": N, "alpha": alpha, "width": width,
        "distance": distance, "n": n, "sides": sides,
    }

    if solve_for == "power":
        if n is None:
            raise ValueError("solve_for='power' requires n")
        d = _finite_d(p, int(n), N, alpha, sides)
        out = {
            "n": int(n),
            "achieved_distance": d,
            "achieved_width": 2.0 * d,
            "lower_limit": max(0.0, p - d),
            "upper_limit": min(1.0, p + d),
        }
    elif solve_for == "n":
        if d_target is None or d_target <= 0:
            raise ValueError(
                "solve_for='n' requires positive width (W=2d) or distance"
            )

        def width_at(k: int) -> float:
            return _finite_d(p, k, N, alpha, sides)

        n_req = _bisect_n(width_at, float(d_target),
                          n_min=n_min, n_max=n_max)
        d = _finite_d(p, n_req, N, alpha, sides)
        out = {
            "n": int(n_req),
            "target_distance": float(d_target),
            "target_width": 2.0 * float(d_target),
            "achieved_distance": d,
            "achieved_width": 2.0 * d,
            "lower_limit": max(0.0, p - d),
            "upper_limit": min(1.0, p + d),
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_one_proportion_finite",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CIT_ONE_PROP_FIN,
    }


# ===========================================================================
# Ch 216 - Confidence Intervals for the Difference Between Two Proportions
# ===========================================================================

_DIFF_METHODS = {
    "pearson",      # simple asymptotic (Wald)
    "yates",        # Pearson + continuity correction
    "wilson",       # Newcombe Method 10 (Wilson score)
    "wilson_cc",    # Newcombe Method 11 (Wilson CC score)
}


def _wilson_roots_for_diff(p_hat: float, n: int, z: float,
                           cc: bool) -> tuple[float, float]:
    if not cc:
        a = 1.0 + z * z / n
        b = -(2.0 * p_hat + z * z / n)
        c = p_hat * p_hat
        disc = max(b * b - 4.0 * a * c, 0.0)
        L = (-b - math.sqrt(disc)) / (2.0 * a)
        U = (-b + math.sqrt(disc)) / (2.0 * a)
    else:
        inner_lo = z * z - 2.0 - 1.0 / n + 4.0 * p_hat * (n * (1.0 - p_hat) + 1.0)
        inner_up = z * z + 2.0 - 1.0 / n + 4.0 * p_hat * (n * (1.0 - p_hat) - 1.0)
        denom = 2.0 * (n + z * z)
        L = ((2.0 * n * p_hat + z * z - 1.0) - z * math.sqrt(max(inner_lo, 0.0))) / denom
        U = ((2.0 * n * p_hat + z * z + 1.0) + z * math.sqrt(max(inner_up, 0.0))) / denom
    return max(0.0, L), min(1.0, U)


def _diff_limits(method: str, p1: float, p2: float, n1: int, n2: int,
                 alpha: float, sides: int, tail: str
                 ) -> tuple[float, float]:
    z = _z_for(alpha, sides)
    diff = p1 - p2
    if method == "pearson":
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        half = z * se
        L = max(-1.0, diff - half)
        U = min(1.0, diff + half)
    elif method == "yates":
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        cc = 0.5 * (1.0 / n1 + 1.0 / n2)
        L = max(-1.0, diff - z * se - cc)
        U = min(1.0, diff + z * se + cc)
    elif method in ("wilson", "wilson_cc"):
        cc = method == "wilson_cc"
        l1, u1 = _wilson_roots_for_diff(p1, n1, z, cc)
        l2, u2 = _wilson_roots_for_diff(p2, n2, z, cc)
        B = math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
        C = math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
        L = diff - B
        U = diff + C
        L = max(-1.0, L)
        U = min(1.0, U)
    else:
        raise ValueError(
            f"method must be one of {sorted(_DIFF_METHODS)}; got {method!r}"
        )
    if sides == 1:
        if tail == "upper":
            L = -1.0
        elif tail == "lower":
            U = 1.0
        else:
            raise ValueError("tail must be 'upper' or 'lower' for sides=1")
    return L, U


def _diff_metric(method: str, p1: float, p2: float, n1: int, n2: int,
                 alpha: float, sides: int, tail: str
                 ) -> tuple[float, float, float]:
    L, U = _diff_limits(method, p1, p2, n1, n2, alpha, sides, tail)
    diff = p1 - p2
    if sides == 2:
        return U - L, L, U
    if tail == "upper":
        return U - diff, L, U
    if tail == "lower":
        return diff - L, L, U
    raise ValueError("unreachable")


def ci_difference_two_proportions(
    *,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation_ratio: float = 1.0,
    sides: int = 2,
    tail: str = "two",
    method: str = "pearson",
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 5_000_000,
) -> dict[str, Any]:
    """Sample size / achieved width for CI on p1 - p2.

    The search assumes ``n2 = ceil(allocation_ratio * n1)`` when both
    sizes are solved, or fixes ``n2`` and searches over ``n1`` when
    ``n2`` is supplied.
    """
    _check_alpha(alpha)
    _check_sides(sides)
    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("p1 and p2 must be in (0, 1)")
    if method not in _DIFF_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_DIFF_METHODS)}; got {method!r}"
        )
    if allocation_ratio <= 0:
        raise ValueError("allocation_ratio must be positive")

    if sides == 2:
        target = width
        target_key = "width"
    else:
        target = distance
        target_key = "distance"
        if tail not in ("upper", "lower"):
            raise ValueError("for sides=1 set tail='upper' or 'lower'")

    if solve_for is None:
        solve_for = "power" if (n1 is not None and n2 is not None) else "n"

    inputs_echo = {
        "p1": p1, "p2": p2, "alpha": alpha, "width": width,
        "distance": distance, "n1": n1, "n2": n2,
        "allocation_ratio": allocation_ratio, "sides": sides,
        "tail": tail, "method": method,
    }

    if solve_for == "power":
        if n1 is None or n2 is None:
            raise ValueError("solve_for='power' requires both n1 and n2")
        actual, L, U = _diff_metric(method, p1, p2, int(n1), int(n2),
                                    alpha, sides, tail)
        out = {
            "n1": int(n1), "n2": int(n2), "n": int(n1) + int(n2),
            f"achieved_{target_key}": actual,
            "lower_limit": L, "upper_limit": U,
        }
    elif solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                f"solve_for='n' requires positive {target_key}"
            )

        if n2 is not None and n1 is None:
            # search n1 with n2 fixed
            fixed_n2 = int(n2)

            def width_at(k: int) -> float:
                w, _, _ = _diff_metric(method, p1, p2, k, fixed_n2,
                                       alpha, sides, tail)
                return w

            n1_req = _bisect_n(width_at, float(target),
                               n_min=n_min, n_max=n_max)
            n2_req = fixed_n2
        elif n1 is not None and n2 is None:
            fixed_n1 = int(n1)

            def width_at(k: int) -> float:
                w, _, _ = _diff_metric(method, p1, p2, fixed_n1, k,
                                       alpha, sides, tail)
                return w

            n2_req = _bisect_n(width_at, float(target),
                               n_min=n_min, n_max=n_max)
            n1_req = fixed_n1
        else:
            def width_at(k: int) -> float:
                n1k = k
                n2k = max(2, math.ceil(allocation_ratio * k))
                w, _, _ = _diff_metric(method, p1, p2, n1k, n2k,
                                       alpha, sides, tail)
                return w

            n1_req = _bisect_n(width_at, float(target),
                               n_min=n_min, n_max=n_max)
            n2_req = max(2, math.ceil(allocation_ratio * n1_req))

        actual, L, U = _diff_metric(method, p1, p2, n1_req, n2_req,
                                    alpha, sides, tail)
        out = {
            "n1": int(n1_req), "n2": int(n2_req),
            "n": int(n1_req) + int(n2_req),
            f"target_{target_key}": float(target),
            f"achieved_{target_key}": actual,
            "lower_limit": L, "upper_limit": U,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_difference_two_proportions",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CIT_DIFF,
    }


# ===========================================================================
# Ch 217 - Confidence Intervals for the Ratio of Two Proportions
# ===========================================================================

_RATIO_METHODS = {
    "katz",       # log(p1/p2) +/- z*sqrt(q1/(n1*p1) + q2/(n2*p2))
    "walters",    # Walters log+0.5: uses x_i + 0.5, n_i + 0.5 style.
}


def _ratio_limits(method: str, p1: float, p2: float, n1: int, n2: int,
                  alpha: float, sides: int, tail: str
                  ) -> tuple[float, float]:
    z = _z_for(alpha, sides)
    if method == "katz":
        if p1 <= 0 or p2 <= 0:
            raise ValueError("p1 and p2 must be > 0 for Katz log CI")
        phi = p1 / p2
        var_log = (1.0 - p1) / (n1 * p1) + (1.0 - p2) / (n2 * p2)
        se_log = math.sqrt(var_log)
        L = phi * math.exp(-z * se_log)
        U = phi * math.exp(z * se_log)
    elif method == "walters":
        # Walters add-1/2 (a = n1*p1, etc.)
        a = n1 * p1
        c = n1 * (1.0 - p1)
        b = n2 * p2
        d = n2 * (1.0 - p2)
        log_phi = (math.log((a + 0.5) / (n1 + 0.5))
                   - math.log((b + 0.5) / (n2 + 0.5)))
        u_hat = (1.0 / (a + 0.5) - 1.0 / (n1 + 0.5)
                 + 1.0 / (b + 0.5) - 1.0 / (n2 + 0.5))
        se = math.sqrt(max(u_hat, 0.0))
        L = math.exp(log_phi - z * se)
        U = math.exp(log_phi + z * se)
    else:
        raise ValueError(
            f"method must be one of {sorted(_RATIO_METHODS)}; got {method!r}"
        )
    if sides == 1:
        if tail == "upper":
            L = 0.0
        elif tail == "lower":
            U = math.inf
        else:
            raise ValueError("tail must be 'upper' or 'lower' for sides=1")
    return L, U


def _ratio_metric(method: str, p1: float, p2: float, n1: int, n2: int,
                  alpha: float, sides: int, tail: str
                  ) -> tuple[float, float, float]:
    L, U = _ratio_limits(method, p1, p2, n1, n2, alpha, sides, tail)
    phi = p1 / p2
    if sides == 2:
        return U - L, L, U
    if tail == "upper":
        return U - phi, L, U
    if tail == "lower":
        return phi - L, L, U
    raise ValueError("unreachable")


def ci_ratio_two_proportions(
    *,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation_ratio: float = 1.0,
    sides: int = 2,
    tail: str = "two",
    method: str = "katz",
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 5_000_000,
) -> dict[str, Any]:
    """Sample size / achieved width for CI on p1 / p2 (relative risk)."""
    _check_alpha(alpha)
    _check_sides(sides)
    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("p1 and p2 must be in (0, 1)")
    if method not in _RATIO_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_RATIO_METHODS)}; got {method!r}"
        )
    if allocation_ratio <= 0:
        raise ValueError("allocation_ratio must be positive")

    if sides == 2:
        target = width
        target_key = "width"
    else:
        target = distance
        target_key = "distance"
        if tail not in ("upper", "lower"):
            raise ValueError("for sides=1 set tail='upper' or 'lower'")

    if solve_for is None:
        solve_for = "power" if (n1 is not None and n2 is not None) else "n"

    inputs_echo = {
        "p1": p1, "p2": p2, "alpha": alpha, "width": width,
        "distance": distance, "n1": n1, "n2": n2,
        "allocation_ratio": allocation_ratio, "sides": sides,
        "tail": tail, "method": method,
    }

    if solve_for == "power":
        if n1 is None or n2 is None:
            raise ValueError("solve_for='power' requires both n1 and n2")
        actual, L, U = _ratio_metric(method, p1, p2, int(n1), int(n2),
                                     alpha, sides, tail)
        out = {
            "n1": int(n1), "n2": int(n2), "n": int(n1) + int(n2),
            f"achieved_{target_key}": actual,
            "lower_limit": L, "upper_limit": U,
        }
    elif solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                f"solve_for='n' requires positive {target_key}"
            )

        if n2 is not None and n1 is None:
            fixed_n2 = int(n2)

            def width_at(k: int) -> float:
                w, _, _ = _ratio_metric(method, p1, p2, k, fixed_n2,
                                        alpha, sides, tail)
                return w

            n1_req = _bisect_n(width_at, float(target),
                               n_min=n_min, n_max=n_max)
            n2_req = fixed_n2
        elif n1 is not None and n2 is None:
            fixed_n1 = int(n1)

            def width_at(k: int) -> float:
                w, _, _ = _ratio_metric(method, p1, p2, fixed_n1, k,
                                        alpha, sides, tail)
                return w

            n2_req = _bisect_n(width_at, float(target),
                               n_min=n_min, n_max=n_max)
            n1_req = fixed_n1
        else:
            def width_at(k: int) -> float:
                n1k = k
                n2k = max(2, math.ceil(allocation_ratio * k))
                w, _, _ = _ratio_metric(method, p1, p2, n1k, n2k,
                                        alpha, sides, tail)
                return w

            n1_req = _bisect_n(width_at, float(target),
                               n_min=n_min, n_max=n_max)
            n2_req = max(2, math.ceil(allocation_ratio * n1_req))

        actual, L, U = _ratio_metric(method, p1, p2, n1_req, n2_req,
                                     alpha, sides, tail)
        out = {
            "n1": int(n1_req), "n2": int(n2_req),
            "n": int(n1_req) + int(n2_req),
            f"target_{target_key}": float(target),
            f"achieved_{target_key}": actual,
            "lower_limit": L, "upper_limit": U,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_ratio_two_proportions",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CIT_RATIO,
    }


# ===========================================================================
# Ch 218 - Confidence Intervals for the Odds Ratio of Two Proportions
# ===========================================================================

_OR_METHODS = {
    "logarithm",         # +0.5 cell correction, log-Wald (Fleiss 2003 default).
    "simple",            # No correction, log-Wald (n*p, etc.).
}


def _or_limits(method: str, p1: float, p2: float, n1: int, n2: int,
               alpha: float, sides: int, tail: str
               ) -> tuple[float, float]:
    z = _z_for(alpha, sides)
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("p1 and p2 must be in (0, 1) for OR CI")
    psi = (p1 / (1 - p1)) / (p2 / (1 - p2))

    if method == "simple":
        var_log = (1.0 / (n1 * p1) + 1.0 / (n1 * (1 - p1))
                   + 1.0 / (n2 * p2) + 1.0 / (n2 * (1 - p2)))
        se_log = math.sqrt(var_log)
        L = psi * math.exp(-z * se_log)
        U = psi * math.exp(z * se_log)
    elif method == "logarithm":
        a = n1 * p1 + 0.5
        c = n1 * (1 - p1) + 0.5
        b = n2 * p2 + 0.5
        d = n2 * (1 - p2) + 0.5
        psi_prime = (a * d) / (b * c)
        var_log = 1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d
        se_log = math.sqrt(var_log)
        L = psi_prime * math.exp(-z * se_log)
        U = psi_prime * math.exp(z * se_log)
        psi = psi_prime  # report point estimate at +0.5
    else:
        raise ValueError(
            f"method must be one of {sorted(_OR_METHODS)}; got {method!r}"
        )
    if sides == 1:
        if tail == "upper":
            L = 0.0
        elif tail == "lower":
            U = math.inf
        else:
            raise ValueError("tail must be 'upper' or 'lower' for sides=1")
    return L, U


def _or_metric(method: str, p1: float, p2: float, n1: int, n2: int,
               alpha: float, sides: int, tail: str
               ) -> tuple[float, float, float]:
    L, U = _or_limits(method, p1, p2, n1, n2, alpha, sides, tail)
    psi = (p1 / (1 - p1)) / (p2 / (1 - p2))
    if sides == 2:
        return U - L, L, U
    if tail == "upper":
        return U - psi, L, U
    if tail == "lower":
        return psi - L, L, U
    raise ValueError("unreachable")


def ci_odds_ratio_two_proportions(
    *,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    width: float | None = None,
    distance: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation_ratio: float = 1.0,
    sides: int = 2,
    tail: str = "two",
    method: str = "logarithm",
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 5_000_000,
) -> dict[str, Any]:
    """Sample size / achieved width for CI on the odds ratio O1/O2."""
    _check_alpha(alpha)
    _check_sides(sides)
    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("p1 and p2 must be in (0, 1)")
    if method not in _OR_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_OR_METHODS)}; got {method!r}"
        )
    if allocation_ratio <= 0:
        raise ValueError("allocation_ratio must be positive")

    if sides == 2:
        target = width
        target_key = "width"
    else:
        target = distance
        target_key = "distance"
        if tail not in ("upper", "lower"):
            raise ValueError("for sides=1 set tail='upper' or 'lower'")

    if solve_for is None:
        solve_for = "power" if (n1 is not None and n2 is not None) else "n"

    inputs_echo = {
        "p1": p1, "p2": p2, "alpha": alpha, "width": width,
        "distance": distance, "n1": n1, "n2": n2,
        "allocation_ratio": allocation_ratio, "sides": sides,
        "tail": tail, "method": method,
    }

    if solve_for == "power":
        if n1 is None or n2 is None:
            raise ValueError("solve_for='power' requires both n1 and n2")
        actual, L, U = _or_metric(method, p1, p2, int(n1), int(n2),
                                  alpha, sides, tail)
        out = {
            "n1": int(n1), "n2": int(n2), "n": int(n1) + int(n2),
            f"achieved_{target_key}": actual,
            "lower_limit": L, "upper_limit": U,
        }
    elif solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                f"solve_for='n' requires positive {target_key}"
            )

        if n2 is not None and n1 is None:
            fixed_n2 = int(n2)

            def width_at(k: int) -> float:
                w, _, _ = _or_metric(method, p1, p2, k, fixed_n2,
                                     alpha, sides, tail)
                return w

            n1_req = _bisect_n(width_at, float(target),
                               n_min=n_min, n_max=n_max)
            n2_req = fixed_n2
        elif n1 is not None and n2 is None:
            fixed_n1 = int(n1)

            def width_at(k: int) -> float:
                w, _, _ = _or_metric(method, p1, p2, fixed_n1, k,
                                     alpha, sides, tail)
                return w

            n2_req = _bisect_n(width_at, float(target),
                               n_min=n_min, n_max=n_max)
            n1_req = fixed_n1
        else:
            def width_at(k: int) -> float:
                n1k = k
                n2k = max(2, math.ceil(allocation_ratio * k))
                w, _, _ = _or_metric(method, p1, p2, n1k, n2k,
                                     alpha, sides, tail)
                return w

            n1_req = _bisect_n(width_at, float(target),
                               n_min=n_min, n_max=n_max)
            n2_req = max(2, math.ceil(allocation_ratio * n1_req))

        actual, L, U = _or_metric(method, p1, p2, n1_req, n2_req,
                                  alpha, sides, tail)
        out = {
            "n1": int(n1_req), "n2": int(n2_req),
            "n": int(n1_req) + int(n2_req),
            f"target_{target_key}": float(target),
            f"achieved_{target_key}": actual,
            "lower_limit": L, "upper_limit": U,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "ci_odds_ratio_two_proportions",
        "solve_for": solve_for,
        **out,
        "inputs_echo": inputs_echo,
        "citations": _CIT_OR,
    }


__all__ = [
    "ci_one_proportion",
    "ci_one_proportion_finite",
    "ci_difference_two_proportions",
    "ci_ratio_two_proportions",
    "ci_odds_ratio_two_proportions",
]
