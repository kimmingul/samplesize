"""Hotelling's One- and Two-Sample T-squared (T²) sample-size / power.

- "Hotelling's One-Sample T2" (Chapter 601)
- "Hotelling's Two-Sample T2" (Chapter 600)

Both are multivariate extensions of the univariate one- and two-sample
t-tests.  Power calculations rely on the relationship between the
non-central T² and the non-central F distribution.

One-sample (N observations, p response variables)::

    T² ~ (p (N-1) / (N-p)) F'(p, N-p, λ)
    λ  = N · Δ²,  Δ² = (μ_A - μ_0)' Σ⁻¹ (μ_A - μ_0)
    df1 = p,  df2 = N - p

Two-sample (N1, N2 observations)::

    T² ~ (p (N1+N2-2) / (N1+N2-p-1)) F'(p, N1+N2-p-1, λ)
    λ  = (N1 N2 / (N1+N2)) · Δ²,  Δ² = (μ_1 - μ_2)' Σ⁻¹ (μ_1 - μ_2)
    df1 = p,  df2 = N1 + N2 - p - 1

The non-centrality parameter λ feeds directly into a non-central F test
at the F-critical value (1 - α quantile of the *central* F with the
same dfs).  Rencher (1998) tabulates worked examples;


Inputs use the standardized effect Δ via either a covariance matrix or
common SD/correlation pattern; arbitrary positive-definite Σ matrices
are accepted (lists of lists).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from samplesize.core import distributions as D

# ---------------------------------------------------------------------------
# Linear-algebra helpers (no numpy dependency at import time)
# ---------------------------------------------------------------------------


def _to_matrix(M: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = [list(map(float, row)) for row in M]
    p = len(rows)
    for r in rows:
        if len(r) != p:
            raise ValueError("covariance matrix must be square")
    return rows


def _solve_spd(M: list[list[float]], b: list[float]) -> list[float]:
    """Solve M x = b for a symmetric positive-definite M using Cholesky."""
    p = len(M)
    # Cholesky decomposition: L L' = M
    L = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1):
            s = M[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError(
                        "covariance matrix is not positive-definite"
                    )
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    # Solve L y = b, then L' x = y
    y = [0.0] * p
    for i in range(p):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * p
    for i in range(p - 1, -1, -1):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, p))) / L[i][i]
    return x


def _quad_form_invSigma(mu: list[float], Sigma: list[list[float]]) -> float:
    """Return mu' Σ⁻¹ mu using Cholesky-based solve."""
    if any(len(row) != len(mu) for row in Sigma) or len(Sigma) != len(mu):
        raise ValueError("dimension mismatch between mean vector and Σ")
    x = _solve_spd(Sigma, mu)
    return float(sum(mu[i] * x[i] for i in range(len(mu))))


def _build_sigma(
    sigma: float | Sequence[float] | None,
    rho: float,
    p: int,
    pattern: str,
) -> list[list[float]]:
    """Build Σ from common-SD scalar or per-variable SD list, plus a
    correlation pattern ('constant' or 'ar1')."""
    if sigma is None:
        raise ValueError("must supply sigma or a covariance matrix")
    if isinstance(sigma, (int, float)):
        sds = [float(sigma)] * p
    else:
        sds = [float(s) for s in sigma]
        if len(sds) != p:
            raise ValueError("len(sigma) must equal p")
    if any(s <= 0 for s in sds):
        raise ValueError("all standard deviations must be positive")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must be strictly between -1 and 1")
    pat = pattern.lower()
    if pat not in ("constant", "ar1", "ar(1)", "1st-order", "autocorrelation"):
        raise ValueError(
            "correlation pattern must be 'constant' or 'ar1'"
        )
    R = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            if i == j:
                R[i][j] = 1.0
            elif pat == "constant":
                R[i][j] = rho
            else:
                R[i][j] = rho ** abs(i - j)
    Sigma = [
        [sds[i] * sds[j] * R[i][j] for j in range(p)] for i in range(p)
    ]
    return Sigma


def _resolve_sigma(
    *,
    sigma_matrix: Sequence[Sequence[float]] | None,
    sd: float | Sequence[float] | None,
    rho: float | None,
    pattern: str,
    p: int,
) -> list[list[float]]:
    if sigma_matrix is not None:
        M = _to_matrix(sigma_matrix)
        if len(M) != p:
            raise ValueError(
                f"sigma_matrix has size {len(M)} but p={p}"
            )
        return M
    if sd is None:
        raise ValueError("must supply either sigma_matrix or sd")
    return _build_sigma(sd, rho if rho is not None else 0.0, p, pattern)


# ---------------------------------------------------------------------------
# One-sample T²
# ---------------------------------------------------------------------------


def _power_one_sample(*, N: int, p: int, delta2: float,
                      alpha: float) -> float:
    if N <= p:
        return 0.0
    df1 = p
    df2 = N - p
    ncp = N * delta2
    from scipy.stats import f as fdist
    f_crit = fdist.ppf(1.0 - alpha, df1, df2)
    return float(1.0 - D.ncf_cdf(f_crit, df1, df2, ncp))


def hotelling_one_sample_t2(
    *,
    mean_diff: Sequence[float],
    sigma_matrix: Sequence[Sequence[float]] | None = None,
    sd: float | Sequence[float] | None = None,
    rho: float | None = None,
    correlation_pattern: str = "constant",
    K: float = 1.0,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
    n_min: int = 3,
    n_max: int = 1_000_000,
) -> dict[str, Any]:
    """Power / sample-size for Hotelling's One-Sample T².

    Parameters
    ----------
    mean_diff : sequence of float
        Hypothesised mean differences μ_A - μ_0, one per response variable.
    sigma_matrix : sequence of sequences of float, optional
        Variance-covariance matrix Σ (p × p, positive-definite).
    sd, rho, correlation_pattern : optional
        Alternative to sigma_matrix: build Σ from a common SD (scalar or
        per-variable vector), a correlation ``rho`` (default 0), and a
        ``"constant"`` (default) or ``"ar1"`` pattern.
    K : float
        Means multiplier ``K``.  Mean differences are multiplied
        by K before the power computation.
    alpha : float
        Significance level.
    n : int, optional
        Sample size (number of independent vectors).
    power : float, optional
        Target power.
    solve_for : str or None
        ``"n"`` or ``"power"``.  Inferred from which of ``n``/``power``
        is None.
    """
    md = [K * float(x) for x in mean_diff]
    p = len(md)
    if p < 1:
        raise ValueError("mean_diff must have at least one element")
    Sigma = _resolve_sigma(
        sigma_matrix=sigma_matrix, sd=sd, rho=rho,
        pattern=correlation_pattern, p=p,
    )
    delta2 = _quad_form_invSigma(md, Sigma)
    if delta2 < 0:
        raise ValueError(
            "implied non-centrality is negative; Σ is not PSD"
        )

    inputs_echo = {
        "mean_diff": list(mean_diff), "K": K, "p": p,
        "alpha": alpha, "n": n, "power": power,
    }

    if n is None and power is None:
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        if n <= p:
            raise ValueError(
                f"n must be > p (p={p}); got n={n}"
            )
        achieved = _power_one_sample(N=int(n), p=p,
                                     delta2=delta2, alpha=alpha)
        result = {
            "n": int(n),
            "p": p,
            "df1": p,
            "df2": int(n) - p,
            "ncp": int(n) * delta2,
            "effect_size_delta": math.sqrt(delta2),
            "achieved_power": achieved,
        }
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        # bracket
        lo = max(p + 1, n_min)
        hi = lo
        while hi <= n_max:
            if _power_one_sample(N=hi, p=p, delta2=delta2,
                                 alpha=alpha) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError(f"failed to bracket N within {n_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _power_one_sample(N=mid, p=p, delta2=delta2,
                                 alpha=alpha) >= power:
                hi = mid
            else:
                lo = mid
        n_solved = hi
        achieved = _power_one_sample(N=n_solved, p=p,
                                     delta2=delta2, alpha=alpha)
        result = {
            "n": n_solved,
            "p": p,
            "df1": p,
            "df2": n_solved - p,
            "ncp": n_solved * delta2,
            "effect_size_delta": math.sqrt(delta2),
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "hotelling_one_sample_t2",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Rencher, A.C. (1998). Multivariate Statistical Inference and "
            "Applications, page 106.",
        ],
    }


# ---------------------------------------------------------------------------
# Two-sample T²
# ---------------------------------------------------------------------------


def _power_two_sample(*, N1: int, N2: int, p: int, delta2: float,
                      alpha: float) -> float:
    df2 = N1 + N2 - p - 1
    if df2 < 1 or N1 < 2 or N2 < 2:
        return 0.0
    df1 = p
    ncp = (N1 * N2 / (N1 + N2)) * delta2
    from scipy.stats import f as fdist
    f_crit = fdist.ppf(1.0 - alpha, df1, df2)
    return float(1.0 - D.ncf_cdf(f_crit, df1, df2, ncp))


def hotelling_two_sample_t2(
    *,
    mean_diff: Sequence[float],
    sigma_matrix: Sequence[Sequence[float]] | None = None,
    sd: float | Sequence[float] | None = None,
    rho: float | None = None,
    correlation_pattern: str = "constant",
    K: float = 1.0,
    alpha: float = 0.05,
    n1: int | None = None,
    n2: int | None = None,
    n_per_group: int | None = None,
    allocation: float = 1.0,
    power: float | None = None,
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 1_000_000,
) -> dict[str, Any]:
    """Power / sample-size for Hotelling's Two-Sample T².

    Parameters
    ----------
    mean_diff : sequence of float
        Hypothesised between-group mean differences μ_1 - μ_2.
    sigma_matrix / sd / rho / correlation_pattern : optional
        Same as the one-sample case; Σ is assumed common to both groups.
    K : float
        ``K`` multiplier on the mean differences.
    n1, n2 : int, optional
        Per-group sample sizes.  When solving for power both must be
        supplied (``n_per_group`` is a shortcut for n1 == n2).
    n_per_group : int, optional
        Convenience: set n1 = n2 = n_per_group.
    allocation : float
        ``n2 / n1`` ratio used when solving for sample size.  Default 1.
    """
    md = [K * float(x) for x in mean_diff]
    p = len(md)
    if p < 1:
        raise ValueError("mean_diff must have at least one element")
    Sigma = _resolve_sigma(
        sigma_matrix=sigma_matrix, sd=sd, rho=rho,
        pattern=correlation_pattern, p=p,
    )
    delta2 = _quad_form_invSigma(md, Sigma)

    if n_per_group is not None:
        if n1 is None:
            n1 = n_per_group
        if n2 is None:
            n2 = n_per_group

    if allocation <= 0:
        raise ValueError("allocation must be > 0")

    inputs_echo = {
        "mean_diff": list(mean_diff), "K": K, "p": p,
        "alpha": alpha, "n1": n1, "n2": n2,
        "allocation": allocation, "power": power,
    }

    have_n = n1 is not None and n2 is not None
    if not have_n and power is None:
        raise ValueError("supply at least one of (power, n1+n2)")

    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _power_two_sample(N1=int(n1), N2=int(n2), p=p,
                                     delta2=delta2, alpha=alpha)
        result = {
            "n1": int(n1), "n2": int(n2),
            "n": int(n1) + int(n2),
            "p": p,
            "df1": p,
            "df2": int(n1) + int(n2) - p - 1,
            "ncp": (int(n1) * int(n2) / (int(n1) + int(n2))) * delta2,
            "effect_size_delta": math.sqrt(delta2),
            "achieved_power": achieved,
        }
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")

        def n2_from(n1v: int) -> int:
            return max(2, math.ceil(allocation * n1v))

        def p_at(n1v: int) -> float:
            return _power_two_sample(
                N1=n1v, N2=n2_from(n1v), p=p,
                delta2=delta2, alpha=alpha,
            )

        lo, hi = max(n_min, 2), max(n_min, 2)
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
        n1_solved = hi
        n2_solved = n2_from(n1_solved)
        achieved = _power_two_sample(N1=n1_solved, N2=n2_solved, p=p,
                                     delta2=delta2, alpha=alpha)
        result = {
            "n1": n1_solved, "n2": n2_solved,
            "n": n1_solved + n2_solved,
            "p": p,
            "df1": p,
            "df2": n1_solved + n2_solved - p - 1,
            "ncp": (n1_solved * n2_solved
                    / (n1_solved + n2_solved)) * delta2,
            "effect_size_delta": math.sqrt(delta2),
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "hotelling_two_sample_t2",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Rencher, A.C. (1998). Multivariate Statistical Inference and "
            "Applications, pages 107-108.",
        ],
    }
