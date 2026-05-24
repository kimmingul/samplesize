"""MANOVA power & sample-size: Pillai's trace approximation.


Pillai-Bartlett trace F approximation
--------------------------------------
Given k groups, p response variables, and n subjects per group (N = k·n):

  H matrix  = (Θ̂ − Θ₀)' [C (X'X)⁻ C']⁻¹ (Θ̂ − Θ₀)   (hypothesis SSCP)
  E matrix  = Σ̂ · (N − r)                              (error SSCP, r = k)
  T         = H + E

  Pillai trace  T_PB = tr(H · T⁻¹)
  s            = min(a, p)   where a = k − 1 (contrast df)
  η            = T_PB / s
  df1          = a · p
  df2          = s · [(N − r) − p + s]
  F            = (η/df1) / ((1−η)/df2)
  λ            = df1 · F

Specifying the non-centrality
------------------------------
For the case of a one-way MANOVA with equal group sizes (n per group,
N = k·n, r = k) using the means matrix M (p × k) and the residual
covariance Σ (p × p) to compute H and E internally.  We expose a
simpler *Cohen-f²-equivalent* interface that lets users specify the
non-centrality via sigma_m (SD of group means for each response) and
sigma (within-group SD for each response, assumed common).

Under equal group sizes and common diagonal Σ:
  Each response dimension contributes independently via f² = σ_m² / σ²,
  so the joint non-centrality for a one-way design is approximated
  through the trace relationship:

    T_PB ≈ trace(H · T⁻¹) evaluated at the hypothesized means/Sigma.

For full-matrix input, users supply means_matrix (shape k × p) and
sigma (scalar or p-vector).  The module computes H and E exactly using
the formula with N subjects (n per group, N = k·n).

References
----------
* Muller, K.E. & Barton, C.N. (1989). JASA 84, 549-555.
* Muller, K.E., LaVange, L.E., Ramey, S.L., Ramey, C.T. (1992). JASA 87, 1209-1226.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Linear-algebra helpers (pure Python, no external dep beyond scipy)
# ---------------------------------------------------------------------------


def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Dense matrix multiply A @ B."""
    r, inner, c = len(A), len(B), len(B[0])
    return [
        [sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(c)]
        for i in range(r)
    ]


def _mat_inv(A: list[list[float]]) -> list[list[float]]:
    """Invert square matrix using numpy (or scipy linalg)."""
    import numpy as np
    return np.linalg.inv(np.array(A, dtype=float)).tolist()


def _mat_trace(A: list[list[float]]) -> float:
    return sum(A[i][i] for i in range(len(A)))


def _mat_add(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(n)]


def _scalar_mul(s: float, A: list[list[float]]) -> list[list[float]]:
    return [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


# ---------------------------------------------------------------------------
# Core power calculation
# ---------------------------------------------------------------------------


def _pillai_power(
    means_matrix: list[list[float]],
    sigma: float | list[float],
    n_per_group: int,
    alpha: float,
) -> tuple[float, float, float, float, float]:
    """Compute Pillai trace power for a one-way MANOVA.

    Parameters
    ----------
    means_matrix : list[list[float]]
        Shape [k][p]: group means (k groups, p responses).
    sigma : float or list[float]
        Within-group SD (scalar → same for all p; list → per response).
        Off-diagonal covariance is assumed zero (diagonal Sigma).
    n_per_group : int
        Subjects per group.

    Returns
    -------
    power, T_PB, F_stat, df1, df2
    """
    k = len(means_matrix)
    p = len(means_matrix[0])
    n_total = k * n_per_group
    r = k  # rank of X (one-way, k groups)

    # Build diagonal Sigma
    if isinstance(sigma, (int, float)):
        sig_vec = [float(sigma)] * p
    else:
        sig_vec = [float(s) for s in sigma]
    # E = Sigma * (N - r)
    E = [[0.0] * p for _ in range(p)]
    for j in range(p):
        E[j][j] = sig_vec[j] ** 2 * (n_total - r)

    # Compute group marginal means and grand mean (per response)
    grand_means = [sum(means_matrix[i][j] for i in range(k)) / k for j in range(p)]

    # H = n_per_group * sum_i (mu_i - mu_grand)(mu_i - mu_grand)'
    H = [[0.0] * p for _ in range(p)]
    for i in range(k):
        d = [means_matrix[i][j] - grand_means[j] for j in range(p)]
        for row in range(p):
            for col in range(p):
                H[row][col] += n_per_group * d[row] * d[col]

    T = _mat_add(H, E)
    try:
        T_inv = _mat_inv(T)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    HT_inv = _mat_mul(H, T_inv)
    T_pb = _mat_trace(HT_inv)

    a = k - 1  # contrast df
    s = min(a, p)
    if s == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    eta = T_pb / s
    eta = max(0.0, min(eta, 1.0 - 1e-12))  # clamp to [0, 1)
    df1 = a * p
    df2 = s * ((n_total - r) - p + s)
    if df1 <= 0 or df2 <= 0:
        return 0.0, T_pb, 0.0, float(df1), float(df2)

    F_stat = (eta / df1) / ((1.0 - eta) / df2)
    ncp = df1 * F_stat

    from scipy.stats import f as fdist
    f_crit = fdist.ppf(1.0 - alpha, df1, df2)
    power = float(1.0 - D.ncf_cdf(f_crit, df1, df2, ncp))
    return power, T_pb, F_stat, float(df1), float(df2)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def manova_pillai(
    *,
    means_matrix: list[list[float]],
    sigma: float | list[float],
    n_per_group: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """One-way MANOVA Pillai's trace power calculator.

    Parameters
    ----------
    means_matrix
        Shape [k][p]: cell means. k = number of groups (≥ 2),
        p = number of response variables (≥ 2).
    sigma
        Within-group standard deviation.  A scalar applies to all p
        responses; a list of length p gives per-response SDs.
        Off-diagonal covariances are assumed zero.
    n_per_group
        Subjects per group (for solve_for='power').
    alpha
        Significance level (default 0.05).
    power
        Target power (for solve_for='n').
    solve_for
        ``'power'`` or ``'n'``.
    """
    if not means_matrix or not means_matrix[0]:
        raise ValueError("means_matrix must be a non-empty 2-D list")
    k = len(means_matrix)
    p = len(means_matrix[0])
    if k < 2:
        raise ValueError("need at least 2 groups (k >= 2)")
    if p < 2:
        raise ValueError("need at least 2 response variables (p >= 2)")
    if isinstance(sigma, (int, float)):
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
    else:
        if any(s <= 0 for s in sigma):
            raise ValueError("all sigma values must be > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    inputs_echo = {
        "means_matrix": means_matrix, "sigma": sigma,
        "n_per_group": n_per_group, "alpha": alpha, "power": power,
    }
    have_n = n_per_group is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n_per_group, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    def pwr(n_g: int) -> float:
        if n_g < 2:
            return 0.0
        pw, *_ = _pillai_power(means_matrix, sigma, n_g, alpha)
        return pw

    if solve_for == "power":
        assert n_per_group is not None
        if n_per_group < 2:
            raise ValueError("n_per_group must be >= 2")
        achieved, T_pb, F_stat, df1, df2 = _pillai_power(
            means_matrix, sigma, n_per_group, alpha
        )
        n_out = n_per_group
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 2, 2
        while hi <= 1_000_000:
            if pwr(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket n_per_group within 1,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pwr(mid) >= power:
                hi = mid
            else:
                lo = mid
        n_out = hi
        achieved, T_pb, F_stat, df1, df2 = _pillai_power(
            means_matrix, sigma, n_out, alpha
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "manova_pillai",
        "solve_for": solve_for,
        "n_per_group": n_out,
        "n_total": k * n_out,
        "achieved_power": achieved,
        "pillai_trace": T_pb,
        "approx_f": F_stat,
        "df1": df1,
        "df2": df2,
        "inputs_echo": inputs_echo,
        "citations": [
            "Muller, K.E. & Barton, C.N. (1989). 'Approximate Power for "
            "Repeated-Measures ANOVA Lacking Sphericity.' JASA 84, 549-555.",
            "Muller, K.E., LaVange, L.E., Ramey, S.L., Ramey, C.T. (1992). "
            "'Power Calculations for General Linear Multivariate Models.' "
            "JASA 87, 1209-1226.",
        ],
    }
