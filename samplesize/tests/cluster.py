"""Cluster-randomized two-group designs.

Implements:
  - ``cluster_randomized_two_means`` — see
    "Tests for Two Means in a Cluster-Randomized Design".
    Two test-statistic variants are supported:
      * ``"t_subjects"`` (default) — t-test with DF based on number of
        subjects (Campbell & Walters 2014; Ahn, Heo & Zhang 2015).
      * ``"t_clusters"`` — t-test with DF based on number of clusters
        (Donner & Klar 1996, 2000).  Donner & Klar ignored cluster-size
        variability, so to reproduce their tables set ``cov = 0``.
  - ``cluster_randomized_two_proportions`` — see
    "Tests for Two Proportions in a Cluster-Randomized Design".
    Donner & Klar (2000) large-sample z-test with variance inflated by
    the cluster design effect :math:`F_g = 1 + (m_g - 1)\\rho`.
    Both pooled and unpooled critical SE are supported.

Notation
--------
``K_i`` is the number of clusters in arm ``i``; ``M_i`` is the average
number of subjects per cluster in arm ``i``; ``rho`` is the intracluster
correlation (ICC).  The variance of each group mean is inflated by

    DE_i = 1 + (M_i - 1) * rho                     # design effect
    RE_i = 1 / (1 - COV^2 * lambda_i * (1 - lambda_i))   # rel. efficiency
    lambda_i = M_i * rho / (M_i * rho + 1 - rho)

so that ``V_i = sigma^2 * DE_i * RE_i / (K_i * M_i)``.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D
from samplesize.core import effect_sizes as E


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _design_effect(m: float, rho: float) -> float:
    return 1.0 + (m - 1.0) * rho


def _relative_efficiency(m: float, rho: float, cov: float) -> float:
    if cov == 0.0:
        return 1.0
    denom = (m * rho) + (1.0 - rho)
    if denom <= 0:
        raise ValueError("invalid combination of m and rho (lambda undefined)")
    lam = (m * rho) / denom
    factor = 1.0 - (cov * cov) * lam * (1.0 - lam)
    if factor <= 0:
        raise ValueError("invalid COV; relative-efficiency factor non-positive")
    return 1.0 / factor


# ---------------------------------------------------------------------------
# Two means in a cluster-randomized design
# ---------------------------------------------------------------------------

VALID_MEAN_TEST_TYPES = {"t_subjects", "t_clusters"}


def _mean_power(
    *,
    mean1: float,
    mean2: float,
    sd: float,
    k1: int,
    k2: int,
    m1: float,
    m2: float,
    rho: float,
    cov: float,
    alpha: float,
    sides: int,
    test_type: str,
) -> float:
    if sd <= 0:
        raise ValueError("sd must be positive")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho (ICC) must lie in [0, 1)")
    if cov < 0:
        raise ValueError("cov must be >= 0")
    if k1 < 2 or k2 < 2:
        return 0.0
    if m1 < 1 or m2 < 1:
        raise ValueError("average cluster sizes must be >= 1")
    if test_type not in VALID_MEAN_TEST_TYPES:
        raise ValueError(
            f"test_type must be one of {sorted(VALID_MEAN_TEST_TYPES)}"
        )

    de1 = _design_effect(m1, rho)
    de2 = _design_effect(m2, rho)
    re1 = _relative_efficiency(m1, rho, cov)
    re2 = _relative_efficiency(m2, rho, cov)
    v1 = sd * sd * de1 * re1 / (k1 * m1)
    v2 = sd * sd * de2 * re2 / (k2 * m2)
    sigma_d = math.sqrt(v1 + v2)
    if sigma_d <= 0:
        return 0.0
    ncp = (mean1 - mean2) / sigma_d

    if test_type == "t_subjects":
        df = k1 * m1 + k2 * m2 - 2.0
    else:  # t_clusters
        df = k1 + k2 - 2.0
    if df <= 0:
        return 0.0

    if sides == 2:
        x1 = D.t_ppf(alpha / 2.0, df)
        x2 = D.t_ppf(1.0 - alpha / 2.0, df)
        p1 = D.nct_cdf(x1, df, ncp)
        p2 = D.nct_cdf(x2, df, ncp)
        return 1.0 - (p2 - p1)
    if sides == 1:
        x = D.t_ppf(1.0 - alpha, df)
        if mean1 >= mean2:
            return 1.0 - D.nct_cdf(x, df, ncp)
        return D.nct_cdf(-x, df, ncp)
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def _k1_for_mean_power(
    *,
    mean1: float,
    mean2: float,
    sd: float,
    m: float,
    rho: float,
    cov: float,
    alpha: float,
    power: float,
    sides: int,
    allocation: float,
    test_type: str,
    k_min: int = 2,
    k_max: int = 1_000_000,
) -> tuple[int, int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if mean1 == mean2:
        raise ValueError("mean1 and mean2 must differ to solve for K")
    if allocation <= 0:
        raise ValueError("allocation (k2/k1) must be > 0")

    def k2_for(k1):
        return max(2, math.ceil(allocation * k1))

    def p_at(k1):
        k2 = k2_for(k1)
        return _mean_power(
            mean1=mean1, mean2=mean2, sd=sd,
            k1=k1, k2=k2, m1=m, m2=m,
            rho=rho, cov=cov, alpha=alpha,
            sides=sides, test_type=test_type,
        )

    lo, hi = k_min, k_min
    while hi <= k_max:
        if p_at(hi) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket K1 within {k_max}")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if p_at(mid) >= power:
            hi = mid
        else:
            lo = mid

    k1 = hi
    k2 = k2_for(k1)
    achieved = _mean_power(
        mean1=mean1, mean2=mean2, sd=sd,
        k1=k1, k2=k2, m1=m, m2=m,
        rho=rho, cov=cov, alpha=alpha,
        sides=sides, test_type=test_type,
    )
    return k1, k2, achieved


def cluster_randomized_two_means(
    *,
    mean1: float,
    mean2: float,
    sd: float,
    m: float,
    icc: float,
    alpha: float = 0.05,
    power: float | None = None,
    k_clusters: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    cov: float = 0.0,
    test_type: str = "t_subjects",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-mean cluster-randomized power / sample-size.

    Parameters
    ----------
    mean1, mean2 : float
        Group means under the alternative.
    sd : float
        Subject-level standard deviation (assumed equal in both arms).
    m : float
        Average cluster size (applied to both arms; ``M1 = M2 = m``).
    icc : float
        Intracluster correlation coefficient ``rho`` (0 <= icc < 1).
    alpha : float
        Type-I error.
    power : float, optional
        Target power; supply this **or** ``k_clusters``.
    k_clusters : int, optional
        Number of clusters in arm 1 (arm-2 derived from ``allocation``);
        supply this **or** ``power``.
    sides : int
        2 (default) or 1.
    allocation : float
        Ratio ``k2 / k1`` (default 1.0 for balanced designs).
    cov : float
        Coefficient of variation of cluster sizes.  Set to 0 to
        reproduce Donner & Klar (1996); typical values 0.4 - 0.9
        (Campbell & Walters 2014).
    test_type : str
        ``"t_subjects"`` (default) uses DF = K1*M1 + K2*M2 - 2 (Campbell
        & Walters / Ahn-Heo-Zhang).  ``"t_clusters"`` uses DF = K1+K2-2
        (Donner & Klar).
    solve_for : {"n", "power"}, optional
        Override the auto-detected target.
    """
    inputs_echo = {
        "mean1": mean1, "mean2": mean2, "sd": sd, "m": m, "icc": icc,
        "alpha": alpha, "power": power, "k_clusters": k_clusters,
        "sides": sides, "allocation": allocation, "cov": cov,
        "test_type": test_type,
    }
    have_k = k_clusters is not None
    have_power = power is not None
    if have_k == have_power:
        raise ValueError("supply exactly one of (power, k_clusters)")
    if solve_for is None:
        solve_for = "n" if not have_k else "power"

    if solve_for == "power":
        assert k_clusters is not None
        k1 = int(k_clusters)
        k2 = max(2, math.ceil(allocation * k1))
        achieved = _mean_power(
            mean1=mean1, mean2=mean2, sd=sd,
            k1=k1, k2=k2, m1=m, m2=m,
            rho=icc, cov=cov, alpha=alpha,
            sides=sides, test_type=test_type,
        )
    elif solve_for == "n":
        assert power is not None
        k1, k2, achieved = _k1_for_mean_power(
            mean1=mean1, mean2=mean2, sd=sd, m=m,
            rho=icc, cov=cov, alpha=alpha, power=power,
            sides=sides, allocation=allocation, test_type=test_type,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    de = _design_effect(m, icc)
    n1 = int(round(k1 * m))
    n2 = int(round(k2 * m))
    return {
        "method_id": "cluster_randomized_two_means",
        "solve_for": solve_for,
        "k1": k1,
        "k2": k2,
        "k_total": k1 + k2,
        "m_per_cluster": m,
        "n1": n1,
        "n2": n2,
        "n_total": n1 + n2,
        "achieved_power": achieved,
        "design_effect": de,
        "effect_d": E.cohens_d(mean1, mean2, sd),
        "inputs_echo": inputs_echo,
        "citations": [
            "Donner, A. and Klar, N. (1996). Statistical Considerations in "
            "the Design and Analysis of Community Intervention Trials. "
            "J. Clin. Epidemiol. 49(4):435-439.",
            "Donner, A. and Klar, N. (2000). Design and Analysis of "
            "Cluster Randomization Trials in Health Research. Arnold.",
            "Campbell, M.J. and Walters, S.J. (2014). How to Design, "
            "Analyse and Report Cluster Randomised Trials in Medicine "
            "and Health Related Research. Wiley.",
            "Ahn, C., Heo, M., and Zhang, S. (2015). Sample Size "
            "Calculations for Clustered and Longitudinal Outcomes in "
            "Clinical Research. CRC Press.",
        ],
    }


# ---------------------------------------------------------------------------
# Two proportions in a cluster-randomized design
# ---------------------------------------------------------------------------

VALID_PROP_TEST_TYPES = {"z_pooled", "z_unpooled"}


def _prop_power(
    *,
    p1: float,
    p2: float,
    k1: int,
    k2: int,
    m1: float,
    m2: float,
    rho: float,
    alpha: float,
    sides: int,
    test_type: str,
) -> float:
    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("p1 and p2 must lie in (0, 1)")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho (ICC) must lie in [0, 1)")
    if k1 < 2 or k2 < 2:
        return 0.0
    if m1 < 1 or m2 < 1:
        raise ValueError("cluster sizes must be >= 1")
    if test_type not in VALID_PROP_TEST_TYPES:
        raise ValueError(
            f"test_type must be one of {sorted(VALID_PROP_TEST_TYPES)}"
        )

    n1 = k1 * m1
    n2 = k2 * m2
    f1 = 1.0 + (m1 - 1.0) * rho
    f2 = 1.0 + (m2 - 1.0) * rho
    diff = p1 - p2

    sigma_u = math.sqrt(p1 * (1 - p1) * f1 / n1 + p2 * (1 - p2) * f2 / n2)
    if test_type == "z_pooled":
        pbar = (n1 * p1 + n2 * p2) / (n1 + n2)
        sigma_crit = math.sqrt(pbar * (1 - pbar) * (f1 / n1 + f2 / n2))
    else:
        sigma_crit = sigma_u

    from scipy.stats import norm
    if sides == 2:
        z = D.norm_ppf(1.0 - alpha / 2.0)
        upper = 1.0 - norm.cdf((z * sigma_crit - diff) / sigma_u)
        lower = norm.cdf((-z * sigma_crit - diff) / sigma_u)
        return float(upper + lower)
    if sides == 1:
        z = D.norm_ppf(1.0 - alpha)
        if diff > 0:
            return float(1.0 - norm.cdf((z * sigma_crit - diff) / sigma_u))
        return float(norm.cdf((-z * sigma_crit - diff) / sigma_u))
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def _k1_for_prop_power(
    *,
    p1: float,
    p2: float,
    m: float,
    rho: float,
    alpha: float,
    power: float,
    sides: int,
    allocation: float,
    test_type: str,
    k_min: int = 2,
    k_max: int = 1_000_000,
) -> tuple[int, int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ to solve for K")
    if allocation <= 0:
        raise ValueError("allocation (k2/k1) must be > 0")

    def k2_for(k1):
        return max(2, math.ceil(allocation * k1))

    def p_at(k1):
        k2 = k2_for(k1)
        return _prop_power(
            p1=p1, p2=p2, k1=k1, k2=k2, m1=m, m2=m,
            rho=rho, alpha=alpha, sides=sides, test_type=test_type,
        )

    lo, hi = k_min, k_min
    while hi <= k_max:
        if p_at(hi) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket K1 within {k_max}")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if p_at(mid) >= power:
            hi = mid
        else:
            lo = mid

    k1 = hi
    k2 = k2_for(k1)
    achieved = _prop_power(
        p1=p1, p2=p2, k1=k1, k2=k2, m1=m, m2=m,
        rho=rho, alpha=alpha, sides=sides, test_type=test_type,
    )
    return k1, k2, achieved


def cluster_randomized_two_proportions(
    *,
    p1: float,
    p2: float,
    m: float,
    icc: float,
    alpha: float = 0.05,
    power: float | None = None,
    k_clusters: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    test_type: str = "z_pooled",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-proportion cluster-randomized power / sample-size.

    Parameters
    ----------
    p1, p2 : float
        Group proportions under the alternative.
    m : float
        Average cluster size (applied to both arms; ``M1 = M2 = m``).
    icc : float
        Intracluster correlation coefficient.
    alpha : float
        Type-I error.
    power : float, optional
        Target power; supply this **or** ``k_clusters``.
    k_clusters : int, optional
        Number of clusters in arm 1; supply this **or** ``power``.
    sides : int
        2 (default) or 1.
    allocation : float
        Ratio ``k2 / k1`` (default 1.0).
    test_type : str
        ``"z_pooled"`` (default) or ``"z_unpooled"``.  These are
        Z-Test (Pooled) and Z-Test (Unpooled); the difference is whether
        the rejection critical SE is computed under the null
        (pooled p̄) or under the alternative (unpooled).
    solve_for : {"n", "power"}, optional
        Override the auto-detected target.
    """
    inputs_echo = {
        "p1": p1, "p2": p2, "m": m, "icc": icc, "alpha": alpha,
        "power": power, "k_clusters": k_clusters, "sides": sides,
        "allocation": allocation, "test_type": test_type,
    }
    have_k = k_clusters is not None
    have_power = power is not None
    if have_k == have_power:
        raise ValueError("supply exactly one of (power, k_clusters)")
    if solve_for is None:
        solve_for = "n" if not have_k else "power"

    if solve_for == "power":
        assert k_clusters is not None
        k1 = int(k_clusters)
        k2 = max(2, math.ceil(allocation * k1))
        achieved = _prop_power(
            p1=p1, p2=p2, k1=k1, k2=k2, m1=m, m2=m,
            rho=icc, alpha=alpha, sides=sides, test_type=test_type,
        )
    elif solve_for == "n":
        assert power is not None
        k1, k2, achieved = _k1_for_prop_power(
            p1=p1, p2=p2, m=m, rho=icc, alpha=alpha, power=power,
            sides=sides, allocation=allocation, test_type=test_type,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    de = _design_effect(m, icc)
    n1 = int(round(k1 * m))
    n2 = int(round(k2 * m))
    return {
        "method_id": "cluster_randomized_two_proportions",
        "solve_for": solve_for,
        "k1": k1,
        "k2": k2,
        "k_total": k1 + k2,
        "m_per_cluster": m,
        "n1": n1,
        "n2": n2,
        "n_total": n1 + n2,
        "achieved_power": achieved,
        "design_effect": de,
        "effect_h": E.cohens_h(p1, p2),
        "inputs_echo": inputs_echo,
        "citations": [
            "Cluster-Randomized Design",
            "Donner, A. and Klar, N. (2000). Design and Analysis of "
            "Cluster Randomization Trials in Health Research. Arnold.",
        ],
    }


# ---------------------------------------------------------------------------
# Shared cluster helper: effective n per arm
# ---------------------------------------------------------------------------

def _cluster_n_eff(k: int, m: float, icc: float) -> float:
    """Effective individual-level n for one arm after cluster design-effect.

    n_eff = k * m / DE  where DE = 1 + (m-1)*rho.
    """
    de = _design_effect(m, icc)
    return k * m / de


# ---------------------------------------------------------------------------
# Farrington-Manning score test helpers (difference scale)
# ---------------------------------------------------------------------------

def _fm_diff_constrained(
    p1: float, p2: float, n_eff: float, d0: float,
) -> tuple[float, float]:
    """Pooled null-hypothesis proportions under H0: p1 - p2 = d0."""
    p2t = (n_eff * (p1 - d0) + n_eff * p2) / (2.0 * n_eff)
    p1t = p2t + d0
    eps = 1e-12
    p1t = min(1.0 - eps, max(eps, p1t))
    p2t = min(1.0 - eps, max(eps, p2t))
    return p1t, p2t


def _fm_diff_power_one_sided(
    p1: float, p2: float, n_eff: float, d0: float, alpha: float,
) -> float:
    """One-sided FM power: H0: p1-p2 <= d0  vs  H1: p1-p2 > d0."""
    from scipy.stats import norm as _norm
    if n_eff < 2:
        return 0.0
    p1t, p2t = _fm_diff_constrained(p1, p2, n_eff, d0)
    se0 = math.sqrt(p1t * (1.0 - p1t) / n_eff + p2t * (1.0 - p2t) / n_eff)
    se1 = math.sqrt(p1 * (1.0 - p1) / n_eff + p2 * (1.0 - p2) / n_eff)
    if se1 <= 0:
        return 0.0
    z_a = D.norm_ppf(1.0 - alpha)
    return float(_norm.cdf(((p1 - p2 - d0) - z_a * se0) / se1))


def _fm_diff_tost_power(
    p1: float, p2: float, n_eff: float, d0l: float, d0u: float, alpha: float,
) -> float:
    """TOST FM power for H0: diff<=d0l or diff>=d0u  vs  H1: d0l<diff<d0u."""
    from scipy.stats import norm as _norm
    if n_eff < 2:
        return 0.0
    # Lower one-sided: H0: diff <= D0L
    pL = _fm_diff_power_one_sided(p1, p2, n_eff, d0l, alpha)
    # Upper one-sided: H0: diff >= D0U  <=>  H0: p2-p1 <= -D0U
    p1t_U, p2t_U = _fm_diff_constrained(p1, p2, n_eff, d0u)
    se0_U = math.sqrt(p1t_U * (1.0 - p1t_U) / n_eff + p2t_U * (1.0 - p2t_U) / n_eff)
    se1 = math.sqrt(p1 * (1.0 - p1) / n_eff + p2 * (1.0 - p2) / n_eff)
    z_a = D.norm_ppf(1.0 - alpha)
    if se1 <= 0:
        return 0.0
    pU = float(_norm.cdf((-(p1 - p2 - d0u) - z_a * se0_U) / se1))
    return max(0.0, pL + pU - 1.0)


# ---------------------------------------------------------------------------
# Farrington-Manning score test helpers (ratio scale)
# ---------------------------------------------------------------------------

def _fm_ratio_constrained(
    p1: float, p2: float, n_eff: float, phi0: float,
) -> tuple[float, float]:
    """Constrained MLEs under H0: p1/p2 = phi0 (Farrington-Manning 1990)."""
    N = 2.0 * n_eff
    x11 = n_eff * p1
    x21 = n_eff * p2
    A = N * phi0
    B = -(n_eff * phi0 + x11 + n_eff + x21 * phi0)
    C = x11 + x21
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        disc = 0.0
    eps = 1e-12
    p2t = (-B - math.sqrt(disc)) / (2.0 * A)
    p2t = min(1.0 - eps, max(eps, p2t))
    p1t = min(1.0 - eps, max(eps, phi0 * p2t))
    return p1t, p2t


def _fm_ratio_power_one_sided(
    p1: float, p2: float, n_eff: float, phi0: float, alpha: float,
    direction: str = "upper",
) -> float:
    """One-sided FM power on ratio scale.

    direction='upper': H0: p1/p2 <= phi0, H1: p1/p2 > phi0
    direction='lower': H0: p1/p2 >= phi0, H1: p1/p2 < phi0
    """
    from scipy.stats import norm as _norm
    if n_eff < 2:
        return 0.0
    p1t, p2t = _fm_ratio_constrained(p1, p2, n_eff, phi0)
    se0 = math.sqrt(p1t * (1.0 - p1t) / n_eff + phi0 ** 2 * p2t * (1.0 - p2t) / n_eff)
    se1 = math.sqrt(p1 * (1.0 - p1) / n_eff + phi0 ** 2 * p2 * (1.0 - p2) / n_eff)
    if se1 <= 0:
        return 0.0
    z_a = D.norm_ppf(1.0 - alpha)
    diff = p1 - phi0 * p2
    if direction == "upper":
        return float(_norm.cdf((diff - z_a * se0) / se1))
    return float(_norm.cdf((-diff - z_a * se0) / se1))


# ---------------------------------------------------------------------------
# Bisection helper shared by NI/equivalence/superiority means methods
# ---------------------------------------------------------------------------

def _k1_for_margin_mean_power(
    *,
    delta: float,
    margin: float,
    sd: float,
    m: float,
    rho: float,
    cov: float,
    alpha: float,
    power: float,
    sides: int,
    allocation: float,
    test_type: str,
    k_min: int = 2,
    k_max: int = 1_000_000,
) -> tuple[int, int, float]:
    """Bisect K1 for NI/Superiority/Equivalence cluster means (one-sided NCP)."""
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if allocation <= 0:
        raise ValueError("allocation (k2/k1) must be > 0")

    def k2_for(k1: int) -> int:
        return max(2, math.ceil(allocation * k1))

    def p_at(k1: int) -> float:
        k2 = k2_for(k1)
        return _margin_mean_power(
            delta=delta, margin=margin, sd=sd,
            k1=k1, k2=k2, m1=m, m2=m,
            rho=rho, cov=cov, alpha=alpha,
            sides=sides, test_type=test_type,
        )

    lo, hi = k_min, k_min
    while hi <= k_max:
        if p_at(hi) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket K1 within {k_max}")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if p_at(mid) >= power:
            hi = mid
        else:
            lo = mid

    k1 = hi
    k2 = k2_for(k1)
    achieved = p_at(k1)
    return k1, k2, achieved


def _margin_mean_power(
    *,
    delta: float,
    margin: float,
    sd: float,
    k1: int,
    k2: int,
    m1: float,
    m2: float,
    rho: float,
    cov: float,
    alpha: float,
    sides: int,
    test_type: str,
) -> float:
    """One-sided power for NI / Superiority margin: NCP = (delta - margin)/sigma_d."""
    if sd <= 0:
        raise ValueError("sd must be positive")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho (ICC) must lie in [0, 1)")
    if k1 < 2 or k2 < 2:
        return 0.0

    de1 = _design_effect(m1, rho)
    de2 = _design_effect(m2, rho)
    re1 = _relative_efficiency(m1, rho, cov)
    re2 = _relative_efficiency(m2, rho, cov)
    v1 = sd * sd * de1 * re1 / (k1 * m1)
    v2 = sd * sd * de2 * re2 / (k2 * m2)
    sigma_d = math.sqrt(v1 + v2)
    if sigma_d <= 0:
        return 0.0

    if test_type == "t_subjects":
        df = k1 * m1 + k2 * m2 - 2.0
    elif test_type == "t_clusters":
        df = k1 + k2 - 2.0
    else:
        raise ValueError(f"test_type must be one of {sorted(VALID_MEAN_TEST_TYPES)}")
    if df <= 0:
        return 0.0

    ncp = (delta - margin) / sigma_d
    # One-sided upper test (reject when t > t_alpha)
    x = D.t_ppf(1.0 - alpha, df)
    return float(1.0 - D.nct_cdf(x, df, ncp))


# ---------------------------------------------------------------------------
# Non-inferiority for cluster two means
# ---------------------------------------------------------------------------

def non_inferiority_cluster_two_means(
    *,
    delta: float,
    nim: float,
    sd: float,
    m: float,
    icc: float,
    alpha: float = 0.025,
    power: float | None = None,
    k_clusters: int | None = None,
    sides: int = 1,
    allocation: float = 1.0,
    cov: float = 0.0,
    test_type: str = "t_subjects",
    higher_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for two means in a cluster-randomized design.

    Parameters
    ----------
    delta : float
        True mean difference mu1 - mu2 (often 0).
    nim : float
        Non-inferiority margin (positive magnitude).
    sd : float
        Subject-level standard deviation.
    m : float
        Average cluster size (both arms).
    icc : float
        Intracluster correlation (0 <= icc < 1).
    alpha : float
        One-sided type-I error (default 0.025).
    higher_better : bool
        If True, H0: delta <= -NIM; if False, H0: delta >= NIM.
    """
    inputs_echo = dict(
        delta=delta, nim=nim, sd=sd, m=m, icc=icc,
        alpha=alpha, power=power, k_clusters=k_clusters,
        sides=sides, allocation=allocation, cov=cov,
        test_type=test_type, higher_better=higher_better,
    )
    have_k = k_clusters is not None
    have_power = power is not None
    if have_k == have_power:
        raise ValueError("supply exactly one of (power, k_clusters)")
    if solve_for is None:
        solve_for = "n" if not have_k else "power"

    # margin sign: if higher_better, NI fails when delta < -NIM -> use margin = -NIM
    margin = -abs(nim) if higher_better else abs(nim)

    if solve_for == "power":
        assert k_clusters is not None
        k1 = int(k_clusters)
        k2 = max(2, math.ceil(allocation * k1))
        achieved = _margin_mean_power(
            delta=delta, margin=margin, sd=sd,
            k1=k1, k2=k2, m1=m, m2=m,
            rho=icc, cov=cov, alpha=alpha,
            sides=1, test_type=test_type,
        )
    else:
        assert power is not None
        k1, k2, achieved = _k1_for_margin_mean_power(
            delta=delta, margin=margin, sd=sd, m=m,
            rho=icc, cov=cov, alpha=alpha, power=power,
            sides=1, allocation=allocation, test_type=test_type,
        )

    de = _design_effect(m, icc)
    n1 = int(round(k1 * m))
    n2 = int(round(k2 * m))
    return {
        "method_id": "non_inferiority_cluster_two_means",
        "solve_for": solve_for,
        "n": n1 + n2,
        "n_clusters": k1 + k2,
        "m_per_cluster": int(m),
        "k1": k1,
        "k2": k2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "design_effect": de,
        "inputs_echo": inputs_echo,
        "citations": [
            "in a Cluster-Randomized Design",
            "Campbell, M.J. and Walters, S.J. (2014). How to Design, Analyse "
            "and Report Cluster Randomised Trials. Wiley.",
            "Ahn, C., Heo, M., and Zhang, S. (2015). Sample Size Calculations "
            "for Clustered and Longitudinal Outcomes. CRC Press.",
        ],
    }


# ---------------------------------------------------------------------------
# Equivalence for cluster two means (TOST)
# ---------------------------------------------------------------------------

def equivalence_cluster_two_means(
    *,
    delta: float,
    eu: float,
    el: float | None = None,
    sd: float,
    m: float,
    icc: float,
    alpha: float = 0.05,
    power: float | None = None,
    k_clusters: int | None = None,
    allocation: float = 1.0,
    cov: float = 0.0,
    test_type: str = "t_subjects",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """TOST equivalence for two means in a cluster-randomized design.

    Parameters
    ----------
    delta : float
        True mean difference mu1 - mu2 (often 0).
    eu : float
        Upper equivalence limit (positive).
    el : float, optional
        Lower equivalence limit (negative); defaults to -eu.
    """
    if el is None:
        el = -abs(eu)
    inputs_echo = dict(
        delta=delta, eu=eu, el=el, sd=sd, m=m, icc=icc,
        alpha=alpha, power=power, k_clusters=k_clusters,
        allocation=allocation, cov=cov, test_type=test_type,
    )
    have_k = k_clusters is not None
    have_power = power is not None
    if have_k == have_power:
        raise ValueError("supply exactly one of (power, k_clusters)")
    if solve_for is None:
        solve_for = "n" if not have_k else "power"

    def _tost_power(k1: int, k2: int) -> float:
        # Phillips/Schuirmann TOST: Power = Phi((EU-D)/SE - t_alpha) - Phi((EL-D)/SE + t_alpha)
        from scipy.stats import norm as _norm
        de1 = _design_effect(m, icc)
        re1 = _relative_efficiency(m, icc, cov)
        v = sd * sd * de1 * re1 / (k1 * m)
        se = math.sqrt(2.0 * v)  # balanced arms
        if se <= 0:
            return 0.0
        if test_type == "t_subjects":
            df = k1 * m + k2 * m - 2.0
        else:
            df = k1 + k2 - 2.0
        t_alpha = D.t_ppf(1.0 - alpha, df)
        upper_term = float(_norm.cdf((eu - delta) / se - t_alpha))
        lower_term = float(_norm.cdf((el - delta) / se + t_alpha))
        return max(0.0, upper_term - lower_term)

    if solve_for == "power":
        assert k_clusters is not None
        k1 = int(k_clusters)
        k2 = max(2, math.ceil(allocation * k1))
        achieved = _tost_power(k1, k2)
    else:
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        if allocation <= 0:
            raise ValueError("allocation must be > 0")

        def k2_for(k1: int) -> int:
            return max(2, math.ceil(allocation * k1))

        lo, hi = 2, 2
        k_max = 1_000_000
        while hi <= k_max:
            if _tost_power(hi, k2_for(hi)) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket K1")

        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _tost_power(mid, k2_for(mid)) >= power:
                hi = mid
            else:
                lo = mid

        k1 = hi
        k2 = k2_for(k1)
        achieved = _tost_power(k1, k2)

    de = _design_effect(m, icc)
    n1 = int(round(k1 * m))
    n2 = int(round(k2 * m))
    return {
        "method_id": "equivalence_cluster_two_means",
        "solve_for": solve_for,
        "n": n1 + n2,
        "n_clusters": k1 + k2,
        "m_per_cluster": int(m),
        "k1": k1,
        "k2": k2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "design_effect": de,
        "inputs_echo": inputs_echo,
        "citations": [
            "in a Cluster-Randomized Design",
            "Campbell, M.J. and Walters, S.J. (2014). How to Design, Analyse "
            "and Report Cluster Randomised Trials. Wiley.",
            "Schuirmann, D. (1987). A Comparison of the Two One-Sided Tests "
            "Procedure and the Power Approach. J. Pharmacokin. Biopharm. 15:657-680.",
        ],
    }


# ---------------------------------------------------------------------------
# Superiority by a margin for cluster two means
# ---------------------------------------------------------------------------

def superiority_by_margin_cluster_two_means(
    *,
    delta: float,
    sm: float,
    sd: float,
    m: float,
    icc: float,
    alpha: float = 0.025,
    power: float | None = None,
    k_clusters: int | None = None,
    allocation: float = 1.0,
    cov: float = 0.0,
    test_type: str = "t_subjects",
    higher_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Superiority by a margin for two means in a cluster-randomized design.

    Parameters
    ----------
    delta : float
        True mean difference mu1 - mu2.
    sm : float
        Superiority margin (positive magnitude).
    higher_better : bool
        If True, H0: delta <= SM; if False, H0: delta >= -SM.
    """
    inputs_echo = dict(
        delta=delta, sm=sm, sd=sd, m=m, icc=icc,
        alpha=alpha, power=power, k_clusters=k_clusters,
        allocation=allocation, cov=cov, test_type=test_type,
        higher_better=higher_better,
    )
    have_k = k_clusters is not None
    have_power = power is not None
    if have_k == have_power:
        raise ValueError("supply exactly one of (power, k_clusters)")
    if solve_for is None:
        solve_for = "n" if not have_k else "power"

    margin = abs(sm) if higher_better else -abs(sm)

    if solve_for == "power":
        assert k_clusters is not None
        k1 = int(k_clusters)
        k2 = max(2, math.ceil(allocation * k1))
        achieved = _margin_mean_power(
            delta=delta, margin=margin, sd=sd,
            k1=k1, k2=k2, m1=m, m2=m,
            rho=icc, cov=cov, alpha=alpha,
            sides=1, test_type=test_type,
        )
    else:
        assert power is not None
        k1, k2, achieved = _k1_for_margin_mean_power(
            delta=delta, margin=margin, sd=sd, m=m,
            rho=icc, cov=cov, alpha=alpha, power=power,
            sides=1, allocation=allocation, test_type=test_type,
        )

    de = _design_effect(m, icc)
    n1 = int(round(k1 * m))
    n2 = int(round(k2 * m))
    return {
        "method_id": "superiority_by_margin_cluster_two_means",
        "solve_for": solve_for,
        "n": n1 + n2,
        "n_clusters": k1 + k2,
        "m_per_cluster": int(m),
        "k1": k1,
        "k2": k2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "design_effect": de,
        "inputs_echo": inputs_echo,
        "citations": [
            "in a Cluster-Randomized Design",
            "Campbell, M.J. and Walters, S.J. (2014). How to Design, Analyse "
            "and Report Cluster Randomised Trials. Wiley.",
        ],
    }


# ---------------------------------------------------------------------------
# Bisection helper for cluster proportion NI/equivalence methods
# ---------------------------------------------------------------------------

def _k1_for_prop_ni_power(
    power_fn,
    *,
    power: float,
    allocation: float,
    k_min: int = 2,
    k_max: int = 1_000_000,
) -> tuple[int, int, float]:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if allocation <= 0:
        raise ValueError("allocation must be > 0")

    def k2_for(k1: int) -> int:
        return max(2, math.ceil(allocation * k1))

    lo, hi = k_min, k_min
    while hi <= k_max:
        if power_fn(hi, k2_for(hi)) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket K1 within {k_max}")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if power_fn(mid, k2_for(mid)) >= power:
            hi = mid
        else:
            lo = mid

    k1 = hi
    k2 = k2_for(k1)
    return k1, k2, power_fn(k1, k2)


# ---------------------------------------------------------------------------
# Non-inferiority for cluster two proportions (difference scale)
# ---------------------------------------------------------------------------

def non_inferiority_cluster_two_proportions(
    *,
    p1: float,
    p2: float,
    d0: float,
    m: float,
    icc: float,
    alpha: float = 0.05,
    power: float | None = None,
    k_clusters: int | None = None,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for two proportions (difference scale) in cluster RCT.

    Parameters
    ----------
    p1 : float
        Actual proportion in group 1 (treatment) under H1.
    p2 : float
        Proportion in group 2 (reference/control).
    d0 : float
        Non-inferiority margin (negative when higher proportions are better,
        e.g. -0.05 means P1 may be up to 5% below P2).
    """
    inputs_echo = dict(
        p1=p1, p2=p2, d0=d0, m=m, icc=icc,
        alpha=alpha, power=power, k_clusters=k_clusters, allocation=allocation,
    )
    have_k = k_clusters is not None
    have_power = power is not None
    if have_k == have_power:
        raise ValueError("supply exactly one of (power, k_clusters)")
    if solve_for is None:
        solve_for = "n" if not have_k else "power"

    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("p1 and p2 must lie in (0, 1)")
    if not 0.0 <= icc < 1.0:
        raise ValueError("icc must lie in [0, 1)")

    def _power_at(k1: int, k2: int) -> float:
        n_eff = _cluster_n_eff(k1, m, icc)
        return _fm_diff_power_one_sided(p1, p2, n_eff, d0, alpha)

    if solve_for == "power":
        assert k_clusters is not None
        k1 = int(k_clusters)
        k2 = max(2, math.ceil(allocation * k1))
        achieved = _power_at(k1, k2)
    else:
        assert power is not None
        k1, k2, achieved = _k1_for_prop_ni_power(
            _power_at, power=power, allocation=allocation,
        )

    de = _design_effect(m, icc)
    n1 = int(round(k1 * m))
    n2 = int(round(k2 * m))
    return {
        "method_id": "non_inferiority_cluster_two_proportions",
        "solve_for": solve_for,
        "n": n1 + n2,
        "n_clusters": k1 + k2,
        "m_per_cluster": int(m),
        "k1": k1,
        "k2": k2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "design_effect": de,
        "inputs_echo": inputs_echo,
        "citations": [
            "of Two Proportions in a Cluster-Randomized Design",
            "Donner, A. and Klar, N. (2000). Design and Analysis of "
            "Cluster Randomization Trials in Health Research. Arnold.",
            "Farrington, C.P. and Manning, G. (1990). Test statistics and sample "
            "size formulae for comparative binomial trials. Stat. Med. 9:1447-1454.",
        ],
    }


# ---------------------------------------------------------------------------
# Non-inferiority for cluster two proportions (ratio scale)
# ---------------------------------------------------------------------------

def non_inferiority_cluster_ratio_two_proportions(
    *,
    p1: float,
    p2: float,
    r0: float,
    m: float,
    icc: float,
    alpha: float = 0.05,
    power: float | None = None,
    k_clusters: int | None = None,
    allocation: float = 1.0,
    higher_better: bool = True,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-inferiority test for two proportions (ratio scale) in cluster RCT.

    Parameters
    ----------
    p1 : float
        Actual proportion in group 1 (treatment) under H1.
    p2 : float
        Proportion in group 2 (reference/control).
    r0 : float
        Non-inferiority ratio R0 = P1.0/P2 (< 1 when higher is better).
    higher_better : bool
        If True, H0: p1/p2 <= r0, H1: p1/p2 > r0.
    """
    inputs_echo = dict(
        p1=p1, p2=p2, r0=r0, m=m, icc=icc,
        alpha=alpha, power=power, k_clusters=k_clusters,
        allocation=allocation, higher_better=higher_better,
    )
    have_k = k_clusters is not None
    have_power = power is not None
    if have_k == have_power:
        raise ValueError("supply exactly one of (power, k_clusters)")
    if solve_for is None:
        solve_for = "n" if not have_k else "power"

    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("p1 and p2 must lie in (0, 1)")
    if not 0.0 <= icc < 1.0:
        raise ValueError("icc must lie in [0, 1)")
    if r0 <= 0:
        raise ValueError("r0 must be > 0")

    direction = "upper" if higher_better else "lower"

    def _power_at(k1: int, k2: int) -> float:
        n_eff = _cluster_n_eff(k1, m, icc)
        return _fm_ratio_power_one_sided(p1, p2, n_eff, r0, alpha, direction)

    if solve_for == "power":
        assert k_clusters is not None
        k1 = int(k_clusters)
        k2 = max(2, math.ceil(allocation * k1))
        achieved = _power_at(k1, k2)
    else:
        assert power is not None
        k1, k2, achieved = _k1_for_prop_ni_power(
            _power_at, power=power, allocation=allocation,
        )

    de = _design_effect(m, icc)
    n1 = int(round(k1 * m))
    n2 = int(round(k2 * m))
    return {
        "method_id": "non_inferiority_cluster_ratio_two_proportions",
        "solve_for": solve_for,
        "n": n1 + n2,
        "n_clusters": k1 + k2,
        "m_per_cluster": int(m),
        "k1": k1,
        "k2": k2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "design_effect": de,
        "inputs_echo": inputs_echo,
        "citations": [
            "of Two Proportions in a Cluster-Randomized Design",
            "Donner, A. and Klar, N. (2000). Design and Analysis of "
            "Cluster Randomization Trials in Health Research. Arnold.",
            "Farrington, C.P. and Manning, G. (1990). Test statistics and sample "
            "size formulae for comparative binomial trials. Stat. Med. 9:1447-1454.",
        ],
    }


# ---------------------------------------------------------------------------
# Equivalence for cluster two proportions (difference scale)
# ---------------------------------------------------------------------------

def equivalence_cluster_two_proportions(
    *,
    p1: float,
    p2: float,
    d0u: float,
    d0l: float | None = None,
    m: float,
    icc: float,
    alpha: float = 0.05,
    power: float | None = None,
    k_clusters: int | None = None,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """TOST equivalence for two proportions (difference scale) in cluster RCT.

    Parameters
    ----------
    p1 : float
        Actual proportion in group 1 under H1.
    p2 : float
        Proportion in group 2 (reference/control).
    d0u : float
        Upper equivalence difference (positive).
    d0l : float, optional
        Lower equivalence difference (negative); defaults to -d0u.
    """
    if d0l is None:
        d0l = -abs(d0u)
    inputs_echo = dict(
        p1=p1, p2=p2, d0u=d0u, d0l=d0l, m=m, icc=icc,
        alpha=alpha, power=power, k_clusters=k_clusters, allocation=allocation,
    )
    have_k = k_clusters is not None
    have_power = power is not None
    if have_k == have_power:
        raise ValueError("supply exactly one of (power, k_clusters)")
    if solve_for is None:
        solve_for = "n" if not have_k else "power"

    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("p1 and p2 must lie in (0, 1)")
    if not 0.0 <= icc < 1.0:
        raise ValueError("icc must lie in [0, 1)")

    def _power_at(k1: int, k2: int) -> float:
        n_eff = _cluster_n_eff(k1, m, icc)
        return _fm_diff_tost_power(p1, p2, n_eff, d0l, d0u, alpha)

    if solve_for == "power":
        assert k_clusters is not None
        k1 = int(k_clusters)
        k2 = max(2, math.ceil(allocation * k1))
        achieved = _power_at(k1, k2)
    else:
        assert power is not None
        k1, k2, achieved = _k1_for_prop_ni_power(
            _power_at, power=power, allocation=allocation,
        )

    de = _design_effect(m, icc)
    n1 = int(round(k1 * m))
    n2 = int(round(k2 * m))
    return {
        "method_id": "equivalence_cluster_two_proportions",
        "solve_for": solve_for,
        "n": n1 + n2,
        "n_clusters": k1 + k2,
        "m_per_cluster": int(m),
        "k1": k1,
        "k2": k2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "design_effect": de,
        "inputs_echo": inputs_echo,
        "citations": [
            "of Two Proportions in a Cluster-Randomized Design",
            "Donner, A. and Klar, N. (2000). Design and Analysis of "
            "Cluster Randomization Trials in Health Research. Arnold.",
            "Farrington, C.P. and Manning, G. (1990). Test statistics and sample "
            "size formulae for comparative binomial trials. Stat. Med. 9:1447-1454.",
        ],
    }


# ---------------------------------------------------------------------------
# Two Poisson rates in a cluster-randomized design (Hayes & Bennett 1999)
# ---------------------------------------------------------------------------

def cluster_two_poisson_rates(
    *,
    lambda1: float,
    lambda2: float,
    ki: int | None = None,
    m: float,
    cv1: float,
    cv2: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-arm cluster-randomized Poisson rate comparison (Hayes & Bennett 1999).

    Parameters
    ----------
    lambda1 : float
        Event rate in group 1 (control).
    lambda2 : float
        Event rate in group 2 (treatment).
    ki : int, optional
        Number of clusters per group (balanced: K1 = K2 = ki).
    m : float
        Person-years per cluster (both groups).
    cv1 : float
        Coefficient of variation of cluster rates in group 1.
    cv2 : float, optional
        CV for group 2; defaults to cv1.
    sides : int
        1 or 2 (default 2).
    """
    if cv2 is None:
        cv2 = cv1
    inputs_echo = dict(
        lambda1=lambda1, lambda2=lambda2, ki=ki, m=m,
        cv1=cv1, cv2=cv2, alpha=alpha, power=power, sides=sides,
    )
    have_ki = ki is not None
    have_power = power is not None
    if have_ki == have_power:
        raise ValueError("supply exactly one of (power, ki)")
    if solve_for is None:
        solve_for = "n" if not have_ki else "power"
    if lambda1 <= 0 or lambda2 <= 0:
        raise ValueError("lambda1 and lambda2 must be > 0")
    if lambda1 == lambda2:
        raise ValueError("lambda1 and lambda2 must differ to solve for ki")
    if cv1 < 0 or cv2 < 0:
        raise ValueError("cv1 and cv2 must be >= 0")

    from scipy.stats import norm as _norm

    def _power_at(k: int) -> float:
        if k < 2:
            return 0.0
        numer = (k - 1) * (lambda2 - lambda1) ** 2
        denom = (lambda1 + lambda2) / m + (cv1 * lambda1) ** 2 + (cv2 * lambda2) ** 2
        if denom <= 0:
            return 0.0
        if sides == 2:
            z_crit = D.norm_ppf(1.0 - alpha / 2.0)
        else:
            z_crit = D.norm_ppf(1.0 - alpha)
        return float(_norm.cdf(math.sqrt(numer / denom) - z_crit))

    if solve_for == "power":
        assert ki is not None
        achieved = _power_at(ki)
        k_val = ki
    else:
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 2, 2
        k_max = 10_000_000
        while hi <= k_max:
            if _power_at(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError(f"failed to bracket Ki within {k_max}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _power_at(mid) >= power:
                hi = mid
            else:
                lo = mid
        k_val = hi
        achieved = _power_at(k_val)

    total_k = 2 * k_val
    n_total = int(round(total_k * m))
    return {
        "method_id": "cluster_two_poisson_rates",
        "solve_for": solve_for,
        "n": n_total,
        "n_clusters": total_k,
        "m_per_cluster": int(m),
        "ki": k_val,
        "k1": k_val,
        "k2": k_val,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "in a Cluster-Randomized Design",
            "Hayes, R.J. and Bennett, S. (1999). Simple sample size calculation "
            "for cluster-randomized trials. Int. J. Epidemiol. 28:319-326.",
            "Hayes, R.J. and Moulton, L.H. (2009). Cluster Randomised Trials. "
            "CRC Press.",
        ],
    }
