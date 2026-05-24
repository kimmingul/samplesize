"""Logrank test in a cluster-randomized design.

Implements ``cluster_logrank`` — see
"Logrank Tests in a Cluster-Randomized Design".

The formula combines Freedman (1982) non-cluster logrank events formula
with the Xie & Waksman (2003) cluster inflation:

    e = (z_{1-alpha/2} + z_{1-beta})^2 * (1 + r*HR)^2 / (r * (1-HR)^2)
    e_c = e * (1 + (Mbar - 1) * rho)

where Mbar = (K1*M1 + K2*M2) / (K1+K2) is the overall average cluster size,
rho is the ICC on the censoring indicator, and
HR = ln(S2) / ln(S1) is the hazard ratio.

Power for a given design is computed by inverting this formula:
actual event count = N * (1-S1 + r*(1-S2)) / (1+r), then

    e_eff = e_actual / (1 + (Mbar-1)*rho)
    Power = Phi(sqrt(e_eff * r * (1-HR)^2 / (1+r*HR)^2) - z_{1-alpha/2})
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


def _logrank_cluster_power(
    *,
    s1: float,
    s2: float,
    k1: int,
    k2: int,
    m1: float,
    m2: float,
    icc: float,
    alpha: float,
    sides: int,
) -> float:
    """Power of cluster logrank test given K1, K2, M1, M2."""
    if not (0.0 < s1 < 1.0 and 0.0 < s2 < 1.0):
        raise ValueError("s1 and s2 must lie in (0, 1)")
    if not 0.0 <= icc < 1.0:
        raise ValueError("icc must lie in [0, 1)")
    if k1 < 2 or k2 < 2:
        return 0.0
    if m1 < 1 or m2 < 1:
        raise ValueError("cluster sizes must be >= 1")

    from scipy.stats import norm as _norm

    hr = math.log(s2) / math.log(s1)
    n1 = k1 * m1
    n2 = k2 * m2
    n_total = n1 + n2
    r = n2 / n1  # allocation ratio
    m_bar = (k1 * m1 + k2 * m2) / (k1 + k2)
    vif = 1.0 + (m_bar - 1.0) * icc

    # Actual event count (proportional-hazards assumption, no censoring)
    event_fraction = (1.0 - s1 + r * (1.0 - s2)) / (1.0 + r)
    e_actual = n_total * event_fraction

    # Effective events after cluster inflation
    e_eff = e_actual / vif
    if e_eff <= 0:
        return 0.0

    denom = (1.0 + r * hr) ** 2
    if denom <= 0:
        return 0.0

    ncp_sq = e_eff * r * (1.0 - hr) ** 2 / denom
    if ncp_sq < 0:
        return 0.0

    if sides == 2:
        z_crit = D.norm_ppf(1.0 - alpha / 2.0)
    else:
        z_crit = D.norm_ppf(1.0 - alpha)
    return float(_norm.cdf(math.sqrt(ncp_sq) - z_crit))


def _k1_for_logrank_cluster(
    *,
    s1: float,
    s2: float,
    m: float,
    icc: float,
    alpha: float,
    power: float,
    sides: int,
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

    def p_at(k1: int) -> float:
        return _logrank_cluster_power(
            s1=s1, s2=s2, k1=k1, k2=k2_for(k1),
            m1=m, m2=m, icc=icc, alpha=alpha, sides=sides,
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
    achieved = _logrank_cluster_power(
        s1=s1, s2=s2, k1=k1, k2=k2, m1=m, m2=m,
        icc=icc, alpha=alpha, sides=sides,
    )
    return k1, k2, achieved


def cluster_logrank(
    *,
    s1: float,
    s2: float,
    m: float,
    icc: float,
    alpha: float = 0.05,
    power: float | None = None,
    k_clusters: int | None = None,
    sides: int = 2,
    allocation: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Logrank test power/sample-size in a cluster-randomized design.

    Parameters
    ----------
    s1 : float
        Proportion surviving (non-events) in group 1 (control).
    s2 : float
        Proportion surviving in group 2 (treatment).
    m : float
        Average cluster size (applied to both arms).
    icc : float
        Intracluster correlation coefficient on the censoring indicator.
    alpha : float
        Type-I error (default 0.05).
    power : float, optional
        Target power; supply this **or** ``k_clusters``.
    k_clusters : int, optional
        Number of clusters in arm 1; supply this **or** ``power``.
    sides : int
        2 (default) or 1.
    allocation : float
        Ratio k2/k1 (default 1.0 for balanced).
    solve_for : {"n", "power"}, optional
        Override auto-detected target.
    """
    inputs_echo = dict(
        s1=s1, s2=s2, m=m, icc=icc, alpha=alpha,
        power=power, k_clusters=k_clusters,
        sides=sides, allocation=allocation,
    )
    have_k = k_clusters is not None
    have_power = power is not None
    if have_k == have_power:
        raise ValueError("supply exactly one of (power, k_clusters)")
    if solve_for is None:
        solve_for = "n" if not have_k else "power"

    if not (0.0 < s1 < 1.0 and 0.0 < s2 < 1.0):
        raise ValueError("s1 and s2 must lie in (0, 1)")
    if not 0.0 <= icc < 1.0:
        raise ValueError("icc must lie in [0, 1)")

    hr = math.log(s2) / math.log(s1)

    if solve_for == "power":
        assert k_clusters is not None
        k1 = int(k_clusters)
        k2 = max(2, math.ceil(allocation * k1))
        achieved = _logrank_cluster_power(
            s1=s1, s2=s2, k1=k1, k2=k2, m1=m, m2=m,
            icc=icc, alpha=alpha, sides=sides,
        )
    elif solve_for == "n":
        assert power is not None
        k1, k2, achieved = _k1_for_logrank_cluster(
            s1=s1, s2=s2, m=m, icc=icc, alpha=alpha, power=power,
            sides=sides, allocation=allocation,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    n1 = int(round(k1 * m))
    n2 = int(round(k2 * m))
    e1 = round(n1 * (1.0 - s1), 1)
    e2 = round(n2 * (1.0 - s2), 1)
    vif = 1.0 + (m - 1.0) * icc
    return {
        "method_id": "cluster_logrank",
        "solve_for": solve_for,
        "n": n1 + n2,
        "n_clusters": k1 + k2,
        "m_per_cluster": int(m),
        "k1": k1,
        "k2": k2,
        "n1": n1,
        "n2": n2,
        "e1": e1,
        "e2": e2,
        "hr": hr,
        "achieved_power": achieved,
        "vif": vif,
        "inputs_echo": inputs_echo,
        "citations": [
            "Xie, T. and Waksman, J. (2003). Design and sample size estimation "
            "in clinical trials with clustered survival times as the primary "
            "endpoint. Statist. Med. 22:2835-2846.",
            "Freedman, L.S. (1982). Tables of the number of patients required "
            "in clinical trials using the logrank test. Stat. Med. 1:121-129.",
            "Campbell, M.J. and Walters, S.J. (2014). How to Design, Analyse "
            "and Report Cluster Randomised Trials. Wiley.",
        ],
    }
