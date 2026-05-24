"""Distribution helpers used across power formulas.

Wraps `scipy.stats` non-central distributions with a consistent API and
provides high-precision fallbacks for the regions where scipy is
unstable. Imports are deferred so that registry/CLI usage does not pay
the scipy import cost.
"""
from __future__ import annotations


def nct_cdf(x: float, df: float, ncp: float) -> float:
    """Non-central t CDF. Thin wrapper for now; will fall back to mpmath
    when |ncp| is large and scipy loses precision.
    """
    from scipy.stats import nct  # local import
    return float(nct.cdf(x, df, ncp))


def nct_ppf(q: float, df: float, ncp: float) -> float:
    from scipy.stats import nct
    return float(nct.ppf(q, df, ncp))


def ncf_cdf(x: float, dfn: float, dfd: float, ncp: float) -> float:
    from scipy.stats import ncf
    return float(ncf.cdf(x, dfn, dfd, ncp))


def norm_ppf(q: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(q))


def t_ppf(q: float, df: float) -> float:
    from scipy.stats import t
    return float(t.ppf(q, df))
