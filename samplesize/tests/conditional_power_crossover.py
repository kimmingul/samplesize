"""Conditional power for a 2×2 cross-over design at an interim look.


At an interim look when ``n_k`` of the planned ``N`` subjects have been
enrolled, the conditional power (CP) of eventually rejecting H0 at the
end of the study is computed from the observed test statistic ``z_k``
(the t-value from the current data).

From Jennison & Turnbull (2000) pp. 205-208 and Chang (2008) p. 71,
the information levels for the 2×2 cross-over t-test are

    I_k = n_k / σ_d²
    I_K = N  / σ_d²

where σ_d is the within-subject SD of the pair differences, and the
conditional power formula is (upper one-sided, Ha: δ1 > δ0)

    CP = Φ( (Z_k·√I_k − z_{1-α}·√I_K + θ·(I_K − I_k)) / √(I_K − I_k) )

with θ = δ1 − δ0 (expected difference under Ha).

The predictive power (Bayesian average of CP) is also returned:

    PP = Φ( (Z_k·√I_K − z_{1-α}·√I_k) / √(I_K − I_k) )

Two-sided and lower one-sided forms follow by symmetry.

References
----------
* Jennison, C. and Turnbull, B.W. (2000). Group Sequential Methods with
  Applications to Clinical Trials. Chapman & Hall/CRC.
* Chang, M. (2008). Classical and Adaptive Clinical Trial Designs.
  John Wiley & Sons.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Core CP/PP kernel
# ---------------------------------------------------------------------------


def _conditional_power_crossover(
    *,
    z_k: float,
    n_k: int,
    n_total: int,
    delta0: float,
    delta1: float,
    sigma_d: float,
    alpha: float,
    sides: int,
) -> tuple[float, float]:
    """Return (conditional_power, predictive_power) for 2×2 cross-over.

    Parameters
    ----------
    z_k
        Observed t-statistic at the interim look.
    n_k
        Subjects enrolled at interim.
    n_total
        Planned total subjects (N).
    delta0
        Mean difference under H0.
    delta1
        Mean difference under Ha (alternative).
    sigma_d
        SD of within-subject differences.
    alpha
        Final type-I error rate.
    sides
        1 or 2.
    """
    if sigma_d <= 0:
        raise ValueError("sigma_d must be positive")
    if n_k >= n_total:
        raise ValueError("n_k must be less than n_total")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    sigma2 = sigma_d ** 2
    i_k = n_k / sigma2
    i_K = n_total / sigma2
    theta = delta1 - delta0

    diff = i_K - i_k
    sqrt_diff = math.sqrt(diff)
    sqrt_ik = math.sqrt(i_k)
    sqrt_iK = math.sqrt(i_K)

    if sides == 1:
        z_crit = D.norm_ppf(1.0 - alpha)
        # Upper one-sided: Ha: δ1 > δ0
        arg_cp = (z_k * sqrt_ik - z_crit * sqrt_iK + theta * diff) / sqrt_diff
        cp = float(norm.cdf(arg_cp))
        # Predictive power
        arg_pp = (z_k * sqrt_iK - z_crit * sqrt_ik) / sqrt_diff
        pp = float(norm.cdf(arg_pp))
    else:
        z_crit = D.norm_ppf(1.0 - alpha / 2.0)
        upper_arg = (z_k * sqrt_ik - z_crit * sqrt_iK + theta * diff) / sqrt_diff
        lower_arg = (-z_k * sqrt_ik - z_crit * sqrt_iK - theta * diff) / sqrt_diff
        cp = float(norm.cdf(upper_arg) + norm.cdf(lower_arg))
        abs_zk = abs(z_k)
        pp_u = (abs_zk * sqrt_iK - z_crit * sqrt_ik) / sqrt_diff
        pp_l = (-abs_zk * sqrt_iK - z_crit * sqrt_ik) / sqrt_diff
        pp = float(norm.cdf(pp_u) + norm.cdf(pp_l))

    return cp, pp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def conditional_power_2x2_crossover(
    *,
    n_total: int,
    n_k: int,
    delta0: float = 0.0,
    delta1: float,
    sigma_d: float,
    z_k: float | None = None,
    mean_diff_observed: float | None = None,
    alpha: float = 0.025,
    sides: int = 1,
) -> dict[str, Any]:
    """Conditional and predictive power for a 2×2 cross-over interim analysis.


    Parameters
    ----------
    n_total
        Total planned sample size N for the study.
    n_k
        Number of subjects enrolled at the interim look.
    delta0
        Null mean difference (usually 0).
    delta1
        Alternative mean difference (the effect assumed under Ha).
    sigma_d
        Standard deviation of within-subject pair differences.
    z_k
        Observed test statistic (t-value) at the interim.  Provide
        exactly one of (z_k, mean_diff_observed).
    mean_diff_observed
        If the interim test statistic is not available, supply the
        observed mean difference instead.  z_k is then computed as
        ``(mean_diff_observed - delta0) * sqrt(n_k / sigma_d²)``.
    alpha
        Final type-I error rate.  Default 0.025 (one-sided).
    sides
        1 (default, one-sided Ha: δ1 > δ0) or 2 (two-sided).
    """
    if n_total < 2:
        raise ValueError("n_total must be >= 2")
    if n_k < 1 or n_k >= n_total:
        raise ValueError("n_k must satisfy 1 <= n_k < n_total")
    if sigma_d <= 0:
        raise ValueError("sigma_d must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    # Resolve z_k
    if z_k is None and mean_diff_observed is None:
        raise ValueError("supply exactly one of (z_k, mean_diff_observed)")
    if z_k is not None and mean_diff_observed is not None:
        raise ValueError("supply exactly one of (z_k, mean_diff_observed), not both")
    if mean_diff_observed is not None:
        # t = (d_bar - delta0) * sqrt(I_k) where I_k = n_k / sigma_d^2
        i_k_hat = n_k / (sigma_d ** 2)
        z_k = (mean_diff_observed - delta0) * math.sqrt(i_k_hat)

    assert z_k is not None

    cp, pp = _conditional_power_crossover(
        z_k=z_k,
        n_k=n_k,
        n_total=n_total,
        delta0=delta0,
        delta1=delta1,
        sigma_d=sigma_d,
        alpha=alpha,
        sides=sides,
    )

    futility_index = 1.0 - cp

    inputs_echo: dict[str, Any] = {
        "n_total": n_total, "n_k": n_k,
        "delta0": delta0, "delta1": delta1,
        "sigma_d": sigma_d, "z_k": z_k,
        "mean_diff_observed": mean_diff_observed,
        "alpha": alpha, "sides": sides,
    }

    return {
        "method_id": "conditional_power_2x2_crossover",
        "solve_for": "power",
        "n": n_total,
        "achieved_power": cp,
        "conditional_power": cp,
        "predictive_power": pp,
        "futility_index": futility_index,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cross-Over Designs.",
            "Jennison, C. and Turnbull, B.W. (2000). Group Sequential "
            "Methods with Applications to Clinical Trials. "
            "Chapman & Hall/CRC.",
            "Chang, M. (2008). Classical and Adaptive Clinical Trial "
            "Designs. John Wiley & Sons.",
            "Proschan, M., Lan, K.K.G., Wittes, J.T. (2006). Statistical "
            "Monitoring of Clinical Trials. Springer.",
        ],
    }
