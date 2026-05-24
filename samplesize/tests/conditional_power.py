"""Conditional and predictive power at an interim look (t-tests).

- Chapter 402 — Conditional Power of One-Sample T-Tests
- Chapter 403 — Conditional Power of Paired T-Tests
- Chapter 433 — Conditional Power of Two-Sample T-Tests

All three share the Jennison & Turnbull (2000, pp. 205-208) /
Chang (2008, pp. 69-70) information-based form.  Given the interim
test statistic ``Z_k`` computed from ``n_k`` of the planned ``N``
subjects, the *upper one-sided* conditional power for rejecting H0 at
the final analysis is

    P_uk(θ) = Φ( ( Z_k·√I_k − z_{1-α}·√I_K + θ·(I_K − I_k) )
                  / √(I_K − I_k) )

with mirror image expressions for ``Ha: μ1 < μ0`` and a two-tailed
sum for ``sides=2``.  The information levels depend on the design:

    one-sample t :     I_k = n_k / σ²              I_K = N / σ²
    paired t      :    I_k = n_k / σ_d²            I_K = N / σ_d²
    two-sample t  :    I_k = (σ₁²/n1k + σ₂²/n2k)⁻¹
                       I_K = (σ₁²/N1  + σ₂²/N2 )⁻¹

The "predictive power" of Jennison & Turnbull (2000, pp. 210-213) is
also returned for completeness.

``Z_k`` may be supplied directly via ``z_observed`` or implicitly via
``mean_observed`` (the interim sample mean / mean difference).
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# ---------------------------------------------------------------------------
# Core conditional / predictive power kernel (works in information units).
# ---------------------------------------------------------------------------


def _cp_from_information(
    *,
    z_k: float,
    i_k: float,
    i_K: float,
    theta: float,
    alpha: float,
    sides: int,
    direction: str = "upper",
) -> tuple[float, float]:
    """Return ``(conditional_power, predictive_power)`` from raw information.

    ``direction`` is ``"upper"`` (Ha: θ > 0), ``"lower"`` (Ha: θ < 0),
    or ignored for ``sides == 2``.
    """
    if i_K <= i_k:
        raise ValueError(
            f"I_K (={i_K}) must exceed I_k (={i_k}); the interim cannot "
            "be at or beyond the final sample size."
        )
    if sides not in (1, 2):
        raise ValueError(f"sides must be 1 or 2, got {sides}")

    from scipy.stats import norm

    diff = i_K - i_k
    sqrt_diff = math.sqrt(diff)
    sqrt_ik = math.sqrt(i_k)
    sqrt_iK = math.sqrt(i_K)

    if sides == 1:
        z_crit = D.norm_ppf(1.0 - alpha)
        if direction == "upper":
            arg = (z_k * sqrt_ik - z_crit * sqrt_iK + theta * diff) / sqrt_diff
            cp = float(norm.cdf(arg))
            # predictive power (upper one-sided)
            pp_arg = (z_k * sqrt_iK - z_crit * sqrt_ik) / sqrt_diff
            pp = float(norm.cdf(pp_arg))
        elif direction == "lower":
            arg = (-z_k * sqrt_ik - z_crit * sqrt_iK - theta * diff) / sqrt_diff
            cp = float(norm.cdf(arg))
            pp_arg = (-z_k * sqrt_iK - z_crit * sqrt_ik) / sqrt_diff
            pp = float(norm.cdf(pp_arg))
        else:
            raise ValueError(
                f"direction must be 'upper' or 'lower', got {direction!r}"
            )
    else:  # sides == 2
        z_crit = D.norm_ppf(1.0 - alpha / 2.0)
        upper_arg = (z_k * sqrt_ik - z_crit * sqrt_iK + theta * diff) / sqrt_diff
        lower_arg = (-z_k * sqrt_ik - z_crit * sqrt_iK - theta * diff) / sqrt_diff
        cp = float(norm.cdf(upper_arg) + norm.cdf(lower_arg))
        # predictive power (two-sided)
        abs_zk = abs(z_k)
        pp_u = (abs_zk * sqrt_iK - z_crit * sqrt_ik) / sqrt_diff
        pp_l = (-abs_zk * sqrt_iK - z_crit * sqrt_ik) / sqrt_diff
        pp = float(norm.cdf(pp_u) + norm.cdf(pp_l))

    # numerical guard — Φ pairs can drift to ~ -1e-16 / 1+1e-16
    cp = max(0.0, min(1.0, cp))
    pp = max(0.0, min(1.0, pp))
    return cp, pp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _direction_from_sides_and_mean(sides: int, theta: float,
                                   direction: str | None) -> str:
    """Resolve the rejection direction for one-sided tests.

    For ``sides == 1`` we infer from the sign of θ unless the caller
    overrides via ``direction``.  For ``sides == 2`` this is unused.
    """
    if sides == 2:
        return "upper"  # placeholder; not consulted by the kernel
    if direction is not None:
        d = direction.lower()
        if d not in ("upper", "lower"):
            raise ValueError(
                f"direction must be 'upper' or 'lower', got {direction!r}"
            )
        return d
    if theta >= 0:
        return "upper"
    return "lower"


def _z_from_mean(mean_observed: float, mean0: float,
                 sqrt_information_k: float) -> float:
    """Z_k = (x̄_k − μ0)·√I_k  (Chang 2008 p.69)."""
    return (mean_observed - mean0) * sqrt_information_k


# ---------------------------------------------------------------------------
# Chapter 402 — One-sample t-test
# ---------------------------------------------------------------------------


def conditional_power_one_sample_t(
    *,
    mean0: float,
    mean1: float,
    sd: float,
    n: int,
    n_t: int,
    z_observed: float | None = None,
    mean_observed: float | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    direction: str | None = None,
) -> dict[str, Any]:
    """Conditional power for a one-sample t-test at an interim look.

    Parameters
    ----------
    mean0, mean1 : float
        Null mean and design alternative mean.  θ = mean1 − mean0.
    sd : float
        Standard deviation of the response (σ).
    n : int
        Planned total sample size (N).
    n_t : int
        Sample size observed through the interim look (n_k).
    z_observed : float, optional
        Interim t-statistic Z_k computed from the first ``n_t`` subjects.
    mean_observed : float, optional
        Interim sample mean.  Used to compute Z_k when ``z_observed`` is
        not supplied: Z_k = (mean_observed − mean0)·√(n_t/σ²).
    alpha : float
        Final-analysis significance level.
    sides : {1, 2}
        Tails of the final test.
    direction : {"upper", "lower"}, optional
        Override for the one-sided alternative.  Defaults to the sign of
        (mean1 − mean0).
    """
    if sd <= 0:
        raise ValueError("sd must be positive")
    if n < 2 or n_t < 1 or n_t >= n:
        raise ValueError(
            f"need 1 <= n_t < n; got n_t={n_t}, n={n}"
        )
    if z_observed is None and mean_observed is None:
        raise ValueError("supply z_observed or mean_observed")
    if z_observed is not None and mean_observed is not None:
        raise ValueError(
            "supply exactly one of z_observed / mean_observed"
        )

    sigma2 = sd * sd
    i_k = n_t / sigma2
    i_K = n / sigma2
    theta = mean1 - mean0

    if z_observed is None:
        assert mean_observed is not None
        z_k = _z_from_mean(mean_observed, mean0, math.sqrt(i_k))
    else:
        z_k = float(z_observed)

    direction_resolved = _direction_from_sides_and_mean(sides, theta, direction)
    cp, pp = _cp_from_information(
        z_k=z_k, i_k=i_k, i_K=i_K, theta=theta,
        alpha=alpha, sides=sides, direction=direction_resolved,
    )

    return {
        "method_id": "conditional_power_one_sample_t",
        "solve_for": "conditional_power",
        "conditional_power": cp,
        "achieved_power": cp,           # alias for fixture key compatibility
        "predictive_power": pp,
        "futility_index": 1.0 - cp,
        "z_observed": z_k,
        "i_k": i_k,
        "i_K": i_K,
        "theta": theta,
        "n": n,
        "n_t": n_t,
        "inputs_echo": {
            "mean0": mean0, "mean1": mean1, "sd": sd, "n": n, "n_t": n_t,
            "z_observed": z_observed, "mean_observed": mean_observed,
            "alpha": alpha, "sides": sides, "direction": direction,
        },
        "citations": [
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods, pp. 205-208.",
            "Chang, M. (2008). Classical and Adaptive Clinical Trial Designs, p. 69.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 403 — Paired t-test
# ---------------------------------------------------------------------------


def conditional_power_paired_t(
    *,
    delta0: float = 0.0,
    delta1: float,
    sd: float,
    n: int,
    n_t: int,
    z_observed: float | None = None,
    mean_observed: float | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    direction: str | None = None,
) -> dict[str, Any]:
    """Conditional power for a paired t-test at an interim look.

    The paired test is mathematically identical to a one-sample test on
    the within-pair differences.  ``sd`` is σ_d, the SD of the
    differences.
    """
    if sd <= 0:
        raise ValueError("sd must be positive")
    if n < 2 or n_t < 1 or n_t >= n:
        raise ValueError(f"need 1 <= n_t < n; got n_t={n_t}, n={n}")
    if z_observed is None and mean_observed is None:
        raise ValueError("supply z_observed or mean_observed")
    if z_observed is not None and mean_observed is not None:
        raise ValueError("supply exactly one of z_observed / mean_observed")

    sigma2 = sd * sd
    i_k = n_t / sigma2
    i_K = n / sigma2
    theta = delta1 - delta0

    if z_observed is None:
        assert mean_observed is not None
        z_k = _z_from_mean(mean_observed, delta0, math.sqrt(i_k))
    else:
        z_k = float(z_observed)

    direction_resolved = _direction_from_sides_and_mean(sides, theta, direction)
    cp, pp = _cp_from_information(
        z_k=z_k, i_k=i_k, i_K=i_K, theta=theta,
        alpha=alpha, sides=sides, direction=direction_resolved,
    )

    return {
        "method_id": "conditional_power_paired_t",
        "solve_for": "conditional_power",
        "conditional_power": cp,
        "achieved_power": cp,
        "predictive_power": pp,
        "futility_index": 1.0 - cp,
        "z_observed": z_k,
        "i_k": i_k,
        "i_K": i_K,
        "theta": theta,
        "n": n,
        "n_t": n_t,
        "inputs_echo": {
            "delta0": delta0, "delta1": delta1, "sd": sd, "n": n, "n_t": n_t,
            "z_observed": z_observed, "mean_observed": mean_observed,
            "alpha": alpha, "sides": sides, "direction": direction,
        },
        "citations": [
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods, pp. 205-208.",
            "Chang, M. (2008). Classical and Adaptive Clinical Trial Designs, p. 69.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 433 — Two-sample t-test
# ---------------------------------------------------------------------------


def conditional_power_two_sample_t(
    *,
    mean1: float,
    mean2: float,
    sd1: float,
    sd2: float | None = None,
    n1: int,
    n2: int | None = None,
    n1_t: int,
    n2_t: int | None = None,
    z_observed: float | None = None,
    mean_diff_observed: float | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    direction: str | None = None,
) -> dict[str, Any]:
    """Conditional power for a two-sample t-test at an interim look.

    Parameters
    ----------
    mean1, mean2 : float
        Design means for groups 1 and 2.  θ = mean2 − mean1.
    sd1 : float
        SD for group 1.
    sd2 : float, optional
        SD for group 2; defaults to ``sd1`` if None.
    n1, n2 : int
        Planned final per-group sample sizes; ``n2`` defaults to ``n1``.
    n1_t, n2_t : int
        Interim per-group sample sizes; ``n2_t`` defaults to ``n1_t``.
    z_observed : float, optional
        Interim t-statistic computed from the data so far.
    mean_diff_observed : float, optional
        Interim observed mean difference (x̄_2 − x̄_1).  Used to compute
        Z_k when ``z_observed`` is None: Z_k = (x̄_2 − x̄_1)·√I_k.
    alpha : float, sides : {1, 2}, direction : {"upper","lower"}
        Final analysis configuration (see one-sample notes).
    """
    if sd2 is None:
        sd2 = sd1
    if n2 is None:
        n2 = n1
    if n2_t is None:
        n2_t = n1_t
    if sd1 <= 0 or sd2 <= 0:
        raise ValueError("sd1 and sd2 must be positive")
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must each be >= 2")
    if n1_t < 1 or n2_t < 1:
        raise ValueError("n1_t and n2_t must each be >= 1")
    if n1_t >= n1 or n2_t >= n2:
        raise ValueError(
            f"interim sample sizes must be strictly < final: "
            f"n1_t={n1_t}/n1={n1}, n2_t={n2_t}/n2={n2}"
        )
    if z_observed is None and mean_diff_observed is None:
        raise ValueError("supply z_observed or mean_diff_observed")
    if z_observed is not None and mean_diff_observed is not None:
        raise ValueError(
            "supply exactly one of z_observed / mean_diff_observed"
        )

    i_k = 1.0 / (sd1 * sd1 / n1_t + sd2 * sd2 / n2_t)
    i_K = 1.0 / (sd1 * sd1 / n1 + sd2 * sd2 / n2)
    theta = mean2 - mean1

    if z_observed is None:
        assert mean_diff_observed is not None
        z_k = float(mean_diff_observed) * math.sqrt(i_k)
    else:
        z_k = float(z_observed)

    direction_resolved = _direction_from_sides_and_mean(sides, theta, direction)
    cp, pp = _cp_from_information(
        z_k=z_k, i_k=i_k, i_K=i_K, theta=theta,
        alpha=alpha, sides=sides, direction=direction_resolved,
    )

    return {
        "method_id": "conditional_power_two_sample_t",
        "solve_for": "conditional_power",
        "conditional_power": cp,
        "achieved_power": cp,
        "predictive_power": pp,
        "futility_index": 1.0 - cp,
        "z_observed": z_k,
        "i_k": i_k,
        "i_K": i_K,
        "theta": theta,
        "n1": n1,
        "n2": n2,
        "n1_t": n1_t,
        "n2_t": n2_t,
        "inputs_echo": {
            "mean1": mean1, "mean2": mean2,
            "sd1": sd1, "sd2": sd2,
            "n1": n1, "n2": n2,
            "n1_t": n1_t, "n2_t": n2_t,
            "z_observed": z_observed,
            "mean_diff_observed": mean_diff_observed,
            "alpha": alpha, "sides": sides, "direction": direction,
        },
        "citations": [
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods, pp. 205-208.",
            "Chang, M. (2008). Classical and Adaptive Clinical Trial Designs, p. 70.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 101 — One-proportion z-test
# ---------------------------------------------------------------------------


def conditional_power_one_proportion(
    *,
    p0: float,
    p1: float,
    n: int,
    n_t: int,
    z_observed: float | None = None,
    p_observed: float | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    direction: str | None = None,
) -> dict[str, Any]:
    """Conditional power for a one-proportion z-test at an interim look.

    Parameters
    ----------
    p0 : float
        Null hypothesis proportion (H0: P = p0).
    p1 : float
        Alternative proportion under Ha.  θ = p1 − p0.
    n : int
        Planned total sample size (N).
    n_t : int
        Number of subjects observed through the interim look (n_k).
    z_observed : float, optional
        Interim z-statistic computed from the observed data.
    p_observed : float, optional
        Interim sample proportion.  Used to compute Z_k when
        ``z_observed`` is not supplied: Z_k = (p_obs − p0)·√I_k
        where I_k = n_k / σ² and σ² = p̄(1 − p̄), p̄ = (p0 + p1)/2.
    alpha : float
        Final-analysis significance level.
    sides : {1, 2}
        Tails of the final test.
    direction : {"upper", "lower"}, optional
        Override for the one-sided alternative.
    """
    if not (0.0 < p0 < 1.0) or not (0.0 < p1 < 1.0):
        raise ValueError("p0 and p1 must be in (0, 1)")
    if n < 2 or n_t < 1 or n_t >= n:
        raise ValueError(f"need 1 <= n_t < n; got n_t={n_t}, n={n}")
    if z_observed is None and p_observed is None:
        raise ValueError("supply z_observed or p_observed")
    if z_observed is not None and p_observed is not None:
        raise ValueError("supply exactly one of z_observed / p_observed")

    # σ² = p̄(1 − p̄),  p̄ = (p0 + p1)/2  (Chang 2008, p. 70)
    p_bar = (p0 + p1) / 2.0
    sigma2 = p_bar * (1.0 - p_bar)
    i_k = n_t / sigma2
    i_K = n / sigma2
    theta = p1 - p0

    if z_observed is None:
        assert p_observed is not None
        z_k = _z_from_mean(float(p_observed), p0, math.sqrt(i_k))
    else:
        z_k = float(z_observed)

    direction_resolved = _direction_from_sides_and_mean(sides, theta, direction)
    cp, pp = _cp_from_information(
        z_k=z_k, i_k=i_k, i_K=i_K, theta=theta,
        alpha=alpha, sides=sides, direction=direction_resolved,
    )

    return {
        "method_id": "conditional_power_one_proportion",
        "solve_for": "conditional_power",
        "conditional_power": cp,
        "achieved_power": cp,
        "predictive_power": pp,
        "futility_index": 1.0 - cp,
        "z_observed": z_k,
        "i_k": i_k,
        "i_K": i_K,
        "theta": theta,
        "n": n,
        "n_t": n_t,
        "inputs_echo": {
            "p0": p0, "p1": p1, "n": n, "n_t": n_t,
            "z_observed": z_observed, "p_observed": p_observed,
            "alpha": alpha, "sides": sides, "direction": direction,
        },
        "citations": [
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods, pp. 205-208.",
            "Chang, M. (2008). Classical and Adaptive Clinical Trial Designs, p. 70.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 202 — Two-proportion z-test
# ---------------------------------------------------------------------------


def conditional_power_two_proportions(
    *,
    p1: float,
    p2: float,
    n1: int,
    n2: int | None = None,
    n1_t: int,
    n2_t: int | None = None,
    z_observed: float | None = None,
    diff_observed: float | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    direction: str | None = None,
) -> dict[str, Any]:
    """Conditional power for a two-proportion z-test at an interim look.

    Parameters
    ----------
    p1, p2 : float
        Proportions in groups 1 and 2 under the alternative.
        θ = p2 − p1.
    n1, n2 : int
        Planned final per-group sample sizes; ``n2`` defaults to ``n1``.
    n1_t, n2_t : int
        Interim per-group sample sizes; ``n2_t`` defaults to ``n1_t``.
    z_observed : float, optional
        Interim z-statistic.
    diff_observed : float, optional
        Interim observed proportion difference (p2_k − p1_k).  Used to
        compute Z_k = (diff_observed)·√I_k when ``z_observed`` is None.
    alpha : float, sides : {1, 2}, direction : {"upper","lower"}
        Final analysis configuration.
    """
    if n2 is None:
        n2 = n1
    if n2_t is None:
        n2_t = n1_t
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0):
        raise ValueError("p1 and p2 must be in (0, 1)")
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must each be >= 2")
    if n1_t < 1 or n2_t < 1:
        raise ValueError("n1_t and n2_t must each be >= 1")
    if n1_t >= n1 or n2_t >= n2:
        raise ValueError(
            f"interim sample sizes must be strictly < final: "
            f"n1_t={n1_t}/n1={n1}, n2_t={n2_t}/n2={n2}"
        )
    if z_observed is None and diff_observed is None:
        raise ValueError("supply z_observed or diff_observed")
    if z_observed is not None and diff_observed is not None:
        raise ValueError("supply exactly one of z_observed / diff_observed")

    # σ² = p̄(1 − p̄),  p̄ = (p1 + p2)/2  (Chang 2008, pp. 70-71)
    p_bar = (p1 + p2) / 2.0
    sigma2 = p_bar * (1.0 - p_bar)
    i_k = (1.0 / (1.0 / n1_t + 1.0 / n2_t)) / sigma2
    i_K = (1.0 / (1.0 / n1 + 1.0 / n2)) / sigma2
    theta = p2 - p1

    if z_observed is None:
        assert diff_observed is not None
        z_k = float(diff_observed) * math.sqrt(i_k)
    else:
        z_k = float(z_observed)

    direction_resolved = _direction_from_sides_and_mean(sides, theta, direction)
    cp, pp = _cp_from_information(
        z_k=z_k, i_k=i_k, i_K=i_K, theta=theta,
        alpha=alpha, sides=sides, direction=direction_resolved,
    )

    return {
        "method_id": "conditional_power_two_proportions",
        "solve_for": "conditional_power",
        "conditional_power": cp,
        "achieved_power": cp,
        "predictive_power": pp,
        "futility_index": 1.0 - cp,
        "z_observed": z_k,
        "i_k": i_k,
        "i_K": i_K,
        "theta": theta,
        "n1": n1,
        "n2": n2,
        "n1_t": n1_t,
        "n2_t": n2_t,
        "inputs_echo": {
            "p1": p1, "p2": p2,
            "n1": n1, "n2": n2,
            "n1_t": n1_t, "n2_t": n2_t,
            "z_observed": z_observed,
            "diff_observed": diff_observed,
            "alpha": alpha, "sides": sides, "direction": direction,
        },
        "citations": [
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods, pp. 205-208.",
            "Chang, M. (2008). Classical and Adaptive Clinical Trial Designs, pp. 70-71.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 701 — Logrank test
# ---------------------------------------------------------------------------


def conditional_power_logrank(
    *,
    hazard_ratio: float,
    events: int,
    events_t: int,
    p1: float = 0.5,
    z_observed: float | None = None,
    alpha: float = 0.05,
    sides: int = 2,
    direction: str | None = None,
) -> dict[str, Any]:
    """Conditional power for a logrank test at an interim look.

    Parameters
    ----------
    hazard_ratio : float
        Hazard ratio of treatment (group 2) to control (group 1) under Ha.
        HR < 1 implies the treatment reduces the hazard.
        θ = log(HR).
    events : int
        Total planned number of events (E) across both groups.
    events_t : int
        Number of events observed through the interim look (E_k).
    p1 : float
        Proportion of subjects assigned to group 1 (control), default 0.5.
    z_observed : float
        Interim logrank z-statistic (required).
    alpha : float
        Final-analysis significance level.
    sides : {1, 2}
        Tails of the final test.
    direction : {"upper", "lower"}, optional
        Override for the one-sided alternative.  Defaults to the sign of
        log(HR).
    """
    if hazard_ratio <= 0 or hazard_ratio == 1.0:
        raise ValueError("hazard_ratio must be positive and not equal to 1")
    if events < 2 or events_t < 1 or events_t >= events:
        raise ValueError(
            f"need 1 <= events_t < events; got events_t={events_t}, events={events}"
        )
    if not (0.0 < p1 < 1.0):
        raise ValueError("p1 must be in (0, 1)")
    if z_observed is None:
        raise ValueError("z_observed is required for the logrank CP calculation")

    # I_k = E_k · P1 · (1 − P1),  I_K = E · P1 · (1 − P1)  (Chang 2008, p. 71)
    i_k = float(events_t) * p1 * (1.0 - p1)
    i_K = float(events) * p1 * (1.0 - p1)
    theta = math.log(hazard_ratio)
    z_k = float(z_observed)

    direction_resolved = _direction_from_sides_and_mean(sides, theta, direction)
    cp, pp = _cp_from_information(
        z_k=z_k, i_k=i_k, i_K=i_K, theta=theta,
        alpha=alpha, sides=sides, direction=direction_resolved,
    )

    return {
        "method_id": "conditional_power_logrank",
        "solve_for": "conditional_power",
        "conditional_power": cp,
        "achieved_power": cp,
        "predictive_power": pp,
        "futility_index": 1.0 - cp,
        "z_observed": z_k,
        "i_k": i_k,
        "i_K": i_K,
        "theta": theta,
        "log_hazard_ratio": theta,
        "hazard_ratio": hazard_ratio,
        "events": events,
        "events_t": events_t,
        "inputs_echo": {
            "hazard_ratio": hazard_ratio, "events": events, "events_t": events_t,
            "p1": p1, "z_observed": z_observed,
            "alpha": alpha, "sides": sides, "direction": direction,
        },
        "citations": [
            "Jennison, C. & Turnbull, B.W. (2000). Group Sequential Methods, pp. 205-208.",
            "Chang, M. (2008). Classical and Adaptive Clinical Trial Designs, p. 71.",
        ],
    }
