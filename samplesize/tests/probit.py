"""Probit analysis power and sample size — Kodell et al. (2010).


Tests H0: ρ=1 vs H1: ρ>1 where ρ=LD50(Treatment)/LD50(Control).

The sample size per dose-group is:

    n = 2·(t_{f,1-α} + t_{f,1-β})² / (β1·log10(ρ))²·Σw_i

The power is:

    t_{f,1-β} = sqrt(n·(β1·log10(ρ))²·Σw_i / 2) - t_{f,1-α}

where f = 2g-3 degrees of freedom, g = number of dose groups, and the
probit weight for dose group i is:

    w_i = φ(Φ⁻¹(P_i))² / (P_i·(1-P_i))

Reference: Kodell, R.L., Lensing, S., Landes, R., Kumar, K.S., and
Hauer-Jensen, M. (2010). Sample Size Calculations for Evaluating a
Radioprotective Agent in a Rodent Lethality Model.
Radiation Research, 173, 237-247.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm as normdist
from scipy.stats import t as tdist


# ---------------------------------------------------------------------------
# Weight and slope computation
# ---------------------------------------------------------------------------


def _probit_weight(p: float) -> float:
    """Probit weight w_i = φ(Φ⁻¹(p))² / (p·(1-p))."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"response proportion {p} must be in (0, 1)")
    z = normdist.ppf(p)
    phi = normdist.pdf(z)
    return phi ** 2 / (p * (1.0 - p))


def _compute_slope_from_doses(
    proportions: list[float],
    doses: list[float],
) -> float:
    """Compute probit slope β1 from a set of doses and response proportions."""
    if len(proportions) != len(doses):
        raise ValueError("proportions and doses must have the same length")
    log_doses = [math.log10(d) for d in doses]
    probit_vals = [normdist.ppf(p) for p in proportions]
    n = len(log_doses)
    mean_x = sum(log_doses) / n
    mean_y = sum(probit_vals) / n
    sxy = sum((log_doses[i] - mean_x) * (probit_vals[i] - mean_y) for i in range(n))
    sxx = sum((log_doses[i] - mean_x) ** 2 for i in range(n))
    if sxx == 0:
        raise ValueError("All log10(doses) are equal; cannot compute slope")
    return sxy / sxx


# ---------------------------------------------------------------------------
# Core power function
# ---------------------------------------------------------------------------


def _power_at_n(
    *,
    n_per_group: int,
    beta1: float,
    rho: float,
    proportions: list[float],
    alpha: float,
) -> float:
    """Power for the probit test at a given per-group sample size."""
    g = len(proportions)
    f = 2 * g - 3
    sum_w = sum(_probit_weight(p) for p in proportions)
    log_rho = math.log10(rho)
    numer = n_per_group * (beta1 * log_rho) ** 2 * sum_w / 2.0
    if numer <= 0:
        return 0.0
    t_alpha = tdist.ppf(1.0 - alpha, f)
    t_beta = math.sqrt(numer) - t_alpha
    return float(tdist.cdf(t_beta, f))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def probit_analysis(
    *,
    rho: float,
    proportions: list[float],
    beta1: float | None = None,
    doses: list[float] | None = None,
    alpha: float = 0.025,
    n_per_group: int | None = None,
    power: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power and sample size for probit analysis (comparative LD50 study).

    Tests H0: ρ=1 (LD50 equal in treatment and control) vs H1: ρ>1
    using the Kodell et al. (2010) formula.


    Parameters
    ----------
    rho
        Relative potency LD50(T)/LD50(C).  Must be > 1.
    proportions
        List of g target lethality proportions (one per dose group).
        E.g. [0.05, 0.275, 0.5, 0.725, 0.95] for a 5-dose design.
    beta1
        Probit slope (log10-dose scale).  Provide this OR ``doses``.
    doses
        Control doses (one per proportion).  Used to compute beta1 if
        ``beta1`` is not supplied.
    alpha
        One-sided type-I error.  Typically 0.025 or 0.05.
    n_per_group
        Per-dose-group sample size (supply when solve_for="power").
    power
        Target power (supply when solve_for="n").
    solve_for
        ``"n"`` or ``"power"``.
    """
    if rho <= 1.0:
        raise ValueError("rho must be > 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if len(proportions) < 2:
        raise ValueError("at least 2 dose groups required")

    # Resolve slope
    if beta1 is None and doses is None:
        raise ValueError("supply either beta1 or doses")
    if beta1 is None:
        assert doses is not None
        beta1 = _compute_slope_from_doses(proportions, doses)
    if beta1 <= 0:
        raise ValueError("beta1 must be positive")

    inputs_echo: dict[str, Any] = {
        "rho": rho,
        "proportions": proportions,
        "beta1": beta1,
        "alpha": alpha,
        "n_per_group": n_per_group,
        "power": power,
    }

    given = sum(x is not None for x in (n_per_group, power))
    if given == 0:
        raise ValueError("supply exactly one of (n_per_group, power)")
    if given == 2 and solve_for is None:
        raise ValueError(
            "both n_per_group and power given; specify solve_for explicitly"
        )
    if solve_for is None:
        solve_for = "n" if n_per_group is None else "power"

    g = len(proportions)
    f = 2 * g - 3
    sum_w = sum(_probit_weight(p) for p in proportions)
    log_rho = math.log10(rho)

    if solve_for == "power":
        assert n_per_group is not None
        achieved = _power_at_n(
            n_per_group=n_per_group, beta1=beta1, rho=rho,
            proportions=proportions, alpha=alpha,
        )
        n_out_per_group = n_per_group

    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        t_alpha = tdist.ppf(1.0 - alpha, f)
        t_beta = tdist.ppf(power, f)
        # Direct formula: n = 2·(t_α+t_β)² / ((β1·log10(ρ))²·Σw_i)
        n_exact = 2.0 * (t_alpha + t_beta) ** 2 / (
            (beta1 * log_rho) ** 2 * sum_w
        )
        n_out_per_group = math.ceil(n_exact)
        # Verify and bump if needed
        while _power_at_n(
            n_per_group=n_out_per_group, beta1=beta1, rho=rho,
            proportions=proportions, alpha=alpha,
        ) < power:
            n_out_per_group += 1
        achieved = _power_at_n(
            n_per_group=n_out_per_group, beta1=beta1, rho=rho,
            proportions=proportions, alpha=alpha,
        )

    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    total_n = n_out_per_group * 2 * g  # 2 groups × g dose levels

    return {
        "method_id": "probit_analysis",
        "solve_for": solve_for,
        "n": total_n,
        "n_per_group": n_out_per_group,
        "achieved_power": achieved,
        "beta1_used": beta1,
        "inputs_echo": inputs_echo,
        "citations": [
            "Kodell, R.L., Lensing, S., Landes, R., Kumar, K.S., and "
            "Hauer-Jensen, M. (2010). Sample Size Calculations for "
            "Evaluating a Radioprotective Agent in a Rodent Lethality "
            "Model. Radiation Research, 173, 237-247.",
        ],
    }
