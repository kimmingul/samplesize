"""Group-sequential logrank test — O'Brien-Fleming / Pocock spending.


The Schoenfeld (1981) events formula gives the total events required for a
fixed-sample logrank test.  For a group-sequential design the required drift
θ is inflated by the sequential correction: D_seq = (θ_seq / θ_fixed)² * D.

This implementation reuses the boundary and drift solver from
``samplesize.tests.group_sequential`` and converts the drift back to sample
size via the logrank events formula:

    θ = |log(HR)| * sqrt(E * p1 * (1-p1))   (Schoenfeld 1981)
    E = θ² / (log(HR)² * p1 * (1-p1))       total events needed
    N = E / Pr_ev                             total subjects

where Pr_ev = p1*Pev1 + (1-p1)*Pev2 is the overall event probability.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D
from samplesize.tests.group_sequential import (
    _exit_probability,
    _solve_boundaries,
    _solve_drift,
)


def _logrank_theta(
    s1: float, s2: float, p1: float, pev1: float, pev2: float,
    events: int,
) -> float:
    """Schoenfeld drift for fixed-sample logrank with E total events."""
    hr = math.log(s2) / math.log(s1)
    if hr <= 0.0 or hr == 1.0:
        raise ValueError("HR must be positive and != 1")
    pr_ev = p1 * pev1 + (1.0 - p1) * pev2
    if pr_ev <= 0.0:
        raise ValueError("overall event probability must be > 0")
    return abs(math.log(hr)) * math.sqrt(events * p1 * (1.0 - p1))


def group_sequential_logrank(
    *,
    s1: float,
    s2: float,
    pev1: float | None = None,
    pev2: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    p1: float = 0.5,
    n_looks: int = 4,
    boundary: str = "obrien-fleming",
    info_frac: list[float] | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Group-sequential logrank test (O'Brien-Fleming or Pocock boundary).

    Group-sequential logrank test (Wang & Tsiatis 1987; Schoenfeld 1981).

    Parameters
    ----------
    s1, s2
        Survival proportions at t=1 for the two groups (S1, S2).
    pev1, pev2
        Probability of event in each group.  Defaults to 1 - s1, 1 - s2
        (assumes all subjects followed to t=1).
    alpha
        Two-sided overall type-I error.
    power
        Target power (solve for N/events when given).
    n
        Total sample size (solve for power when given).
    p1
        Proportion in group 1.
    n_looks
        Number of interim analyses (including final).
    boundary
        Spending function: ``"obrien-fleming"`` or ``"pocock"``.
    info_frac
        Optional information fractions; default equally spaced.
    sides
        2 (default) for two-sided test.
    """
    if s1 <= 0 or s1 >= 1 or s2 <= 0 or s2 >= 1:
        raise ValueError("s1 and s2 must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    boundary_key = boundary.lower().replace("_", "-")

    if pev1 is None:
        pev1 = 1.0 - s1
    if pev2 is None:
        pev2 = 1.0 - s2

    hr = math.log(s2) / math.log(s1)
    if hr <= 0.0 or hr == 1.0:
        raise ValueError("HR derived from (s1, s2) must be positive and != 1")

    pr_ev = p1 * pev1 + (1.0 - p1) * pev2
    if pr_ev <= 0.0:
        raise ValueError("overall event probability must be > 0")

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply exactly one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if info_frac is None:
        info_frac = [(k + 1) / n_looks for k in range(n_looks)]
    else:
        info_frac = list(info_frac)
        n_looks = len(info_frac)
    if abs(info_frac[-1] - 1.0) > 1e-9:
        raise ValueError("info_frac must end at 1.0")

    boundaries = _solve_boundaries(info_frac, alpha, boundary_key, sides)
    log_hr = math.log(hr)

    if solve_for == "power":
        assert n is not None
        events = int(round(n * pr_ev))
        theta = abs(log_hr) * math.sqrt(events * p1 * (1.0 - p1))
        achieved = _exit_probability(boundaries, info_frac,
                                     drift=theta, sides=sides)
        n_total = n
        events_total = events
    else:
        assert power is not None
        # Find drift needed, then back out events and N
        theta_seq = _solve_drift(boundaries, info_frac, power, sides)
        # E = theta² / (log(HR)² * p1 * (1-p1))
        events_float = theta_seq ** 2 / (log_hr ** 2 * p1 * (1.0 - p1))
        events_total = max(1, math.ceil(events_float - 1e-9))
        # Recompute achieved with integer events
        theta_actual = abs(log_hr) * math.sqrt(
            events_total * p1 * (1.0 - p1))
        achieved = _exit_probability(boundaries, info_frac,
                                     drift=theta_actual, sides=sides)
        # If rounding caused undershoot, bump events
        while achieved < power and events_total < 100_000_000:
            events_total += 1
            theta_actual = abs(log_hr) * math.sqrt(
                events_total * p1 * (1.0 - p1))
            achieved = _exit_probability(boundaries, info_frac,
                                         drift=theta_actual, sides=sides)
        n_float = events_total / pr_ev
        n_total = max(2, math.ceil(n_float - 1e-9))

    n1 = round(n_total * p1)
    n2 = n_total - n1
    events_per_look = [
        round(events_total * f) for f in info_frac
    ]

    return {
        "method_id": "group_sequential_logrank",
        "solve_for": solve_for,
        "n": n_total,
        "n1": n1,
        "n2": n2,
        "events": events_total,
        "events_per_look": events_per_look,
        "achieved_power": achieved,
        "z_boundaries": list(boundaries),
        "info_frac": list(info_frac),
        "n_looks": n_looks,
        "hazard_ratio": hr,
        "inputs_echo": dict(
            s1=s1, s2=s2, pev1=pev1, pev2=pev2, alpha=alpha,
            power=power, n=n, p1=p1, n_looks=n_looks,
            boundary=boundary, sides=sides,
        ),
        "citations": [
            "Lan, K.K.G. & DeMets, D.L. (1983). Discrete sequential "
            "boundaries for clinical trials. Biometrika 70.",
            "Schoenfeld, D.A. (1981). The asymptotic properties of "
            "nonparametric tests for comparing survival distributions. "
            "Biometrika 68:316-319.",
            "O'Brien, P.C. & Fleming, T.R. (1979). A multiple testing "
            "procedure for clinical trials. Biometrics 35.",
        ],
    }
