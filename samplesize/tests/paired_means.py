"""Paired t-test power / sample-size.

procedure uses the same mechanics as Tests for One Mean (applied to
the differences D = X1 - X2).  We therefore reduce the paired test to
a one-sample test on the differences with mean0=0.
"""
from __future__ import annotations

from typing import Any

from samplesize.tests.one_mean import one_sample_t


def paired_t(
    *,
    mean_diff: float | None = None,
    sd_diff: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    sd_known: bool = False,
    nonparametric: str = "ignore",
    solve_for: str | None = None,
    direction: str = "above",
) -> dict[str, Any]:
    """Two-sided / one-sided paired t-test solver.

    `mean_diff` is the alternative mean difference (H1).  The null
    hypothesis is that the population mean difference is zero.
    """
    inputs_echo = {
        "mean_diff": mean_diff, "sd_diff": sd_diff, "alpha": alpha,
        "power": power, "n": n, "sides": sides, "sd_known": sd_known,
        "nonparametric": nonparametric, "direction": direction,
    }

    inner = one_sample_t(
        mean0=0.0,
        mean1=mean_diff,
        sd=sd_diff,
        alpha=alpha,
        power=power,
        n=n,
        sides=sides,
        sd_known=sd_known,
        nonparametric=nonparametric,
        solve_for=solve_for,
        direction=direction,
    )

    out: dict[str, Any] = {
        "method_id": "paired_t",
        "solve_for": inner["solve_for"],
        "n": inner["n"],
        "achieved_power": inner["achieved_power"],
        "effect_d": inner["effect_d"],
    }
    if "mean1" in inner:
        # solve_for == "effect": the inner mean1 is the detectable mean
        # difference (since mean0=0).
        out["mean_diff"] = inner["mean1"]
    out["inputs_echo"] = inputs_echo
    out["citations"] = [
        "Zar, J.H. (1984). Biostatistical Analysis, 2nd ed.",
    ]
    return out
