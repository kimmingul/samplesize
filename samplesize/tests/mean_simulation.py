"""Monte-Carlo simulation tests for means.

Implements five simulation-based mean-test power procedures that share a common simulation core:

  * Chapter 410 — Tests for One Mean (Simulation)
    -> :func:`tests_one_mean_simulation`

  * Chapter 440 — Tests for Two Means (Simulation)
    -> :func:`tests_two_means_simulation`

  * Chapter 490 — Tests for Paired Means (Simulation)
    -> :func:`tests_paired_means_simulation`

  * Chapter 465 — Equivalence Tests for Two Means (Simulation)
    -> :func:`equivalence_two_means_simulation`

  * Chapter 495 — Equivalence Tests for Paired Means (Simulation)
    -> :func:`equivalence_paired_means_simulation`

All functions use Monte-Carlo simulation (default n_sims=10000, seed=42)
with the standard one-sample or two-sample t-test as the primary
statistic (standard practice for simulation-based tests).

The simulation procedure uses a pool-based sampling strategy; we
use direct numpy RNG sampling which is equivalent in the asymptotic limit.

Tolerance on power: ±0.015 (combined MC noise).

------------------
Chapters 410, 440, 465, 490, 495.
Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap.
Chapman & Hall.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as _stats


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _t_pvalue_one_sample(x: np.ndarray, mu0: float, sides: int) -> np.ndarray:
    """Vectorised one-sample t p-value.  x shape (n_sims, n)."""
    n = x.shape[1]
    xbar = x.mean(axis=1)
    se = x.std(axis=1, ddof=1) / math.sqrt(n)
    se = np.where(se <= 0, np.finfo(float).tiny, se)
    t = (xbar - mu0) / se
    df = n - 1
    if sides == 2:
        pv = 2.0 * _stats.t.sf(np.abs(t), df)
    else:
        pv = _stats.t.sf(t, df)
    return pv


def _t_pvalue_two_sample(x1: np.ndarray, x2: np.ndarray,
                          diff0: float, sides: int) -> np.ndarray:
    """Vectorised pooled-variance two-sample t p-value.
    x1 shape (n_sims, n1), x2 shape (n_sims, n2)."""
    n1, n2 = x1.shape[1], x2.shape[1]
    xbar1 = x1.mean(axis=1)
    xbar2 = x2.mean(axis=1)
    s1sq = x1.var(axis=1, ddof=1)
    s2sq = x2.var(axis=1, ddof=1)
    sp2 = ((n1 - 1) * s1sq + (n2 - 1) * s2sq) / (n1 + n2 - 2)
    sp2 = np.where(sp2 <= 0, np.finfo(float).tiny, sp2)
    se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    t = (xbar1 - xbar2 - diff0) / se
    df = n1 + n2 - 2
    if sides == 2:
        pv = 2.0 * _stats.t.sf(np.abs(t), df)
    else:
        pv = _stats.t.sf(t, df)
    return pv


def _tost_pvalue_two_sample(x1: np.ndarray, x2: np.ndarray,
                              eq_lower: float, eq_upper: float) -> np.ndarray:
    """TOST p-value (max of two one-sided p-values) for two-sample."""
    n1, n2 = x1.shape[1], x2.shape[1]
    xbar1 = x1.mean(axis=1)
    xbar2 = x2.mean(axis=1)
    s1sq = x1.var(axis=1, ddof=1)
    s2sq = x2.var(axis=1, ddof=1)
    sp2 = ((n1 - 1) * s1sq + (n2 - 1) * s2sq) / (n1 + n2 - 2)
    sp2 = np.where(sp2 <= 0, np.finfo(float).tiny, sp2)
    se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    df = n1 + n2 - 2
    diff = xbar1 - xbar2
    # Upper test: H0 diff >= EU  vs H1 diff < EU
    t_upper = (diff - eq_upper) / se
    pv_upper = _stats.t.cdf(t_upper, df)
    # Lower test: H0 diff <= EL  vs H1 diff > EL
    t_lower = (diff - eq_lower) / se
    pv_lower = _stats.t.sf(t_lower, df)
    return np.maximum(pv_upper, pv_lower)


def _tost_pvalue_one_sample(x: np.ndarray, eq_lower: float,
                              eq_upper: float) -> np.ndarray:
    """TOST p-value for one-sample (paired differences)."""
    n = x.shape[1]
    xbar = x.mean(axis=1)
    se = x.std(axis=1, ddof=1) / math.sqrt(n)
    se = np.where(se <= 0, np.finfo(float).tiny, se)
    df = n - 1
    t_upper = (xbar - eq_upper) / se
    pv_upper = _stats.t.cdf(t_upper, df)
    t_lower = (xbar - eq_lower) / se
    pv_lower = _stats.t.sf(t_lower, df)
    return np.maximum(pv_upper, pv_lower)


def _sim_power(reject: np.ndarray) -> float:
    return float(reject.mean())


def _solve_n_sim(power_fn, target_power: float,
                 n_min: int = 2, n_max: int = 100_000) -> tuple[int, float]:
    """Binary-search for smallest n achieving >= target_power."""
    lo, hi = n_min, n_min
    while hi <= n_max:
        if power_fn(hi) >= target_power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"Could not achieve power={target_power} for n ≤ {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if power_fn(mid) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi, power_fn(hi)


# ---------------------------------------------------------------------------
# Chapter 410: Tests for One Mean (Simulation)
# ---------------------------------------------------------------------------

def tests_one_mean_simulation(
    *,
    mean0: float,
    mean1: float,
    sd: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    dist: str = "normal",
    n_sims: int = 10000,
    seed: int = 42,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Simulation-based power for one-mean t-test.

    Parameters
    ----------
    mean0 : float
        Mean under H₀.
    mean1 : float
        True mean under H₁ (at which power is computed).
    sd : float
        Standard deviation.
    alpha : float
        Significance level.
    power : float or None
        Target power (provide when solving for n).
    n : int or None
        Sample size (provide when solving for power).
    sides : int
        1 or 2.
    dist : str
        Distribution: "normal" (default), "t", "uniform", "exponential".
    n_sims : int
        Number of simulation replications.
    seed : int
        Random seed for reproducibility.
    solve_for : str or None
        "n" or "power".
    """
    if sd <= 0:
        raise ValueError("sd must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    inputs_echo: dict[str, Any] = {
        "mean0": mean0, "mean1": mean1, "sd": sd, "alpha": alpha,
        "power": power, "n": n, "sides": sides, "dist": dist,
        "n_sims": n_sims, "seed": seed,
    }

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    rng = np.random.default_rng(seed)

    def _draw(n_val: int, mean: float) -> np.ndarray:
        """Draw (n_sims, n_val) array with given mean and sd."""
        if dist == "normal":
            return rng.normal(mean, sd, size=(n_sims, n_val))
        if dist == "t":
            df_t = 10.0
            scale = math.sqrt(df_t / (df_t - 2.0))
            return mean + sd * rng.standard_t(df_t, size=(n_sims, n_val)) / scale
        if dist == "uniform":
            s3 = math.sqrt(3.0)
            return mean + sd * rng.uniform(-s3, s3, size=(n_sims, n_val))
        if dist == "exponential":
            return mean + sd * (rng.standard_exponential(size=(n_sims, n_val)) - 1.0)
        raise ValueError(f"unsupported dist {dist!r}")

    def power_at_n(n_val: int) -> float:
        x = _draw(n_val, mean1)
        pv = _t_pvalue_one_sample(x, mean0, sides)
        return _sim_power(pv < alpha)

    if solve_for == "power":
        assert n is not None
        achieved = power_at_n(n)
        result: dict[str, Any] = {"n": n, "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        if not 0 < power < 1:
            raise ValueError("power must be in (0, 1)")
        n_found, achieved = _solve_n_sim(power_at_n, power)
        result = {"n": n_found, "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "tests_one_mean_simulation",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 440: Tests for Two Means (Simulation)
# ---------------------------------------------------------------------------

def tests_two_means_simulation(
    *,
    mean1: float,
    mean2: float,
    sd: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    sides: int = 2,
    diff0: float = 0.0,
    dist: str = "normal",
    n_sims: int = 10000,
    seed: int = 42,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Simulation-based power for two-means t-test.

    Parameters
    ----------
    mean1, mean2 : float
        True group means under H₁.
    sd : float
        Common within-group standard deviation.
    alpha : float
        Significance level.
    power : float or None
        Target power.
    n1, n2 : int or None
        Group sample sizes.
    allocation : float
        n2 = ceil(allocation * n1) when n2 is not given.
    sides : int
        1 or 2.
    diff0 : float
        Null hypothesis difference (default 0).
    dist : str
        "normal", "t", "uniform", or "exponential".
    n_sims, seed : int
        Simulation controls.
    """
    if sd <= 0:
        raise ValueError("sd must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))

    inputs_echo: dict[str, Any] = {
        "mean1": mean1, "mean2": mean2, "sd": sd, "alpha": alpha,
        "power": power, "n1": n1, "n2": n2, "allocation": allocation,
        "sides": sides, "diff0": diff0, "dist": dist,
        "n_sims": n_sims, "seed": seed,
    }

    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply one of (n1/n2, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    rng = np.random.default_rng(seed)

    def _draw(n_val: int, mean: float) -> np.ndarray:
        if dist == "normal":
            return rng.normal(mean, sd, size=(n_sims, n_val))
        if dist == "t":
            df_t = 10.0
            scale = math.sqrt(df_t / (df_t - 2.0))
            return mean + sd * rng.standard_t(df_t, size=(n_sims, n_val)) / scale
        if dist == "uniform":
            s3 = math.sqrt(3.0)
            return mean + sd * rng.uniform(-s3, s3, size=(n_sims, n_val))
        if dist == "exponential":
            return mean + sd * (rng.standard_exponential(size=(n_sims, n_val)) - 1.0)
        raise ValueError(f"unsupported dist {dist!r}")

    def power_at_n(n1_val: int) -> float:
        n2_val = max(2, math.ceil(allocation * n1_val))
        x1 = _draw(n1_val, mean1)
        x2 = _draw(n2_val, mean2)
        pv = _t_pvalue_two_sample(x1, x2, diff0, sides)
        return _sim_power(pv < alpha)

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        x1 = _draw(n1, mean1)
        x2 = _draw(n2, mean2)
        pv = _t_pvalue_two_sample(x1, x2, diff0, sides)
        achieved = _sim_power(pv < alpha)
        result: dict[str, Any] = {"n1": n1, "n2": n2, "n": n1 + n2,
                                   "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        if not 0 < power < 1:
            raise ValueError("power must be in (0, 1)")
        n1_found, achieved = _solve_n_sim(power_at_n, power)
        n2_found = max(2, math.ceil(allocation * n1_found))
        result = {"n1": n1_found, "n2": n2_found, "n": n1_found + n2_found,
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "tests_two_means_simulation",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 490: Tests for Paired Means (Simulation)
# ---------------------------------------------------------------------------

def tests_paired_means_simulation(
    *,
    mean_diff: float,
    mean_diff0: float = 0.0,
    sd: float,
    corr: float = 0.0,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    dist: str = "normal",
    n_sims: int = 10000,
    seed: int = 42,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Simulation-based power for paired t-test.

    Paired data are simulated as correlated bivariate normals; the
    difference has mean ``mean_diff`` and SD
    ``sd_diff = sd * sqrt(2 * (1 - corr))``.

    Parameters
    ----------
    mean_diff : float
        True paired mean difference (A – B) under H₁.
    mean_diff0 : float
        Null hypothesis mean difference (default 0).
    sd : float
        SD of each item A and B (assumed equal).
    corr : float
        Correlation between items A and B (default 0).
    alpha : float
        Significance level.
    power, n : float/int or None
        Supply one; the other is solved for.
    sides : int
        1 or 2.
    dist : str
        "normal" (default), others use the same rescaling as the ANOVA
        simulation modules.
    n_sims, seed : int
        Simulation controls.
    """
    if sd <= 0:
        raise ValueError("sd must be positive")
    if not -1 < corr < 1:
        raise ValueError("corr must be in (-1, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    inputs_echo: dict[str, Any] = {
        "mean_diff": mean_diff, "mean_diff0": mean_diff0, "sd": sd,
        "corr": corr, "alpha": alpha, "power": power, "n": n,
        "sides": sides, "dist": dist, "n_sims": n_sims, "seed": seed,
    }

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    # SD of the difference
    sd_diff = sd * math.sqrt(2.0 * (1.0 - corr))
    rng = np.random.default_rng(seed)

    def _draw_diff(n_val: int) -> np.ndarray:
        """Paired differences shape (n_sims, n_val)."""
        if dist == "normal":
            return rng.normal(mean_diff, sd_diff, size=(n_sims, n_val))
        if dist == "t":
            df_t = 10.0
            scale = math.sqrt(df_t / (df_t - 2.0))
            return (mean_diff
                    + sd_diff * rng.standard_t(df_t, size=(n_sims, n_val)) / scale)
        if dist == "uniform":
            s3 = math.sqrt(3.0)
            return mean_diff + sd_diff * rng.uniform(-s3, s3, size=(n_sims, n_val))
        if dist == "exponential":
            return (mean_diff
                    + sd_diff * (rng.standard_exponential(size=(n_sims, n_val)) - 1.0))
        raise ValueError(f"unsupported dist {dist!r}")

    def power_at_n(n_val: int) -> float:
        diffs = _draw_diff(n_val)
        pv = _t_pvalue_one_sample(diffs, mean_diff0, sides)
        return _sim_power(pv < alpha)

    if solve_for == "power":
        assert n is not None
        achieved = power_at_n(n)
        result: dict[str, Any] = {"n": n, "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        if not 0 < power < 1:
            raise ValueError("power must be in (0, 1)")
        n_found, achieved = _solve_n_sim(power_at_n, power)
        result = {"n": n_found, "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "tests_paired_means_simulation",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 465: Equivalence Tests for Two Means (Simulation)
# ---------------------------------------------------------------------------

def equivalence_two_means_simulation(
    *,
    mean1: float,
    mean2: float,
    sd: float,
    eq_lower: float,
    eq_upper: float,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    allocation: float = 1.0,
    dist: str = "normal",
    n_sims: int = 10000,
    seed: int = 42,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """TOST simulation for two means.

    Tests H₀: (μ₁−μ₂) ≤ eq_lower or (μ₁−μ₂) ≥ eq_upper
    vs   H₁: eq_lower < (μ₁−μ₂) < eq_upper

    using Schuirmann's TOST (two one-sided t-tests).

    Parameters
    ----------
    mean1, mean2 : float
        True group means under H₁ (H₁ mean diff = mean1 - mean2).
    sd : float
        Common within-group SD.
    eq_lower, eq_upper : float
        Equivalence limits (eq_lower < 0 < eq_upper for symmetric).
    alpha : float
        Per-test significance level (0.05 typical).
    power, n1/n2 : float/int or None
        Supply one pair; the other is solved for.
    allocation : float
        n2 = ceil(allocation * n1).
    dist : str
        "normal", "t", "uniform", "exponential".
    n_sims, seed : int
        Simulation controls.
    """
    if sd <= 0:
        raise ValueError("sd must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if eq_lower >= eq_upper:
        raise ValueError("eq_lower must be < eq_upper")
    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))

    inputs_echo: dict[str, Any] = {
        "mean1": mean1, "mean2": mean2, "sd": sd,
        "eq_lower": eq_lower, "eq_upper": eq_upper,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2,
        "allocation": allocation, "dist": dist, "n_sims": n_sims, "seed": seed,
    }

    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply one of (n1/n2, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    rng = np.random.default_rng(seed)

    def _draw(n_val: int, mean: float) -> np.ndarray:
        if dist == "normal":
            return rng.normal(mean, sd, size=(n_sims, n_val))
        if dist == "t":
            df_t = 10.0
            scale = math.sqrt(df_t / (df_t - 2.0))
            return mean + sd * rng.standard_t(df_t, size=(n_sims, n_val)) / scale
        if dist == "uniform":
            s3 = math.sqrt(3.0)
            return mean + sd * rng.uniform(-s3, s3, size=(n_sims, n_val))
        if dist == "exponential":
            return mean + sd * (rng.standard_exponential(size=(n_sims, n_val)) - 1.0)
        raise ValueError(f"unsupported dist {dist!r}")

    def power_at_n(n1_val: int) -> float:
        n2_val = max(2, math.ceil(allocation * n1_val))
        x1 = _draw(n1_val, mean1)
        x2 = _draw(n2_val, mean2)
        pv = _tost_pvalue_two_sample(x1, x2, eq_lower, eq_upper)
        return _sim_power(pv < alpha)

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        x1 = _draw(n1, mean1)
        x2 = _draw(n2, mean2)
        pv = _tost_pvalue_two_sample(x1, x2, eq_lower, eq_upper)
        achieved = _sim_power(pv < alpha)
        result: dict[str, Any] = {"n1": n1, "n2": n2, "n": n1 + n2,
                                   "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        if not 0 < power < 1:
            raise ValueError("power must be in (0, 1)")
        n1_found, achieved = _solve_n_sim(power_at_n, power)
        n2_found = max(2, math.ceil(allocation * n1_found))
        result = {"n1": n1_found, "n2": n2_found, "n": n1_found + n2_found,
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "equivalence_two_means_simulation",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Schuirmann, D.J. (1987). A comparison of the two one-sided tests "
            "procedure and the power approach for assessing bioequivalence. "
            "Journal of Pharmacokinetics and Biopharmaceutics, 15, 657-680.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 495: Equivalence Tests for Paired Means (Simulation)
# ---------------------------------------------------------------------------

def equivalence_paired_means_simulation(
    *,
    mean_diff: float,
    sd: float,
    eq_lower: float,
    eq_upper: float,
    corr: float = 0.0,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    dist: str = "normal",
    n_sims: int = 10000,
    seed: int = 42,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """TOST simulation for paired means.

    Tests H₀: diff ≤ eq_lower or diff ≥ eq_upper
    vs   H₁: eq_lower < diff < eq_upper

    using paired TOST (two one-sided paired t-tests).

    Parameters
    ----------
    mean_diff : float
        True paired mean difference under H₁.
    sd : float
        SD of each item (assumed equal for both members of pair).
    eq_lower, eq_upper : float
        Equivalence limits.
    corr : float
        Correlation between paired items (default 0).
    alpha : float
        Per-test significance level.
    power, n : float/int or None
        Supply one; the other is solved for.
    dist : str
        "normal" (default), "t", "uniform", "exponential".
    n_sims, seed : int
        Simulation controls.
    """
    if sd <= 0:
        raise ValueError("sd must be positive")
    if not -1 < corr < 1:
        raise ValueError("corr must be in (-1, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if eq_lower >= eq_upper:
        raise ValueError("eq_lower must be < eq_upper")

    inputs_echo: dict[str, Any] = {
        "mean_diff": mean_diff, "sd": sd,
        "eq_lower": eq_lower, "eq_upper": eq_upper,
        "corr": corr, "alpha": alpha, "power": power, "n": n,
        "dist": dist, "n_sims": n_sims, "seed": seed,
    }

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    sd_diff = sd * math.sqrt(2.0 * (1.0 - corr))
    rng = np.random.default_rng(seed)

    def _draw_diff(n_val: int) -> np.ndarray:
        if dist == "normal":
            return rng.normal(mean_diff, sd_diff, size=(n_sims, n_val))
        if dist == "t":
            df_t = 10.0
            scale = math.sqrt(df_t / (df_t - 2.0))
            return (mean_diff
                    + sd_diff * rng.standard_t(df_t, size=(n_sims, n_val)) / scale)
        if dist == "uniform":
            s3 = math.sqrt(3.0)
            return mean_diff + sd_diff * rng.uniform(-s3, s3, size=(n_sims, n_val))
        if dist == "exponential":
            return (mean_diff
                    + sd_diff * (rng.standard_exponential(size=(n_sims, n_val)) - 1.0))
        raise ValueError(f"unsupported dist {dist!r}")

    def power_at_n(n_val: int) -> float:
        diffs = _draw_diff(n_val)
        pv = _tost_pvalue_one_sample(diffs, eq_lower, eq_upper)
        return _sim_power(pv < alpha)

    if solve_for == "power":
        assert n is not None
        achieved = power_at_n(n)
        result: dict[str, Any] = {"n": n, "achieved_power": achieved}
    elif solve_for == "n":
        assert power is not None
        if not 0 < power < 1:
            raise ValueError("power must be in (0, 1)")
        n_found, achieved = _solve_n_sim(power_at_n, power)
        result = {"n": n_found, "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "equivalence_paired_means_simulation",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Schuirmann, D.J. (1987). A comparison of the two one-sided tests "
            "procedure and the power approach for assessing bioequivalence. "
            "Journal of Pharmacokinetics and Biopharmaceutics, 15, 657-680.",
        ],
    }
