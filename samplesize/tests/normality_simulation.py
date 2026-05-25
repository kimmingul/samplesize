"""Monte-Carlo simulation-based normality tests.

Implements power analysis by simulation for standard goodness-of-fit
normality tests:

* "Normality Tests (Simulation)"
  -> :func:`normality_tests_simulation`

Supported test statistics:
- Shapiro-Wilk (1965)
- Anderson-Darling (1954)
- D'Agostino-Pearson K^2 omnibus (1971/1990)

The function draws ``n_sims`` samples of size ``n`` from the specified
alternative distribution, applies the selected test, and reports the fraction
of rejections at the given alpha level as the estimated power.

Alternative distributions supported: ``normal`` (size/type-I only),
``lognormal``, ``t``, ``uniform``, ``exponential``, ``laplace``, ``beta``,
``gamma``, ``cauchy``, ``logistic``, ``weibull``.

References
----------
Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance test for
normality (complete samples). Biometrika, 52, 591-611.

Anderson, T. W. and Darling, D. A. (1954). A test of goodness of fit.
Journal of the American Statistical Association, 49, 765-769.

D'Agostino, R. B. and Pearson, E. S. (1973). Tests for departure from
normality. Biometrika, 60, 613-622.

D'Agostino, R., Belanger, A. and D'Agostino, R. B. Jr. (1990).
A suggestion for using powerful and informative tests of normality.
American Statistician, 44, 316-321.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Supported tests and distributions
# ---------------------------------------------------------------------------

_SUPPORTED_TESTS = ("shapiro_wilk", "anderson_darling", "dagostino_pearson")

_SUPPORTED_DISTS = (
    "normal", "lognormal", "t", "uniform", "exponential",
    "laplace", "beta", "gamma", "cauchy", "logistic", "weibull",
)


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def _draw_samples(
    rng: np.random.Generator,
    *,
    n: int,
    n_sims: int,
    distribution: str,
    dist_params: dict,
) -> np.ndarray:
    """Return (n_sims, n) array from the specified alternative distribution.

    Parameters
    ----------
    distribution
        One of ``_SUPPORTED_DISTS``.
    dist_params
        Distribution-specific parameters (see docstring of
        :func:`normality_tests_simulation`).
    """
    p = dist_params or {}

    if distribution == "normal":
        loc = float(p.get("loc", 0.0))
        scale = float(p.get("scale", 1.0))
        return rng.normal(loc=loc, scale=scale, size=(n_sims, n))

    if distribution == "lognormal":
        # Parameterised via underlying normal (mu_ln, sigma_ln)
        mu_ln = float(p.get("mu_ln", 0.0))
        sigma_ln = float(p.get("sigma_ln", 1.0))
        return rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=(n_sims, n))

    if distribution == "t":
        df = float(p.get("df", 5.0))
        if df <= 0:
            raise ValueError("t distribution requires df > 0")
        return rng.standard_t(df=df, size=(n_sims, n))

    if distribution == "uniform":
        low = float(p.get("low", 0.0))
        high = float(p.get("high", 1.0))
        return rng.uniform(low=low, high=high, size=(n_sims, n))

    if distribution == "exponential":
        scale = float(p.get("scale", 1.0))
        return rng.exponential(scale=scale, size=(n_sims, n))

    if distribution == "laplace":
        loc = float(p.get("loc", 0.0))
        scale = float(p.get("scale", 1.0))
        return rng.laplace(loc=loc, scale=scale, size=(n_sims, n))

    if distribution == "beta":
        a = float(p.get("a", 2.0))
        b = float(p.get("b", 5.0))
        return rng.beta(a=a, b=b, size=(n_sims, n))

    if distribution == "gamma":
        shape = float(p.get("shape", 2.0))
        scale = float(p.get("scale", 1.0))
        return rng.gamma(shape=shape, scale=scale, size=(n_sims, n))

    if distribution == "cauchy":
        loc = float(p.get("loc", 0.0))
        scale = float(p.get("scale", 1.0))
        return stats.cauchy.rvs(
            loc=loc, scale=scale, size=(n_sims, n), random_state=rng
        )

    if distribution == "logistic":
        loc = float(p.get("loc", 0.0))
        scale = float(p.get("scale", 1.0))
        return rng.logistic(loc=loc, scale=scale, size=(n_sims, n))

    if distribution == "weibull":
        shape = float(p.get("shape", 2.0))
        scale = float(p.get("scale", 1.0))
        # numpy weibull takes c = shape parameter; scipy convention differs
        return scale * rng.weibull(a=shape, size=(n_sims, n))

    raise ValueError(
        f"unsupported distribution {distribution!r}; choose one of {_SUPPORTED_DISTS}"
    )


# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------

def _run_test(samples: np.ndarray, test: str, alpha: float) -> float:
    """Run the selected normality test on each row of ``samples``.

    Parameters
    ----------
    samples : (n_sims, n) array
    test : str
    alpha : float

    Returns
    -------
    float
        Fraction of simulations that reject H0 (estimated power).
    """
    n_sims = samples.shape[0]
    rejects = 0

    if test == "shapiro_wilk":
        for row in samples:
            _, p = stats.shapiro(row)
            if p < alpha:
                rejects += 1

    elif test == "anderson_darling":
        # scipy anderson returns critical values, not p-values.
        # We compare the statistic to the critical value at the requested
        # significance level.  scipy provides critical values at
        # significance levels [15, 10, 5, 2.5, 1] percent.
        alpha_pct = alpha * 100.0
        # Find the closest supported significance level
        supported_pct = [15.0, 10.0, 5.0, 2.5, 1.0]
        # Pick nearest
        closest_idx = int(
            np.argmin([abs(alpha_pct - s) for s in supported_pct])
        )
        for row in samples:
            res = stats.anderson(row, dist="norm")
            crit = res.critical_values[closest_idx]
            if res.statistic > crit:
                rejects += 1

    elif test == "dagostino_pearson":
        for row in samples:
            _, p = stats.normaltest(row)
            if p < alpha:
                rejects += 1

    else:
        raise ValueError(
            f"unsupported test {test!r}; choose one of {_SUPPORTED_TESTS}"
        )

    return rejects / n_sims


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def normality_tests_simulation(
    *,
    n: int,
    distribution: str = "normal",
    dist_params: dict | None = None,
    test: str = "shapiro_wilk",
    alpha: float = 0.05,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte-Carlo simulation power for normality tests.

    Draws ``n_sims`` samples of size ``n`` from the specified alternative
    distribution, applies the chosen normality test, and reports the fraction
    of rejections as the estimated power.

    When ``distribution='normal'`` this estimates the empirical type-I error
    rate (should be approximately alpha).

    Parameters
    ----------
    n
        Sample size per simulation replicate (>= 3).
    distribution
        Alternative distribution to sample from.  One of:
        ``'normal'``, ``'lognormal'``, ``'t'``, ``'uniform'``,
        ``'exponential'``, ``'laplace'``, ``'beta'``, ``'gamma'``,
        ``'cauchy'``, ``'logistic'``, ``'weibull'``.
    dist_params
        Dict of distribution parameters (see below).  Defaults to each
        distribution's standard parameterisation.

        - ``normal``: ``loc`` (default 0), ``scale`` (default 1).
        - ``lognormal``: ``mu_ln`` (default 0), ``sigma_ln`` (default 1).
        - ``t``: ``df`` (default 5).
        - ``uniform``: ``low`` (default 0), ``high`` (default 1).
        - ``exponential``: ``scale`` (default 1).
        - ``laplace``: ``loc`` (default 0), ``scale`` (default 1).
        - ``beta``: ``a`` (default 2), ``b`` (default 5).
        - ``gamma``: ``shape`` (default 2), ``scale`` (default 1).
        - ``cauchy``: ``loc`` (default 0), ``scale`` (default 1).
        - ``logistic``: ``loc`` (default 0), ``scale`` (default 1).
        - ``weibull``: ``shape`` (default 2), ``scale`` (default 1).
    test
        Normality test to use.  One of:
        ``'shapiro_wilk'``, ``'anderson_darling'``, ``'dagostino_pearson'``.
    alpha
        Significance level (default 0.05).
    n_sims
        Number of Monte-Carlo replicates (default 10,000).
    seed
        RNG seed for reproducibility (default 42).

    Returns
    -------
    dict
        Standard envelope: method_id, solve_for, n, achieved_power,
        inputs_echo, citations.
    """
    if n < 3:
        raise ValueError("n must be >= 3")
    if distribution not in _SUPPORTED_DISTS:
        raise ValueError(
            f"distribution {distribution!r} not in {_SUPPORTED_DISTS}"
        )
    if test not in _SUPPORTED_TESTS:
        raise ValueError(f"test {test!r} not in {_SUPPORTED_TESTS}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_sims < 100:
        raise ValueError("n_sims must be >= 100")

    if dist_params is None:
        dist_params = {}

    rng = np.random.default_rng(seed)
    samples = _draw_samples(
        rng, n=n, n_sims=n_sims,
        distribution=distribution, dist_params=dist_params,
    )

    achieved = _run_test(samples, test=test, alpha=alpha)

    inputs_echo = {
        "n": n,
        "distribution": distribution,
        "dist_params": dist_params,
        "test": test,
        "alpha": alpha,
        "n_sims": n_sims,
        "seed": seed,
    }
    return {
        "method_id": "normality_tests_simulation",
        "solve_for": "power",
        "n": n,
        "achieved_power": achieved,
        "stochastic": True,
        "n_sims": n_sims,
        "seed": seed,
        "inputs_echo": inputs_echo,
        "citations": [
            "Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance test "
            "for normality (complete samples). Biometrika, 52, 591-611.",
            "Anderson, T. W. and Darling, D. A. (1954). A test of goodness of fit. "
            "Journal of the American Statistical Association, 49, 765-769.",
            "D'Agostino, R., Belanger, A. and D'Agostino, R. B. Jr. (1990). "
            "A suggestion for using powerful and informative tests of normality. "
            "American Statistician, 44, 316-321.",
        ],
        "notes": (
            "Monte-Carlo estimate; achieved_power is stochastic.  With "
            f"n_sims={n_sims}, the 95% binomial CI half-width is ~"
            f"{1.96 * math.sqrt(max(achieved, 1e-9) * (1 - min(achieved, 1 - 1e-9)) / n_sims):.4f}."
        ),
    }
