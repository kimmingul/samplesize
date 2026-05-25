"""Monte-Carlo simulation-based normal-scores two-sample tests.

Implements two simulation-based power procedures:

* "Terry-Hoeffding Normal-Scores Tests of Means (Simulation)"
  -> :func:`terry_hoeffding_simulation`
* "Van der Waerden Normal Quantiles Tests of Means (Simulation)"
  -> :func:`van_der_waerden_simulation`

Both methods assign scores to ranks and then apply a two-sample t-test on
those scores.  The Terry-Hoeffding c1-statistic uses expected order statistics
(E[Z_(i:N)]) as scores; the Van der Waerden statistic uses normal quantiles
Phi^{-1}(i/(N+1)).

For large N the two score sets are nearly identical; they differ for small N.

References
----------
Terry, M. E. (1952). Some rank order tests which are most powerful against
specific parametric alternatives. Annals of Mathematical Statistics, 23, 346-366.

Hoeffding, W. (1951). Optimum nonparametric tests. Proceedings of the Second
Berkeley Symposium, 83-92.

Van der Waerden, B. L. (1953). Ein neuer Test fuer das Problem der zwei
Stichproben. Mathematische Annalen, 126, 93-107.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Score generation
# ---------------------------------------------------------------------------

def _terry_scores(N: int) -> np.ndarray:
    """Expected order statistics E[Z_(i:N)] for i=1..N (Terry/Hoeffding scores).

    Uses the Blom (1958) approximation:
        E[Z_(i:N)] ≈ Phi^{-1}((i - 3/8) / (N + 1/4))

    This is the standard approximation used in practice.

    Reference: Blom, G. (1958). Statistical Estimates and Transformed
    Beta-Variables. Wiley.
    """
    i = np.arange(1, N + 1, dtype=float)
    return stats.norm.ppf((i - 3.0 / 8.0) / (N + 0.25))


def _van_der_waerden_scores(N: int) -> np.ndarray:
    """Van der Waerden scores: Phi^{-1}(i / (N + 1)) for i=1..N."""
    i = np.arange(1, N + 1, dtype=float)
    return stats.norm.ppf(i / (N + 1.0))


# ---------------------------------------------------------------------------
# Vectorised test p-values
# ---------------------------------------------------------------------------

def _normal_scores_pvalues(
    s1: np.ndarray,
    s2: np.ndarray,
    score_fn,
    *,
    sides: int,
) -> np.ndarray:
    """Vectorised p-values for a normal-scores two-sample test.

    Parameters
    ----------
    s1 : (n_sims, n1) array
    s2 : (n_sims, n2) array
    score_fn : callable(N) -> (N,) array of scores
    sides : 1 or 2

    The procedure:
    1. Pool and rank both groups together.
    2. Replace ranks with the prescribed scores.
    3. Compute a two-sample t-statistic on the scores.
    4. Refer to t(N-2) distribution.
    """
    n_sims, n1 = s1.shape
    n2 = s2.shape[1]
    N = n1 + n2

    # Pre-compute scores lookup
    score_lookup = score_fn(N)  # shape (N,)

    pooled = np.concatenate([s1, s2], axis=1)  # (n_sims, N)
    # rankdata returns 1-based ranks
    ranks = stats.rankdata(pooled, method="average", axis=1).astype(int)  # (n_sims, N)
    # Map rank -> score; ranks are 1-based
    scores = score_lookup[ranks - 1]  # (n_sims, N)

    sc1 = scores[:, :n1]  # (n_sims, n1)
    sc2 = scores[:, n1:]  # (n_sims, n2)

    mean1 = sc1.mean(axis=1)
    mean2 = sc2.mean(axis=1)
    var1 = sc1.var(axis=1, ddof=1)
    var2 = sc2.var(axis=1, ddof=1)

    # Pooled variance t-test
    sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (N - 2)
    sp2 = np.where(sp2 <= 0, np.finfo(float).tiny, sp2)
    se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    t_stat = (mean1 - mean2) / se

    df = N - 2
    if sides == 1:
        pval = stats.t.sf(np.abs(t_stat), df=df)
    else:
        pval = 2.0 * stats.t.sf(np.abs(t_stat), df=df)

    return pval


# ---------------------------------------------------------------------------
# Shared distribution sampler (reuse pattern from nonparametric_simulation)
# ---------------------------------------------------------------------------

_SUPPORTED_DISTS = (
    "normal", "lognormal", "logistic", "cauchy", "uniform", "exponential",
)


def _sample_group(
    rng: np.random.Generator,
    *,
    n: int,
    mean: float,
    sigma: float,
    dist: str,
    n_sims: int,
) -> np.ndarray:
    """Return (n_sims, n) array of replicates from the specified distribution."""
    if dist == "normal":
        return mean + sigma * rng.standard_normal(size=(n_sims, n))

    if dist == "lognormal":
        if mean > 0:
            sigma_ln2 = math.log(1.0 + (sigma / mean) ** 2)
            sigma_ln = math.sqrt(sigma_ln2)
            mu_ln = math.log(mean) - sigma_ln2 / 2.0
            return rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=(n_sims, n))
        else:
            raw = rng.lognormal(mean=0.0, sigma=1.0, size=(n_sims, n))
            raw_mean = math.exp(0.5)
            raw_sd = math.sqrt((math.exp(1.0) - 1.0) * math.exp(1.0))
            return (raw - raw_mean) / raw_sd * sigma + mean

    if dist == "logistic":
        scale = sigma * math.sqrt(3.0) / math.pi
        return rng.logistic(loc=mean, scale=scale, size=(n_sims, n))

    if dist == "cauchy":
        return stats.cauchy.rvs(
            loc=mean, scale=sigma, size=(n_sims, n), random_state=rng
        )

    if dist == "uniform":
        half = math.sqrt(3.0) * sigma
        return rng.uniform(mean - half, mean + half, size=(n_sims, n))

    if dist == "exponential":
        raw = rng.exponential(scale=sigma, size=(n_sims, n))
        return raw + (mean - sigma)

    raise ValueError(
        f"unsupported distribution {dist!r}; choose one of {_SUPPORTED_DISTS}"
    )


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------

def terry_hoeffding_simulation(
    *,
    n1: int,
    n2: int,
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    distribution1: str = "normal",
    distribution2: str = "normal",
    alpha: float = 0.05,
    sides: int = 2,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte-Carlo simulation power for the Terry-Hoeffding c1 normal-scores test.

    The Terry-Hoeffding procedure replaces ranks with expected normal order
    statistics (Blom approximation) and applies a two-sample t-test on those
    scores.  This test is asymptotically equivalent to the van der Waerden
    test but uses E[Z_(i:N)] rather than Phi^{-1}(i/(N+1)).

    Parameters
    ----------
    n1
        Sample size in group 1 (>= 2).
    n2
        Sample size in group 2 (>= 2).
    mu1
        Mean (location) of group 1 under H1.
    mu2
        Mean (location) of group 2 under H1.
    sigma1
        Scale (SD) of group 1.
    sigma2
        Scale (SD) of group 2.
    distribution1
        Distribution family for group 1.  One of: 'normal', 'lognormal',
        'logistic', 'cauchy', 'uniform', 'exponential'.
    distribution2
        Distribution family for group 2.
    alpha
        Significance level (default 0.05).
    sides
        1 for one-sided, 2 for two-sided (default).
    n_sims
        Monte-Carlo replicates (default 10,000).
    seed
        RNG seed (default 42).

    Returns
    -------
    dict
        Standard envelope: method_id, solve_for, n, achieved_power,
        inputs_echo, citations.
    """
    if n1 < 2:
        raise ValueError("n1 must be >= 2")
    if n2 < 2:
        raise ValueError("n2 must be >= 2")
    if sigma1 <= 0 or sigma2 <= 0:
        raise ValueError("sigma1 and sigma2 must be positive")
    if distribution1 not in _SUPPORTED_DISTS:
        raise ValueError(f"distribution1 {distribution1!r} not in {_SUPPORTED_DISTS}")
    if distribution2 not in _SUPPORTED_DISTS:
        raise ValueError(f"distribution2 {distribution2!r} not in {_SUPPORTED_DISTS}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if n_sims < 100:
        raise ValueError("n_sims must be >= 100")

    rng = np.random.default_rng(seed)
    s1 = _sample_group(rng, n=n1, mean=mu1, sigma=sigma1,
                       dist=distribution1, n_sims=n_sims)
    s2 = _sample_group(rng, n=n2, mean=mu2, sigma=sigma2,
                       dist=distribution2, n_sims=n_sims)

    pvals = _normal_scores_pvalues(s1, s2, _terry_scores, sides=sides)
    achieved = float(np.mean(pvals < alpha))

    inputs_echo = {
        "n1": n1, "n2": n2,
        "mu1": mu1, "mu2": mu2,
        "sigma1": sigma1, "sigma2": sigma2,
        "distribution1": distribution1, "distribution2": distribution2,
        "alpha": alpha, "sides": sides,
        "n_sims": n_sims, "seed": seed,
    }
    return {
        "method_id": "terry_hoeffding_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "stochastic": True,
        "n_sims": n_sims,
        "seed": seed,
        "inputs_echo": inputs_echo,
        "citations": [
            "Terry, M. E. (1952). Some rank order tests which are most powerful "
            "against specific parametric alternatives. Annals of Mathematical "
            "Statistics, 23, 346-366.",
            "Hoeffding, W. (1951). Optimum nonparametric tests. Proceedings of the "
            "Second Berkeley Symposium on Mathematical Statistics and Probability, 83-92.",
            "Blom, G. (1958). Statistical Estimates and Transformed Beta-Variables. Wiley.",
        ],
        "notes": (
            "Monte-Carlo estimate; achieved_power is stochastic.  With "
            f"n_sims={n_sims}, the 95% binomial CI half-width is ~"
            f"{1.96 * math.sqrt(max(achieved, 1e-9) * (1 - min(achieved, 1 - 1e-9)) / n_sims):.4f}."
        ),
    }


def van_der_waerden_simulation(
    *,
    n1: int,
    n2: int,
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    distribution1: str = "normal",
    distribution2: str = "normal",
    alpha: float = 0.05,
    sides: int = 2,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte-Carlo simulation power for the Van der Waerden X normal-scores test.

    The Van der Waerden procedure replaces ranks with normal quantile scores
    Phi^{-1}(i/(N+1)) and applies a two-sample t-test on those scores.

    Parameters
    ----------
    n1
        Sample size in group 1 (>= 2).
    n2
        Sample size in group 2 (>= 2).
    mu1
        Mean (location) of group 1 under H1.
    mu2
        Mean (location) of group 2 under H1.
    sigma1
        Scale (SD) of group 1.
    sigma2
        Scale (SD) of group 2.
    distribution1
        Distribution family for group 1.  One of: 'normal', 'lognormal',
        'logistic', 'cauchy', 'uniform', 'exponential'.
    distribution2
        Distribution family for group 2.
    alpha
        Significance level (default 0.05).
    sides
        1 for one-sided, 2 for two-sided (default).
    n_sims
        Monte-Carlo replicates (default 10,000).
    seed
        RNG seed (default 42).

    Returns
    -------
    dict
        Standard envelope: method_id, solve_for, n, achieved_power,
        inputs_echo, citations.
    """
    if n1 < 2:
        raise ValueError("n1 must be >= 2")
    if n2 < 2:
        raise ValueError("n2 must be >= 2")
    if sigma1 <= 0 or sigma2 <= 0:
        raise ValueError("sigma1 and sigma2 must be positive")
    if distribution1 not in _SUPPORTED_DISTS:
        raise ValueError(f"distribution1 {distribution1!r} not in {_SUPPORTED_DISTS}")
    if distribution2 not in _SUPPORTED_DISTS:
        raise ValueError(f"distribution2 {distribution2!r} not in {_SUPPORTED_DISTS}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if n_sims < 100:
        raise ValueError("n_sims must be >= 100")

    rng = np.random.default_rng(seed)
    s1 = _sample_group(rng, n=n1, mean=mu1, sigma=sigma1,
                       dist=distribution1, n_sims=n_sims)
    s2 = _sample_group(rng, n=n2, mean=mu2, sigma=sigma2,
                       dist=distribution2, n_sims=n_sims)

    pvals = _normal_scores_pvalues(s1, s2, _van_der_waerden_scores, sides=sides)
    achieved = float(np.mean(pvals < alpha))

    inputs_echo = {
        "n1": n1, "n2": n2,
        "mu1": mu1, "mu2": mu2,
        "sigma1": sigma1, "sigma2": sigma2,
        "distribution1": distribution1, "distribution2": distribution2,
        "alpha": alpha, "sides": sides,
        "n_sims": n_sims, "seed": seed,
    }
    return {
        "method_id": "van_der_waerden_simulation",
        "solve_for": "power",
        "n": n1 + n2,
        "n1": n1,
        "n2": n2,
        "achieved_power": achieved,
        "stochastic": True,
        "n_sims": n_sims,
        "seed": seed,
        "inputs_echo": inputs_echo,
        "citations": [
            "Van der Waerden, B. L. (1953). Ein neuer Test fuer das Problem der "
            "zwei Stichproben. Mathematische Annalen, 126, 93-107.",
            "Conover, W. J. (1999). Practical Nonparametric Statistics, 3rd ed. Wiley.",
        ],
        "notes": (
            "Monte-Carlo estimate; achieved_power is stochastic.  With "
            f"n_sims={n_sims}, the 95% binomial CI half-width is ~"
            f"{1.96 * math.sqrt(max(achieved, 1e-9) * (1 - min(achieved, 1 - 1e-9)) / n_sims):.4f}."
        ),
    }
