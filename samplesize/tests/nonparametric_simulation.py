"""Monte-Carlo simulation-based nonparametric two-sample tests.

Implements two nonparametric simulation-based power procedures:

* Chapter 430 — "Mann-Whitney-Wilcoxon Tests (Simulation)"
  -> :func:`mann_whitney_wilcoxon_simulation`
* Chapter 556 — "Kruskal-Wallis Tests (Simulation)"
  -> :func:`kruskal_wallis_simulation`

For each scenario the function generates ``n_sims`` datasets from the user-
specified per-group distributions (under H1), runs the test on each
replicate, and reports the proportion of rejections as the simulated power.

The Mann-Whitney-Wilcoxon test uses the normal approximation with
continuity correction and tie correction to the standard deviation, as
described in Gibbons (1985).

The Kruskal-Wallis test uses the chi-square approximation with tie
correction as described in Conover (1999).

Distributional choices include the subset that is mathematically
meaningful for rank-based tests: ``normal``, ``lognormal``, ``logistic``,
``cauchy``, ``uniform``, ``exponential``.  All variates except Cauchy (which
has no finite mean/variance) are shifted so the theoretical location equals
the requested mean parameter.  Lognormal, logistic, and cauchy are
parameterised so that the within-group SD matches ``sigma`` where possible.

Results are stochastic: with the default ``n_sims=10000`` the binomial
95% CI half-width is ~±0.010 at power=0.5 and ~±0.004 at power=0.95.
Fixtures therefore use a wider tolerance (~±0.015 absolute on power) than
analytic methods.  ``numpy.random`` is seeded (default ``seed=42``) so
results are deterministic under a fixed test harness.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Supported distributions
# ---------------------------------------------------------------------------

_SUPPORTED_DISTS = (
    "normal", "lognormal", "logistic", "cauchy", "uniform", "exponential",
)


def _sample_group_np(
    rng: np.random.Generator,
    *,
    n: int,
    mean: float,
    sigma: float,
    dist: str,
    n_sims: int,
) -> np.ndarray:
    """Return an ``(n_sims, n)`` array of replicates for one group.

    Each row is one simulated dataset of ``n`` observations drawn from the
    requested distribution centred at ``mean`` with scale ``sigma``.

    - ``normal``:      N(mean, sigma²)
    - ``lognormal``:   lognormal with median=exp(mu_ln), rescaled to have
                       SD ≈ sigma around location ``mean``.  Uses the
                       meanlog/sdlog parameterisation: the underlying
                       log-normal has log-scale sigma_ln chosen so that
                       its theoretical SD equals ``sigma``, then shifted
                       so E[X] = mean.
    - ``logistic``:    Logistic(mean, s) where s = sigma * sqrt(3) / pi,
                       so Var = sigma².
    - ``cauchy``:      Cauchy(mean, sigma).  Has no finite mean/variance;
                       sigma is the scale (half-width at half-maximum).
                       No location shift beyond ``mean`` is needed.
    - ``uniform``:     Uniform shifted so E[X]=mean and Var=sigma².
                       a = mean - sqrt(3)*sigma, b = mean + sqrt(3)*sigma.
    - ``exponential``: Shifted exponential with mean=1/rate shifted so
                       E[X]=mean; sigma controls the scale (SD = sigma).
    """
    if dist == "normal":
        return mean + sigma * rng.standard_normal(size=(n_sims, n))

    if dist == "lognormal":
        # Lognormal: if Y~LN(mu_ln, sigma_ln), then
        # E[Y] = exp(mu_ln + sigma_ln²/2), Var[Y] = (exp(sigma_ln²)-1)*exp(2*mu_ln+sigma_ln²)
        # We want E[X]=mean and SD[X]=sigma; solve for sigma_ln and mu_ln.
        # sigma_ln² = log(1 + (sigma/mean)²)  -- only valid if mean > 0
        # mu_ln = log(mean) - sigma_ln²/2
        # If mean <= 0, fall back to shifting: sample LN(0, 1) scaled and shifted.
        if mean > 0:
            sigma_ln2 = math.log(1.0 + (sigma / mean) ** 2)
            sigma_ln = math.sqrt(sigma_ln2)
            mu_ln = math.log(mean) - sigma_ln2 / 2.0
            return rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=(n_sims, n))
        else:
            # mean ≤ 0: sample standard lognormal (mu=0,sigma=1) and shift
            raw = rng.lognormal(mean=0.0, sigma=1.0, size=(n_sims, n))
            raw_mean = math.exp(0.5)  # E[LN(0,1)]
            raw_sd = math.sqrt((math.exp(1.0) - 1.0) * math.exp(1.0))
            return (raw - raw_mean) / raw_sd * sigma + mean

    if dist == "logistic":
        # Logistic(loc, scale): Var = pi²*scale²/3  => scale = sigma*sqrt(3)/pi
        scale = sigma * math.sqrt(3.0) / math.pi
        return rng.logistic(loc=mean, scale=scale, size=(n_sims, n))

    if dist == "cauchy":
        # Cauchy has no finite moments; sigma is the scale parameter directly
        return stats.cauchy.rvs(
            loc=mean, scale=sigma, size=(n_sims, n), random_state=rng
        )

    if dist == "uniform":
        # Uniform[a, b]: E = (a+b)/2 = mean, Var = (b-a)²/12 = sigma²
        half = math.sqrt(3.0) * sigma
        return rng.uniform(mean - half, mean + half, size=(n_sims, n))

    if dist == "exponential":
        # Shifted Exp: E[X] = 1/rate = sigma (so rate = 1/sigma)
        # then shift by (mean - sigma) so E[shifted X] = mean
        raw = rng.exponential(scale=sigma, size=(n_sims, n))
        return raw + (mean - sigma)

    raise ValueError(
        f"unsupported distribution {dist!r}; choose one of {_SUPPORTED_DISTS}"
    )


# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------

def _mww_pvalues(
    s1: np.ndarray,
    s2: np.ndarray,
    *,
    sides: int,
) -> np.ndarray:
    """Vectorised Mann-Whitney-Wilcoxon p-values (normal approximation).

    ``s1`` shape ``(n_sims, n1)``, ``s2`` shape ``(n_sims, n2)``.

    Uses the Gibbons (1985) formulation with continuity correction and
    tie-corrected standard deviation (Gibbons 1985).
    """
    n_sims, n1 = s1.shape
    n2 = s2.shape[1]
    N = n1 + n2

    pooled = np.concatenate([s1, s2], axis=1)               # (n_sims, N)
    ranks = stats.rankdata(pooled, method="average", axis=1) # (n_sims, N)

    W1 = ranks[:, :n1].sum(axis=1)  # sum of ranks for group 1

    # Expected W1 under H0
    mu_W = n1 * (N + 1) / 2.0

    # Tie-corrected variance
    sorted_r = np.sort(ranks, axis=1)
    diff_r = np.diff(sorted_r, axis=1) == 0  # consecutive equal ranks
    T = np.zeros(n_sims)
    for i in range(n_sims):
        run = 1
        for j in range(N - 1):
            if diff_r[i, j]:
                run += 1
            else:
                if run > 1:
                    T[i] += run ** 3 - run
                run = 1
        if run > 1:
            T[i] += run ** 3 - run

    var_W = (n1 * n2 / 12.0) * ((N + 1) - T / (N * (N - 1)))
    var_W = np.where(var_W <= 0, np.finfo(float).tiny, var_W)
    s_W = np.sqrt(var_W)

    numerator = W1 - mu_W
    # continuity correction: move numerator toward zero by 0.5
    C = np.where(numerator < 0, 0.5, -0.5)
    z = (numerator + C) / s_W

    if sides == 2:
        pval = 2.0 * stats.norm.sf(np.abs(z))
    else:
        # one-sided: H1: group1 > group2, i.e. W1 is large
        pval = stats.norm.sf(z)

    return pval


def _kw_pvalues(samples: list[np.ndarray]) -> np.ndarray:
    """Vectorised Kruskal-Wallis H-test p-values.

    Each ``samples[g]`` is shape ``(n_sims, n_g)``.  Implements
    the tie-corrected H statistic, distributed as chi²(g-1) under H0.
    See Conover (1999).
    """
    k = len(samples)
    n_sims = samples[0].shape[0]
    ni = np.array([s.shape[1] for s in samples])
    N = int(ni.sum())

    pooled = np.concatenate(samples, axis=1)                      # (n_sims, N)
    ranks = stats.rankdata(pooled, method="average", axis=1)      # (n_sims, N)

    starts = np.cumsum(np.concatenate(([0], ni)))
    H = np.zeros(n_sims)
    for g in range(k):
        Rg = ranks[:, starts[g]:starts[g + 1]].sum(axis=1)
        H += Rg ** 2 / ni[g]
    H = (12.0 / (N * (N + 1))) * H - 3.0 * (N + 1)

    # Tie correction
    sorted_r = np.sort(ranks, axis=1)
    diff_r = np.diff(sorted_r, axis=1) == 0
    T_vec = np.zeros(n_sims)
    for i in range(n_sims):
        run = 1
        for j in range(N - 1):
            if diff_r[i, j]:
                run += 1
            else:
                if run > 1:
                    T_vec[i] += run ** 3 - run
                run = 1
        if run > 1:
            T_vec[i] += run ** 3 - run
    C = 1.0 - T_vec / (N ** 3 - N)
    C = np.where(C <= 0, 1.0, C)
    H = H / C
    H = np.where(H < 0, 0.0, H)

    return 1.0 - stats.chi2.cdf(H, df=k - 1)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_mww_inputs(
    *,
    n1: int,
    n2: int,
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    distribution1: str,
    distribution2: str,
    alpha: float,
    sides: int,
    n_sims: int,
) -> None:
    if n1 < 2:
        raise ValueError("n1 must be >= 2")
    if n2 < 2:
        raise ValueError("n2 must be >= 2")
    if sigma1 <= 0:
        raise ValueError("sigma1 must be positive")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    if distribution1 not in _SUPPORTED_DISTS:
        raise ValueError(
            f"distribution1 {distribution1!r} not in {_SUPPORTED_DISTS}"
        )
    if distribution2 not in _SUPPORTED_DISTS:
        raise ValueError(
            f"distribution2 {distribution2!r} not in {_SUPPORTED_DISTS}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if n_sims < 100:
        raise ValueError("n_sims must be >= 100")


def _validate_kw_inputs(
    *,
    n_per_group: int | Sequence[int],
    group_means: Sequence[float],
    group_sds: float | Sequence[float],
    distribution: str,
    alpha: float,
    n_sims: int,
) -> tuple[list[int], list[float], list[float]]:
    means_l = list(group_means)
    k = len(means_l)
    if k < 2:
        raise ValueError("need at least 2 groups")

    if isinstance(n_per_group, int):
        ni_l = [n_per_group] * k
    else:
        ni_l = list(n_per_group)
        if len(ni_l) != k:
            raise ValueError("len(n_per_group) must equal len(group_means)")
    if any(n < 2 for n in ni_l):
        raise ValueError("each group n must be >= 2")

    if isinstance(group_sds, (int, float)):
        sds_l = [float(group_sds)] * k
    else:
        sds_l = list(group_sds)
        if len(sds_l) != k:
            raise ValueError("len(group_sds) must equal len(group_means)")
    if any(s <= 0 for s in sds_l):
        raise ValueError("all group SDs must be positive")

    if distribution not in _SUPPORTED_DISTS:
        raise ValueError(
            f"distribution {distribution!r} not in {_SUPPORTED_DISTS}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_sims < 100:
        raise ValueError("n_sims must be >= 100")

    return ni_l, means_l, sds_l


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------

def mann_whitney_wilcoxon_simulation(
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
    """Monte-Carlo simulation power for the Mann-Whitney-Wilcoxon test.

    Generates ``n_sims`` paired datasets of sizes ``n1`` and ``n2`` from
    the specified H1 distributions, computes the MWW normal-approximation
    test statistic with continuity and tie correction, and reports the
    proportion of rejections as the simulated power.

    Parameters
    ----------
    n1
        Sample size in group 1 (≥ 2).
    n2
        Sample size in group 2 (≥ 2).
    mu1
        Mean (location) of group 1 distribution under H1.
    mu2
        Mean (location) of group 2 distribution under H1.
    sigma1
        Scale (SD) of group 1 distribution.  For Cauchy this is the
        scale parameter (half-width at half-maximum), not the SD.
    sigma2
        Scale (SD) of group 2 distribution.
    distribution1
        Distribution family for group 1.  One of: ``'normal'``,
        ``'lognormal'``, ``'logistic'``, ``'cauchy'``, ``'uniform'``,
        ``'exponential'``.
    distribution2
        Distribution family for group 2 (same choices).
    alpha
        Target significance level (default 0.05).
    sides
        1 for one-sided (H1: mu1 > mu2), 2 for two-sided (default).
    n_sims
        Number of Monte-Carlo replicates (default 10,000).
    seed
        Seed for ``numpy.random.default_rng`` (default 42).

    Returns
    -------
    dict
        Standard envelope with ``method_id``, ``solve_for``, ``n``,
        ``achieved_power``, ``inputs_echo``, ``citations``.
    """
    _validate_mww_inputs(
        n1=n1, n2=n2, mu1=mu1, mu2=mu2,
        sigma1=sigma1, sigma2=sigma2,
        distribution1=distribution1, distribution2=distribution2,
        alpha=alpha, sides=sides, n_sims=n_sims,
    )

    rng = np.random.default_rng(seed)
    s1 = _sample_group_np(rng, n=n1, mean=mu1, sigma=sigma1,
                          dist=distribution1, n_sims=n_sims)
    s2 = _sample_group_np(rng, n=n2, mean=mu2, sigma=sigma2,
                          dist=distribution2, n_sims=n_sims)

    pvals = _mww_pvalues(s1, s2, sides=sides)
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
        "method_id": "mann_whitney_wilcoxon_simulation",
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
            "Gibbons, J. D. (1985). Nonparametric Methods for Quantitative Analysis. "
            "2nd ed. American Sciences Press.",
        ],
        "notes": (
            "Monte-Carlo estimate; achieved_power is stochastic.  With "
            f"n_sims={n_sims}, the 95% binomial CI half-width is ~"
            f"{1.96 * math.sqrt(max(achieved, 1e-9) * (1 - min(achieved, 1 - 1e-9)) / n_sims):.4f}."
        ),
    }


def kruskal_wallis_simulation(
    *,
    n_per_group: int | Sequence[int],
    group_means: Sequence[float],
    group_sds: float | Sequence[float],
    distribution: str = "normal",
    alpha: float = 0.05,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte-Carlo simulation power for the Kruskal-Wallis H-test.

    Generates ``n_sims`` multi-group datasets from the specified H1
    distributions (all groups share the same family; means and SDs vary),
    computes the tie-corrected KW H statistic, and reports the proportion
    of rejections as the simulated power.

    Parameters
    ----------
    n_per_group
        Per-group sample size (scalar for equal allocation, or list of
        length k for unequal allocation; each ≥ 2).
    group_means
        Means (locations) under H1, one per group (k ≥ 2).
    group_sds
        Within-group SDs, scalar (common) or list of length k.
    distribution
        Distribution family shared by all groups.  One of: ``'normal'``,
        ``'lognormal'``, ``'logistic'``, ``'cauchy'``, ``'uniform'``,
        ``'exponential'``.
    alpha
        Target significance level (default 0.05).
    n_sims
        Number of Monte-Carlo replicates (default 10,000).
    seed
        Seed for ``numpy.random.default_rng`` (default 42).

    Returns
    -------
    dict
        Standard envelope with ``method_id``, ``solve_for``, ``n``,
        ``n_per_group``, ``achieved_power``, ``inputs_echo``, ``citations``.
    """
    ni_l, means_l, sds_l = _validate_kw_inputs(
        n_per_group=n_per_group,
        group_means=group_means,
        group_sds=group_sds,
        distribution=distribution,
        alpha=alpha,
        n_sims=n_sims,
    )

    rng = np.random.default_rng(seed)
    samples = [
        _sample_group_np(rng, n=ni_l[g], mean=means_l[g], sigma=sds_l[g],
                         dist=distribution, n_sims=n_sims)
        for g in range(len(means_l))
    ]

    pvals = _kw_pvalues(samples)
    achieved = float(np.mean(pvals < alpha))
    n_total = sum(ni_l)

    inputs_echo = {
        "n_per_group": ni_l,
        "group_means": means_l,
        "group_sds": sds_l,
        "distribution": distribution,
        "alpha": alpha,
        "n_sims": n_sims,
        "seed": seed,
    }
    return {
        "method_id": "kruskal_wallis_simulation",
        "solve_for": "power",
        "n": n_total,
        "n_per_group": ni_l,
        "achieved_power": achieved,
        "stochastic": True,
        "n_sims": n_sims,
        "seed": seed,
        "inputs_echo": inputs_echo,
        "citations": [
            "Conover, W. J. (1999). Practical Nonparametric Statistics. 3rd ed. Wiley.",
        ],
        "notes": (
            "Monte-Carlo estimate; achieved_power is stochastic.  With "
            f"n_sims={n_sims}, the 95% binomial CI half-width is ~"
            f"{1.96 * math.sqrt(max(achieved, 1e-9) * (1 - min(achieved, 1 - 1e-9)) / n_sims):.4f}."
        ),
    }
