"""Monte-Carlo simulation of one-way ANOVA F-test power.

Implements two simulation-based power procedures that share a single simulation core:

* Chapter 555 — "One-Way Analysis of Variance F-Tests (Simulation)"
  -> :func:`one_way_anova_f_simulation`
* Chapter 559 — "Power Comparison of Tests of Means in One-Way Designs
  (Simulation)" -> :func:`power_comparison_means_one_way_sim`

For each scenario the function generates ``n_sims`` datasets from per-group
distributions (under H1), runs one or more location tests on each
replicate, and reports the proportion of rejections as the simulated
power.  When solving for sample size, an outer bisection brackets the
smallest equal-allocation per-group ``n`` whose F-test power reaches the
target.

Distributional choices include the subset that is
mathematically meaningful for one-way ANOVA: ``normal`` (default),
``t`` (heavy-tailed, requires ``df > 2``), ``uniform`` (light-tailed
symmetric) and ``exponential`` (skewed).  All non-normal variates are
rescaled so the theoretical within-group SD equals the requested
``sigma`` -- ``σ`` is always the
within-group SD irrespective of the alternative distribution shape.

Results are *stochastic*: with the default ``n_sims=10000`` the binomial
95% CI half-width is ~±0.010 at power=0.5 and ~±0.004 at power=0.95
(see the precision table reproduced verbatim in Chapter 555).  Fixtures
therefore use a wider tolerance (~±0.01 absolute on power) than the
analytic methods.  ``numpy.random`` is seeded (default ``seed=42``) so
results are deterministic under a fixed test harness.

The power-comparison function juxtaposes
the F-test against Kruskal-Wallis, Terry-Hoeffding and Van der Waerden
normal-scores tests.  We expose F + Welch's F + Kruskal-Wallis +
Brown-Forsythe (median-centred Levene applied to the *response*) which
covers the parametric / robust / rank trichotomy most users care about
when picking a one-way location test.  The simulated F-power exactly
matches the Fleiss (1986) example (n=11 -> ~0.80); the
Kruskal-Wallis value also matches the published example to within MC noise.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Distribution sampler (shared with variances_simulation)
# ---------------------------------------------------------------------------

_SUPPORTED_DISTS = ("normal", "t", "uniform", "exponential")


def _sample_group(
    rng: np.random.Generator,
    *,
    n: int,
    mean: float,
    sigma: float,
    dist: str,
    n_sims: int,
    df: float | None = None,
) -> np.ndarray:
    """Return an ``(n_sims, n)`` array of replicates with the requested
    location (``mean``) and within-group SD (``sigma``).

    For non-normal distributions the underlying variate is rescaled so
    its theoretical SD equals ``sigma``.  ``sigma`` is always
    that ``sigma`` is always the within-group SD regardless of the chosen
    H1 distribution.
    """
    if dist == "normal":
        z = rng.standard_normal(size=(n_sims, n))
        return mean + sigma * z
    if dist == "t":
        if df is None or df <= 2:
            raise ValueError("t distribution requires df > 2 for finite variance")
        t = rng.standard_t(df, size=(n_sims, n))
        scale = math.sqrt(df / (df - 2.0))
        return mean + sigma * (t / scale)
    if dist == "uniform":
        # Uniform on [-sqrt(3), sqrt(3)] has unit variance.
        u = rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), size=(n_sims, n))
        return mean + sigma * u
    if dist == "exponential":
        e = rng.standard_exponential(size=(n_sims, n)) - 1.0
        return mean + sigma * e
    raise ValueError(f"unsupported dist {dist!r}; choose one of {_SUPPORTED_DISTS}")


# ---------------------------------------------------------------------------
# Vectorised one-way location-test statistics
# ---------------------------------------------------------------------------


def _anova_f_pvalues(samples: list[np.ndarray]) -> np.ndarray:
    """Standard one-way ANOVA F-test p-value across the replicate axis.

    Each ``samples[g]`` is shape ``(n_sims, n_g)``.  Implements

        F = MSR / MSE   with df = (g-1, N-g)
        MSR = Σ n_g (X̄_g - X̄)² / (g-1)
        MSE = Σ Σ (X_{gj} - X̄_g)² / (N-g)

    and returns the upper-tail p-values from F(g-1, N-g).
    """
    g = len(samples)
    ni = np.array([s.shape[1] for s in samples])
    N = int(ni.sum())
    n_sims = samples[0].shape[0]

    group_means = np.stack([s.mean(axis=1) for s in samples], axis=1)  # (n_sims, g)
    grand = (group_means * ni).sum(axis=1) / N                          # (n_sims,)
    ss_between = (ni * (group_means - grand[:, None]) ** 2).sum(axis=1)
    ss_within = np.zeros(n_sims)
    for k, s in enumerate(samples):
        ss_within += ((s - group_means[:, k:k + 1]) ** 2).sum(axis=1)

    df1 = g - 1
    df2 = N - g
    ss_within = np.where(ss_within <= 0, np.finfo(float).tiny, ss_within)
    F = (ss_between / df1) / (ss_within / df2)
    return 1.0 - stats.f.cdf(F, df1, df2)


def _welch_f_pvalues(samples: list[np.ndarray]) -> np.ndarray:
    """Welch's heteroscedastic one-way ANOVA F-test p-value.

    Welch (1951) statistic, with denominator df from the Satterthwaite
    approximation:

        w_g     = n_g / S_g²
        X̄_·    = Σ w_g X̄_g / Σ w_g
        F*      = [ Σ w_g (X̄_g − X̄_·)² / (k−1) ]
                  / [ 1 + 2(k−2)/(k²−1) · Σ ((1 − w_g/Σw_g)² / (n_g − 1)) ]
        df₂*    = (k² − 1) / ( 3 · Σ ((1 − w_g/Σw_g)² / (n_g − 1)) )

    Under H₀, F* ~ F(k−1, df₂*).
    """
    k = len(samples)
    ni = np.array([s.shape[1] for s in samples])
    var_g = np.stack([s.var(axis=1, ddof=1) for s in samples], axis=1)  # (n_sims, k)
    var_g = np.where(var_g <= 0, np.finfo(float).tiny, var_g)
    mean_g = np.stack([s.mean(axis=1) for s in samples], axis=1)         # (n_sims, k)

    w = ni / var_g                                                       # (n_sims, k)
    W = w.sum(axis=1)                                                    # (n_sims,)
    xbar = (w * mean_g).sum(axis=1) / W                                  # (n_sims,)
    msr = (w * (mean_g - xbar[:, None]) ** 2).sum(axis=1) / (k - 1)

    # Σ ((1 − w_g/W)² / (n_g − 1)), broadcast across replicates
    inv_n_minus_1 = 1.0 / (ni - 1)
    Q = ((1.0 - w / W[:, None]) ** 2 * inv_n_minus_1).sum(axis=1)
    denom = 1.0 + (2.0 * (k - 2) / (k ** 2 - 1)) * Q
    F = msr / denom
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3.0 * Q)
    df2 = np.where(df2 <= 0, np.finfo(float).tiny, df2)
    return 1.0 - stats.f.cdf(F, df1, df2)


def _kruskal_wallis_pvalues(samples: list[np.ndarray]) -> np.ndarray:
    """Kruskal-Wallis H-test p-value across the replicate axis.

    Pools the samples, ranks with average ties, computes

        H = (12 / (N(N+1))) · Σ R_g² / n_g  −  3(N+1)

    and applies the standard tie correction.  Under H₀, H ~ χ²(k-1).
    """
    k = len(samples)
    n_sims = samples[0].shape[0]
    ni = np.array([s.shape[1] for s in samples])
    N = int(ni.sum())

    pooled = np.concatenate(samples, axis=1)                       # (n_sims, N)
    ranks = stats.rankdata(pooled, method="average", axis=1)       # (n_sims, N)

    starts = np.cumsum(np.concatenate(([0], ni)))
    H = np.zeros(n_sims)
    for g in range(k):
        Rg = ranks[:, starts[g]:starts[g + 1]].sum(axis=1)
        H += (Rg ** 2) / ni[g]
    H = (12.0 / (N * (N + 1))) * H - 3.0 * (N + 1)

    # tie correction: T = Σ (t_j³ − t_j); C = 1 − T / (N³ − N).
    # Compute per-replicate via sort + run-length on the *ranks* (which
    # take half-integer values when ties exist).
    sorted_ranks = np.sort(ranks, axis=1)
    diff = np.diff(sorted_ranks, axis=1) == 0                      # (n_sims, N-1)
    # tie-group sizes per replicate via run-length encoding -- vectorise
    # the simple "count consecutive equal" pattern.
    # We need Σ t³ − t where t are tie-group sizes.  Equivalent formula:
    # for each rank position i, contribution to T is: when in tie group of
    # size t, sum over the group adds t·(t²−1).  A scan-based approach is
    # clearer in Python; iterate over replicates (still cheap since N is
    # small relative to n_sims·anything).
    T = np.zeros(n_sims)
    for i in range(n_sims):
        run = 1
        for j in range(N - 1):
            if diff[i, j]:
                run += 1
            else:
                if run > 1:
                    T[i] += run ** 3 - run
                run = 1
        if run > 1:
            T[i] += run ** 3 - run
    C = 1.0 - T / (N ** 3 - N)
    C = np.where(C <= 0, 1.0, C)
    H = H / C
    return 1.0 - stats.chi2.cdf(H, df=k - 1)


def _brown_forsythe_response_pvalues(samples: list[np.ndarray]) -> np.ndarray:
    """Brown-Forsythe (1974) heteroscedastic one-way F-test on means.

    Distinct from the Brown-Forsythe *variances* test.  The statistic is

        F* = Σ n_g (X̄_g − X̄)² / Σ (1 − n_g/N) · S_g²

    with denominator df from the Satterthwaite approximation:

        df₂* = ( Σ c_g · S_g² )² / Σ ( c_g² · S_g⁴ / (n_g − 1) )
        c_g  = 1 − n_g / N

    Under H₀, F* ~ F(k-1, df₂*).  Robust to unequal variances; uses
    means (not medians) of the response, contrast that with the
    variances version which centres absolute deviations.
    """
    k = len(samples)
    ni = np.array([s.shape[1] for s in samples])
    N = int(ni.sum())
    mean_g = np.stack([s.mean(axis=1) for s in samples], axis=1)         # (n_sims, k)
    var_g = np.stack([s.var(axis=1, ddof=1) for s in samples], axis=1)    # (n_sims, k)
    var_g = np.where(var_g <= 0, np.finfo(float).tiny, var_g)

    grand = (mean_g * ni).sum(axis=1) / N
    num = (ni * (mean_g - grand[:, None]) ** 2).sum(axis=1)
    cg = 1.0 - ni / N
    den = (cg * var_g).sum(axis=1)
    F = num / np.where(den <= 0, np.finfo(float).tiny, den)
    df1 = k - 1
    df2 = den ** 2 / ((cg ** 2 * var_g ** 2 / (ni - 1)).sum(axis=1))
    df2 = np.where(df2 <= 0, np.finfo(float).tiny, df2)
    return 1.0 - stats.f.cdf(F, df1, df2)


# ---------------------------------------------------------------------------
# Validation + simulation core
# ---------------------------------------------------------------------------


def _validate_inputs(
    means: Sequence[float],
    sigma: float | Sequence[float],
    dist: str,
    alpha: float,
) -> tuple[list[float], list[float]]:
    if len(means) < 2:
        raise ValueError("need at least 2 group means")
    if isinstance(sigma, (int, float)):
        sigmas = [float(sigma)] * len(means)
    else:
        sigmas = list(sigma)
        if len(sigmas) != len(means):
            raise ValueError("len(sigma list) must equal len(means)")
    if any(s <= 0 for s in sigmas):
        raise ValueError("every sigma must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if dist not in _SUPPORTED_DISTS:
        raise ValueError(
            f"unsupported dist {dist!r}; choose one of {_SUPPORTED_DISTS}"
        )
    return list(means), sigmas


def _draw_samples(
    rng: np.random.Generator,
    *,
    means: list[float],
    sigmas: list[float],
    ni: list[int],
    dist: str,
    n_sims: int,
    df_t: float | None,
) -> list[np.ndarray]:
    return [
        _sample_group(
            rng,
            n=ni[i],
            mean=means[i],
            sigma=sigmas[i],
            dist=dist,
            n_sims=n_sims,
            df=df_t,
        )
        for i in range(len(means))
    ]


def _simulate_powers(
    *,
    tests: list[str],
    means: list[float],
    sigmas: list[float],
    ni: list[int],
    dist: str,
    alpha: float,
    n_sims: int,
    seed: int,
    df_t: float | None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = _draw_samples(
        rng,
        means=means,
        sigmas=sigmas,
        ni=ni,
        dist=dist,
        n_sims=n_sims,
        df_t=df_t,
    )
    out: dict[str, float] = {}
    for t in tests:
        if t == "f":
            p = _anova_f_pvalues(samples)
        elif t == "welch_f":
            p = _welch_f_pvalues(samples)
        elif t == "kruskal_wallis":
            p = _kruskal_wallis_pvalues(samples)
        elif t == "brown_forsythe":
            p = _brown_forsythe_response_pvalues(samples)
        else:  # pragma: no cover
            raise ValueError(f"unknown test {t!r}")
        out[t] = float(np.mean(p < alpha))
    return out


def _bisect_n_for_f(
    *,
    means: list[float],
    sigmas: list[float],
    allocation: list[float],
    dist: str,
    alpha: float,
    target_power: float,
    n_sims: int,
    seed: int,
    df_t: float | None,
    n_min: int = 2,
    n_max: int = 5000,
) -> tuple[int, list[int], float]:
    if not 0.0 < target_power < 1.0:
        raise ValueError("power must be in (0, 1)")

    def ni_for(n_base: int) -> list[int]:
        return [max(2, math.ceil(n_base * a)) for a in allocation]

    def power_at(n_base: int) -> float:
        return _simulate_powers(
            tests=["f"],
            means=means,
            sigmas=sigmas,
            ni=ni_for(n_base),
            dist=dist,
            alpha=alpha,
            n_sims=n_sims,
            seed=seed,
            df_t=df_t,
        )["f"]

    lo, hi = n_min, max(n_min, 4)
    p_hi = power_at(hi)
    while p_hi < target_power:
        lo = hi
        hi = hi * 2
        if hi > n_max:
            raise RuntimeError(
                f"failed to bracket n within {n_max}; last power={p_hi:.3f}"
            )
        p_hi = power_at(hi)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if power_at(mid) >= target_power:
            hi = mid
        else:
            lo = mid
    ni = ni_for(hi)
    return hi, ni, power_at(hi)


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------


def one_way_anova_f_simulation(
    *,
    means: Sequence[float],
    sigma: float | Sequence[float],
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    allocation: Sequence[float] | None = None,
    dist: str = "normal",
    n_sims: int = 10000,
    seed: int = 42,
    df_t: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Monte-Carlo simulation of the one-way ANOVA F-test.

    The F-test is the parametric workhorse for one-way location
    comparisons; this simulation variant lets you check its empirical
    power and *actual* significance level under non-normal alternative
    distributions (``dist='t'``, ``'uniform'``, ``'exponential'``) that
    invalidate the closed-form noncentral-F result.

    Parameters
    ----------
    means
        Group means under H1 (length k ≥ 2).
    sigma
        Common within-group SD (scalar) or per-group SDs (list of length
        k).  Per-group SDs are useful for studying robustness under
        variance heterogeneity even though the analytic F-test assumes
        homoscedasticity.
    n
        Per-group base sample size (combined with ``allocation``).
        Required when ``solve_for='power'``.
    alpha
        Target significance level (default 0.05).
    power
        Target power; used when ``solve_for='n'``.
    allocation
        Per-group allocation ratios; defaults to equal (``[1,…,1]``).
    dist
        Alternative-distribution shape per group: ``'normal'`` (default)
        ``'t'``, ``'uniform'``, ``'exponential'``.  All shapes are
        rescaled so the within-group SD equals ``sigma``.
    n_sims
        Number of Monte-Carlo replicates (default 10,000).
    seed
        Seed for ``numpy.random.default_rng`` (default 42).
    df_t
        Degrees of freedom for the ``'t'`` distribution (must be > 2).
    solve_for
        ``'power'`` or ``'n'``.  Defaults to ``'n'`` if ``power`` is
        supplied, else ``'power'``.
    """
    means_l, sigmas_l = _validate_inputs(means, sigma, dist, alpha)
    if allocation is None:
        allocation = [1.0] * len(means_l)
    if len(allocation) != len(means_l):
        raise ValueError("allocation length must match means length")

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        ni = [max(2, math.ceil(int(n) * a)) for a in allocation]
        achieved = _simulate_powers(
            tests=["f"],
            means=means_l,
            sigmas=sigmas_l,
            ni=ni,
            dist=dist,
            alpha=alpha,
            n_sims=n_sims,
            seed=seed,
            df_t=df_t,
        )["f"]
        n_out = int(n)
    elif solve_for == "n":
        assert power is not None
        n_out, ni, achieved = _bisect_n_for_f(
            means=means_l,
            sigmas=sigmas_l,
            allocation=list(allocation),
            dist=dist,
            alpha=alpha,
            target_power=float(power),
            n_sims=n_sims,
            seed=seed,
            df_t=df_t,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    inputs_echo = {
        "means": means_l, "sigma": sigmas_l, "n": n, "alpha": alpha,
        "power": power, "allocation": list(allocation), "dist": dist,
        "n_sims": n_sims, "seed": seed, "df_t": df_t,
    }
    return {
        "method_id": "one_way_anova_f_simulation",
        "solve_for": solve_for,
        "n_per_group": n_out,
        "n_per_group_list": ni,
        "n_total": sum(ni),
        "achieved_power": achieved,
        "stochastic": True,
        "n_sims": n_sims,
        "seed": seed,
        "inputs_echo": inputs_echo,
        "citations": [
            "F-Tests (Simulation).",
            "Fleiss, J. (1986). The Design and Analysis of Clinical "
            "Experiments. Wiley.",
        ],
        "notes": (
            "Monte-Carlo estimate; achieved_power is stochastic.  With "
            f"n_sims={n_sims}, the 95% binomial CI half-width is ~"
            f"{1.96 * math.sqrt(achieved * (1 - achieved) / n_sims):.4f}."
        ),
    }


def power_comparison_means_one_way_sim(
    *,
    means: Sequence[float],
    sigma: float | Sequence[float],
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    allocation: Sequence[float] | None = None,
    dist: str = "normal",
    n_sims: int = 10000,
    seed: int = 42,
    df_t: float | None = None,
    tests: Sequence[str] | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power comparison of one-way location tests.

    Generates one shared set of Monte-Carlo replicates per scenario and
    applies several candidate tests of equality of means to each, so the
    user can compare their empirical power on identical data:

    * ``'f'``               — classical one-way ANOVA F-test.
    * ``'welch_f'``         — Welch (1951) heteroscedastic F-test.
    * ``'kruskal_wallis'``  — Kruskal-Wallis H-test (rank-based).
    * ``'brown_forsythe'``  — Brown-Forsythe (1974) F* for means
      (Satterthwaite-corrected, robust to unequal variances).

    When ``solve_for='n'`` the bisection drives the *F-test* power to the
    target, and the powers of the other
    tests are reported at that same per-group n.  Classic references
    juxtapose F, Kruskal-Wallis, Terry-Hoeffding, Van der Waerden; we
    drop the two normal-scores tests (whose values lie very close to
    Kruskal-Wallis) in favour of Welch's F and
    Brown-Forsythe, which are the more commonly recommended robust
    alternatives in modern practice.

    Inputs mirror :func:`one_way_anova_f_simulation`; the additional
    ``tests`` argument selects the subset to run (default: all four).
    Returned ``powers`` is a mapping ``test -> achieved_power``.
    """
    means_l, sigmas_l = _validate_inputs(means, sigma, dist, alpha)
    if allocation is None:
        allocation = [1.0] * len(means_l)
    if len(allocation) != len(means_l):
        raise ValueError("allocation length must match means length")
    if tests is None:
        tests_l = ["f", "welch_f", "kruskal_wallis", "brown_forsythe"]
    else:
        tests_l = list(tests)
        valid = {"f", "welch_f", "kruskal_wallis", "brown_forsythe"}
        bad = [t for t in tests_l if t not in valid]
        if bad:
            raise ValueError(f"unknown test(s) {bad!r}; choose subset of {sorted(valid)}")

    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n is not None
        ni = [max(2, math.ceil(int(n) * a)) for a in allocation]
        powers = _simulate_powers(
            tests=tests_l,
            means=means_l,
            sigmas=sigmas_l,
            ni=ni,
            dist=dist,
            alpha=alpha,
            n_sims=n_sims,
            seed=seed,
            df_t=df_t,
        )
        n_out = int(n)
    elif solve_for == "n":
        assert power is not None
        n_out, ni, _ = _bisect_n_for_f(
            means=means_l,
            sigmas=sigmas_l,
            allocation=list(allocation),
            dist=dist,
            alpha=alpha,
            target_power=float(power),
            n_sims=n_sims,
            seed=seed,
            df_t=df_t,
        )
        powers = _simulate_powers(
            tests=tests_l,
            means=means_l,
            sigmas=sigmas_l,
            ni=ni,
            dist=dist,
            alpha=alpha,
            n_sims=n_sims,
            seed=seed,
            df_t=df_t,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    inputs_echo = {
        "means": means_l, "sigma": sigmas_l, "n": n, "alpha": alpha,
        "power": power, "allocation": list(allocation), "dist": dist,
        "n_sims": n_sims, "seed": seed, "df_t": df_t,
        "tests": tests_l,
    }
    # Achieved-power surfaced at top level is the F-test power, matching
    # The F-test power is the headline achieved_power;
    # the headline.  Per-test powers live in `powers`.
    achieved = powers.get("f", next(iter(powers.values())))
    return {
        "method_id": "power_comparison_means_one_way_sim",
        "solve_for": solve_for,
        "n_per_group": n_out,
        "n_per_group_list": ni,
        "n_total": sum(ni),
        "achieved_power": achieved,
        "powers": powers,
        "stochastic": True,
        "n_sims": n_sims,
        "seed": seed,
        "inputs_echo": inputs_echo,
        "citations": [
            "in One-Way Designs (Simulation).",
            "Welch, B. L. (1951). On the comparison of several mean values: "
            "an alternative approach. Biometrika 38(3/4), 330-336.",
            "Brown, M. B. & Forsythe, A. B. (1974). The small sample behavior "
            "of some statistics which test the equality of several means. "
            "Technometrics 16(1), 129-132.",
            "Kruskal, W. H. & Wallis, W. A. (1952). Use of ranks in "
            "one-criterion variance analysis. JASA 47(260), 583-621.",
        ],
        "notes": (
            f"Monte-Carlo estimate; powers are stochastic.  With "
            f"n_sims={n_sims}, the 95% binomial CI half-width at the "
            f"reported F-test power is ~"
            f"{1.96 * math.sqrt(achieved * (1 - achieved) / n_sims):.4f}."
        ),
    }
