"""Monte-Carlo simulation-based homogeneity-of-variance tests.

Implements simulation-based power for variance homogeneity tests:

* "Levene Test of Variances (Simulation)"  (Chapter 553)
* "Bartlett Test of Variances (Simulation)" (Chapter 552)
* "Brown-Forsythe Test of Variances (Simulation)" (Chapter 554)
* "Conover Test of Variances (Simulation)" (Chapter 561)

For each scenario the function generates ``n_sims`` datasets from the user-
specified per-group distribution, runs the selected variance test on
each replicate, and reports the proportion of rejections as the
simulated power.  When solving for sample size, an outer bisection
brackets the smallest equal-allocation per-group ``n`` that achieves
the requested power.

Results are *stochastic*: with the default ``n_sims=10000`` the binomial
95% CI around a true power of 0.5 is ~+/-0.010 and ~+/-0.004 around
0.95.  Fixtures therefore
use a wider tolerance (~+/-0.01 absolute on power) than analytic
methods.  All simulations seed ``numpy.random`` (default ``seed=42``) for
deterministic reproducibility under a fixed test harness.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Distribution sampler
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
    """Return an (n_sims, n) array of replicates with the requested
    location (``mean``) and scale (``sigma``).

    For non-normal distributions the underlying variate is rescaled so
    that its theoretical SD equals ``sigma``.  ``sigma`` is always the within-group SD regardless
    of the chosen H1 distribution.
    """
    if dist == "normal":
        z = rng.standard_normal(size=(n_sims, n))
        return mean + sigma * z
    if dist == "t":
        if df is None or df <= 2:
            raise ValueError("t distribution requires df > 2 for finite variance")
        # standard t has variance df/(df-2); rescale to unit variance
        t = rng.standard_t(df, size=(n_sims, n))
        scale = math.sqrt(df / (df - 2.0))
        return mean + sigma * (t / scale)
    if dist == "uniform":
        # Uniform on [-sqrt(3), sqrt(3)] has variance 1.
        u = rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), size=(n_sims, n))
        return mean + sigma * u
    if dist == "exponential":
        # Exp(1) has mean 1, variance 1; centre it so the sample has the
        # requested mean while keeping variance = sigma^2.
        e = rng.standard_exponential(size=(n_sims, n)) - 1.0
        return mean + sigma * e
    raise ValueError(f"unsupported dist {dist!r}; choose one of {_SUPPORTED_DISTS}")


# ---------------------------------------------------------------------------
# Vectorised test statistics
# ---------------------------------------------------------------------------


def _levene_pvalues(samples: list[np.ndarray], *, center: str = "median") -> np.ndarray:
    """Vectorised Levene W statistic across the leading replicate axis.

    Each entry of ``samples`` is shape ``(n_sims, nᵢ)``.  Returns an
    array of length ``n_sims`` containing the p-values from the standard
    F distribution with df = (k-1, N-k).
    """
    k = len(samples)
    n_sims = samples[0].shape[0]
    ni = np.array([s.shape[1] for s in samples])
    N = int(ni.sum())

    if center == "median":
        z_groups = [np.abs(s - np.median(s, axis=1, keepdims=True)) for s in samples]
    elif center == "mean":
        z_groups = [np.abs(s - np.mean(s, axis=1, keepdims=True)) for s in samples]
    else:
        raise ValueError("center must be 'median' or 'mean'")

    # group means of z
    z_bar_g = np.stack([z.mean(axis=1) for z in z_groups], axis=1)        # (n_sims, k)
    # overall mean of z (weighted by ni)
    z_bar = (z_bar_g * ni).sum(axis=1) / N                                 # (n_sims,)

    # between-group SS
    ss_between = (ni * (z_bar_g - z_bar[:, None]) ** 2).sum(axis=1)
    # within-group SS
    ss_within = np.zeros(n_sims)
    for i, z in enumerate(z_groups):
        ss_within += ((z - z_bar_g[:, i:i + 1]) ** 2).sum(axis=1)

    df1 = k - 1
    df2 = N - k
    # Guard against zero within-group dispersion (degenerate replicates)
    ss_within = np.where(ss_within <= 0, np.finfo(float).tiny, ss_within)
    W = (ss_between / df1) / (ss_within / df2)
    p = 1.0 - stats.f.cdf(W, df1, df2)
    return p


def _bartlett_pvalues(samples: list[np.ndarray]) -> np.ndarray:
    """Vectorised Bartlett's test p-value across the replicate axis.

    Implements the standard formula

        T = ((N-k) ln(Sp²) − Σ (nᵢ-1) ln(Sᵢ²)) / C
        C = 1 + 1/(3(k-1)) · (Σ 1/(nᵢ-1) − 1/(N-k))

    Sᵢ² is the unbiased per-group variance (ddof=1).  Under H0,
    T ~ χ²(k-1).
    """
    k = len(samples)
    ni = np.array([s.shape[1] for s in samples])
    N = int(ni.sum())
    if np.any(ni < 2):
        raise ValueError("Bartlett requires every group to have nᵢ >= 2")

    # per-group variances: shape (n_sims, k)
    var_g = np.stack([s.var(axis=1, ddof=1) for s in samples], axis=1)
    var_g = np.where(var_g <= 0, np.finfo(float).tiny, var_g)

    nu = ni - 1                                                # df per group
    Sp2 = (nu * var_g).sum(axis=1) / (N - k)                   # pooled variance
    numer = (N - k) * np.log(Sp2) - (nu * np.log(var_g)).sum(axis=1)
    C = 1.0 + (1.0 / (3.0 * (k - 1))) * ((1.0 / nu).sum() - 1.0 / (N - k))
    T = numer / C
    p = 1.0 - stats.chi2.cdf(T, df=k - 1)
    return p


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------


def _validate_inputs(
    k_groups: int,
    sigmas: Sequence[float],
    means: Sequence[float] | None,
    dist: str,
    alpha: float,
    sides: int,
) -> tuple[list[float], list[float]]:
    if k_groups < 2:
        raise ValueError("k_groups must be >= 2")
    if len(sigmas) != k_groups:
        raise ValueError(
            f"len(sigmas)={len(sigmas)} does not match k_groups={k_groups}"
        )
    if any(s <= 0 for s in sigmas):
        raise ValueError("every sigma must be positive")
    mus = list(means) if means is not None else [0.0] * k_groups
    if len(mus) != k_groups:
        raise ValueError(
            f"len(means)={len(mus)} does not match k_groups={k_groups}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if dist not in _SUPPORTED_DISTS:
        raise ValueError(
            f"unsupported dist {dist!r}; choose one of {_SUPPORTED_DISTS}"
        )
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    return list(sigmas), mus


def _simulate_power(
    *,
    test: str,
    k_groups: int,
    sigmas: Sequence[float],
    means: Sequence[float],
    n_per_group: int | list[int],
    dist: str,
    alpha: float,
    n_sims: int,
    seed: int,
    df_t: float | None,
) -> float:
    rng = np.random.default_rng(seed)
    if isinstance(n_per_group, int):
        ni = [n_per_group] * k_groups
    else:
        ni = list(n_per_group)
        if len(ni) != k_groups:
            raise ValueError("len(n_per_group list) must match k_groups")
    samples = [
        _sample_group(
            rng,
            n=ni[i],
            mean=means[i],
            sigma=sigmas[i],
            dist=dist,
            n_sims=n_sims,
            df=df_t,
        )
        for i in range(k_groups)
    ]
    if test == "levene":
        # (mean-centred). The median-centred Brown-Forsythe variant is more
        # robust to non-normality; the mean form is the default.
        p = _levene_pvalues(samples, center="mean")
    elif test == "bartlett":
        p = _bartlett_pvalues(samples)
    elif test == "brown_forsythe":
        # Median-centred Levene (Brown-Forsythe 1974): Z_ki = |Y_ki - median(Y_k)|.
        p = _levene_pvalues(samples, center="median")
    elif test == "conover":
        p = _conover_pvalues(samples)
    else:  # pragma: no cover
        raise ValueError(f"unknown test {test!r}")
    return float(np.mean(p < alpha))


def _bisect_n(
    *,
    test: str,
    k_groups: int,
    sigmas: list[float],
    means: list[float],
    dist: str,
    alpha: float,
    target_power: float,
    n_sims: int,
    seed: int,
    df_t: float | None,
    n_min: int = 4,
    n_max: int = 5000,
) -> tuple[int, float]:
    if not 0.0 < target_power < 1.0:
        raise ValueError("power must be in (0, 1)")

    def power_at(n: int) -> float:
        return _simulate_power(
            test=test,
            k_groups=k_groups,
            sigmas=sigmas,
            means=means,
            n_per_group=n,
            dist=dist,
            alpha=alpha,
            n_sims=n_sims,
            seed=seed,
            df_t=df_t,
        )

    # geometric bracket
    lo = n_min
    hi = max(n_min, 8)
    p_hi = power_at(hi)
    while p_hi < target_power:
        lo = hi
        hi = hi * 2
        if hi > n_max:
            raise RuntimeError(
                f"failed to bracket n within {n_max}; last power={p_hi:.3f}"
            )
        p_hi = power_at(hi)

    # bisect
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if power_at(mid) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi, power_at(hi)


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------


def _solve(
    *,
    test: str,
    method_id: str,
    chapter: str,
    k_groups: int,
    sigmas: Sequence[float],
    means: Sequence[float] | None,
    dist: str,
    alpha: float,
    sides: int,
    n_per_group: int | None,
    power: float | None,
    n_sims: int,
    seed: int,
    df_t: float | None,
    solve_for: str | None,
    citations: list[str] | None = None,
) -> dict[str, Any]:
    sigmas_list, means_list = _validate_inputs(
        k_groups, sigmas, means, dist, alpha, sides
    )

    have_n = n_per_group is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n_per_group, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    if solve_for == "power":
        assert n_per_group is not None
        achieved = _simulate_power(
            test=test,
            k_groups=k_groups,
            sigmas=sigmas_list,
            means=means_list,
            n_per_group=n_per_group,
            dist=dist,
            alpha=alpha,
            n_sims=n_sims,
            seed=seed,
            df_t=df_t,
        )
        n_out = int(n_per_group)
    elif solve_for == "n":
        assert power is not None
        n_out, achieved = _bisect_n(
            test=test,
            k_groups=k_groups,
            sigmas=sigmas_list,
            means=means_list,
            dist=dist,
            alpha=alpha,
            target_power=float(power),
            n_sims=n_sims,
            seed=seed,
            df_t=df_t,
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    n_total = n_out * k_groups
    inputs_echo = {
        "k_groups": k_groups,
        "sigmas": sigmas_list,
        "means": means_list,
        "dist": dist,
        "alpha": alpha,
        "sides": sides,
        "n_per_group": n_per_group,
        "power": power,
        "n_sims": n_sims,
        "seed": seed,
        "df_t": df_t,
    }
    return {
        "method_id": method_id,
        "solve_for": solve_for,
        "n_per_group": n_out,
        "n_total": n_total,
        "achieved_power": achieved,
        "stochastic": True,
        "n_sims": n_sims,
        "seed": seed,
        "inputs_echo": inputs_echo,
        "citations": citations if citations is not None else [
            f"Brown & Forsythe (1974) / Conover (1999) variance test simulation ({chapter}).",
            "Levene, H. (1960). Robust tests for equality of variances. "
            "In I. Olkin (Ed.), Contributions to probability and statistics.",
            "Bartlett, M. S. (1937). Properties of sufficiency and statistical tests. "
            "Proc. Roy. Soc. London Ser. A 160, 268-282.",
        ],
        "notes": (
            "Monte-Carlo estimate; achieved_power is stochastic. With "
            f"n_sims={n_sims}, the 95% binomial CI half-width is ~"
            f"{1.96 * math.sqrt(achieved * (1 - achieved) / n_sims):.4f}."
        ),
    }


def levene_variances_simulation(
    *,
    k_groups: int,
    sigmas: Sequence[float],
    means: Sequence[float] | None = None,
    dist: str = "normal",
    alpha: float = 0.05,
    sides: int = 2,
    n_per_group: int | None = None,
    power: float | None = None,
    n_sims: int = 10000,
    seed: int = 42,
    df_t: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Levene's homogeneity-of-variance test, simulation-based power.

    Uses the median-centred ("Brown-Forsythe" form of Levene) test which
    Brown-Forsythe (1974) test is the default for robustness.  Per-group SDs are
    ``sigmas`` and (Levene/Bartlett are invariant to the group means
    under both H0 and H1) ``means`` defaults to zeros.

    Pass either ``n_per_group`` (-> solve for power) or ``power`` (->
    bisect ``n``).  Returned ``achieved_power`` is a Monte-Carlo estimate
    and is therefore stochastic; the 95% binomial half-width is given in
    ``notes``.  The default ``seed=42`` makes results reproducible.
    """
    return _solve(
        test="levene",
        method_id="levene_variances_simulation",
        chapter="Levene Test of Variances (Simulation)",
        k_groups=k_groups,
        sigmas=sigmas,
        means=means,
        dist=dist,
        alpha=alpha,
        sides=sides,
        n_per_group=n_per_group,
        power=power,
        n_sims=n_sims,
        seed=seed,
        df_t=df_t,
        solve_for=solve_for,
    )


def bartlett_variances_simulation(
    *,
    k_groups: int,
    sigmas: Sequence[float],
    means: Sequence[float] | None = None,
    dist: str = "normal",
    alpha: float = 0.05,
    sides: int = 2,
    n_per_group: int | None = None,
    power: float | None = None,
    n_sims: int = 10000,
    seed: int = 42,
    df_t: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Bartlett's homogeneity-of-variance test, simulation-based power.

    Bartlett is the likelihood-ratio test under normality; under
    non-normal ``dist`` is allowed so the user
    can read the actual (vs. target) alpha and judge robustness.

    Inputs and return contract mirror :func:`levene_variances_simulation`.
    """
    return _solve(
        test="bartlett",
        method_id="bartlett_variances_simulation",
        chapter="Bartlett Test of Variances (Simulation)",
        k_groups=k_groups,
        sigmas=sigmas,
        means=means,
        dist=dist,
        alpha=alpha,
        sides=sides,
        n_per_group=n_per_group,
        power=power,
        n_sims=n_sims,
        seed=seed,
        df_t=df_t,
        solve_for=solve_for,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _conover_pvalues(samples: list[np.ndarray]) -> np.ndarray:
    """Vectorised Conover (1999) squared-rank test for equal variances.

    Per the Conover (1999) formula:

        Z_ki = |Y_ki - mean(Y_k)|                  (absolute deviations
                                                    from the group mean)
        R_ki = Rank(Z_ki) over the *pooled* sample (average ties)
        S_k  = Σ_i R_ki²                            (sum of squared ranks
                                                    within group k)
        S̄    = (1/N) Σ_k S_k
        D²   = (1/(N-1)) · ( Σ_{k,i} R_ki⁴  -  N · S̄² )
        T    = (1/D²) · ( Σ_k S_k²/n_k  -  N · S̄² )

    Under H0, T ~ χ²(g-1).  This is the classic Conover squared-rank
    test (see Conover, *Practical Nonparametric Statistics*, 3rd ed.,
    1999, §5.3).

    Each entry of ``samples`` is shape ``(n_sims, n_k)``.  Returns an
    ``(n_sims,)`` vector of p-values.
    """
    k = len(samples)
    n_sims = samples[0].shape[0]
    ni = np.array([s.shape[1] for s in samples])
    N = int(ni.sum())

    # 1) absolute deviations from each group's mean (Levene 1960 uses the
    #    mean; the median-centred variant exists but Conover's published
    #    chi-square approximation is for mean-centred deviations).
    z_groups = [np.abs(s - np.mean(s, axis=1, keepdims=True)) for s in samples]

    # 2) pool and rank across the simulation axis, averaging ties.
    pooled = np.concatenate(z_groups, axis=1)                # (n_sims, N)
    ranks = stats.rankdata(pooled, method="average", axis=1)  # (n_sims, N)

    # 3) per-group rank sums S_k and pooled sum_R^4
    starts = np.cumsum(np.concatenate(([0], ni)))
    S_k = np.zeros((n_sims, k))
    sum_R4 = np.zeros(n_sims)
    for g in range(k):
        Rg = ranks[:, starts[g]:starts[g + 1]]
        S_k[:, g] = (Rg ** 2).sum(axis=1)
        sum_R4 += (Rg ** 4).sum(axis=1)

    Sbar = S_k.sum(axis=1) / N                                # (n_sims,)
    D2 = (sum_R4 - N * Sbar ** 2) / (N - 1)
    D2 = np.where(D2 <= 0, np.finfo(float).tiny, D2)
    T = ((S_k ** 2 / ni).sum(axis=1) - N * Sbar ** 2) / D2
    return 1.0 - stats.chi2.cdf(T, df=k - 1)


def brown_forsythe_variances_simulation(
    *,
    k_groups: int,
    sigmas: Sequence[float],
    means: Sequence[float] | None = None,
    dist: str = "normal",
    alpha: float = 0.05,
    sides: int = 2,
    n_per_group: int | None = None,
    power: float | None = None,
    n_sims: int = 10000,
    seed: int = 42,
    df_t: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Brown-Forsythe homogeneity-of-variance test (simulation-based).

    Brown & Forsythe (1974) is the median-centred form of Levene's test:

        Z_ki = | Y_ki - median(Y_k) |

    followed by an ordinary one-way ANOVA F-statistic on the ``Z_ki``
    with degrees of freedom ``(g-1, N-g)``.  Median centring makes the
    test markedly more robust to heavy-tailed/skewed distributions than
    the original mean-centred Levene form, which is why it is
    as a separate procedure (Chapter 554).

    Inputs and return contract mirror :func:`levene_variances_simulation`.
    ``achieved_power`` is a Monte-Carlo estimate at ``seed`` with
    ``n_sims`` replicates; the 95% binomial half-width is reported in
    ``notes``.
    """
    return _solve(
        test="brown_forsythe",
        method_id="brown_forsythe_variances_simulation",
        chapter="Brown-Forsythe Test of Variances (Simulation)",
        k_groups=k_groups,
        sigmas=sigmas,
        means=means,
        dist=dist,
        alpha=alpha,
        sides=sides,
        n_per_group=n_per_group,
        power=power,
        n_sims=n_sims,
        seed=seed,
        df_t=df_t,
        solve_for=solve_for,
        citations=[
            "Brown, M. B. & Forsythe, A. B. (1974). Robust tests for the equality of "
            "variances. Journal of the American Statistical Association 69(346), 364-367.",
            "Levene, H. (1960). Robust tests for equality of variances. "
            "In I. Olkin (Ed.), Contributions to probability and statistics.",
        ],
    )


def conover_variances_simulation(
    *,
    k_groups: int,
    sigmas: Sequence[float],
    means: Sequence[float] | None = None,
    dist: str = "normal",
    alpha: float = 0.05,
    sides: int = 2,
    n_per_group: int | None = None,
    power: float | None = None,
    n_sims: int = 10000,
    seed: int = 42,
    df_t: float | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Conover squared-rank homogeneity-of-variance test (simulation).

    Conover (1999) is a rank-based test on the within-group absolute
    deviations ``Z_ki = |Y_ki − Ȳ_k|``; the deviations are pooled,
    ranked (with average ties), squared, and assembled into a chi-square
    statistic with ``g−1`` degrees of freedom.  The formula
    (Chapter 561) is implemented in :func:`_conover_pvalues`.

    Because the test only looks at the ranks of the absolute deviations,
    it is distribution-free under H0 and noticeably more robust than
    Bartlett's test to non-normal data.  Inputs and return contract
    mirror :func:`levene_variances_simulation`.
    """
    return _solve(
        test="conover",
        method_id="conover_variances_simulation",
        chapter="Conover Test of Variances (Simulation)",
        k_groups=k_groups,
        sigmas=sigmas,
        means=means,
        dist=dist,
        alpha=alpha,
        sides=sides,
        n_per_group=n_per_group,
        power=power,
        n_sims=n_sims,
        seed=seed,
        df_t=df_t,
        solve_for=solve_for,
        citations=[
            "Conover, W. J. (1999). Practical Nonparametric Statistics (3rd ed.), "
            "Wiley, §5.3.",
        ],
    )
