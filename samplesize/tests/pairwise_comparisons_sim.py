"""Monte-Carlo simulation-based multiple-comparison power analyses.

Implements two pairwise-comparison simulation procedures that share a common simulation engine:

* "Pair-Wise Multiple Comparisons (Simulation)" — Chapter 580.
  All C(k, 2) pair-wise comparisons among ``k`` group means with
  family-wise-error-rate (FWER) control.  Supported adjustments:

    - ``"tukey"``     — Tukey-Kramer studentised-range procedure
                       (default; pooled variance).
    - ``"bonferroni"`` — Bonferroni-adjusted Student-t.
    - ``"scheffe"``   — Scheffé's projection method using the F
                       distribution; protects against any contrast.
    - ``"none"``      — unadjusted pair-wise Student-t (for
                       reference / FWER inflation studies).

* "Multiple Comparisons of Treatments vs a Control (Simulation)" —
  Chapter 585.  Each of the ``k-1`` treatment groups is compared
  against a control group (assumed to be the first group).  Supported
  adjustments:

    - ``"dunnett"``    — Dunnett's classical two-sided treatments-
                        vs-control procedure.  We use the analytic
                        equicorrelated multivariate-t critical value
                        when sample sizes are equal (closed-form via
                        scipy's multivariate-t CDF) and fall back to a
                        Bonferroni-conservative critical value for
                        the unbalanced case.
    - ``"bonferroni"`` — Bonferroni-adjusted Student-t over the
                        ``k-1`` comparisons.
    - ``"none"``      — unadjusted Student-t.

Both functions return a dict with ``achieved_power`` interpreted under
the user-selected ``power_definition`` ∈ {``"any"``, ``"all"``,
``"individual"``}.  These power types are:

* ``"any"``        — any-pair / any-comparison power: probability of
                    rejecting *at least one* truly different pair.
* ``"all"``        — all-pairs / all-comparisons power: probability of
                    rejecting *all* truly different pairs simultaneously.
* ``"individual"`` — average of the per-pair rejection rates over the
                    truly different pairs (the "Mean Pairs Power"
                    column).  Equal to a single pair's power when only
                    one pair differs.

Results are *stochastic*.  With the default ``n_sims=10000`` the 95%
binomial CI half-width is ~+/-0.010 at power=0.5 and ~+/-0.004 at
power=0.95.  Fixtures use a
±0.01 tolerance.  All simulations seed ``numpy.random`` (default
``seed=42``) for reproducibility.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_common(
    k_groups: int,
    means: Sequence[float],
    sd: float | Sequence[float],
    alpha: float,
    n_sims: int,
) -> tuple[list[float], list[float]]:
    if k_groups < 2:
        raise ValueError("k_groups must be >= 2")
    if len(means) != k_groups:
        raise ValueError(f"len(means)={len(means)} does not match k_groups={k_groups}")
    if isinstance(sd, (int, float)):
        sds = [float(sd)] * k_groups
    else:
        sds = [float(s) for s in sd]
        if len(sds) != k_groups:
            raise ValueError(
                f"len(sd)={len(sds)} does not match k_groups={k_groups}"
            )
    if any(s <= 0 for s in sds):
        raise ValueError("every sd must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_sims < 100:
        raise ValueError("n_sims must be at least 100")
    return [float(m) for m in means], sds


def _resolve_n(
    n_per_group: int | Sequence[int],
    k_groups: int,
) -> list[int]:
    if isinstance(n_per_group, int):
        return [int(n_per_group)] * k_groups
    ni = [int(n) for n in n_per_group]
    if len(ni) != k_groups:
        raise ValueError("len(n_per_group list) must match k_groups")
    if any(n < 2 for n in ni):
        raise ValueError("each per-group n must be >= 2")
    return ni


# ---------------------------------------------------------------------------
# Vectorised data simulation
# ---------------------------------------------------------------------------


def _simulate_means_and_pooled_sd(
    *,
    rng: np.random.Generator,
    n_sims: int,
    means: Sequence[float],
    sds: Sequence[float],
    ni: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised normal sampler.

    Returns:
        ybar : (n_sims, k) array of group sample means.
        s2_g : (n_sims, k) array of unbiased per-group variances.
        s2_p : (n_sims,)   pooled within-group variance (ANOVA MSE).
    """
    k = len(means)
    ni_arr = np.asarray(ni)
    df_within = int(ni_arr.sum() - k)

    ybar = np.empty((n_sims, k))
    s2_g = np.empty((n_sims, k))
    ss_within = np.zeros(n_sims)
    for j in range(k):
        z = rng.standard_normal(size=(n_sims, ni[j]))
        x = means[j] + sds[j] * z
        m = x.mean(axis=1)
        # unbiased variance, ddof=1
        v = x.var(axis=1, ddof=1)
        ybar[:, j] = m
        s2_g[:, j] = v
        ss_within += (ni[j] - 1) * v
    s2_p = ss_within / df_within
    return ybar, s2_g, s2_p


# ---------------------------------------------------------------------------
# Critical values for the pair-wise / control adjustments
# ---------------------------------------------------------------------------


def _crit_tukey(alpha: float, k: int, df: int) -> float:
    """Studentised-range upper-α critical value q_{α,k,df}.

    Tukey-Kramer rejects |Ȳ_i - Ȳ_j| / sqrt((s²/2)(1/nᵢ + 1/nⱼ)) ≥ q.
    """
    return float(stats.studentized_range.ppf(1.0 - alpha, k, df))


def _crit_bonferroni_t(alpha: float, n_comparisons: int, df: int) -> float:
    """Two-sided Bonferroni-adjusted Student-t critical value."""
    return float(stats.t.ppf(1.0 - alpha / (2.0 * n_comparisons), df))


def _crit_scheffe(alpha: float, k: int, df: int) -> float:
    """Scheffé critical value for |contrast|/SE ≥ sqrt((k-1) F_{α,k-1,df})."""
    return float(math.sqrt((k - 1) * stats.f.ppf(1.0 - alpha, k - 1, df)))


def _crit_dunnett_two_sided(alpha: float, k_minus_1: int, df: int) -> float:
    """Equicorrelated two-sided Dunnett critical value.

    For balanced one-way ANOVA where every treatment is compared against a
    common control, the pairwise t-statistics share a common pairwise
    correlation ρ = 1/2.  The two-sided critical value is the solution
    in d of

        P( max_i |T_i| ≤ d ) = 1 - α

    where (T_1, ..., T_{k-1}) is a (k-1)-variate central t-distribution
    with df=df and equicorrelation matrix Σ with diagonal 1 and
    off-diagonal 1/2.

    We compute this via scipy.stats.multivariate_t.cdf with a small
    bisection.  Falls back to a Bonferroni-conservative critical value
    if scipy lacks ``multivariate_t``.
    """
    if k_minus_1 < 1:
        raise ValueError("k_minus_1 must be >= 1")
    if k_minus_1 == 1:
        return float(stats.t.ppf(1.0 - alpha / 2.0, df))

    # Equicorrelation Σ with ρ = 1/2.
    rho = 0.5
    sigma = np.full((k_minus_1, k_minus_1), rho)
    np.fill_diagonal(sigma, 1.0)
    try:
        mvt = stats.multivariate_t(loc=np.zeros(k_minus_1), shape=sigma, df=df)
    except (AttributeError, TypeError):  # pragma: no cover - very old scipy
        return _crit_bonferroni_t(alpha, k_minus_1, df)

    target = 1.0 - alpha

    def coverage(d: float) -> float:
        lo = np.full(k_minus_1, -d)
        hi = np.full(k_minus_1, d)
        try:
            return float(mvt.cdf(hi, lower_limit=lo))
        except TypeError:
            # Older scipy: cdf only supports upper -> use Bonferroni fallback.
            return float("nan")

    # Bracket: Bonferroni-Sidak gives a safe upper bound; t_{α/2,df} a lower.
    lo = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    hi = _crit_bonferroni_t(alpha, k_minus_1, df) + 0.5
    c_lo = coverage(lo)
    if math.isnan(c_lo):
        return _crit_bonferroni_t(alpha, k_minus_1, df)
    c_hi = coverage(hi)
    # Expand hi if needed.
    while c_hi < target and hi < 50.0:
        hi += 0.5
        c_hi = coverage(hi)
    # Bisection.
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        c = coverage(mid)
        if c >= target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Power simulation core
# ---------------------------------------------------------------------------


_ADJUSTMENTS_PAIRWISE = ("tukey", "bonferroni", "scheffe", "none")
_ADJUSTMENTS_VS_CONTROL = ("dunnett", "bonferroni", "none")
_POWER_DEFS = ("any", "all", "individual", "complete")  # complete == all


def _power_from_rejections(
    rej: np.ndarray,            # (n_sims, n_comparisons) boolean
    truly_diff: np.ndarray,     # (n_comparisons,)        boolean
    power_definition: str,
) -> float:
    if power_definition in ("complete", "all"):
        if not truly_diff.any():
            return float("nan")
        return float(rej[:, truly_diff].all(axis=1).mean())
    if power_definition == "any":
        if not truly_diff.any():
            return float("nan")
        return float(rej[:, truly_diff].any(axis=1).mean())
    if power_definition == "individual":
        if not truly_diff.any():
            return float("nan")
        # average over the truly different pairs of per-pair rejection prob.
        return float(rej[:, truly_diff].mean())
    raise ValueError(
        f"unknown power_definition {power_definition!r}; "
        f"choose from {_POWER_DEFS}"
    )


def _simulate_pairwise_power(
    *,
    k_groups: int,
    means: Sequence[float],
    sds: Sequence[float],
    ni: Sequence[int],
    alpha: float,
    adjustment: str,
    n_sims: int,
    seed: int,
    power_definition: str,
    equal_tol: float = 1e-8,
) -> float:
    rng = np.random.default_rng(seed)
    ybar, s2_g, s2_p = _simulate_means_and_pooled_sd(
        rng=rng, n_sims=n_sims, means=means, sds=sds, ni=ni,
    )
    k = k_groups
    pairs = list(combinations(range(k), 2))
    n_pairs = len(pairs)
    df_within = int(sum(ni) - k)
    s_p = np.sqrt(s2_p)                                  # (n_sims,)

    # Pre-compute critical value(s).
    adj = adjustment.lower()
    if adj == "tukey":
        crit = _crit_tukey(alpha, k, df_within)
    elif adj == "bonferroni":
        crit = _crit_bonferroni_t(alpha, n_pairs, df_within)
    elif adj == "scheffe":
        crit = _crit_scheffe(alpha, k, df_within)
    elif adj == "none":
        crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_within))
    else:
        raise ValueError(
            f"adjustment {adjustment!r} not supported; choose from "
            f"{_ADJUSTMENTS_PAIRWISE}"
        )

    rej = np.empty((n_sims, n_pairs), dtype=bool)
    truly_diff = np.empty(n_pairs, dtype=bool)
    for p, (i, j) in enumerate(pairs):
        diff = ybar[:, i] - ybar[:, j]
        if adj == "tukey":
            # Standard Tukey-Kramer statistic q = |Ȳ_i - Ȳ_j| /
            #   sqrt((s²/2)(1/nᵢ + 1/nⱼ)).
            se = np.sqrt(0.5 * s2_p * (1.0 / ni[i] + 1.0 / ni[j]))
            stat = np.abs(diff) / se
            rej[:, p] = stat >= crit
        else:
            # t-style statistic for Bonferroni / Scheffé / none.  Scheffé
            # compares the studentised contrast directly: the equivalent
            # form is |Ȳ_i - Ȳ_j| / sqrt(s²(1/nᵢ + 1/nⱼ)) ≥ sqrt((k-1)F).
            se = np.sqrt(s2_p * (1.0 / ni[i] + 1.0 / ni[j]))
            stat = np.abs(diff) / se
            rej[:, p] = stat >= crit
        truly_diff[p] = abs(means[i] - means[j]) > equal_tol

    return _power_from_rejections(rej, truly_diff, power_definition)


def _simulate_vs_control_power(
    *,
    k_groups: int,
    means: Sequence[float],
    sds: Sequence[float],
    ni: Sequence[int],
    alpha: float,
    adjustment: str,
    n_sims: int,
    seed: int,
    power_definition: str,
    equal_tol: float = 1e-8,
) -> float:
    rng = np.random.default_rng(seed)
    ybar, s2_g, s2_p = _simulate_means_and_pooled_sd(
        rng=rng, n_sims=n_sims, means=means, sds=sds, ni=ni,
    )
    k = k_groups
    n_comp = k - 1
    df_within = int(sum(ni) - k)

    adj = adjustment.lower()
    if adj == "dunnett":
        # Balanced design uses the analytic Dunnett critical value with
        # ρ = 1/2.  Fall back to Bonferroni for unbalanced designs.
        if len(set(ni)) == 1:
            crit = _crit_dunnett_two_sided(alpha, n_comp, df_within)
        else:
            crit = _crit_bonferroni_t(alpha, n_comp, df_within)
    elif adj == "bonferroni":
        crit = _crit_bonferroni_t(alpha, n_comp, df_within)
    elif adj == "none":
        crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_within))
    else:
        raise ValueError(
            f"adjustment {adjustment!r} not supported; choose from "
            f"{_ADJUSTMENTS_VS_CONTROL}"
        )

    rej = np.empty((n_sims, n_comp), dtype=bool)
    truly_diff = np.empty(n_comp, dtype=bool)
    for p, j in enumerate(range(1, k)):
        diff = ybar[:, j] - ybar[:, 0]
        se = np.sqrt(s2_p * (1.0 / ni[j] + 1.0 / ni[0]))
        stat = np.abs(diff) / se
        rej[:, p] = stat >= crit
        truly_diff[p] = abs(means[j] - means[0]) > equal_tol

    return _power_from_rejections(rej, truly_diff, power_definition)


# ---------------------------------------------------------------------------
# n-bisection
# ---------------------------------------------------------------------------


def _bisect_n(
    *,
    power_fn,
    target_power: float,
    n_min: int = 3,
    n_max: int = 2000,
) -> tuple[int, float]:
    if not 0.0 < target_power < 1.0:
        raise ValueError("power must be in (0, 1)")
    lo = n_min
    hi = max(n_min, 8)
    p_hi = power_fn(hi)
    while p_hi < target_power:
        lo = hi
        hi *= 2
        if hi > n_max:
            raise RuntimeError(
                f"failed to bracket n within {n_max}; last power={p_hi:.3f}"
            )
        p_hi = power_fn(hi)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        p_mid = power_fn(mid)
        if p_mid >= target_power:
            hi = mid
        else:
            lo = mid
    return hi, power_fn(hi)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pair_wise_multiple_comparisons_sim(
    *,
    k_groups: int,
    means: Sequence[float],
    sd: float | Sequence[float],
    alpha: float = 0.05,
    n_per_group: int | Sequence[int] | None = None,
    power: float | None = None,
    adjustment: str = "tukey",
    power_definition: str = "any",
    n_sims: int = 10000,
    seed: int = 42,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Pair-Wise Multiple Comparisons simulation-based power calculator.

    Simulates ``n_sims`` one-way ANOVA datasets from
    ``N(means[i], sd² or sd[i]²)`` with sample sizes ``n_per_group``
    and reports the proportion in which the selected pair-wise
    procedure rejects according to ``power_definition``:

    * ``"any"``        any-pair power (probability of detecting at
                       least one truly different pair),
    * ``"all"``/``"complete"`` all-pairs power,
    * ``"individual"`` average per-pair rejection probability over the
                       truly different pairs.

    ``adjustment`` selects the FWER-controlling procedure:
    ``"tukey"`` (default, studentised range), ``"bonferroni"``,
    ``"scheffe"``, or ``"none"`` (unadjusted Student-t for reference).

    Pass either ``n_per_group`` (→ solve for power) or ``power`` (→
    bisect equal-allocation ``n``).  ``achieved_power`` is a Monte-Carlo
    estimate; the 95% binomial half-width appears in ``notes``.
    """
    means_list, sds_list = _validate_common(
        k_groups, list(means), sd, alpha, n_sims
    )
    if power_definition not in _POWER_DEFS:
        raise ValueError(
            f"power_definition {power_definition!r} not supported; choose "
            f"from {_POWER_DEFS}"
        )

    have_n = n_per_group is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n_per_group, power)")
    if solve_for is None:
        solve_for = "power" if have_n else "n"

    if solve_for == "power":
        assert n_per_group is not None
        ni = _resolve_n(n_per_group, k_groups)
        achieved = _simulate_pairwise_power(
            k_groups=k_groups,
            means=means_list,
            sds=sds_list,
            ni=ni,
            alpha=alpha,
            adjustment=adjustment,
            n_sims=n_sims,
            seed=seed,
            power_definition=power_definition,
        )
        n_out = ni if isinstance(n_per_group, (list, tuple)) else int(ni[0])
    elif solve_for == "n":
        assert power is not None

        def power_at(n: int) -> float:
            return _simulate_pairwise_power(
                k_groups=k_groups,
                means=means_list,
                sds=sds_list,
                ni=[n] * k_groups,
                alpha=alpha,
                adjustment=adjustment,
                n_sims=n_sims,
                seed=seed,
                power_definition=power_definition,
            )

        n_out, achieved = _bisect_n(power_fn=power_at, target_power=float(power))
        ni = [n_out] * k_groups
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    n_total = sum(ni)
    inputs_echo = {
        "k_groups": k_groups,
        "means": means_list,
        "sd": sds_list if not all(s == sds_list[0] for s in sds_list) else sds_list[0],
        "alpha": alpha,
        "adjustment": adjustment,
        "power_definition": power_definition,
        "n_per_group": n_per_group,
        "power": power,
        "n_sims": n_sims,
        "seed": seed,
    }
    return {
        "method_id": "pair_wise_multiple_comparisons_sim",
        "solve_for": solve_for,
        "n_per_group": n_out,
        "n_total": n_total,
        "achieved_power": achieved,
        "adjustment": adjustment,
        "power_definition": power_definition,
        "stochastic": True,
        "n_sims": n_sims,
        "seed": seed,
        "inputs_echo": inputs_echo,
        "citations": [
            "Tukey, J. W. (1953). The problem of multiple comparisons. "
            "Unpublished manuscript, Princeton University.",
            "Kramer, C. Y. (1956). Extension of multiple range tests to group "
            "means with unequal numbers of replications. Biometrics 12, 307-310.",
            "Ramsey, P. H. (1978). Power differences between pairwise multiple "
            "comparisons. JASA 73(363), 479-485.",
        ],
        "notes": (
            "Monte-Carlo estimate; achieved_power is stochastic. With "
            f"n_sims={n_sims}, the 95% binomial CI half-width is ~"
            f"{1.96 * math.sqrt(max(achieved, 1e-6) * (1 - max(achieved, 1e-6)) / n_sims):.4f}. "
            f"adjustment={adjustment!r}, power_definition={power_definition!r}."
        ),
    }


def multiple_comparisons_vs_control_sim(
    *,
    k_groups: int,
    means: Sequence[float],
    sd: float | Sequence[float],
    alpha: float = 0.05,
    n_per_group: int | Sequence[int] | None = None,
    power: float | None = None,
    adjustment: str = "dunnett",
    power_definition: str = "any",
    n_sims: int = 10000,
    seed: int = 42,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Multiple Comparisons of Treatments vs a Control simulation-based power calculator.

    Compares each of the ``k-1`` treatment groups (indices 1..k-1) to
    the control (index 0) using the selected FWER-controlling procedure
    (``"dunnett"``, ``"bonferroni"``, or ``"none"``).  Inputs and return
    contract mirror :func:`pair_wise_multiple_comparisons_sim` but with
    only ``k-1`` comparisons.

    For balanced equal-allocation designs ``"dunnett"`` uses the exact
    equicorrelated multivariate-t critical value (ρ=1/2) computed via
    scipy's multivariate-t CDF; for unbalanced designs we fall back to
    the (slightly conservative) Bonferroni critical value to avoid the
    much heavier general Dunnett quadrature.
    """
    means_list, sds_list = _validate_common(
        k_groups, list(means), sd, alpha, n_sims
    )
    if power_definition not in _POWER_DEFS:
        raise ValueError(
            f"power_definition {power_definition!r} not supported; choose "
            f"from {_POWER_DEFS}"
        )

    have_n = n_per_group is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n_per_group, power)")
    if solve_for is None:
        solve_for = "power" if have_n else "n"

    if solve_for == "power":
        assert n_per_group is not None
        ni = _resolve_n(n_per_group, k_groups)
        achieved = _simulate_vs_control_power(
            k_groups=k_groups,
            means=means_list,
            sds=sds_list,
            ni=ni,
            alpha=alpha,
            adjustment=adjustment,
            n_sims=n_sims,
            seed=seed,
            power_definition=power_definition,
        )
        n_out = ni if isinstance(n_per_group, (list, tuple)) else int(ni[0])
    elif solve_for == "n":
        assert power is not None

        def power_at(n: int) -> float:
            return _simulate_vs_control_power(
                k_groups=k_groups,
                means=means_list,
                sds=sds_list,
                ni=[n] * k_groups,
                alpha=alpha,
                adjustment=adjustment,
                n_sims=n_sims,
                seed=seed,
                power_definition=power_definition,
            )

        n_out, achieved = _bisect_n(power_fn=power_at, target_power=float(power))
        ni = [n_out] * k_groups
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    n_total = sum(ni)
    inputs_echo = {
        "k_groups": k_groups,
        "means": means_list,
        "sd": sds_list if not all(s == sds_list[0] for s in sds_list) else sds_list[0],
        "alpha": alpha,
        "adjustment": adjustment,
        "power_definition": power_definition,
        "n_per_group": n_per_group,
        "power": power,
        "n_sims": n_sims,
        "seed": seed,
    }
    return {
        "method_id": "multiple_comparisons_vs_control_sim",
        "solve_for": solve_for,
        "n_per_group": n_out,
        "n_total": n_total,
        "achieved_power": achieved,
        "adjustment": adjustment,
        "power_definition": power_definition,
        "stochastic": True,
        "n_sims": n_sims,
        "seed": seed,
        "inputs_echo": inputs_echo,
        "citations": [
            "a Control (Simulation).",
            "Dunnett, C. W. (1955). A multiple comparison procedure for "
            "comparing several treatments with a control. JASA 50(272), "
            "1096-1121.",
            "Ramsey, P. H. (1978). Power differences between pairwise multiple "
            "comparisons. JASA 73(363), 479-485.",
        ],
        "notes": (
            "Monte-Carlo estimate; achieved_power is stochastic. With "
            f"n_sims={n_sims}, the 95% binomial CI half-width is ~"
            f"{1.96 * math.sqrt(max(achieved, 1e-6) * (1 - max(achieved, 1e-6)) / n_sims):.4f}. "
            f"adjustment={adjustment!r}, power_definition={power_definition!r}. "
            "Balanced Dunnett uses the exact equicorrelated multivariate-t "
            "critical value (rho=1/2); unbalanced falls back to Bonferroni."
        ),
    }
