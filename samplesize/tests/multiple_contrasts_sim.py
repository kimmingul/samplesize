"""Monte-Carlo simulation-based multiple-contrast power analyses.

simultaneously testing a user-defined family of linear contrasts of
``k`` group means with family-wise-error-rate (FWER) control.

A contrast is a vector ``c = (c_1, ..., c_k)`` with ``Σ c_i = 0``;
the null hypothesis is ``Σ c_i μ_i = 0`` and the two-sided alternative
is ``Σ c_i μ_i ≠ 0``.

Supported adjustments (``adjustment=`` argument):

* ``"bonferroni"`` — Dunn-Bonferroni: each contrast is
                     tested with Student-t at α/(2·C) (C = number of
                     contrasts), using the pooled within-group
                     variance and ``df = N − k``.  Eq.~Dunn (1961).
* ``"welch"``      — Dunn-Welch: each contrast uses its own variance
                     estimate Σ c_i² s_i²/n_i and Welch-Satterthwaite
                     degrees of freedom; useful when group variances
                     differ.
* ``"scheffe"``    — Scheffé's S-method (extends to *any* contrast):
                     rejects when the absolute studentised contrast
                     exceeds ``sqrt((k-1) F_{α,k-1,N-k})``.  Conservative
                     for a small finite family but exact for the
                     infinite contrast family.
* ``"none"``       — unadjusted Student-t; for reference / FWER
                     inflation studies.

``power_definition`` is one of:

* ``"any"``        — any-contrast power (probability of rejecting at
                     least one non-zero contrast),
* ``"all"`` / ``"complete"`` — all-contrasts power,
* ``"individual"`` — average of the per-contrast rejection
                     probabilities over the truly non-zero contrasts
                     (mean contrast power across truly non-zero contrasts).

Results are *stochastic*.  At ``n_sims=10000`` the 95% binomial CI
half-width is ~+/-0.010 at power=0.5 and ~+/-0.004 at power=0.95
Fixtures therefore use a
±0.01 tolerance.  ``seed=42`` is fixed by default for reproducibility.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats


_ADJUSTMENTS = ("bonferroni", "welch", "scheffe", "none")
_POWER_DEFS = ("any", "all", "individual", "complete")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    k_groups: int,
    means: Sequence[float],
    sd: float | Sequence[float],
    contrasts: Sequence[Sequence[float]],
    alpha: float,
    n_sims: int,
    coef_sum_tol: float = 1e-6,
) -> tuple[list[float], list[float], np.ndarray]:
    if k_groups < 2:
        raise ValueError("k_groups must be >= 2")
    if len(means) != k_groups:
        raise ValueError(f"len(means)={len(means)} does not match k_groups={k_groups}")
    if isinstance(sd, (int, float)):
        sds = [float(sd)] * k_groups
    else:
        sds = [float(s) for s in sd]
        if len(sds) != k_groups:
            raise ValueError(f"len(sd)={len(sds)} does not match k_groups={k_groups}")
    if any(s <= 0 for s in sds):
        raise ValueError("every sd must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_sims < 100:
        raise ValueError("n_sims must be at least 100")
    if len(contrasts) < 1:
        raise ValueError("at least one contrast must be supplied")

    C = np.asarray(contrasts, dtype=float)
    if C.ndim != 2 or C.shape[1] != k_groups:
        raise ValueError(
            f"contrasts must be a 2-D array with k_groups={k_groups} columns; "
            f"got shape {C.shape}"
        )
    sums = C.sum(axis=1)
    if np.any(np.abs(sums) > coef_sum_tol):
        bad = np.where(np.abs(sums) > coef_sum_tol)[0].tolist()
        raise ValueError(
            f"contrast rows must sum to zero; rows {bad} sum to {sums[bad].tolist()}"
        )

    return [float(m) for m in means], sds, C


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
# Data sim (shared style with pairwise_comparisons_sim)
# ---------------------------------------------------------------------------


def _simulate_means_and_variances(
    *,
    rng: np.random.Generator,
    n_sims: int,
    means: Sequence[float],
    sds: Sequence[float],
    ni: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = len(means)
    ni_arr = np.asarray(ni)
    df_within = int(ni_arr.sum() - k)

    ybar = np.empty((n_sims, k))
    s2_g = np.empty((n_sims, k))
    ss_within = np.zeros(n_sims)
    for j in range(k):
        z = rng.standard_normal(size=(n_sims, ni[j]))
        x = means[j] + sds[j] * z
        ybar[:, j] = x.mean(axis=1)
        s2_g[:, j] = x.var(axis=1, ddof=1)
        ss_within += (ni[j] - 1) * s2_g[:, j]
    s2_p = ss_within / df_within
    return ybar, s2_g, s2_p


# ---------------------------------------------------------------------------
# Power simulation core
# ---------------------------------------------------------------------------


def _power_from_rejections(
    rej: np.ndarray, truly_non_zero: np.ndarray, power_definition: str,
) -> float:
    if not truly_non_zero.any():
        return float("nan")
    if power_definition in ("complete", "all"):
        return float(rej[:, truly_non_zero].all(axis=1).mean())
    if power_definition == "any":
        return float(rej[:, truly_non_zero].any(axis=1).mean())
    if power_definition == "individual":
        return float(rej[:, truly_non_zero].mean())
    raise ValueError(
        f"unknown power_definition {power_definition!r}; choose from {_POWER_DEFS}"
    )


def _simulate_contrasts_power(
    *,
    k_groups: int,
    means: Sequence[float],
    sds: Sequence[float],
    ni: Sequence[int],
    C: np.ndarray,
    alpha: float,
    adjustment: str,
    n_sims: int,
    seed: int,
    power_definition: str,
    zero_tol: float = 1e-8,
) -> float:
    rng = np.random.default_rng(seed)
    ybar, s2_g, s2_p = _simulate_means_and_variances(
        rng=rng, n_sims=n_sims, means=means, sds=sds, ni=ni,
    )
    n_contrasts = C.shape[0]
    ni_arr = np.asarray(ni, dtype=float)
    df_within = int(ni_arr.sum() - k_groups)
    adj = adjustment.lower()

    if adj not in _ADJUSTMENTS:
        raise ValueError(
            f"adjustment {adjustment!r} not supported; choose from {_ADJUSTMENTS}"
        )

    # Numerator: estimated contrast L̂_c = Σ c_i Ȳ_i  -> (n_sims, n_contrasts)
    L_hat = ybar @ C.T

    # True contrast values for marking "truly non-zero".
    L_true = C @ np.asarray(means)
    truly_non_zero = np.abs(L_true) > zero_tol

    rej = np.empty((n_sims, n_contrasts), dtype=bool)

    if adj in ("bonferroni", "none", "scheffe"):
        # Pooled-variance SE: SE_c = sqrt(s²_p · Σ c_i²/n_i)
        sum_c2_over_n = (C ** 2 / ni_arr).sum(axis=1)        # (n_contrasts,)
        se = np.sqrt(s2_p[:, None] * sum_c2_over_n[None, :])  # (n_sims, C)
        stat = np.abs(L_hat) / se
        if adj == "bonferroni":
            crit = float(stats.t.ppf(1.0 - alpha / (2.0 * n_contrasts), df_within))
            rej[:] = stat >= crit
        elif adj == "none":
            crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_within))
            rej[:] = stat >= crit
        else:  # scheffe
            crit = math.sqrt((k_groups - 1) * stats.f.ppf(1.0 - alpha, k_groups - 1, df_within))
            rej[:] = stat >= crit
    else:  # "welch"
        # Per-contrast: SE = sqrt(Σ c_i² s²_i / n_i)
        # df by Welch-Satterthwaite.  Both vary across replicates.
        # Use Bonferroni alpha adjustment (Dunn-Welch).
        alpha_adj = alpha / (2.0 * n_contrasts)
        for cidx in range(n_contrasts):
            c = C[cidx]
            terms = (c ** 2) * s2_g / ni_arr                  # (n_sims, k)
            var = terms.sum(axis=1)                           # (n_sims,)
            denom = (terms ** 2 / (ni_arr - 1)).sum(axis=1)   # (n_sims,)
            denom = np.where(denom <= 0, np.finfo(float).tiny, denom)
            df_w = (var ** 2) / denom
            df_w = np.clip(df_w, 1.0, None)
            se = np.sqrt(var)
            stat = np.abs(L_hat[:, cidx]) / se
            crit = stats.t.ppf(1.0 - alpha_adj, df_w)
            rej[:, cidx] = stat >= crit

    return _power_from_rejections(rej, truly_non_zero, power_definition)


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


def multiple_contrasts_sim(
    *,
    k_groups: int,
    means: Sequence[float],
    sd: float | Sequence[float],
    contrasts: Sequence[Sequence[float]],
    alpha: float = 0.05,
    n_per_group: int | Sequence[int] | None = None,
    power: float | None = None,
    adjustment: str = "bonferroni",
    power_definition: str = "any",
    n_sims: int = 10000,
    seed: int = 42,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Multiple Contrasts simulation-based power calculator.

    Simulates ``n_sims`` one-way ANOVA datasets from
    ``N(means[i], sd² or sd[i]²)`` with sample sizes ``n_per_group``
    and tests the supplied ``contrasts`` (each a length-k vector whose
    coefficients sum to zero) using one of the FWER-controlling
    adjustments (``"bonferroni"``, ``"welch"``, ``"scheffe"``, or
    ``"none"``).

    ``power_definition`` ∈ ``{"any", "all"/"complete", "individual"}``.
    ``"individual"`` averages the per-contrast rejection probability
    over the contrasts that are truly non-zero under H1 (the
    "Mean Cont. Power" column).

    Pass either ``n_per_group`` (→ solve for power) or ``power`` (→
    bisect equal-allocation ``n``).  ``achieved_power`` is a Monte-Carlo
    estimate; the 95% binomial half-width is reported in ``notes``.
    """
    means_list, sds_list, C = _validate_inputs(
        k_groups, list(means), sd, contrasts, alpha, n_sims
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
        achieved = _simulate_contrasts_power(
            k_groups=k_groups,
            means=means_list,
            sds=sds_list,
            ni=ni,
            C=C,
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
            return _simulate_contrasts_power(
                k_groups=k_groups,
                means=means_list,
                sds=sds_list,
                ni=[n] * k_groups,
                C=C,
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
        "contrasts": C.tolist(),
        "alpha": alpha,
        "adjustment": adjustment,
        "power_definition": power_definition,
        "n_per_group": n_per_group,
        "power": power,
        "n_sims": n_sims,
        "seed": seed,
    }
    return {
        "method_id": "multiple_contrasts_sim",
        "solve_for": solve_for,
        "n_per_group": n_out,
        "n_total": n_total,
        "achieved_power": achieved,
        "adjustment": adjustment,
        "power_definition": power_definition,
        "n_contrasts": int(C.shape[0]),
        "stochastic": True,
        "n_sims": n_sims,
        "seed": seed,
        "inputs_echo": inputs_echo,
        "citations": [
            "Dunn, O. J. (1961). Multiple comparisons among means. JASA "
            "56(293), 52-64.",
            "Welch, B. L. (1947). The generalization of `Student's' problem "
            "when several different population variances are involved. "
            "Biometrika 34(1/2), 28-35.",
            "Kirk, R. E. (1982). Experimental Design (2nd ed.), Brooks/Cole, "
            "pp. 100-109.",
        ],
        "notes": (
            "Monte-Carlo estimate; achieved_power is stochastic. With "
            f"n_sims={n_sims}, the 95% binomial CI half-width is ~"
            f"{1.96 * math.sqrt(max(achieved, 1e-6) * (1 - max(achieved, 1e-6)) / n_sims):.4f}. "
            f"adjustment={adjustment!r}, power_definition={power_definition!r}, "
            f"n_contrasts={int(C.shape[0])}."
        ),
    }
