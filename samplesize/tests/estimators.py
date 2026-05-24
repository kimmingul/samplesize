"""Estimator utilities for sample-size planning.

These are *not* sample-size methods. They are pure-function calculators
that produce an estimate (an effect size, an SD, Cohen's w, or kappa)
that other sample-size routines then consume.

Chapters covered:

- "Standard Deviation Estimator"
- "Standard Deviation of Means Calculator"
- "Chi-Square Effect Size Estimator"
- "Kappa Estimator"

Each function returns the standard envelope:

    {
        "method_id":   <str>,
        "solve_for":   "estimate",
        "result":      <single number>,
        "inputs_echo": <dict of the inputs as supplied>,
        "citations":   [...],
        ...method-specific extras (e.g. df, components)...
    }
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1. Standard Deviation Estimator
# ---------------------------------------------------------------------------


def standard_deviation_estimator(
    *,
    # Pooled-from-prior-studies mode
    sds: Sequence[float] | None = None,
    ns: Sequence[int] | None = None,
    # Data-values mode (Example 1 / Example 2 in the chapter)
    values: Sequence[float] | None = None,
    counts: Sequence[int] | None = None,
    use_n_minus_1: bool = True,
    # Standard-error mode
    se: float | None = None,
    n_for_se: int | None = None,
    # Coefficient-of-variation mode
    cov: float | None = None,
    mean: float | None = None,
    # Range mode (population or sample)
    range_value: float | None = None,
    range_divisor: float | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    """Estimate a single σ̂ from one of several input modes.

    Modes (mutually exclusive; ``method`` may force a particular one):

    - ``"pooled"``        — weighted pool of prior-study SDs.
                             σ̂ = sqrt(Σ(nᵢ-1)·sᵢ² / Σ(nᵢ-1)).
    - ``"values"``        — list of raw values (with optional counts).
                             SD with N-1 (default) or N divisor.
    - ``"standard_error"``— SD = SE·sqrt(N).
    - ``"cov"``           — SD = COV · mean.
    - ``"range"``         — SD = Range / C   (C in {4, 5, 6, ...} or
                                              the sample-size dependent
                                              median range divisor).

    Mode auto-detection: if ``method`` is ``None`` the first set of
    arguments that is fully supplied wins, checked in the order above.
    """
    inputs_echo = {
        "sds": list(sds) if sds is not None else None,
        "ns": list(ns) if ns is not None else None,
        "values": list(values) if values is not None else None,
        "counts": list(counts) if counts is not None else None,
        "use_n_minus_1": use_n_minus_1,
        "se": se, "n_for_se": n_for_se,
        "cov": cov, "mean": mean,
        "range_value": range_value, "range_divisor": range_divisor,
        "method": method,
    }

    # Pick mode if not given.
    if method is None:
        if sds is not None and ns is not None:
            method = "pooled"
        elif values is not None:
            method = "values"
        elif se is not None and n_for_se is not None:
            method = "standard_error"
        elif cov is not None and mean is not None:
            method = "cov"
        elif range_value is not None and range_divisor is not None:
            method = "range"
        else:
            raise ValueError(
                "supply one valid input mode: (sds+ns), (values), "
                "(se+n_for_se), (cov+mean), or (range_value+range_divisor)"
            )

    extras: dict[str, Any] = {"method": method}

    if method == "pooled":
        if sds is None or ns is None:
            raise ValueError("pooled mode requires both 'sds' and 'ns'")
        if len(sds) != len(ns):
            raise ValueError("sds and ns must have the same length")
        if any(n < 2 for n in ns):
            raise ValueError("each study must have n >= 2")
        if any(s < 0 for s in sds):
            raise ValueError("SDs must be non-negative")
        num = sum((n - 1) * (s * s) for s, n in zip(sds, ns))
        den = sum(n - 1 for n in ns)
        if den == 0:
            raise ValueError("total degrees of freedom is zero")
        sd_hat = math.sqrt(num / den)
        extras["total_df"] = den
        extras["k_studies"] = len(sds)

    elif method == "values":
        if values is None:
            raise ValueError("values mode requires 'values'")
        if counts is not None:
            if len(counts) != len(values):
                raise ValueError("counts and values must have the same length")
            if any(c < 0 for c in counts):
                raise ValueError("counts must be non-negative")
            expanded: list[float] = []
            for v, c in zip(values, counts):
                expanded.extend([float(v)] * int(c))
            data = expanded if expanded else [float(v) for v in values]
        else:
            data = [float(v) for v in values]
        n = len(data)
        if n < 2:
            raise ValueError("need at least 2 observations")
        mean_val = sum(data) / n
        ss = sum((x - mean_val) ** 2 for x in data)
        divisor = (n - 1) if use_n_minus_1 else n
        if divisor == 0:
            raise ValueError("divisor is zero")
        sd_hat = math.sqrt(ss / divisor)
        extras["n"] = n
        extras["mean"] = mean_val
        extras["divisor"] = "n-1" if use_n_minus_1 else "n"

    elif method == "standard_error":
        if se is None or n_for_se is None:
            raise ValueError("standard_error mode requires 'se' and 'n_for_se'")
        if n_for_se < 1:
            raise ValueError("n_for_se must be >= 1")
        if se < 0:
            raise ValueError("se must be non-negative")
        sd_hat = se * math.sqrt(n_for_se)

    elif method == "cov":
        if cov is None or mean is None:
            raise ValueError("cov mode requires 'cov' and 'mean'")
        sd_hat = cov * mean

    elif method == "range":
        if range_value is None or range_divisor is None:
            raise ValueError(
                "range mode requires 'range_value' and 'range_divisor'"
            )
        if range_divisor <= 0:
            raise ValueError("range_divisor must be positive")
        sd_hat = range_value / range_divisor

    else:
        raise ValueError(f"unknown method: {method!r}")

    return {
        "method_id": "standard_deviation_estimator",
        "solve_for": "estimate",
        "result": sd_hat,
        "sd_estimate": sd_hat,
        **extras,
        "inputs_echo": inputs_echo,
        "citations": [
        ],
    }


# ---------------------------------------------------------------------------
# 2. Standard Deviation of Means Calculator
# ---------------------------------------------------------------------------


def standard_deviation_of_means_calculator(
    *,
    means: Sequence[float] | Sequence[Sequence[float]],
    factor: str | None = None,
) -> dict[str, Any]:
    """σ_m from a list of hypothesised means (effects table convention).

    The definition used here is

        σ_m = sqrt( Σ_i (μ_i - μ̄)² / k )

    where k is the number of levels in the factor. For a 1-D list of
    means this is simply the population SD of the means. For a 2-D
    table of means (one-way × one-way crossed) supply ``factor`` to
    request a particular effects component:

    - ``"row"``         — main effect for the row factor (k = R)
    - ``"col"``         — main effect for the column factor (k = C)
    - ``"interaction"`` — interaction effect (k = R·C)

    If ``factor`` is omitted on a 2-D input, the *grand* (overall) σ_m
    is returned, which equals the population SD of all R·C means.
    """
    arr = np.asarray(means, dtype=float)
    inputs_echo = {
        "means": arr.tolist(),
        "factor": factor,
    }

    if arr.ndim == 1:
        if arr.size < 2:
            raise ValueError("need at least 2 means")
        k = arr.size
        grand_mean = float(arr.mean())
        effects = arr - grand_mean
        sm = math.sqrt(float((effects ** 2).sum()) / k)
        return {
            "method_id": "standard_deviation_of_means_calculator",
            "solve_for": "estimate",
            "result": sm,
            "sm": sm,
            "k": k,
            "grand_mean": grand_mean,
            "effects": effects.tolist(),
            "inputs_echo": inputs_echo,
            "citations": [
                "of Means Calculator.",
            ],
        }

    if arr.ndim == 2:
        R, C = arr.shape
        if R < 2 or C < 2:
            raise ValueError("2-D means must be at least 2x2")
        grand = float(arr.mean())
        row_means = arr.mean(axis=1)
        col_means = arr.mean(axis=0)
        row_eff = row_means - grand
        col_eff = col_means - grand
        inter = arr - row_means[:, None] - col_means[None, :] + grand

        sm_row = math.sqrt(float((row_eff ** 2).sum()) / R)
        sm_col = math.sqrt(float((col_eff ** 2).sum()) / C)
        sm_inter = math.sqrt(float((inter ** 2).sum()) / (R * C))
        sm_grand = math.sqrt(float(((arr - grand) ** 2).sum()) / (R * C))

        if factor is None:
            sm = sm_grand
            k = R * C
        elif factor == "row":
            sm = sm_row
            k = R
        elif factor == "col":
            sm = sm_col
            k = C
        elif factor == "interaction":
            sm = sm_inter
            k = R * C
        else:
            raise ValueError(
                f"factor must be one of None/'row'/'col'/'interaction', "
                f"got {factor!r}"
            )

        return {
            "method_id": "standard_deviation_of_means_calculator",
            "solve_for": "estimate",
            "result": sm,
            "sm": sm,
            "sm_row": sm_row,
            "sm_col": sm_col,
            "sm_interaction": sm_inter,
            "sm_grand": sm_grand,
            "k": k,
            "grand_mean": grand,
            "row_effects": row_eff.tolist(),
            "col_effects": col_eff.tolist(),
            "interaction_effects": inter.tolist(),
            "inputs_echo": inputs_echo,
            "citations": [
                "of Means Calculator.",
            ],
        }

    raise ValueError("means must be a 1-D or 2-D array-like")


# ---------------------------------------------------------------------------
# 3. Chi-Square Effect Size Estimator
# ---------------------------------------------------------------------------


def chi_square_effect_size_estimator(
    *,
    p1: Sequence[float] | Sequence[Sequence[float]],
    p0: Sequence[float] | Sequence[Sequence[float]] | None = None,
    table_kind: str = "auto",
) -> dict[str, Any]:
    """Cohen's w from a probability/percentage/count matrix.

    Parameters
    ----------
    p1 : array-like
        Alternative-hypothesis cell proportions, percentages, or counts.
        Percentages (values > 1) are rescaled to sum to 1; counts are
        normalised by the total.
    p0 : array-like, optional
        Null-hypothesis cell proportions. For a contingency table this
        is the independence model derived from the marginals of ``p1``;
        omit ``p0`` to use that default. For a multinomial / goodness-
        of-fit problem you must supply ``p0`` explicitly (one entry per
        cell).
    table_kind : {"auto", "contingency", "multinomial"}
        ``"auto"`` infers from the shape of ``p1``: 2-D → contingency,
        1-D → multinomial.

    Returns
    -------
    dict with ``result`` = Cohen's w. Additional fields: ``df``,
    ``chi_square_at_N`` (None unless inputs were counts), ``p0``, ``p1``.
    """
    a1 = np.asarray(p1, dtype=float)
    inputs_echo = {
        "p1": a1.tolist(),
        "p0": (np.asarray(p0).tolist() if p0 is not None else None),
        "table_kind": table_kind,
    }

    if a1.ndim not in (1, 2):
        raise ValueError("p1 must be 1-D or 2-D")
    if np.any(a1 < 0):
        raise ValueError("p1 entries must be non-negative")

    kind = table_kind
    if kind == "auto":
        kind = "contingency" if a1.ndim == 2 else "multinomial"

    total = float(a1.sum())
    if total <= 0:
        raise ValueError("p1 sum must be > 0")

    # If user passed counts (sum > ~1.5) we normalise by total.
    # The sum-near-1 heuristic also catches proportions.
    is_counts = total > 1.5 + 1e-9
    p1_norm = a1 / total

    if kind == "contingency":
        if a1.ndim != 2:
            raise ValueError("contingency tables must be 2-D")
        R, C = p1_norm.shape
        if p0 is None:
            row_marg = p1_norm.sum(axis=1, keepdims=True)
            col_marg = p1_norm.sum(axis=0, keepdims=True)
            p0_norm = row_marg @ col_marg
        else:
            a0 = np.asarray(p0, dtype=float)
            if a0.shape != p1_norm.shape:
                raise ValueError("p0 must have the same shape as p1")
            s0 = float(a0.sum())
            if s0 <= 0:
                raise ValueError("p0 sum must be > 0")
            p0_norm = a0 / s0
        df = (R - 1) * (C - 1)

    elif kind == "multinomial":
        if a1.ndim != 1:
            raise ValueError("multinomial input must be 1-D")
        if p0 is None:
            raise ValueError("multinomial mode requires p0 (null proportions)")
        a0 = np.asarray(p0, dtype=float)
        if a0.shape != p1_norm.shape:
            raise ValueError("p0 must have the same shape as p1")
        s0 = float(a0.sum())
        if s0 <= 0:
            raise ValueError("p0 sum must be > 0")
        p0_norm = a0 / s0
        df = p1_norm.size - 1

    else:
        raise ValueError(f"unknown table_kind: {table_kind!r}")

    if np.any(p0_norm <= 0):
        raise ValueError("p0 must be strictly positive in every cell")

    w = math.sqrt(float(((p1_norm - p0_norm) ** 2 / p0_norm).sum()))

    chi_square_at_N: float | None = None
    N_total: int | None = None
    if is_counts:
        N_total = int(round(total))
        chi_square_at_N = float(N_total) * w * w

    return {
        "method_id": "chi_square_effect_size_estimator",
        "solve_for": "estimate",
        "result": w,
        "w": w,
        "df": df,
        "table_kind": kind,
        "p0": p0_norm.tolist(),
        "p1": p1_norm.tolist(),
        "chi_square_at_N": chi_square_at_N,
        "N": N_total,
        "inputs_echo": inputs_echo,
        "citations": [
            "Estimator.",
            "Cohen, J. (1988). Statistical Power Analysis for the "
            "Behavioral Sciences (2nd ed.). Hillsdale, NJ: Lawrence Erlbaum.",
        ],
    }


# ---------------------------------------------------------------------------
# 4. Kappa Estimator
# ---------------------------------------------------------------------------


def kappa_estimator(
    *,
    table: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Expected Cohen's κ from a square agreement matrix.

    ``table`` may contain counts or proportions; it is normalised by
    its total. The marginals are then used to compute

        P_O = Σ_i p_ii                       (observed agreement)
        P_E = Σ_i (row_i · col_i)            (chance agreement)
        κ   = (P_O - P_E) / (1 - P_E)

    Both Fleiss and Cohen large-sample SD(κ) approximations are
    returned alongside ``result`` = κ.
    """
    arr = np.asarray(table, dtype=float)
    inputs_echo = {"table": arr.tolist()}

    if arr.ndim != 2:
        raise ValueError("table must be 2-D")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("table must be square")
    if np.any(arr < 0):
        raise ValueError("table entries must be non-negative")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("table sum must be > 0")

    P = arr / total
    k = P.shape[0]
    if k < 2:
        raise ValueError("table must be at least 2x2")

    po = float(np.trace(P))
    row_marg = P.sum(axis=1)
    col_marg = P.sum(axis=0)
    pe = float((row_marg * col_marg).sum())
    if 1.0 - pe == 0.0:
        raise ValueError("PE = 1, kappa is undefined")
    kappa = (po - pe) / (1.0 - pe)

    # Large-sample SE approximations.
    # Use total as N when counts are supplied; otherwise leave N optional
    # so SDs are reported as None.
    is_counts = total > 1.5 + 1e-9
    N = int(round(total)) if is_counts else None

    fleiss_sd: float | None = None
    cohen_sd: float | None = None
    if N is not None and N > 1 and 0.0 < pe < 1.0:
        # Fleiss (1969) large-sample variance, fully expanded.
        # Var(κ) = [ A + B - C ] / [ N · (1 - P_E)² ]
        #   A = Σ_i p_ii · (1 - (row_i + col_i)(1 - κ))²
        #   B = (1-κ)² · Σ_{i≠j} p_ij · (col_i + row_j)²
        #   C = (κ - P_E (1 - κ))²
        A = 0.0
        for i in range(k):
            A += P[i, i] * (1.0 - (row_marg[i] + col_marg[i]) * (1.0 - kappa)) ** 2
        B = 0.0
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                B += P[i, j] * (col_marg[i] + row_marg[j]) ** 2
        B *= (1.0 - kappa) ** 2
        C = (kappa - pe * (1.0 - kappa)) ** 2
        var_fleiss = (A + B - C) / (N * (1.0 - pe) ** 2)
        if var_fleiss > 0:
            fleiss_sd = math.sqrt(var_fleiss)

        # Cohen's approximate SE (no off-diagonal cross terms).
        # Var(κ) ≈ [ P_O (1 - P_O) ] / [ N · (1 - P_E)² ]
        var_cohen = po * (1.0 - po) / (N * (1.0 - pe) ** 2)
        if var_cohen > 0:
            cohen_sd = math.sqrt(var_cohen)

    return {
        "method_id": "kappa_estimator",
        "solve_for": "estimate",
        "result": kappa,
        "kappa": kappa,
        "po": po,
        "pe": pe,
        "fleiss_sd_kappa": fleiss_sd,
        "cohen_sd_kappa": cohen_sd,
        "row_marginals": row_marg.tolist(),
        "col_marginals": col_marg.tolist(),
        "N": N,
        "k": k,
        "inputs_echo": inputs_echo,
        "citations": [
            "Fleiss, J. L. (1981). Statistical Methods for Rates and "
            "Proportions (2nd ed.). New York: Wiley.",
            "Cohen, J. (1960). A coefficient of agreement for nominal "
            "scales. Educ. Psychol. Meas., 20, 37-46.",
        ],
    }


# Re-exports for the registry.
__all__ = [
    "standard_deviation_estimator",
    "standard_deviation_of_means_calculator",
    "chi_square_effect_size_estimator",
    "kappa_estimator",
]
