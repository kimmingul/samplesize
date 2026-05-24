"""Multiple one-sample / paired and two-sample t-tests with multiplicity
control.

- "Multiple One-Sample or Paired T-Tests" (Chapter 610)
- "Multiple Two-Sample T-Tests" (Chapter 615)

Both chapters describe family-wise control of the per-test alpha when
``k_tests`` tests are run simultaneously.  Two adjustment families are
supported:

EWER (Experiment-Wise Error Rate, Bonferroni-style)
    ``alpha_adj = family_alpha / k_tests``

FDR (Benjamini & Hochberg 1995 / Jung 2005, Chow Shao & Wang 2008)
    ``alpha_adj = K * (1 - beta) * fdr / ((k_tests - K) * (1 - fdr))``
    Solved iteratively because ``alpha_adj`` depends on ``beta``.

For each fixed per-test alpha the individual power is the standard
one-sample (or paired) / two-sample t/z power.  "Complete power" is the
probability of declaring all K truly-different tests significant
simultaneously (Prob to Detect All K = power**K under independence).

This module exposes two top-level callables wired into the registry:

* ``multiple_one_sample_or_paired_t``
* ``multiple_two_sample_t``

The ``adjustment`` parameter accepts ``"bonferroni"`` (alias EWER),
``"fdr"`` (Benjamini-Hochberg via Jung), and ``"none"`` (no
adjustment).  ``"holm"`` and ``"hochberg"`` are accepted names but the
step-down/step-up procedures; we therefore fall back to the Bonferroni
single-test alpha which is a conservative lower bound for the power of
Holm and Hochberg (both are uniformly at least as powerful as
Bonferroni).
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D

# ---------------------------------------------------------------------------
# Per-test power kernels
# ---------------------------------------------------------------------------


def _safe_nct_lower(x: float, df: float, ncp: float) -> float:
    """``nct.cdf(-x, df, ncp)`` clamped to 0 when scipy returns NaN.

    For large positive ``ncp`` (large effect) the left-tail probability
    is exponentially small; scipy's series fails and returns NaN where
    the true value is essentially zero.
    """
    val = D.nct_cdf(-x, df, ncp)
    if math.isnan(val):
        return 0.0
    return val


def _power_t_one(D_eff: float, S: float, n: int,
                 alpha: float, sides: int) -> float:
    """One-sample / paired t-test power for mean-diff = ``D_eff``."""
    if n < 2:
        return 0.0
    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return 1.0
    df = n - 1
    se = S / math.sqrt(n)
    ncp = D_eff / se
    if sides == 2:
        t_crit = D.t_ppf(1 - alpha / 2.0, df)
        upper = 1.0 - D.nct_cdf(t_crit, df, ncp)
        if math.isnan(upper):
            upper = 0.0
        lower = _safe_nct_lower(t_crit, df, ncp)
        return upper + lower
    if sides == 1:
        t_crit = D.t_ppf(1 - alpha, df)
        if D_eff >= 0:
            val = 1.0 - D.nct_cdf(t_crit, df, ncp)
            return 0.0 if math.isnan(val) else val
        return _safe_nct_lower(t_crit, df, -ncp)  # mirror
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def _power_z_one(D_eff: float, S: float, n: int,
                 alpha: float, sides: int) -> float:
    """One-sample / paired z-test power (SD known)."""
    from scipy.stats import norm
    if n < 1:
        return 0.0
    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return 1.0
    se = S / math.sqrt(n)
    ncp = D_eff / se
    if sides == 2:
        z_crit = D.norm_ppf(1 - alpha / 2.0)
        return float((1.0 - norm.cdf(z_crit - ncp))
                     + norm.cdf(-z_crit - ncp))
    if sides == 1:
        z_crit = D.norm_ppf(1 - alpha)
        if D_eff >= 0:
            return float(1.0 - norm.cdf(z_crit - ncp))
        return float(norm.cdf(-z_crit - ncp))
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def _power_t_two(D_eff: float, S1: float, S2: float, n1: int, n2: int,
                 alpha: float, sides: int, equal_var: bool) -> float:
    """Two-sample t-test power.  ``equal_var=True`` uses pooled t with
    ``df = n1+n2-2``.  ``equal_var=False`` uses Welch with the
    adjusted-df expression (case 4) minus 2.
    """
    if n1 < 2 or n2 < 2:
        return 0.0
    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return 1.0
    if equal_var:
        sigma_xbar = S1 * math.sqrt(1.0 / n1 + 1.0 / n2)
        df = n1 + n2 - 2
    else:
        v1 = (S1 ** 2) / n1
        v2 = (S2 ** 2) / n2
        sigma_xbar = math.sqrt(v1 + v2)
        denom = (S1 ** 4) / (n1 ** 2 * (n1 + 1)) + \
                (S2 ** 4) / (n2 ** 2 * (n2 + 1))
        df = (sigma_xbar ** 4) / denom - 2.0
        df = max(1.0, df)
    ncp = D_eff / sigma_xbar
    if sides == 2:
        t_crit = D.t_ppf(1 - alpha / 2.0, df)
        upper = 1.0 - D.nct_cdf(t_crit, df, ncp)
        if math.isnan(upper):
            upper = 0.0
        lower = _safe_nct_lower(t_crit, df, ncp)
        return upper + lower
    if sides == 1:
        t_crit = D.t_ppf(1 - alpha, df)
        if D_eff >= 0:
            val = 1.0 - D.nct_cdf(t_crit, df, ncp)
            return 0.0 if math.isnan(val) else val
        return _safe_nct_lower(t_crit, df, -ncp)
    raise ValueError(f"sides must be 1 or 2, got {sides}")


def _power_z_two(D_eff: float, S1: float, S2: float, n1: int, n2: int,
                 alpha: float, sides: int) -> float:
    from scipy.stats import norm
    if n1 < 1 or n2 < 1:
        return 0.0
    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return 1.0
    se = math.sqrt(S1 ** 2 / n1 + S2 ** 2 / n2)
    ncp = D_eff / se
    if sides == 2:
        z_crit = D.norm_ppf(1 - alpha / 2.0)
        return float((1.0 - norm.cdf(z_crit - ncp))
                     + norm.cdf(-z_crit - ncp))
    if sides == 1:
        z_crit = D.norm_ppf(1 - alpha)
        if D_eff >= 0:
            return float(1.0 - norm.cdf(z_crit - ncp))
        return float(norm.cdf(-z_crit - ncp))
    raise ValueError(f"sides must be 1 or 2, got {sides}")


# ---------------------------------------------------------------------------
# Adjusted-alpha solver
# ---------------------------------------------------------------------------


def _alpha_adj_ewer(family_alpha: float, k_tests: int) -> float:
    if k_tests < 1:
        raise ValueError("k_tests must be >= 1")
    return family_alpha / k_tests


def _alpha_adj_fdr(beta: float, *, K: int, k_tests: int, fdr: float) -> float:
    """Jung (2005) / Chow Shao Wang (2008) FDR-adjusted single-test alpha."""
    if K < 1 or K >= k_tests:
        raise ValueError("K must satisfy 1 <= K < k_tests")
    if not 0.0 < fdr < 1.0:
        raise ValueError("fdr (family_alpha) must be in (0, 1) for FDR")
    return K * (1.0 - beta) * fdr / ((k_tests - K) * (1.0 - fdr))


def _resolve_alpha_and_power(
    power_kernel,
    *,
    adjustment: str,
    family_alpha: float,
    k_tests: int,
    K: int | None,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> tuple[float, float]:
    """Return ``(individual_power, alpha_adj)`` for a single-test power
    kernel ``power_kernel(alpha) -> power``.

    For EWER / Bonferroni the alpha is fixed and a single evaluation
    suffices.  For FDR the alpha depends on beta = 1 - power and we
    iterate until convergence.
    """
    adj = adjustment.lower()
    if adj in ("none", "independent", "pcer"):
        alpha = family_alpha
        return power_kernel(alpha), alpha
    if adj in ("bonferroni", "ewer", "holm", "hochberg"):
        alpha = _alpha_adj_ewer(family_alpha, k_tests)
        return power_kernel(alpha), alpha
    if adj in ("fdr", "benjamini-hochberg", "bh"):
        if K is None:
            raise ValueError("FDR adjustment requires K (number of true H1)")
        # Fixed-point iteration on beta.
        beta = 0.5
        power = 0.5
        for _ in range(max_iter):
            alpha = _alpha_adj_fdr(beta, K=K, k_tests=k_tests, fdr=family_alpha)
            alpha = min(max(alpha, 0.0), 1.0)
            power = power_kernel(alpha)
            new_beta = 1.0 - power
            if abs(new_beta - beta) < tol:
                beta = new_beta
                break
            beta = new_beta
        alpha = _alpha_adj_fdr(beta, K=K, k_tests=k_tests, fdr=family_alpha)
        return power, alpha
    raise ValueError(
        f"unknown adjustment: {adjustment!r} "
        "(expected bonferroni / ewer / fdr / none / holm / hochberg)"
    )


# ---------------------------------------------------------------------------
# Public callables
# ---------------------------------------------------------------------------


def _individual_to_complete(power: float, K: int) -> float:
    """Probability of detecting all K truly-different tests
    (assuming independence)."""
    if K <= 0:
        return float("nan")
    return power ** K


def _individual_to_any(power: float, K: int) -> float:
    """Probability of detecting at least one truly-different test."""
    if K <= 0:
        return float("nan")
    return 1.0 - (1.0 - power) ** K


def _bracket_n(power_at, target_power: float, n_min: int, n_max: int) -> int:
    """Return smallest integer n in [n_min, n_max] with power_at(n) >= target."""
    lo = n_min
    hi = n_min
    while hi <= n_max:
        if power_at(hi) >= target_power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if power_at(mid) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi


# ---- One-sample / paired ---------------------------------------------------


def multiple_one_sample_or_paired_t(
    *,
    k_tests: int,
    mean_diff: float,
    sd: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    adjustment: str = "bonferroni",
    K: int | None = None,
    sd_known: bool = False,
    power_definition: str = "individual",
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 1_000_000,
) -> dict[str, Any]:
    """Multiple one-sample or paired t-tests power / sample-size.

    Parameters
    ----------
    k_tests : int
        Total number of simultaneous tests (e.g. genes, m).
    mean_diff : float
        Minimum mean difference to be detected per test.
    sd : float
        Standard deviation of paired differences.
    alpha : float
        Family-wise alpha (EWER) or false-discovery rate (FDR) target,
        depending on ``adjustment``.
    power : float or None
        Target individual-test power (1 - beta).
    n : int or None
        Sample size (number of arrays / replicates).
    sides : int
        1 (one-sided) or 2 (two-sided).
    adjustment : str
        ``"bonferroni"`` / ``"ewer"`` (alpha/k), ``"fdr"`` (Jung 2005),
        ``"none"`` (no adjustment).  ``"holm"`` / ``"hochberg"`` reduce
        to Bonferroni for the *expected* power calculation.
    K : int or None
        Number of tests with true mean difference > D (required for FDR).
    sd_known : bool
        If True, use the z-test variant.
    power_definition : str
        ``"individual"`` (default), ``"complete"`` (Prob to
        Detect All K), or ``"any"`` (probability of detecting at least
        one truly-different test).
    solve_for : str or None
        ``"n"`` or ``"power"``.  Inferred if exactly one of (n, power) is
        None.
    """
    if k_tests < 2:
        raise ValueError("k_tests must be >= 2 for multiplicity control")
    if mean_diff == 0:
        raise ValueError("mean_diff must be nonzero")
    if sd <= 0:
        raise ValueError("sd must be positive")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if power_definition not in ("individual", "complete", "any"):
        raise ValueError(
            "power_definition must be 'individual', 'complete', or 'any'"
        )

    inputs_echo = {
        "k_tests": k_tests, "mean_diff": mean_diff, "sd": sd,
        "alpha": alpha, "power": power, "n": n, "sides": sides,
        "adjustment": adjustment, "K": K, "sd_known": sd_known,
        "power_definition": power_definition,
    }

    given = sum(x is not None for x in (power, n))
    if given < 1:
        raise ValueError("supply at least one of (power, n)")
    if solve_for is None:
        if n is None:
            solve_for = "n"
        elif power is None:
            solve_for = "power"
        else:
            raise ValueError(
                "both power and n provided; set one to None or specify solve_for"
            )

    def per_test_power_at(n_val: int) -> tuple[float, float]:
        """Return (individual_power, alpha_adj) for given n."""
        if sd_known:
            kernel = lambda a: _power_z_one(mean_diff, sd, n_val, a, sides)
        else:
            kernel = lambda a: _power_t_one(mean_diff, sd, n_val, a, sides)
        return _resolve_alpha_and_power(
            kernel, adjustment=adjustment, family_alpha=alpha,
            k_tests=k_tests, K=K,
        )

    def report_power(individual: float) -> float:
        if power_definition == "individual":
            return individual
        if K is None:
            raise ValueError(
                f"power_definition={power_definition!r} requires K"
            )
        if power_definition == "complete":
            return _individual_to_complete(individual, K)
        return _individual_to_any(individual, K)

    if solve_for == "power":
        assert n is not None
        individual, alpha_adj = per_test_power_at(int(n))
        reported = report_power(individual)
        result = {
            "n": int(n),
            "achieved_power": reported,
            "individual_power": individual,
            "alpha_adj": alpha_adj,
        }
        if K is not None:
            result["prob_detect_all_K"] = _individual_to_complete(individual, K)

    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")

        def power_at(n_val: int) -> float:
            individual, _ = per_test_power_at(n_val)
            return report_power(individual)

        n_solved = _bracket_n(power_at, power, n_min, n_max)
        individual, alpha_adj = per_test_power_at(n_solved)
        reported = report_power(individual)
        result = {
            "n": n_solved,
            "achieved_power": reported,
            "individual_power": individual,
            "alpha_adj": alpha_adj,
        }
        if K is not None:
            result["prob_detect_all_K"] = _individual_to_complete(individual, K)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "multiple_one_sample_or_paired_t",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Jung, S.-H. (2005). Sample size for FDR-control in microarray "
            "data analysis. Bioinformatics 21(14): 3097-3104.",
            "Benjamini, Y. & Hochberg, Y. (1995). Controlling the false "
            "discovery rate. JRSS-B 57(1): 289-300.",
            "Chow, S.-C., Shao, J. & Wang, H. (2008). Sample Size "
            "Calculations in Clinical Research, 2e.",
        ],
    }


# ---- Two-sample ------------------------------------------------------------


def multiple_two_sample_t(
    *,
    k_tests: int,
    mean_diff: float,
    sd1: float,
    sd2: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    sides: int = 2,
    adjustment: str = "bonferroni",
    K: int | None = None,
    sd_known: bool = False,
    allocation: float = 1.0,
    power_definition: str = "individual",
    solve_for: str | None = None,
    n_min: int = 2,
    n_max: int = 1_000_000,
) -> dict[str, Any]:
    """Multiple two-sample t-tests power / sample-size.

    Parameters
    ----------
    k_tests : int
        Total number of tests.
    mean_diff : float
        Minimum detectable mean1 - mean2.
    sd1, sd2 : float
        Group standard deviations.  ``sd2 is None`` → equal-variance T.
        If ``sd2 == sd1`` the unequal-variance Welch
        formula (Case 4) is still used; equal-variance pooling is used ONLY when
        sd2 is None / omitted.
    alpha : float
        Family-wise alpha (EWER) or FDR target.
    power, n1, n2 : float / int / None
        Provide at most one of ``power`` / sample size to be solved.
        ``n2`` defaults to ``ceil(allocation * n1)``.
    sides : int
        1 or 2.
    adjustment : str
        See ``multiple_one_sample_or_paired_t``.
    K : int or None
        Required for FDR.
    sd_known : bool
        If True, use Z-test branch.
    allocation : float
        n2 / n1 ratio when solving for n.
    power_definition : str
        ``"individual"`` / ``"complete"`` / ``"any"``.
    """
    if k_tests < 2:
        raise ValueError("k_tests must be >= 2 for multiplicity control")
    if mean_diff == 0:
        raise ValueError("mean_diff must be nonzero")
    if sd1 <= 0:
        raise ValueError("sd1 must be positive")
    if sd2 is not None and sd2 <= 0:
        raise ValueError("sd2 must be positive")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if allocation <= 0:
        raise ValueError("allocation must be > 0")
    if power_definition not in ("individual", "complete", "any"):
        raise ValueError(
            "power_definition must be 'individual', 'complete', or 'any'"
        )

    equal_var = sd2 is None
    sd2_eff = sd1 if sd2 is None else sd2

    inputs_echo = {
        "k_tests": k_tests, "mean_diff": mean_diff, "sd1": sd1, "sd2": sd2,
        "alpha": alpha, "power": power, "n1": n1, "n2": n2, "sides": sides,
        "adjustment": adjustment, "K": K, "sd_known": sd_known,
        "allocation": allocation, "power_definition": power_definition,
    }

    if n1 is not None and n2 is None:
        n2 = max(2, math.ceil(allocation * n1))
    elif n2 is not None and n1 is None:
        n1 = max(2, math.ceil(n2 / allocation))

    have_n = n1 is not None and n2 is not None
    have_power = power is not None

    if not have_n and not have_power:
        raise ValueError("supply at least one of (power, n1)")

    if solve_for is None:
        if not have_n:
            solve_for = "n"
        elif not have_power:
            solve_for = "power"
        else:
            raise ValueError(
                "both power and n supplied; specify solve_for explicitly"
            )

    def per_test_power_at(n1_val: int, n2_val: int) -> tuple[float, float]:
        if sd_known:
            kernel = lambda a: _power_z_two(
                mean_diff, sd1, sd2_eff, n1_val, n2_val, a, sides
            )
        else:
            kernel = lambda a: _power_t_two(
                mean_diff, sd1, sd2_eff, n1_val, n2_val, a, sides, equal_var
            )
        return _resolve_alpha_and_power(
            kernel, adjustment=adjustment, family_alpha=alpha,
            k_tests=k_tests, K=K,
        )

    def report_power(individual: float) -> float:
        if power_definition == "individual":
            return individual
        if K is None:
            raise ValueError(
                f"power_definition={power_definition!r} requires K"
            )
        if power_definition == "complete":
            return _individual_to_complete(individual, K)
        return _individual_to_any(individual, K)

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        individual, alpha_adj = per_test_power_at(int(n1), int(n2))
        reported = report_power(individual)
        result = {
            "n1": int(n1), "n2": int(n2), "n": int(n1) + int(n2),
            "achieved_power": reported,
            "individual_power": individual,
            "alpha_adj": alpha_adj,
        }
        if K is not None:
            result["prob_detect_all_K"] = _individual_to_complete(individual, K)
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")

        def power_at(n1_val: int) -> float:
            n2_val = max(2, math.ceil(allocation * n1_val))
            individual, _ = per_test_power_at(n1_val, n2_val)
            return report_power(individual)

        n1_solved = _bracket_n(power_at, power, n_min, n_max)
        n2_solved = max(2, math.ceil(allocation * n1_solved))
        individual, alpha_adj = per_test_power_at(n1_solved, n2_solved)
        reported = report_power(individual)
        result = {
            "n1": n1_solved, "n2": n2_solved, "n": n1_solved + n2_solved,
            "achieved_power": reported,
            "individual_power": individual,
            "alpha_adj": alpha_adj,
        }
        if K is not None:
            result["prob_detect_all_K"] = _individual_to_complete(individual, K)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "multiple_two_sample_t",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Jung, S.-H. (2005). Sample size for FDR-control in microarray "
            "data analysis. Bioinformatics 21(14): 3097-3104.",
            "Benjamini, Y. & Hochberg, Y. (1995). Controlling the false "
            "discovery rate. JRSS-B 57(1): 289-300.",
            "Chow, S.-C., Shao, J. & Wang, H. (2008). Sample Size "
            "Calculations in Clinical Research, 2e.",
        ],
    }
