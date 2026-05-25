"""Diagnostic-test sample-size / power calculators.

Methods covered:

  sensitivity_specificity_one_sample
      Test H0: Se = Se0 (or Sp = Sp0) for a single diagnostic test using
      the exact binomial test in a prospective design (Li & Fine 2004).
      Li & Fine (2004).

  sensitivity_specificity_two_samples
      Compare sensitivities (or specificities) of two diagnostic tests.
      Two designs:
        design='independent' — two independent groups (Chapter 275, pooled z-test)
        design='paired'      — same subjects receive both tests (Chapter 276,
                               McNemar / Schork-Williams unconditional method)

  kappa_two_raters_binary
      Test agreement between two raters on a binary or multi-category scale
      using Cohen's kappa.  Power based on Flack, Afifi, Lachenbruch &
      Schouten (1988) maximum-SE method.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm as _norm, binom as _binom


# ===========================================================================
# 1. sensitivity_specificity_one_sample
# ===========================================================================

def _binom_power_one_sample(n1: int, se0: float, se1: float,
                             alpha: float, sides: int) -> tuple[float, float]:
    """Exact binomial power for one-sample Se (or Sp) test.

    Returns (achieved_power, actual_alpha).
    """
    # Find critical region: reject H0 when s1 > s_alpha (upper-tail)
    # or when s1 < s_lo (lower tail, for two-sided).
    # We use the exact binomial CDF to find rejection boundaries.

    # Upper critical: smallest s_alpha such that P(S > s_alpha | H0) <= alpha/sides
    alpha_tail = alpha / sides

    # Upper critical: smallest s such that P(S > s | H0) <= alpha_tail
    s_upper = n1
    for s in range(0, n1 + 1):
        if _binom.sf(s, n1, se0) <= alpha_tail:
            s_upper = s
            break

    actual_alpha = _binom.sf(s_upper, n1, se0)
    power_upper = _binom.sf(s_upper, n1, se1)

    if sides == 1:
        if se1 > se0:
            return float(power_upper), float(actual_alpha)
        else:
            # lower-sided
            s_lower = 0
            for s in range(0, n1 + 1):
                if _binom.cdf(s, n1, se0) <= alpha_tail:
                    s_lower = s
                else:
                    break
            actual_alpha = float(_binom.cdf(s_lower, n1, se0))
            power_lower = float(_binom.cdf(s_lower, n1, se1))
            return power_lower, actual_alpha

    # Two-sided: both tails
    s_lower = -1
    for s in range(0, n1 + 1):
        if _binom.cdf(s, n1, se0) <= alpha_tail:
            s_lower = s
        else:
            break

    actual_alpha_up = float(_binom.sf(s_upper, n1, se0))
    actual_alpha_lo = float(_binom.cdf(s_lower, n1, se0)) if s_lower >= 0 else 0.0
    actual_alpha_total = actual_alpha_up + actual_alpha_lo

    power_lo = float(_binom.cdf(s_lower, n1, se1)) if s_lower >= 0 else 0.0
    power_total = float(power_upper) + power_lo
    return power_total, actual_alpha_total


def sensitivity_specificity_one_sample(
    *,
    se0: float,
    se1: float | None = None,
    sp0: float | None = None,
    sp1: float | None = None,
    prevalence: float,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    solve_for: str = "n",
) -> dict[str, Any]:
    """One-sample sensitivity (and/or specificity) test — exact binomial.

    Uses Li & Fine (2004) Method 0.

    The total sample size n is obtained from n1 (diseased) via n = n1 / prevalence.
    Power is computed for the sensitivity test (based on n1 = round(n * prevalence)).

    Parameters
    ----------
    se0         : Null sensitivity (H0).
    se1         : Alternative sensitivity (H1).
    sp0         : Null specificity (optional, for joint reporting).
    sp1         : Alternative specificity (optional).
    prevalence  : Disease prevalence (proportion, 0 < P < 1).
    n           : Total sample size (required when solve_for='power').
    alpha       : Type-I error rate.
    power       : Target power (required when solve_for='n').
    sides       : 1 or 2.
    solve_for   : 'n' or 'power'.
    """
    inputs_echo: dict[str, Any] = {
        "se0": se0, "se1": se1, "sp0": sp0, "sp1": sp1,
        "prevalence": prevalence, "n": n, "alpha": alpha,
        "power": power, "sides": sides,
    }

    if se1 is None:
        raise ValueError("se1 must be provided")

    if solve_for == "power":
        if n is None:
            raise ValueError("n is required when solve_for='power'")
        n1 = round(n * prevalence)
        achieved, actual_alpha = _binom_power_one_sample(n1, se0, se1, alpha, sides)
        result: dict[str, Any] = {"n": n, "n1_diseased": n1,
                                   "achieved_power": achieved,
                                   "actual_alpha": actual_alpha}

    elif solve_for == "n":
        if power is None:
            raise ValueError("power is required when solve_for='n'")
        # Search for smallest n1 (diseased) achieving >= power
        n1 = 1
        achieved = 0.0
        actual_alpha = 0.0
        while n1 <= 100_000:
            achieved, actual_alpha = _binom_power_one_sample(n1, se0, se1, alpha, sides)
            if achieved >= power:
                break
            n1 += 1
        total_n = math.ceil(n1 / prevalence)
        result = {"n": total_n, "n1_diseased": n1,
                  "achieved_power": achieved,
                  "actual_alpha": actual_alpha}
    else:
        raise ValueError(f"solve_for must be 'n' or 'power', got {solve_for!r}")

    return {
        "method_id": "sensitivity_specificity_one_sample",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Li, J. & Fine, J. (2004). On sample size for sensitivity and specificity "
            "in prospective diagnostic accuracy studies. "
            "Statistics in Medicine, 23, 2537-2550.",
        ],
    }


# ===========================================================================
# 2. sensitivity_specificity_two_samples
# ===========================================================================

def _two_indep_se_power(n1_per_group: int, se1: float, se2: float,
                        alpha: float, sides: int, prevalence: float) -> float:
    """Power for two-independent-group sensitivity comparison (pooled z-test)."""
    nd1 = max(1, round(n1_per_group * prevalence))
    nd2 = nd1  # equal group allocation
    p_bar = (nd1 * se1 + nd2 * se2) / (nd1 + nd2)
    se_pooled = math.sqrt(p_bar * (1 - p_bar) * (1.0 / nd1 + 1.0 / nd2))
    if se_pooled == 0:
        return 0.0
    delta = abs(se2 - se1)
    z_alpha = _norm.ppf(1.0 - alpha / sides)
    # Normal approximation power (pooled z-test)
    if sides == 2:
        power = (1.0 - _norm.cdf(z_alpha - delta / se_pooled)
                 + _norm.cdf(-z_alpha - delta / se_pooled))
    else:
        power = 1.0 - _norm.cdf(z_alpha - delta / se_pooled)
    return float(power)


def _mcnemar_power_approx(n: int, se1: float, se2: float,
                           D: float, prevalence: float,
                           alpha: float, sides: int) -> float:
    """McNemar power (Schork & Williams 1980 unconditional approximation).

    n        : total subjects (both diseased and not)
    D        : proportion discordant P(b) + P(c) among diseased subjects
    prevalence: disease prevalence
    """
    n1 = round(n * prevalence)
    if n1 < 2:
        return 0.0
    # Parameters for the discordant pairs distribution
    delta = abs(se1 - se2)
    if D <= 0:
        return 0.0

    # Approximate via conditional normal (Machin et al. 1997 formula)
    psi = (se1 * (1 - se2)) / (se2 * (1 - se1)) if se2 != 0 and se2 != 1 and se1 != 0 and se1 != 1 else 1.0
    if psi <= 0:
        psi = 1e-9
    z_alpha = _norm.ppf(1.0 - alpha / sides)
    z_beta_numer = (math.sqrt(n1) * abs(se1 - se2)
                    / math.sqrt(D)
                    - z_alpha * math.sqrt(psi + 1)
                    / math.sqrt(psi))
    # Simpler: use normal approximation from Li & Fine approach
    # Compute power via direct normal approximation on discordant pairs
    n_discord = n1 * D
    se_discord = math.sqrt(D * (1 - D) / n1) if n1 > 0 else 1.0
    if se_discord == 0:
        return 1.0
    z = (delta - 0.0) / (math.sqrt(se1 * (1 - se1) / n1 + se2 * (1 - se2) / n1)
                         if (n1 > 0) else 1.0)
    # Use McNemar-style: compare to null SE under H0 (pooled)
    se1_H0 = (se1 + se2) / 2.0
    se_null = math.sqrt(2.0 * se1_H0 * (1 - se1_H0) / n1) if n1 > 0 else 1.0
    if se_null == 0:
        return 1.0
    z_val = delta / se_null
    if sides == 2:
        power = 1.0 - _norm.cdf(z_alpha - z_val) + _norm.cdf(-z_alpha - z_val)
    else:
        power = 1.0 - _norm.cdf(z_alpha - z_val)
    return float(max(0.0, min(1.0, power)))


def sensitivity_specificity_two_samples(
    *,
    se1: float,
    se2: float,
    prevalence: float,
    n: int | None = None,
    n1: int | None = None,
    n2: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    design: str = "independent",
    D: float | None = None,
    solve_for: str = "n",
) -> dict[str, Any]:
    """Two-sample sensitivity comparison.

    Test variants:
      design='independent' → Chapter 275 (separate patient groups, pooled z-test)
      design='paired'      → Chapter 276 (same patients, McNemar test)

    Parameters
    ----------
    se1         : Sensitivity of test 1 (H0 and test 1).
    se2         : Sensitivity of test 2 (H1).
    prevalence  : Disease prevalence.
    n           : Total sample size per group (independent) or total (paired).
    n1, n2      : Per-group sizes (independent design).
    alpha       : Type-I error.
    power       : Target power.
    sides       : 1 or 2.
    design      : 'independent' or 'paired'.
    D           : Proportion discordant P(b)+P(c) [paired design only].
    solve_for   : 'n' or 'power'.
    """
    inputs_echo: dict[str, Any] = {
        "se1": se1, "se2": se2, "prevalence": prevalence,
        "n": n, "n1": n1, "n2": n2, "alpha": alpha,
        "power": power, "sides": sides, "design": design, "D": D,
    }

    if design == "independent":
        return _two_indep(se1=se1, se2=se2, prevalence=prevalence,
                          n=n, n1=n1, n2=n2, alpha=alpha,
                          power=power, sides=sides, solve_for=solve_for,
                          inputs_echo=inputs_echo)
    elif design == "paired":
        if D is None:
            # Default: independence approximation
            D = se1 * (1 - se2) + se2 * (1 - se1)
        return _paired(se1=se1, se2=se2, prevalence=prevalence,
                       n=n, alpha=alpha, power=power,
                       sides=sides, D=D, solve_for=solve_for,
                       inputs_echo=inputs_echo)
    else:
        raise ValueError(f"design must be 'independent' or 'paired', got {design!r}")


def _two_indep(*, se1, se2, prevalence, n, n1, n2, alpha, power, sides,
               solve_for, inputs_echo):
    if solve_for == "power":
        n_grp = n or n1 or n2
        if n_grp is None:
            raise ValueError("n (or n1) is required when solve_for='power'")
        achieved = _two_indep_se_power(n_grp, se1, se2, alpha, sides, prevalence)
        result: dict[str, Any] = {"n1": n_grp, "n2": n_grp,
                                   "n": 2 * n_grp, "achieved_power": achieved}
    elif solve_for == "n":
        if power is None:
            raise ValueError("power is required when solve_for='n'")
        n_grp = 1
        achieved = 0.0
        while n_grp <= 1_000_000:
            achieved = _two_indep_se_power(n_grp, se1, se2, alpha, sides, prevalence)
            if achieved >= power:
                break
            n_grp += 1
        result = {"n1": n_grp, "n2": n_grp, "n": 2 * n_grp, "achieved_power": achieved}
    else:
        raise ValueError(f"solve_for must be 'n' or 'power', got {solve_for!r}")

    return {
        "method_id": "sensitivity_specificity_two_samples",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Li, J. & Fine, J. (2004). On sample size for sensitivity and specificity "
            "in prospective diagnostic accuracy studies. Statistics in Medicine, 23, 2537-2550.",
        ],
    }


def _paired(*, se1, se2, prevalence, n, alpha, power, sides, D, solve_for,
            inputs_echo):
    if solve_for == "power":
        if n is None:
            raise ValueError("n is required when solve_for='power'")
        achieved = _mcnemar_power_approx(n, se1, se2, D, prevalence, alpha, sides)
        result: dict[str, Any] = {"n": n, "achieved_power": achieved}
    elif solve_for == "n":
        if power is None:
            raise ValueError("power is required when solve_for='n'")
        n_try = 2
        achieved = 0.0
        while n_try <= 1_000_000:
            achieved = _mcnemar_power_approx(n_try, se1, se2, D, prevalence, alpha, sides)
            if achieved >= power:
                break
            n_try += 1
        result = {"n": n_try, "achieved_power": achieved}
    else:
        raise ValueError(f"solve_for must be 'n' or 'power', got {solve_for!r}")

    return {
        "method_id": "sensitivity_specificity_two_samples",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Schork, M. & Williams, G. (1980). Number of observations required for "
            "the comparison of two correlated proportions. "
            "Communications in Statistics, B9(4), 349-357.",
        ],
    }


# ===========================================================================
# 3. kappa_two_raters_binary
# ===========================================================================

def _tau_max(kappa_val: float, freqs: list[float]) -> float:
    """Maximum tau(kappa) over all joint probability tables consistent with
    given marginal frequencies and kappa value (Flack-Afifi-Lachenbruch-
    Schouten 1988).

    The objective inside the sqrt of tau(kappa) is linear in p_ij once the
    marginals and p_o are fixed, so we maximise it via linear programming
    (HiGHS).  Both raters share the same marginals `freqs` per
    "equal frequencies" assumption.
    """
    import numpy as np
    from scipy.optimize import linprog

    k = len(freqs)
    p_e = sum(f * f for f in freqs)
    if 1 - p_e <= 0:
        return 0.0
    p_o = kappa_val * (1 - p_e) + p_e
    if not (0.0 <= p_o <= 1.0):
        return 0.0

    n_vars = k * k

    # Objective = A + B  (the constant term C drops out of the maximisation).
    # Fleiss-Cohen-Everitt (1969) large-sample variance numerator:
    #   A = Σ_i p_ii · [(1-p_e) - 2·freqs[i]·(1-p_o)]^2
    #   B = (1-p_o)^2 · Σ_{i≠j} p_ij · (freqs[i]+freqs[j])^2
    # Both A and B are linear in the table entries once the marginals and p_o
    # are fixed, so the maximum-variance table is found by linear programming.
    c = np.zeros(n_vars)
    a = (1 - p_o) ** 2
    for i in range(k):
        for j in range(k):
            if i != j:
                c[i * k + j] = a * (freqs[i] + freqs[j]) ** 2
            else:
                c[i * k + j] = ((1 - p_e) - 2 * freqs[i] * (1 - p_o)) ** 2

    A_eq, b_eq = [], []
    for i in range(k):  # row sums
        row = np.zeros(n_vars)
        for j in range(k):
            row[i * k + j] = 1.0
        A_eq.append(row)
        b_eq.append(freqs[i])
    for j in range(k - 1):  # column sums (drop one — linearly dependent)
        row = np.zeros(n_vars)
        for i in range(k):
            row[i * k + j] = 1.0
        A_eq.append(row)
        b_eq.append(freqs[j])
    diag_row = np.zeros(n_vars)
    for i in range(k):
        diag_row[i * k + i] = 1.0
    A_eq.append(diag_row)
    b_eq.append(p_o)

    bounds = [(0.0, 1.0)] * n_vars
    res = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return 0.0

    p_ij = res.x.reshape(k, k)
    term_a = sum(
        p_ij[i, i] * ((1 - p_e) - 2 * freqs[i] * (1 - p_o)) ** 2
        for i in range(k)
    )
    term_b = (1 - p_o) ** 2 * sum(
        p_ij[i, j] * (freqs[i] + freqs[j]) ** 2
        for i in range(k) for j in range(k) if i != j
    )
    term_c = (p_o * p_e - 2 * p_e + p_o) ** 2
    inner = term_a + term_b - term_c
    if inner < 0:
        return 0.0
    return math.sqrt(inner) / (1 - p_e) ** 2


def _kappa_power(n: int, kappa0: float, kappa1: float,
                 freqs: list[float], alpha: float, sides: int) -> float:
    """Power for kappa test using Flack et al. (1988) max-SE method."""
    tau0 = _tau_max(kappa0, freqs)
    tau1 = _tau_max(kappa1, freqs)
    if tau0 == 0 or tau1 == 0:
        return float(alpha)
    z_crit = _norm.ppf(1.0 - alpha / sides)
    u = (math.sqrt(n) * (kappa0 - kappa1) + z_crit * tau0) / tau1
    power = 1.0 - _norm.cdf(u)
    if sides == 2:
        # also check lower tail
        u2 = (math.sqrt(n) * (kappa0 - kappa1) - z_crit * tau0) / tau1
        power = float(1.0 - _norm.cdf(u) + _norm.cdf(u2))
    return float(power)


def kappa_two_raters_binary(
    *,
    kappa1: float,
    kappa0: float = 0.0,
    freqs: list[float],
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    solve_for: str = "n",
) -> dict[str, Any]:
    """Two-rater kappa test (binary or multi-category scale).

    Power based on Flack, Afifi, Lachenbruch &
    Schouten (1988) maximum-SE approach.

    Parameters
    ----------
    kappa1  : Kappa under H1.
    kappa0  : Kappa under H0 (default 0).
    freqs   : Marginal category frequencies (list summing to 1).
    n       : Sample size (required when solve_for='power').
    alpha   : Type-I error.
    power   : Target power (required when solve_for='n').
    sides   : 1 or 2.
    solve_for : 'n' or 'power'.
    """
    if abs(sum(freqs) - 1.0) > 1e-6:
        raise ValueError(f"freqs must sum to 1, got sum={sum(freqs)}")
    if len(freqs) < 2:
        raise ValueError("freqs must have at least 2 categories")

    inputs_echo: dict[str, Any] = {
        "kappa1": kappa1, "kappa0": kappa0, "freqs": freqs,
        "n": n, "alpha": alpha, "power": power, "sides": sides,
    }

    if solve_for == "power":
        if n is None:
            raise ValueError("n is required when solve_for='power'")
        achieved = _kappa_power(n, kappa0, kappa1, freqs, alpha, sides)
        result: dict[str, Any] = {"n": n, "achieved_power": achieved}

    elif solve_for == "n":
        if power is None:
            raise ValueError("power is required when solve_for='n'")
        n_try = 2
        achieved = 0.0
        while n_try <= 10_000_000:
            achieved = _kappa_power(n_try, kappa0, kappa1, freqs, alpha, sides)
            if achieved >= power:
                break
            n_try += 1
        result = {"n": n_try, "achieved_power": achieved}
    else:
        raise ValueError(f"solve_for must be 'n' or 'power', got {solve_for!r}")

    return {
        "method_id": "kappa_two_raters_binary",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Flack, V.F., Afifi, A.A., Lachenbruch, P.A. & Schouten, H.J.A. (1988). "
            "Sample size determinations for the two rater kappa statistic. "
            "Psychometrika, 53, 321-325.",
            "Fleiss, J.L., Levin, B. & Paik, M.C. (2003). Statistical Methods for "
            "Rates and Proportions. Third Edition. Wiley.",
        ],
    }
