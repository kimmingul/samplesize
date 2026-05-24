"""ROC curve sample-size / power calculators.

Three methods are covered:

  roc_auc_one_sample
      Test H0: AUC = AUC0 vs H1: AUC = AUC1 for a single diagnostic test.
      Variance formula: Obuchowski & McClish (1997) for discrete (rating)
      data; Hanley & McNeil (1983) for continuous data.
      Hanley & McNeil (1982).

  roc_auc_two_independent_samples
      Compare AUCs of two diagnostic tests applied to INDEPENDENT groups
      (correlation r+ = r- = 0).  Uses Obuchowski & McClish (1997) for
      discrete data; Hanley & McNeil (1983) for continuous data.
      Hanley & McNeil (1983).

  roc_auc_two_correlated_samples
      Compare AUCs of two diagnostic tests applied to the SAME subjects
      (paired / correlated design).  Requires r+ and r- (positive-group
      and negative-group correlations between the two test scores).
      Uses Obuchowski & McClish (1997) for discrete data; Hanley & McNeil
      (1983) for continuous data.
      Hanley & McNeil (1983).
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm as _norm


# ---------------------------------------------------------------------------
# Variance helpers
# ---------------------------------------------------------------------------

def _obuchowski_var(theta: float, R: float, B: float,
                   fpr1: float = 0.0, fpr2: float = 1.0) -> float:
    """Variance of theta-hat (discrete/rating data) per Obuchowski 1997.

    theta : AUC (full or partial, as AUC' after rescaling)
    R     : N- / N+
    B     : SD- / SD+  (set B=1 for conservative)
    fpr1, fpr2 : FPR limits (0, 1 for full AUC)
    """
    A = math.sqrt(1.0 + B * B) * _norm.ppf(theta)
    E1 = math.exp(-A * A / (2.0 * (1.0 + B * B)))
    E2 = 1.0 + B * B
    # integration limits in the z-scale
    c1 = _norm.ppf(fpr1) if fpr1 > 0 else -math.inf
    c2 = _norm.ppf(fpr2) if fpr2 < 1 else math.inf
    E3 = _norm.cdf(c2) - _norm.cdf(c1)
    # E4: (phi(c1) - phi(c2))
    phi_c1 = math.exp(-0.5 * c1 * c1) / math.sqrt(2 * math.pi) if math.isfinite(c1) else 0.0
    phi_c2 = math.exp(-0.5 * c2 * c2) / math.sqrt(2 * math.pi) if math.isfinite(c2) else 0.0
    E4 = phi_c1 - phi_c2

    f = E1 * E3 / math.sqrt(2 * math.pi * E2)
    g = (E1 * E4 / math.sqrt(2 * math.pi * E2)
         - A * B * E1 * E3 / math.sqrt(2 * math.pi * E2 ** 3))

    var = f * f * (1.0 + B * B / R + A * A / 2.0) + g * g * B * B * (1.0 + R) / (2.0 * R)
    return var


def _hanley_var(theta: float, R: float) -> float:
    """Variance of theta-hat (continuous data) per Hanley & McNeil (1983)."""
    return (theta / (R * (2.0 - theta))
            + 2.0 * theta ** 2 / (1.0 + theta)
            - theta ** 2 * (1.0 + R) / R)


def _obuchowski_cov(theta1: float, theta2: float, R: float,
                    B1: float, B2: float, r_pos: float, r_neg: float,
                    fpr1: float = 0.0, fpr2: float = 1.0) -> float:
    """Covariance between two theta-hats (discrete/rating, correlated design)."""
    A1 = math.sqrt(1.0 + B1 * B1) * _norm.ppf(theta1)
    A2 = math.sqrt(1.0 + B2 * B2) * _norm.ppf(theta2)

    def _ef(A, B, fpr1=fpr1, fpr2=fpr2):
        E1 = math.exp(-A * A / (2.0 * (1.0 + B * B)))
        E2 = 1.0 + B * B
        c1 = _norm.ppf(fpr1) if fpr1 > 0 else -math.inf
        c2 = _norm.ppf(fpr2) if fpr2 < 1 else math.inf
        E3 = _norm.cdf(c2) - _norm.cdf(c1)
        phi_c1 = math.exp(-0.5 * c1 * c1) / math.sqrt(2 * math.pi) if math.isfinite(c1) else 0.0
        phi_c2 = math.exp(-0.5 * c2 * c2) / math.sqrt(2 * math.pi) if math.isfinite(c2) else 0.0
        E4 = phi_c1 - phi_c2
        f = E1 * E3 / math.sqrt(2 * math.pi * E2)
        g = (E1 * E4 / math.sqrt(2 * math.pi * E2)
             - A * B * E1 * E3 / math.sqrt(2 * math.pi * E2 ** 3))
        return f, g, A

    f1, g1, A1 = _ef(A1, B1)
    f2, g2, A2 = _ef(A2, B2)

    cov = (f1 * f2 * (r_pos + r_neg * B1 * B2 / R + r_pos ** 2 * A1 * A2 / 2.0)
           + g1 * g2 * B1 * B2 * (r_neg ** 2 + R * r_pos ** 2) / (2.0 * R)
           + f1 * g2 * A1 * B2 * r_pos ** 2 / 2.0
           + f2 * g1 * A2 * B1 * r_pos ** 2 / 2.0)
    return cov


def _hanley_cov(theta1: float, theta2: float, R: float, r: float) -> float:
    """Covariance for continuous correlated design (Hanley & McNeil 1983)."""
    v1 = _hanley_var(theta1, R)
    v2 = _hanley_var(theta2, R)
    return 2.0 * r * math.sqrt(v1 * v2)


# ---------------------------------------------------------------------------
# AUC / partial-AUC transform (Obuchowski & McClish 1997)
# ---------------------------------------------------------------------------

def _auc_prime_to_auc(auc_prime: float, fpr1: float, fpr2: float) -> float:
    """Convert AUC' (rescaled) to actual AUC for partial range."""
    if fpr1 == 0.0 and fpr2 == 1.0:
        return auc_prime
    max_ = fpr2 - fpr1
    min_ = max_ / 2.0 * (fpr2 + fpr1)
    auc = (2.0 * auc_prime - 1.0) * (max_ - min_) + min_
    return auc


# ---------------------------------------------------------------------------
# Power / N core for one-sample ROC
# ---------------------------------------------------------------------------

def _roc_one_sample_power(theta0: float, theta1: float, n_pos: int,
                          R: float, alpha: float, sides: int,
                          data_type: str, B: float,
                          fpr1: float, fpr2: float) -> float:
    """Power for one-sample ROC AUC test."""
    if data_type == "continuous":
        v0 = _hanley_var(theta0, R) / n_pos
        v1 = _hanley_var(theta1, R) / n_pos
    else:
        v0 = _obuchowski_var(theta0, R, B, fpr1, fpr2) / n_pos
        v1 = _obuchowski_var(theta1, R, B, fpr1, fpr2) / n_pos

    delta = abs(theta1 - theta0)
    if v0 <= 0 or v1 <= 0:
        return 0.0
    z_alpha = _norm.ppf(1.0 - alpha / sides)
    power = _norm.cdf((delta * math.sqrt(n_pos) - z_alpha * math.sqrt(v0 * n_pos))
                      / math.sqrt(v1 * n_pos))
    return float(power)


def _roc_one_sample_n(theta0: float, theta1: float, R: float, alpha: float,
                      power: float, sides: int, data_type: str, B: float,
                      fpr1: float, fpr2: float) -> tuple[int, float]:
    """Smallest N+ achieving >= power for one-sample ROC AUC test."""
    if data_type == "continuous":
        v0 = _hanley_var(theta0, R)
        v1 = _hanley_var(theta1, R)
    else:
        v0 = _obuchowski_var(theta0, R, B, fpr1, fpr2)
        v1 = _obuchowski_var(theta1, R, B, fpr1, fpr2)

    delta = abs(theta1 - theta0)
    z_alpha = _norm.ppf(1.0 - alpha / sides)
    z_beta = _norm.ppf(power)

    n_pos_cont = ((z_alpha * math.sqrt(v0) + z_beta * math.sqrt(v1)) / delta) ** 2
    n_pos = max(2, math.ceil(n_pos_cont))

    achieved = _roc_one_sample_power(theta0, theta1, n_pos, R, alpha, sides,
                                     data_type, B, fpr1, fpr2)
    while achieved < power:
        n_pos += 1
        achieved = _roc_one_sample_power(theta0, theta1, n_pos, R, alpha, sides,
                                         data_type, B, fpr1, fpr2)
    return n_pos, achieved


# ---------------------------------------------------------------------------
# Power / N core for two-sample ROC (independent or correlated)
# ---------------------------------------------------------------------------

def _roc_two_sample_power(theta1: float, theta2: float, n_pos: int,
                          R: float, alpha: float, sides: int,
                          data_type: str, B1: float, B2: float,
                          r_pos: float, r_neg: float,
                          fpr1: float, fpr2: float) -> float:
    delta = abs(theta1 - theta2)
    if data_type == "continuous":
        v1 = _hanley_var(theta1, R)
        v2 = _hanley_var(theta2, R)
        cov_null = _hanley_cov(theta1, theta1, R, r_pos)  # under H0 both = theta1
        cov_alt = _hanley_cov(theta1, theta2, R, r_pos)
        v0_delta = (v1 + v1 - 2.0 * cov_null) / n_pos
        v_alt_delta = (v1 + v2 - 2.0 * cov_alt) / n_pos
    else:
        v1 = _obuchowski_var(theta1, R, B1, fpr1, fpr2)
        v2 = _obuchowski_var(theta2, R, B2, fpr1, fpr2)
        cov_null = _obuchowski_cov(theta1, theta1, R, B1, B2, r_pos, r_neg, fpr1, fpr2)
        cov_alt = _obuchowski_cov(theta1, theta2, R, B1, B2, r_pos, r_neg, fpr1, fpr2)
        v0_delta = (v1 + v1 - 2.0 * cov_null) / n_pos
        v_alt_delta = (v1 + v2 - 2.0 * cov_alt) / n_pos

    if v0_delta <= 0 or v_alt_delta <= 0:
        return 0.0
    z_alpha = _norm.ppf(1.0 - alpha / sides)
    power = _norm.cdf(
        (delta * math.sqrt(n_pos) - z_alpha * math.sqrt(v0_delta * n_pos))
        / math.sqrt(v_alt_delta * n_pos)
    )
    return float(power)


def _roc_two_sample_n(theta1: float, theta2: float, R: float, alpha: float,
                      power: float, sides: int, data_type: str, B1: float, B2: float,
                      r_pos: float, r_neg: float,
                      fpr1: float, fpr2: float) -> tuple[int, float]:
    delta = abs(theta1 - theta2)
    if data_type == "continuous":
        v1 = _hanley_var(theta1, R)
        v2 = _hanley_var(theta2, R)
        cov_null = _hanley_cov(theta1, theta1, R, r_pos)
        cov_alt = _hanley_cov(theta1, theta2, R, r_pos)
    else:
        v1 = _obuchowski_var(theta1, R, B1, fpr1, fpr2)
        v2 = _obuchowski_var(theta2, R, B2, fpr1, fpr2)
        cov_null = _obuchowski_cov(theta1, theta1, R, B1, B2, r_pos, r_neg, fpr1, fpr2)
        cov_alt = _obuchowski_cov(theta1, theta2, R, B1, B2, r_pos, r_neg, fpr1, fpr2)

    v0_delta = v1 + v1 - 2.0 * cov_null
    v_alt_delta = v1 + v2 - 2.0 * cov_alt

    z_alpha = _norm.ppf(1.0 - alpha / sides)
    z_beta = _norm.ppf(power)

    n_pos_cont = ((z_alpha * math.sqrt(v0_delta) + z_beta * math.sqrt(v_alt_delta)) / delta) ** 2
    n_pos = max(2, math.ceil(n_pos_cont))

    achieved = _roc_two_sample_power(theta1, theta2, n_pos, R, alpha, sides,
                                     data_type, B1, B2, r_pos, r_neg, fpr1, fpr2)
    while achieved < power:
        n_pos += 1
        achieved = _roc_two_sample_power(theta1, theta2, n_pos, R, alpha, sides,
                                         data_type, B1, B2, r_pos, r_neg, fpr1, fpr2)
    return n_pos, achieved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def roc_auc_one_sample(
    *,
    auc0: float,
    auc1: float | None = None,
    n_diseased: int | None = None,
    R: float = 1.0,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    data_type: str = "discrete",
    B: float = 1.0,
    fpr1: float = 0.0,
    fpr2: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """One-sample ROC AUC test (H0: AUC=AUC0).

    Tests for one ROC curve (Hanley & McNeil 1982). Two variance options:
      data_type='discrete' : Obuchowski & McClish (1997)
      data_type='continuous': Hanley & McNeil (1983)

    Parameters
    ----------
    auc0        : AUC under H0.
    auc1        : AUC under H1 (required unless solve_for='effect').
    n_diseased  : N+ (diseased subjects).
    R           : N- / N+ allocation ratio.
    alpha       : Type-I error rate.
    power       : Target power (0 < power < 1).
    sides       : 1 or 2.
    data_type   : 'discrete' (rating scale) or 'continuous'.
    B           : SD- / SD+ ratio (discrete data only; use 1.0 for conservative).
    fpr1, fpr2  : FPR integration limits (0,1 for full AUC).
    """
    inputs_echo: dict[str, Any] = {
        "auc0": auc0, "auc1": auc1, "n_diseased": n_diseased, "R": R,
        "alpha": alpha, "power": power, "sides": sides,
        "data_type": data_type, "B": B, "fpr1": fpr1, "fpr2": fpr2,
    }

    if solve_for is None:
        if n_diseased is None:
            solve_for = "n"
        elif power is None:
            solve_for = "power"
        else:
            raise ValueError("Supply exactly two of (auc1, power, n_diseased); "
                             "leave the third None or set solve_for explicitly.")

    if solve_for == "power":
        assert auc1 is not None and n_diseased is not None
        achieved = _roc_one_sample_power(auc0, auc1, n_diseased, R, alpha, sides,
                                         data_type, B, fpr1, fpr2)
        result: dict[str, Any] = {"n_diseased": n_diseased,
                                   "n_nondiseased": math.ceil(R * n_diseased),
                                   "achieved_power": achieved}

    elif solve_for == "n":
        assert auc1 is not None and power is not None
        n_pos, achieved = _roc_one_sample_n(auc0, auc1, R, alpha, power, sides,
                                            data_type, B, fpr1, fpr2)
        result = {"n_diseased": n_pos,
                  "n_nondiseased": math.ceil(R * n_pos),
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "roc_auc_one_sample",
        "solve_for": solve_for,
        "n": result["n_diseased"] + result["n_nondiseased"],
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Obuchowski, N.A. & McClish, D.K. (1997). Sample size determination for "
            "diagnostic accuracy studies involving binormal ROC curve indices. "
            "Statistics in Medicine, 16, 1529-1542.",
            "Hanley, J.A. & McNeil, B.J. (1983). A method of comparing the areas "
            "under receiver operating characteristic curves derived from the same cases. "
            "Radiology, 148, 839-843.",
        ],
    }


def roc_auc_two_independent_samples(
    *,
    auc1: float,
    auc2: float | None = None,
    n_diseased: int | None = None,
    R: float = 1.0,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    data_type: str = "discrete",
    B1: float = 1.0,
    B2: float = 1.0,
    fpr1: float = 0.0,
    fpr2: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-sample ROC AUC test for INDEPENDENT (separate-patient) groups.

    Two correlated ROC curves with no inter-test correlation (Hanley & McNeil 1983).
    """
    return _roc_two_sample(
        auc1=auc1, auc2=auc2, n_diseased=n_diseased, R=R,
        alpha=alpha, power=power, sides=sides,
        data_type=data_type, B1=B1, B2=B2,
        r_pos=0.0, r_neg=0.0,
        fpr1=fpr1, fpr2=fpr2,
        solve_for=solve_for,
        method_id="roc_auc_two_independent_samples",
    )


def roc_auc_two_correlated_samples(
    *,
    auc1: float,
    auc2: float | None = None,
    n_diseased: int | None = None,
    R: float = 1.0,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    data_type: str = "discrete",
    B1: float = 1.0,
    B2: float = 1.0,
    r_pos: float = 0.6,
    r_neg: float = 0.6,
    fpr1: float = 0.0,
    fpr2: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-sample ROC AUC test for CORRELATED (paired) designs.

    Two correlated ROC curves with positive inter-test correlations r_pos and r_neg.
    """
    return _roc_two_sample(
        auc1=auc1, auc2=auc2, n_diseased=n_diseased, R=R,
        alpha=alpha, power=power, sides=sides,
        data_type=data_type, B1=B1, B2=B2,
        r_pos=r_pos, r_neg=r_neg,
        fpr1=fpr1, fpr2=fpr2,
        solve_for=solve_for,
        method_id="roc_auc_two_correlated_samples",
    )


# ---------------------------------------------------------------------------
# Confidence Interval for AUC
# ---------------------------------------------------------------------------

def _hanley_se_auc(auc: float, n1: int, n2: int) -> float:
    """SE of AUC estimator per Hanley & McNeil (1982).

    SE(AUC) = sqrt(
        [AUC(1-AUC) + (N1-1)(Q1 - AUC²) + (N2-1)(Q2 - AUC²)] / (N1*N2)
    )
    where Q1 = AUC / (2 - AUC),  Q2 = 2*AUC² / (1 + AUC).

    N1 = positive (diseased) group, N2 = negative (non-diseased) group.
    """
    Q1 = auc / (2.0 - auc)
    Q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1.0 - auc)
           + (n1 - 1) * (Q1 - auc * auc)
           + (n2 - 1) * (Q2 - auc * auc)) / (n1 * n2)
    return math.sqrt(max(var, 0.0))


def ci_auc_roc(
    *,
    auc: float,
    width: float | None = None,
    half_width: float | None = None,
    n1: int | None = None,
    n2: int | None = None,
    R: float = 1.0,
    alpha: float = 0.05,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Confidence interval for the area under an ROC curve.

    Uses the Hanley & McNeil (1982) variance formula for the AUC estimator.
    Sample size is chosen so that the CI width ≤ requested ``width``
    (two-sided) or half-width ≤ ``half_width`` (one-sided or two-sided).

    Parameters
    ----------
    auc : float
        Anticipated AUC value (0.5–1.0).
    width : float or None
        Total CI width (UCL – LCL).  Provide either ``width`` or
        ``half_width``; they are related by width = 2 * half_width for
        a two-sided interval.
    half_width : float or None
        Distance from AUC to each limit (|AUC – LCL| = |UCL – AUC|).
    n1 : int or None
        Number of positive (diseased) subjects.
    n2 : int or None
        Number of negative (non-diseased) subjects.
        When not given, derived as n2 = ceil(R * n1).
    R : float
        Allocation ratio N2/N1 (default 1.0 for equal groups).
    alpha : float
        Error rate; CI level = 1 – alpha (default 0.05 → 95% CI).
    sides : int
        1 (one-sided) or 2 (two-sided, default).
    solve_for : str or None
        ``"n"`` (default when n1 is None) or ``"width"``.

    Returns
    -------
    dict
        Standard envelope with ``n1``, ``n2``, ``n``, ``achieved_width``,
        and ``achieved_power`` set to null (CI-only method).
    """
    if not 0.5 <= auc <= 1.0:
        raise ValueError("auc must be in [0.5, 1.0]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if R <= 0:
        raise ValueError("R must be positive")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    # Normalise width / half_width
    if width is not None and half_width is None:
        half_width = width / 2.0
    elif half_width is not None and width is None:
        width = half_width * 2.0 if sides == 2 else half_width
    elif width is None and half_width is None:
        if n1 is not None:
            solve_for = solve_for or "width"
        else:
            raise ValueError("supply one of (width, half_width, n1)")

    if solve_for is None:
        solve_for = "n" if n1 is None else "width"

    z = _norm.ppf(1.0 - alpha / sides)

    inputs_echo: dict[str, Any] = {
        "auc": auc, "width": width, "half_width": half_width,
        "n1": n1, "n2": n2, "R": R, "alpha": alpha, "sides": sides,
    }

    if n1 is not None and n2 is None:
        n2 = max(1, math.ceil(R * n1))

    def _achieved_hw(n1_val: int) -> float:
        n2_val = max(1, math.ceil(R * n1_val))
        return z * _hanley_se_auc(auc, n1_val, n2_val)

    if solve_for == "width":
        if n1 is None:
            raise ValueError("n1 required for solve_for='width'")
        assert n2 is not None
        hw = _achieved_hw(n1)
        aw = hw * 2.0 if sides == 2 else hw
        result: dict[str, Any] = {
            "n1": n1, "n2": n2, "n": n1 + n2,
            "achieved_width": aw,
            "achieved_power": None,
        }

    elif solve_for == "n":
        if half_width is None:
            raise ValueError("width or half_width required for solve_for='n'")
        target_hw = half_width

        # Closed-form starting estimate: SE = hw/z → n1 ~ ...
        # Iteratively search upward.
        n1_val = 2
        while n1_val <= 10_000_000:
            if _achieved_hw(n1_val) <= target_hw:
                break
            n1_val += 1
        else:
            raise RuntimeError("Could not find n1 ≤ 10,000,000")

        n2_val = max(1, math.ceil(R * n1_val))
        aw_hw = _achieved_hw(n1_val)
        aw = aw_hw * 2.0 if sides == 2 else aw_hw
        result = {
            "n1": n1_val, "n2": n2_val, "n": n1_val + n2_val,
            "achieved_width": aw,
            "achieved_power": None,
        }
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "ci_auc_roc",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Area Under an ROC Curve",
            "Hanley, J.A. & McNeil, B.J. (1982). The meaning and use of "
            "the area under a receiver operating characteristic (ROC) curve. "
            "Radiology, 143, 29-36.",
        ],
    }


def _roc_two_sample(
    *, auc1: float, auc2: float | None, n_diseased: int | None,
    R: float, alpha: float, power: float | None, sides: int,
    data_type: str, B1: float, B2: float,
    r_pos: float, r_neg: float,
    fpr1: float, fpr2: float,
    solve_for: str | None,
    method_id: str,
) -> dict[str, Any]:
    inputs_echo: dict[str, Any] = {
        "auc1": auc1, "auc2": auc2, "n_diseased": n_diseased, "R": R,
        "alpha": alpha, "power": power, "sides": sides,
        "data_type": data_type, "B1": B1, "B2": B2,
        "r_pos": r_pos, "r_neg": r_neg, "fpr1": fpr1, "fpr2": fpr2,
    }

    if solve_for is None:
        if n_diseased is None:
            solve_for = "n"
        elif power is None:
            solve_for = "power"
        else:
            raise ValueError("Supply exactly two of (auc2, power, n_diseased); "
                             "leave the third None or set solve_for explicitly.")

    if solve_for == "power":
        assert auc2 is not None and n_diseased is not None
        achieved = _roc_two_sample_power(auc1, auc2, n_diseased, R, alpha, sides,
                                         data_type, B1, B2, r_pos, r_neg, fpr1, fpr2)
        result: dict[str, Any] = {"n_diseased": n_diseased,
                                   "n_nondiseased": math.ceil(R * n_diseased),
                                   "achieved_power": achieved}

    elif solve_for == "n":
        assert auc2 is not None and power is not None
        n_pos, achieved = _roc_two_sample_n(auc1, auc2, R, alpha, power, sides,
                                            data_type, B1, B2, r_pos, r_neg, fpr1, fpr2)
        result = {"n_diseased": n_pos,
                  "n_nondiseased": math.ceil(R * n_pos),
                  "achieved_power": achieved}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": method_id,
        "solve_for": solve_for,
        "n": result["n_diseased"] + result["n_nondiseased"],
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Obuchowski, N.A. & McClish, D.K. (1997). Sample size determination for "
            "diagnostic accuracy studies involving binormal ROC curve indices. "
            "Statistics in Medicine, 16, 1529-1542.",
            "Hanley, J.A. & McNeil, B.J. (1983). A method of comparing the areas "
            "under receiver operating characteristic curves derived from the same cases. "
            "Radiology, 148, 839-843.",
        ],
    }
