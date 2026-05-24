"""Tests for Two Means in a 2x2 Cross-Over Design using Differences.

using Differences" (Chow & Liu 1999; Julious 2004).

In a 2x2 (AB/BA) cross-over each subject receives both treatments
separated by a washout period.  Power is driven by the *within-subject*
SD ``sd_w`` rather than between-subject variability.  The test statistic
is a paired-style t with degrees of freedom ``df = N - 2`` (one df lost
to sequence) and standard error ``sd_w * sqrt(2 / N)``.

Power formulae (df = N - 2, λ = δ * sqrt(N) / (σ_w * sqrt(2))):

  one-tailed Ha: mean1 > mean2 (δ > 0):
      power = 1 - T'(t_α; df, λ)

  one-tailed Ha: mean1 < mean2 (δ < 0):
      power = T'(-t_α; df, -λ) = T'(-t_α; df, λ) when sign handled.

  two-tailed Ha: mean1 ≠ mean2:
      power = [1 - T'(t_{α/2}; df, λ)] + T'(-t_{α/2}; df, λ)

Only *even* values of N are searched because the design is assumed
balanced (equal numbers in the AB and BA sequences); we follow the same
convention.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# ---- power given fixed N ---------------------------------------------------


def _power_cross_over(diff: float, sd_w: float, n: int,
                      alpha: float, sides: int) -> float:
    """Noncentral-t power for the 2x2 cross-over t-test.

    ``diff`` is mean1 - mean2 (the alternative effect; H0 difference is
    assumed to be zero; Diff0 = 0).
    ``sd_w`` is the within-subject SD (Sw, = sqrt(WMSE) from the ANOVA).
    """
    if n < 3:
        return 0.0
    df = n - 2
    se = sd_w * math.sqrt(2.0 / n)
    ncp = diff / se
    if sides == 2:
        t_crit = D.t_ppf(1 - alpha / 2.0, df)
        upper = 1.0 - D.nct_cdf(t_crit, df, ncp)
        lower = D.nct_cdf(-t_crit, df, ncp)
        return upper + lower
    elif sides == 1:
        t_crit = D.t_ppf(1 - alpha, df)
        if diff >= 0:
            return 1.0 - D.nct_cdf(t_crit, df, ncp)
        else:
            return D.nct_cdf(-t_crit, df, ncp)
    else:
        raise ValueError(f"sides must be 1 or 2, got {sides}")


def power_at_n(*, mean1: float, mean2: float, sd_w: float, n: int,
               alpha: float, sides: int = 2) -> float:
    """Compute power at a fixed total sample size N (across both sequences)."""
    return _power_cross_over(mean1 - mean2, sd_w, n, alpha, sides)


# ---- solve for N -----------------------------------------------------------


def _next_even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def n_for_power(*, mean1: float, mean2: float, sd_w: float, alpha: float,
                power: float, sides: int = 2,
                n_min: int = 4, n_max: int = 1_000_000) -> tuple[int, float]:
    """Smallest *even* N >= n_min achieving >= ``power``.

    Returns (n, achieved_power).  Only even N is searched because the
    design is balanced across the two sequences.
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if mean1 == mean2:
        raise ValueError("mean1 and mean2 must differ to solve for N")

    diff = mean1 - mean2
    lo = _next_even(max(n_min, 4))
    hi = lo
    while hi <= n_max:
        p = _power_cross_over(diff, sd_w, hi, alpha, sides)
        if p >= power:
            break
        lo = hi
        hi = max(hi + 2, hi * 2)
        hi = _next_even(hi)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")

    # Bisect on even values.
    while lo + 2 < hi:
        mid = _next_even((lo + hi) // 2)
        if mid == lo:
            mid += 2
        if mid == hi:
            break
        p = _power_cross_over(diff, sd_w, mid, alpha, sides)
        if p >= power:
            hi = mid
        else:
            lo = mid

    achieved = _power_cross_over(diff, sd_w, hi, alpha, sides)
    return hi, achieved


# ---- top-level entry point used by the registry ---------------------------


def cross_over_two_means(
    *,
    mean1: float | None = None,
    mean2: float | None = None,
    sd_w: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """2x2 cross-over (AB/BA) two-mean test power / sample-size solver.

    Inputs
    ------
    mean1, mean2
        Treatment effect means (the test concerns ``δ = mean1 - mean2``;
        Diff0 = 0, Diff1 = δ.  Provide both.
    sd_w
        Within-subject SD of the response (Sw = sqrt(WMSE)).  This is
        the standard deviation that drives the cross-over power
        calculation, *not* the between-subject SD.  If you only have the
        SD of the period differences (SdPeriod), pass
        ``sd_w = SdPeriod * sqrt(2)``.  If you have the SD of paired
        differences (SdPaired), pass ``sd_w = SdPaired / sqrt(2)``.
    alpha
        Type-I error rate.
    power, n
        Provide exactly one of these; the other is solved for.
        ``n`` is the *total* sample size across both sequences.
    sides
        1 or 2 (default 2).

    Returns
    -------
    dict with keys ``method_id``, ``solve_for``, ``n``,
    ``n_per_sequence`` (n / 2), ``achieved_power``, ``inputs_echo``,
    ``citations``.
    """
    if mean1 is None or mean2 is None:
        raise ValueError("supply both mean1 and mean2")
    if sd_w <= 0:
        raise ValueError("sd_w must be positive")

    inputs_echo = {
        "mean1": mean1, "mean2": mean2, "sd_w": sd_w, "alpha": alpha,
        "power": power, "n": n, "sides": sides,
    }

    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError(
            "supply exactly one of (power, n); leave the other None"
        )

    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        if n % 2 != 0:
            # but warn via the result by echoing what they passed.
            pass
        achieved = power_at_n(
            mean1=mean1, mean2=mean2, sd_w=sd_w, n=n,
            alpha=alpha, sides=sides,
        )
        result = {
            "n": n,
            "n_per_sequence": n // 2,
            "achieved_power": achieved,
        }

    elif solve_for == "n":
        assert power is not None
        n_req, achieved = n_for_power(
            mean1=mean1, mean2=mean2, sd_w=sd_w, alpha=alpha,
            power=power, sides=sides,
        )
        result = {
            "n": n_req,
            "n_per_sequence": n_req // 2,
            "achieved_power": achieved,
        }

    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "cross_over_two_means",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Design using Differences",
            "Chow, S.C. & Liu, J.P. (1999). Design and Analysis of "
            "Bioavailability and Bioequivalence Studies, 2nd ed.",
            "Julious, S.A. (2004). Sample sizes for clinical trials with "
            "Normal data. Statistics in Medicine 23: 1921-1986.",
        ],
    }


# ---------------------------------------------------------------------------
# M-Period Cross-Over Designs using Contrasts
# ---------------------------------------------------------------------------
#
# Each subject is measured at M time points (periods) and receives some
# sequence of treatments.  The hypothesis tested is H0: C'mu = 0 versus
# Ha: C'mu != 0, where C is a user-supplied contrast (coefficients sum to 0)
# and mu is the vector of period means.
#
# noncentral F with df1 = 1, df2 = N - S (with S = 1 sequence group in the
# multivariate analysis; S > 1 only when the user explicitly separates
# sequences in a mixed-model analysis, in which case the formula reduces
# df2 = N - S).  The noncentrality parameter is
#
#     lambda = N * (C' mu)^2 / (C' Sigma C)
#
# Under compound symmetry with all rho's equal, C' Sigma C reduces to
# sigma_w^2 * sum(c_i^2) where sigma_w^2 = sigma^2 * (1 - rho).  More
# generally we accept the user-supplied within-subject SD ``sd_w`` such
# that the effective denominator is ``sd_w^2 * sum(c_i^2)``.  This matches
# both the All-rho-Equal example (Example 1) and the AR(1) hand-calculation
# example when sd_w is set so that
# sd_w^2 * sum(c_i^2) == C' Sigma C.
#
# Power = P(F' > F_{1-alpha,1,df2,0} | F' ~ F_{1,df2,lambda}).


def _cprime_mu(contrasts: list[float], means: list[float] | None,
               mean_contrast: float | None) -> float:
    """Return the value of the contrast applied to the period means.

    Either ``means`` (a list the same length as ``contrasts``) or a
    pre-computed scalar ``mean_contrast`` may be supplied; if both are
    given, they must agree.
    """
    if mean_contrast is None and means is None:
        raise ValueError("supply either means or mean_contrast")
    computed = None
    if means is not None:
        if len(means) != len(contrasts):
            raise ValueError(
                f"means length {len(means)} != contrasts length {len(contrasts)}"
            )
        computed = sum(c * m for c, m in zip(contrasts, means))
    if mean_contrast is None:
        return computed  # type: ignore[return-value]
    if computed is not None and not math.isclose(
        computed, mean_contrast, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            f"mean_contrast={mean_contrast} disagrees with C'mu={computed} "
            f"computed from means+contrasts"
        )
    return mean_contrast


def _power_m_period(
    *, contrasts: list[float], cmu: float, sd_w: float, n: int,
    s: int, alpha: float, sides: int,
) -> float:
    """Noncentral-F power for the M-period cross-over contrast test.

    Uses df1 = 1, df2 = N - S, NCP = N * (C'mu)^2 / (sd_w^2 * sum c_i^2).
    The F-test is intrinsically two-sided; for ``sides == 1`` we use the
    equivalent noncentral-t form with the same NCP (sqrt of F NCP) and
    df = df2.
    """
    sum_csq = sum(c * c for c in contrasts)
    if sum_csq <= 0:
        raise ValueError("contrasts must have non-zero sum of squares")
    df2 = n - s
    if df2 < 1:
        return 0.0
    lam = n * (cmu ** 2) / (sum_csq * sd_w * sd_w)
    if sides == 2:
        f_crit = D.t_ppf(1 - alpha / 2.0, df2)  # |T| critical, but we use F
        # Use the F-based formula directly: power = 1 - F'(F_crit; 1, df2, lam)
        # with F_crit = t_{1-alpha/2,df2}^2.  Equivalently:
        from samplesize.core import distributions as _D
        f_crit_val = _D.t_ppf(1 - alpha / 2.0, df2) ** 2
        return 1.0 - _D.ncf_cdf(f_crit_val, 1, df2, lam)
    elif sides == 1:
        # One-sided t-test using sign of C'mu.
        t_ncp = math.copysign(math.sqrt(lam), cmu)
        t_crit = D.t_ppf(1 - alpha, df2)
        if cmu >= 0:
            return 1.0 - D.nct_cdf(t_crit, df2, t_ncp)
        else:
            return D.nct_cdf(-t_crit, df2, t_ncp)
    else:
        raise ValueError(f"sides must be 1 or 2, got {sides}")


def _solve_n_m_period(
    *, contrasts: list[float], cmu: float, sd_w: float,
    s: int, alpha: float, power: float, sides: int,
    n_per_sequence: int | None,
    n_min: int = 4, n_max: int = 1_000_000,
) -> tuple[int, float]:
    """Smallest N >= n_min achieving the target power.

    If ``n_per_sequence`` is given, the search is restricted to multiples
    of ``s * 1`` consistent with that, i.e., N = n_per_sequence * S; the
    function then returns the smallest such N. Otherwise N is searched
    one subject at a time (reporting the exact integer
    N from its search grid).
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if cmu == 0:
        raise ValueError("C'mu must be non-zero to solve for N")

    step = 1
    if n_per_sequence is not None:
        # If user fixes n_per_sequence, N must be a multiple of S (with
        # n_per_sequence subjects in each).  We then search S = N / n_per_sequence
        # integer values.  Here we treat the more common case of a fixed
        # number of sequences ``s`` and increment by ``s`` so that
        # n_per_sequence stays an integer.
        step = s

    lo = max(n_min, s + 1)
    if step > 1:
        # round lo up to nearest multiple of step
        lo = ((lo + step - 1) // step) * step
    hi = lo
    while hi <= n_max:
        p = _power_m_period(
            contrasts=contrasts, cmu=cmu, sd_w=sd_w, n=hi,
            s=s, alpha=alpha, sides=sides,
        )
        if p >= power:
            break
        lo = hi
        hi = max(hi + step, hi * 2)
        if step > 1:
            hi = ((hi + step - 1) // step) * step
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")

    while lo + step < hi:
        mid = ((lo + hi) // 2)
        if step > 1:
            mid = ((mid + step - 1) // step) * step
        if mid <= lo:
            mid = lo + step
        if mid >= hi:
            break
        p = _power_m_period(
            contrasts=contrasts, cmu=cmu, sd_w=sd_w, n=mid,
            s=s, alpha=alpha, sides=sides,
        )
        if p >= power:
            hi = mid
        else:
            lo = mid

    achieved = _power_m_period(
        contrasts=contrasts, cmu=cmu, sd_w=sd_w, n=hi,
        s=s, alpha=alpha, sides=sides,
    )
    return hi, achieved


def m_period_cross_over_contrasts(
    *,
    contrasts: list[float],
    mean_contrast: float | None = None,
    means: list[float] | None = None,
    sd_w: float,
    n_per_sequence: int | None = None,
    n_sequences: int = 1,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """M-Period Cross-Over contrast test.

    Inputs
    ------
    contrasts
        List of M contrast coefficients ``c_i``; must sum to zero.  By
        It is recommended (but not required) that
        ``sum(|c_i|) == 2``.
    mean_contrast
        The true value of ``C'mu = sum(c_i * mu_i)`` under the
        alternative.  Either provide this directly or provide ``means``
        below; if both are given they must match.
    means
        Optional list of M period means.  If supplied, ``C'mu`` is
        computed as the dot product with ``contrasts``.
    sd_w
        Within-subject SD entering the test denominator.  Defined so
        that the noncentrality parameter is
        ``N * (C'mu)^2 / (sd_w^2 * sum c_i^2)``.  Equivalently, for the
        compound-symmetry / All-rho-Equal model
        ``sd_w = sigma * sqrt(1 - rho)``; for AR(1) and other covariance
        structures pass the value of ``sd_w`` such that
        ``sd_w^2 * sum c_i^2`` equals ``C' Sigma C``.
    n_per_sequence
        Optional: subjects per sequence.  If supplied, N is searched on
        multiples of ``n_sequences`` so that each sequence has an
        integer count.  Reported as ``n_per_sequence`` in the result.
    n_sequences
        Number of treatment sequences ``S`` used for degrees of freedom
        (df2 = N - S).  Default 1 matches the multivariate Hotelling
        T^2 output (df2 = N - 1).
    alpha
        Type I error rate.
    power, n
        Provide exactly one; the other is solved for.  ``n`` is the
        total number of subjects across all sequences.
    sides
        1 or 2 (default 2).  The two-sided test is recommended.

    Returns
    -------
    dict with keys ``method_id``, ``solve_for``, ``n``,
    ``n_per_sequence``, ``n_sequences``, ``mean_contrast``,
    ``achieved_power``, ``inputs_echo``, ``citations``.
    """
    if not contrasts or len(contrasts) < 2:
        raise ValueError("contrasts must have at least 2 entries")
    if sd_w <= 0:
        raise ValueError("sd_w must be positive")
    if n_sequences < 1:
        raise ValueError("n_sequences must be >= 1")
    c_sum = sum(contrasts)
    if not math.isclose(c_sum, 0.0, abs_tol=1e-9):
        raise ValueError(
            f"contrasts must sum to zero (got sum={c_sum})"
        )

    cmu = _cprime_mu(contrasts, means, mean_contrast)

    inputs_echo = {
        "contrasts": list(contrasts),
        "mean_contrast": mean_contrast,
        "means": list(means) if means is not None else None,
        "sd_w": sd_w,
        "n_per_sequence": n_per_sequence,
        "n_sequences": n_sequences,
        "alpha": alpha,
        "power": power,
        "n": n,
        "sides": sides,
    }

    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError(
            "supply exactly one of (power, n); leave the other None"
        )

    if solve_for is None:
        solve_for = "n" if n is None else "power"

    if solve_for == "power":
        assert n is not None
        achieved = _power_m_period(
            contrasts=contrasts, cmu=cmu, sd_w=sd_w, n=n,
            s=n_sequences, alpha=alpha, sides=sides,
        )
        n_used = n

    elif solve_for == "n":
        assert power is not None
        n_used, achieved = _solve_n_m_period(
            contrasts=contrasts, cmu=cmu, sd_w=sd_w,
            s=n_sequences, alpha=alpha, power=power, sides=sides,
            n_per_sequence=n_per_sequence,
        )

    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    nps_out = (
        n_used // n_sequences if n_sequences > 0 else None
    )

    return {
        "method_id": "m_period_cross_over_contrasts",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": nps_out,
        "n_sequences": n_sequences,
        "mean_contrast": cmu,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "using Contrasts.",
            "Jones, B. & Kenward, M.G. (2015). Design and Analysis of "
            "Cross-Over Trials, 3rd ed. CRC Press.",
            "Davis, C.S. (2002). Statistical Methods for the Analysis of "
            "Repeated Measurements. Springer.",
            "Maxwell, S.E. & Delaney, H.D. (2003). Designing Experiments "
            "and Analyzing Data, 2nd ed.",
        ],
    }


# ============================================================================
# Bioequivalence (2x2 cross-over) on log-transformed ratios
# ----------------------------------------------------------------------------
#   "Equivalence Tests for Two Means in a 2x2 Cross-Over Design using Ratios"
#
# References:
#   Schuirmann, D.J. (1987).  A comparison of the two one-sided tests
#     procedure and the power approach for assessing the equivalence of
#     average bioavailability.  J. Pharmacokin. Biopharm. 15: 657-680.
#   Phillips, K.F. (1990).  Power of the two one-sided tests procedure in
#     bioequivalence.  J. Pharmacokin. Biopharm. 18: 137-144.
#   Diletti, E., Hauschke, D., Steinijans, V.W. (1991).  Sample size
#     determination for bioequivalence assessment by means of confidence
#     intervals.  Int. J. Clin. Pharmacol. Ther. Toxicol. 29: 1-8.
#   Julious, S.A. (2004).  Sample sizes for clinical trials with Normal
#     data.  Statistics in Medicine 23: 1921-1986.
#
# Mechanics:
#   PK metrics (AUC, Cmax) are assumed log-normal.  After log-transform
#   the within-subject SD on the log scale is
#       sigma_w = sqrt(ln(CV^2 + 1))
#   where CV is the coefficient of variation on the original scale.
#
#   The TOST procedure tests
#       H0:  phi <= phi_L  OR  phi >= phi_U
#       H1:  phi_L < phi < phi_U
#   with phi = mu_T / mu_R.  On the log scale the equivalence limits
#   become (ln phi_L, ln phi_U) and the effect is D = ln(R1).
#
#   Exact Phillips (1990) / Diletti (1991) power via Owen's Q:
#       Power = Q_v(-t_alpha; delta_U; 0, B) - Q_v(t_alpha; delta_L; 0, B)
#   where
#       SE      = sigma_w * sqrt(2/N)
#       df = v  = N - 2
#       delta_L = (ln R1 - ln RL) / SE
#       delta_U = (ln R1 - ln RU) / SE
#       B       = sqrt(v) * (delta_L - delta_U) / (2 * t_alpha)
#       Q_v(t; d; a, b) = integral_a^b Phi(t*x/sqrt(v) - d) f_chi(x; v) dx
# ----------------------------------------------------------------------------


def _owens_q(v: int, t_val: float, delta: float, a: float, b: float) -> float:
    """Owen's Q function used by Phillips/Diletti exact power.

    Q_v(t; delta; a, b) = integral_a^b Phi(t*x/sqrt(v) - delta) f_chi(x; v) dx

    Evaluated by adaptive Gauss-Kronrod quadrature.
    """
    from scipy.stats import chi, norm
    from scipy.integrate import quad

    sqrt_v = math.sqrt(v)

    def integrand(x: float) -> float:
        return float(norm.cdf(t_val * x / sqrt_v - delta) * chi.pdf(x, v))

    val, _ = quad(integrand, a, b, limit=200, epsabs=1e-12, epsrel=1e-10)
    return float(val)


def _bioeq_power(ratio: float, lower_limit: float, upper_limit: float,
                 cv: float, n: int, alpha: float) -> float:
    """Exact Phillips (1990) / Diletti (1991) TOST power on log ratios.

    Equivalence Tests for Two Means in a 2x2 Cross-Over
    Design using Ratios" to >=4 decimals on the chapter's worked
    examples.
    """
    if n < 3:
        return 0.0
    if cv <= 0:
        raise ValueError("cv must be positive")
    if not (0 < lower_limit < 1 < upper_limit):
        raise ValueError(
            "must have 0 < lower_limit < 1 < upper_limit"
        )
    if ratio <= 0:
        raise ValueError("ratio must be positive")

    df = n - 2
    sigma_w = math.sqrt(math.log(cv * cv + 1.0))
    se = sigma_w * math.sqrt(2.0 / n)

    log_R1 = math.log(ratio)
    log_RL = math.log(lower_limit)
    log_RU = math.log(upper_limit)

    t_alpha = D.t_ppf(1.0 - alpha, df)
    if t_alpha <= 0:
        return 0.0

    delta_L = (log_R1 - log_RL) / se
    delta_U = (log_R1 - log_RU) / se
    if delta_L <= delta_U:
        return 0.0

    B = math.sqrt(df) * (delta_L - delta_U) / (2.0 * t_alpha)
    q_upper = _owens_q(df, -t_alpha, delta_U, 0.0, B)
    q_lower = _owens_q(df, t_alpha, delta_L, 0.0, B)
    return max(0.0, q_upper - q_lower)


def bioeq_power_at_n(*, ratio: float, lower_limit: float, upper_limit: float,
                     cv: float, n: int, alpha: float) -> float:
    """Power at fixed total sample size N for a 2x2 cross-over BE test."""
    return _bioeq_power(ratio, lower_limit, upper_limit, cv, n, alpha)


def bioeq_n_for_power(*, ratio: float, lower_limit: float, upper_limit: float,
                      cv: float, alpha: float, power: float,
                      n_min: int = 4,
                      n_max: int = 1_000_000) -> tuple[int, float]:
    """Smallest *even* total N achieving >= ``power`` on the log scale.

    Only even N is searched because the design is balanced between
    sequences AB and BA.
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if not (lower_limit < ratio < upper_limit):
        raise ValueError(
            "true ratio must lie strictly between the equivalence limits "
            "for sample-size calculation"
        )

    lo = _next_even(max(n_min, 4))
    hi = lo
    while hi <= n_max:
        p = _bioeq_power(ratio, lower_limit, upper_limit, cv, hi, alpha)
        if p >= power:
            break
        lo = hi
        hi = max(hi + 2, hi * 2)
        hi = _next_even(hi)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")

    while lo + 2 < hi:
        mid = _next_even((lo + hi) // 2)
        if mid == lo:
            mid += 2
        if mid == hi:
            break
        p = _bioeq_power(ratio, lower_limit, upper_limit, cv, mid, alpha)
        if p >= power:
            hi = mid
        else:
            lo = mid

    achieved = _bioeq_power(ratio, lower_limit, upper_limit, cv, hi, alpha)
    return hi, achieved


def bioequivalence_two_means_ratios(
    *,
    ratio: float = 1.0,
    lower_limit: float = 0.80,
    upper_limit: float = 1.25,
    cv: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """2x2 cross-over bioequivalence power / sample-size solver on ratios.

    Cross-Over Design using Ratios" via the Phillips (1990) / Diletti
    (1991) exact TOST power formula on the log scale.

    Inputs
    ------
    ratio
        True geometric mean ratio ``phi = mu_T / mu_R`` at which power
        is computed.  Defaults to 1.0 (treatments truly equal).
    lower_limit, upper_limit
        Equivalence limits on the ratio scale.  Defaults to the
        FDA/EMA standard 0.80 / 1.25.  Must satisfy
        ``0 < lower_limit < 1 < upper_limit``.
    cv
        Coefficient of variation on the *original* (unlogged) scale,
        as a decimal (e.g. 0.30 for 30%).  Converted internally to the
        log-scale within-subject SD via
        ``sigma_w = sqrt(ln(cv^2 + 1))``.
    alpha
        Per one-sided test (TOST) significance level.  Default 0.05.
        The corresponding two-sided confidence-interval level is
        ``1 - 2*alpha`` (i.e. the usual 90% CI for alpha=0.05).
    power, n
        Provide exactly one; the other is solved for.  ``n`` is the
        total sample size across both AB and BA sequences and is
        constrained to even values for balance.

    Returns
    -------
    dict with keys ``method_id``, ``solve_for``, ``n``,
    ``n_per_sequence`` (= n / 2), ``achieved_power``, ``sigma_w``,
    ``inputs_echo``, ``citations``.
    """
    if cv <= 0:
        raise ValueError("cv must be positive")
    if not (0 < lower_limit < 1):
        raise ValueError("lower_limit must lie in (0, 1)")
    if upper_limit <= 1:
        raise ValueError("upper_limit must be > 1")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")

    inputs_echo = {
        "ratio": ratio, "lower_limit": lower_limit,
        "upper_limit": upper_limit, "cv": cv, "alpha": alpha,
        "power": power, "n": n,
    }

    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError(
            "supply exactly one of (power, n); leave the other None"
        )

    if solve_for is None:
        solve_for = "n" if n is None else "power"

    sigma_w = math.sqrt(math.log(cv * cv + 1.0))

    if solve_for == "power":
        assert n is not None
        achieved = bioeq_power_at_n(
            ratio=ratio, lower_limit=lower_limit, upper_limit=upper_limit,
            cv=cv, n=n, alpha=alpha,
        )
        result = {
            "n": n,
            "n_per_sequence": n // 2,
            "achieved_power": achieved,
        }
    elif solve_for == "n":
        assert power is not None
        n_req, achieved = bioeq_n_for_power(
            ratio=ratio, lower_limit=lower_limit, upper_limit=upper_limit,
            cv=cv, alpha=alpha, power=power,
        )
        result = {
            "n": n_req,
            "n_per_sequence": n_req // 2,
            "achieved_power": achieved,
        }
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "bioequivalence_two_means_ratios",
        "solve_for": solve_for,
        **result,
        "sigma_w": sigma_w,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cross-Over Design using Ratios",
            "Schuirmann, D.J. (1987). A comparison of the two one-sided "
            "tests procedure and the power approach for assessing the "
            "equivalence of average bioavailability. J. Pharmacokin. "
            "Biopharm. 15: 657-680.",
            "Phillips, K.F. (1990). Power of the two one-sided tests "
            "procedure in bioequivalence. J. Pharmacokin. Biopharm. "
            "18: 137-144.",
            "Diletti, E., Hauschke, D. & Steinijans, V.W. (1991). "
            "Sample size determination for bioequivalence assessment by "
            "means of confidence intervals. Int. J. Clin. Pharmacol. "
            "Ther. Toxicol. 29: 1-8.",
            "Julious, S.A. (2004). Sample sizes for clinical trials "
            "with Normal data. Statistics in Medicine 23: 1921-1986.",
        ],
    }


# ============================================================================
# Non-Inferiority / Equivalence / Superiority-by-Margin in 2x2 cross-over.
# ----------------------------------------------------------------------------
# All six routines below share the same denominator as ``cross_over_two_means``
# (SE = sd_w * sqrt(2/N), df = N - 2) and assume a balanced AB/BA design.  We
# Search only even values of N.
#
# Reference: Chow, S.C., Shao, J. & Wang, H. (2003).  Sample Size Calculations
#   in Clinical Research, 1st ed., Chapter 5 (cross-over designs).  Julious
#   (2004) gives consolidated formulae and worked examples for
#   chapter validation.
# ============================================================================


_CITATIONS_DIFF = [
    "Chow, S.C., Shao, J. & Wang, H. (2003). Sample Size Calculations "
    "in Clinical Research, 1st ed. Marcel Dekker.",
    "Julious, S.A. (2004). Sample sizes for clinical trials with "
    "Normal data. Statistics in Medicine 23: 1921-1986.",
    "Chow, S.C. & Liu, J.P. (1999). Design and Analysis of "
    "Bioavailability and Bioequivalence Studies, 2nd ed.",
]

_CITATIONS_RATIOS = [
    "using Ratios (and NI / Sup-by-Margin companion chapters).",
    "Julious, S.A. (2004). Sample sizes for clinical trials with "
    "Normal data. Statistics in Medicine 23: 1921-1986.",
    "Hauschke, D., Steinijans, V. & Pigeot, I. (2007). Bioequivalence "
    "Studies in Drug Development. Wiley.",
    "Chow, S.C. & Liu, J.P. (1999). Design and Analysis of "
    "Bioavailability and Bioequivalence Studies, 2nd ed.",
]


def _solve_even_n(
    power_fn,
    target_power: float,
    n_min: int = 4,
    n_max: int = 1_000_000,
) -> tuple[int, float]:
    """Bisect for the smallest *even* N >= ``n_min`` with ``power_fn(N) >= target``.

    Used by every NI / Eq / Sup-by-Margin routine in this module.
    """
    if not 0.0 < target_power < 1.0:
        raise ValueError("power must be in (0, 1)")
    lo = _next_even(max(n_min, 4))
    hi = lo
    while hi <= n_max:
        if power_fn(hi) >= target_power:
            break
        lo = hi
        hi = max(hi + 2, hi * 2)
        hi = _next_even(hi)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")

    while lo + 2 < hi:
        mid = _next_even((lo + hi) // 2)
        if mid <= lo:
            mid += 2
        if mid >= hi:
            break
        if power_fn(mid) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi, power_fn(hi)


# ---------------------------------------------------------------------------
# 1. Non-Inferiority Tests for Two Means in a 2x2 Cross-Over (Differences)
# ---------------------------------------------------------------------------


def _power_ni_diff(
    diff: float, margin: float, sd_w: float, n: int, alpha: float,
    higher_means: str,
) -> float:
    """Noncentral-t power for the NI 2x2 cross-over test on differences.

    ``higher_means`` is "better" (upper-tailed H1: Diff > -NIM) or "worse"
    (lower-tailed H1: Diff < NIM).  ``margin`` is the *magnitude* (>0).
    """
    if n < 3:
        return 0.0
    df = n - 2
    se = sd_w * math.sqrt(2.0 / n)
    t_crit = D.t_ppf(1.0 - alpha, df)
    if higher_means == "better":
        # H0: Diff <= -|NIM|, H1: Diff > -|NIM|
        ncp = (diff - (-margin)) / se
        return 1.0 - D.nct_cdf(t_crit, df, ncp)
    elif higher_means == "worse":
        # H0: Diff >= |NIM|, H1: Diff < |NIM|
        ncp = (diff - margin) / se
        return D.nct_cdf(-t_crit, df, ncp)
    else:
        raise ValueError(
            f"higher_means must be 'better' or 'worse', got {higher_means!r}"
        )


def non_inferiority_cross_over_diff(
    *,
    margin: float,
    diff: float = 0.0,
    sd_w: float,
    alpha: float = 0.025,
    power: float | None = None,
    n: int | None = None,
    higher_means: str = "better",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-Inferiority Tests for Two Means in a 2x2 Cross-Over (Differences).

    Cross-Over Design using Differences" (Chapter 510).

    Inputs
    ------
    margin
        Non-inferiority margin magnitude (positive; NIM).
    diff
        True mean difference D = mean_T - mean_R at which power is
        computed (default 0).
    sd_w
        Within-subject SD (Sw = sqrt(WMSE)).  The user may
        enter SdPeriod or SdPaired instead; convert to Sw via
        ``Sw = SdPeriod * sqrt(2) = SdPaired / sqrt(2)``.
    alpha
        One-sided Type-I error.  Default 0.025.
    power, n
        Provide exactly one; the other is solved for.  N is total
        (across both sequences) and is constrained to even values.
    higher_means
        "better" (default) places the rejection region on Diff > -NIM;
        "worse" places it on Diff < +NIM.

    Returns
    -------
    dict with ``method_id``, ``solve_for``, ``n``, ``n_per_sequence``,
    ``achieved_power``, ``inputs_echo``, ``citations``.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    if sd_w <= 0:
        raise ValueError("sd_w must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")

    inputs_echo = {
        "margin": margin, "diff": diff, "sd_w": sd_w, "alpha": alpha,
        "power": power, "n": n, "higher_means": higher_means,
    }
    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError("supply exactly one of (power, n)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    def pwr(nn: int) -> float:
        return _power_ni_diff(diff, margin, sd_w, nn, alpha, higher_means)

    if solve_for == "power":
        assert n is not None
        achieved = pwr(n)
        n_used = n
    elif solve_for == "n":
        assert power is not None
        n_used, achieved = _solve_even_n(pwr, power)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "non_inferiority_cross_over_diff",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // 2,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cross-Over Design using Differences (Chapter 510).",
            *_CITATIONS_DIFF,
        ],
    }


# ---------------------------------------------------------------------------
# 2. Equivalence Tests for Two Means in a 2x2 Cross-Over (Differences)
# ---------------------------------------------------------------------------


def _power_eq_diff(
    diff: float, lower_limit: float, upper_limit: float,
    sd_w: float, n: int, alpha: float,
) -> float:
    """Exact Phillips (1990) / Owen-Q TOST power on the difference scale.

    Identical mechanics to ``_bioeq_power`` but operates on raw
    differences instead of log-ratios.
    """
    if n < 3:
        return 0.0
    if lower_limit >= upper_limit:
        raise ValueError("lower_limit must be strictly less than upper_limit")
    if not (lower_limit < diff < upper_limit):
        # caller may still pass extreme diff for power evaluation; allow
        # but the resulting power will be (essentially) zero.
        pass

    df = n - 2
    se = sd_w * math.sqrt(2.0 / n)
    t_alpha = D.t_ppf(1.0 - alpha, df)
    if t_alpha <= 0:
        return 0.0

    delta_L = (diff - lower_limit) / se
    delta_U = (diff - upper_limit) / se
    if delta_L <= delta_U:
        return 0.0

    B = math.sqrt(df) * (delta_L - delta_U) / (2.0 * t_alpha)
    q_upper = _owens_q(df, -t_alpha, delta_U, 0.0, B)
    q_lower = _owens_q(df, t_alpha, delta_L, 0.0, B)
    return max(0.0, q_upper - q_lower)


def equivalence_cross_over_diff(
    *,
    lower_limit: float,
    upper_limit: float,
    diff: float = 0.0,
    sd_w: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Equivalence (TOST) Tests for Two Means in a 2x2 Cross-Over (Differences).

    Cross-Over Design using Differences" (Chapter 520) via the exact
    Phillips (1990) / Owen-Q power formula on the difference scale.

    Inputs
    ------
    lower_limit, upper_limit
        Equivalence bounds (EL < 0 < EU).  The null of non-equivalence is
        ``Diff <= EL OR Diff >= EU``.
    diff
        True mean difference at which power is computed (default 0).
        Must lie strictly between the equivalence limits for finite
        power.
    sd_w
        Within-subject SD (Sw = sqrt(WMSE)).
    alpha
        Per one-sided test (TOST) significance level.  Default 0.05.
    power, n
        Provide exactly one; the other is solved for.  N is total and
        constrained to even values.
    """
    if upper_limit <= lower_limit:
        raise ValueError("upper_limit must exceed lower_limit")
    if sd_w <= 0:
        raise ValueError("sd_w must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")

    inputs_echo = {
        "lower_limit": lower_limit, "upper_limit": upper_limit,
        "diff": diff, "sd_w": sd_w, "alpha": alpha,
        "power": power, "n": n,
    }
    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError("supply exactly one of (power, n)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    def pwr(nn: int) -> float:
        return _power_eq_diff(diff, lower_limit, upper_limit, sd_w, nn, alpha)

    if solve_for == "power":
        assert n is not None
        achieved = pwr(n)
        n_used = n
    elif solve_for == "n":
        assert power is not None
        if not (lower_limit < diff < upper_limit):
            raise ValueError(
                "true diff must lie strictly between the equivalence "
                "limits for sample-size calculation"
            )
        n_used, achieved = _solve_even_n(pwr, power)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "equivalence_cross_over_diff",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // 2,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cross-Over Design using Differences (Chapter 520).",
            "Schuirmann, D.J. (1987). A comparison of the two one-sided "
            "tests procedure and the power approach for assessing the "
            "equivalence of average bioavailability. J. Pharmacokin. "
            "Biopharm. 15: 657-680.",
            "Phillips, K.F. (1990). Power of the two one-sided tests "
            "procedure in bioequivalence. J. Pharmacokin. Biopharm. "
            "18: 137-144.",
            *_CITATIONS_DIFF,
        ],
    }


# ---------------------------------------------------------------------------
# 3. Superiority-by-Margin Tests for Two Means in a 2x2 Cross-Over (Diff)
# ---------------------------------------------------------------------------


def _power_supm_diff(
    diff: float, margin: float, sd_w: float, n: int, alpha: float,
    higher_means: str,
) -> float:
    """Noncentral-t power for the sup-by-margin 2x2 cross-over test (diff).

    ``higher_means``: "better"  -> H0: Diff <= +SM, H1: Diff > +SM.
                       "worse" -> H0: Diff >= -SM, H1: Diff < -SM.
    """
    if n < 3:
        return 0.0
    df = n - 2
    se = sd_w * math.sqrt(2.0 / n)
    t_crit = D.t_ppf(1.0 - alpha, df)
    if higher_means == "better":
        ncp = (diff - margin) / se
        return 1.0 - D.nct_cdf(t_crit, df, ncp)
    elif higher_means == "worse":
        ncp = (diff - (-margin)) / se
        return D.nct_cdf(-t_crit, df, ncp)
    else:
        raise ValueError(
            f"higher_means must be 'better' or 'worse', got {higher_means!r}"
        )


def superiority_by_margin_cross_over_diff(
    *,
    margin: float,
    diff: float,
    sd_w: float,
    alpha: float = 0.025,
    power: float | None = None,
    n: int | None = None,
    higher_means: str = "better",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Superiority-by-Margin Tests for Two Means in a 2x2 Cross-Over (Diff).

    Means in a 2x2 Cross-Over Design using Differences" (Chapter 508).
    Mechanics are identical to the NI difference variant with the
    rejection region shifted to the *positive* margin side (Diff > +SM
    when higher is better; Diff < -SM when higher is worse).

    Inputs
    ------
    margin
        Superiority margin magnitude SM (>0).
    diff
        True mean difference D = mean_T - mean_R at which power is
        computed.  Should exceed +SM in magnitude on the relevant tail
        for finite power.
    sd_w
        Within-subject SD (Sw = sqrt(WMSE)).
    alpha
        One-sided Type-I error.  Default 0.025.
    power, n
        Provide exactly one.
    higher_means
        "better" or "worse" (selects upper- vs lower-tailed test).
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    if sd_w <= 0:
        raise ValueError("sd_w must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")

    inputs_echo = {
        "margin": margin, "diff": diff, "sd_w": sd_w, "alpha": alpha,
        "power": power, "n": n, "higher_means": higher_means,
    }
    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError("supply exactly one of (power, n)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    def pwr(nn: int) -> float:
        return _power_supm_diff(diff, margin, sd_w, nn, alpha, higher_means)

    if solve_for == "power":
        assert n is not None
        achieved = pwr(n)
        n_used = n
    elif solve_for == "n":
        assert power is not None
        # caller must supply a difference on the correct side of the
        # margin; otherwise the search will not converge to power.
        if higher_means == "better" and diff <= margin:
            raise ValueError("with higher_means='better' need diff > margin")
        if higher_means == "worse" and diff >= -margin:
            raise ValueError("with higher_means='worse' need diff < -margin")
        n_used, achieved = _solve_even_n(pwr, power)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "superiority_by_margin_cross_over_diff",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // 2,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "in a 2x2 Cross-Over Design using Differences (Chapter 508).",
            *_CITATIONS_DIFF,
        ],
    }


# ---------------------------------------------------------------------------
# 4. Tests for Two Means in a 2x2 Cross-Over using Ratios (inequality)
# ---------------------------------------------------------------------------


def _sigma_w_from_cv(cv: float) -> float:
    """Within-subject SD on the log scale induced by an original-scale CV."""
    return math.sqrt(math.log(cv * cv + 1.0))


def _power_ratios_test(
    r1: float, r0: float, cv: float, n: int, alpha: float, sides: int,
) -> float:
    """Noncentral-t power for the log-ratio inequality test in 2x2 cross-over."""
    if n < 3:
        return 0.0
    if r0 <= 0 or r1 <= 0:
        raise ValueError("r0 and r1 must be positive")
    if r1 == r0:
        # Under H0; no power above alpha.
        return alpha
    sigma_w = _sigma_w_from_cv(cv)
    df = n - 2
    se = sigma_w * math.sqrt(2.0 / n)
    diff = math.log(r1) - math.log(r0)
    ncp = diff / se
    if sides == 2:
        t_crit = D.t_ppf(1.0 - alpha / 2.0, df)
        return (1.0 - D.nct_cdf(t_crit, df, ncp)) + D.nct_cdf(-t_crit, df, ncp)
    elif sides == 1:
        t_crit = D.t_ppf(1.0 - alpha, df)
        if diff >= 0:
            return 1.0 - D.nct_cdf(t_crit, df, ncp)
        else:
            return D.nct_cdf(-t_crit, df, ncp)
    else:
        raise ValueError(f"sides must be 1 or 2, got {sides}")


def tests_two_means_cross_over_ratios(
    *,
    r1: float,
    r0: float = 1.0,
    cv: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Tests for Two Means in a 2x2 Cross-Over Design using Ratios.

    Design using Ratios" (Chapter 505) by transforming to the log scale
    (sigma_w = sqrt(ln(CV^2 + 1))) and applying the difference-scale
    cross-over t-test.

    Inputs
    ------
    r1
        True ratio mu_T / mu_R at which power is computed.
    r0
        Null ratio (default 1.0).  ``r1 != r0`` is required to solve N.
    cv
        Coefficient of variation on the *original* scale (decimal, e.g.
        0.30 for 30%).
    alpha
        Significance level (per side for two-sided test).  Default 0.05.
    power, n
        Provide exactly one.  N is total across both sequences,
        constrained to even values.
    sides
        1 or 2 (default 2).
    """
    if r0 <= 0 or r1 <= 0:
        raise ValueError("r0 and r1 must be positive")
    if cv <= 0:
        raise ValueError("cv must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")

    inputs_echo = {
        "r1": r1, "r0": r0, "cv": cv, "alpha": alpha,
        "power": power, "n": n, "sides": sides,
    }
    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError("supply exactly one of (power, n)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    sigma_w = _sigma_w_from_cv(cv)

    def pwr(nn: int) -> float:
        return _power_ratios_test(r1, r0, cv, nn, alpha, sides)

    if solve_for == "power":
        assert n is not None
        achieved = pwr(n)
        n_used = n
    elif solve_for == "n":
        assert power is not None
        if r1 == r0:
            raise ValueError("r1 and r0 must differ to solve for N")
        n_used, achieved = _solve_even_n(pwr, power)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "tests_two_means_cross_over_ratios",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // 2,
        "achieved_power": achieved,
        "sigma_w": sigma_w,
        "inputs_echo": inputs_echo,
        "citations": [
            "Design using Ratios (Chapter 505).",
            *_CITATIONS_RATIOS,
        ],
    }


# ---------------------------------------------------------------------------
# 5. Non-Inferiority Tests for Two Means in a 2x2 Cross-Over (Ratios)
# ---------------------------------------------------------------------------


def _power_ni_ratios(
    r1: float, margin: float, cv: float, n: int, alpha: float,
    higher_means: str,
) -> float:
    """Noncentral-t power for NI on ratios: shifts log-scale rejection region.

    When ``higher_means == 'better'``: H0: phi <= 1 - NIM, H1: phi > 1 - NIM.
    When ``higher_means == 'worse'`` : H0: phi >= 1 + NIM, H1: phi < 1 + NIM.
    """
    if n < 3:
        return 0.0
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    if margin <= 0:
        raise ValueError("margin must be positive")
    sigma_w = _sigma_w_from_cv(cv)
    df = n - 2
    se = sigma_w * math.sqrt(2.0 / n)
    t_crit = D.t_ppf(1.0 - alpha, df)
    if higher_means == "better":
        if margin >= 1:
            raise ValueError("margin must be < 1 when higher_means='better'")
        log_lb = math.log(1.0 - margin)
        ncp = (math.log(r1) - log_lb) / se
        return 1.0 - D.nct_cdf(t_crit, df, ncp)
    elif higher_means == "worse":
        log_ub = math.log(1.0 + margin)
        ncp = (math.log(r1) - log_ub) / se
        return D.nct_cdf(-t_crit, df, ncp)
    else:
        raise ValueError(
            f"higher_means must be 'better' or 'worse', got {higher_means!r}"
        )


def non_inferiority_cross_over_ratios(
    *,
    margin: float,
    r1: float = 1.0,
    cv: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    higher_means: str = "better",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Non-Inferiority Tests for Two Means in a 2x2 Cross-Over (Ratios).

    2x2 Cross-Over Design using Ratios" (Chapter 515).  Uses the log
    transform sigma_w = sqrt(ln(CV^2 + 1)) and reduces to the NI
    difference-scale test with margin |ln(1 - NIM)| (better) or
    |ln(1 + NIM)| (worse).

    Inputs
    ------
    margin
        Non-inferiority margin on the ratio scale (positive; NIM).
        For ``higher_means='better'`` the lower equivalence bound is
        ``1 - NIM`` (must be > 0, i.e. NIM < 1).
    r1
        True mean ratio mu_T / mu_R at which power is computed.  Default
        1.0.
    cv
        Original-scale coefficient of variation (decimal).
    alpha
        One-sided significance level.  Default 0.05.
    power, n
        Exactly one of these is provided; the other is solved for.
    higher_means
        "better" (default) or "worse".
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    if cv <= 0:
        raise ValueError("cv must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if higher_means == "better" and margin >= 1:
        raise ValueError("margin must be < 1 when higher_means='better'")

    inputs_echo = {
        "margin": margin, "r1": r1, "cv": cv, "alpha": alpha,
        "power": power, "n": n, "higher_means": higher_means,
    }
    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError("supply exactly one of (power, n)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    sigma_w = _sigma_w_from_cv(cv)

    def pwr(nn: int) -> float:
        return _power_ni_ratios(r1, margin, cv, nn, alpha, higher_means)

    if solve_for == "power":
        assert n is not None
        achieved = pwr(n)
        n_used = n
    elif solve_for == "n":
        assert power is not None
        n_used, achieved = _solve_even_n(pwr, power)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "non_inferiority_cross_over_ratios",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // 2,
        "achieved_power": achieved,
        "sigma_w": sigma_w,
        "inputs_echo": inputs_echo,
        "citations": [
            "Cross-Over Design using Ratios (Chapter 515).",
            *_CITATIONS_RATIOS,
        ],
    }


# ---------------------------------------------------------------------------
# 6. Superiority-by-Margin Tests for Two Means in a 2x2 Cross-Over (Ratios)
# ---------------------------------------------------------------------------


def _power_supm_ratios(
    r1: float, margin: float, cv: float, n: int, alpha: float,
    higher_means: str,
) -> float:
    """Noncentral-t power for sup-by-margin on ratios.

    When ``higher_means == 'better'``: H0: phi <= 1 + SM, H1: phi > 1 + SM.
    When ``higher_means == 'worse'`` : H0: phi >= 1 - SM, H1: phi < 1 - SM.
    """
    if n < 3:
        return 0.0
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    if margin <= 0:
        raise ValueError("margin must be positive")
    sigma_w = _sigma_w_from_cv(cv)
    df = n - 2
    se = sigma_w * math.sqrt(2.0 / n)
    t_crit = D.t_ppf(1.0 - alpha, df)
    if higher_means == "better":
        log_lb = math.log(1.0 + margin)
        ncp = (math.log(r1) - log_lb) / se
        return 1.0 - D.nct_cdf(t_crit, df, ncp)
    elif higher_means == "worse":
        if margin >= 1:
            raise ValueError("margin must be < 1 when higher_means='worse'")
        log_ub = math.log(1.0 - margin)
        ncp = (math.log(r1) - log_ub) / se
        return D.nct_cdf(-t_crit, df, ncp)
    else:
        raise ValueError(
            f"higher_means must be 'better' or 'worse', got {higher_means!r}"
        )


def superiority_by_margin_cross_over_ratios(
    *,
    margin: float,
    r1: float,
    cv: float,
    alpha: float = 0.05,
    power: float | None = None,
    n: int | None = None,
    higher_means: str = "better",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Superiority-by-Margin Tests for Two Means in a 2x2 Cross-Over (Ratios).

    Means in a 2x2 Cross-Over Design using Ratios" (Chapter 513).
    Mechanics mirror the NI-ratios test with the rejection region shifted
    to the positive (better) or negative (worse) margin side.

    Inputs
    ------
    margin
        Superiority margin on the ratio scale (positive; SM).
        For higher_means='worse' must satisfy ``margin < 1``.
    r1
        True ratio mu_T / mu_R at which power is computed.  Should
        satisfy ``r1 > 1 + SM`` (better) or ``r1 < 1 - SM`` (worse) for
        finite power.
    cv
        Coefficient of variation on the original scale (decimal).
    alpha
        One-sided significance level.  Default 0.05.
    power, n
        Provide exactly one.
    higher_means
        "better" (default) or "worse".
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    if r1 <= 0:
        raise ValueError("r1 must be positive")
    if cv <= 0:
        raise ValueError("cv must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if higher_means == "worse" and margin >= 1:
        raise ValueError("margin must be < 1 when higher_means='worse'")

    inputs_echo = {
        "margin": margin, "r1": r1, "cv": cv, "alpha": alpha,
        "power": power, "n": n, "higher_means": higher_means,
    }
    given = sum(x is not None for x in (power, n))
    if given != 1:
        raise ValueError("supply exactly one of (power, n)")
    if solve_for is None:
        solve_for = "n" if n is None else "power"

    sigma_w = _sigma_w_from_cv(cv)

    def pwr(nn: int) -> float:
        return _power_supm_ratios(r1, margin, cv, nn, alpha, higher_means)

    if solve_for == "power":
        assert n is not None
        achieved = pwr(n)
        n_used = n
    elif solve_for == "n":
        assert power is not None
        if higher_means == "better" and r1 <= 1 + margin:
            raise ValueError(
                "with higher_means='better' need r1 > 1 + margin"
            )
        if higher_means == "worse" and r1 >= 1 - margin:
            raise ValueError(
                "with higher_means='worse' need r1 < 1 - margin"
            )
        n_used, achieved = _solve_even_n(pwr, power)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    return {
        "method_id": "superiority_by_margin_cross_over_ratios",
        "solve_for": solve_for,
        "n": n_used,
        "n_per_sequence": n_used // 2,
        "achieved_power": achieved,
        "sigma_w": sigma_w,
        "inputs_echo": inputs_echo,
        "citations": [
            "in a 2x2 Cross-Over Design using Ratios (Chapter 513).",
            *_CITATIONS_RATIOS,
        ],
    }
