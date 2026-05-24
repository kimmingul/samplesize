"""One-Way Repeated Measures ANOVA F-test power & sample size.


Within-subject one-factor design: each of N subjects is measured under K
repeated conditions/time points.  Hypotheses about equality of the K
condition means are tested with the univariate ANOVA F statistic,
optionally with the Geisser-Greenhouse / Box / Huynh-Feldt sphericity
correction ``ε`` applied to the F degrees of freedom.

Under compound symmetry the within-subject error variance is
``σ_e² = σ_w² · (1 - ρ)`` so the noncentral-F NCP simplifies to

    λ_0 = N · K · σ_m² / [σ_w² (1 - ρ)]
        = N · K · f²   / (1 - ρ)              (with f² = σ_m²/σ_w²)
    df₁ = (K - 1) · ε
    df₂ = (N - 1) · (K - 1) · ε
    λ   = λ_0 · ε                              (Box / Geisser-Greenhouse)

When ``ε = 1`` (sphericity satisfied) this reduces to the standard
univariate F test with df₁ = K-1 and df₂ = (N-1)(K-1).  For ε < 1 both
the degrees of freedom and the noncentrality are scaled by ε, which is
the conservative adjustment recommended by Box (1954)
when "Univariate: Geisser-Greenhouse F Test" is selected.

where
    σ_m² = Σ (μᵢ - μ̄)² / K   (population between-condition variance)
    μ̄    = Σ μᵢ / K

Inputs
------
* ``means``    list of K (≥2) condition means under H1
* ``sigma_w``  between-subject standard deviation at a single time point
               (assumed equal across time, as required by the F test)
* ``rho``      autocorrelation between repeated measurements
               (compound-symmetry correlation, 0 ≤ ρ < 1; default 0)
* ``epsilon``  sphericity / Geisser-Greenhouse correction
               (1 / (K-1) ≤ ε ≤ 1; default 1 → no correction)
* ``n``        number of subjects (solve for power)
* ``alpha``    significance level (default 0.05)
* ``power``    target power (solve for n)

Outputs
-------
``dict`` with ``method_id``, ``solve_for``, ``n``, ``achieved_power``,
``sigma_m`` (population SD of means), ``effect_f`` (Cohen's f =
σ_m / σ_w), ``df1``, ``df2``, ``ncp``, ``inputs_echo``, ``citations``.

References
----------
* Muller, K.E. and Barton, C.N. (1989).
* Muller, K.E., LaVange, L.E., Ramey, S.L., Ramey, C.T. (1992).
* Maxwell, S.E. and Delaney, H.D. (2003). *Designing Experiments and
  Analyzing Data*, 2nd Ed.
* Davis, C.S. (2002). *Statistical Methods for the Analysis of Repeated
  Measurements*.
"""
from __future__ import annotations

import math
from typing import Any

from samplesize.core import distributions as D


# ---------- helpers ---------------------------------------------------------


def _check_inputs(means, sigma_w, rho, epsilon):
    if len(means) < 2:
        raise ValueError("need at least 2 condition means (K >= 2)")
    if sigma_w <= 0:
        raise ValueError("sigma_w must be > 0")
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0, 1)")
    k = len(means)
    eps_min = 1.0 / (k - 1) if k > 1 else 1.0
    # allow small numerical slack
    if not (eps_min - 1e-9 <= epsilon <= 1.0 + 1e-9):
        raise ValueError(
            f"epsilon must be in [1/(K-1), 1] = [{eps_min:.4f}, 1.0]"
        )


def _sigma_m_squared(means: list[float]) -> float:
    """Population variance of the K means (divisor K, not K-1)."""
    k = len(means)
    mu_bar = sum(means) / k
    return sum((m - mu_bar) ** 2 for m in means) / k


# ---------- power at a given N ---------------------------------------------


def _power_rm(means, sigma_w, rho, epsilon, n, alpha):
    k = len(means)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    df1 = (k - 1) * epsilon
    df2 = (n - 1) * (k - 1) * epsilon
    if df1 <= 0 or df2 <= 0:
        return 0.0, df1, df2, 0.0
    sigma_m2 = _sigma_m_squared(means)
    ncp_base = n * k * sigma_m2 / (sigma_w ** 2 * (1.0 - rho))
    # Box / Geisser-Greenhouse: scale NCP by epsilon in tandem with df.
    ncp = ncp_base * epsilon
    from scipy.stats import f as fdist
    f_crit = fdist.ppf(1.0 - alpha, df1, df2)
    power = float(1.0 - D.ncf_cdf(f_crit, df1, df2, ncp))
    return power, df1, df2, ncp


def power_at_n(*, means, sigma_w, n, alpha=0.05, rho=0.0,
               epsilon=1.0) -> float:
    """Achieved power for given subject count ``n``."""
    _check_inputs(means, sigma_w, rho, epsilon)
    p, _, _, _ = _power_rm(means, sigma_w, rho, epsilon, n, alpha)
    return p


# ---------- solve for N -----------------------------------------------------


def n_for_power(*, means, sigma_w, alpha, power, rho=0.0, epsilon=1.0,
                n_min: int = 2, n_max: int = 1_000_000) -> tuple[int, float]:
    """Smallest N for which achieved power >= target power."""
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    _check_inputs(means, sigma_w, rho, epsilon)

    def p_at(n):
        p, *_ = _power_rm(means, sigma_w, rho, epsilon, n, alpha)
        return p

    lo, hi = n_min, n_min
    while hi <= n_max:
        if p_at(hi) >= power:
            break
        lo = hi
        hi = max(hi + 1, hi * 2)
    else:
        raise RuntimeError(f"failed to bracket N within {n_max}")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if p_at(mid) >= power:
            hi = mid
        else:
            lo = mid
    return hi, p_at(hi)


# ---------- public solver ---------------------------------------------------


def _build_cov_matrix(m: int, sigma: float, rho: float,
                      cov_type: str) -> list[list[float]]:
    """Build M x M covariance matrix for the specified correlation pattern."""
    sigma2 = sigma * sigma
    cov = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                cov[i][j] = sigma2
            else:
                gap = abs(i - j)
                if cov_type == "cs":            # compound symmetry (all rho equal)
                    cov[i][j] = sigma2 * rho
                elif cov_type == "ar1":         # AR(1)
                    cov[i][j] = sigma2 * (rho ** gap)
                elif cov_type == "banded1":     # Banded(1)
                    cov[i][j] = sigma2 * rho if gap == 1 else 0.0
                elif cov_type == "banded2":     # Banded(2)
                    cov[i][j] = sigma2 * rho if gap <= 2 else 0.0
                else:
                    raise ValueError(
                        f"cov_type must be one of 'cs', 'ar1', 'banded1', "
                        f"'banded2'; got {cov_type!r}"
                    )
    return cov


def _contrast_variance(contrast: list[float],
                       cov: list[list[float]]) -> float:
    """C'VC for contrast vector C and covariance matrix V."""
    m = len(contrast)
    total = 0.0
    for i in range(m):
        for j in range(m):
            total += contrast[i] * cov[i][j] * contrast[j]
    return total


def ci_one_way_repeated_measures_contrasts(
    *,
    m: int,
    contrast: list[float],
    sigma: float,
    rho: float,
    alpha: float = 0.05,
    distance: float | None = None,
    width: float | None = None,
    n: int | None = None,
    sides: int = 2,
    cov_type: str = "ar1",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """CI sample size for a contrast in a one-way repeated measures design.

    the confidence interval for the contrast ``C'μ`` is

        D = t_{1 - alpha/2, N-1} * sqrt(C'VC / N)

    where ``V = σ² * R`` is the M x M covariance matrix (specified via
    ``cov_type`` and ``rho``), C is the M-dimensional contrast coefficient
    vector, and N is the number of subjects.

    The smallest integer N with D ≤ target distance (half-width) is
    returned when ``solve_for='n'``; pass ``n`` to obtain the achieved D.

    Parameters
    ----------
    m : int
        Number of repeated measurements (time points) per subject. Must
        be >= 2.
    contrast : list of float
        Contrast coefficient vector of length M.  By convention the
        coefficients should sum to 0, but this is not enforced.
    sigma : float
        Within-subject standard deviation (assumed equal across time
        points).  Must be > 0.
    rho : float
        Correlation parameter for the covariance structure.  Must be in
        ``[0, 1)``.
    alpha : float
        Significance level; confidence level is ``1 - alpha``.
    distance : float, optional
        Target half-width D (required when ``sides=2`` and solving for N;
        "Distance from Contrast to Limit".
    width : float, optional
        Alias for ``2 * distance`` when sides=2.
    n : int, optional
        Fixed number of subjects; pass to compute achieved D.
    sides : int
        ``2`` for two-sided, ``1`` for one-sided.
    cov_type : {"cs", "ar1", "banded1", "banded2"}
        Covariance pattern: ``"cs"`` = compound symmetry (all ρ equal),
        ``"ar1"`` = AR(1), ``"banded1"`` = Banded(1), ``"banded2"`` =
        Banded(2).
    solve_for : {"n", "distance"}, optional
        Forced solve target. Defaults to ``"n"`` when ``n is None``.
    """
    if m < 2:
        raise ValueError("m must be >= 2")
    if len(contrast) != m:
        raise ValueError(
            f"contrast must have length m={m}; got {len(contrast)}"
        )
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    c = list(contrast)
    cov = _build_cov_matrix(m, sigma, rho, cov_type)
    ctvc = _contrast_variance(c, cov)
    contrast_sd = math.sqrt(ctvc)  # sqrt(C'VC)

    target = distance
    if target is None and width is not None:
        if sides != 2:
            raise ValueError("width can only be supplied when sides=2")
        target = width / 2.0

    if solve_for is None:
        solve_for = "n" if n is None else "distance"

    inputs_echo = {
        "m": m, "contrast": c, "sigma": sigma, "rho": rho,
        "alpha": alpha, "distance": distance, "width": width,
        "n": n, "sides": sides, "cov_type": cov_type,
    }

    from scipy.stats import t as tdist

    def _distance_at(nn: int) -> float:
        """Achieved half-width D = t_{1-α/sides/2, nn-1} * sqrt(C'VC/nn)."""
        if nn < 2:
            return math.inf
        q = 1.0 - alpha / (2.0 if sides == 2 else 1.0)
        t_val = float(tdist.ppf(q, nn - 1))
        return t_val * math.sqrt(ctvc / nn)

    n_floor = 2

    if solve_for == "n":
        if target is None or target <= 0:
            raise ValueError(
                "supply a positive `distance` or `width` when solving for n"
            )
        lo, hi = n_floor, n_floor
        while hi <= 10_000_000:
            if _distance_at(hi) <= target:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 10,000,000")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _distance_at(mid) <= target:
                hi = mid
            else:
                lo = mid
        n_req = hi
    elif solve_for == "distance":
        if n is None or n < n_floor:
            raise ValueError(
                f"supply n >= {n_floor} when solving for distance"
            )
        n_req = int(n)
    else:
        raise ValueError(f"unknown solve_for: {solve_for!r}")

    achieved_distance = _distance_at(n_req)

    return {
        "method_id": "ci_one_way_repeated_measures_contrasts",
        "solve_for": solve_for,
        "n": int(n_req),
        "m": int(m),
        "achieved_distance": float(achieved_distance),
        "achieved_width": float(2.0 * achieved_distance),
        "contrast_sd": float(contrast_sd),
        "inputs_echo": inputs_echo,
        "citations": [
            "One-Way Repeated Measures Contrasts.",
            "Rencher, A. C. (1998). Multivariate Statistical Inference "
            "and Applications. John Wiley, New York.",
            "Maxwell, S. E. and Delaney, H. D. (2003). Designing "
            "Experiments and Analyzing Data, 2nd Ed. Lawrence Erlbaum, "
            "New Jersey.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 431 — Two means in a repeated measures design (TAD)
# ---------------------------------------------------------------------------


def _build_R_matrix(m: int, rho: float, cov_type: str):
    """Build the M×M within-subject correlation matrix R.

    ``cov_type`` must be one of ``'cs'`` (compound symmetry), ``'ar1'``,
    ``'banded1'``, or ``'simple'``.
    """
    import numpy as np

    R = np.eye(m)
    for i in range(m):
        for j in range(m):
            if i != j:
                gap = abs(i - j)
                if cov_type == "cs":
                    R[i, j] = rho
                elif cov_type == "ar1":
                    R[i, j] = rho ** gap
                elif cov_type == "banded1":
                    R[i, j] = rho if gap == 1 else 0.0
                elif cov_type == "simple":
                    R[i, j] = 0.0
                else:
                    raise ValueError(
                        f"cov_type must be 'cs', 'ar1', 'banded1', or 'simple'; "
                        f"got {cov_type!r}"
                    )
    return R


def _var_beta1_tad(n1: int, n2: int, m: int, sigma2: float,
                   rho: float, cov_type: str) -> float:
    """Variance of β̂₁ (TAD treatment effect).

    Liu & Wu (2005) Eq. 4 / Diggle et al. (1994) p. 31:

        var(β̂₁) = σ² · (1/n1 + 1/n2) · (1'R1) / m²

    where ``1'R1`` is the sum of all elements of the M×M correlation
    matrix R, and m is the number of repeated measurements.
    """
    import numpy as np

    R = _build_R_matrix(m, rho, cov_type)
    ones = np.ones(m)
    ones_R_ones = float(ones @ R @ ones)   # 1'R1
    var = sigma2 * (1.0 / n1 + 1.0 / n2) * ones_R_ones / (m * m)
    return var


def _power_tad_means(n1: int, n2: int, m: int, d: float, sigma: float,
                     rho: float, cov_type: str, alpha: float,
                     sides: int) -> float:
    """Power for the TAD two-means test (Liu & Wu 2005 / Diggle et al. 1994)."""
    if n1 < 1 or n2 < 1:
        return 0.0
    sigma2 = sigma * sigma
    var_b1 = _var_beta1_tad(n1, n2, m, sigma2, rho, cov_type)
    sd_b1 = math.sqrt(var_b1)
    from scipy.stats import norm
    z_alpha = D.norm_ppf(1.0 - alpha / (1.0 if sides == 1 else 2.0))
    z_effect = abs(d) / sd_b1
    if sides == 2:
        return float(norm.cdf(z_effect - z_alpha) + norm.cdf(-z_effect - z_alpha))
    else:
        return float(1.0 - norm.cdf(z_alpha - z_effect))


def two_means_repeated_measures(
    *,
    d: float,
    m: int,
    sigma: float,
    rho: float = 0.0,
    cov_type: str = "cs",
    n1: int | None = None,
    n2: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-arm repeated measures design for the time-averaged difference in means.

    Measures Design) following Brown & Prescott (2006) / Liu & Wu (2005).

    Parameters
    ----------
    d : float
        Time-averaged difference (D1 = mean_group1 − mean_group2).
    m : int
        Number of repeated measurements per subject (M ≥ 1).
    sigma : float
        Standard deviation of a single observation (equal for both groups).
    rho : float
        Within-subject autocorrelation (0 ≤ ρ < 1).
    cov_type : {"cs", "ar1", "banded1", "simple"}
        Covariance structure: compound symmetry (cs), AR(1), Banded(1),
        or Simple (independent measurements).
    n1, n2 : int, optional
        Per-group sample sizes.  When ``solve_for='n'``, ``n1=n2`` is
        solved; when solving for power, both must be supplied.
    alpha : float
        Significance level.
    power : float, optional
        Target power (supply when ``solve_for='n'``).
    sides : {1, 2}
        One- or two-sided test.
    solve_for : {"n", "power"}, optional
        Defaults to ``"n"`` when ``n1`` is None.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    have_n = n1 is not None
    have_power = power is not None
    if not have_n and not have_power:
        raise ValueError("supply at least one of (n1, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    inputs_echo = {
        "d": d, "m": m, "sigma": sigma, "rho": rho, "cov_type": cov_type,
        "n1": n1, "n2": n2, "alpha": alpha, "power": power, "sides": sides,
    }

    if solve_for == "power":
        if n1 is None:
            raise ValueError("supply n1 when solve_for='power'")
        n1_out = int(n1)
        n2_out = int(n2) if n2 is not None else n1_out
        achieved = _power_tad_means(
            n1_out, n2_out, m, d, sigma, rho, cov_type, alpha, sides
        )
        n_out = n1_out + n2_out

    elif solve_for == "n":
        if not have_power or power is None:
            raise ValueError("supply power when solve_for='n'")
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")

        def p_at(nn: int) -> float:
            return _power_tad_means(nn, nn, m, d, sigma, rho, cov_type, alpha, sides)

        # bracket then bisect
        lo, hi = 2, 2
        while hi <= 1_000_000:
            if p_at(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 1,000,000 per group")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if p_at(mid) >= power:
                hi = mid
            else:
                lo = mid
        n1_out = hi
        n2_out = hi
        n_out = n1_out + n2_out
        achieved = p_at(n1_out)

    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "two_means_repeated_measures",
        "solve_for": solve_for,
        "n": n_out,
        "n1": n1_out,
        "n2": n2_out,
        "m": m,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Brown, H. & Prescott, R. (2006). Applied Mixed Models in Medicine, 2nd Ed., Ch. 6.",
            "Liu, H. & Wu, T. (2005). Sample Size Calculation and Power Analysis of "
            "Time-Averaged Difference. JMASM 4(2), 434-445.",
            "Diggle, P.J., Liang, K.Y., & Zeger, S.L. (1994). Analysis of Longitudinal "
            "Data. Oxford University Press.",
        ],
    }


# ---------------------------------------------------------------------------
# Chapter 201 — Two proportions in a repeated measures design (TAD)
# ---------------------------------------------------------------------------


def _power_tad_proportions_logit(
    n1: int, n2: int, m: int, p1: float, p2: float,
    rho: float, cov_type: str, alpha: float, sides: int,
) -> float:
    """Power for two-proportions TAD test on the logit scale (Brown & Prescott 2006).

    Uses the GLS information matrix with group-specific working variances:

        Vz_j^{-1} = p_j·q_j · R^{-1}

    so the total information for β̂₁ is

        [X'Vz^{-1}X]_{11} = s·n1·p1·q1·n2·p2·q2 / (n1·p1·q1 + n2·p2·q2)

    where s = 1'R^{-1}1 = sum of all elements of R^{-1}.  Under H0 the
    working variances coincide at p̄·q̄ for both groups.

    Power = 1 − Φ(r·z_α − d/σ_{H1}) + Φ(−r·z_α − d/σ_{H1})   [two-sided]
    where r = σ_{H0}/σ_{H1}  and d = |logit(p1) − logit(p2)|.
    """
    if n1 < 1 or n2 < 1:
        return 0.0
    import numpy as np
    from scipy.stats import norm

    R = _build_R_matrix(m, rho, cov_type)
    Rinv = np.linalg.inv(R)
    s = float(Rinv.sum())   # 1'R^{-1}1

    q1 = 1.0 - p1
    q2 = 1.0 - p2
    p_bar = (n1 * p1 + n2 * p2) / (n1 + n2)
    q_bar = 1.0 - p_bar

    # var(β̂₁ | H0):  working variance = p̄·q̄ for both groups
    # [X'Vz_H0^{-1}X]_{11} = s · n1·n2 / (n1+n2)  * 1/(p̄q̄) ... wait:
    # Under H0: all B_ii = p̄q̄, so Vz^{-1} = p̄q̄·R^{-1} per subject
    # X' Vz_H0^{-1} X = p̄q̄ · [[( n1+n2)s, n1·s],[n1·s, n1·s]]
    # [inv]_{11} = (n1+n2)s / (p̄q̄ · det)
    # det = p̄q̄ · s · [ n1·s·(n1+n2) - n1²·s ] · p̄q̄ = (p̄q̄)² · s² · n1·n2
    # => [inv]_{11} = 1 / (p̄q̄ · s · n2)  ... hmm, this yields asymmetric formula
    # Simpler: use the unbalanced formula (1/n1 + 1/n2) / (p̄q̄·s) matching the
    # chapter's text which says sigma^2_H0 = pooled p̄q̄ estimate.
    var_h0 = (1.0 / n1 + 1.0 / n2) / (p_bar * q_bar * s)

    # var(β̂₁ | H1):  group-specific working variances
    # [X'Vz_H1^{-1}X]_{11} = n1·p1q1·s  (lower-right corner of 2×2)
    # Full 2×2: A = [[(n1p1q1+n2p2q2)s, n1p1q1·s], [n1p1q1·s, n1p1q1·s]]
    # det(A) = s² · n1·p1q1 · n2·p2q2
    # [A^{-1}]_{11} = (n1p1q1+n2p2q2)·s / (s² · n1·p1q1 · n2·p2q2)
    var_h1 = (n1 * p1 * q1 + n2 * p2 * q2) / (s * n1 * p1 * q1 * n2 * p2 * q2)

    sd_h0 = math.sqrt(var_h0)
    sd_h1 = math.sqrt(var_h1)
    d = abs(math.log(p1 / q1) - math.log(p2 / q2))

    z_alpha = D.norm_ppf(1.0 - alpha / (1.0 if sides == 1 else 2.0))
    ratio = sd_h0 / sd_h1
    arg = ratio * z_alpha - d / sd_h1
    power = float(1.0 - norm.cdf(arg))
    if sides == 2:
        power += float(norm.cdf(-ratio * z_alpha - d / sd_h1))
    return min(1.0, max(0.0, power))


def two_proportions_repeated_measures(
    *,
    p1: float,
    p2: float,
    m: int,
    rho: float = 0.0,
    cov_type: str = "cs",
    n1: int | None = None,
    n2: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two-arm repeated measures design for the time-averaged difference in proportions.

    Measures Design) using the logit-link test statistic following Brown &
    Prescott (2006) / Liu & Wu (2005).

    Parameters
    ----------
    p1, p2 : float
        Success proportions in groups 1 and 2 under H1.
        The effect is d = logit(p1) − logit(p2) = log(OR).
    m : int
        Number of repeated measurements per subject (M ≥ 1).
    rho : float
        Within-subject autocorrelation (0 ≤ ρ < 1).
    cov_type : {"cs", "ar1", "banded1", "simple"}
        Covariance structure.
    n1, n2 : int, optional
        Per-group sample sizes.
    alpha : float
        Significance level.
    power : float, optional
        Target power (supply when ``solve_for='n'``).
    sides : {1, 2}
        One- or two-sided test.
    solve_for : {"n", "power"}, optional
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0):
        raise ValueError("p1 and p2 must be in (0, 1)")
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    have_n = n1 is not None
    have_power = power is not None
    if not have_n and not have_power:
        raise ValueError("supply at least one of (n1, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    inputs_echo = {
        "p1": p1, "p2": p2, "m": m, "rho": rho, "cov_type": cov_type,
        "n1": n1, "n2": n2, "alpha": alpha, "power": power, "sides": sides,
    }

    if solve_for == "power":
        if n1 is None:
            raise ValueError("supply n1 when solve_for='power'")
        n1_out = int(n1)
        n2_out = int(n2) if n2 is not None else n1_out
        achieved = _power_tad_proportions_logit(
            n1_out, n2_out, m, p1, p2, rho, cov_type, alpha, sides
        )
        n_out = n1_out + n2_out

    elif solve_for == "n":
        if not have_power or power is None:
            raise ValueError("supply power when solve_for='n'")
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")

        def p_at(nn: int) -> float:
            return _power_tad_proportions_logit(
                nn, nn, m, p1, p2, rho, cov_type, alpha, sides
            )

        lo, hi = 2, 2
        while hi <= 1_000_000:
            if p_at(hi) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 1,000,000 per group")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if p_at(mid) >= power:
                hi = mid
            else:
                lo = mid
        n1_out = hi
        n2_out = hi
        n_out = n1_out + n2_out
        achieved = p_at(n1_out)

    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    # Compute odds ratio for reporting
    odds_ratio = (p1 / (1.0 - p1)) / (p2 / (1.0 - p2))

    return {
        "method_id": "two_proportions_repeated_measures",
        "solve_for": solve_for,
        "n": n_out,
        "n1": n1_out,
        "n2": n2_out,
        "m": m,
        "odds_ratio": odds_ratio,
        "achieved_power": achieved,
        "inputs_echo": inputs_echo,
        "citations": [
            "Brown, H. & Prescott, R. (2006). Applied Mixed Models in Medicine, 2nd Ed., p. 270.",
            "Liu, H. & Wu, T. (2005). Sample Size Calculation and Power Analysis of "
            "Time-Averaged Difference. JMASM 4(2), 434-445.",
            "Diggle, P.J., Liang, K.Y., & Zeger, S.L. (1994). Analysis of Longitudinal "
            "Data. Oxford University Press.",
        ],
    }


def one_way_repeated_measures(
    *,
    means: list[float],
    sigma_w: float,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    rho: float = 0.0,
    epsilon: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Solve a one-way repeated-measures ANOVA F-test design.

    Provide ``n`` to solve for ``achieved_power``, or ``power`` to solve
    for the minimum ``n`` that achieves it.  ``rho`` is the compound-
    symmetry correlation between repeated measurements; ``epsilon`` is
    the Geisser-Greenhouse / Box sphericity correction (set 1 for the
    pure F-test, 1/(K-1) for the most conservative bound).
    """
    inputs_echo = {
        "means": list(means), "sigma_w": sigma_w, "n": n, "alpha": alpha,
        "power": power, "rho": rho, "epsilon": epsilon,
    }
    have_n = n is not None
    have_power = power is not None
    if not (have_n or have_power):
        raise ValueError("supply at least one of (n, power)")
    if solve_for is None:
        solve_for = "n" if not have_n else "power"

    _check_inputs(means, sigma_w, rho, epsilon)
    sigma_m2 = _sigma_m_squared(means)
    sigma_m = math.sqrt(sigma_m2)
    effect_f = sigma_m / sigma_w
    k = len(means)

    if solve_for == "power":
        assert n is not None
        achieved, df1, df2, ncp = _power_rm(
            means, sigma_w, rho, epsilon, n, alpha
        )
        n_out = n
    elif solve_for == "n":
        assert power is not None
        n_out, achieved = n_for_power(
            means=means, sigma_w=sigma_w, alpha=alpha, power=power,
            rho=rho, epsilon=epsilon,
        )
        _, df1, df2, ncp = _power_rm(
            means, sigma_w, rho, epsilon, n_out, alpha
        )
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "one_way_repeated_measures",
        "solve_for": solve_for,
        "n": n_out,
        "k_conditions": k,
        "achieved_power": achieved,
        "sigma_m": sigma_m,
        "effect_f": effect_f,
        "df1": df1,
        "df2": df2,
        "ncp": ncp,
        "inputs_echo": inputs_echo,
        "citations": [
            "Muller, K.E., LaVange, L.E., Ramey, S.L., & Ramey, C.T. (1992). "
            "Power Calculations for General Linear Multivariate Models "
            "Including Repeated Measures Applications. JASA 87(420), 1209-1226.",
            "Muller, K.E. & Barton, C.N. (1989). Approximate Power for "
            "Repeated-Measures ANOVA Lacking Sphericity. JASA 84(406), 549-555.",
            "Maxwell, S.E. & Delaney, H.D. (2003). Designing Experiments "
            "and Analyzing Data, 2nd Ed.",
        ],
    }
