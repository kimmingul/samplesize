"""Specialty correlation tests.

Four chapters are implemented in this module:

* ``tests_two_correlations`` -- comparison of two independent correlations
  via the Fisher z-transform (see Meng, Rosenthal & Rubin 1992;
  Ch. 805).  Test statistic ``z = (z1 - z2) / sqrt(1/(N1-3) + 1/(N2-3))``.
* ``lin_concordance_correlation`` -- Lin's concordance correlation
  coefficient (see Lin 1989;
  Ch. 812).  One-sided z-test on the Fisher-z transform of CCC with the
  Lin (1989, 1992, 2002, 2012) variance formula.
* ``tests_intraclass_correlation`` -- one-way random-effects ICC F-test
  (Walter et al. 1998; Shrout & Fleiss 1979).  Power is
  ``1 - F_cdf( C0 * F_{1-alpha,N-1,N(K-1)} )`` per Walter, Eliasziw &
  Donner (1998) and Winer (1991).
* ``point_biserial_correlation`` -- point-biserial correlation test with
  a random binary covariate (Tate 1954; see
  Tests", Ch. 807).  Power is a binomial mixture over n1 of non-central
  t tail probabilities with noncentrality
  ``delta_R = rho/sqrt(1-rho^2) * sqrt(n1*n0/(n*p*q))`` per Tate (1954).

All routines follow the project convention:

* keyword-only parameters,
* the user supplies exactly one of ``n``/``power`` (or ``n1`` / ``power``
  for the two-correlation routine),
* return a dict ``{method_id, solve_for, n (or n1, n2), achieved_power,
  inputs_echo, citations}``.
"""
from __future__ import annotations

import math
from typing import Any

from scipy.stats import binom, f as fdist, nct, norm, t as tdist


# ---------------------------------------------------------------------------
# 1. Tests for Two Correlations  (Fisher z-difference)
# ---------------------------------------------------------------------------

def _fisher_z(rho: float) -> float:
    return 0.5 * math.log((1.0 + rho) / (1.0 - rho))


def _two_corr_power(
    rho1: float, rho2: float, n1: int, n2: int,
    alpha: float, sides: int, alternative: str,
) -> float:
    if n1 <= 3 or n2 <= 3:
        return 0.0
    Z1 = _fisher_z(rho1)
    Z2 = _fisher_z(rho2)
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    delta = Z1 - Z2
    if sides == 2:
        z = norm.ppf(1.0 - alpha / 2.0)
        upper = 1.0 - norm.cdf(z - delta / se)
        lower = norm.cdf(-z - delta / se)
        return float(upper + lower)
    z = norm.ppf(1.0 - alpha)
    if alternative == "greater":  # H1: rho1 > rho2
        return float(1.0 - norm.cdf(z - delta / se))
    if alternative == "less":  # H1: rho1 < rho2
        return float(norm.cdf(-z - delta / se))
    raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less', got {alternative!r}")


def tests_two_correlations(
    *,
    rho1: float,
    rho2: float,
    n1: int | None = None,
    n2: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    alternative: str = "two-sided",
    ratio: float = 1.0,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Two independent correlation comparison (Fisher z difference).

    Solve modes:
      * supply both ``n1`` and ``n2`` and omit ``power`` -- returns power.
      * supply ``power`` and omit ``n1``/``n2`` -- solves for the per-group
        sample sizes with ``n2 = ceil(ratio * n1)`` (default ratio=1 i.e.
        balanced design).
    """
    inputs_echo = {
        "rho1": rho1, "rho2": rho2, "n1": n1, "n2": n2,
        "alpha": alpha, "power": power, "sides": sides,
        "alternative": alternative, "ratio": ratio,
    }
    if not -1.0 < rho1 < 1.0 or not -1.0 < rho2 < 1.0:
        raise ValueError("rho values must lie in (-1, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")

    have_n = n1 is not None and n2 is not None
    have_power = power is not None
    if have_n == have_power:
        raise ValueError("supply exactly one of (n1+n2) or power")

    if solve_for is None:
        solve_for = "power" if have_n else "n"

    if solve_for == "power":
        assert n1 is not None and n2 is not None
        achieved = _two_corr_power(rho1, rho2, n1, n2,
                                   alpha, sides, alternative)
        result = {"n1": int(n1), "n2": int(n2),
                  "achieved_power": float(achieved)}
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        if rho1 == rho2:
            raise ValueError("cannot solve for N when rho1 == rho2")
        # search per-group sample size n1 with n2 = ceil(ratio * n1)
        lo, hi = 4, 4
        max_n = 10_000_000
        while hi <= max_n:
            n2_hi = max(4, int(math.ceil(ratio * hi)))
            if _two_corr_power(rho1, rho2, hi, n2_hi,
                               alpha, sides, alternative) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N1 within 1e7")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            n2_mid = max(4, int(math.ceil(ratio * mid)))
            if _two_corr_power(rho1, rho2, mid, n2_mid,
                               alpha, sides, alternative) >= power:
                hi = mid
            else:
                lo = mid
        n1_req = hi
        n2_req = max(4, int(math.ceil(ratio * n1_req)))
        achieved = _two_corr_power(rho1, rho2, n1_req, n2_req,
                                   alpha, sides, alternative)
        result = {"n1": int(n1_req), "n2": int(n2_req),
                  "achieved_power": float(achieved)}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "tests_two_correlations",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Fisher, R.A. (1921). On the 'probable error' of a coefficient "
            "of correlation deduced from a small sample.",
            "Zar, J.H. (1984). Biostatistical Analysis (2nd ed.), pp.312-315.",
        ],
    }


# ---------------------------------------------------------------------------
# 2. Lin's Concordance Correlation Coefficient
# ---------------------------------------------------------------------------

def _ccc(rho: float, v: float, omega: float) -> float:
    return rho * 2.0 / (v * v + omega + 1.0 / omega)


def _lin_sigma2(rho: float, v: float, omega: float, n: int) -> float:
    """Lin (2002 Eq. on p.260; 2012 Ch.4) asymptotic variance of
    tanh^{-1}(CCC_hat)."""
    CCC = _ccc(rho, v, omega)
    one_minus_CCC2 = 1.0 - CCC * CCC
    if one_minus_CCC2 <= 0 or rho == 0:
        return float("inf")
    rho2 = rho * rho
    term1 = (1.0 - rho2) * CCC * CCC / (one_minus_CCC2 * rho2)
    term2 = 2.0 * CCC**3 * (1.0 - CCC) * v * v \
        / (rho * one_minus_CCC2 ** 2)
    term3 = CCC**4 * v**4 / (2.0 * rho2 * one_minus_CCC2 ** 2)
    return (1.0 / (n - 2)) * (term1 + term2 - term3)


def _lin_power(
    rho0: float, rho1: float,
    v0: float, v1: float,
    w0: float, w1: float,
    n: int, alpha: float,
) -> float:
    if n <= 2:
        return 0.0
    CCC0 = _ccc(rho0, v0, w0)
    CCC1 = _ccc(rho1, v1, w1)
    if not -1.0 < CCC0 < 1.0 or not -1.0 < CCC1 < 1.0:
        raise ValueError("derived CCC must lie in (-1, 1)")
    lam0 = math.atanh(CCC0)
    lam1 = math.atanh(CCC1)
    sigma0 = math.sqrt(_lin_sigma2(rho0, v0, w0, n))
    sigma1 = math.sqrt(_lin_sigma2(rho1, v1, w1, n))
    z_alpha = norm.ppf(1.0 - alpha)
    arg = ((lam0 - lam1) + z_alpha * sigma0) / sigma1
    return float(1.0 - norm.cdf(arg))


def lin_concordance_correlation(
    *,
    rho0: float,
    rho1: float,
    v0: float = 0.0,
    v1: float = 0.0,
    omega0: float = 1.0,
    omega1: float = 1.0,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    solve_for: str | None = None,
    ccc0: float | None = None,
    ccc1: float | None = None,
) -> dict[str, Any]:
    """Power / sample size for Lin's CCC one-sided non-inferiority test.

    Hypotheses are H0: CCC <= CCC0 vs H1: CCC > CCC0, evaluated at
    ``CCC1`` via the Fisher-z transform of CCC.  In the *simple* mode
    pass ``ccc0`` and ``ccc1`` (which set ``rho = CCC``, ``v = 0``,
    ``omega = 1``).  In *general* mode pass ``rho0, rho1, v0, v1,
    omega0, omega1``.
    """
    if ccc0 is not None or ccc1 is not None:
        if ccc0 is None or ccc1 is None:
            raise ValueError("supply both ccc0 and ccc1 in simple mode")
        rho0, rho1 = ccc0, ccc1
        v0 = v1 = 0.0
        omega0 = omega1 = 1.0

    inputs_echo = {
        "rho0": rho0, "rho1": rho1,
        "v0": v0, "v1": v1, "omega0": omega0, "omega1": omega1,
        "n": n, "alpha": alpha, "power": power,
    }
    if rho0 <= 0 or rho1 <= 0:
        raise ValueError("rho0 and rho1 must be positive (CCC sample-size formula assumes positive precision)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    have_n = n is not None
    have_power = power is not None
    if have_n == have_power:
        raise ValueError("supply exactly one of (n, power)")

    if solve_for is None:
        solve_for = "power" if have_n else "n"

    if solve_for == "power":
        assert n is not None
        achieved = _lin_power(rho0, rho1, v0, v1, omega0, omega1, n, alpha)
        result = {"n": int(n), "achieved_power": float(achieved),
                  "ccc0": _ccc(rho0, v0, omega0),
                  "ccc1": _ccc(rho1, v1, omega1)}
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 3, 3
        max_n = 10_000_000
        while hi <= max_n:
            if _lin_power(rho0, rho1, v0, v1, omega0, omega1,
                          hi, alpha) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 1e7")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _lin_power(rho0, rho1, v0, v1, omega0, omega1,
                          mid, alpha) >= power:
                hi = mid
            else:
                lo = mid
        n_req = hi
        achieved = _lin_power(rho0, rho1, v0, v1, omega0, omega1,
                              n_req, alpha)
        result = {"n": int(n_req), "achieved_power": float(achieved),
                  "ccc0": _ccc(rho0, v0, omega0),
                  "ccc1": _ccc(rho1, v1, omega1)}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "lin_concordance_correlation",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Lin, L.I-K. (1989). A concordance correlation coefficient to "
            "evaluate reproducibility. Biometrics 45, 255-268.",
            "Lin, L.I-K. (1992). Assay validation using the concordance "
            "correlation coefficient. Biometrics 48, 599-604.",
            "Lin, L., Hedayat, A.S., Sinha, B, Yang, M. (2002). Statistical "
            "methods in assessing agreement. JASA 97, 257-270.",
            "Lin, L., Hedayat, A.S., Wu, W. (2012). Statistical Tools for "
            "Measuring Agreement. Springer.",
        ],
    }


# ---------------------------------------------------------------------------
# 3. Tests for Intraclass Correlation (one-way random-effects F test)
# ---------------------------------------------------------------------------

def _icc_power(
    rho0: float, rho1: float, N: int, K: int,
    alpha: float, sides: int,
) -> float:
    if N < 2 or K < 2:
        return 0.0
    if not 0.0 <= rho0 < 1.0 or not 0.0 <= rho1 < 1.0:
        raise ValueError("rho values must lie in [0, 1)")
    df1 = N - 1
    df2 = N * (K - 1)
    C0 = (1.0 + K * rho0 / (1.0 - rho0)) / (1.0 + K * rho1 / (1.0 - rho1))
    # critical value in the reported outputs; the printed
    # "F_{1-alpha/2}" in the formula box is a transcription artefact
    # — Walter et al. (1998) and the chapter's worked examples both
    # use F_{1-alpha}.
    a = alpha / 2.0 if sides == 2 else alpha
    F_crit = fdist.ppf(1.0 - a, df1, df2)
    return float(1.0 - fdist.cdf(C0 * F_crit, df1, df2))


def tests_intraclass_correlation(
    *,
    rho0: float,
    rho1: float,
    K: int,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 1,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Intraclass correlation F-test (one-way random effects).

    ``n`` is the number of subjects; ``K`` is the
    number of replications per subject.  The hypothesis is one-sided
    by default (``H0: rho = rho0`` vs ``H1: rho = rho1 > rho0``), which
    matches Walter et al. (1998).
    810.  ``sides=2`` uses ``alpha/2`` for the F critical value.
    """
    inputs_echo = {
        "rho0": rho0, "rho1": rho1, "K": K, "n": n,
        "alpha": alpha, "power": power, "sides": sides,
    }
    if rho1 <= rho0:
        raise ValueError("rho1 must be greater than rho0")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    if K < 2:
        raise ValueError("K (observations per subject) must be >= 2")

    have_n = n is not None
    have_power = power is not None
    if have_n == have_power:
        raise ValueError("supply exactly one of (n, power)")

    if solve_for is None:
        solve_for = "power" if have_n else "n"

    if solve_for == "power":
        assert n is not None
        achieved = _icc_power(rho0, rho1, n, K, alpha, sides)
        result = {"n": int(n), "achieved_power": float(achieved)}
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 2, 2
        max_n = 1_000_000
        while hi <= max_n:
            if _icc_power(rho0, rho1, hi, K, alpha, sides) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 1e6")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _icc_power(rho0, rho1, mid, K, alpha, sides) >= power:
                hi = mid
            else:
                lo = mid
        n_req = hi
        achieved = _icc_power(rho0, rho1, n_req, K, alpha, sides)
        result = {"n": int(n_req), "achieved_power": float(achieved)}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "tests_intraclass_correlation",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Walter, S.D., Eliasziw, M., Donner, A. (1998). Sample size and "
            "optimal designs for reliability studies. Statistics in Medicine "
            "17, 101-110.",
            "Winer, B.J. (1991). Statistical Principles in Experimental "
            "Design (3rd ed.).",
        ],
    }


# ---------------------------------------------------------------------------
# 4. Point-Biserial Correlation Tests (random binary covariate)
# ---------------------------------------------------------------------------

def _pb_power(
    rho1: float, rho0: float, n: int, p: float,
    alpha: float, sides: int, alternative: str,
) -> float:
    if n < 4:
        return 0.0
    if rho0 != 0.0:
        # The classic Tate / Lev derivation sets the
        # critical value under H0 from a *central* t with df=n-2,
        # which is exact only when rho0 = 0.  We retain rho0 as an
        # argument so the API is consistent with the rest of the
        # correlation family, but warn the caller via ValueError when
        # they try a non-zero null --- supporting rho0 != 0 would
        # require iterating until the size of the test equals alpha.
        raise ValueError(
            "rho0 != 0 is not supported by the Tate (1954) point-biserial "
            "power formula implemented here; use the simulation backend "
            "for non-zero nulls."
        )
    if not -1.0 < rho1 < 1.0:
        raise ValueError("rho1 must lie in (-1, 1)")
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    df = n - 2
    q = 1.0 - p
    if sides == 2:
        t_crit = tdist.ppf(1.0 - alpha / 2.0, df)
    elif sides == 1:
        t_crit = tdist.ppf(1.0 - alpha, df)
    else:
        raise ValueError("sides must be 1 or 2")

    total = 0.0
    sign = 1.0 if rho1 >= 0 else -1.0
    rho1_abs = abs(rho1)
    for n1 in range(1, n):
        n0 = n - n1
        w = binom.pmf(n1, n, p)
        if w <= 0.0:
            continue
        nc = sign * rho1_abs / math.sqrt(1.0 - rho1_abs * rho1_abs) \
            * math.sqrt(n1 * n0 / (n * p * q))
        if sides == 2:
            pwr_n1 = (1.0 - nct.cdf(t_crit, df, nc)) + nct.cdf(-t_crit, df, nc)
        else:
            if alternative == "greater":
                pwr_n1 = 1.0 - nct.cdf(t_crit, df, nc)
            elif alternative == "less":
                pwr_n1 = nct.cdf(-t_crit, df, nc)
            else:
                raise ValueError(
                    "alternative must be 'greater' or 'less' for one-sided tests"
                )
        total += w * pwr_n1
    return float(total)


def point_biserial_correlation(
    *,
    rho1: float,
    p: float,
    rho0: float = 0.0,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    sides: int = 2,
    alternative: str = "two-sided",
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Point-biserial correlation test against zero (random design).

    Continuous Y with N(mu_1, sigma) when X=1 and N(mu_0, sigma) when
    X=0, where X is Bernoulli(p).  Implements the Tate (1954) binomial
    mixture of non-central t powers (Tate 1954).  Only
    ``rho0 = 0`` is currently supported.
    """
    inputs_echo = {
        "rho1": rho1, "rho0": rho0, "p": p, "n": n,
        "alpha": alpha, "power": power, "sides": sides,
        "alternative": alternative,
    }

    have_n = n is not None
    have_power = power is not None
    if have_n == have_power:
        raise ValueError("supply exactly one of (n, power)")

    if solve_for is None:
        solve_for = "power" if have_n else "n"

    if solve_for == "power":
        assert n is not None
        achieved = _pb_power(rho1, rho0, n, p, alpha, sides, alternative)
        result = {"n": int(n), "achieved_power": float(achieved)}
    elif solve_for == "n":
        assert power is not None
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")
        lo, hi = 4, 4
        max_n = 200_000
        while hi <= max_n:
            if _pb_power(rho1, rho0, hi, p, alpha, sides,
                         alternative) >= power:
                break
            lo = hi
            hi = max(hi + 1, hi * 2)
        else:
            raise RuntimeError("failed to bracket N within 2e5")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _pb_power(rho1, rho0, mid, p, alpha, sides,
                         alternative) >= power:
                hi = mid
            else:
                lo = mid
        n_req = hi
        achieved = _pb_power(rho1, rho0, n_req, p, alpha, sides, alternative)
        result = {"n": int(n_req), "achieved_power": float(achieved)}
    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")

    return {
        "method_id": "point_biserial_correlation",
        "solve_for": solve_for,
        **result,
        "inputs_echo": inputs_echo,
        "citations": [
            "Tate, R.F. (1954). Correlation between a discrete and a "
            "continuous variable. Point-biserial correlation. Annals of "
            "Mathematical Statistics 25, 603-607.",
            "Lev, J. (1949). The point biserial coefficient of correlation. "
            "Annals of Mathematical Statistics 20, 125-126.",
        ],
    }
