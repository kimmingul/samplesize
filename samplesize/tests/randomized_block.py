"""Power and sample-size for Randomized Complete Block Design ANOVA.

Implements:

* "Randomized Block Analysis of Variance"
  -> :func:`randomized_block_anova`

The randomized complete block design (RCBD) has ``k`` treatments applied once
each within each of ``b`` blocks.  The model is:

    Y_ij = mu + tau_i + beta_j + epsilon_ij

where tau_i are fixed treatment effects and beta_j are random block effects
(blocking variable treated as a random factor in the power computation).

Power formula
-------------
The F-test for treatment effects in the RCBD uses the mean square ratio:

    F = MS_Treatments / MS_Error

Under H1 with treatment means mu_1, ..., mu_k and within-cell error sigma:

    df_treatments = k - 1
    df_error      = (k - 1)(b - 1)
    lambda (NCP)  = b * sum_i (mu_i - mu_bar)^2 / sigma^2

where mu_bar is the unweighted mean of the treatment means and b is the
number of blocks (= number of replicates per treatment).

This follows from the standard noncentral-F power derivation for the
randomized block design as given in Cochran & Cox (1957) and Fleiss (1986).

References
----------
Cochran, W. G. and Cox, G. M. (1957). Experimental Designs, 2nd ed.
Wiley, New York.  Chapter 4.

Fleiss, J. L. (1986). The Design and Analysis of Clinical Experiments.
Wiley.  Chapter 14.

Lenth, R. V. (2001). Some practical guidelines for effective sample size
determination. American Statistician, 55, 187-193.
"""
from __future__ import annotations

from typing import Any

from scipy.stats import f as _fdist


# ---------------------------------------------------------------------------
# Core power function
# ---------------------------------------------------------------------------

def _rcbd_power(
    *,
    means: list[float],
    sigma: float,
    n_blocks: int,
    alpha: float,
) -> float:
    """Power of the RCBD F-test for treatment effects.

    Parameters
    ----------
    means
        Treatment means under H1 (length k >= 2).
    sigma
        Within-cell error standard deviation.
    n_blocks
        Number of blocks (b >= 2).
    alpha
        Significance level.

    Returns
    -------
    float
        Power (probability of rejecting H0).
    """
    k = len(means)
    b = n_blocks
    if b < 2 or k < 2:
        return 0.0

    mu_bar = sum(means) / k
    ss_treatments = sum((m - mu_bar) ** 2 for m in means)

    # NCP = b * SS_treatments / sigma^2
    ncp = b * ss_treatments / (sigma ** 2)

    df1 = k - 1
    df2 = (k - 1) * (b - 1)

    f_crit = float(_fdist.ppf(1.0 - alpha, df1, df2))
    # Power = 1 - F'(f_crit; df1, df2, ncp)  where F' is noncentral F CDF
    from scipy.stats import ncf as _ncf
    power = float(1.0 - _ncf.cdf(f_crit, df1, df2, ncp))
    return power


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def randomized_block_anova(
    *,
    means: list[float],
    sigma: float,
    alpha: float = 0.05,
    power: float | None = None,
    n_blocks: int | None = None,
    solve_for: str | None = None,
) -> dict[str, Any]:
    """Power / sample size for a randomized complete block design ANOVA.

    Tests H0: all treatment means equal vs H1: at least one treatment
    mean differs, in a two-way layout with ``k`` treatments and ``b`` blocks
    (one observation per treatment-block cell).

    Provide exactly two of (power, n_blocks).

    Parameters
    ----------
    means
        Treatment means under H1 (list of k >= 2 floats).
    sigma
        Within-cell error standard deviation (> 0).
    alpha
        Significance level (default 0.05).
    power
        Target power (required when solve_for='n' or omitting n_blocks).
    n_blocks
        Number of blocks b (required when solve_for='power' or omitting power).
    solve_for
        ``'n'`` to solve for number of blocks, ``'power'`` to compute power.
        Inferred automatically when omitted.

    Returns
    -------
    dict
        Standard envelope: method_id, solve_for, n, n_blocks,
        achieved_power, inputs_echo, citations.
    """
    # Input validation
    if len(means) < 2:
        raise ValueError("means must have at least 2 elements")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    inputs_echo = {
        "means": means,
        "sigma": sigma,
        "alpha": alpha,
        "power": power,
        "n_blocks": n_blocks,
        "solve_for": solve_for,
    }

    # Infer solve_for
    if solve_for is None:
        if n_blocks is None and power is not None:
            solve_for = "n"
        elif n_blocks is not None and power is None:
            solve_for = "power"
        else:
            raise ValueError("supply exactly one of (n_blocks, power)")

    citations = [
        "Cochran, W. G. and Cox, G. M. (1957). Experimental Designs, 2nd ed. "
        "Wiley, New York.",
        "Fleiss, J. L. (1986). The Design and Analysis of Clinical Experiments. "
        "Wiley.",
        "Lenth, R. V. (2001). Some practical guidelines for effective sample size "
        "determination. American Statistician, 55, 187-193.",
    ]

    k = len(means)

    if solve_for == "power":
        if n_blocks is None:
            raise ValueError("n_blocks is required when solve_for='power'")
        if n_blocks < 2:
            raise ValueError("n_blocks must be >= 2")
        achieved = _rcbd_power(
            means=means, sigma=sigma, n_blocks=n_blocks, alpha=alpha
        )
        return {
            "method_id": "randomized_block_anova",
            "solve_for": "power",
            "n_blocks": n_blocks,
            "n_treatments": k,
            "n": n_blocks * k,
            "achieved_power": achieved,
            "inputs_echo": inputs_echo,
            "citations": citations,
        }

    elif solve_for == "n":
        if power is None:
            raise ValueError("power is required when solve_for='n'")
        if not 0.0 < power < 1.0:
            raise ValueError("power must be in (0, 1)")

        # Check if all means are equal — no power possible
        mu_bar = sum(means) / k
        if all(abs(m - mu_bar) < 1e-15 for m in means):
            raise ValueError(
                "all treatment means are equal; cannot solve for n_blocks"
            )

        # Search for smallest b >= 2 that achieves target power
        b = 2
        b_max = 1_000_000
        while b <= b_max:
            achieved = _rcbd_power(
                means=means, sigma=sigma, n_blocks=b, alpha=alpha
            )
            if achieved >= power:
                break
            b += 1
        else:
            raise RuntimeError("failed to find n_blocks within limit")

        return {
            "method_id": "randomized_block_anova",
            "solve_for": "n",
            "n_blocks": b,
            "n_treatments": k,
            "n": b * k,
            "achieved_power": achieved,
            "inputs_echo": inputs_echo,
            "citations": citations,
        }

    else:
        raise ValueError(f"unsupported solve_for: {solve_for!r}")
