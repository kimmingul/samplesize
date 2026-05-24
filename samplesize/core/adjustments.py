"""Post-hoc adjustments applied to a base sample size.

Dropout/loss-to-followup inflation, multiple-testing α adjustment, and
finite-population correction. Each is a pure function so it can be
composed and audited.
"""
from __future__ import annotations

import math


def inflate_for_dropout(n: int, dropout_rate: float) -> int:
    """Inflate base N to compensate for expected dropout.

    Standard formula: N_inflated = ceil(N / (1 - dropout_rate)).
    """
    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(f"dropout_rate must be in [0, 1): got {dropout_rate}")
    return math.ceil(n / (1.0 - dropout_rate))


def bonferroni(alpha: float, n_tests: int) -> float:
    """Bonferroni-corrected α for `n_tests` independent tests."""
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    return alpha / n_tests


def finite_population_correction(n: int, population: int) -> int:
    """Reduce N when sampling fraction is non-negligible.

    Finite-population correction: sigma1^2 = (1 - n/N) * sigma^2.  Here we just return the
    adjusted N solving n_adj / (1 + (n_adj - 1)/population) = n.
    """
    if population <= 0:
        raise ValueError("population must be > 0")
    if n >= population:
        return population
    return math.ceil(n * population / (n + population - 1))
