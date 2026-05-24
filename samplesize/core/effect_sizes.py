"""Effect-size conversions and helpers.

Cohen's d / h / f, hazard ratio, odds ratio conversions. Kept
side-effect-free; never echoes or rounds — that is the calculation
function's job.
"""
from __future__ import annotations

import math


def cohens_d(mean1: float, mean2: float, sd: float) -> float:
    """Standardized mean difference (pooled SD assumed by caller)."""
    return (mean1 - mean2) / sd


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions (arcsine transformation)."""
    phi1 = 2.0 * math.asin(math.sqrt(p1))
    phi2 = 2.0 * math.asin(math.sqrt(p2))
    return phi1 - phi2


def odds_ratio(p1: float, p2: float) -> float:
    return (p1 / (1.0 - p1)) / (p2 / (1.0 - p2))


def hazard_ratio_from_survivals(s1: float, s2: float) -> float:
    """HR = log(S2) / log(S1), the proportional-hazards relationship
    between two survival probabilities at a common time."""
    return math.log(s2) / math.log(s1)
