# Method Coverage

Tracks which methods are implemented and validated.
Update whenever a method moves between states.

## Legend

- ⚪ Not implemented
- 🟡 Implemented, not validated against reference examples
- 🟢 Implemented and validated to documented tolerance against reference examples

## Tier 1 (MVP) — 8 / 8 complete

| id | name | status | fixtures |
|---|---|---|---|
| `one_sample_t` | Tests for One Mean | 🟢 | 6 |
| `two_sample_t_equal_var` | Two-Sample T-Tests Assuming Equal Variance | 🟢 | 3 |
| `paired_t` | Tests for Paired Means | 🟢 | 4 |
| `one_proportion` | Tests for One Proportion | 🟢 | 3 |
| `two_proportions` | Tests for Two Proportions | 🟢 | 5 |
| `pearson_correlation` | Pearson's Correlation Tests | 🟢 | 3 |
| `one_way_anova_f` | One-Way Analysis of Variance F-Tests | 🟢 | 3 |
| `logrank_freedman` | Logrank Tests (Freedman) | 🟢 | 1 |

Pearson correlation uses the exact Guenther/Hotelling density via
₂F₁ + numerical integration; matches reference to ≥4 sig.fig. A
`method="fisher-z"` backend is retained for the textbook approximation.

**Total: 28 worked-example fixtures, all passing within tolerance.**

## Tier 2 (in progress) — 11 / ~25 methods done

| id | name | status | fixtures |
|---|---|---|---|
| `non_inferiority_two_means` | NI Tests for Two Means using Differences | 🟢 | 2 |
| `non_inferiority_two_proportions` | NI Tests for the Difference Between Two Proportions | 🟢 | 4 |
| `equivalence_two_means` | Equivalence Tests for Two Means using Differences | 🟢 | 3 |
| `equivalence_two_proportions` | Equivalence Tests for the Difference Between Two Proportions | 🟢 | 3 |
| `superiority_by_margin_two_means` | Superiority by a Margin Tests for Two Means using Differences | 🟢 | 4 |
| `superiority_by_margin_two_proportions` | Superiority by a Margin for the Difference Between Two Proportions | 🟢 | 4 |
| `cox_regression` | Cox Regression (Hsieh-Lavori) | 🟢 | 5 |
| `mcnemar` | Tests for Two Correlated Proportions (McNemar) | 🟢 | 4 |
| `chi_square` | Chi-Square Tests (general r×c) | 🟢 | 5 |
| `non_inferiority_one_mean` | Non-Inferiority Tests for One Mean | 🟢 | 5 |
| `equivalence_one_mean` | Equivalence Tests for One Mean | 🟢¹ | 4 |

¹ Equivalence one-mean uses σ-known + central-t approximation; mildly
   optimistic at very small N. Exact bivariate-noncentral-t backend is
   a v0.3 target.

**Tier 1 + Tier 2 total: 70 worked-example fixtures, 73/73 tests passing.**

## Tier 3 — 7 / +

| id | name | status | fixtures |
|---|---|---|---|
| `non_inferiority_one_proportion` | NI Tests for One Proportion | 🟢 | 4 |
| `equivalence_one_proportion` | Equivalence Tests for One Proportion | 🟢 | 4 |
| `superiority_by_margin_one_proportion` | Superiority by a Margin for One Proportion | 🟢 | 4 |
| `cluster_randomized_two_means` | Tests for Two Means in a Cluster-Randomized Design | 🟢 | 7 |
| `cluster_randomized_two_proportions` | Tests for Two Proportions in a Cluster-Randomized Design | 🟢 | 6 |
| `cross_over_two_means` | Tests for Two Means in a 2x2 Cross-Over Design using Differences | 🟢 | 6 |
| `group_sequential_two_means` | Group-Sequential Tests for Two Means (OBF + Pocock) | 🟢 | 2 |

**Cumulative: 26 methods, 103 fixtures, 167/167 validation tests passing.**

## Long tail

The full catalogue of methods is in `samplesize/registry/methods.json`.
Each method will appear here in priority order as it is picked up.
