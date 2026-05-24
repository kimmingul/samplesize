# Roadmap

## v0.1 — Tier-1 MVP (target: 2 weeks)

8 most-used methods, fully validated against reference examples:

- [x] one_sample_t  (6 fixtures passing)
- [x] two_sample_t_equal_var  (3 fixtures passing)
- [x] paired_t  (4 fixtures passing)
- [x] one_proportion  (3 fixtures passing)
- [x] two_proportions  (5 fixtures passing)
- [x] pearson_correlation  (3 fixtures, exact Guenther/Hotelling backend matches reference to ≥4 sig.fig.)
- [x] one_way_anova_f  (3 fixtures passing)
- [x] logrank_freedman  (1 fixture passing)

**Tier 1 complete: 28 fixtures, 31/31 tests passing.**

Required infra (all done):

- [x] `samplesize.core.distributions` (nct, ncf, norm, t)
- [x] `samplesize.core.effect_sizes` (Cohen's d/h/f, HR, OR)
- [x] `samplesize.core.adjustments` (dropout, Bonferroni, FPC)
- [x] `samplesize.reporting.audit` (audit JSON writer)
- [x] `samplesize.reporting.plots` (power curve PNG)
- [x] `samplesize.reporting.tables` (sensitivity grid)
- [x] `samplesize.reporting.protocol` (ICH E9 + grant, i18n YAML)
- [x] `samplesize.reporting.code_export` (R / SAS)
- [x] `samplesize.cli` (`list`, `show`, `calc`, `report`, `doctor`)
- [x] `samplesize.doctor` (9 integrity checks; CI gate)
- [x] Plugin commands `/ss-design`, `/ss-calc`, `/ss-power`, `/ss-curve`, `/ss-report`, `/ss-validate`

## v0.2 — Tier-2 (target: +4 weeks)

Add the families clinicians ask for most:

- [x] Non-inferiority for two means using differences  (2 fixtures)
- [x] Non-inferiority for two proportions (difference)  (4 fixtures)
- [x] Equivalence for two means using differences  (3 fixtures)
- [x] Equivalence for two proportions (difference)  (3 fixtures)
- [x] Non-inferiority + equivalence for one mean (Tier-2)
- [x] Non-inferiority + equivalence + superiority-by-margin for one proportion
- [x] Superiority by a margin (one/two means, one/two proportions)
- [x] McNemar / paired proportions (binomial enumeration)
- [x] Chi-square (general r×c with Cohen w)
- [x] Cox regression (Hsieh-Lavori)
- [ ] Mann-Whitney (simulation)
- [ ] CI for one mean / one proportion

## v0.3 — Tier-3 specialty designs (in progress)

- [x] Cluster-randomized two means + two proportions (Donner-Klar)
- [x] 2x2 cross-over for two means
- [x] Group-sequential two means (OBF + Pocock alpha-spending)
- [ ] Higher-order cross-over (3+ periods)
- [ ] Logrank Lachin/Foulkes, Lakatos (Markov)
- [ ] Repeated measures
- [ ] Bioequivalence (R `PowerTOST` subprocess)

## v0.4 — Reporting & integrations

- [x] R/SAS code export for verification  (10 methods covered for R; 7 for SAS)
- [x] Protocol section templates (ICH E9 R1, English + Korean i18n)
- [x] Multi-scenario comparison grid (sensitivity table CLI)
- [ ] Pilot-data CSV → effect size estimator
- [ ] Session save/load (`/ss save`, `/ss load`)
- [ ] Additional protocol languages (ja, zh, es)

## v1.0 — Coverage push

Aim for 50+ methods implemented and validated. Identify the
research-frequency long-tail that needs filling.

## Stretch

- [ ] Bayesian sample size (probability of success)
- [ ] Simulation-based fallbacks for every analytic method (sanity)
- [ ] Web UI via FastAPI (only if researcher demand emerges)
