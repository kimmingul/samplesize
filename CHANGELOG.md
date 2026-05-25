# Changelog

All notable changes to `samplesize` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-05-25

First public release.

### Added
- **234 sample-size / power methods**, every one implemented **and** validated
  against published worked-example fixtures (819 fixtures across 234 files).
  Families include means (one/two/paired, NI/equivalence/superiority-by-margin),
  proportions (one/two/correlated, ratio/odds variants), correlation, ANOVA/GLM,
  survival (logrank, Cox, exponential hazard rates), group-sequential,
  cluster-randomized, cross-over (2×2 and higher-order), Poisson/negative-binomial
  rates, ROC/diagnostic accuracy, reference intervals, nonparametric simulations,
  and specialty designs.
- Method registry (`samplesize/registry/methods.json`) as the canonical catalogue,
  with a `samplesize doctor` integrity check (registry callables, fixtures,
  plugin manifest, reporting templates).
- Worked-example validation harness (`tests/validation/`) with per-method
  reference fixtures and documented tolerances.
- Claude Code plugin (`plugin/`) with design / calculate / report / validate
  skills and supporting agents.
- MkDocs documentation site, auto-generated method-coverage matrix
  (`scripts/gen_method_coverage.py`), and PyPI Trusted-Publisher release workflow.

### Notes
- Equivalence/TOST means tests use a central-t formulation that matches the exact
  bivariate-noncentral-t method at moderate-to-large N and is conservative at very
  small N (documented per fixture).
- `kappa_two_raters_binary` uses the Fleiss-Cohen-Everitt variance maximised over
  tables (Flack et al. 1988 conservative approach).

[0.1.0]: https://github.com/kimmingul/samplesize-copilot/releases/tag/v0.1.0
