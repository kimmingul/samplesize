# Changelog

All notable changes to `samplesize` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- **Reporting layer correctness**:
  - `samplesize report --kind power-curve` no longer produces silent NaN curves
    for two-sample methods; the sweep key is now derived from the registry
    signature (`n1` when present, else `n`) instead of a non-existent `params`
    field (`samplesize/reporting/plots.py`).
  - `samplesize report --kind r-code` / `--kind sas-code` no longer crash with
    `TypeError` on confidence-interval methods whose `achieved_power` is `None`;
    formatters now emit `"N/A"` cleanly (`samplesize/reporting/code_export.py`).
  - `samplesize report --kind protocol` / `--kind grant` no longer crash and no
    longer emit literal `"None"` or the placeholder `"α"` for CI methods;
    `power`, `alpha`, and `target_power` are pre-formatted with a localized
    "n/a" / "해당 없음" fallback (`samplesize/reporting/protocol.py`,
    `samplesize/reporting/templates/protocol.{en,ko}.yaml`).
- **Sensitivity report**: `samplesize report --kind sensitivity` now drops
  `n`/`n1`/`n2`/`power` from the base inputs before sweeping, so a `solve_for=power`
  audit no longer collides with the sweep semantics (`samplesize/cli.py`).
- **Grant text**: `samplesize report --kind grant --lang ko` now actually honors
  `--lang` (was always English) (`samplesize/cli.py`).
- **Method-routing integrity (per CLAUDE.md "no fallback heuristics")**: the
  decision tree no longer silently downgrades to an approximate method when the
  catalogue lacks the requested one. K-group survival, k-group Poisson, and
  one-sample ordinal terminate at explicit `unimplemented:` markers with a stated
  reason rather than routing to a different method
  (`samplesize/registry/decision_tree.yaml`). The `count` and `ordinal` branches
  also gained the missing group-count question.
- **Performance**: `reference_intervals_clinical_lab` now binary-searches the
  sample size (was linear); the worst observed case (`target_k≈1.97`) drops
  from ~5.5 s to ~0.004 s (`samplesize/tests/reference_intervals.py`).
- **Audit file uniqueness**: same-method calls in the same second no longer
  overwrite each other; filenames now include a microsecond suffix
  (`samplesize/reporting/audit.py`).
- **Drift detector**: `scripts/gen_method_coverage.py --check` now uses a
  word-boundary regex instead of substring matching, so headline counts can't
  be silently shadowed by a different number containing the same digits.
- **inputs_echo shape**: `randomized_block_anova` and `reference_intervals_clinical_lab`
  now include `solve_for` in their `inputs_echo` for parity with other solvers.
- **Dependency floor**: `scipy>=1.17` (was `>=1.12`) — the Anderson-Darling
  simulation path uses `stats.anderson(..., method="interpolate")`, available
  from 1.17 onward (`pyproject.toml`).

### Added
- **Defense-in-depth hardening** (no current exploit; reduces blast radius if a
  future entry point opens):
  - `_resolve_callable` now allowlists module prefixes (`samplesize.tests.`)
    (`samplesize/registry/__init__.py`).
  - `samplesize calc` rejects kwargs not in the resolved method's signature
    and gives a clean error instead of raising a Python `TypeError`
    (`samplesize/cli.py`).
  - `samplesize calc --json-args-file <path>` flag (mutually exclusive with
    `--json-args`), so plugin SKILLs can pass kwargs without shell-quoting.
  - `samplesize calc --json-args '<bad>'` now reports a clean stderr message
    on `JSONDecodeError` and exits 2.
  - Audit filename `method_id` is sanitized (`[^A-Za-z0-9_.-]` → `_`).
  - `samplesize-calculate` and `samplesize-validate` plugin SKILLs gained
    explicit "shell-injection defense" preambles instructing the LLM to
    registry-validate `method_id` against `^[a-z][a-z0-9_]*$` and to prefer
    `--json-args-file` over inline shell quoting.
- **New `samplesize doctor` checks** (now 11/11):
  - `registry.decision-tree-leaves-exist` — every `leaf:` in
    `decision_tree.yaml` is a registered method id (or a deliberate
    `unimplemented:` marker).
  - `plugin.skill-kind-choices-exist` — every `--kind <name>` referenced in
    `plugin/*.md` is in the argparse `choices` for `cmd_report`.
- **CI release gate**: `.github/workflows/release.yml` now runs a `verify` job
  (doctor + coverage `--check` + full pytest + ruff) before `build`/`publish`,
  so a `v*` tag on an unchecked commit cannot ship.
- **SKILL documentation**: `samplesize-report` now enumerates all six `--kind`
  choices (was missing `sensitivity`, `r-code`, `sas-code`).

### Notes
- `plugin.skill-cli-flags-exist` doctor check now also scans `docs/**/*.md`
  and `README.md` for the same drift class.

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
