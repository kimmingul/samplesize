# samplesize-copilot — Sample-size and power calculations for clinical and applied research

[![CI](https://github.com/kimmingul/samplesize-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/kimmingul/samplesize-copilot/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Methods](https://img.shields.io/badge/methods-234-brightgreen)](docs/METHOD_COVERAGE.md)
[![Tests](https://img.shields.io/badge/tests-819%20passing-brightgreen)](tests/validation/test_fixture_consistency.py)

A Python package + Claude Code plugin implementing 234 sample-size and power-calculation
methods validated against worked examples from established statistical references.

## Status

**v0.1 — 234 methods implemented and validated, 819 worked-example fixture tests passing.**
Doctor passes 9/9 integrity checks across registry, callables, plugin manifest, and
reporting templates. Roadmap in `docs/ROADMAP.md`; live coverage matrix in
`docs/METHOD_COVERAGE.md`.

## Layout

```
samplesize-copilot/
├── samplesize/             # Python package — pure-Python calculators
│   ├── core/               # distributions, effect sizes, adjustments
│   ├── tests/              # per-method calculator modules
│   ├── reporting/          # plots, tables, protocol text, audit, R/SAS export
│   │   └── templates/      # i18n templates (protocol.en.yaml, protocol.ko.yaml, ...)
│   ├── registry/           # methods.json — categorical metadata only
│   ├── cli.py              # `python -m samplesize ...`
│   └── doctor.py           # `samplesize doctor` integrity checks
├── plugin/                 # Claude Code plugin
│   ├── .claude-plugin/plugin.json
│   ├── skills/             # design / calculate / report / validate
│   ├── commands/           # /ss-design, /ss-calc, /ss-power, /ss-curve, /ss-report
│   └── agents/             # methodologist, calculator, validator
├── reference/              # Local-only knowledge base (gitignored, user-supplied)
│   └── ...                 # Validation reference material — not bundled in repo
├── tests/                  # pytest suites
│   ├── validation/         # worked-example regression tests
│   └── unit/               # registry / doctor / signature parity
└── docs/                   # ARCHITECTURE, ROADMAP, METHOD_COVERAGE, COOKBOOK, TROUBLESHOOTING
```

## Installation

```sh
pip install -e ".[dev]"
```

## Quick start

```sh
samplesize list                                          # available methods
samplesize show two_sample_t_equal_var                   # full metadata + kwargs
samplesize calc two_sample_t_equal_var \
  --json-args '{"mean1":10,"mean2":0,"sd":20,"alpha":0.05,"power":0.80,"sides":2}'
# → n1=64, n2=64, achieved_power=0.8015; audit JSON saved

# follow-ups on the audit just printed
AUDIT=$(ls -t .samplesize/audit/*.json | head -1)
samplesize report "$AUDIT" --kind power-curve --out curve.png
samplesize report "$AUDIT" --kind protocol --lang en
samplesize report "$AUDIT" --kind sensitivity --vary "sd=15,20,25,30"
samplesize report "$AUDIT" --kind r-code        # pwr::pwr.t.test(...) equivalent
samplesize report "$AUDIT" --kind sas-code      # PROC POWER equivalent

# sanity gate
samplesize doctor
```

**More recipes.** `docs/COOKBOOK.md` has 15 worked study scenarios
(RCT, NI, equivalence, survival, Cox, McNemar, χ², ANOVA, correlation).
Hit an error? `docs/TROUBLESHOOTING.md`.

## Using inside Claude Code (plugin)

Two ways to make the slash commands and skills available:

**Ephemeral — load for one session**:
```sh
claude --plugin-dir /path/to/samplesize-copilot/plugin
```

**Persistent — register the marketplace and install**:
```sh
claude plugin marketplace add kimmingul/samplesize-copilot   # from GitHub
# …or from a local clone (repo root): claude plugin marketplace add /path/to/samplesize-copilot
claude plugin install samplesize-copilot@samplesize-copilot  # requires CC ≥ 2.2
```

Once loaded, these commands work inside Claude Code:

- `/samplesize-copilot:ss-design <study description>` — pick the right test
- `/samplesize-copilot:ss-calc <method> ...` — run a calculation
- `/samplesize-copilot:ss-power ...` — solve for power at fixed N
- `/samplesize-copilot:ss-curve` — emit a power-curve PNG for the latest result
- `/samplesize-copilot:ss-report` — generate ICH E9 protocol / grant text
- `/samplesize-copilot:ss-validate <method?>` — run worked-example validation tests

## Coverage

234 methods across:

- Means (one-sample, two-sample, paired, non-inferiority, equivalence, superiority-by-margin)
- Proportions (one, two, McNemar, NI/equivalence variants)
- Correlation (Pearson exact and Fisher-z)
- ANOVA / GLM (one-way F, chi-square)
- Survival (logrank Freedman, Cox regression Hsieh-Lavori)
- Group-sequential (O'Brien-Fleming, Pocock alpha-spending)
- Cluster-randomized (two means, two proportions, Donner-Klar)
- Cross-over (2×2 design)
- Phase II (Simon two-stage)
- ROC / diagnostic
- And more — see `docs/METHOD_COVERAGE.md`

## Validation

819 fixture tests passing. Methods are validated against worked examples from
established statistical software references. Reference content itself is
user-supplied (see `reference/` — not bundled in this repository).

Fixtures live under `tests/validation/fixtures/<method_id>.yaml`.

```sh
pytest tests/validation/
```

## License

Apache License 2.0 — see `LICENSE`.

## Acknowledgments

Method implementations draw on the primary statistical literature, including:

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.)
- Donner, A. & Klar, N. (1996). Statistical considerations in the design and analysis of community intervention trials.
- Hsieh, F. Y. & Lavori, P. W. (2000). Sample-size calculations for the Cox proportional hazards regression model with nonbinary covariates.
- Schoenfeld, D. (1981). The asymptotic properties of nonparametric tests for comparing survival distributions.
- Bonett, D. G. & Wright, T. A. (2000). Sample size requirements for estimating Pearson, Kendall and Spearman correlations.
- Hanley, J. A. & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve.
- Simon, R. (1989). Optimal two-stage designs for phase II clinical trials.
- Wang, S. K. & Tsiatis, A. A. (1987). Approximately optimal one-parameter boundaries for group sequential trials.
- Flack, V. F. et al. (1988). Sample size determinations for the two rater kappa statistic.
