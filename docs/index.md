# samplesize

**Sample-size and power calculations for clinical and applied research.**

A Python package + Claude Code plugin implementing 234 sample-size and
power-calculation methods, validated against worked examples from
established statistical references.

[![CI](https://github.com/kimmingul/samplesize/actions/workflows/ci.yml/badge.svg)](https://github.com/kimmingul/samplesize/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/kimmingul/samplesize/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/kimmingul/samplesize/blob/main/pyproject.toml)
[![Methods](https://img.shields.io/badge/methods-234-brightgreen)](METHOD_COVERAGE.md)
[![Tests](https://img.shields.io/badge/tests-819%20passing-brightgreen)](https://github.com/kimmingul/samplesize/blob/main/tests/validation/test_fixture_consistency.py)

## What's here

- **234 methods** spanning means, proportions, correlation, survival,
  ANOVA/GLM, ROC/diagnostic accuracy, group-sequential, cluster-randomized,
  cross-over, Phase II clinical trials, and specialty designs.
- **819 worked-example fixture tests** — every method has at least two
  pinned reference examples with documented tolerances.
- **Doctor integrity gate** — 9 cross-checks on the method registry,
  callable imports, fixture references, plugin manifest, and reporting
  templates.
- **Audit log** — every calculation writes a JSON record with inputs,
  outputs, library versions, and method citation.

## Installation

```bash
pip install -e .
```

(PyPI release pending.)

## Quick start

```python
from samplesize.tests.one_mean import one_sample_t

result = one_sample_t(
    mu0=0.0,
    mu1=0.5,
    sigma=1.0,
    alpha=0.05,
    power=0.80,
    sides=2,
    solve_for="n",
)
print(result["n"], result["achieved_power"])
```

CLI:

```bash
samplesize list
samplesize calc one_sample_t mu0=0 mu1=0.5 sigma=1 alpha=0.05 power=0.80 sides=2
samplesize doctor
```

## Where to go next

- **[Architecture](ARCHITECTURE.md)** — how the package, registry, and
  plugin fit together.
- **[Cookbook](COOKBOOK.md)** — realistic study sketches, end-to-end.
- **[Method Coverage](METHOD_COVERAGE.md)** — which methods are
  implemented and which are validated.
- **[Roadmap](ROADMAP.md)** — what's next.
- **[Troubleshooting](TROUBLESHOOTING.md)** — common errors and fixes.

## License

MIT — see [LICENSE](https://github.com/kimmingul/samplesize/blob/main/LICENSE).

## Acknowledgments

Calculator implementations follow the worked formulas from standard
power-analysis references — Cohen (1988), Schoenfeld (1981), Hsieh &
Lavori (2000), Donner & Klar (1996), Wang & Tsiatis (1987), Hanley &
McNeil (1982), Bonett & Wright (2000), Simon (1989), Flack et al. (1988),
and many others cited per-method in the calculator source.
