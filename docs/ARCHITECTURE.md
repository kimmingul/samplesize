# Architecture

## Two layers

```
┌─────────────────────────────────────────────┐
│  plugin/  — Claude Code interface           │
│    skills, commands, agents, templates      │
└─────────────────────────────────────────────┘
                    │ invokes via `python -m samplesize ...`
                    ▼
┌─────────────────────────────────────────────┐
│  samplesize/  — Python package              │
│    core (distributions, effect, adjust)     │
│    tests (calculators)                      │
│    reporting (plots, audit, citations)      │
│    registry (catalogue, decision tree)      │
└─────────────────────────────────────────────┘
```

The plugin layer never computes anything itself. Every calculation
crosses the CLI boundary so the same numeric path is used whether the
caller is Claude, a Jupyter notebook, or a CI test.

## Method registry

`samplesize/registry/methods.json` is the single source of truth for
"what methods exist and which are implemented". The plugin's
`methodologist` agent reads it to pick a method, and the `calculator`
agent reads it to resolve a method id → Python callable.

A method entry has:

- `id` — stable snake_case identifier.
- `name`, `category` — for display, taken from the reference chapter.
- `manual_path` — relative path to the chapter under `reference/`.
- `callable` — `module:function` resolved lazily.
- `implemented`, `validated` — independent flags; a method can be
  implemented but unvalidated.
- `study_types`, `outcome`, `design`, `hypothesis` — facets used by the
  decision tree.
- `params.required`, `params.optional`, `params.solve_for_options` —
  parameter contract.

## Calculation function contract

Every Tier-1 calculator returns a dict with at least:

```python
{
  "method_id": "one_sample_t",
  "solve_for": "n" | "power" | "effect",
  "n": int | None,
  "achieved_power": float | None,
  "effect": float | None,
  "inputs_echo": {...},
  "citations": [...],
}
```

The CLI wraps this with audit metadata and a re-runnable Python
snippet.

## Validation pipeline

```
Reference chapter (markdown) ──► extract Example sections ──► YAML fixture
                                                                    │
                                                                    ▼
                                                           pytest parametrise
                                                                    │
                                                                    ▼
                                                         compare to reference values
                                                           within tolerance
```

Fixtures are the durable record of "reference said N=47 with these
inputs". They never change to make tests pass.
