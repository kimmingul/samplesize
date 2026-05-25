# Project: samplesize-copilot

Goal: a Python sample-size and power-calculation package (`samplesize`) and a
Claude Code plugin (`samplesize-copilot`, in `plugin/`).

## Working agreements

- **Never compute statistics in chat.** Always invoke the Python package
  (`python -m samplesize ...` or import a module). Hand-derived numbers
  are not trustworthy.
- **Reference examples are ground truth.** When implementing a method, source
  worked examples from a published statistical reference and put them under
  `tests/validation/fixtures/`. Only declare a method `validated: true` once
  every fixture passes within tolerance.
- **Method registry is the catalogue.** Touch
  `samplesize/registry/methods.json` when adding a method; never invent
  method names elsewhere.
- **No fallback heuristics.** If a method is not yet implemented, the
  CLI returns an error. Do not silently approximate with a different
  method.
- **Audit every calculation.** Each calculation writes a JSON record
  via `samplesize.reporting.audit.write_audit` containing inputs,
  outputs, library versions, and the reference chapter cited.

## Where things live

- `samplesize/tests/<family>.py` — calculators (one file per family:
  `one_mean.py`, `two_means.py`, `proportions.py`, ...).
- `samplesize/registry/methods.json` — canonical method catalogue.
- `reference/` — local-only, gitignored; users must supply their own
  validation source material (see "Reference content" below).
- `tests/validation/fixtures/<method_id>.yaml` — worked-example regression
  fixtures.

## Conventions

- Sides: always explicit (`sides=1` or `sides=2`), never default
  silently to 2.
- Rounding: `n` is `math.ceil` to integer; report achieved power at the
  rounded N as well.
- Effect sizes: do not assume Cohen "small/medium/large" defaults.
  Always require explicit numeric input.
- Dropout: apply `inflate_for_dropout` last, after the base N is
  computed.

## Reference content

Reference content under `reference/` is local-only; users must supply their own
validation source material. The repository does not bundle any third-party
copyrighted content.

## Plugin shape

- `plugin/skills/samplesize-design` — pick a method from a study
  description.
- `plugin/skills/samplesize-calculate` — collect params + run.
- `plugin/skills/samplesize-report` — generate curves, tables, text.
- `plugin/skills/samplesize-validate` — verify against worked examples.

The skills delegate to subagents in `plugin/agents/` for execution.
