---
name: samplesize-validate
description: Use to verify that an implemented method reproduces the worked-example reference answers within 3 significant figures. Run on demand or before declaring a Tier-1 method "production".
---

# Validating against reference examples

Goal: prove numerical agreement with the published worked examples in
the reference fixtures.

## Safety preamble (shell-injection defense)

The pytest `-k <method_id>` selector is a shell argument: a stray space,
quote, or option character lets a hostile `method_id` inject arbitrary
pytest flags. Before invoking pytest:

- **Validate `<method_id>` against `^[a-z][a-z0-9_]*$`.** No spaces, no
  dashes, no shell metacharacters.
- **Registry-validate it.** Confirm `<method_id>` appears in
  `python -m samplesize list` output. If either check fails, STOP and
  ask the user to clarify — do **not** invoke pytest.
- **Never pass the raw user description as a shell argument.**

## Process

1. **Resolve `method_id` (registry-validated).** Run
   `python -m samplesize list`, pick the matching `id`, and confirm it
   satisfies `^[a-z][a-z0-9_]*$`.
2. **Pick the method.** Read the YAML fixture under
   `tests/validation/fixtures/<method_id>.yaml`. Each fixture lists 1–N
   examples with inputs, expected outputs, and tolerances.
3. **If no fixture exists**, extract one from the reference documentation:
   - Open `reference/md/<chapter>/hybrid_auto/<chapter>.md`
   - Find "Example 1", "Example 2", ... sections — these contain inputs
     and the published answer
   - Translate to YAML with explicit `inputs`, `expected`, `tolerance`
4. **Run pytest** on the validation suite (only after the safety checks
   above pass):
   ```sh
   pytest tests/validation/ -k <method_id> -v
   ```
5. **Report the comparison table**: reference value vs ours vs absolute and
   relative error per example.
6. **If any example fails tolerance**, flag the discrepancy
   explicitly — do not adjust tolerance to make tests pass.

## Tolerance defaults

- Sample size N: exact integer match (ceiling rounding)
- Power: ±0.001 absolute
- Critical values, effect sizes: 3 significant figures relative
- Simulation-based methods: ±0.01 absolute on power (Monte Carlo noise)
