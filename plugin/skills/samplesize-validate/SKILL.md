---
name: samplesize-validate
description: Use to verify that an implemented method reproduces the worked-example reference answers within 3 significant figures. Run on demand or before declaring a Tier-1 method "production".
---

# Validating against reference examples

Goal: prove numerical agreement with the published worked examples in
the reference fixtures.

## Process

1. **Pick the method.** Read the YAML fixture under
   `tests/validation/fixtures/<method_id>.yaml`. Each fixture lists 1–N
   examples with inputs, expected outputs, and tolerances.
2. **If no fixture exists**, extract one from the reference documentation:
   - Open `reference/md/<chapter>/hybrid_auto/<chapter>.md`
   - Find "Example 1", "Example 2", ... sections — these contain inputs
     and the published answer
   - Translate to YAML with explicit `inputs`, `expected`, `tolerance`
3. **Run pytest** on the validation suite:
   ```sh
   pytest tests/validation/ -k <method_id> -v
   ```
4. **Report the comparison table**: reference value vs ours vs absolute and
   relative error per example.
5. **If any example fails tolerance**, flag the discrepancy
   explicitly — do not adjust tolerance to make tests pass.

## Tolerance defaults

- Sample size N: exact integer match (ceiling rounding)
- Power: ±0.001 absolute
- Critical values, effect sizes: 3 significant figures relative
- Simulation-based methods: ±0.01 absolute on power (Monte Carlo noise)
