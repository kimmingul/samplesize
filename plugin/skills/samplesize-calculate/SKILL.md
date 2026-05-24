---
name: samplesize-calculate
description: Use after a sample-size method has been identified (by samplesize-design or user-named) and you need to collect parameters, run the calculation, and present results. Always invokes the validated Python package, never computes formulas inline.
---

# Running a sample-size calculation

Goal: produce a verified N (or power, or detectable effect) with full
reproducibility metadata.

## Process

1. **Look up the method's exact API**:
   ```sh
   python -m samplesize show <method_id>
   ```
   This prints the registry entry **plus** the actual callable's
   keyword-only parameters (`signature.kwargs`) — that signature block
   is the source of truth for parameter names and defaults. Do **not**
   read `methods.json` directly; the registry's `params.optional`
   listing is informative only.
2. **Collect parameters.** Use `AskUserQuestion` for structured fields
   (α, power, sides, allocation ratio, dropout). For continuous
   parameters (means, SDs, effect sizes), accept free text but echo back
   the parsed numeric value before computing.
3. **Run via the CLI**, not Python imports inside Claude:
   ```sh
   python -m samplesize calc <method_id> --json-args '<args>'
   ```
   This guarantees the same code path users get programmatically.
4. **Present the result** with three blocks:
   - **Headline**: required N, achieved power, key inputs
   - **Sanity check**: flag implausible inputs (effect size < 0.05, N >
     5000, etc.) and suggest the user verify
   - **Reproducibility**: paste the generated Python snippet, audit JSON
     path, and method citation
5. **Offer next steps**: power curve (`samplesize-report`), sensitivity
   analysis, protocol section text.

## Inputs Claude must never invent

- Effect sizes — always ask, never assume Cohen's "small/medium/large"
  defaults
- Variance / SD — ask, do not guess from "typical" values
- Allocation ratio — default to 1:1 only when user explicitly accepts
- Sides — explicit `1` or `2`, no implicit two-sided

## Rounding convention

- Always round N up to the next integer (ceiling)
- Report achieved power for the rounded N, not the fractional N
- Cluster designs: round per-cluster N up, then number of clusters up
