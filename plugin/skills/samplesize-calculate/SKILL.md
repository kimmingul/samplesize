---
name: samplesize-calculate
description: Use after a sample-size method has been identified (by samplesize-design or user-named) and you need to collect parameters, run the calculation, and present results. Always invokes the validated Python package, never computes formulas inline.
---

# Running a sample-size calculation

Goal: produce a verified N (or power, or detectable effect) with full
reproducibility metadata.

## Safety preamble (shell-injection defense)

When you build the shell command for `python -m samplesize`, treat
`<method_id>` and parameter values as untrusted text:

- **Validate `<method_id>` BEFORE substituting it into any shell string.**
  It MUST match the regex `^[a-z][a-z0-9_]*$` AND appear as an `id`
  in the output of `python -m samplesize list`. If either check fails,
  STOP and ask the user to clarify — do **not** invoke the CLI.
- **Never pass the raw user description as a shell argument.** Translate
  it into a validated `method_id` first.
- **Prefer `--json-args-file <tmpfile>` for kwargs.** Write the JSON to a
  temp file via your file-write tool (e.g. `/tmp/<method_id>.json`),
  then reference that path. This avoids shell-quoting the user's
  parameter values entirely. Use inline `--json-args '<json>'` **only**
  when every value is a known-safe scalar you constructed yourself.

## Process

1. **Resolve `method_id` (registry-validated).** Run
   `python -m samplesize list`, pick the matching `id`, and confirm it
   satisfies `^[a-z][a-z0-9_]*$`.
2. **Look up the method's exact API**:
   ```sh
   python -m samplesize show <method_id>
   ```
   This prints the registry entry **plus** the actual callable's
   keyword-only parameters (`signature.kwargs`) — that signature block
   is the source of truth for parameter names and defaults. Do **not**
   read `methods.json` directly; the registry's `params.optional`
   listing is informative only.
3. **Collect parameters.** Use `AskUserQuestion` for structured fields
   (α, power, sides, allocation ratio, dropout). For continuous
   parameters (means, SDs, effect sizes), accept free text but echo back
   the parsed numeric value before computing.
4. **Write kwargs JSON to a temp file**, e.g. `/tmp/<method_id>.json`,
   using your file-write tool — do **not** interpolate user values into
   the shell command.
5. **Run via the CLI**, not Python imports inside Claude:
   ```sh
   python -m samplesize calc <method_id> --json-args-file /tmp/<method_id>.json
   ```
   This guarantees the same code path users get programmatically and
   keeps untrusted text off the shell command line.
6. **Present the result** with three blocks:
   - **Headline**: required N, achieved power, key inputs
   - **Sanity check**: flag implausible inputs (effect size < 0.05, N >
     5000, etc.) and suggest the user verify
   - **Reproducibility**: paste the generated Python snippet, audit JSON
     path, and method citation
7. **Offer next steps**: power curve (`samplesize-report`), sensitivity
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
