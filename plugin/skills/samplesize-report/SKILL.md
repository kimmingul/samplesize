---
name: samplesize-report
description: Use to generate downstream artefacts after a calculation - power curves, sensitivity tables, ICH E9 protocol text, R/SAS equivalent code, and grant-ready writeups.
---

# Reporting outputs from a sample-size calculation

Goal: turn a single N into research-grade deliverables.

## Process

1. **Locate the calculation's audit JSON** (path printed by
   `samplesize-calculate`).
2. **Ask what the user wants**, multi-select via `AskUserQuestion`:
   - Power curve (matplotlib PNG)
   - Sensitivity table (vary effect size × dropout)
   - Protocol section (ICH E9 R1 style, English / Korean)
   - Grant-aims paragraph (NIH/NRF style)
   - R/SAS equivalent code (for verification by collaborators)
   - Markdown summary table
3. **Run the appropriate CLI command** (`--kind` selects the artefact):
   ```sh
   python -m samplesize report <audit_json> --kind power-curve --out plot.png
   python -m samplesize report <audit_json> --kind protocol --lang en
   python -m samplesize report <audit_json> --kind grant
   ```
   Available `--kind` values: `power-curve`, `protocol`, `grant`.
   Available `--lang` values are discovered from
   `samplesize/reporting/templates/protocol.<lang>.yaml` (default `en`).
4. **Show inline** for short outputs, **link** to file path for figures
   and longer text.

## Templates

Translatable protocol / grant-aim phrasing lives as data, not code:

  `samplesize/reporting/templates/protocol.<lang>.yaml`

Each file declares `templates.ich_e9_section`, `templates.grant_aims`,
plus `words` and `n_phrase` lookups used at format time. Adding a new
language is one file (no code changes); the CLI's `--lang` choices are
generated from this directory.

## Quality bar

A finished writeup paragraph must include: method name, effect-size
metric and value, α, target power, achieved power, total N (with per-arm
breakdown), dropout assumption, and a citation to the method plus the
underlying methodological reference (Cohen 1988, Lachin &
Foulkes 1986, etc.).
