---
name: samplesize-design
description: Use when a user wants help picking the right sample-size method — they describe a study (groups, outcome type, hypothesis) but haven't named a specific test. Walks them through a structured decision tree, then hands off to samplesize-calculate.
---

# Choosing the right sample-size method

Goal: given a free-form study description, identify the single method
that matches and confirm with the user before any calculation.

## Process

1. **Read the registry first** — never guess method names from memory.
   The authoritative way to discover candidates is the CLI:
   ```sh
   python -m samplesize list           # all method ids and status flags
   python -m samplesize show <id>      # full metadata + actual signature
   ```
   The `decision_tree.yaml` at
   `samplesize/registry/decision_tree.yaml` is a useful aid but the
   `show` output is the source of truth for what parameters a method
   actually accepts.
2. **Ask only the questions the decision tree requires.** Use
   `AskUserQuestion` with structured options. Typical branching axes:
   - outcome type: continuous / binary / time-to-event / count /
     ordinal / categorical
   - number of groups / arms
   - independence: independent / paired / clustered / cross-over
   - hypothesis flavour: superiority / equivalence / non-inferiority /
     superiority-by-margin
   - design: parallel / group-sequential / multi-stage / adaptive
3. **Surface the match.** Quote the chapter name, link to its markdown
   reference (`reference/md/<chapter>/hybrid_auto/<chapter>.md`), and
   summarise the test's assumptions in one paragraph drawn from the
   "Assumptions" section.
4. **Confirm with the user**, then hand off to `samplesize-calculate`
   with the method id resolved.

## Anti-patterns

- Do not propose a method that is marked `implemented: false` in the
  registry without warning the user that no validated calculator exists
  yet for it.
- Do not skip ahead to inputting numbers. Method selection precedes
  parameter entry.
- Do not synthesise hybrid methods ("a one-sample test with a
  non-inferiority margin" should map to an actual registered method, not
  an invented composition).
