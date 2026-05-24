# Troubleshooting

Errors you may see, what they mean, and what to do.

The first line of triage for anything strange is always:

```sh
python -m samplesize doctor
```

If `doctor` is green and you still see a runtime error, scan the list
below.

---

## Argument and input errors

### `ValueError: supply exactly two of (mean1, power, n)`

Source: every parametric calculator (one-sample t, two-sample t, etc.).
You supplied 0 or 1 of these solver inputs, or supplied all three.

The CLI infers what to solve for from which slot is `None`:

| `mean1` | `power` | `n` | solve_for |
|---|---|---|---|
| given | given | `None` | `n` |
| given | `None` | given | `power` |
| `None` | given | given | `effect` |
| missing two | — | — | error |

Pass exactly two; leave the third out (or set to `null`).

### `ValueError: mean0 and mean1 must differ to solve for N`

Identical means → effect size 0 → no power for any N. Re-check your
hypothesised effect, or use `solve_for: "power"` if you really want
the power at a fixed N (always ≈ α).

### `ValueError: margin must be > 0`

`margin` carries the *magnitude* of the NI/Eq/Sup-by-margin boundary
(direction comes from `higher_is_better`). Pass a positive number.

### `ValueError: p1, p2 must be in (0, 1)` (proportions)

Proportions on a closed boundary aren't allowed. Use 0.001 / 0.999 if
you really want to model near-deterministic outcomes.

### `ValueError: supply only one of (p1, allocation), not both`

`logrank_freedman` accepts either convention but not both at once.
- `allocation` is the **n2/n1 ratio** (matches every other two-arm
  method).
- `p1` is the **fraction of N assigned to group 1**.
- Default is balanced 1:1 (`allocation=1.0`, equivalent to `p1=0.5`).

### `RuntimeError: failed to bracket N within 10000000`

The solver could not find a sample size for which the requested power
is achievable. Usual causes:

| Method family | Likely cause |
|---|---|
| Means tests | `mean1 == mean2`, or SD so large that the effect is below 0.001 SD |
| Proportions | `p1 ≈ p2`; consider whether your design is sensitive enough |
| Non-inferiority | true effect on the *wrong* side of zero relative to the margin |
| Superiority-by-margin | **`|mean1 - mean2| ≤ margin`** — the design cannot demonstrate the superiority you want; raise the assumed effect or lower the margin |
| Cox regression | extremely small `B` or extremely low `event_rate` |

The fix is statistical, not technical: revise the assumptions.

---

## CLI / argparse errors

### `argument --kind: invalid choice: 'foo'`

`samplesize report --kind` accepts only:

```
power-curve | protocol | grant | sensitivity | r-code | sas-code
```

For the most up-to-date list:
```sh
samplesize report --help
```

### `argument --lang: invalid choice: 'ja'`

A protocol template file is missing for that language. Available
languages are derived from
`samplesize/reporting/templates/protocol.<lang>.yaml`. Add a file there
and `--lang ja` works immediately (the choices list is built at start
time, so restart the process after adding a template).

### `--vary required for --kind sensitivity`

Supply at least one `--vary` spec, e.g.:

```sh
samplesize report <audit> --kind sensitivity --vary "sd=15,20,25"
```

For a 2-D grid pass `--vary` twice (one for row, one for column).
More than 2 dimensions is rejected (output would be unreadable).

### `--vary spec must be 'key=v1,v2,...'`

Use the exact form `key=v1,v2,v3`. Spaces inside the comma-separated
list are ignored; spaces around the `=` are fine.

### `audit file not found: ...`

`samplesize report` expects the path printed by an earlier `calc`. The
file lives under `.samplesize/audit/` of the working directory by
default (override with `SAMPLESIZE_AUDIT_DIR`).

---

## Conceptual surprises (not bugs)

### "My N differs from a reference answer by 1"

Some reference implementations round based on the noncentral-t CDF at a
target power threshold. scipy's `nct.cdf` differs from certain reference
tools by ~5e-5 near `power = 0.90`, which is enough to push the integer N
by one when you sit right on the boundary. Documented for
`non_inferiority_two_means` (Julious) and `superiority_by_margin_two_means`.
The achieved-power values agree.

### "My Pearson power differs from a textbook"

`pearson_correlation` defaults to `method="exact"` (Guenther/Hotelling
density via ₂F₁ + scipy.integrate). This matches validated reference
examples to ≥4 sig.fig.

If you want the textbook (`pwr` package) approximation, pass
`method="fisher-z"` explicitly. The two backends usually differ by
≈0.005–0.02 in power at small N, and by 0–1 in integer N.

### "TOST/equivalence α convention"

Two conventions are common:

- **TOST one-sided α**: each of the two one-sided tests at level α,
  total Type I = α. Confidence interval shown is at `1-2α`. The
  bio­equivalence guidance (FDA, EMA) uses α = 0.05 (90 % CI).
- **Two-sided CI convention**: confidence interval at `1-α`, each
  side at α/2.

Pass whichever your protocol uses. Our `equivalence_two_means` takes
α per one-sided test, so `alpha=0.025` corresponds to a 95 % CI and
`alpha=0.05` to a 90 % CI.

### "Non-inferiority `alpha` defaults to 0.025"

Our `non_inferiority_*` calculators use `alpha=0.025` by default —
that is the FDA convention (equivalent to a two-sided 95 % CI
excluding the margin). Override explicitly if your protocol differs.

### "Superiority by a margin needs a true effect bigger than margin"

If `mean1 - mean2 ≤ margin`, no finite N can demonstrate superiority by
that margin. The solver returns `failed to bracket N`. Either raise
your hypothesised effect (perhaps too conservative) or lower the
margin (perhaps too ambitious).

---

## When in doubt

1. **Run doctor.** `python -m samplesize doctor` is the
   one-command sanity check.
2. **Run a known-good fixture.** `pytest tests/validation -k <method>`
   re-runs worked examples for that method.
3. **Inspect the actual signature.**
   `python -m samplesize show <method_id>` prints every accepted
   kwarg, default, and whether it's required.
4. **Look at an audit.** `.samplesize/audit/*.json` records exactly
   what was sent in, what came out, library versions, citation. If you
   open a bug report, attach this file.
