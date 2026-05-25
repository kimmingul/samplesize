# Example 3: Cox regression sample size

A two-arm survival trial aims to detect a hazard ratio of HR = 0.7
(treatment reduces the hazard by 30%).  The expected event rate over the
follow-up window is 20%.  We want 80% power at α = 0.05, two-sided.

The Hsieh & Lavori (2000) formula is used, which works directly with the
log-hazard-ratio coefficient `B = log(HR)` and the standard deviation of
the covariate.  For a balanced binary treatment indicator (50 % in each
arm) `sd_x = 0.5`.

## Compute sample size

```python
import math
from samplesize.tests.cox import cox_regression

result = cox_regression(
    B=math.log(0.7),      # log(HR) = -0.3567
    sd_x=0.5,             # SD of binary arm indicator
    event_rate=0.20,      # proportion who experience the event
    alpha=0.05,
    power=0.80,
    sides=2,
    solve_for="n",
)

print(f"n (total)      = {result['n']}")
print(f"events needed  = {result['events']}")
print(f"achieved power = {result['achieved_power']:.4f}")
```

Expected output:

```
n (total)      = 1234
events needed  = 247
achieved power = 0.8001
```

## Inspect the envelope

```python
{
    "method_id": "cox_regression",
    "solve_for": "n",
    "n": 1234,
    "events": 247,
    "achieved_power": 0.8001,
    "inputs_echo": {"B": -0.3567, "sd_x": 0.5, "event_rate": 0.2, ...},
    "citations": [
        "Hsieh, F.Y. and Lavori, P.W. (2000)...",
        "Schoenfeld, D.A. (1983)...",
    ],
}
```

The result reports both the total headcount `n` and the number of
`events` required — the latter is the quantity that directly drives power
in event-driven trials and is often the figure that appears in a DSMB
charter.

## Solve for power at a fixed N

If enrolment is capped at 1,232 participants:

```python
result = cox_regression(
    B=math.log(0.7),
    sd_x=0.5,
    event_rate=0.20,
    alpha=0.05,
    n=1232,
    sides=2,
    solve_for="power",
)
print(f"power at n=1232: {result['achieved_power']:.4f}")
```

```
power at n=1232: 0.7994
```

Losing two participants has a negligible effect on power at this scale.

## Sensitivity table

Power requirements change sharply as the assumed HR moves toward the null:

```python
import math
for hr in (0.5, 0.6, 0.7, 0.8, 0.9):
    r = cox_regression(
        B=math.log(hr), sd_x=0.5, event_rate=0.20,
        alpha=0.05, power=0.80, sides=2, solve_for="n",
    )
    print(f"HR = {hr:.1f} → n = {r['n']}")
```

```
HR = 0.5 → n = 327
HR = 0.6 → n = 602
HR = 0.7 → n = 1234
HR = 0.8 → n = 3153
HR = 0.9 → n = 14142
```

The near-null HR = 0.9 scenario demands more than 14,000 participants —
a useful reality check when a sponsor proposes ambitious enrolment caps.

## Notes on the `r_squared` parameter

When the treatment indicator is correlated with other prognostic
covariates already in the Cox model, supply `r_squared` (the R² of the
treatment variable regressed on those covariates).  The default `r_squared=0.0`
assumes an uncorrelated covariate, which is the standard assumption for
a randomised trial.

## Audit record

Every call writes a JSON audit record to `.samplesize/<timestamp>.json`
containing inputs, outputs, library versions, and the method citation —
ready to attach to a study protocol or IRB submission.
