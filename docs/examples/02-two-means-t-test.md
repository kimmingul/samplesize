# Example 2: Two-sample independent t-test (equal variance)

A clinical team is planning a randomised trial comparing a new treatment
against a control group. Based on prior data they expect a standardised
effect of d = 0.5 (half a standard deviation difference in the primary
outcome), and they want 80% power at α = 0.05, two-sided.

## Compute sample size

```python
from samplesize.tests.two_means import two_sample_t_equal_var

result = two_sample_t_equal_var(
    mean1=0.0,
    mean2=0.5,
    sd=1.0,
    alpha=0.05,
    power=0.80,
    sides=2,
    solve_for="n",
)

print(f"n per group = {result['n1']}")
print(f"total n     = {result['n']}")
print(f"achieved power = {result['achieved_power']:.4f}")
```

Expected output:

```
n per group = 64
total n     = 128
achieved power = 0.8015
```

## Inspect the envelope

```python
{
    "method_id": "two_sample_t_equal_var",
    "solve_for": "n",
    "n1": 64,
    "n2": 64,
    "n": 128,
    "achieved_power": 0.8015,
    "effect_d": -0.5,
    "inputs_echo": {"mean1": 0.0, "mean2": 0.5, "sd": 1.0, ...},
    "citations": ["Julious, S.A. (2010)...", "Machin et al. (1997)..."],
}
```

The result reports both per-group sizes (`n1`, `n2`) and the total (`n`),
making it easy to fill in both fields of a protocol template.  The
`allocation` parameter (default 1.0) lets you request an unbalanced
design; `n2` will be scaled accordingly.

## Solve for power at a fixed N

Budget constrains enrolment to 60 participants per arm:

```python
result = two_sample_t_equal_var(
    mean1=0.0,
    mean2=0.5,
    sd=1.0,
    alpha=0.05,
    n1=60,
    sides=2,
    solve_for="power",
)
print(f"power at n1=60: {result['achieved_power']:.4f}")
```

```
power at n1=60: 0.7753
```

A shortfall of four participants per arm drops power from 80% to ~78%.

## Sensitivity table

How total enrolment scales with the assumed effect:

```python
for delta in (0.3, 0.4, 0.5, 0.6, 0.7):
    r = two_sample_t_equal_var(
        mean1=0, mean2=delta, sd=1.0,
        alpha=0.05, power=0.80, sides=2, solve_for="n",
    )
    print(f"d = {delta:.1f} → n per group = {r['n1']}")
```

```
d = 0.3 → n per group = 176
d = 0.4 → n per group = 100
d = 0.5 → n per group = 64
d = 0.6 → n per group = 45
d = 0.7 → n per group = 34
```

Halving the detectable effect from d = 0.6 to d = 0.3 nearly quintuples
the required sample.

## Audit record

Every call writes a JSON audit record to `.samplesize/<timestamp>.json`
containing inputs, outputs, library versions, and the method citation —
ready to attach to a study protocol or IRB submission.
