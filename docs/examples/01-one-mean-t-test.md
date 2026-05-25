# Example 1: One-sample t-test

A pilot study suggests a single-group mean of 0.5 (standardised) against a
null of 0.0, with σ = 1.0. We want 80% power at α = 0.05 (two-sided).

## Compute sample size

```python
from samplesize.tests.one_mean import one_sample_t

result = one_sample_t(
    mean0=0.0,
    mean1=0.5,
    sd=1.0,
    alpha=0.05,
    power=0.80,
    sides=2,
    solve_for="n",
)

print(f"n = {result['n']}")
print(f"achieved power = {result['achieved_power']:.4f}")
```

Expected output:

```
n = 34
achieved power = 0.8078
```

## Inspect the envelope

Every calculator returns a standard envelope:

```python
{
    "method_id": "one_sample_t",
    "solve_for": "n",
    "n": 34,
    "achieved_power": 0.8078,
    "effect_d": 0.5,
    "inputs_echo": {"mean0": 0.0, "mean1": 0.5, "sd": 1.0, ...},
    "citations": ["Cohen, J. (1988). Statistical Power Analysis ..."],
}
```

The `inputs_echo` echoes back exactly what you passed in, so the result
record is self-describing — useful for audit logs and protocol writing.

## Solve for power at a fixed N

```python
result = one_sample_t(
    mean0=0.0,
    mean1=0.5,
    sd=1.0,
    alpha=0.05,
    n=30,
    sides=2,
    solve_for="power",
)
print(f"power at n=30: {result['achieved_power']:.4f}")
```

```
power at n=30: 0.7540
```

## Sensitivity table

How sample size scales with effect size:

```python
for delta in (0.3, 0.4, 0.5, 0.6, 0.7):
    r = one_sample_t(mean0=0, mean1=delta, sd=1.0,
                     alpha=0.05, power=0.80, sides=2, solve_for="n")
    print(f"effect = {delta:.1f} → n = {r['n']}")
```

```
effect = 0.3 → n = 90
effect = 0.4 → n = 52
effect = 0.5 → n = 34
effect = 0.6 → n = 24
effect = 0.7 → n = 19
```

## Audit record

Every call writes a JSON audit record to `.samplesize/<timestamp>.json`
with the inputs, outputs, library versions, and method citation —
useful for reproducibility and protocol attachments.
