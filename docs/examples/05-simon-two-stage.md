# Example 5: Simon's two-stage Phase II design

A Phase II oncology trial is testing a new agent.  The uninteresting
response rate is p₀ = 0.10; the response rate that would warrant further
development is p₁ = 0.30.  We want type-I error α ≤ 0.10 and type-II
error β ≤ 0.10 (i.e. at least 90% power).

Simon's two-stage design minimises the expected sample size under H₀,
allowing early stopping for futility after the first stage.

## Compute the optimal design

```python
from samplesize.tests.phase_ii import simon_optimal_two_stage

result = simon_optimal_two_stage(
    p0=0.10,
    p1=0.30,
    alpha=0.10,
    beta=0.10,
)

d = result["design"]
print(f"Stage 1: enrol n1={d['n1']}, stop if ≤ r1={d['r1']} responses")
print(f"Stage 2: enrol to n={d['n']}, reject if ≤ r={d['r']} responses")
print(f"Achieved alpha = {d['alpha_actual']:.4f}")
print(f"Achieved power = {result['achieved_power']:.4f}")
print(f"E[N | H0]      = {d['EN_under_h0']:.1f}")
print(f"PET (H0)       = {d['PET']:.4f}")
```

Expected output:

```
Stage 1: enrol n1=18, stop if ≤ r1=2 responses
Stage 2: enrol to n=26, reject if ≤ r=4 responses
Achieved alpha = 0.0995
Achieved power = 0.9037
E[N | H0]      = 20.1
PET (H0)       = 0.7338
```

## Interpret the design

| Quantity | Value | Meaning |
|---|---|---|
| n1 | 18 | Patients in stage 1 |
| r1 | 2 | Stop for futility if ≤ 2 responses in stage 1 |
| n | 26 | Maximum total patients (both stages) |
| r | 4 | Declare inactive if ≤ 4 total responses |
| E[N \| H₀] | 20.1 | Expected enrolment if drug is truly inactive |
| PET | 0.734 | Probability of stopping after stage 1 under H₀ |

A PET of 73% means that in nearly three-quarters of trials where the
drug is truly inactive, the study will stop after only 18 patients —
the key efficiency gain over a single-stage design.

## Inspect the envelope

```python
{
    "method_id": "simon_optimal_two_stage",
    "solve_for": "n",
    "n": 26,
    "design": {
        "r1": 2, "n1": 18,
        "r": 4,  "n": 26,
        "alpha_actual": 0.0995,
        "beta_actual": 0.0963,
        "EN_under_h0": 20.1,
        "PET": 0.7338,
    },
    "achieved_power": 0.9037,
    "inputs_echo": {"p0": 0.1, "p1": 0.3, "alpha": 0.1, "beta": 0.1, ...},
    "citations": ["Simon, R. (1989)..."],
}
```

## Sensitivity table

How the design changes as the target response rate p₁ varies:

```python
for p1 in (0.20, 0.25, 0.30, 0.35, 0.40):
    r = simon_optimal_two_stage(p0=0.10, p1=p1, alpha=0.10, beta=0.10)
    d = r["design"]
    print(
        f"p1={p1:.2f} → n1={d['n1']:3d}, n={d['n']:3d}, "
        f"r1={d['r1']}, r={d['r']}"
    )
```

```
p1=0.20 → n1= 49, n= 90, r1=5, r=12
p1=0.25 → n1= 21, n= 50, r1=2, r= 7
p1=0.30 → n1= 18, n= 26, r1=2, r= 4
p1=0.35 → n1= 11, n= 19, r1=1, r= 3
p1=0.40 → n1=  5, n= 18, r1=0, r= 3
```

Detecting only a doubling of response rate (p₁ = 0.20) requires 90
patients versus 26 for a tripling (p₁ = 0.30) — a common argument for
selecting agents with larger expected signals for Phase II evaluation.

## Minimax alternative

`simon_minimax_two_stage` minimises the maximum sample size instead of
the expected sample size.  For the same design parameters:

```python
from samplesize.tests.phase_ii import simon_minimax_two_stage

r = simon_minimax_two_stage(p0=0.10, p1=0.30, alpha=0.10, beta=0.10)
d = r["design"]
print(f"Minimax: n1={d['n1']}, n={d['n']}, r1={d['r1']}, r={d['r']}")
print(f"E[N | H0] = {d['EN_under_h0']:.1f},  PET = {d['PET']:.4f}")
```

```
Minimax: n1=16, n=25, r1=1, r=4
E[N | H0] = 20.4,  PET = 0.5147
```

The minimax design saves one patient at maximum (25 vs 26) but has a
lower PET (0.51 vs 0.73), so it is less efficient at early stopping.
The choice between optimal and minimax depends on whether the trial
prioritises expected or worst-case enrolment.

## Audit record

Every call writes a JSON audit record to `.samplesize/<timestamp>.json`
containing inputs, outputs, library versions, and the method citation —
ready to attach to a study protocol or IRB submission.
