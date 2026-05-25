# Example 4: Cluster-randomised two-means trial

A school-based mental-health intervention will randomise whole schools
rather than individual students.  Each school contributes approximately
30 students, the intracluster correlation (ICC) is estimated at 0.05,
and the expected standardised effect is d = 0.4.  We want 80% power at
α = 0.05, two-sided.

Clustering inflates the required sample relative to an individual-level
RCT.  The design effect here is **2.45** (= 1 + (30 − 1) × 0.05),
meaning roughly 2.45 times as many subjects are needed to achieve the
same power.

## Compute sample size

```python
from samplesize.tests.cluster import cluster_randomized_two_means

result = cluster_randomized_two_means(
    mean1=0.0,
    mean2=0.4,
    sd=1.0,
    m=30,          # students per school
    icc=0.05,
    alpha=0.05,
    power=0.80,
    sides=2,
    solve_for="n",
)

print(f"clusters per arm = {result['k1']}")
print(f"total clusters   = {result['k_total']}")
print(f"subjects per arm = {result['n1']}")
print(f"total subjects   = {result['n_total']}")
print(f"design effect    = {result['design_effect']}")
print(f"achieved power   = {result['achieved_power']:.4f}")
```

Expected output:

```
clusters per arm = 9
total clusters   = 18
subjects per arm = 270
total subjects   = 540
design effect    = 2.45
achieved power   = 0.8423
```

## Inspect the envelope

```python
{
    "method_id": "cluster_randomized_two_means",
    "solve_for": "n",
    "k1": 9, "k2": 9, "k_total": 18,
    "m_per_cluster": 30,
    "n1": 270, "n2": 270, "n_total": 540,
    "achieved_power": 0.8423,
    "design_effect": 2.45,
    "effect_d": -0.4,
    "inputs_echo": {"mean1": 0.0, "mean2": 0.4, "icc": 0.05, ...},
    "citations": ["Donner & Klar (1996, 2000)...", ...],
}
```

The result distinguishes cluster counts (`k1`, `k_total`) from subject
counts (`n1`, `n_total`), which map directly onto the two levels of the
study protocol.

## Solve for power at a fixed number of clusters

If only 8 schools per arm can be recruited:

```python
result = cluster_randomized_two_means(
    mean1=0.0,
    mean2=0.4,
    sd=1.0,
    m=30,
    icc=0.05,
    alpha=0.05,
    k_clusters=9,
    sides=2,
    solve_for="power",
)
print(f"power at k=9 per arm: {result['achieved_power']:.4f}")
```

```
power at k=9 per arm: 0.8423
```

## Sensitivity table

ICC uncertainty is often the dominant planning assumption.  Here is how
the required number of clusters scales across plausible ICC values:

```python
for icc in (0.01, 0.03, 0.05, 0.10, 0.15):
    r = cluster_randomized_two_means(
        mean1=0.0, mean2=0.4, sd=1.0,
        m=30, icc=icc,
        alpha=0.05, power=0.80, sides=2, solve_for="n",
    )
    print(f"ICC = {icc:.2f} → k per arm = {r['k1']}, n_total = {r['n_total']}")
```

```
ICC = 0.01 → k per arm = 5, n_total = 300
ICC = 0.03 → k per arm = 7, n_total = 420
ICC = 0.05 → k per arm = 9, n_total = 540
ICC = 0.10 → k per arm = 13, n_total = 780
ICC = 0.15 → k per arm = 18, n_total = 1080
```

Tripling the ICC from 0.05 to 0.15 doubles the required number of
clusters — a strong argument for collecting ICC pilot data before
finalising the protocol.

## Notes on the `cov` parameter

If cluster sizes are unequal, supply `cov` (coefficient of variation of
cluster size; typical values 0.4–0.9).  The default `cov=0.0` reproduces
the Donner & Klar (1996) equal-cluster formula used here.

## Audit record

Every call writes a JSON audit record to `.samplesize/<timestamp>.json`
containing inputs, outputs, library versions, and the method citation —
ready to attach to a study protocol or IRB submission.
