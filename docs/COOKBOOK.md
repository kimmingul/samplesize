# Cookbook — realistic studies, end to end

Each recipe is a real study sketch: design, why a particular method
matches, the exact CLI call, expected output, and interpretation
notes. Recipes are independent — open the one closest to your study,
adapt the numbers.

All examples assume:

```sh
cd /path/to/samplesize-copilot   # or wherever you cloned
# (no install needed for CLI use; `pip install -e .` makes it path-free)
```

Replace `<audit_json>` with the path printed at the end of each `calc`
run (look under `.samplesize/audit/`).

---

## 1. Two-arm RCT, continuous outcome — antihypertensive vs placebo

**Design.** Phase III RCT. Primary endpoint: change in systolic BP at
12 weeks. Expected mean reduction 10 mmHg in treatment, 0 in control.
Pooled SD 20 mmHg. Two-sided α = 0.05, target power 0.80.

**Method.** `two_sample_t_equal_var`. Continuous outcome, 2 independent
arms, equal variance plausible.

```sh
python3 -m samplesize calc two_sample_t_equal_var \
  --json-args '{"mean1":10,"mean2":0,"sd":20,"alpha":0.05,"power":0.80,"sides":2}'
```

**Expect.** N = 128 (64 + 64). Achieved power 0.8015.
Cohen's d = 0.5 (medium effect).

**Follow-ups.**
- Sensitivity to SD assumption:
  ```sh
  samplesize report <audit> --kind sensitivity --vary "sd=15,20,25,30"
  ```
- Protocol text:
  ```sh
  samplesize report <audit> --kind protocol --lang en
  ```

---

## 2. Pre-post weight-loss study — paired t

**Design.** Same individuals measured before and after a 12-week
exercise programme. Expected weight drop of 5 lbs; SD of differences
~10 lbs. Two-sided α = 0.05, power 0.80.

**Method.** `paired_t`. Pre/post on the same subjects ⇒ paired
differences, one-sample test on the difference.

```sh
samplesize calc paired_t \
  --json-args '{"mean_diff":-5,"sd_diff":10,"alpha":0.05,"power":0.80,"sides":2}'
```

**Expect.** N = 34 pairs, achieved 0.808.

**Variant.** SD of differences unknown — try several values:
```sh
samplesize report <audit> --kind sensitivity --vary "sd_diff=8,10,12,15"
```

---

## 3. Single-arm pilot vs historical control — one-sample t

**Design.** A biomarker has historical mean 100 (known from a published
cohort, σ ≈ 40). Pilot trial aims to detect a shift to 110 (or larger)
with N = 50 patients.

**Method.** `one_sample_t`. Single sample, mean test against known μ₀.

```sh
samplesize calc one_sample_t \
  --json-args '{"mean0":100,"mean1":110,"sd":40,"n":50,"alpha":0.05,"sides":2}'
```

This solves for **power**, not N. Expect achieved power ≈ 0.32 — the
pilot is under-powered if the true effect is only 0.25 SD.

**Re-plan** for 80 % power:
```sh
samplesize calc one_sample_t \
  --json-args '{"mean0":100,"mean1":110,"sd":40,"power":0.80,"alpha":0.05,"sides":2}'
```
N ≈ 128.

---

## 4. Dose-finding — one-way ANOVA, 4 arms

**Design.** Comparing 4 dose levels of a drug; primary endpoint is a
continuous biomarker. Hypothesised means 9.78 / 12.0 / 12.0 / 14.23,
pooled SD 3. Two-sided α = 0.05, power 0.80.

**Method.** `one_way_anova_f`. ≥3 independent groups, continuous
outcome, single overall F-test.

```sh
samplesize calc one_way_anova_f \
  --json-args '{"means":[9.775,12,12,14.225],"sigma":3,"alpha":0.05,"power":0.80}'
```

**Expect.** n = 11 per arm (Fleiss 1986 validation example). Total
N = 44.

---

## 5. Vaccine efficacy — two-arm trial, binary endpoint

**Design.** Two-arm randomised trial. Expected 6-month infection rate
12 % in placebo, 4 % in vaccine. Equal allocation, two-sided α = 0.05,
power 0.80. Z-pooled test.

**Method.** `two_proportions`. 2 independent arms, binary outcome.

```sh
samplesize calc two_proportions \
  --json-args '{"p1":0.04,"p2":0.12,"alpha":0.05,"power":0.80,"sides":2,"test_type":"z_pooled"}'
```

**Expect.** Per-arm ≈ 300, total ≈ 600.

**Sensitivity to placebo rate**:
```sh
samplesize report <audit> --kind sensitivity --vary "p2=0.10,0.12,0.15"
```

---

## 6. Single-arm Phase II — one-proportion

**Design.** Phase II oncology trial. Historical response rate 30 %.
New regimen expected to push to 50 %. One-sided α = 0.05, power 0.80.

**Method.** `one_proportion` with `z_s0` test.

```sh
samplesize calc one_proportion \
  --json-args '{"p0":0.30,"p1":0.50,"alpha":0.05,"power":0.80,"sides":1,"test_type":"z_s0"}'
```

**Expect.** N ≈ 35–40 depending on test variant.

**Variant.** With continuity correction:
```sh
... "test_type":"z_s0_cc"
```

---

## 7. Diagnostic test agreement — McNemar (paired binary)

**Design.** New rapid test vs reference, both applied to N patients.
Pre-specify discordant proportions: p₁₀ = 0.20 (positive on new,
negative on reference), p₀₁ = 0.10 (vice versa). Two-sided α = 0.05,
power 0.80.

**Method.** `mcnemar`. Paired binary outcomes.

```sh
samplesize calc mcnemar \
  --json-args '{"p10":0.20,"p01":0.10,"alpha":0.05,"power":0.80,"sides":2}'
```

**Expect.** N ≈ 130–150 paired observations (binomial enumeration).

---

## 8. Survey analysis — chi-square contingency

**Design.** 3 × 3 contingency table (e.g., 3 age groups × 3 attitude
levels). Expect Cohen's w = 0.30 (medium effect). df = (3−1)(3−1) = 4.
α = 0.05, power 0.80.

**Method.** `chi_square`.

```sh
samplesize calc chi_square \
  --json-args '{"w":0.30,"df":4,"alpha":0.05,"power":0.80}'
```

**Expect.** N = 133 (reference example 2).

---

## 9. Two-arm survival — logrank (Freedman)

**Design.** RCT for cancer. 5-year survival: 25 % (standard) → 50 %
(new). One-sided α = 0.05, power 0.90, equal allocation, no
loss-to-follow-up.

**Method.** `logrank_freedman`. Two-arm survival, proportional hazards.

```sh
samplesize calc logrank_freedman \
  --json-args '{"s1":0.25,"s2":0.50,"alpha":0.05,"power":0.90,"sides":1,"allocation":1.0}'
```

**Expect.** N = 124 (62 + 62), HR ≈ 0.5, achieved power 0.9014.

**With 15 % dropout assumed:**
```sh
... "loss_to_followup":0.15
```

---

## 10. Cox regression with covariate — Hsieh–Lavori

**Design.** Cox regression study. Primary covariate is a continuous
biomarker (SD 0.5). Hypothesised log-HR per SD = ln(1.5) ≈ 0.4055.
Expected event rate 71 %. R² with other covariates ≈ 0 (no
confounders). α = 0.05 two-sided, power 0.80.

**Method.** `cox_regression`.

```sh
samplesize calc cox_regression \
  --json-args '{"B":0.4055,"sd_x":0.5,"event_rate":0.71,"alpha":0.05,"power":0.80,"sides":2}'
```

**Expect.** N = 212 (Schoenfeld 1983 validation).

---

## 11. Psychometric validation — Pearson correlation

**Design.** Validating a new questionnaire against an established
measure. Expected correlation ρ = 0.4. α = 0.05 two-sided, power 0.80.

**Method.** `pearson_correlation` (default `method="exact"` uses the
Guenther/Hotelling density).

```sh
samplesize calc pearson_correlation \
  --json-args '{"r":0.40,"alpha":0.05,"power":0.80,"sides":2}'
```

**Expect.** N ≈ 46.

**Compare with Fisher-z approximation:**
```sh
... "method":"fisher-z"
```
Usually matches within 1 subject.

---

## 12. Non-inferiority trial — two-proportion, FDA convention

**Design.** New treatment vs reference. Historical reference success
rate 65 %. Plan: enrol enough to rule out a margin of −10 percentage
points. Assumed new treatment effect = same as reference (+0 pp).
One-sided α = 0.025 (FDA convention: equivalent to two-sided 95 % CI).
Power 0.80.

**Method.** `non_inferiority_two_proportions`, higher-is-better.

```sh
samplesize calc non_inferiority_two_proportions \
  --json-args '{"p1":0.65,"p2":0.65,"margin":0.10,"alpha":0.025,"power":0.80,"higher_is_better":true}'
```

**Expect.** Per-arm ≈ 300 (mirrors Julius & Campbell 2012 Table XIII).

**Grant-style writeup:**
```sh
samplesize report <audit> --kind grant
```

---

## 13. Equivalence trial — two means, TOST

**Design.** Generic drug vs reference. Both groups expected to perform
identically (mean1 = mean2). Equivalence window ±10 in original units;
within-subject SD 100. α = 0.025 per side (95 % CI). Power 0.90.

**Method.** `equivalence_two_means`, symmetric margin.

```sh
samplesize calc equivalence_two_means \
  --json-args '{"mean1":0,"mean2":0,"sd":100,"margin":10,"alpha":0.025,"power":0.90}'
```

**Expect.** N = 5,200 (2,600 + 2,600). The N is large because SD ≈
margin — feasibility check warranted.

**Sensitivity to SD:**
```sh
samplesize report <audit> --kind sensitivity --vary "sd=80,100,120"
```

---

## 14. Superiority by a margin — two means

**Design.** Demonstrate that the treatment mean exceeds control by
**at least** a clinically-meaningful margin of 0.575 (raw units). True
expected gap = 1.725, pooled SD = 3. One-sided α = 0.025, power 0.90.

**Method.** `superiority_by_margin_two_means`. True effect must exceed
the margin — otherwise the test has no power.

```sh
samplesize calc superiority_by_margin_two_means \
  --json-args '{"mean1":1.725,"mean2":0,"sd":3.0,"margin":0.575,"alpha":0.025,"power":0.90,"higher_is_better":true}'
```

**Expect.** Per-arm = 144, achieved 0.90004.

**Common pitfall.** Setting `mean1 - mean2 ≤ margin` makes the test
impossible — the function raises "failed to bracket N". The true
expected effect must be **larger** than the margin you want to rule
out below.

---

## 15. End-to-end demo: design → calc → report → R code

Putting it all together for the same 5-year-survival RCT (recipe 9):

```sh
# 1. discover method
samplesize show logrank_freedman | head -20

# 2. compute
samplesize calc logrank_freedman \
  --json-args '{"s1":0.25,"s2":0.50,"alpha":0.05,"power":0.90,"sides":1}'

# 3. capture the audit JSON path it printed, then build reports
AUDIT=$(ls -t .samplesize/audit/*.json | head -1)

# Power curve
samplesize report "$AUDIT" --kind power-curve --out curve.png

# Sensitivity to S2 (treatment-arm survival)
samplesize report "$AUDIT" --kind sensitivity --vary "s2=0.40,0.45,0.50,0.55"

# ICH E9 protocol section
samplesize report "$AUDIT" --kind protocol --lang en

# R equivalent code (for collaborator verification)
samplesize report "$AUDIT" --kind r-code
```

---

## Tips for adapting these recipes to your study

- **`samplesize show <id>`** lists every accepted kwarg with its
  default — adapt blindly to your scenario by checking this first.
- **`samplesize doctor`** validates the install end-to-end if anything
  feels wrong.
- **Audit JSON** files in `.samplesize/audit/` contain inputs, results,
  library versions, and citation — exactly what reviewers ask for.
- **Multiple `--vary`** in `--kind sensitivity` produces a 2-D grid;
  one `--vary` gives a 1-D column.
- **`--method "fisher-z"`** on `pearson_correlation` recovers the
  textbook (pwr-package) approximation when you need cross-tool
  comparison.
