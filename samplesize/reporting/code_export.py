"""Export equivalent R / SAS code for a finished sample-size result.

The audit JSON carries inputs and `method_id`.  We dispatch by method id
to a per-method generator that emits a runnable R script (using
`pwr` / `gsDesign` / `powerSurvEpi`) or a SAS PROC POWER block.  Where
no widely-installed package matches this procedure, we emit a
clearly-labelled fallback: the closed-form formula, simulation hint,
or an explicit "no direct equivalent" comment.

Registry pattern: append `r_<method_id>` / `sas_<method_id>` helpers
and route via `_R_DISPATCH` / `_SAS_DISPATCH`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def _load_audit(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


# ---------- R generators -----------------------------------------------------

def _r_two_sample_t(inputs: dict, result: dict) -> str:
    d = abs(inputs["mean1"] - inputs["mean2"]) / inputs["sd"]
    sides = inputs.get("sides", 2)
    alt = "two.sided" if sides == 2 else "greater"
    return (
        '# R equivalent — Two-sample t-test (equal variance)\n'
        'library(pwr)\n'
        f'pwr.t.test(d = {d:.6g},\n'
        f'           sig.level = {inputs["alpha"]},\n'
        f'           power = {inputs.get("power") or result["achieved_power"]:.4f},\n'
        f'           type = "two.sample",\n'
        f'           alternative = "{alt}")\n'
        '# Total N = 2 * ceiling(n) for balanced design\n'
    )


def _r_one_sample_t(inputs: dict, result: dict) -> str:
    d = abs(inputs["mean1"] - inputs["mean0"]) / inputs["sd"]
    sides = inputs.get("sides", 2)
    alt = "two.sided" if sides == 2 else "greater"
    return (
        '# R equivalent — One-sample t-test\n'
        'library(pwr)\n'
        f'pwr.t.test(d = {d:.6g},\n'
        f'           sig.level = {inputs["alpha"]},\n'
        f'           power = {inputs.get("power") or result["achieved_power"]:.4f},\n'
        f'           type = "one.sample",\n'
        f'           alternative = "{alt}")\n'
    )


def _r_paired_t(inputs: dict, result: dict) -> str:
    d = abs(inputs["mean_diff"]) / inputs["sd_diff"]
    sides = inputs.get("sides", 2)
    alt = "two.sided" if sides == 2 else "greater"
    return (
        '# R equivalent — Paired t-test (operates on differences)\n'
        'library(pwr)\n'
        f'pwr.t.test(d = {d:.6g},\n'
        f'           sig.level = {inputs["alpha"]},\n'
        f'           power = {inputs.get("power") or result["achieved_power"]:.4f},\n'
        f'           type = "paired",\n'
        f'           alternative = "{alt}")\n'
    )


def _r_one_proportion(inputs: dict, result: dict) -> str:
    import math
    p0, p1 = inputs["p0"], inputs["p1"]
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0))
    sides = inputs.get("sides", 2)
    alt = "two.sided" if sides == 2 else "greater"
    return (
        '# R equivalent — One-sample proportion (Cohen h via pwr)\n'
        '# Note: pwr uses Cohen h on the arcsine scale.\n'
        'library(pwr)\n'
        f'pwr.p.test(h = {h:.6g},\n'
        f'           sig.level = {inputs["alpha"]},\n'
        f'           power = {inputs.get("power") or result["achieved_power"]:.4f},\n'
        f'           alternative = "{alt}")\n'
    )


def _r_two_proportions(inputs: dict, result: dict) -> str:
    import math
    p1, p2 = inputs["p1"], inputs["p2"]
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    sides = inputs.get("sides", 2)
    alt = "two.sided" if sides == 2 else "greater"
    return (
        '# R equivalent — Two-sample proportion (Cohen h via pwr)\n'
        '# pwr uses arcsine-h; see pwr package documentation for available test statistics.\n'
        'library(pwr)\n'
        f'pwr.2p.test(h = {h:.6g},\n'
        f'            sig.level = {inputs["alpha"]},\n'
        f'            power = {inputs.get("power") or result["achieved_power"]:.4f},\n'
        f'            alternative = "{alt}")\n'
    )


def _r_pearson_correlation(inputs: dict, result: dict) -> str:
    sides = inputs.get("sides", 2)
    alt = "two.sided" if sides == 2 else "greater"
    return (
        '# R equivalent — Pearson correlation\n'
        'library(pwr)\n'
        f'pwr.r.test(r = {inputs["r"]},\n'
        f'           sig.level = {inputs["alpha"]},\n'
        f'           power = {inputs.get("power") or result["achieved_power"]:.4f},\n'
        f'           alternative = "{alt}")\n'
    )


def _r_one_way_anova_f(inputs: dict, result: dict) -> str:
    import math
    means = inputs["means"]
    sigma = inputs["sigma"]
    mu_bar = sum(means) / len(means)
    f = math.sqrt(sum((m - mu_bar) ** 2 for m in means) / len(means)) / sigma
    return (
        '# R equivalent — One-way ANOVA (Cohen f effect size)\n'
        'library(pwr)\n'
        f'pwr.anova.test(k = {len(means)},\n'
        f'               f = {f:.6g},\n'
        f'               sig.level = {inputs["alpha"]},\n'
        f'               power = {inputs.get("power") or result["achieved_power"]:.4f})\n'
    )


def _r_chi_square(inputs: dict, result: dict) -> str:
    return (
        '# R equivalent — Chi-square (Cohen w)\n'
        'library(pwr)\n'
        f'pwr.chisq.test(w = {inputs["w"]},\n'
        f'               df = {inputs["df"]},\n'
        f'               sig.level = {inputs["alpha"]},\n'
        f'               power = {inputs.get("power") or result["achieved_power"]:.4f})\n'
    )


def _r_logrank_freedman(inputs: dict, result: dict) -> str:
    return (
        '# R equivalent — Logrank Freedman; no widely-installed pwr.* function.\n'
        '# Use `nph::sample.size.NPH()` or implement Freedman formula directly:\n'
        '#   HR = log(S2) / log(S1)\n'
        '#   phi = (1 - p1) / p1\n'
        '#   N = ((z_alpha + z_beta) * (1 + phi*HR) / (|HR-1| * sqrt(phi)))^2\n'
        '#       / (((1 - S1) + phi*(1 - S2)) / (1 + phi))\n'
        f'# Inputs: S1={inputs["s1"]} S2={inputs["s2"]} alpha={inputs["alpha"]} '
        f'power={inputs.get("power") or result["achieved_power"]:.4f}\n'
        f'# Result: N = {result.get("n")} total ({result.get("n1")} : {result.get("n2")})\n'
    )


def _r_cox_regression(inputs: dict, result: dict) -> str:
    return (
        '# R equivalent — Cox regression (Hsieh & Lavori 2000)\n'
        '# Use the `powerSurvEpi` package for full functionality:\n'
        '#   library(powerSurvEpi)\n'
        f'#   powerEpi(...)   # uses similar inputs\n'
        '# Closed form (Hsieh-Lavori): events D = (z_alpha + z_beta)^2 / ((1-R^2) * sigma_x^2 * B^2)\n'
        f'# Inputs: B={inputs["B"]} sd_x={inputs["sd_x"]} '
        f'event_rate={inputs["event_rate"]} R^2={inputs.get("r_squared", 0.0)}\n'
        f'# Result: N = {result.get("n")} ({result.get("events")} events)\n'
    )


def _r_unsupported(method_id: str):
    def gen(inputs, result):
        ap = result.get("achieved_power")
        ap_str = f"{ap:.4f}" if ap is not None else "N/A"
        return (
            f'# No widely-installed R package replicates this procedure for '
            f'{method_id}.\n'
            f'# Verify via simulation or refer to the closed-form audit:\n'
            f'#   inputs: {inputs}\n'
            f'#   result: N={result.get("n") or result.get("n_total")} '
            f'power={ap_str}\n'
        )
    return gen


_R_DISPATCH: dict[str, Callable[[dict, dict], str]] = {
    "one_sample_t": _r_one_sample_t,
    "two_sample_t_equal_var": _r_two_sample_t,
    "paired_t": _r_paired_t,
    "one_proportion": _r_one_proportion,
    "two_proportions": _r_two_proportions,
    "pearson_correlation": _r_pearson_correlation,
    "one_way_anova_f": _r_one_way_anova_f,
    "chi_square": _r_chi_square,
    "logrank_freedman": _r_logrank_freedman,
    "cox_regression": _r_cox_regression,
}


# ---------- SAS generators ---------------------------------------------------

def _sas_two_sample_t(inputs: dict, result: dict) -> str:
    sides = inputs.get("sides", 2)
    return (
        '/* SAS equivalent — Two-sample t-test (equal variance) */\n'
        'proc power;\n'
        '  twosamplemeans test=diff\n'
        f'    meandiff = {inputs["mean1"] - inputs["mean2"]}\n'
        f'    stddev = {inputs["sd"]}\n'
        f'    alpha = {inputs["alpha"]}\n'
        f'    power = {inputs.get("power") or result["achieved_power"]:.4f}\n'
        f'    sides = {sides}\n'
        '    ntotal = .;\n'
        'run;\n'
    )


def _sas_one_sample_t(inputs: dict, result: dict) -> str:
    sides = inputs.get("sides", 2)
    return (
        '/* SAS equivalent — One-sample t-test */\n'
        'proc power;\n'
        '  onesamplemeans\n'
        f'    mean = {inputs["mean1"]}\n'
        f'    nullmean = {inputs["mean0"]}\n'
        f'    stddev = {inputs["sd"]}\n'
        f'    alpha = {inputs["alpha"]}\n'
        f'    power = {inputs.get("power") or result["achieved_power"]:.4f}\n'
        f'    sides = {sides}\n'
        '    ntotal = .;\n'
        'run;\n'
    )


def _sas_paired_t(inputs: dict, result: dict) -> str:
    sides = inputs.get("sides", 2)
    return (
        '/* SAS equivalent — Paired t-test (one-sample on differences) */\n'
        'proc power;\n'
        '  pairedmeans test=diff\n'
        f'    meandiff = {inputs["mean_diff"]}\n'
        f'    stddev = {inputs["sd_diff"]}\n'
        f'    alpha = {inputs["alpha"]}\n'
        f'    power = {inputs.get("power") or result["achieved_power"]:.4f}\n'
        f'    sides = {sides}\n'
        '    npairs = .;\n'
        'run;\n'
    )


def _sas_two_proportions(inputs: dict, result: dict) -> str:
    sides = inputs.get("sides", 2)
    return (
        '/* SAS equivalent — Two-sample proportions */\n'
        'proc power;\n'
        '  twosamplefreq test=pchi\n'
        f'    groupproportions = ({inputs["p1"]} {inputs["p2"]})\n'
        f'    alpha = {inputs["alpha"]}\n'
        f'    power = {inputs.get("power") or result["achieved_power"]:.4f}\n'
        f'    sides = {sides}\n'
        '    ntotal = .;\n'
        'run;\n'
    )


def _sas_one_proportion(inputs: dict, result: dict) -> str:
    sides = inputs.get("sides", 2)
    return (
        '/* SAS equivalent — One-sample proportion */\n'
        'proc power;\n'
        '  onesamplefreq test=z method=normal\n'
        f'    proportion = {inputs["p1"]}\n'
        f'    nullproportion = {inputs["p0"]}\n'
        f'    alpha = {inputs["alpha"]}\n'
        f'    power = {inputs.get("power") or result["achieved_power"]:.4f}\n'
        f'    sides = {sides}\n'
        '    ntotal = .;\n'
        'run;\n'
    )


def _sas_pearson_correlation(inputs: dict, result: dict) -> str:
    sides = inputs.get("sides", 2)
    return (
        '/* SAS equivalent — Pearson correlation */\n'
        'proc power;\n'
        '  onecorr\n'
        f'    corr = {inputs["r"]}\n'
        f'    nullcorr = {inputs.get("rho0", 0.0)}\n'
        f'    alpha = {inputs["alpha"]}\n'
        f'    power = {inputs.get("power") or result["achieved_power"]:.4f}\n'
        f'    sides = {sides}\n'
        '    ntotal = .;\n'
        'run;\n'
    )


def _sas_one_way_anova_f(inputs: dict, result: dict) -> str:
    means = " ".join(str(m) for m in inputs["means"])
    return (
        '/* SAS equivalent — One-way ANOVA F-test */\n'
        'proc power;\n'
        '  onewayanova\n'
        f'    groupmeans = ({means})\n'
        f'    stddev = {inputs["sigma"]}\n'
        f'    alpha = {inputs["alpha"]}\n'
        f'    power = {inputs.get("power") or result["achieved_power"]:.4f}\n'
        '    npergroup = .;\n'
        'run;\n'
    )


def _sas_unsupported(method_id: str):
    def gen(inputs, result):
        ap = result.get("achieved_power")
        ap_str = f"{ap:.4f}" if ap is not None else "N/A"
        return (
            f'/* No direct SAS PROC POWER mapping for {method_id} */\n'
            f'/* Inputs: {inputs} */\n'
            f'/* Result: N={result.get("n") or result.get("n_total")} '
            f'power={ap_str} */\n'
        )
    return gen


_SAS_DISPATCH: dict[str, Callable[[dict, dict], str]] = {
    "one_sample_t": _sas_one_sample_t,
    "two_sample_t_equal_var": _sas_two_sample_t,
    "paired_t": _sas_paired_t,
    "one_proportion": _sas_one_proportion,
    "two_proportions": _sas_two_proportions,
    "pearson_correlation": _sas_pearson_correlation,
    "one_way_anova_f": _sas_one_way_anova_f,
}


# ---------- public entry -----------------------------------------------------

def r_code(audit_path: str | Path) -> str:
    rec = _load_audit(audit_path)
    method_id = rec.get("method_id") or rec["result"]["method_id"]
    res = rec["result"]
    inputs = {k: v for k, v in res.get("inputs_echo", {}).items() if v is not None}
    gen = _R_DISPATCH.get(method_id, _r_unsupported(method_id))
    return gen(inputs, res)


def sas_code(audit_path: str | Path) -> str:
    rec = _load_audit(audit_path)
    method_id = rec.get("method_id") or rec["result"]["method_id"]
    res = rec["result"]
    inputs = {k: v for k, v in res.get("inputs_echo", {}).items() if v is not None}
    gen = _SAS_DISPATCH.get(method_id, _sas_unsupported(method_id))
    return gen(inputs, res)
