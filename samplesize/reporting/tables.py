"""Sensitivity tables for sample-size results.

Given a base calculation, sweep one or two input parameters and emit
the resulting required-N (or achieved-power) as a markdown grid.  No
plotting — markdown is the right substrate for protocols.
"""
from __future__ import annotations

from typing import Any, Iterable

from samplesize.registry import resolve_method


def sensitivity_grid(
    *,
    method_id: str,
    base_inputs: dict[str, Any],
    row_param: str,
    row_values: Iterable[Any],
    col_param: str | None = None,
    col_values: Iterable[Any] | None = None,
    report_key: str = "n",
) -> str:
    """Return a markdown table of `report_key` over the row × col grid."""
    method = resolve_method(method_id)
    if method is None or not method.get("implemented"):
        raise ValueError(f"method {method_id!r} not implemented")
    fn = method["_callable"]

    row_vals = list(row_values)
    if col_param is None:
        col_vals = [None]
    else:
        col_vals = list(col_values or [])

    def run(rv, cv):
        kwargs = dict(base_inputs)
        kwargs[row_param] = rv
        if col_param is not None:
            kwargs[col_param] = cv
        try:
            res = fn(**kwargs)
            v = res.get(report_key)
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)
        except Exception as e:
            return f"err: {type(e).__name__}"

    lines: list[str] = []
    if col_param is None:
        lines.append(f"| {row_param} | {report_key} |")
        lines.append("|---|---|")
        for rv in row_vals:
            lines.append(f"| {rv} | {run(rv, None)} |")
    else:
        header = f"| {row_param} \\ {col_param} | " + " | ".join(
            str(cv) for cv in col_vals
        ) + " |"
        sep = "|---" * (len(col_vals) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for rv in row_vals:
            row = f"| {rv} | " + " | ".join(run(rv, cv) for cv in col_vals) + " |"
            lines.append(row)
    return "\n".join(lines) + "\n"
