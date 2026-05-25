"""Power-curve and sensitivity plots for sample-size results.

Output is a PNG file written via matplotlib's `Agg` backend so no
display is required.  Curves are deliberately plain (one chart, one
purpose) so they drop cleanly into protocols and grant documents.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from samplesize.registry import resolve_method


def _label(name: str) -> str:
    return name.replace("_", " ").title()


def power_curve(
    *,
    method_id: str,
    base_inputs: dict[str, Any],
    n_range: tuple[int, int] | None = None,
    n_points: int = 40,
    target_power: float | None = 0.80,
    title: str | None = None,
    out_path: str | Path = "power_curve.png",
) -> Path:
    """Plot achieved power vs N for one method holding effect/SD/α fixed.

    `base_inputs` should be the keyword arguments needed by the method's
    calculator with `n` or `n1` omitted (we'll vary it).  For two-sample
    methods the helper sweeps `n1` and lets the function fill in `n2`
    via `allocation`.
    """
    method = resolve_method(method_id)
    if method is None or not method.get("implemented"):
        raise ValueError(f"method {method_id!r} not implemented")
    fn = method["_callable"]

    # decide which N parameter to sweep
    sweep_key = "n"
    sig_params = method.get("params", {}).get("required", []) + \
                 list(method.get("params", {}).get("optional", {}).keys())
    if "n1" in sig_params:
        sweep_key = "n1"

    base = {k: v for k, v in base_inputs.items() if k not in {"n", "n1", "power"}}

    # auto-range: 5 -> 5 * (solve-for-N at target_power)
    if n_range is None:
        try:
            solved = fn(power=target_power or 0.80, **base)
            top = solved.get(sweep_key) or solved.get("n") or 100
            n_range = (max(4, top // 5), max(20, top * 2))
        except Exception:
            n_range = (10, 300)

    n_lo, n_hi = n_range
    step = max(1, (n_hi - n_lo) // n_points)
    xs: list[int] = list(range(n_lo, n_hi + 1, step))
    ys: list[float] = []
    for n in xs:
        try:
            result = fn(**{sweep_key: n}, **base)
            ys.append(float(result["achieved_power"]))
        except Exception:
            ys.append(float("nan"))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    ax.plot(xs, ys, linewidth=2.0, color="#1f77b4")
    if target_power is not None:
        ax.axhline(target_power, linestyle="--", color="gray", linewidth=1.0,
                   label=f"target power = {target_power:.2f}")
        ax.legend(loc="lower right")
    ax.set_xlabel(f"sample size ({sweep_key})")
    ax.set_ylabel("power")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title(title or f"{_label(method_id)} — power vs N")
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out)
    plt.close(fig)
    return out
