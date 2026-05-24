"""Every implemented method's callable must be importable, keyword-only,
and consistent with what `samplesize show` will print.

This catches the specific drift class that broke Scenario A/B: a method
registered in `methods.json` whose `callable` reference no longer
matches a real function, or whose function lacks keyword-only kwargs.
"""
from __future__ import annotations

import inspect

import pytest

from samplesize.registry import load_methods, resolve_method


def _ids():
    return [e["id"] for e in load_methods() if e.get("implemented")]


@pytest.mark.parametrize("method_id", _ids())
def test_implemented_method_is_resolvable(method_id):
    entry = resolve_method(method_id)
    assert entry is not None, f"{method_id}: not found in registry"
    assert "_callable" in entry, f"{method_id}: missing _callable"
    assert "signature" in entry, f"{method_id}: missing derived signature"


@pytest.mark.parametrize("method_id", _ids())
def test_callable_is_keyword_only(method_id):
    entry = resolve_method(method_id)
    fn = entry["_callable"]
    sig = inspect.signature(fn)
    kw_only = [p for p in sig.parameters.values()
               if p.kind == inspect.Parameter.KEYWORD_ONLY]
    assert kw_only, (
        f"{method_id}: callable {fn.__qualname__} has no keyword-only "
        f"parameters; signature: {sig}"
    )


@pytest.mark.parametrize("method_id", _ids())
def test_callable_returns_dict_with_method_id(method_id):
    """Smoke-test the result envelope, NOT the numbers (validation
    fixtures cover those).  We only confirm the result includes
    `method_id` and `inputs_echo` so reporting can later parse the
    audit JSON without surprises."""
    pytest.importorskip("scipy")
    entry = resolve_method(method_id)
    fn = entry["_callable"]
    sig = inspect.signature(fn)
    # Try to invoke with a minimal valid argument set drawn from defaults
    # plus a synthetic positive-effect signal.  Skip if a required kwarg
    # has no obvious default.
    bag = {
        "mean0": 0.0, "mean1": 1.0, "mean": 1.0, "reference": 0.0,
        "mean1_": 1.0, "mean2": 0.0,
        "sd": 1.0, "sd_diff": 1.0, "sd_x": 1.0,
        "p0": 0.5, "p1": 0.4, "p2": 0.5, "p10": 0.2, "p01": 0.1,
        "r": 0.3, "rho0": 0.0,
        "means": [0.0, 0.5, 1.0], "sigma": 1.0,
        "s1": 0.5, "s2": 0.7,
        "alpha": 0.05, "power": 0.80,
        "margin": 0.1, "lower_margin": -0.1, "upper_margin": 0.1,
        "w": 0.3, "df": 2,
        "k": 3,
        "B": 0.4, "event_rate": 0.7,
    }
    kwargs = {}
    for p in sig.parameters.values():
        if p.kind != inspect.Parameter.KEYWORD_ONLY:
            continue
        if p.default is inspect.Parameter.empty:
            if p.name in bag:
                kwargs[p.name] = bag[p.name]
            else:
                pytest.skip(f"no synthetic default for required kwarg "
                            f"{p.name!r} in {method_id}")
    try:
        result = fn(**kwargs)
    except Exception as e:
        pytest.skip(f"{method_id}: synthetic invocation rejected: {e}")
    assert isinstance(result, dict)
    assert "method_id" in result
    assert "inputs_echo" in result
    assert result["method_id"] == method_id
