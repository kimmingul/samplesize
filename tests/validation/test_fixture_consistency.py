"""Numerical agreement of `samplesize` implementations with reference fixtures.

Skipped automatically for any method whose registry entry has
`implemented: false`. Failures here block promoting a method to
`validated: true`.
"""
from __future__ import annotations

import math

import pytest

from samplesize.registry import load_methods, resolve_method


def _within(actual, expected, tol):
    if isinstance(expected, int) and isinstance(tol, int) and tol == 0:
        return actual == expected
    return math.isclose(actual, expected, abs_tol=tol, rel_tol=0.0)


def test_fixture_example(example):
    method = resolve_method(example["method"])
    assert method is not None, f"method {example['method']} not in registry"
    if not method.get("implemented"):
        pytest.skip(f"{example['method']} not implemented yet")

    fn = method["_callable"]
    result = fn(**example["inputs"])
    for key, expected_val in example["expected"].items():
        tol = example.get("tolerance", {}).get(key, 0.001)
        actual_val = result[key]
        assert _within(actual_val, expected_val, tol), (
            f"{example['method']}::{example['id']} key={key!r}: "
            f"expected {expected_val}, got {actual_val} (tol={tol})"
        )


def pytest_generate_tests(metafunc):
    if "example" not in metafunc.fixturenames:
        return
    from tests.validation.conftest import _load_all_examples
    examples = _load_all_examples()
    metafunc.parametrize(
        "example",
        examples,
        ids=[f"{ex['method']}::{ex['id']}" for ex in examples],
    )
