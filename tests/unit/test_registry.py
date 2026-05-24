"""Smoke tests for the method registry."""
from samplesize.registry import load_methods, resolve_method


def test_registry_nonempty():
    methods = load_methods()
    assert len(methods) >= 8, "MVP Tier-1 should declare at least 8 methods"


def test_unique_ids():
    methods = load_methods()
    ids = [m["id"] for m in methods]
    assert len(ids) == len(set(ids)), "duplicate method ids in registry"


def test_resolve_unknown_returns_none():
    assert resolve_method("definitely_not_a_method") is None
