"""Validation-test fixtures loader.

Each YAML fixture under `fixtures/<method_id>.yaml` declares one or more
worked examples with input parameters, expected outputs, and
tolerances. `pytest` parametrises automatically on this collection.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_all_examples():
    examples = []
    for yml in sorted(FIXTURES_DIR.glob("*.yaml")):
        if yml.name.startswith("_"):
            continue  # template / docs, not a real fixture
        with yml.open() as f:
            doc = yaml.safe_load(f)
        if not doc:
            continue
        method = doc["method"]
        for ex in doc.get("examples", []):
            examples.append({
                "method": method,
                "fixture": yml.name,
                **ex,
            })
    return examples


@pytest.fixture(scope="session")
def validation_examples():
    return _load_all_examples()
