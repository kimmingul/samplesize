"""Run every `samplesize doctor` check in CI as well.

Each check is parametrised so pytest reports them individually — if a
single check regresses the failure points at the exact source of drift,
not "doctor failed".
"""
from __future__ import annotations

import pytest

from samplesize.doctor import CHECKS


@pytest.mark.parametrize("name,check", CHECKS, ids=[n for n, _ in CHECKS])
def test_doctor_check(name, check):
    errs = check()
    assert errs == [], f"doctor.{name} reported failures:\n  - " + "\n  - ".join(errs)
