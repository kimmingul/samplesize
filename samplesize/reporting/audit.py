"""Audit-log writer.

Every calculation emits a JSON record with inputs, outputs, library
versions, environment, and a citation to the methodology. The file is
written to a configurable cache dir (default: `.samplesize/audit/`).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_AUDIT_DIR = Path(
    os.environ.get("SAMPLESIZE_AUDIT_DIR", ".samplesize/audit")
)


def write_audit(record: dict[str, Any], audit_dir: Path | None = None) -> Path:
    """Write `record` to a timestamped JSON file. Returns the path."""
    audit_dir = audit_dir or DEFAULT_AUDIT_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    method = record.get("method_id", "unknown")
    path = audit_dir / f"{ts}_{method}.json"
    enriched = {
        **record,
        "_meta": {
            "timestamp": ts,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    with path.open("w") as f:
        json.dump(enriched, f, indent=2, default=str)
    return path
