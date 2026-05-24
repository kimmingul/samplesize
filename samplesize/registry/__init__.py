"""Method registry loader.

Reads `methods.json` (a top-level JSON array) and, for implemented
methods, attaches the **actual function signature** as the source of
truth for parameter names and defaults.

Design rule: never declare parameters separately from the callable.
The registry JSON carries categorical metadata (id, category, manual
path, hypothesis flavour, citations, synonyms, `solve_for_options`)
only.  Anything that smells like an API contract — parameter names,
defaults, requiredness — is derived from `inspect.signature(callable)`.
"""
from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent
_METHODS_PATH = _HERE / "methods.json"

_REQUIRED_TOP_KEYS = {
    "id", "name", "category", "manual_path", "callable",
    "implemented", "validated",
}


def _signature_kwargs(fn) -> list[dict[str, Any]]:
    """Return keyword-only parameters of `fn` as registry-friendly dicts."""
    sig = inspect.signature(fn)
    out: list[dict[str, Any]] = []
    for p in sig.parameters.values():
        if p.kind != inspect.Parameter.KEYWORD_ONLY:
            continue
        default = None if p.default is inspect.Parameter.empty else p.default
        out.append({
            "name": p.name,
            "default": default,
            "required": p.default is inspect.Parameter.empty,
        })
    return out


def _resolve_callable(spec: str):
    module_name, attr = spec.rsplit(":", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def load_methods() -> list[dict[str, Any]]:
    """Return the raw registry array (no signature attachment, no I/O cost
    beyond reading the JSON file).
    """
    if not _METHODS_PATH.exists():
        return []
    with _METHODS_PATH.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(
            f"registry methods.json must be a top-level JSON array, got "
            f"{type(data).__name__}"
        )
    return data


def resolve_method(method_id: str) -> dict[str, Any] | None:
    """Look up a method by id and (when implemented) attach the callable
    and its derived signature."""
    for entry in load_methods():
        if entry["id"] != method_id:
            continue
        out = dict(entry)
        if entry.get("implemented"):
            if "callable" not in entry:
                raise ValueError(
                    f"method {method_id!r} is implemented but has no `callable`"
                )
            fn = _resolve_callable(entry["callable"])
            out["_callable"] = fn
            out["signature"] = {"kwargs": _signature_kwargs(fn)}
        return out
    return None


def all_implemented_methods() -> list[dict[str, Any]]:
    """Resolve every implemented method (eager) for batch checks / docs."""
    return [
        resolve_method(e["id"])
        for e in load_methods()
        if e.get("implemented")
    ]
