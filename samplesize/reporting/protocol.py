"""Protocol / grant text generators for a finished sample-size result.

The plugin operates in English by default.  Localised output is
data-driven: a translator drops a `protocol.<lang>.yaml` file under
`samplesize/reporting/templates/` and the renderer picks it up via the
`lang` argument.  No translated strings live in the code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

_TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_LANG = "en"


def _load_audit(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def _load_lang(lang: str) -> dict[str, Any]:
    path = _TEMPLATE_DIR / f"protocol.{lang}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no protocol template for lang={lang!r}; available: "
            f"{available_languages()}"
        )
    with path.open() as f:
        return yaml.safe_load(f)


def available_languages() -> list[str]:
    return sorted(p.stem.split(".", 1)[1] for p in _TEMPLATE_DIR.glob("protocol.*.yaml"))


def _n_phrase(record: dict[str, Any], pack: dict[str, Any]) -> str:
    np = pack["n_phrase"]
    if "n_total" in record:
        return np["with_groups"].format(
            total=record["n_total"],
            per_group=", ".join(str(x) for x in record.get("n_per_group_list", [])),
        )
    if "n1" in record and "n2" in record:
        return np["two_arm"].format(
            total=record["n1"] + record["n2"],
            n1=record["n1"], n2=record["n2"],
        )
    if "n" in record:
        return np["single"].format(n=record["n"])
    return np["fallback"]


def _common_fields(rec: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    res = rec["result"]
    inputs = res.get("inputs_echo", {})
    sides = inputs.get("sides", 2)
    sides_word = pack["words"]["one_sided" if sides == 1 else "two_sided"]
    citations = res.get("citations", [])
    return {
        "method_name": rec.get("method_name", res.get("method_id", "this procedure")),
        "alpha": inputs.get("alpha", "α"),
        "power": float(res.get("achieved_power", inputs.get("power") or 0.0)),
        "target_power": inputs.get("power") or res.get("achieved_power"),
        "sides_word": sides_word,
        "n_phrase": _n_phrase(res, pack),
        "citation": citations[0] if citations else "",
    }


def ich_e9_section(audit_path: str | Path, *, lang: str = DEFAULT_LANG) -> str:
    rec = _load_audit(audit_path)
    pack = _load_lang(lang)
    fields = _common_fields(rec, pack)
    template = pack["templates"]["ich_e9_section"]
    return template.format(**fields).replace("\n", " ").strip()


def grant_aims(audit_path: str | Path, *, lang: str = DEFAULT_LANG) -> str:
    rec = _load_audit(audit_path)
    pack = _load_lang(lang)
    fields = _common_fields(rec, pack)
    template = pack["templates"]["grant_aims"]
    return template.format(**fields).replace("\n", " ").strip()
