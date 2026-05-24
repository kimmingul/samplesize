"""`samplesize doctor` — integrity checks that catch drift before users hit it.

Each check returns a list of failures.  Doctor exits 0 only when every
check returns an empty list.  Run from CLI:

    python -m samplesize doctor          # human output, exits non-zero on fail
    python -m samplesize doctor --json   # machine-readable output

Add a new check by writing `_check_<name>()` returning `list[str]` and
appending it to `CHECKS`.
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parents[1]
_PLUGIN = _ROOT / "plugin"
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
_REQUIRED_TOP_KEYS = {
    "id", "name", "category", "manual_path", "callable",
    "implemented", "validated",
}


def _frontmatter(path: Path) -> dict[str, str] | None:
    text = path.read_text()
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm


# --- individual checks -------------------------------------------------------


def _check_registry_shape() -> list[str]:
    errs: list[str] = []
    path = _ROOT / "samplesize/registry/methods.json"
    if not path.exists():
        return [f"missing {path}"]
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"{path}: invalid JSON: {e}"]
    if not isinstance(data, list):
        errs.append(f"{path}: must be a top-level JSON array, got "
                    f"{type(data).__name__}")
        return errs
    ids = set()
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            errs.append(f"entry #{i} is not an object")
            continue
        missing = _REQUIRED_TOP_KEYS - set(entry)
        if missing:
            errs.append(f"entry {entry.get('id', f'#{i}')}: missing keys "
                        f"{sorted(missing)}")
        for bool_key in ("implemented", "validated"):
            if bool_key in entry and not isinstance(entry[bool_key], bool):
                errs.append(f"entry {entry.get('id')}: {bool_key!r} must be "
                            f"bool, got {type(entry[bool_key]).__name__}")
        mid = entry.get("id")
        if mid in ids:
            errs.append(f"duplicate id: {mid}")
        ids.add(mid)
    return errs


def _check_callables_importable() -> list[str]:
    from samplesize.registry import load_methods
    errs: list[str] = []
    for entry in load_methods():
        if not entry.get("implemented"):
            continue
        spec = entry.get("callable", "")
        if ":" not in spec:
            errs.append(f"{entry['id']}: callable {spec!r} missing 'module:fn'")
            continue
        mod_name, attr = spec.rsplit(":", 1)
        try:
            mod = __import__(mod_name, fromlist=[attr])
        except ImportError as e:
            errs.append(f"{entry['id']}: cannot import {mod_name}: {e}")
            continue
        fn = getattr(mod, attr, None)
        if fn is None:
            errs.append(f"{entry['id']}: {mod_name} has no {attr!r}")
            continue
        sig = inspect.signature(fn)
        if not any(p.kind == inspect.Parameter.KEYWORD_ONLY
                   for p in sig.parameters.values()):
            errs.append(f"{entry['id']}: callable has no keyword-only "
                        f"parameters; signature: {sig}")
    return errs


def _check_manual_paths() -> list[str]:
    from samplesize.registry import load_methods
    errs: list[str] = []
    for entry in load_methods():
        if not entry.get("implemented"):
            continue
        mp = entry.get("manual_path")
        if not mp:
            continue
        path = _ROOT / mp
        if not path.exists():
            errs.append(f"{entry['id']}: manual_path {mp} does not exist")
    return errs


def _check_plugin_manifest() -> list[str]:
    errs: list[str] = []
    manifest = _PLUGIN / ".claude-plugin/plugin.json"
    if not manifest.exists():
        return [f"missing {manifest}"]
    try:
        data = json.loads(manifest.read_text())
    except json.JSONDecodeError as e:
        return [f"{manifest}: invalid JSON: {e}"]
    for key in ("name", "version", "description"):
        if key not in data:
            errs.append(f"plugin.json missing {key!r}")
    return errs


def _check_commands_have_descriptions() -> list[str]:
    errs: list[str] = []
    for md in sorted((_PLUGIN / "commands").glob("*.md")):
        fm = _frontmatter(md)
        if fm is None:
            errs.append(f"{md.relative_to(_ROOT)}: missing YAML front-matter")
            continue
        if "description" not in fm:
            errs.append(f"{md.relative_to(_ROOT)}: front-matter missing "
                        f"'description'")
    return errs


def _check_skills_have_metadata() -> list[str]:
    errs: list[str] = []
    skills_dir = _PLUGIN / "skills"
    if not skills_dir.exists():
        return []
    for sk in sorted(skills_dir.iterdir()):
        sm = sk / "SKILL.md"
        if not sm.exists():
            errs.append(f"skills/{sk.name}: missing SKILL.md")
            continue
        fm = _frontmatter(sm)
        if fm is None:
            errs.append(f"{sm.relative_to(_ROOT)}: missing YAML front-matter")
            continue
        for key in ("name", "description"):
            if key not in fm:
                errs.append(f"{sm.relative_to(_ROOT)}: front-matter missing "
                            f"{key!r}")
    return errs


def _check_fixtures_reference_real_methods() -> list[str]:
    import yaml
    from samplesize.registry import load_methods
    known = {e["id"] for e in load_methods()}
    errs: list[str] = []
    fixtures_dir = _ROOT / "tests/validation/fixtures"
    for yml in sorted(fixtures_dir.glob("*.yaml")):
        if yml.name.startswith("_"):
            continue
        try:
            doc = yaml.safe_load(yml.read_text())
        except yaml.YAMLError as e:
            errs.append(f"{yml.relative_to(_ROOT)}: invalid YAML: {e}")
            continue
        mid = (doc or {}).get("method")
        if not mid:
            errs.append(f"{yml.relative_to(_ROOT)}: top-level 'method' missing")
        elif mid not in known:
            errs.append(f"{yml.relative_to(_ROOT)}: method {mid!r} not in registry")
    return errs


def _check_skill_cli_flags_exist() -> list[str]:
    """Every `python -m samplesize <sub> --<flag>` mentioned in a SKILL.md
    must exist in the live argparse spec."""
    from samplesize.cli import _build_parser  # type: ignore[attr-defined]
    parser = _build_parser()
    # Map subcommand -> set of valid options.
    sub_options: dict[str, set[str]] = {}
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices and \
                isinstance(action.choices, dict):
            for name, sub_parser in action.choices.items():
                opts = set()
                for a in sub_parser._actions:
                    opts.update(a.option_strings)
                sub_options[name] = opts

    pat = re.compile(r"python -m samplesize (\w+)([^\n`]*)")
    errs: list[str] = []
    for md in sorted((_PLUGIN).rglob("*.md")):
        text = md.read_text()
        for m in pat.finditer(text):
            sub = m.group(1)
            rest = m.group(2)
            if sub not in sub_options:
                errs.append(f"{md.relative_to(_ROOT)}: unknown subcommand "
                            f"{sub!r} (have: {sorted(sub_options)})")
                continue
            for flag in re.findall(r"(--[\w-]+)", rest):
                if flag not in sub_options[sub]:
                    errs.append(f"{md.relative_to(_ROOT)}: `samplesize {sub}` "
                                f"has no flag {flag!r}")
    return errs


def _check_protocol_templates() -> list[str]:
    errs: list[str] = []
    tdir = _ROOT / "samplesize/reporting/templates"
    if not tdir.exists():
        return [f"missing {tdir}"]
    if not (tdir / "protocol.en.yaml").exists():
        errs.append(f"missing default English protocol template at "
                    f"{tdir / 'protocol.en.yaml'}")
    return errs


CHECKS: list[tuple[str, Callable[[], list[str]]]] = [
    ("registry.shape", _check_registry_shape),
    ("registry.callables-importable", _check_callables_importable),
    ("registry.manual-paths", _check_manual_paths),
    ("registry.fixtures-reference-real-methods",
     _check_fixtures_reference_real_methods),
    ("plugin.manifest", _check_plugin_manifest),
    ("plugin.commands-have-descriptions", _check_commands_have_descriptions),
    ("plugin.skills-have-metadata", _check_skills_have_metadata),
    ("plugin.skill-cli-flags-exist", _check_skill_cli_flags_exist),
    ("reporting.protocol-templates", _check_protocol_templates),
]


def run_doctor(json_output: bool = False) -> int:
    results: dict[str, list[str]] = {}
    for name, check in CHECKS:
        try:
            results[name] = check()
        except Exception as e:
            results[name] = [f"check crashed: {type(e).__name__}: {e}"]
    fail_count = sum(1 for errs in results.values() if errs)
    if json_output:
        print(json.dumps({"ok": fail_count == 0, "checks": results}, indent=2))
    else:
        for name, errs in results.items():
            tag = "OK  " if not errs else "FAIL"
            print(f"[{tag}] {name}")
            for e in errs:
                print(f"        - {e}")
        print()
        print(f"{len(results) - fail_count}/{len(results)} checks passed")
    return 0 if fail_count == 0 else 1
