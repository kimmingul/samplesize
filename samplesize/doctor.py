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
    md_paths: list[Path] = list(_PLUGIN.rglob("*.md"))
    docs_dir = _ROOT / "docs"
    if docs_dir.exists():
        md_paths.extend(docs_dir.rglob("*.md"))
    readme = _ROOT / "README.md"
    if readme.exists():
        md_paths.append(readme)
    for md in sorted(md_paths):
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


def _check_decision_tree_leaves_exist() -> list[str]:
    """Every `leaf:` value in decision_tree.yaml must be a registered method id."""
    import yaml
    from samplesize.registry import load_methods
    errs: list[str] = []
    tree_path = _ROOT / "samplesize/registry/decision_tree.yaml"
    if not tree_path.exists():
        return [f"missing {tree_path.relative_to(_ROOT)}"]
    try:
        doc = yaml.safe_load(tree_path.read_text())
    except yaml.YAMLError as e:
        return [f"{tree_path.relative_to(_ROOT)}: invalid YAML: {e}"]
    known = {e["id"] for e in load_methods()}

    def walk(node, path: str) -> None:
        if not isinstance(node, dict):
            return
        # A node is "terminal" if it has `leaf` OR `unimplemented`.
        # `unimplemented: true` (or a string reason) documents a known
        # catalogue gap — per CLAUDE.md "no fallback heuristics", these
        # are valid terminals (the CLI should error rather than silently
        # downgrade to a different method).
        if "unimplemented" in node:
            return
        if "leaf" in node:
            mid = node["leaf"]
            if mid not in known:
                errs.append(f"decision_tree.yaml at {path}: leaf "
                            f"{mid!r} is not a registered method id")
            return
        opts = node.get("options")
        if isinstance(opts, dict):
            for key, child in opts.items():
                walk(child, f"{path}.{key}")

    if isinstance(doc, dict):
        for top_key, top_node in doc.items():
            walk(top_node, top_key)
    return errs


def _check_skill_kind_choices_exist() -> list[str]:
    """Every `--kind <name>` mentioned in a plugin markdown file must be a
    valid argparse `choices=` value for that subcommand.

    Currently covers `cmd_report`'s `--kind`. Extracts the choices list
    from cli.py's source with a regex so we stay in lockstep with the
    style of `_check_skill_cli_flags_exist`.
    """
    # NOTE: regex assumes `--kind` and `choices=[...]` appear inside the
    # same `add_argument(...)` call; if cli.py refactors to a module-level
    # CHOICES constant, update both this check and the call site together.
    errs: list[str] = []
    cli_path = _ROOT / "samplesize/cli.py"
    if not cli_path.exists():
        return [f"missing {cli_path.relative_to(_ROOT)}"]
    src = cli_path.read_text()

    # Find `--kind` argparse add_argument call and extract its choices=[...].
    kind_block = re.search(
        r'add_argument\(\s*"--kind"[^)]*choices\s*=\s*\[([^\]]*)\]',
        src,
        re.DOTALL,
    )
    if not kind_block:
        return [f"{cli_path.relative_to(_ROOT)}: could not locate --kind "
                f"argparse choices"]
    choices = set(re.findall(r'"([^"]+)"', kind_block.group(1)))
    if not choices:
        return [f"{cli_path.relative_to(_ROOT)}: --kind choices list is empty"]

    pat = re.compile(r"--kind\s+([\w-]+)")
    for md in sorted(_PLUGIN.rglob("*.md")):
        text = md.read_text()
        for m in pat.finditer(text):
            name = m.group(1)
            if name not in choices:
                errs.append(f"{md.relative_to(_ROOT)}: `--kind {name}` is "
                            f"not in argparse choices "
                            f"(have: {sorted(choices)})")
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
    ("registry.decision-tree-leaves-exist",
     _check_decision_tree_leaves_exist),
    ("plugin.manifest", _check_plugin_manifest),
    ("plugin.commands-have-descriptions", _check_commands_have_descriptions),
    ("plugin.skills-have-metadata", _check_skills_have_metadata),
    ("plugin.skill-cli-flags-exist", _check_skill_cli_flags_exist),
    ("plugin.skill-kind-choices-exist", _check_skill_kind_choices_exist),
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
