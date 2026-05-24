#!/usr/bin/env python3
"""Validate the samplesize plugin structure.

Checks that every file Claude Code will load can be parsed:
- plugin.json valid JSON with required keys
- commands/*.md have a description in front-matter
- skills/<name>/SKILL.md has name + description in front-matter
- agents/*.md have name + description in front-matter

Run from the plugin/ directory or pass --root.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FM_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def _front_matter(path: Path) -> dict[str, str]:
    text = path.read_text()
    m = FM_RE.match(text)
    if not m:
        raise ValueError(f"{path}: missing YAML front-matter")
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".",
                    help="path to plugin/ root (default: cwd)")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    errors: list[str] = []

    manifest = root / ".claude-plugin/plugin.json"
    if not manifest.exists():
        errors.append(f"missing manifest: {manifest}")
    else:
        try:
            data = json.loads(manifest.read_text())
            for key in ("name", "version", "description"):
                if key not in data:
                    errors.append(f"plugin.json missing key: {key}")
            print(f"  ok  plugin.json  name={data.get('name')} "
                  f"version={data.get('version')}")
        except json.JSONDecodeError as e:
            errors.append(f"plugin.json: {e}")

    for sub, label in [("commands", "command"),
                       ("agents", "agent")]:
        for md in sorted((root / sub).glob("*.md")):
            try:
                fm = _front_matter(md)
            except ValueError as e:
                errors.append(str(e))
                continue
            if "description" not in fm:
                errors.append(f"{md}: front-matter missing 'description'")
            print(f"  ok  {sub}/{md.name}  description={fm.get('description', '')!r:.80s}")

    skills_dir = root / "skills"
    if skills_dir.exists():
        for sk in sorted(skills_dir.iterdir()):
            sm = sk / "SKILL.md"
            if not sm.exists():
                errors.append(f"{sk}: missing SKILL.md")
                continue
            try:
                fm = _front_matter(sm)
            except ValueError as e:
                errors.append(str(e))
                continue
            for key in ("name", "description"):
                if key not in fm:
                    errors.append(f"{sm}: front-matter missing {key!r}")
            print(f"  ok  skills/{sk.name}  name={fm.get('name')!r}")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
