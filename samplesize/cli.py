"""Command-line entry: `python -m samplesize <subcommand> ...`.

Currently a stub; subcommands wired in as Tier-1 methods land.
"""
import argparse
import json
import sys


def cmd_list(_args: argparse.Namespace) -> int:
    """List registered methods."""
    from samplesize.registry import load_methods
    methods = load_methods()
    for m in methods:
        flag = "✓" if m.get("implemented") else " "
        print(f"  [{flag}] {m['id']:<40s} {m['name']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Print a method's registry entry + actual callable signature."""
    import inspect
    from samplesize.registry import resolve_method
    method = resolve_method(args.method)
    if method is None:
        print(f"unknown method: {args.method}", file=sys.stderr)
        return 2
    out = {k: v for k, v in method.items() if k != "_callable"}
    if method.get("implemented") and "_callable" in method:
        sig = inspect.signature(method["_callable"])
        out["signature"] = {
            "kwargs": [
                {
                    "name": p.name,
                    "default": (None if p.default is inspect.Parameter.empty
                                else p.default),
                    "required": p.default is inspect.Parameter.empty,
                }
                for p in sig.parameters.values()
                if p.kind == inspect.Parameter.KEYWORD_ONLY
            ],
        }
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_calc(args: argparse.Namespace) -> int:
    """Run a calculation for the given method id."""
    from samplesize.registry import resolve_method
    from samplesize.reporting.audit import write_audit

    method = resolve_method(args.method)
    if method is None:
        print(f"unknown method: {args.method}", file=sys.stderr)
        return 2
    if not method.get("implemented"):
        print(f"method '{args.method}' is registered but not yet implemented",
              file=sys.stderr)
        return 3
    kwargs = json.loads(args.json_args) if args.json_args else {}
    fn = method["_callable"]
    result = fn(**kwargs)

    audit_path = write_audit({
        "method_id": args.method,
        "method_name": method["name"],
        "chapter": method.get("manual_path"),
        "result": result,
    })
    result["_audit_path"] = str(audit_path)
    result["_repro_snippet"] = _build_snippet(args.method, kwargs)

    print(json.dumps(result, indent=2, default=str))
    return 0


def _build_snippet(method_id: str, kwargs: dict) -> str:
    """Standalone Python that reproduces this calculation."""
    return (
        "from samplesize.registry import resolve_method\n"
        f"fn = resolve_method({method_id!r})['_callable']\n"
        f"print(fn(**{kwargs!r}))\n"
    )


def cmd_report(args: argparse.Namespace) -> int:
    """Generate downstream artefacts (curves, tables, write-ups)."""
    import json as _json
    from pathlib import Path

    audit_path = Path(args.audit_json)
    if not audit_path.exists():
        print(f"audit file not found: {audit_path}", file=sys.stderr)
        return 2

    rec = _json.loads(audit_path.read_text())
    method_id = rec.get("method_id") or rec["result"]["method_id"]
    inputs = dict(rec["result"]["inputs_echo"])

    if args.kind == "power-curve":
        from samplesize.reporting.plots import power_curve
        # base for the curve: drop n / n1 / power so the plotter can sweep N
        base = {k: v for k, v in inputs.items() if v is not None}
        out = power_curve(
            method_id=method_id,
            base_inputs=base,
            target_power=inputs.get("power") or 0.80,
            out_path=args.out or "power_curve.png",
        )
        print(f"wrote {out}")
        return 0

    if args.kind == "protocol":
        from samplesize.reporting.protocol import ich_e9_section
        text = ich_e9_section(audit_path, lang=args.lang)
        if args.out:
            Path(args.out).write_text(text + "\n")
            print(f"wrote {args.out}")
        else:
            print(text)
        return 0

    if args.kind == "grant":
        from samplesize.reporting.protocol import grant_aims
        text = grant_aims(audit_path)
        if args.out:
            Path(args.out).write_text(text + "\n")
            print(f"wrote {args.out}")
        else:
            print(text)
        return 0

    if args.kind == "r-code":
        from samplesize.reporting.code_export import r_code
        text = r_code(audit_path)
        if args.out:
            Path(args.out).write_text(text)
            print(f"wrote {args.out}")
        else:
            print(text, end="")
        return 0

    if args.kind == "sas-code":
        from samplesize.reporting.code_export import sas_code
        text = sas_code(audit_path)
        if args.out:
            Path(args.out).write_text(text)
            print(f"wrote {args.out}")
        else:
            print(text, end="")
        return 0

    if args.kind == "sensitivity":
        from samplesize.reporting.tables import sensitivity_grid
        if not args.vary:
            print("--vary required for --kind sensitivity "
                  "(e.g. --vary 'sd=15,20,25')", file=sys.stderr)
            return 2
        sweeps = [_parse_vary(s) for s in args.vary]
        if len(sweeps) > 2:
            print("--vary accepts at most 2 sweep dimensions", file=sys.stderr)
            return 2
        base = {k: v for k, v in inputs.items() if v is not None}
        for key, _ in sweeps:
            base.pop(key, None)
        if len(sweeps) == 1:
            (row_key, row_vals) = sweeps[0]
            text = sensitivity_grid(
                method_id=method_id, base_inputs=base,
                row_param=row_key, row_values=row_vals,
                report_key=args.report_key,
            )
        else:
            (row_key, row_vals), (col_key, col_vals) = sweeps
            text = sensitivity_grid(
                method_id=method_id, base_inputs=base,
                row_param=row_key, row_values=row_vals,
                col_param=col_key, col_values=col_vals,
                report_key=args.report_key,
            )
        if args.out:
            Path(args.out).write_text(text)
            print(f"wrote {args.out}")
        else:
            print(text, end="")
        return 0

    print(f"unknown report kind: {args.kind}", file=sys.stderr)
    return 2


def _coerce_scalar(raw: str):
    """Best-effort int/float/string parsing of a sweep value."""
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_vary(spec: str) -> tuple[str, list]:
    """Parse 'key=v1,v2,v3' into ('key', [v1, v2, v3]) with smart typing."""
    if "=" not in spec:
        raise ValueError(f"--vary spec must be 'key=v1,v2,...': got {spec!r}")
    key, _, vals = spec.partition("=")
    return key.strip(), [_coerce_scalar(v) for v in vals.split(",")]


def cmd_doctor(args: argparse.Namespace) -> int:
    from samplesize.doctor import run_doctor
    return run_doctor(json_output=args.json)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="samplesize")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub_list = sub.add_parser("list", help="list registered methods")
    sub_list.set_defaults(func=cmd_list)

    sub_show = sub.add_parser("show",
                              help="show a method's registry entry and actual kwargs")
    sub_show.add_argument("method", help="method id (see `samplesize list`)")
    sub_show.set_defaults(func=cmd_show)

    sub_calc = sub.add_parser("calc", help="run a calculation")
    sub_calc.add_argument("method", help="method id (see `samplesize list`)")
    sub_calc.add_argument("--json-args", default="{}",
                          help="JSON-encoded parameter dict")
    sub_calc.set_defaults(func=cmd_calc)

    sub_report = sub.add_parser("report", help="generate report artefacts")
    sub_report.add_argument("audit_json", help="audit JSON from a prior calc")
    sub_report.add_argument("--kind", required=True,
                            choices=["power-curve", "protocol", "grant",
                                     "sensitivity", "r-code", "sas-code"],
                            help="artefact to produce")
    sub_report.add_argument("--out", help="output file path")
    sub_report.add_argument("--vary", action="append",
                            help="(sensitivity only) sweep spec "
                                 "'key=v1,v2,v3'; pass twice for a 2D grid")
    sub_report.add_argument("--report-key", default="n",
                            help="(sensitivity only) result field to tabulate "
                                 "(n, achieved_power, n1, n2, ...)")
    from samplesize.reporting.protocol import available_languages
    sub_report.add_argument("--lang", default="en",
                            choices=available_languages(),
                            help="language for protocol/grant text")
    sub_report.set_defaults(func=cmd_report)

    sub_doctor = sub.add_parser("doctor",
                                help="run integrity checks on the plugin")
    sub_doctor.add_argument("--json", action="store_true",
                            help="machine-readable JSON output")
    sub_doctor.set_defaults(func=cmd_doctor)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
