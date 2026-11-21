"""Command-line entry point for openenv-conformance."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .probe import parse_headers, probe
from .report import Report, grade_at_least, render, severity_at_least


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="openenv-conformance",
        description="Live conformance probe for OpenEnv environment servers (reset/step/state over HTTP).",
    )
    ap.add_argument("url", nargs="?", help="base URL of the environment, e.g. http://localhost:8000")
    ap.add_argument("--demo", choices=["healthy", "broken"], help="start a bundled sample env and probe it")
    ap.add_argument("--action", help="JSON action to send to /step (default: built from /schema)")
    ap.add_argument("--mode", choices=["simulation", "production"], default="simulation",
                    help="production: only /health, /schema, /metadata should be served")
    ap.add_argument("--max-steps", type=int, default=50, help="max steps while running one episode to done (default 50)")
    ap.add_argument("--max-p95-ms", type=float, default=1000.0, help="step p95 latency budget in ms (default 1000)")
    ap.add_argument("--seed", type=int, default=1234, help="seed used for the determinism check")
    ap.add_argument("--no-ws", action="store_true", help="skip the /ws session checks (HTTP-only environments)")
    ap.add_argument("--timeout", type=float, default=10.0, help="per-request timeout in seconds")
    ap.add_argument("-H", "--header", action="append", default=[], help="extra header, e.g. 'Authorization: Bearer $HF_TOKEN'")
    ap.add_argument("-f", "--format", default="console", choices=["console", "json", "markdown", "badge", "badge-endpoint"])
    ap.add_argument("--min-grade", metavar="GRADE", help="exit 1 if the grade is worse than this (A-F)")
    ap.add_argument("--fail-on", choices=["critical", "high", "medium", "low"], help="exit 1 on any finding at or above this severity")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    server = None
    if args.demo:
        from .samples.envs import serve

        server = serve(args.demo)
        url = "http://%s:%d" % server.server_address[:2]
    elif args.url:
        url = args.url
    else:
        ap.print_usage(sys.stderr)
        print("error: give an environment URL or --demo healthy|broken", file=sys.stderr)
        return 2
    try:
        action = json.loads(args.action) if args.action else None
        if args.demo and action is None:
            action = {"guess": 10}  # the broken sample hides /schema, so the demo supplies the action
        headers = parse_headers(args.header)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        res, stats = probe(url, action=action, mode=args.mode, max_steps=args.max_steps,
                           max_p95_ms=args.max_p95_ms, seed=args.seed, timeout=args.timeout, headers=headers,
                           ws=not args.no_ws)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
    target = f"demo:{args.demo}" if args.demo else url
    report = Report("openenv-conformance", target, res.findings, stats)
    if args.format == "json":
        data = report.to_dict()
        data["checks"] = [{"id": c.id, "passed": c.passed, "detail": c.detail} for c in res.checks]
        print(json.dumps(data, indent=2))
    else:
        print(render(report, args.format))
    if any(f.rule == "health-reachable" for f in res.findings):
        return 2
    rc = 0
    if args.min_grade and not grade_at_least(report.grade, args.min_grade):
        rc = 1
    if args.fail_on and any(severity_at_least(f.severity, args.fail_on) for f in res.findings):
        rc = 1
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
