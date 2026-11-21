# AGENTS.md

Guidance for coding agents working in this repository.

## Project

`openenv-conformance` probes a running OpenEnv environment server and grades how well it honours the OpenEnv contract. Package: `openenv_conformance/`.

- `probe.py`: the HTTP client, the `Prober` and every check. HTTP checks live in `_http_contract`, session checks in `_ws_contract`.
- `wsproto.py`: minimal RFC 6455 WebSocket client and server helpers. Stdlib only.
- `report.py`: findings, grading, renderers.
- `cli.py`: arguments, `--demo`, exit codes.
- `samples/envs.py`: the `healthy` sample must pass every check; the `broken` sample must trip every rule listed in `tests/test_probe.py::BrokenEnvTests::test_catches_planted_faults`.

## Commands

- Run tests: `python -m unittest discover -s tests -v`
- Try it: `python -m openenv_conformance --demo broken`

## Rules for changes

- You MUST keep runtime dependencies at zero. No `requests`, `websockets` or `httpx`.
- You MUST support Python 3.9.
- HTTP `/reset`, `/step`, `/state` are stateless in `openenv-core`. You MUST NOT add checks that assume state carries across HTTP requests; put episode checks on `/ws`.
- When you add a check, you MUST add it to the README tables and add a test that shows it firing on the broken sample (extend the sample if needed) and passing on the healthy one.
- Every failing check SHOULD carry a concrete `fix` string.
- The probe MUST NOT send destructive or unbounded traffic. Keep the request count small (around 40 today).
- Tests MUST close every socket and server they open.
