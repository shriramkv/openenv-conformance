# openenv-conformance

[![ci](https://github.com/shriramkv/openenv-conformance/actions/workflows/ci.yml/badge.svg)](https://github.com/shriramkv/openenv-conformance/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/dependencies-0-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)

A live conformance probe for **[OpenEnv](https://github.com/meta-pytorch/OpenEnv)** environment servers (the AAIF interface library for RL post-training with environments).

OpenEnv environments are microservices with a Gymnasium-style contract: `reset()`, `step(action)` and `state()`, served over HTTP and over a `/ws` WebSocket session. Trainers such as TRL, torchforge and SkyRL assume that contract holds exactly. When it doesn't, nothing crashes loudly. Training just goes quietly wrong:

- `done` is missing, so the rollout loop never ends or ends at once
- `reward` arrives as the string `"-0.1"`, and advantage estimates are garbage
- `step_count` never moves, so episode-length metrics flatline
- seeds are ignored, so runs are not reproducible
- a malformed action returns **500**, and workers that retry on 5xx hammer the env
- two `/ws` sessions share one environment instance, so parallel rollouts corrupt each other

`openenv-conformance` points at a running environment (local container, Hugging Face Space, cluster) and checks all of that in a couple of seconds. It has **zero dependencies** (stdlib HTTP plus a small RFC 6455 WebSocket client), grades A-F, and gates CI.

## Quick start

```bash
pip install git+https://github.com/shriramkv/openenv-conformance

openenv-conformance --demo healthy      # bundled correct env: grade A, 59/59 checks
openenv-conformance --demo broken       # bundled env with 12 planted faults: grade F

openenv-conformance http://localhost:8000
openenv-conformance https://my-org-my-env.hf.space -H "Authorization: Bearer $HF_TOKEN"
openenv-conformance http://localhost:8000 --action '{"message": "hello"}' --min-grade B
```

No install needed for a quick try: `python -m openenv_conformance --demo broken`.

## What it checks

The probe follows how `openenv-core` actually serves an environment. HTTP `POST /reset`, `POST /step` and `GET /state` are **stateless**: each request runs against a fresh environment instance. Episodes live on the **`/ws` session**, one environment per connection. Checks are split the same way.

**Server and discovery**

| Check | Severity | Expectation |
|---|---|---|
| `health-reachable` / `health-status` | critical / medium | `GET /health` returns 200 `{"status": "healthy"}` |
| `metadata` | low | `GET /metadata` has `name` and `description` |
| `schema` | medium | `GET /schema` has `action`, `observation`, `state` JSON Schemas (the probe builds its default action from it) |
| `json-content-type` | low | responses are `application/json` |
| `simulation-routes-hidden` | high | with `--mode production`, `/reset`, `/step`, `/state` are not served |
| `alive-after-probe` | critical | still healthy after every negative input |

**HTTP request/response contract**

| Check | Severity | Expectation |
|---|---|---|
| `reset-works` / `step-works` | critical | 200 with a JSON object |
| `observation-is-object`, `done-is-bool`, `reward-is-number` | high | `{observation: {...}, reward: number \| null, done: bool}`, rewards finite |
| `reset-not-done` | high | a fresh episode starts with `done=false` |
| `state-works` / `state-shape` | high / medium | `GET /state` has `episode_id` and an integer `step_count` |
| `seed-deterministic` / `seed-accepted` | medium | three resets with the same seed give identical observations |
| `rejects-non-object-action`, `rejects-missing-action`, `rejects-malformed-json` | high on 5xx, medium on 2xx | invalid input gets a 4xx (422), never 500, never silently accepted |
| `step-latency` | medium | step p95 within `--max-p95-ms` (default 1000) |

**`/ws` session contract**

| Check | Severity | Expectation |
|---|---|---|
| `ws-available` | medium | `/ws` accepts a WebSocket session |
| `ws-reset`, `ws-step`, `ws-state` | high | `observation` / `state` reply messages |
| `ws-episode-id`, `ws-step-count-zero`, `ws-step-count-increments` | medium | state tracks the episode: id set, 0 after reset, 1 after one step |
| `episode-terminates` | low | `done=true` within `--max-steps` |
| `ws-step-after-done` | medium | stepping a finished episode doesn't kill the session |
| `ws-episode-id-changes`, `ws-episode-id-honored` | low | fresh id per reset, client-supplied `episode_id` respected |
| `ws-sessions-isolated` | high | a second concurrent session does not disturb the first |
| `ws-concurrent-sessions` | low | a second session can be opened at all |
| `ws-rejects-invalid-action`, `ws-rejects-unknown-type`, `ws-rejects-malformed-json` | high if the session drops, medium otherwise | reply `{type: "error"}` and keep the session open |

Repeats of the same defect (a string reward on every step, say) are collapsed into one finding with a `(seen Nx)` count.

## Validated against real `openenv-core`

Besides the bundled samples, the probe was run against environments served by `openenv-core` 0.3.0 (`create_fastapi_app`):

| Server | Result |
|---|---|
| correct env, default settings | **A (97)**: one low finding, because the default server allows only one `/ws` session |
| correct env, `SUPPORTS_CONCURRENT_SESSIONS = True`, `max_concurrent_envs=4` | **A (100)** |
| same, but the factory returns one shared env instance | **B (80)**: `ws-sessions-isolated` (high) catches the shared state |

## Grading and CI

Each distinct defect costs its severity once (critical 40, high 20, medium 10, low 3). Repeats add 25% each, capped at 2x. Any critical caps the grade at **D**. A is 90+, B 80+, C 70+, D 60+, F below.

```yaml
- name: OpenEnv conformance
  run: |
    docker run -d -p 8000:8000 my-env:latest
    pip install git+https://github.com/shriramkv/openenv-conformance
    openenv-conformance http://localhost:8000 --min-grade B --fail-on high
```

Exit codes: `0` passed, `1` a gate failed, `2` the env was unreachable or the arguments were invalid.

Options: `--action JSON` (default: built from `/schema`), `--mode simulation|production`, `--no-ws` (HTTP-only envs), `--max-steps N`, `--max-p95-ms MS`, `--seed N`, `--timeout S`, `-H "Name: value"` (repeatable), `-f console|json|markdown|badge|badge-endpoint`, `--min-grade`, `--fail-on`.

`-f json` also includes every individual check with pass/fail, handy for dashboards.

## Bundled sample environments

`openenv_conformance/samples/envs.py` holds two stdlib-only "guess the number" envs that mirror `openenv-core`'s serving model (stateless HTTP plus `/ws` sessions). The **broken** one plants: no `/schema`, `reset` without `done`, string rewards, 500s on bad input, ignored seeds, `text/plain` bodies, and on `/ws` a frozen `step_count`, ignored `episode_id`, a shared game across sessions, and a dropped socket on invalid actions.

Run one directly: `python -m openenv_conformance.samples.envs healthy 8000`.

## Where it fits in the AAIF toolchain

The first conformance tool for OpenEnv in the [AAIF](https://aaif.io) ecosystem, alongside the protocol probes for MCP ([mcp-conformance-kit](https://github.com/shriramkv/mcp-conformance-kit)), Open Responses ([open-responses-conformance](https://github.com/shriramkv/open-responses-conformance)) and A2A ([a2a-conformance-probe](https://github.com/shriramkv/a2a-conformance-probe)). See [aaif-agentic-toolkit](https://github.com/shriramkv/aaif-agentic-toolkit) for the full set.

## Development

```bash
python -m unittest discover -s tests -v
```

See [AGENTS.md](AGENTS.md) for conventions when contributing with a coding agent.

## License

MIT
