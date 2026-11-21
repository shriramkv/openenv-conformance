"""Live conformance probe for OpenEnv environment servers over HTTP.

OpenEnv environments expose a Gymnasium-style contract over HTTP:
``POST /reset`` and ``POST /step`` return ``{observation, reward, done}``,
``GET /state`` returns ``{episode_id, step_count, ...}``, and ``/health``,
``/schema`` and ``/metadata`` describe the server. Training frameworks rely
on those details holding exactly: a ``done`` that is missing, a reward sent as
a string, a step counter that never moves, or a 500 on a malformed action
each turn into silent training bugs. This module checks them against a
running server.
"""
from __future__ import annotations

import json
import math
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .report import Finding
from .wsproto import WSClient


@dataclass
class Resp:
    status: int
    body: Any
    raw: bytes
    ctype: str
    ms: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300

    @property
    def is_json_obj(self) -> bool:
        return isinstance(self.body, dict)


@dataclass
class Check:
    id: str
    passed: bool
    detail: str = ""


@dataclass
class ProbeResult:
    findings: List[Finding] = field(default_factory=list)
    checks: List[Check] = field(default_factory=list)
    latencies: Dict[str, List[float]] = field(default_factory=dict)

    def stats(self) -> Dict[str, Any]:
        all_ms = sorted(x for v in self.latencies.values() for x in v)
        step = sorted(self.latencies.get("step", []))
        out: Dict[str, Any] = {
            "checks": len(self.checks),
            "passed": sum(c.passed for c in self.checks),
            "requests": len(all_ms),
        }
        ws = sorted(self.latencies.get("ws", []))
        if ws:
            out["ws_p95_ms"] = round(_pct(ws, 95), 1)
        if step:
            out["step_p50_ms"] = round(statistics.median(step), 1)
            out["step_p95_ms"] = round(_pct(step, 95), 1)
        return out


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    k = max(0, min(len(xs) - 1, int(math.ceil(p / 100 * len(xs))) - 1))
    return xs[k]


class Client:
    def __init__(self, base: str, timeout: float = 10.0, headers: Optional[Dict[str, str]] = None):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}

    def call(self, method: str, path: str, body: Any = None, raw: Optional[bytes] = None) -> Resp:
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        for k, v in self.headers.items():
            req.add_header(k, v)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload, status, ctype = r.read(), r.status, r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            payload, status, ctype = e.read(), e.code, e.headers.get("Content-Type", "") if e.headers else ""
            e.close()
        except Exception as e:  # connection refused, timeout, reset ...
            return Resp(0, None, b"", "", (time.perf_counter() - t0) * 1000, f"{type(e).__name__}: {e}")
        ms = (time.perf_counter() - t0) * 1000
        try:
            parsed = json.loads(payload) if payload else None
        except ValueError:
            parsed = None
        return Resp(status, parsed, payload, ctype, ms)


def example_from_schema(schema: Any, depth: int = 0) -> Any:
    """Build a minimal valid instance from a JSON Schema (enough to drive /step)."""
    if not isinstance(schema, dict) or depth > 6:
        return {}
    if "default" in schema:
        return schema["default"]
    if "examples" in schema and isinstance(schema["examples"], list) and schema["examples"]:
        return schema["examples"][0]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list) and schema[key]:
            return example_from_schema(schema[key][0], depth + 1)
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0] if t else None)
    if t == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        required = schema.get("required") or list(props)
        return {k: example_from_schema(props.get(k, {}), depth + 1) for k in required}
    if t == "array":
        return []
    if t == "string":
        return "probe"
    if t == "integer":
        lo = schema.get("minimum", 0)
        return int(lo) if isinstance(lo, (int, float)) else 0
    if t == "number":
        lo = schema.get("minimum", 0.0)
        return float(lo) if isinstance(lo, (int, float)) else 0.0
    if t == "boolean":
        return False
    return {}


def _is_reward(v: Any) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v))


class Prober:
    def __init__(self, client: Client, action: Optional[dict] = None, mode: str = "simulation",
                 max_steps: int = 50, max_p95_ms: float = 1000.0, seed: int = 1234, ws: bool = True):
        self.c = client
        self.action = action
        self.mode = mode
        self.max_steps = max_steps
        self.max_p95_ms = max_p95_ms
        self.seed = seed
        self.ws = ws
        self.r = ProbeResult()

    # -- helpers -----------------------------------------------------------
    def _req(self, bucket: str, method: str, path: str, body: Any = None, raw: Optional[bytes] = None) -> Resp:
        resp = self.c.call(method, path, body, raw)
        if resp.error is None:
            self.r.latencies.setdefault(bucket, []).append(resp.ms)
        return resp

    def check(self, cid: str, passed: bool, sev: str, msg: str, fix: str = "", where: str = "") -> bool:
        self.r.checks.append(Check(cid, passed, "" if passed else msg))
        if not passed:
            prior = next((f for f in self.r.findings if f.rule == cid), None)
            if prior is None:
                self.r.findings.append(Finding(cid, sev, msg, where, fix))
            else:  # same defect seen again (e.g. on every step): count it once
                n = int(prior.message.rsplit("(seen ", 1)[1].split("x")[0]) + 1 if "(seen " in prior.message else 2
                prior.message = prior.message.split(" (seen ")[0] + f" (seen {n}x)"
        return passed

    def _shape(self, resp: Resp, where: str, critical_id: str) -> bool:
        if not self.check(critical_id, resp.ok and resp.is_json_obj, "critical",
                          f"{where} returned {_describe(resp)}; expected 200 with a JSON object.",
                          f"Make {where} return 200 and a JSON body.", where):
            return False
        b = resp.body
        self.check("observation-is-object", isinstance(b.get("observation"), dict), "high",
                         f"{where}: 'observation' must be a JSON object, got {type(b.get('observation')).__name__}.",
                         "Return the Observation model serialised as an object under 'observation'.", where)
        self.check("done-is-bool", isinstance(b.get("done"), bool), "high",
                         f"{where}: 'done' must be a boolean, got {_t(b.get('done'))}.",
                         "Always include done: true|false; clients treat a missing done as an error or as False forever.", where)
        self.check("reward-is-number", _is_reward(b.get("reward")), "high",
                         f"{where}: 'reward' must be a finite number or null, got {_t(b.get('reward'))} {b.get('reward')!r}.",
                         "Send reward as a JSON number (or null); string or NaN rewards corrupt advantage estimates.", where)
        return True  # reachable and parseable; shape problems are recorded but do not stop the probe

    # -- checks --------------------------------------------------------------
    def run(self) -> ProbeResult:
        h = self._req("meta", "GET", "/health")
        if not self.check("health-reachable", h.ok, "critical", f"/health returned {_describe(h)}.",
                          "Check the URL and that the container is up; /health must return 200.", "/health"):
            return self.r
        self.check("health-status", h.is_json_obj and h.body.get("status") == "healthy", "medium",
                   f"/health body should be {{\"status\": \"healthy\"}}, got {h.raw[:80]!r}.",
                   "Return the HealthResponse model.", "/health")
        self.check("json-content-type", "json" in h.ctype.lower(), "low",
                   f"Responses use Content-Type {h.ctype or '(none)'!r} instead of application/json.",
                   "Set Content-Type: application/json so strict clients parse the body.", "/health")

        m = self._req("meta", "GET", "/metadata")
        self.check("metadata", m.ok and m.is_json_obj and bool(m.body.get("name")) and "description" in m.body, "low",
                   f"/metadata returned {_describe(m)}; expected name and description.",
                   "Expose EnvironmentMetadata (name, description, version).", "/metadata")

        s = self._req("meta", "GET", "/schema")
        schema_ok = self.check("schema", s.ok and s.is_json_obj and all(isinstance(s.body.get(k), dict) for k in ("action", "observation", "state")),
                               "medium", f"/schema returned {_describe(s)}; expected action, observation and state JSON Schemas.",
                               "Expose /schema so agents and trainers can discover the action format.", "/schema")

        if self.mode == "production":
            for path, method in (("/reset", "POST"), ("/step", "POST"), ("/state", "GET")):
                r = self._req("meta", method, path, {} if method == "POST" else None)
                self.check("simulation-routes-hidden", r.status in (404, 405), "high",
                           f"{method} {path} answered {r.status} in production mode; simulation control should not be exposed.",
                           "Register routes with ServerMode.PRODUCTION so only /health, /schema, /metadata and /ws are served.", path)
            return self._liveness_and_latency()

        action = self.action
        if action is None:
            action = example_from_schema(s.body.get("action")) if schema_ok else {}
        self.action = action
        self.r.checks.append(Check("action-used", True, json.dumps(action)))

        self._http_contract(action)
        if self.ws:
            self._ws_contract(action)
        return self._liveness_and_latency()

    # -- HTTP: stateless request/response contract ------------------------
    def _http_contract(self, action: dict) -> None:
        rs = self._req("reset", "POST", "/reset", {})
        if self._shape(rs, "POST /reset", "reset-works"):
            self.check("reset-not-done", rs.body.get("done") is not True, "high",
                       "POST /reset returned done=true; a fresh episode must start not-done.", "Return done=false from reset.", "POST /reset")
        st = self._req("state", "GET", "/state")
        if self.check("state-works", st.ok and st.is_json_obj, "high", f"GET /state returned {_describe(st)}.",
                      "Expose GET /state returning the State model.", "GET /state"):
            sc = st.body.get("step_count")
            self.check("state-shape", isinstance(sc, int) and not isinstance(sc, bool) and sc >= 0 and "episode_id" in st.body, "medium",
                       f"GET /state should carry episode_id and a non-negative integer step_count, got {json.dumps(st.body)[:120]}.",
                       "Return the State model (episode_id, step_count, plus your own fields).", "GET /state")

        sp = self._req("step", "POST", "/step", {"action": action})
        if not self._shape(sp, "POST /step", "step-works") and sp.status in (400, 409, 422):
            self.r.findings[-1].fix = ("The probe sent action=" + json.dumps(action)
                                       + ". If that is not a valid action for this env, pass one with --action '{...}'.")
        for _ in range(4):  # a few more samples for the latency check
            self._req("step", "POST", "/step", {"action": action})

        obs = []
        for _ in range(3):
            r = self._req("reset", "POST", "/reset", {"seed": self.seed})
            if not (r.ok and r.is_json_obj):
                self.check("seed-accepted", False, "medium", f"reset(seed={self.seed}) returned {_describe(r)}.",
                           "Accept an optional non-negative integer seed in ResetRequest.", "POST /reset")
                break
            obs.append(json.dumps(r.body.get("observation"), sort_keys=True))
        else:
            self.check("seed-deterministic", len(set(obs)) == 1, "medium",
                       f"Three resets with seed={self.seed} produced {len(set(obs))} different initial observations.",
                       "Seed every RNG from ResetRequest.seed so runs are reproducible.", "POST /reset")

        for cid, body, raw, label in (
            ("rejects-non-object-action", {"action": "not-an-object"}, None, "a string action"),
            ("rejects-missing-action", {}, None, "a body without 'action'"),
            ("rejects-malformed-json", None, b"{not json", "malformed JSON"),
        ):
            bad = self._req("neg", "POST", "/step", body, raw)
            if bad.error or bad.status >= 500:
                self.check(cid, False, "high", f"POST /step with {label} returned {_describe(bad)} (server error).",
                           "Validate the request and return 422; a 5xx on bad input crashes RL workers that retry on 5xx.", "POST /step")
            elif 200 <= bad.status < 300:
                self.check(cid, False, "medium", f"POST /step with {label} was accepted with {bad.status}.",
                           "Reject invalid actions with 422 instead of stepping the env with garbage.", "POST /step")
            else:
                self.check(cid, True, "low", "")

    # -- /ws: stateful session contract -----------------------------------
    def _ws_open(self):
        url = self.c.base.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
        return WSClient(url, timeout=self.c.timeout, headers=self.c.headers)

    def _ws_reopen(self, old):
        try:
            old.close(send=False)
        except Exception:
            pass
        return self._ws_open()

    def _ws_call(self, ws, msg: Any, raw: Optional[str] = None) -> Optional[dict]:
        t0 = time.perf_counter()
        ws.send_text(raw if raw is not None else json.dumps(msg))
        text = ws.recv_text()
        self.r.latencies.setdefault("ws", []).append((time.perf_counter() - t0) * 1000)
        if text is None:
            return None
        try:
            data = json.loads(text)
        except ValueError:
            return {"type": "<non-json>", "data": {"raw": text[:80]}}
        return data if isinstance(data, dict) else {"type": "<non-object>", "data": {}}

    def _ws_contract(self, action: dict) -> None:
        W = "/ws"
        try:
            ws = self._ws_open()
        except Exception as exc:
            self.check("ws-available", False, "medium",
                       f"Could not open a WebSocket session at /ws ({exc}). EnvClient uses /ws for stateful episodes; "
                       "over plain HTTP every request hits a fresh environment.",
                       "Serve the app with create_app/create_fastapi_app (which mounts /ws), or pass --no-ws for HTTP-only envs.", W)
            return
        self.r.checks.append(Check("ws-available", True))
        try:
            obs = self._ws_call(ws, {"type": "reset", "data": {}})
            if not self.check("ws-reset", bool(obs) and obs.get("type") == "observation" and isinstance(obs.get("data"), dict),
                              "high", f"/ws reset answered {_ws_desc(obs)}; expected an observation message.",
                              "Reply to {type: reset} with {type: observation, data: {observation, reward, done}}.", W):
                return
            self._ws_shape(obs["data"], "ws reset")
            st = self._ws_call(ws, {"type": "state"})
            ep1 = None
            if self.check("ws-state", bool(st) and st.get("type") == "state" and isinstance(st.get("data"), dict), "high",
                          f"/ws state answered {_ws_desc(st)}; expected a state message.",
                          "Reply to {type: state} with {type: state, data: {episode_id, step_count}}.", W):
                ep1 = st["data"].get("episode_id")
                self.check("ws-episode-id", isinstance(ep1, str) and ep1 != "", "medium",
                           f"state.episode_id should be a non-empty string after reset, got {ep1!r}.",
                           "Assign an episode_id on reset; trainers use it to group trajectories.", W)
                self.check("ws-step-count-zero", st["data"].get("step_count") == 0, "medium",
                           f"state.step_count should be 0 after reset, got {st['data'].get('step_count')!r}.",
                           "Reset step_count to 0 in reset().", W)

            res = self._ws_call(ws, {"type": "step", "data": action})
            if not self.check("ws-step", bool(res) and res.get("type") == "observation" and isinstance(res.get("data"), dict), "high",
                              f"/ws step with action {json.dumps(action)} answered {_ws_desc(res)}.",
                              "Reply to {type: step, data: <action>} with an observation message; pass a valid --action if needed.", W):
                return
            self._ws_shape(res["data"], "ws step")
            st = self._ws_call(ws, {"type": "state"})
            if st and isinstance(st.get("data"), dict):
                self.check("ws-step-count-increments", st["data"].get("step_count") == 1, "medium",
                           f"state.step_count should be 1 after one step, got {st['data'].get('step_count')!r}.",
                           "Increment step_count in step(); episode-length metrics depend on it.", W)

            done, n = bool(res["data"].get("done")), 1
            while not done and n < self.max_steps:
                nxt = self._ws_call(ws, {"type": "step", "data": action})
                if not nxt or nxt.get("type") != "observation":
                    self.check("ws-episode-runs", False, "high", f"step {n + 1} over /ws answered {_ws_desc(nxt)}.",
                               "Steps after the first must keep working until done.", W)
                    return
                self._ws_shape(nxt["data"], f"ws step #{n + 1}")
                done, n = bool(nxt["data"].get("done")), n + 1
            self.check("episode-terminates", done, "low",
                       f"Episode did not reach done=true within {self.max_steps} steps using action {json.dumps(action)}.",
                       "Add a step limit (truncation) so rollouts always end; raise --max-steps if episodes are long by design.", W)
            if done:
                after = self._ws_call(ws, {"type": "step", "data": action})
                self.check("ws-step-after-done", after is not None, "medium",
                           "Stepping a finished episode closed the WebSocket session.",
                           "Answer with an error message (or done=true) and keep the session open.", W)
                if after is None:
                    ws = self._ws_reopen(ws)

            self._ws_call(ws, {"type": "reset", "data": {}})
            st = self._ws_call(ws, {"type": "state"})
            if ep1 and st and isinstance(st.get("data"), dict):
                self.check("ws-episode-id-changes", st["data"].get("episode_id") != ep1, "low",
                           "reset() reused the previous episode_id; trajectories from different episodes become indistinguishable.",
                           "Generate a fresh episode_id per reset.", W)
            self._ws_call(ws, {"type": "reset", "data": {"episode_id": "openenv-conformance-probe"}})
            st = self._ws_call(ws, {"type": "state"})
            if st and isinstance(st.get("data"), dict):
                self.check("ws-episode-id-honored", st["data"].get("episode_id") == "openenv-conformance-probe", "low",
                           f"reset(episode_id=...) was ignored (state.episode_id={st['data'].get('episode_id')!r}).",
                           "Pass ResetRequest.episode_id through to reset().", W)

            # session isolation: a second session must not disturb this one
            self._ws_call(ws, {"type": "reset", "data": {}})
            try:
                other = self._ws_open()
                self._ws_call(other, {"type": "reset", "data": {}})
                self._ws_call(other, {"type": "step", "data": action})
                self._ws_call(other, {"type": "step", "data": action})
                other_state = self._ws_call(other, {"type": "state"})
                other.close()
                mine = self._ws_call(ws, {"type": "state"})
                if mine and other_state and isinstance(mine.get("data"), dict):
                    same_ep = mine["data"].get("episode_id") == (other_state.get("data") or {}).get("episode_id")
                    self.check("ws-sessions-isolated", not same_ep and mine["data"].get("step_count") == 0, "high",
                               "Two concurrent /ws sessions share one environment: activity in session B changed session A's state.",
                               "Pass create_app an env class or factory that builds a new instance per session, not a lambda returning one shared instance.", W)
            except Exception as exc:
                self.check("ws-concurrent-sessions", False, "low", f"Could not open a second concurrent session ({exc}).",
                           "The server allows one session (openenv-core default). For parallel rollouts set SUPPORTS_CONCURRENT_SESSIONS = True on an env that isolates state, and raise max_concurrent_envs.", W)

            # negative messages: must answer with an error message and keep the session alive
            for cid, msg, raw, label in (
                ("ws-rejects-invalid-action", {"type": "step", "data": {"__openenv_conformance__": "not-a-valid-action"}}, None, "an invalid action"),
                ("ws-rejects-unknown-type", {"type": "teleport", "data": {}}, None, "an unknown message type"),
                ("ws-rejects-malformed-json", None, "{not json", "malformed JSON"),
            ):
                if ws.closed:
                    ws = self._ws_reopen(ws)
                self._ws_call(ws, {"type": "reset", "data": {}})
                try:
                    ans = self._ws_call(ws, msg, raw)
                except Exception:
                    ans = None
                if ans is None:
                    self.check(cid, False, "high", f"Sending {label} over /ws dropped the session.",
                               "Catch validation errors per message and reply {type: error, data: {code, message}}.", W)
                    ws = self._ws_reopen(ws)
                    continue
                self.check(cid, ans.get("type") == "error", "medium",
                           f"Sending {label} over /ws answered {_ws_desc(ans)} instead of an error message.",
                           "Reply {type: error, data: {code, message}} (VALIDATION_ERROR, UNKNOWN_TYPE, INVALID_JSON).", W)
            try:
                ws.send_text(json.dumps({"type": "close"}))
            except OSError:
                pass
        except Exception as exc:  # session blew up mid-probe
            self.check("ws-session-stable", False, "high", f"/ws session failed mid-probe: {type(exc).__name__}: {exc}",
                       "Keep the session open across messages; log server-side exceptions.", W)
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _ws_shape(self, data: dict, where: str) -> None:
        self.check("observation-is-object", isinstance(data.get("observation"), dict), "high",
                   f"{where}: 'observation' must be a JSON object, got {_t(data.get('observation'))}.",
                   "Return the Observation model serialised as an object under 'observation'.", "/ws")
        self.check("done-is-bool", isinstance(data.get("done"), bool), "high",
                   f"{where}: 'done' must be a boolean, got {_t(data.get('done'))}.",
                   "Always include done: true|false.", "/ws")
        self.check("reward-is-number", _is_reward(data.get("reward")), "high",
                   f"{where}: 'reward' must be a finite number or null, got {_t(data.get('reward'))} {data.get('reward')!r}.",
                   "Send reward as a JSON number (or null).", "/ws")

    def _liveness_and_latency(self) -> ProbeResult:
        h = self._req("meta", "GET", "/health")
        self.check("alive-after-probe", h.ok, "critical", f"/health returned {_describe(h)} after the probe; the server died.",
                   "Catch exceptions per request; one bad input must not take the environment down.", "/health")
        step = sorted(self.r.latencies.get("step", []))
        if step:
            p95 = _pct(step, 95)
            self.check("step-latency", p95 <= self.max_p95_ms, "medium",
                       f"step p95 latency {p95:.0f} ms exceeds {self.max_p95_ms:.0f} ms.",
                       "Profile step(); slow envs starve GPU rollouts. Adjust --max-p95-ms if this env is slow by design.", "POST /step")
        return self.r


def _t(v: Any) -> str:
    return "missing" if v is None else type(v).__name__


def _ws_desc(m: Optional[dict]) -> str:
    if m is None:
        return "nothing (session closed)"
    return json.dumps(m)[:100]


def _describe(r: Resp) -> str:
    if r.error:
        return f"no response ({r.error})"
    snippet = r.raw[:60].decode("utf-8", "replace").replace("\n", " ")
    return f"HTTP {r.status} {snippet!r}"


def parse_headers(items: List[str]) -> Dict[str, str]:
    out = {}
    for it in items:
        k, sep, v = it.partition(":")
        if not sep:
            raise ValueError(f"header must be 'Name: value', got {it!r}")
        out[k.strip()] = v.strip()
    return out


def probe(url: str, **kw) -> Tuple[ProbeResult, Dict[str, Any]]:
    headers = kw.pop("headers", None)
    timeout = kw.pop("timeout", 10.0)
    p = Prober(Client(url, timeout=timeout, headers=headers), **kw)
    res = p.run()
    return res, res.stats()
