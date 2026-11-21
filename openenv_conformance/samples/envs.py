"""Two tiny OpenEnv-style environment servers (stdlib only) for demos and tests.

They mirror how ``openenv-core`` serves an environment:

* HTTP ``POST /reset``, ``POST /step`` and ``GET /state`` are stateless: each
  request runs against a fresh environment instance.
* ``/ws`` is the stateful session channel: one environment per connection,
  driven by ``{"type": "reset"|"step"|"state"|"close", "data": {...}}``
  messages and answered with ``observation`` / ``state`` / ``error`` messages.

``healthy`` follows the contract. ``broken`` looks similar but plants faults:
no /schema, reset without ``done``, rewards sent as strings, 500s on bad
input, seeds ignored, text/plain bodies, and on /ws a step counter that never
moves, episode_id ignored, one game shared by every session, and the socket
dropped on an invalid action.

The game: guess a hidden integer in [0, 20]. Action: {"guess": int}.
"""
from __future__ import annotations

import json
import random
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from ..wsproto import ServerConn, WSClosed, server_handshake

MAX_STEPS = 6

SCHEMA = {
    "action": {
        "type": "object",
        "properties": {"guess": {"type": "integer", "minimum": 0, "maximum": 20}},
        "required": ["guess"],
    },
    "observation": {"type": "object", "properties": {"hint": {"type": "string"}}},
    "state": {
        "type": "object",
        "properties": {"episode_id": {"type": "string"}, "step_count": {"type": "integer"}},
    },
}


class Game:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self, seed=None, episode_id=None):
        rng = random.Random(seed) if seed is not None else random.Random()
        self.secret = rng.randint(0, 20)
        self.step_count = 0
        self.done = False
        self.episode_id = episode_id or str(uuid.uuid4())
        # secret_hash lets a client check seeded determinism without revealing the secret
        return {"hint": "start", "range": [0, 20], "secret_hash": self.secret * 41 % 101}

    def step(self, guess: int):
        self.step_count += 1
        if guess == self.secret:
            hint, reward, self.done = "correct", 1.0, True
        else:
            hint, reward = ("higher" if guess < self.secret else "lower"), -0.1
            if self.step_count >= MAX_STEPS:
                self.done = True
        return {"hint": hint, "last_guess": guess}, reward, self.done


def _valid_action(a) -> bool:
    return isinstance(a, dict) and isinstance(a.get("guess"), int) and not isinstance(a.get("guess"), bool)


def _make_handler(broken: bool):
    shared = Game()  # the broken server (wrongly) shares this across sessions

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code: int, body, ctype: str = "application/json") -> None:
            raw = (json.dumps(body) if not isinstance(body, str) else body).encode()
            if broken and ctype == "application/json":
                ctype = "text/plain"
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _body(self) -> Tuple[Optional[dict], Optional[str]]:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if not raw:
                return {}, None
            try:
                data = json.loads(raw)
            except ValueError:
                return None, "invalid JSON body"
            if not isinstance(data, dict):
                return None, "body must be a JSON object"
            return data, None

        # -- HTTP (stateless) ---------------------------------------------
        def do_GET(self):
            if self.path == "/ws":
                if server_handshake(self):
                    return self._ws_session()
                return self._send(400, {"detail": "expected websocket upgrade"})
            if self.path == "/health":
                return self._send(200, {"status": "healthy"})
            if self.path == "/metadata":
                return self._send(200, {"name": "guess_env" + ("_broken" if broken else ""),
                                        "description": "Guess the hidden number.", "version": "0.1.0"})
            if self.path == "/schema" and not broken:
                return self._send(200, SCHEMA)
            if self.path == "/state":
                g = Game()
                return self._send(200, {"episode_id": g.episode_id, "step_count": g.step_count})
            return self._send(404, {"detail": "Not Found"})

        def do_POST(self):
            body, err = self._body()
            if self.path == "/reset":
                if err:
                    return self._send(500 if broken else 422, {"detail": err})
                seed = None if broken else body.get("seed")
                if seed is not None and (not isinstance(seed, int) or seed < 0):
                    return self._send(422, {"detail": "seed must be a non-negative integer"})
                obs = Game().reset(seed, body.get("episode_id"))
                resp = {"observation": obs, "reward": None}
                if not broken:
                    resp["done"] = False
                return self._send(200, resp)
            if self.path == "/step":
                if err:
                    return self._send(500 if broken else 422, {"detail": err})
                if not _valid_action(body.get("action")):
                    if broken:
                        return self._send(500, "Internal Server Error", "text/plain")
                    return self._send(422, {"detail": "action must be an object with integer 'guess'"})
                obs, reward, done = Game().step(body["action"]["guess"])
                return self._send(200, {"observation": obs, "reward": str(reward) if broken else reward, "done": done})
            return self._send(404, {"detail": "Not Found"})

        # -- /ws (stateful sessions) --------------------------------------
        def _ws_session(self):
            conn = ServerConn(self)
            game = shared if broken else Game()
            self.close_connection = True

            def reply(kind, data):
                conn.send_text(json.dumps({"type": kind, "data": data}))

            def error(code, message):
                reply("error", {"code": code, "message": message})

            while True:
                try:
                    raw = conn.recv_text()
                except (WSClosed, OSError):
                    return
                if raw is None:
                    return
                try:
                    msg = json.loads(raw)
                except ValueError:
                    error("INVALID_JSON", "Invalid JSON")
                    continue
                kind = msg.get("type") if isinstance(msg, dict) else None
                data = (msg.get("data") or {}) if isinstance(msg, dict) else {}
                with game.lock:
                    if kind == "reset":
                        seed = None if broken else data.get("seed")
                        ep = None if broken else data.get("episode_id")
                        obs = game.reset(seed, ep)
                        reply("observation", {"observation": obs, "reward": None, "done": False})
                    elif kind == "step":
                        if not _valid_action(data):
                            if broken:
                                return conn.drop()
                            error("VALIDATION_ERROR", "action must be an object with integer 'guess'")
                            continue
                        if game.done and not broken:
                            error("EXECUTION_ERROR", "episode is done; send reset")
                            continue
                        obs, reward, done = game.step(data["guess"])
                        reply("observation", {"observation": obs, "reward": reward, "done": done})
                    elif kind == "state":
                        reply("state", {"episode_id": game.episode_id, "step_count": 0 if broken else game.step_count})
                    elif kind == "close":
                        return
                    else:
                        error("UNKNOWN_TYPE", f"Unknown message type: {kind!r}")

    return Handler


def serve(kind: str = "healthy", host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Start a sample env in a daemon thread and return the server."""
    if kind not in ("healthy", "broken"):
        raise ValueError("kind must be 'healthy' or 'broken'")
    srv = ThreadingHTTPServer((host, port), _make_handler(kind == "broken"))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":  # python -m openenv_conformance.samples.envs healthy 8000
    import sys

    kind = sys.argv[1] if len(sys.argv) > 1 else "healthy"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    print(f"{kind} sample env on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), _make_handler(kind == "broken")).serve_forever()
