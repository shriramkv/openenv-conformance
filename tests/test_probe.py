import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from openenv_conformance import cli  # noqa: E402
from openenv_conformance.probe import Client, Prober, _pct, example_from_schema, parse_headers  # noqa: E402
from openenv_conformance.samples.envs import serve  # noqa: E402
from openenv_conformance.wsproto import WSClient, WSError, accept_key  # noqa: E402


def url(srv):
    return "http://%s:%d" % srv.server_address[:2]


def cli_run(*argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(list(argv))
    return rc, buf.getvalue()


class HealthyEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = serve("healthy")
        cls.res = Prober(Client(url(cls.srv))).run()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_no_findings(self):
        self.assertEqual(self.res.findings, [], [f.message for f in self.res.findings])

    def test_action_derived_from_schema(self):
        used = next(c for c in self.res.checks if c.id == "action-used")
        self.assertEqual(json.loads(used.detail), {"guess": 0})

    def test_ws_checks_ran(self):
        ids = {c.id for c in self.res.checks}
        for cid in ("ws-available", "ws-step-count-increments", "ws-sessions-isolated",
                    "ws-rejects-invalid-action", "ws-rejects-unknown-type", "ws-rejects-malformed-json",
                    "episode-terminates", "ws-episode-id-honored"):
            self.assertIn(cid, ids)

    def test_stats(self):
        st = self.res.stats()
        self.assertEqual(st["checks"], st["passed"])
        self.assertIn("step_p95_ms", st)
        self.assertIn("ws_p95_ms", st)


class BrokenEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = serve("broken")
        cls.res = Prober(Client(url(cls.srv)), action={"guess": 10}).run()
        cls.rules = {f.rule: f for f in cls.res.findings}

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_catches_planted_faults(self):
        expected = {
            "done-is-bool", "reward-is-number", "schema", "json-content-type",
            "rejects-non-object-action", "rejects-missing-action", "rejects-malformed-json",
            "ws-step-count-increments", "ws-episode-id-honored", "ws-sessions-isolated",
            "ws-rejects-invalid-action",
        }
        self.assertTrue(expected <= set(self.rules), expected - set(self.rules))

    def test_server_errors_are_high(self):
        self.assertEqual(self.rules["rejects-malformed-json"].severity, "high")
        self.assertIn("dropped the session", self.rules["ws-rejects-invalid-action"].message)

    def test_repeat_defects_are_collapsed(self):
        self.assertEqual(sum(f.rule == "reward-is-number" for f in self.res.findings), 1)

    def test_server_survives(self):
        self.assertNotIn("alive-after-probe", self.rules)


class ModeAndOptionTests(unittest.TestCase):
    def setUp(self):
        self.srv = serve("healthy")

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_production_mode_flags_exposed_simulation_routes(self):
        res = Prober(Client(url(self.srv)), mode="production").run()
        self.assertIn("simulation-routes-hidden", {f.rule for f in res.findings})

    def test_no_ws_skips_session_checks(self):
        res = Prober(Client(url(self.srv)), ws=False).run()
        self.assertFalse(any(c.id.startswith("ws-") for c in res.checks))
        self.assertEqual(res.findings, [])

    def test_wrong_action_gets_helpful_fix(self):
        res = Prober(Client(url(self.srv)), action={"move": "left"}, ws=False).run()
        f = next(f for f in res.findings if f.rule == "step-works")
        self.assertIn("--action", f.fix)

    def test_latency_budget(self):
        res = Prober(Client(url(self.srv)), max_p95_ms=0.0001, ws=False).run()
        self.assertIn("step-latency", {f.rule for f in res.findings})

    def test_ws_client_roundtrip(self):
        ws = WSClient(url(self.srv).replace("http", "ws") + "/ws")
        ws.send_text(json.dumps({"type": "reset", "data": {"seed": 3}}))
        msg = json.loads(ws.recv_text())
        self.assertEqual(msg["type"], "observation")
        self.assertFalse(msg["data"]["done"])
        ws.close()

    def test_ws_handshake_failure(self):
        with self.assertRaises(WSError):
            WSClient(url(self.srv).replace("http", "ws") + "/health")


class CliTests(unittest.TestCase):
    def test_demo_healthy_json(self):
        rc, out = cli_run("--demo", "healthy", "-f", "json", "--min-grade", "A")
        data = json.loads(out)
        self.assertEqual((rc, data["grade"]), (0, "A"))
        self.assertTrue(all(c["passed"] for c in data["checks"]))

    def test_demo_broken_gates(self):
        self.assertEqual(cli_run("--demo", "broken", "--min-grade", "C")[0], 1)
        self.assertEqual(cli_run("--demo", "broken", "--fail-on", "high")[0], 1)

    def test_unreachable_returns_2(self):
        rc, out = cli_run("http://127.0.0.1:9", "--timeout", "1")
        self.assertEqual(rc, 2)
        self.assertIn("health-reachable", out)

    def test_bad_action_json_returns_2(self):
        self.assertEqual(cli_run("--demo", "healthy", "--action", "{nope")[0], 2)

    def test_formats(self):
        _, md = cli_run("--demo", "broken", "-f", "markdown")
        self.assertIn("| Severity | Rule |", md)
        _, badge = cli_run("--demo", "healthy", "-f", "badge")
        self.assertIn("brightgreen", badge)


class HelperTests(unittest.TestCase):
    def test_example_from_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "message": {"type": "string", "minLength": 1},
                "n": {"type": "integer", "minimum": 3},
                "mode": {"enum": ["fast", "slow"]},
                "opt": {"type": ["null", "number"]},
                "tags": {"type": "array"},
            },
            "required": ["message", "n", "mode"],
        }
        self.assertEqual(example_from_schema(schema), {"message": "probe", "n": 3, "mode": "fast"})
        self.assertEqual(example_from_schema({"anyOf": [{"type": "boolean"}]}), False)
        self.assertEqual(example_from_schema({"type": "object", "default": {"a": 1}}), {"a": 1})

    def test_headers_and_percentile(self):
        self.assertEqual(parse_headers(["Authorization: Bearer x"]), {"Authorization": "Bearer x"})
        with self.assertRaises(ValueError):
            parse_headers(["nocolon"])
        self.assertEqual(_pct([1, 2, 3, 4, 100], 95), 100)

    def test_accept_key_rfc_example(self):
        self.assertEqual(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")


if __name__ == "__main__":
    unittest.main()
