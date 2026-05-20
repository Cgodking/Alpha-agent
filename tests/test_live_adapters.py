from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alpha.clients import BrainHTTPClient, OpenAICompatibleAIClient
from alpha.models import SimulationFailure


class FakeResponse:
    def __init__(self, status_code=200, data=None, headers=None, text=""):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._data


class FakeSession:
    def __init__(self):
        self.calls = []
        self.routes = {}

    def route(self, method, path, response):
        self.routes[(method.upper(), path)] = response

    def _response(self, method, path):
        response = self.routes[(method.upper(), path)]
        if isinstance(response, list):
            if len(response) > 1:
                return response.pop(0)
            return response[0]
        return response

    def post(self, url, **kwargs):
        path = url.replace("https://api.worldquantbrain.com", "")
        self.calls.append(("POST", path, kwargs))
        return self._response("POST", path)

    def get(self, url, **kwargs):
        path = url.replace("https://api.worldquantbrain.com", "")
        self.calls.append(("GET", path, kwargs))
        return self._response("GET", path)


class LiveAdapterTests(unittest.TestCase):
    def test_openai_compatible_ai_client_parses_json_candidates(self):
        def transport(payload):
            self.assertEqual(payload["model"], "test-model")
            content = json.dumps(
                {
                    "candidates": [
                        {"expression": "rank(close)", "settings": {"neutralization": "INDUSTRY"}},
                        {"expression": "rank(-returns)"},
                    ]
                }
            )
            return {"choices": [{"message": {"content": content}}]}

        client = OpenAICompatibleAIClient(api_key="test", model="test-model", transport=transport)
        candidates = client.generate_candidates(2, {"region": "USA", "delay": 1})

        self.assertEqual([item.expression for item in candidates], ["rank(close)", "rank(-returns)"])
        self.assertEqual(candidates[0].settings["region"], "USA")
        self.assertEqual(candidates[0].settings["neutralization"], "INDUSTRY")
        self.assertEqual(candidates[0].source, "openai_compatible")

    def test_brain_http_client_simulates_then_fetches_submission_checks(self):
        session = FakeSession()
        session.route("POST", "/simulations", FakeResponse(201, headers={"Location": "/simulations/1"}))
        session.route("GET", "/simulations/1", FakeResponse(200, {"alpha": "abc123"}))
        session.route(
            "GET",
            "/alphas/abc123",
            FakeResponse(200, {"is": {"sharpe": 2.0, "fitness": 1.1, "turnover": 0.2}}),
        )
        session.route("POST", "/alphas/abc123/check", FakeResponse(200, headers={}))
        session.route(
            "GET",
            "/alphas/abc123/check",
            FakeResponse(
                200,
                [
                    {"name": "SELF_CORRELATION", "result": "PASS", "value": 0.2},
                    {"name": "PROD_CORRELATION", "result": "PASS", "value": 0.3},
                    {"name": "DATA_DIVERSITY", "result": "PASS"},
                    {"name": "REGULAR_SUBMISSION", "result": "PASS"},
                ],
            ),
        )
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        result = client.simulate("rank(close)", {"region": "USA", "universe": "TOP3000", "delay": 1})

        self.assertEqual(result.alpha_id, "abc123")
        self.assertEqual(result.metrics["sharpe"], 2.0)
        self.assertEqual(result.checks["PROD_CORRELATION"]["status"], "PASS")
        self.assertEqual(session.calls[0][0:2], ("POST", "/simulations"))

    def test_brain_http_client_uses_multisimulation_for_candidate_batches(self):
        session = FakeSession()
        session.route("POST", "/simulations", FakeResponse(201, headers={"Location": "/simulations/multi"}))
        session.route("GET", "/simulations/multi", FakeResponse(200, {"children": ["child1", "child2"]}))
        session.route("GET", "/simulations/child1", FakeResponse(200, {"alpha": "alpha1"}))
        session.route("GET", "/simulations/child2", FakeResponse(200, {"alpha": "alpha2"}))
        session.route(
            "GET",
            "/alphas/alpha1",
            FakeResponse(200, {"is": {"sharpe": 2.0, "fitness": 1.1}, "tests": {"SELF_CORRELATION": "PASS"}}),
        )
        session.route(
            "GET",
            "/alphas/alpha2",
            FakeResponse(200, {"is": {"sharpe": 1.8, "fitness": 1.0}, "tests": {"SELF_CORRELATION": "PASS"}}),
        )
        session.route("POST", "/alphas/alpha1/check", FakeResponse(200, headers={}))
        session.route("POST", "/alphas/alpha2/check", FakeResponse(200, headers={}))
        session.route("GET", "/alphas/alpha1/check", FakeResponse(200, {}))
        session.route("GET", "/alphas/alpha2/check", FakeResponse(200, {}))
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        results = client.simulate_many(
            [
                ("rank(close)", {"region": "USA", "universe": "TOP3000", "delay": 1}),
                ("rank(-returns)", {"region": "USA", "universe": "TOP3000", "delay": 1}),
            ]
        )

        self.assertEqual([result.alpha_id for result in results], ["alpha1", "alpha2"])
        post_call = session.calls[0]
        self.assertEqual(post_call[0:2], ("POST", "/simulations"))
        self.assertIsInstance(post_call[2]["json"], list)
        self.assertEqual(len(post_call[2]["json"]), 2)
        self.assertEqual(post_call[2]["json"][0]["regular"], "rank(close)")

    def test_brain_http_client_returns_partial_multisimulation_failures(self):
        session = FakeSession()
        session.route("POST", "/simulations", FakeResponse(201, headers={"Location": "/simulations/multi"}))
        session.route("GET", "/simulations/multi", FakeResponse(200, {"children": ["child1", "child2"]}))
        session.route("GET", "/simulations/child1", FakeResponse(200, {"alpha": "alpha1"}))
        session.route(
            "GET",
            "/simulations/child2",
            FakeResponse(200, {"status": "ERROR", "detail": "Operator ts_backfill does not support event inputs"}),
        )
        session.route("GET", "/alphas/alpha1", FakeResponse(200, {"is": {"sharpe": 2.0, "fitness": 1.1}}))
        session.route("POST", "/alphas/alpha1/check", FakeResponse(200, headers={}))
        session.route("GET", "/alphas/alpha1/check", FakeResponse(200, {}))
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        results = client.simulate_many(
            [
                ("rank(close)", {"region": "USA", "universe": "TOP3000", "delay": 1}),
                (
                    "rank(ts_backfill(analyst_sentence_count_presentation, 120))",
                    {"region": "USA", "universe": "TOP3000", "delay": 1},
                ),
            ]
        )

        self.assertEqual(results[0].alpha_id, "alpha1")
        self.assertIsInstance(results[1], SimulationFailure)
        self.assertIn("ts_backfill", results[1].error)

    def test_brain_http_client_retries_alpha_detail_404_after_multisimulation_child_ready(self):
        session = FakeSession()
        session.route("POST", "/simulations", FakeResponse(201, headers={"Location": "/simulations/multi"}))
        session.route("GET", "/simulations/multi", FakeResponse(200, {"children": ["child1", "child2"]}))
        session.route("GET", "/simulations/child1", FakeResponse(200, {"alpha": "alpha1"}))
        session.route("GET", "/simulations/child2", FakeResponse(200, {"alpha": "alpha2"}))
        session.route(
            "GET",
            "/alphas/alpha1",
            [
                FakeResponse(404, {"message": "not ready"}),
                FakeResponse(200, {"is": {"sharpe": 2.0, "fitness": 1.1}}),
            ],
        )
        session.route("GET", "/alphas/alpha2", FakeResponse(200, {"is": {"sharpe": 1.7, "fitness": 1.0}}))
        session.route("POST", "/alphas/alpha1/check", FakeResponse(200, headers={}))
        session.route("POST", "/alphas/alpha2/check", FakeResponse(200, headers={}))
        session.route("GET", "/alphas/alpha1/check", FakeResponse(200, {}))
        session.route("GET", "/alphas/alpha2/check", FakeResponse(200, {}))
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        results = client.simulate_many(
            [
                ("rank(close)", {"region": "USA", "universe": "TOP3000", "delay": 1}),
                ("rank(-returns)", {"region": "USA", "universe": "TOP3000", "delay": 1}),
            ]
        )

        self.assertEqual(results[0].alpha_id, "alpha1")
        self.assertEqual(results[0].metrics["sharpe"], 2.0)
        alpha_detail_calls = [call for call in session.calls if call[0:2] == ("GET", "/alphas/alpha1")]
        self.assertEqual(len(alpha_detail_calls), 2)

    def test_brain_http_client_does_not_use_simulation_id_as_alpha_id(self):
        session = FakeSession()
        session.route("POST", "/simulations", FakeResponse(201, headers={"Location": "/simulations/sim1"}))
        session.route(
            "GET",
            "/simulations/sim1",
            [
                FakeResponse(200, {"id": "sim1", "status": "COMPLETE"}),
                FakeResponse(200, {"id": "sim1", "status": "COMPLETE", "alpha": "alpha1"}),
            ],
        )
        session.route("GET", "/alphas/sim1", FakeResponse(404, {"message": "simulation id is not an alpha"}))
        session.route("GET", "/alphas/alpha1", FakeResponse(200, {"is": {"sharpe": 2.0, "fitness": 1.1}}))
        session.route("POST", "/alphas/alpha1/check", FakeResponse(200, headers={}))
        session.route("GET", "/alphas/alpha1/check", FakeResponse(200, {}))
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        result = client.simulate("rank(close)", {"region": "USA", "universe": "TOP3000", "delay": 1})

        self.assertEqual(result.alpha_id, "alpha1")
        self.assertNotIn(("GET", "/alphas/sim1", {}), session.calls)

    def test_brain_http_client_raises_platform_simulation_error(self):
        session = FakeSession()
        session.route("POST", "/simulations", FakeResponse(201, headers={"Location": "/simulations/err"}))
        session.route("GET", "/simulations/err", FakeResponse(200, {"status": "ERROR", "detail": "Unknown variable"}))
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        with self.assertRaises(RuntimeError) as ctx:
            client.simulate("rank(missing_field)", {"region": "USA"})

        self.assertIn("Unknown variable", str(ctx.exception))

    def test_brain_http_client_uses_alpha_detail_tests_when_check_endpoint_is_empty(self):
        session = FakeSession()
        session.route("POST", "/simulations", FakeResponse(201, headers={"Location": "/simulations/2"}))
        session.route("GET", "/simulations/2", FakeResponse(200, {"alpha": "abc456"}))
        session.route(
            "GET",
            "/alphas/abc456",
            FakeResponse(
                200,
                {
                    "is": {"sharpe": 1.9, "fitness": 1.1},
                    "tests": {
                        "SELF_CORRELATION": {"status": "PASS", "value": 0.2},
                        "PROD_CORRELATION": {"status": "FAIL", "value": 0.82},
                    },
                },
            ),
        )
        session.route("POST", "/alphas/abc456/check", FakeResponse(200, headers={}))
        session.route("GET", "/alphas/abc456/check", FakeResponse(200, {}))
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        result = client.simulate("rank(close)", {"region": "USA"})

        self.assertEqual(result.checks["PROD_CORRELATION"]["status"], "FAIL")
        self.assertEqual(result.checks["PROD_CORRELATION"]["value"], 0.82)

    def test_brain_http_client_uses_alpha_detail_is_checks_when_check_endpoint_is_empty(self):
        session = FakeSession()
        session.route("POST", "/simulations", FakeResponse(201, headers={"Location": "/simulations/3"}))
        session.route("GET", "/simulations/3", FakeResponse(200, {"alpha": "abc789"}))
        session.route(
            "GET",
            "/alphas/abc789",
            FakeResponse(
                200,
                {
                    "is": {
                        "sharpe": 3.33,
                        "fitness": 2.17,
                        "checks": [
                            {"name": "LOW_SHARPE", "result": "PASS", "value": 3.33},
                            {"name": "SELF_CORRELATION", "result": "PENDING"},
                            {"name": "DATA_DIVERSITY", "result": "PENDING"},
                            {"name": "PROD_CORRELATION", "result": "PENDING"},
                            {"name": "REGULAR_SUBMISSION", "result": "PENDING"},
                        ],
                    }
                },
            ),
        )
        session.route("POST", "/alphas/abc789/check", FakeResponse(200, headers={}))
        session.route("GET", "/alphas/abc789/check", FakeResponse(200, {}))
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        result = client.simulate("rank(close)", {"region": "IND"})

        self.assertEqual(result.checks["LOW_SHARPE"]["status"], "PASS")
        self.assertEqual(result.checks["SELF_CORRELATION"]["status"], "PENDING")
        self.assertEqual(result.checks["DATA_DIVERSITY"]["status"], "PENDING")

    def test_brain_http_submit_counts_success_only_after_os_verification(self):
        session = FakeSession()
        session.route("POST", "/alphas/abc123/submit", FakeResponse(201, {}))
        session.route(
            "GET",
            "/alphas/abc123",
            FakeResponse(200, {"stage": "OS", "dateSubmitted": "2026-04-29T12:00:00Z"}),
        )
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        result = client.submit_alpha("abc123", dry_run=False)

        self.assertTrue(result.submitted)
        self.assertEqual(result.stage, "OS")

    def test_brain_http_submit_rejects_when_platform_does_not_move_to_os(self):
        session = FakeSession()
        session.route("POST", "/alphas/abc123/submit", FakeResponse(201, {}))
        session.route("GET", "/alphas/abc123", FakeResponse(200, {"stage": "IS", "dateSubmitted": None}))
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        result = client.submit_alpha("abc123", dry_run=False)

        self.assertFalse(result.submitted)
        self.assertEqual(result.stage, "IS")

    def test_brain_http_client_counts_submitted_alphas(self):
        session = FakeSession()
        session.route("GET", "/users/self/alphas", FakeResponse(200, {"results": [{"id": "a"}, {"id": "b"}]}))
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        count = client.count_submitted_alphas("2026-04-29T00:00:00Z", "2026-04-30T00:00:00Z")

        self.assertEqual(count, 2)
        method, path, kwargs = session.calls[0]
        self.assertEqual((method, path), ("GET", "/users/self/alphas"))
        self.assertEqual(kwargs["params"]["stage"], "OS")
        self.assertEqual(kwargs["params"]["dateSubmitted>"], "2026-04-29T00:00:00Z")
        self.assertEqual(kwargs["params"]["dateSubmitted<"], "2026-04-30T00:00:00Z")

    def test_brain_http_client_discovers_datafields_for_scope(self):
        session = FakeSession()
        session.route(
            "GET",
            "/data-fields",
            FakeResponse(
                200,
                {
                    "count": 2,
                    "results": [
                        {"id": "mdl_score", "dataset": {"id": "model1"}, "type": "MATRIX"},
                        {"id": "mdl_score", "dataset": {"id": "model1"}, "type": "MATRIX"},
                        {"id": "analyst_signal", "dataset": {"id": "analyst1"}, "type": "MATRIX"},
                    ],
                },
            ),
        )
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        fields = client.discover_datafields(
            {"region": "EUR", "universe": "TOP2500", "delay": 1},
            search_terms=["model"],
            max_fields=10,
        )

        self.assertEqual([field["id"] for field in fields], ["mdl_score", "analyst_signal"])
        method, path, kwargs = session.calls[0]
        self.assertEqual((method, path), ("GET", "/data-fields"))
        self.assertEqual(kwargs["params"]["region"], "EUR")
        self.assertEqual(kwargs["params"]["universe"], "TOP2500")
        self.assertEqual(kwargs["params"]["delay"], 1)
        self.assertEqual(kwargs["params"]["search"], "model")

    def test_brain_http_client_retries_datafield_rate_limit(self):
        class RateLimitSession:
            def __init__(self):
                self.calls = []

            def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                if len(self.calls) == 1:
                    return FakeResponse(429, {"message": "rate limit"}, headers={"Retry-After": "0"})
                return FakeResponse(200, {"results": [{"id": "field_after_retry"}]})

        session = RateLimitSession()
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        fields = client.discover_datafields({"region": "USA", "universe": "TOP3000", "delay": 1}, max_fields=5)

        self.assertEqual(fields[0]["id"], "field_after_retry")
        self.assertEqual(len(session.calls), 2)

    def test_brain_http_client_paginates_datafield_discovery(self):
        session = FakeSession()
        first_page = [{"id": f"field_{idx}", "type": "MATRIX"} for idx in range(50)]
        second_page = [{"id": f"field_{idx}", "type": "MATRIX"} for idx in range(50, 75)]
        session.route(
            "GET",
            "/data-fields",
            [
                FakeResponse(200, {"results": first_page}),
                FakeResponse(200, {"results": second_page}),
            ],
        )
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        fields = client.discover_datafields({"region": "USA", "universe": "TOP500", "delay": 0}, max_fields=75)

        self.assertEqual(len(fields), 75)
        offsets = [call[2]["params"]["offset"] for call in session.calls]
        self.assertEqual(offsets, [0, 50])

    def test_brain_http_client_spreads_datafield_discovery_across_search_terms(self):
        class SearchAwareSession:
            def __init__(self):
                self.calls = []

            def get(self, url, **kwargs):
                path = url.replace("https://api.worldquantbrain.com", "")
                self.calls.append(("GET", path, kwargs))
                params = kwargs.get("params", {})
                search = params.get("search", "")
                offset = int(params.get("offset", 0))
                if search == "news":
                    return FakeResponse(200, {"results": [{"id": f"news_field_{idx}", "type": "MATRIX"} for idx in range(10)]})
                return FakeResponse(
                    200,
                    {"results": [{"id": f"general_field_{idx}", "type": "MATRIX"} for idx in range(offset, offset + 50)]},
                )

        session = SearchAwareSession()
        client = BrainHTTPClient(session=session, sleep=lambda _seconds: None)

        fields = client.discover_datafields(
            {"region": "USA", "universe": "TOP500", "delay": 0},
            search_terms=["", "news"],
            max_fields=60,
        )

        ids = [field["id"] for field in fields]
        self.assertIn("news_field_0", ids)
        searches = [call[2]["params"].get("search", "") for call in session.calls]
        self.assertIn("news", searches)

    def test_brain_http_client_loads_credentials_file_from_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            cred_path = Path(tmp) / "brain_credentials.txt"
            cred_path.write_text(json.dumps(["user@example.com", "secret"]), encoding="utf-8")
            with patch.dict(os.environ, {"BRAIN_CREDENTIALS_FILE": str(cred_path)}, clear=True):
                email, password = BrainHTTPClient.credentials_from_env()

        self.assertEqual(email, "user@example.com")
        self.assertEqual(password, "secret")


if __name__ == "__main__":
    unittest.main()
