from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from reviewer.cli import main
from reviewer.config import ReviewConfig
from reviewer.github import AcquisitionError
from reviewer.models import ModelError, OpenAIReviewer, _post
from reviewer.pipeline import review_prompt, run_baseline
from reviewer.schemas import BaselineRun, FindingBatch, ReviewInput
from reviewer.storage import save_run
from test_github import DIFF
from test_schemas import finding_data, pr_data

MODEL = "gpt-4.1-mini-2025-04-14"


def config(**updates):
    return ReviewConfig(max_output_tokens=4096, max_attempts=1, input_token_limit=16000, **updates)


def review_input(diff=DIFF.decode()):
    return ReviewInput(pr=pr_data(), diff=diff, diff_complete=True, diff_base="c" * 40)


def finding(**changes):
    return finding_data() | {"file": "src/a.py", "start_line": 1, "end_line": 1} | changes


def response(findings=None, *, raw=None):
    return {"id": "resp_fixture", "model": MODEL, "status": "completed", "error": None,
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": raw if raw is not None else
                                      json.dumps({"findings": [] if findings is None else findings})}]}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                      "input_tokens_details": {"cached_tokens": 10}, "output_tokens_details": {"reasoning_tokens": 0}}}


class BaselineTests(unittest.TestCase):
    def setUp(self):
        guard = patch("socket.create_connection", side_effect=AssertionError("offline tests only"))
        guard.start(); self.addCleanup(guard.stop)

    def provider(self, output=None, **kwargs):
        transport = Mock(return_value=response() if output is None else output)
        return OpenAIReviewer(MODEL, config(), api_key="test-secret", transport=transport, **kwargs), transport

    def test_success_zero_and_multiple_findings(self):
        for findings in ([], [finding()], [finding(), finding(title="Second defect", category="reliability")]):
            with self.subTest(count=len(findings)):
                provider, transport = self.provider(response(findings))
                run = run_baseline(review_input(), provider)
                self.assertEqual(run.result.status, "success")
                self.assertEqual(len(run.result.findings), len(findings))
                self.assertEqual(run.requested_model, MODEL)
                self.assertEqual(run.attempts[0].returned_model, MODEL)
                self.assertEqual(run.attempts[0].usage.total_tokens, 120)
                self.assertEqual(run.attempts[0].usage.cached_input_tokens, 10)
                self.assertIsNone(run.cost_usd)
                self.assertFalse(transport.call_args.args[0]["store"])

    def test_incomplete_status_and_allowlisted_reason_persist_without_secrets(self):
        for details, expected in (({'reason': 'max_output_tokens', 'extra': 'SECRET_SENTINEL'}, 'max_output_tokens'),
                                  ({'reason': 'content_filter'}, 'content_filter'),
                                  ({'reason': 'SECRET_SENTINEL'}, 'unknown'),
                                  ('SECRET_SENTINEL', 'unknown'), (None, None)):
            with self.subTest(expected=expected):
                raw = response()
                raw.update(status='incomplete', incomplete_details=details, output=[])
                provider, transport = self.provider(raw)
                run = run_baseline(review_input(), provider)
                self.assertEqual(run.result.status, 'failed')
                self.assertEqual(run.attempts[0].response_status, 'incomplete')
                reason = run.attempts[0].incomplete_details
                self.assertEqual(reason.reason if reason else None, expected)
                with TemporaryDirectory() as temp:
                    path = save_run(run, Path(temp).resolve())
                    saved = path.read_text()
                    self.assertNotIn('SECRET_SENTINEL', saved)
                    self.assertEqual(BaselineRun.model_validate_json(saved), run)
                transport.assert_called_once()

    def test_unknown_response_status_is_redacted(self):
        raw = response()
        raw['status'] = 'SECRET_SENTINEL'
        provider, _ = self.provider(raw)
        run = run_baseline(review_input(), provider)
        self.assertEqual(run.attempts[0].response_status, 'unknown')
        self.assertNotIn('SECRET_SENTINEL', run.model_dump_json())

    def test_boundary_is_exactly_diff_and_versioned_prompt(self):
        outside = "PRIVATE_REPOSITORY_CONTEXT_SENTINEL"
        signals = "BANDIT_STATIC_SIGNALS_SENTINEL"
        verification = "VERIFIER_VERDICT_SENTINEL"
        golden = "REVIEWBENCH_GOLDEN_LABEL_SENTINEL"
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            for filename, value in (("README.md", outside), ("bandit.json", signals),
                                    ("verification.json", verification), ("golden.json", golden)):
                (root / filename).write_text(value)
            provider, transport = self.provider()
            run = run_baseline(review_input(), provider, input_artifact_directory=str(root))
        sent = transport.call_args.args[0]
        self.assertEqual(json.loads(sent["input"][0]["content"]), {"diff": DIFF.decode()})
        self.assertEqual(sent["instructions"], review_prompt())
        self.assertEqual(sent["text"]["format"]["schema"], FindingBatch.model_json_schema())
        self.assertEqual(set(sent), {"model", "instructions", "input", "text", "max_output_tokens", "store", "truncation"})
        wire = json.dumps(sent)
        for forbidden in (outside, signals, verification, golden, pr_data()["repo"], pr_data()["base"], "test-secret"):
            self.assertNotIn(forbidden, wire)
        self.assertNotIn("tools", sent)
        self.assertNotIn("previous_response_id", sent)
        self.assertNotIn("conversation", sent)
        self.assertEqual(run.result.status, "success")

    def test_invalid_structured_output_is_failed_without_repair(self):
        for raw in ('{"findings":', '```json\n{"findings": []}\n```', '{"findings": [], "findings": []}',
                    json.dumps({"findings": [finding(confidence="0.9")]}),
                    json.dumps({"findings": [finding(file="../escape")]}),
                    json.dumps({"findings": [finding(end_line=0)]}),
                    '{"findings": [], "golden": []}'):
            with self.subTest(raw=raw):
                provider, _ = self.provider(response(raw=raw))
                run = run_baseline(review_input(), provider)
                self.assertEqual(run.result.status, "failed")
                self.assertIn("invalid_output", run.result.error)
                self.assertEqual(run.result.findings, [])
                self.assertEqual(run.output_text, raw)
                self.assertEqual(run.attempts[0].usage.total_tokens, 120)

    def test_impossible_anchors_rejected(self):
        for change in ({"file": "unseen.py"}, {"start_line": 2, "end_line": 2},
                       {"start_line": 1, "end_line": 10**12}):
            provider, _ = self.provider(response([finding(**change)]))
            run = run_baseline(review_input(), provider)
            self.assertEqual(run.result.status, "failed")
            self.assertIn("invalid_reference", run.result.error)

    def test_rename_deleted_file_and_unseen_gap_anchors(self):
        renamed = 'diff --git a/old.py b/new.py\nsimilarity index 80%\nrename from old.py\nrename to new.py\n--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-old\n+new\n'
        for path, status in (("old.py", "failed"), ("new.py", "success")):
            provider, _ = self.provider(response([finding(file=path)]))
            self.assertEqual(run_baseline(review_input(renamed), provider).result.status, status)
        deleted = '--- a/src/a.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n'
        provider, _ = self.provider(response([finding()]))
        self.assertIn("invalid_reference", run_baseline(review_input(deleted), provider).result.error)
        gapped = '--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n@@ -3 +3 @@\n-c\n+d\n'
        provider, _ = self.provider(response([finding(end_line=3)]))
        self.assertIn("invalid_reference", run_baseline(review_input(gapped), provider).result.error)

    def test_context_anchor_for_deletion_is_allowed(self):
        diff = '--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1 @@\n-deleted\n surviving\n'
        provider, _ = self.provider(response([finding()]))
        self.assertEqual(run_baseline(review_input(diff), provider).result.status, "success")

    def test_latency_capture_uses_injected_monotonic_clock(self):
        ticks = iter([1.0, 2.0, 3.0, 5.0])
        provider, _ = self.provider(clock=lambda: next(ticks))
        pipeline_ticks = iter([10.0, 16.0])
        run = run_baseline(review_input(), provider, acquisition_latency_seconds=2.0, clock=lambda: next(pipeline_ticks))
        self.assertEqual(run.attempts[0].latency_seconds, 1.0)
        self.assertEqual(run.model_latency_seconds, 4.0)
        self.assertEqual(run.total_latency_seconds, 8.0)

    def test_missing_usage_is_null_not_zero(self):
        payload = response(); del payload["usage"]
        provider, _ = self.provider(payload)
        run = run_baseline(review_input(), provider)
        self.assertEqual(run.result.status, "success")
        self.assertIsNone(run.attempts[0].usage)

    def test_refusal_incomplete_and_malformed_api_responses(self):
        cases = [None, [], {}, response() | {"status": "incomplete"}, response() | {"output": None},
                 response() | {"usage": {"input_tokens": "100"}},
                 response() | {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 4}}]
        refusal = response(); refusal["output"][0]["content"] = [{"type": "refusal", "refusal": "secret text"}]
        cases.append(refusal)
        for payload in cases:
            provider = OpenAIReviewer(MODEL, config(), api_key="test-secret", transport=Mock(return_value=payload))
            run = run_baseline(review_input(), provider)
            self.assertEqual(run.result.status, "failed")
            self.assertEqual(run.result.findings, [])
            self.assertNotIn("secret text", run.result.error)

    def test_key_and_supported_settings_preflight(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ModelError, "OPENAI_API_KEY"):
            OpenAIReviewer(MODEL, config())
        with self.assertRaisesRegex(ModelError, "omit temperature"):
            OpenAIReviewer("reasoning-model", config(temperature=0.0), api_key="key")
        provider = OpenAIReviewer(MODEL, config(temperature=0.0), api_key="key")
        self.assertEqual(provider.payload("", review_prompt())["temperature"], 0.0)
        provider = OpenAIReviewer("reasoning-model", config(), api_key="key")
        self.assertNotIn("temperature", provider.payload("", review_prompt()))

    def test_budget_and_bad_diff_prevent_model_calls(self):
        provider, transport = self.provider()
        provider.config.input_token_limit = 1
        run = run_baseline(review_input(), provider)
        self.assertIn("input_budget", run.result.error)
        self.assertIsNone(run.request_sha256)
        transport.assert_not_called()
        provider, transport = self.provider()
        self.assertIn("invalid_diff", run_baseline(review_input("broken"), provider).result.error)
        transport.assert_not_called()

    def test_retry_records_each_attempt_and_does_not_change_payload(self):
        settings = config(); settings.max_attempts = 2
        transport = Mock(side_effect=[ModelError("rate_limit", "retry later", retryable=True), response()])
        sleep = Mock()
        provider = OpenAIReviewer(MODEL, settings, api_key="key", transport=transport, sleep=sleep)
        run = run_baseline(review_input(), provider)
        self.assertEqual(run.result.status, "success")
        self.assertEqual(len(run.attempts), 2)
        self.assertIsNone(run.attempts[0].usage)
        self.assertEqual(run.attempts[0].error_code, "rate_limit")
        self.assertEqual(transport.call_args_list[0].args[0], transport.call_args_list[1].args[0])
        sleep.assert_called_once_with(1.0)

    def test_persistence_roundtrip_failure_and_no_overwrite(self):
        provider, _ = self.provider(response(raw="bad JSON"))
        run = run_baseline(review_input(), provider)
        with TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            path = save_run(run, parent, run_id="fixed")
            first = path.read_bytes()
            self.assertEqual(BaselineRun.model_validate_json(first), run)
            self.assertNotIn(b"test-secret", first)
            self.assertEqual(json.loads(first)["result"]["status"], "failed")
            with self.assertRaises(FileExistsError):
                save_run(run, parent, run_id="fixed")
            self.assertEqual(path.read_bytes(), first)
            self.assertNotEqual(save_run(run, parent), save_run(run, parent))


class ModelTransportTests(unittest.TestCase):
    def test_auth_rate_timeout_and_api_errors_are_sanitized(self):
        failures = [(HTTPError("url", 401, "secret", {}, io.BytesIO(b"secret")), "authentication"),
                    (HTTPError("url", 429, "secret", {}, io.BytesIO()), "rate_limit"),
                    (HTTPError("url", 400, "secret", {}, io.BytesIO()), "unsupported_configuration"),
                    (HTTPError("url", 500, "secret", {}, io.BytesIO()), "api_error"),
                    (TimeoutError("secret"), "timeout"), (URLError("secret"), "network")]
        for failure, code in failures:
            opener = Mock(); opener.open.side_effect = failure
            with patch("reviewer.models.build_opener", return_value=opener):
                with self.assertRaises(ModelError) as error:
                    _post({}, "test-secret", 1.0)
            self.assertEqual(error.exception.code, code)
            self.assertNotIn("secret", str(error.exception))
            self.assertEqual(opener.open.call_args.args[0].method, "POST")
            self.assertEqual(opener.open.call_args.args[0].full_url, "https://api.openai.com/v1/responses")


class CLIBaselineTests(unittest.TestCase):
    def test_cli_uses_existing_acquisition_and_persists(self):
        with TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            acquired = SimpleNamespace(review_input=review_input(), artifact_directory=parent / "input",
                                       title="GOLDEN_TITLE_SENTINEL", retrieved_context="CONTEXT_SENTINEL",
                                       static_signals="BANDIT_SENTINEL", verification="VERIFICATION_SENTINEL")
            provider = OpenAIReviewer(MODEL, config(), api_key="key", transport=Mock(return_value=response([finding()])))
            output, errors = io.StringIO(), io.StringIO()
            with patch.dict("os.environ", {"REVIEW_PROVIDER": "openai", "REVIEW_MODEL": MODEL}, clear=True), \
                    patch("reviewer.cli.OpenAIReviewer", return_value=provider), \
                    patch("reviewer.cli.GitHubClient") as github, redirect_stdout(output), redirect_stderr(errors):
                github.return_value.acquire.return_value = acquired
                status = main(["review", "--repo", "owner/repo", "--pr", "7", "--mode", "baseline",
                               "--input-token-limit", "16000", "--results-dir", str(parent / "results")])
            self.assertEqual(status, 0, errors.getvalue())
            self.assertIn("Review complete", output.getvalue())
            self.assertIn("[HIGH]", output.getvalue())
            self.assertIn("Observed tokens", output.getvalue())
            self.assertFalse(github.return_value.acquire.call_args.kwargs["snapshots"])
            self.assertEqual(len(list((parent / "results").glob("*/baseline.json"))), 1)
            wire = json.dumps(provider._transport.call_args.args[0])
            for value in (acquired.title, acquired.retrieved_context, acquired.static_signals, acquired.verification):
                self.assertNotIn(value, wire)

    def test_cli_configuration_errors_precede_acquisition(self):
        with patch.dict("os.environ", {"REVIEW_PROVIDER": "openai", "REVIEW_MODEL": MODEL}, clear=True), \
                patch("reviewer.cli.GitHubClient") as github, redirect_stderr(io.StringIO()) as errors:
            status = main(["review", "--repo", "o/r", "--pr", "1", "--input-token-limit", "16000"])
        self.assertEqual(status, 2)
        self.assertIn("OPENAI_API_KEY", errors.getvalue())
        github.assert_not_called()

    def test_cli_acquisition_and_model_failure_are_nonzero(self):
        for acquisition_failure in (True, False):
            with TemporaryDirectory() as temp:
                provider = OpenAIReviewer(MODEL, config(), api_key="key",
                                          transport=Mock(side_effect=ModelError("authentication", "denied")))
                with patch.dict("os.environ", {"REVIEW_PROVIDER": "openai", "REVIEW_MODEL": MODEL}, clear=True), \
                        patch("reviewer.cli.OpenAIReviewer", return_value=provider), patch("reviewer.cli.GitHubClient") as github, \
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    if acquisition_failure:
                        github.return_value.acquire.side_effect = AcquisitionError("not_found", "missing")
                    else:
                        github.return_value.acquire.return_value = SimpleNamespace(review_input=review_input(), artifact_directory=Path(temp).resolve())
                    status = main(["review", "--repo", "o/r", "--pr", "1", "--input-token-limit", "16000",
                                   "--results-dir", str(Path(temp).resolve() / "results")])
                self.assertEqual(status, 1)
                if not acquisition_failure:
                    paths = list((Path(temp) / "results").glob("*/baseline.json"))
                    self.assertEqual(json.loads(paths[0].read_text())["result"]["status"], "failed")
