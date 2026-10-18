"""Offline Phase 8 tests; all model transports are local mocks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout, redirect_stderr
import io
from unittest.mock import Mock, patch

from reviewer.archives import sha256
from reviewer.benchmark import export_predictions
from reviewer.cli import main as cli_main
from reviewer.config import load_config
from reviewer.models import ModelError, OpenAIReviewer
from reviewer.pipeline import run_full, review_prompt
from reviewer.schemas import (Finding, RepositoryContext, ContextChunk, ReviewInput)
from reviewer.static_analysis import BanditFinding, StaticAnalysisResult
from reviewer.verification import (PROMPT_VERSION, proposal_identifier, verification_prompt,
                                   verify_proposals)
from test_github import DIFF
from test_schemas import finding_data, pr_data
from test_benchmark import response


def proposal(**updates):
    return Finding.model_validate(finding_data() | {'file': 'src/a.py', 'start_line': 1,
                                                     'end_line': 1} | updates)


def verify_response(verdict='SUPPORTED', rationale='The changed line demonstrates the defect.'):
    raw = json.dumps({'verdict': verdict, 'rationale': rationale,
                      'evidence_references': ['src/a.py:1'], 'confidence': 0.9})
    return {'id': 'resp_verify', 'model': 'gpt-6-luna', 'status': 'completed',
            'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': raw}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 12, 'total_tokens': 22,
                      'input_tokens_details': {'cached_tokens': 0},
                      'output_tokens_details': {'reasoning_tokens': 2}}}


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / 'configs/experiment.toml')
        self.config.review.input_token_limit = 32000
        self.inp = ReviewInput(pr=pr_data(), diff=DIFF.decode(), diff_complete=True,
                               diff_base='c' * 40)
        self.archive_hash = 'd' * 64
        self.context = RepositoryContext(
            head=self.inp.pr.head, diff_sha256=sha256(self.inp.diff.encode()),
            archive_sha256=self.archive_hash, repository='owner/repo',
            ranking='structural_lexical_v1', config={},
            budget_unit='serialized_utf8_bytes_conservative_token_estimate',
            budget=100, used_budget=5,
            chunks=[ContextChunk(file='src/helpers.py', start_line=1, end_line=1,
                                 content='def helper(): pass', content_sha256='e' * 64,
                                 score=1, reasons=['test'], selected=True, selection_reason='selected'),
                    ContextChunk(file='README.md', start_line=1, end_line=1,
                                 content='UNRELATED_CONTEXT_SENTINEL', content_sha256='f' * 64,
                                 score=0, reasons=['test'], selected=True, selection_reason='selected')])
        bandit_finding = BanditFinding(test_id='B307', test_name='blacklist', issue_severity='MEDIUM',
            issue_confidence='HIGH', issue_text='Use of eval detected.', file='src/a.py',
            line_number=1, line_range=[1], more_info='https://bandit.readthedocs.io/')
        self.static = StaticAnalysisResult(tool='bandit', version='1.7.8', head=self.inp.pr.head,
            archive_sha256=self.archive_hash, analyzed_files=['src/a.py', 'src/helpers.py'], config={},
            timeout_seconds=10, status='success', error=None, findings=[bandit_finding], metrics={}, signals_sent=[])

    def provider(self, transport):
        return OpenAIReviewer('gpt-6-luna', self.config.review, api_key='offline-key', transport=transport)

    def test_verdict_policy_and_provenance(self):
        for verdict, expected_count in [('SUPPORTED', 1), ('UNSUPPORTED', 0), ('UNCERTAIN', 0)]:
            with self.subTest(verdict=verdict):
                provider = self.provider(Mock(return_value=verify_response(verdict)))
                record = verify_proposals(self.inp, [proposal()], provider)
                self.assertEqual(record.status, 'success')
                self.assertEqual(record.decisions[0].verdict, verdict)
                self.assertEqual(len([p for p, d in zip([proposal()], record.decisions)
                                     if d.verdict == 'SUPPORTED']), expected_count)
                self.assertEqual(record.prompt_version, PROMPT_VERSION)
                self.assertEqual(record.model, 'gpt-6-luna')
                self.assertEqual(record.attempts[0].usage.total_tokens, 22)

    def test_empty_proposals_make_no_verifier_call(self):
        transport = Mock()
        record = verify_proposals(self.inp, [], self.provider(transport))
        self.assertEqual(record.status, 'success')
        self.assertEqual(record.decisions, [])
        transport.assert_not_called()

    def test_malformed_and_api_failure_fail_closed(self):
        malformed = verify_response() | {'output': [{'type': 'message', 'role': 'assistant',
            'status': 'completed', 'content': [{'type': 'output_text', 'text': '{bad'}]}]}
        for transport in (Mock(return_value=malformed),
                          Mock(side_effect=ModelError('authentication', 'denied'))):
            record = verify_proposals(self.inp, [proposal()], self.provider(transport))
            self.assertEqual(record.status, 'failed')
            self.assertEqual(record.decisions, [])
            self.assertEqual(record.error, 'verification request or output validation failed')

    def test_verifier_schema_cannot_return_new_findings(self):
        response_with_finding = verify_response() | {'output': [{'type': 'message', 'role': 'assistant',
            'status': 'completed', 'content': [{'type': 'output_text', 'text': json.dumps(
                {'verdict': 'SUPPORTED', 'rationale': 'supported', 'evidence_references': [],
                 'confidence': 0.9, 'findings': [proposal().model_dump()]})}]}]}
        record = verify_proposals(self.inp, [proposal()], self.provider(Mock(return_value=response_with_finding)))
        self.assertEqual(record.status, 'failed')
        self.assertEqual(record.decisions, [])

    def test_verifier_only_receives_target_evidence_and_versioned_contract(self):
        transport = Mock(return_value=verify_response())
        verifier = self.provider(transport)
        target = proposal(evidence=['Related behavior is implemented in src/helpers.py.'])
        record = verify_proposals(self.inp, [target], verifier, context=self.context,
                                  static_analysis=self.static, signal_budget=1000)
        payload = transport.call_args.args[0]
        body = json.loads(payload['input'][0]['content'])
        evidence = json.loads(body['diff'])
        self.assertEqual(payload['instructions'], verification_prompt())
        self.assertIn('proposed_finding', evidence)
        self.assertEqual([x['file'] for x in evidence['repository_context']], ['src/helpers.py'])
        self.assertEqual(evidence['static_analysis_signals'][0]['test_id'], 'B307')
        self.assertNotIn('UNRELATED_CONTEXT_SENTINEL', json.dumps(payload))
        for forbidden in ('GOLDEN_LABEL_SENTINEL', 'BENCHMARK_DEFECT_SENTINEL', 'ReviewBench task id'):
            self.assertNotIn(forbidden, json.dumps(payload))
        self.assertEqual(record.request_sha256, sha256(json.dumps([sha256(json.dumps(payload, sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode())], separators=(',', ':')).encode()))

    def test_stable_proposal_identifiers(self):
        self.assertEqual(proposal_identifier(0, proposal()), proposal_identifier(0, proposal()))
        self.assertNotEqual(proposal_identifier(0, proposal()), proposal_identifier(1, proposal()))

    def test_full_reuses_tools_proposal_and_exports_supported_only(self):
        review_transport = Mock(return_value=response([proposal().model_dump()]))
        verify_transport = Mock(return_value=verify_response('SUPPORTED'))
        reviewer, verifier = self.provider(review_transport), self.provider(verify_transport)
        static = self.static.model_copy(update={'findings': []})
        run = run_full(self.inp, reviewer, verifier, context=self.context,
                       static_analysis=static, signal_budget=1000)
        self.assertEqual(review_transport.call_count, 1)
        self.assertEqual(verify_transport.call_count, 1)
        self.assertEqual(run.result.mode, 'full')
        self.assertEqual(run.result.findings, run.proposed_findings)
        self.assertEqual(run.proposed_findings, [proposal()])
        self.assertEqual(run.verification.proposal_ids, [proposal_identifier(0, proposal())])
        candidate = export_predictions(run)
        self.assertEqual(len(candidate['findings']), 1)
        self.assertNotIn('verification', json.dumps(candidate))

    def test_full_removes_uncertain_and_persists_verdict_separately(self):
        reviewer = self.provider(Mock(return_value=response([proposal().model_dump()])))
        verifier = self.provider(Mock(return_value=verify_response('UNCERTAIN')))
        run = run_full(self.inp, reviewer, verifier, context=self.context, static_analysis=self.static)
        self.assertEqual(run.result.findings, [])
        self.assertEqual(run.proposed_findings, [proposal()])
        self.assertEqual(run.verification.decisions[0].verdict, 'UNCERTAIN')
        self.assertEqual(export_predictions(run)['findings'], [])

    def test_full_removes_unsupported_finding(self):
        reviewer = self.provider(Mock(return_value=response([proposal().model_dump()])))
        verifier = self.provider(Mock(return_value=verify_response('UNSUPPORTED')))
        run = run_full(self.inp, reviewer, verifier, context=self.context, static_analysis=self.static)
        self.assertEqual(run.result.findings, [])
        self.assertEqual(run.verification.decisions[0].verdict, 'UNSUPPORTED')

    def test_full_bandit_failure_prevents_both_model_calls(self):
        reviewer_transport, verifier_transport = Mock(), Mock()
        reviewer, verifier = self.provider(reviewer_transport), self.provider(verifier_transport)
        failed_static = self.static.model_copy(update={'status': 'failed', 'error': 'bandit unavailable'})
        run = run_full(self.inp, reviewer, verifier, context=self.context,
                       static_analysis=failed_static)
        self.assertEqual(run.result.status, 'failed')
        self.assertIn('static_analysis_failed', run.result.error)
        self.assertIsNone(run.verification)
        reviewer_transport.assert_not_called()
        verifier_transport.assert_not_called()

    def test_verifier_failure_fails_full_result_not_mode_c_fallback(self):
        reviewer = self.provider(Mock(return_value=response([proposal().model_dump()])))
        verifier = self.provider(Mock(side_effect=ModelError('rate_limit', 'later')))
        run = run_full(self.inp, reviewer, verifier, context=self.context, static_analysis=self.static)
        self.assertEqual(run.result.status, 'failed')
        self.assertEqual(run.result.mode, 'full')
        self.assertEqual(run.result.findings, [])
        self.assertEqual(run.proposed_findings, [proposal()])
        self.assertEqual(run.verification.status, 'failed')
        with self.assertRaisesRegex(ValueError, 'failed review'):
            export_predictions(run)

    def test_cli_full_mode_acquires_snapshot_and_persists_verification(self):
        reviewer = self.provider(Mock(return_value=response([proposal().model_dump()])))
        verifier = self.provider(Mock(return_value=verify_response()))
        acquired = SimpleNamespace(review_input=self.inp, artifact_directory=Path('/tmp/input'),
                                   head_snapshot=object())
        with TemporaryDirectory() as temp, \
             patch.dict('os.environ', {'REVIEW_PROVIDER': 'openai', 'REVIEW_MODEL': 'gpt-6-luna',
                                       'VERIFY_PROVIDER': 'openai', 'VERIFY_MODEL': 'gpt-6-luna',
                                       'OPENAI_API_KEY': 'offline-key'}, clear=True), \
             patch('reviewer.cli.OpenAIReviewer', side_effect=[reviewer, verifier]), \
             patch('reviewer.cli.GitHubClient') as github, \
             patch('reviewer.cli.prepare_context', return_value=self.context), \
             patch('reviewer.cli.prepare_static_analysis', return_value=self.static), \
             redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            config_path = Path(__file__).resolve().parents[1] / 'configs/experiment.toml'
            github.return_value.acquire.return_value = acquired
            status = cli_main(['review', '--repo', 'example/project', '--pr', '123', '--mode', 'full',
                               '--config', str(config_path), '--input-token-limit', '32000',
                               '--results-dir', str(Path(temp).resolve())])
            self.assertEqual(status, 0, stderr.getvalue())
            self.assertIn('Mode: full', stdout.getvalue())
            self.assertTrue(github.return_value.acquire.call_args.kwargs['snapshots'])
            saved = list(Path(temp).glob('*/full.json'))
            self.assertEqual(len(saved), 1)
            from reviewer.schemas import BaselineRun
            run = BaselineRun.model_validate_json(saved[0].read_bytes())
            self.assertEqual(run.result.findings, run.proposed_findings)
            self.assertEqual(run.verification.status, 'success')


if __name__ == '__main__':
    unittest.main()
