"""Resume acceptance and rejection tests with synthetic archives and model mocks."""
from contextlib import redirect_stdout
import io
import json
from unittest import TestCase, main
from unittest.mock import Mock, patch

from reviewer.benchmark import export_predictions
from reviewer.cli import main as cli_main
from reviewer.evaluation import run_development_subset
from reviewer.models import OpenAIReviewer
from reviewer.resume import ResumeError, validate_resume
from reviewer.schemas import BaselineRun
import test_benchmark as fixtures
from test_benchmark import ROOT, SENTINEL, response, finding


class ResumeTests(TestCase):
    setUp = fixtures.BenchmarkTests.setUp
    load = fixtures.BenchmarkTests.load
    provider = fixtures.BenchmarkTests.provider
    client = fixtures.BenchmarkTests.client

    def interrupted(self):
        incomplete = response()
        incomplete.update(status='incomplete', output=[],
                          incomplete_details={'reason': 'max_output_tokens'})
        provider = self.provider([response([finding()]), incomplete])
        path, report = run_development_subset(self.benchmark, provider, self.client()[0], self.root / 'results')
        self.assertEqual(report['model_attempts'], 2)
        self.assertEqual(report['completed_reviews'], 1)
        return path.parent

    def resume(self, source, provider=None, **kwargs):
        provider = provider or self.provider()
        client = Mock()
        client.frozen_input.side_effect = AssertionError('resume must use saved inputs')
        path, report = run_development_subset(self.benchmark, provider, client, self.root / 'results',
                                             resume_from=source, **kwargs)
        client.frozen_input.assert_not_called()
        return path, report, provider

    def test_resume_skips_success_preserves_bytes_and_continues_four_tasks(self):
        source = self.interrupted()
        original = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
        provider = self.provider()
        provider.config.max_output_tokens = 8192
        path, report, provider = self.resume(source, provider, retry_failed=True, allow_output_budget_change=True)
        self.assertNotEqual(path.parent, source)
        self.assertEqual(report['completed_reviews'], 5)
        self.assertEqual(report['reused_reviews'], 1)
        self.assertEqual(report['model_attempts'], 4)
        self.assertEqual(report['recorded_model_attempts_including_history'], 6)
        self.assertEqual(provider._transport.call_count, 4)
        self.assertEqual(original, {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()})
        first = report['tasks'][0]
        self.assertEqual((path.parent / first['result_file']).read_bytes(), (source / first['result_file']).read_bytes())
        self.assertEqual(first['settings']['max_output_tokens'], 4096)
        self.assertEqual(len(report['tasks'][1]['previous_results']), 1)
        self.assertEqual(report['tasks'][1]['previous_results'][0]['status'], 'failed')
        self.assertEqual(len(list((path.parent / 'candidates').glob('*.json'))), 5)
        for record in report['tasks']:
            raw = BaselineRun.model_validate_json((path.parent / record['result_file']).read_bytes())
            self.assertEqual(json.loads((path.parent / record['prediction_file']).read_bytes()), export_predictions(raw))
            if not record['reused']:
                self.assertEqual(raw.settings['max_output_tokens'], 8192)
        for call in provider._transport.call_args_list:
            payload = call.args[0]
            self.assertEqual(payload['max_output_tokens'], 8192)
            self.assertEqual(set(json.loads(payload['input'][0]['content'])), {'diff'})
            self.assertNotIn(SENTINEL, json.dumps(payload))

    def test_failed_retry_requires_explicit_flag(self):
        source = self.interrupted()
        provider = self.provider()
        with self.assertRaisesRegex(ResumeError, 'retry-failed'):
            self.resume(source, provider)
        provider._transport.assert_not_called()

    def test_output_increase_requires_flag_and_other_settings_remain_fixed(self):
        source = self.interrupted()
        for attr, value, allow in [('max_output_tokens', 8192, False),
                                    ('max_output_tokens', 2048, True),
                                    ('input_token_limit', 64000, True),
                                    ('timeout_seconds', 120.0, True)]:
            with self.subTest(attr=attr, value=value):
                provider = self.provider()
                setattr(provider.config, attr, value)
                with self.assertRaises(ResumeError):
                    self.resume(source, provider, retry_failed=True, allow_output_budget_change=allow)
                provider._transport.assert_not_called()

    def test_model_prompt_and_subset_mismatches_reject_before_calls(self):
        source = self.interrupted()
        provider = self.provider()
        provider.model = 'other-model'
        with self.assertRaises(ResumeError): self.resume(source, provider, retry_failed=True)
        provider._transport.assert_not_called()
        provider = self.provider()
        with patch('reviewer.resume.review_prompt', return_value='changed prompt'):
            with self.assertRaises(ResumeError): self.resume(source, provider, retry_failed=True)
        provider._transport.assert_not_called()
        self.benchmark.subset.tasks[0].pr.head = '0'*40
        with self.assertRaises(ResumeError): self.resume(source, provider, retry_failed=True)
        provider._transport.assert_not_called()

    def test_tampered_raw_schema_request_hash_and_candidate_are_rejected(self):
        source = self.interrupted()
        key = self.task.task_id
        targets = [
            (f'runs/{key}/baseline.json', lambda d: d['settings'].update(temperature=1.0)),
            (f'runs/{key}/baseline.json', lambda d: d.update(request_sha256='0'*64)),
            (f'runs/{key}/baseline.json', lambda d: d.update(output_schema={})),
            (f'runs/{key}/baseline.json', lambda d: d['result']['pr'].update(head='0'*40)),
            (f'inputs/{key}.json', lambda d: d.update(diff=d['diff']+'\n')),
            (f'candidates/{key}.json', lambda d: d.update(findings=[])),
            ('summary.json', lambda d: d.update(benchmark_revision='0'*40)),
            ('summary.json', lambda d: d.update(mode='retrieval')),
        ]
        for relative, mutate in targets:
            with self.subTest(relative=relative):
                path = source / relative
                before = path.read_bytes()
                data = json.loads(before)
                mutate(data)
                path.write_text(json.dumps(data))
                try:
                    provider = self.provider()
                    with self.assertRaises(ResumeError): self.resume(source, provider, retry_failed=True)
                    provider._transport.assert_not_called()
                finally:
                    path.write_bytes(before)

    def test_failed_retry_stops_batch_and_retains_history(self):
        source = self.interrupted()
        incomplete = response()
        incomplete.update(status='incomplete', output=[], incomplete_details={'reason': 'max_output_tokens'})
        path, report, provider = self.resume(source, self.provider([incomplete]), retry_failed=True)
        self.assertEqual(provider._transport.call_count, 1)
        self.assertEqual(report['completed_reviews'], 1)
        self.assertEqual(report['tasks'][1]['status'], 'failed')
        self.assertEqual(report['tasks'][2]['status'], 'ready')
        self.assertEqual(len(list((path.parent / 'candidates').glob('*.json'))), 1)
        path2, report2, provider2 = self.resume(path.parent, retry_failed=True)
        self.assertEqual(report2['completed_reviews'], 5)
        self.assertEqual(len(report2['tasks'][1]['previous_results']), 2)
        self.assertEqual(provider2._transport.call_count, 4)

    def test_completed_run_resume_makes_zero_calls(self):
        source, _ = run_development_subset(self.benchmark, self.provider(), self.client()[0], self.root / 'results')
        _, report, provider = self.resume(source.parent)
        self.assertEqual(report['reused_reviews'], 5)
        self.assertEqual(report['model_attempts'], 0)
        provider._transport.assert_not_called()

    def test_missing_candidate_or_inconsistent_ready_state_fails_closed(self):
        source = self.interrupted()
        candidate = source / 'candidates' / (self.task.task_id + '.json')
        candidate.unlink()  # Synthetic fixture only; real run artifacts are untouched.
        with self.assertRaises(ResumeError): self.resume(source, retry_failed=True)

    def test_unknown_attempt_on_ready_task_is_not_replayed(self):
        source = self.interrupted()
        key = self.benchmark.subset.tasks[2].task_id
        directory = source / 'runs' / key
        directory.mkdir()
        (directory / 'baseline.json').write_text('{}')
        provider = self.provider()
        with self.assertRaises(ResumeError): self.resume(source, provider, retry_failed=True)
        provider._transport.assert_not_called()

    def test_corrupt_inherited_history_rejects_before_calls(self):
        source = self.interrupted()
        path, _, _ = self.resume(source, retry_failed=True)
        summary = json.loads(path.read_bytes())
        summary['tasks'][1]['previous_results'][0]['result_sha256'] = '0'*64
        path.write_text(json.dumps(summary))
        provider = self.provider()
        with self.assertRaises(ResumeError): self.resume(path.parent, provider)
        provider._transport.assert_not_called()

    def test_legacy_artifacts_without_new_metadata_are_accepted(self):
        source = self.interrupted()
        summary = json.loads((source / 'summary.json').read_bytes())
        summary.pop('review_settings')
        summary.pop('prompt_sha256')
        for record in summary['tasks']:
            record.pop('result_sha256', None)
            for a in record.get('attempts', []):
                a.pop('response_status', None)
                a.pop('incomplete_details', None)
            if record.get('result_file'):
                path = source / record['result_file']
                raw = json.loads(path.read_bytes())
                for a in raw['attempts']:
                    a.pop('response_status', None)
                    a.pop('incomplete_details', None)
                path.write_text(json.dumps(raw))
        (source / 'summary.json').write_text(json.dumps(summary))
        _, report, provider = self.resume(source, retry_failed=True)
        self.assertEqual(report['completed_reviews'], 5)
        self.assertEqual(provider._transport.call_count, 4)

    def test_cli_output_override_and_resume_reach_shared_runner(self):
        source = self.interrupted()
        transport = Mock(return_value=response())
        def factory(model, config):
            return OpenAIReviewer(model, config, api_key='offline-secret', transport=transport)
        with patch.dict('os.environ', {'REVIEW_PROVIDER': 'openai', 'REVIEW_MODEL': 'gpt-6-luna'}, clear=True), \
                patch('reviewer.benchmark.load_benchmark', return_value=self.benchmark), \
                patch('reviewer.cli.OpenAIReviewer', side_effect=factory), \
                patch('reviewer.cli.GitHubClient') as client, redirect_stdout(io.StringIO()) as out:
            code = cli_main(['evaluate', '--benchmark', 'reviewbench', '--modes', 'baseline',
                             '--benchmark-archive', str(self.archive), '--input-token-limit', '32000',
                             '--max-output-tokens', '8192', '--resume-from', str(source),
                             '--retry-failed', '--allow-output-budget-change',
                             '--results-dir', str(self.root / 'results'),
                             '--config', str(ROOT / 'configs/experiment.toml')])
        self.assertEqual(code, 0)
        self.assertIn('Reused: 1; new model attempts: 4', out.getvalue())
        self.assertEqual(transport.call_count, 4)
        self.assertTrue(all(c.args[0]['max_output_tokens'] == 8192 for c in transport.call_args_list))
        client.return_value.frozen_input.assert_not_called()


if __name__ == '__main__':
    main()
