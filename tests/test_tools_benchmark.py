"""A/B/C benchmark orchestration tests; never run real ReviewBench."""
from contextlib import redirect_stdout
import io
import json
from unittest import TestCase, main
from unittest.mock import Mock, patch

from reviewer.cli import main as cli_main
from reviewer.config import load_config
from reviewer.evaluation import run_development_subset
from reviewer.models import ModelError
from reviewer.schemas import BaselineRun
from reviewer.static_analysis import StaticAnalysisResult
import test_benchmark as fixtures
from test_benchmark import ROOT, SENTINEL
from test_retrieval import snapshot


class ToolsBenchmarkTests(TestCase):
    setUp = fixtures.BenchmarkTests.setUp
    load = fixtures.BenchmarkTests.load
    provider = fixtures.BenchmarkTests.provider
    client = fixtures.BenchmarkTests.client

    def snapshot_client(self):
        client = self.client()[0]
        def frozen(pr, **kwargs):
            return snapshot(self.root, {
                'src/a.py': 'new\n',
                'src/helpers.py': 'def new():\n    return 1\n',
                'golden/labels.py': SENTINEL
            }, pr.head, pr.head)
        client.frozen_head = Mock(side_effect=frozen)
        return client

    def test_abc_modes_share_frozen_input_prompt_model_and_separate_exports(self):
        """Modes A/B/C get identical diff, B/C get same context, only C gets static signals."""
        client, provider = self.snapshot_client(), self.provider()
        config = load_config(ROOT / 'configs/experiment.toml')

        with patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run') as mock_bandit:
            mock_bandit.return_value = Mock(returncode=0, stdout='{"results": [], "metrics": {}}')

            path, report = run_development_subset(
                self.benchmark, provider, client, self.root / 'results',
                modes=['baseline', 'retrieval', 'tools'],
                retrieval_config=config.retrieval,
                cache_dir=self.root / 'cache'
            )

        self.assertEqual(report['completed_reviews'], 15)  # 5 tasks × 3 modes
        self.assertEqual(provider._transport.call_count, 15)
        self.assertEqual(client.frozen_head.call_count, 5)  # Snapshot once per task
        self.assertEqual(mock_bandit.call_count, 5)  # Static analysis once per task

        # Check each mode has separate candidate exports
        for mode in ['baseline', 'retrieval', 'tools']:
            candidates = list((path.parent / 'modes' / mode / 'candidates').glob('*.json'))
            self.assertEqual(len(candidates), 5)

        # Verify payloads: baseline gets diff only, retrieval gets diff+context, tools gets all three
        for i in range(0, 15, 3):
            baseline_call = provider._transport.call_args_list[i]
            retrieval_call = provider._transport.call_args_list[i + 1]
            tools_call = provider._transport.call_args_list[i + 2]

            baseline_payload = baseline_call.args[0]
            retrieval_payload = retrieval_call.args[0]
            tools_payload = tools_call.args[0]

            # All use same model and instructions
            self.assertEqual(baseline_payload['model'], retrieval_payload['model'])
            self.assertEqual(retrieval_payload['model'], tools_payload['model'])
            self.assertEqual(baseline_payload['instructions'], tools_payload['instructions'])

            # Parse input content
            baseline_data = json.loads(baseline_payload['input'][0]['content'])
            retrieval_data = json.loads(retrieval_payload['input'][0]['content'])
            tools_data = json.loads(tools_payload['input'][0]['content'])

            # All modes get same diff
            self.assertEqual(baseline_data['diff'], retrieval_data['diff'])
            self.assertEqual(retrieval_data['diff'], tools_data['diff'])

            # Only baseline lacks context
            self.assertEqual(set(baseline_data.keys()), {'diff'})

            # Retrieval and tools get same context
            self.assertTrue(retrieval_data['repository_context'])
            self.assertEqual(retrieval_data['repository_context'], tools_data['repository_context'])

            # Only tools gets static signals
            self.assertNotIn('static_analysis_signals', baseline_data)
            self.assertNotIn('static_analysis_signals', retrieval_data)
            self.assertIn('static_analysis_signals', tools_data)

            # No benchmark label leakage
            self.assertNotIn(SENTINEL, json.dumps(baseline_payload))
            self.assertNotIn(SENTINEL, json.dumps(retrieval_payload))
            self.assertNotIn(SENTINEL, json.dumps(tools_payload))

    def test_tools_mode_persists_static_analysis_in_result(self):
        """Tools mode results include static analysis provenance."""
        client, provider = self.snapshot_client(), self.provider()
        config = load_config(ROOT / 'configs/experiment.toml')

        with patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run') as mock_bandit:
            mock_bandit.return_value = Mock(returncode=0, stdout='{"results": [], "metrics": {"_totals": {"loc": 42}}}')

            path, report = run_development_subset(
                self.benchmark, provider, client, self.root / 'results',
                modes=['baseline', 'tools'],
                retrieval_config=config.retrieval,
                cache_dir=self.root / 'cache'
            )

        # Check one tools result
        tools_record = [r for r in report['tasks'] if r['mode'] == 'tools'][0]
        run = BaselineRun.model_validate_json((path.parent / tools_record['result_file']).read_bytes())

        self.assertEqual(run.result.mode, 'tools')
        self.assertIsNotNone(run.static_analysis)
        self.assertEqual(run.static_analysis['tool'], 'bandit')
        self.assertEqual(run.static_analysis['version'], '1.7.5')
        self.assertGreater(run.static_analysis_latency_seconds, 0)
        self.assertEqual(run.static_analysis['signals_sent'], [])

        candidate_path = path.parent / 'modes/tools/candidates' / (tools_record['task_id'] + '.json')
        candidate = json.loads(candidate_path.read_text())
        serialized_candidate = json.dumps(candidate)
        self.assertNotIn('static_analysis', serialized_candidate)
        self.assertNotIn('signals_sent', serialized_candidate)
        self.assertNotIn(SENTINEL, serialized_candidate)

        # Baseline result has no static analysis
        baseline_record = [r for r in report['tasks'] if r['mode'] == 'baseline'][0]
        baseline_run = BaselineRun.model_validate_json((path.parent / baseline_record['result_file']).read_bytes())
        self.assertIsNone(baseline_run.static_analysis)

    def test_cli_accepts_tools_in_comma_separated_modes(self):
        """CLI benchmark evaluation accepts --modes baseline,retrieval,tools."""
        config = load_config(ROOT / 'configs/experiment.toml')
        config.cache_dir = str(self.root / 'cache')
        provider = self.provider()

        with patch.dict('os.environ', {'REVIEW_PROVIDER': 'openai', 'REVIEW_MODEL': 'gpt-6-luna'}, clear=True), \
             patch('reviewer.cli.load_config', return_value=config), \
             patch('reviewer.benchmark.load_benchmark', return_value=self.benchmark), \
             patch('reviewer.cli.OpenAIReviewer', return_value=provider), \
             patch('reviewer.cli.GitHubClient', return_value=self.snapshot_client()), \
             patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run') as mock_bandit, \
             redirect_stdout(io.StringIO()) as output:

            mock_bandit.return_value = Mock(returncode=0, stdout='{"results": [], "metrics": {}}')

            code = cli_main([
                'evaluate', '--benchmark', 'reviewbench',
                '--modes', 'baseline,retrieval,tools',
                '--benchmark-archive', str(self.archive),
                '--input-token-limit', '32000',
                '--results-dir', str(self.root / 'results')
            ])

        self.assertEqual(code, 0)
        self.assertIn('15/15', output.getvalue())  # 5 tasks × 3 modes

    def test_tools_preflight_failure_prevents_model_calls(self):
        """If static analysis setup fails during preflight, no model calls are made."""
        client, provider = self.snapshot_client(), self.provider()
        config = load_config(ROOT / 'configs/experiment.toml')

        with patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run', side_effect=Exception('bandit failed')):
            _, report = run_development_subset(
                self.benchmark, provider, client, self.root / 'results',
                modes=['baseline', 'tools'],
                retrieval_config=config.retrieval,
                cache_dir=self.root / 'cache'
            )

        self.assertEqual(report['status'], 'failed')
        # May have made some baseline calls before hitting tools failure
        self.assertEqual(provider._transport.call_count, 0)
        self.assertEqual(report['tasks'][0]['mode'], 'tools')
        self.assertEqual(report['tasks'][0]['error_code'], 'static_analysis_failed')

    def test_tools_resume_explicitly_unsupported(self):
        """Tools mode resume is explicitly unsupported like retrieval."""
        config = load_config(ROOT / 'configs/experiment.toml')
        provider = self.provider()

        with self.assertRaisesRegex(ValueError, 'resume is not implemented'):
            run_development_subset(
                self.benchmark, provider, self.client()[0], self.root / 'results',
                modes=['tools'], retrieval_config=config.retrieval,
                resume_from=self.root
            )

        provider._transport.assert_not_called()

    def test_full_verifies_exact_tools_proposals_without_second_review_call(self):
        """C and D share each task's one generated tools proposal batch."""
        client, provider = self.snapshot_client(), self.provider()
        verifier_transport = Mock(side_effect=[self.verification_response() for _ in range(5)])
        verifier = self.provider()
        verifier._transport = verifier_transport
        config = load_config(ROOT / 'configs/experiment.toml')
        with patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run') as mock_bandit:
            mock_bandit.return_value = Mock(returncode=0, stdout='{"results": [], "metrics": {}}')
            path, report = run_development_subset(
                self.benchmark, provider, client, self.root / 'results',
                modes=['tools', 'full'], retrieval_config=config.retrieval,
                static_config=config.static_analysis, cache_dir=self.root / 'cache',
                verifier=verifier, verify_model='gpt-6-luna')

        self.assertEqual(provider._transport.call_count, 5)  # No second reviewer generation for D.
        self.assertEqual(verifier_transport.call_count, 5)
        self.assertEqual(mock_bandit.call_count, 5)
        self.assertEqual(report['completed_reviews'], 10)
        for task in self.benchmark.subset.tasks:
            records = [r for r in report['tasks'] if r['task_id'] == task.task_id]
            tools_record = next(r for r in records if r['mode'] == 'tools')
            full_record = next(r for r in records if r['mode'] == 'full')
            tools_path = path.parent / tools_record['result_file']
            full_path = path.parent / full_record['result_file']
            tools_run = BaselineRun.model_validate_json(tools_path.read_bytes())
            full_run = BaselineRun.model_validate_json(full_path.read_bytes())
            self.assertEqual(full_run.proposed_findings, tools_run.result.findings)
            self.assertEqual(full_run.result.findings, tools_run.result.findings)
            candidate = json.loads((path.parent / full_record['prediction_file']).read_text())
            self.assertEqual(len(candidate['findings']), 1)
            self.assertNotIn('verification', json.dumps(candidate))

    @staticmethod
    def verification_response(verdict='SUPPORTED'):
        raw = json.dumps({'verdict': verdict, 'rationale': 'The diff supports this defect.',
                          'evidence_references': ['src/a.py:1'], 'confidence': 0.9})
        return {'id': 'verify-fixture', 'model': 'gpt-6-luna', 'status': 'completed',
                'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                            'content': [{'type': 'output_text', 'text': raw}]}],
                'usage': {'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30,
                          'input_tokens_details': {'cached_tokens': 0},
                          'output_tokens_details': {'reasoning_tokens': 2}}}

    def test_full_verifier_failure_fails_observation_and_has_no_candidate(self):
        client, provider = self.snapshot_client(), self.provider()
        verifier = self.provider(responses=[ModelError('authentication', 'denied')])
        config = load_config(ROOT / 'configs/experiment.toml')
        with patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run') as mock_bandit:
            mock_bandit.return_value = Mock(returncode=0, stdout='{"results": [], "metrics": {}}')
            path, report = run_development_subset(
                self.benchmark, provider, client, self.root / 'results', modes=['full'],
                retrieval_config=config.retrieval, static_config=config.static_analysis,
                cache_dir=self.root / 'cache', verifier=verifier, verify_model='gpt-6-luna')
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(provider._transport.call_count, 1)
        self.assertEqual(verifier._transport.call_count, 1)
        full_record = report['tasks'][0]
        self.assertEqual(full_record['mode'], 'full')
        self.assertEqual(full_record['status'], 'failed')
        run_file = path.parent / full_record['result_file']
        full_run = BaselineRun.model_validate_json(run_file.read_bytes())
        self.assertEqual(full_run.result.status, 'failed')
        self.assertEqual(len(full_run.proposed_findings), 1)
        self.assertFalse((path.parent / 'modes/full/candidates' /
                          (full_record['task_id'] + '.json')).exists())

    def test_full_only_hides_internal_tools_run_and_exports_no_intermediate_candidate(self):
        client, provider = self.snapshot_client(), self.provider()
        verifier = self.provider(responses=[self.verification_response() for _ in range(5)])
        config = load_config(ROOT / 'configs/experiment.toml')
        with patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run') as mock_bandit:
            mock_bandit.return_value = Mock(returncode=0, stdout='{"results": [], "metrics": {}}')
            path, report = run_development_subset(
                self.benchmark, provider, client, self.root / 'results', modes=['full'],
                retrieval_config=config.retrieval, static_config=config.static_analysis,
                cache_dir=self.root / 'cache', verifier=verifier, verify_model='gpt-6-luna')
        self.assertEqual(len(report['tasks']), 5)
        self.assertTrue(all(record['mode'] == 'full' for record in report['tasks']))
        self.assertEqual(provider._transport.call_count, 5)
        self.assertEqual(verifier._transport.call_count, 5)
        self.assertEqual(report['model_attempts'], 10)
        self.assertFalse((path.parent / 'modes/tools/candidates').exists())
        self.assertFalse((path.parent / 'modes/tools/runs').exists())
        self.assertEqual(len(list((path.parent / 'modes/full/candidates').glob('*.json'))), 5)


if __name__ == '__main__':
    main()
