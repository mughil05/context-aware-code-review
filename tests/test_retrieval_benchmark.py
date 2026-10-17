"""A/B orchestration tests on synthetic tasks; never run the real ReviewBench set."""
from contextlib import redirect_stdout
import io
import json
from unittest import TestCase, main
from unittest.mock import Mock, patch

from reviewer.cli import main as cli_main
from reviewer.config import load_config
from reviewer.evaluation import run_development_subset
from reviewer.schemas import BaselineRun
import test_benchmark as fixtures
from test_benchmark import ROOT, SENTINEL
from test_retrieval import snapshot


class RetrievalBenchmarkTests(TestCase):
    setUp = fixtures.BenchmarkTests.setUp
    load = fixtures.BenchmarkTests.load
    provider = fixtures.BenchmarkTests.provider
    client = fixtures.BenchmarkTests.client

    def snapshot_client(self):
        client = self.client()[0]
        def frozen(pr, **kwargs):
            return snapshot(self.root, {'src/a.py': 'new\n', 'src/helpers.py': 'def new():\n    return 1\n',
                                         'golden/labels.py': SENTINEL}, pr.head, pr.head)
        client.frozen_head = Mock(side_effect=frozen)
        return client

    def test_paired_modes_share_frozen_input_prompt_model_and_separate_exports(self):
        client, provider = self.snapshot_client(), self.provider()
        config = load_config(ROOT / 'configs/experiment.toml')
        path, report = run_development_subset(self.benchmark, provider, client, self.root / 'results',
                                              modes=['baseline', 'retrieval'], retrieval_config=config.retrieval,
                                              cache_dir=self.root / 'cache')
        self.assertEqual(report['completed_reviews'], 10)
        self.assertEqual(provider._transport.call_count, 10)
        self.assertEqual(client.frozen_head.call_count, 5)
        for mode in ['baseline', 'retrieval']:
            self.assertEqual(len(list((path.parent / 'modes' / mode / 'candidates').glob('*.json'))), 5)
        for i in range(0, 10, 2):
            a, b = (provider._transport.call_args_list[j].args[0] for j in (i, i + 1))
            self.assertEqual(a['model'], b['model'])
            self.assertEqual(a['instructions'], b['instructions'])
            self.assertEqual(a['text'], b['text'])
            da, db = (json.loads(p['input'][0]['content']) for p in (a, b))
            self.assertEqual(set(da), {'diff'})
            self.assertEqual(da['diff'], db['diff'])
            self.assertTrue(db['repository_context'])
            self.assertNotIn(SENTINEL, json.dumps(a) + json.dumps(b))
        for record in report['tasks']:
            run = BaselineRun.model_validate_json((path.parent / record['result_file']).read_bytes())
            self.assertEqual(run.result.mode, record['mode'])
            self.assertEqual(run.retrieval is not None, record['mode'] == 'retrieval')
            candidate = json.loads((path.parent / record['prediction_file']).read_bytes())
            self.assertEqual(candidate['agent'], 'context-aware-code-review/' + record['mode'])
            self.assertEqual(candidate['pr'], record['pr'])
            self.assertEqual(set(candidate), {'pr', 'agent', 'findings', 'usage'})

    def test_cli_accepts_comma_separated_modes(self):
        config = load_config(ROOT / 'configs/experiment.toml')
        config.cache_dir = str(self.root / 'cache')
        provider = self.provider()
        with patch.dict('os.environ', {'REVIEW_PROVIDER': 'openai', 'REVIEW_MODEL': 'gpt-6-luna'}, clear=True), \
                patch('reviewer.cli.load_config', return_value=config), \
                patch('reviewer.benchmark.load_benchmark', return_value=self.benchmark), \
                patch('reviewer.cli.OpenAIReviewer', return_value=provider), \
                patch('reviewer.cli.GitHubClient', return_value=self.snapshot_client()), redirect_stdout(io.StringIO()) as output:
            code = cli_main(['evaluate', '--benchmark', 'reviewbench', '--modes', 'baseline,retrieval',
                             '--benchmark-archive', str(self.archive), '--input-token-limit', '32000',
                             '--results-dir', str(self.root / 'results')])
        self.assertEqual(code, 0)
        self.assertIn('10/10', output.getvalue())

    def test_retrieval_preflight_failure_prevents_every_model_call(self):
        client, provider = self.snapshot_client(), self.provider()
        client.frozen_head.side_effect = ValueError('frozen snapshot unavailable')
        config = load_config(ROOT / 'configs/experiment.toml')
        _, report = run_development_subset(self.benchmark, provider, client, self.root / 'results',
                                           modes=['baseline', 'retrieval'], retrieval_config=config.retrieval,
                                           cache_dir=self.root / 'cache')
        self.assertEqual(report['status'], 'failed')
        provider._transport.assert_not_called()

    def test_retrieval_resume_is_explicitly_unsupported(self):
        config = load_config(ROOT / 'configs/experiment.toml')
        provider = self.provider()
        with self.assertRaisesRegex(ValueError, 'resume is not implemented'):
            run_development_subset(self.benchmark, provider, self.client()[0], self.root / 'results',
                                   modes=['retrieval'], retrieval_config=config.retrieval, resume_from=self.root)
        provider._transport.assert_not_called()


if __name__ == '__main__':
    main()
