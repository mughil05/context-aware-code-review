"""Configuration diagnostics fail before acquisition or model execution."""
import io
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from reviewer.cli import main

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive = str(Path(self.temp.name) / 'missing.tar.gz')
        self.args = ['evaluate', '--benchmark', 'reviewbench', '--modes', 'baseline',
                     '--config', str(ROOT / 'configs/experiment.toml'),
                     '--benchmark-archive', self.archive,
                     '--subset', str(ROOT / 'configs/reviewbench-dev5.json'),
                     '--input-token-limit', '32000']

    def invoke(self, expected, *, args=None, key='offline-secret', extra_env=None):
        env = {'REVIEW_PROVIDER': 'openai', 'REVIEW_MODEL': 'gpt-6-luna',
               'OPENAI_API_KEY': key} | (extra_env or {})
        output = io.StringIO()
        with patch.dict(os.environ, env, clear=True), \
             patch('socket.socket', side_effect=AssertionError('network forbidden')), \
             patch('reviewer.evaluation.run_development_subset') as runner, \
             patch('reviewer.cli.GitHubClient') as client, \
             patch('sys.stderr', output):
            self.assertEqual(main(args or self.args), 2)
            runner.assert_not_called()
            client.assert_not_called()
        self.assertIn(f'failed: {expected}.', output.getvalue())
        self.assertNotIn('offline-secret', output.getvalue())
        self.assertNotIn('SECRET_SENTINEL', output.getvalue())

    def test_real_missing_archive_is_specific(self):
        self.invoke('archive_missing')

    def test_missing_subset_is_specific(self):
        args = self.args.copy()
        args[args.index('--subset') + 1] = str(Path(self.temp.name) / 'missing.json')
        self.invoke('subset_missing', args=args)

    def test_hash_and_revision_failures_are_not_bypassed(self):
        failures = {
            'subset manifest hash differs from pinned manifest': 'subset_manifest_hash_mismatch',
            'benchmark manifest hash mismatch': 'manifest_hash_mismatch',
            'golden file hash mismatch': 'golden_hash_mismatch',
            'unexpected archive root': 'archive_revision_mismatch',
            'SECRET_SENTINEL': 'invalid_benchmark_data',
        }
        for message, code in failures.items():
            with self.subTest(code=code), patch('reviewer.benchmark.load_benchmark', side_effect=ValueError(message)):
                self.invoke(code)

    def test_invalid_budget_and_environment_are_safe(self):
        for budget in ('0', '-1'):
            args = self.args.copy()
            args[-1] = budget
            self.invoke('invalid_input_budget', args=args)
        self.invoke('invalid_model_configuration', extra_env={'REVIEW_PROVIDER': 'SECRET_SENTINEL'})

    def test_missing_key_and_bad_pricing_are_specific(self):
        with patch('reviewer.benchmark.load_benchmark', return_value=object()):
            self.invoke('missing_api_key', key='')
            self.invoke('missing_api_key', key='SECRET_SENTINEL\n')
            self.invoke('invalid_pricing_configuration',
                        args=self.args + ['--pricing', str(Path(self.temp.name) / 'missing-pricing.json')])


if __name__ == '__main__':
    unittest.main()
