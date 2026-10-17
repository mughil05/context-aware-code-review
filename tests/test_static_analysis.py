"""Offline tests for Bandit static analysis integration."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from reviewer.config import load_config
from reviewer.cli import main as cli_main
from reviewer.pipeline import prepare_static_analysis, run_review, review_prompt
from reviewer.static_analysis import analyze_snapshot, get_bandit_version, run_bandit, StaticAnalysisResult, BanditFinding
from reviewer.schemas import BaselineRun
from test_retrieval import snapshot, DIFF, FILES
from test_schemas import pr_data


class BanditAdapterTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_bandit_unavailable_returns_specific_status(self):
        """When Bandit is not installed, return tool_unavailable status."""
        result = run_bandit(self.root, ['test.py'], timeout_seconds=60, head='a' * 40, archive_sha256='b' * 64)
        self.assertEqual(result.status, 'tool_unavailable')
        self.assertEqual(result.tool, 'bandit')
        self.assertIsNone(result.version)
        self.assertEqual(result.findings, [])
        self.assertIn('not found', result.error)

    def test_zero_findings_returns_no_findings_status(self):
        """When Bandit runs but finds nothing, return no_findings status."""
        with patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout='{"results": [], "metrics": {}}')
            result = run_bandit(self.root / 'src', ['test.py'], timeout_seconds=60, head='a' * 40, archive_sha256='b' * 64)
            self.assertEqual(result.status, 'no_findings')
            self.assertEqual(result.version, '1.7.5')
            self.assertEqual(result.findings, [])
            self.assertIsNone(result.error)

    def test_bandit_json_parsing(self):
        """Parse representative Bandit JSON output into internal schema."""
        bandit_output = {
            "results": [
                {
                    "filename": str(self.root / "src/auth.py"),
                    "test_id": "B105",
                    "test_name": "hardcoded_password_string",
                    "issue_severity": "LOW",
                    "issue_confidence": "MEDIUM",
                    "issue_text": "Possible hardcoded password: 'password'",
                    "line_number": 42,
                    "line_range": [42],
                    "more_info": "https://bandit.readthedocs.io/en/latest/plugins/b105_hardcoded_password_string.html"
                }
            ],
            "metrics": {"_totals": {"loc": 100}}
        }

        with patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout=json.dumps(bandit_output))
            result = run_bandit(self.root, ['src/auth.py'], timeout_seconds=60, head='a' * 40, archive_sha256='b' * 64)

            self.assertEqual(result.status, 'success')
            self.assertEqual(len(result.findings), 1)
            finding = result.findings[0]
            self.assertEqual(finding.test_id, 'B105')
            self.assertEqual(finding.file, 'src/auth.py')
            self.assertEqual(finding.line_number, 42)
            self.assertEqual(finding.issue_severity, 'LOW')
            self.assertEqual(finding.issue_confidence, 'MEDIUM')

    def test_malformed_json_output_fails_gracefully(self):
        """When Bandit returns invalid JSON, return failed status."""
        with patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout='not json')
            result = run_bandit(self.root, ['test.py'], timeout_seconds=60, head='a' * 40, archive_sha256='b' * 64)
            self.assertEqual(result.status, 'failed')
            self.assertIn('not valid JSON', result.error)

    def test_timeout_handling(self):
        """When Bandit times out, return timeout status."""
        with patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run', side_effect=Exception('timeout')):
            result = run_bandit(self.root, ['test.py'], timeout_seconds=1, head='a' * 40, archive_sha256='b' * 64)
            self.assertEqual(result.status, 'failed')

    def test_payload_respects_budget(self):
        """StaticAnalysisResult.payload() fits findings within budget."""
        findings = [
            BanditFinding(
                test_id=f'B{i:03d}',
                test_name='test',
                issue_severity='HIGH',
                issue_confidence='HIGH',
                issue_text='A' * 500,
                file='test.py',
                line_number=i,
                line_range=[i],
                more_info='http://example.com'
            )
            for i in range(1, 101)
        ]

        result = StaticAnalysisResult(
            tool='bandit',
            version='1.7.5',
            head='a' * 40,
            archive_sha256='b' * 64,
            analyzed_files=['test.py'],
            config={},
            timeout_seconds=60,
            status='success',
            error=None,
            findings=findings,
            metrics=None
        )

        small_budget = 500
        payload = result.payload(small_budget)
        self.assertGreater(len(findings), len(payload))
        serialized_size = len(json.dumps({'static_analysis_signals': payload}).encode())
        self.assertLessEqual(serialized_size, small_budget)


class StaticAnalysisIntegrationTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        from reviewer.schemas import ReviewInput, PRMetadata
        pr_meta = PRMetadata.model_validate(pr_data())
        self.inp = ReviewInput(pr=pr_meta, diff=DIFF, diff_complete=True)
        self.snap = snapshot(self.root, FILES, self.inp.pr.head, 'head-snap')
        self.config = load_config('configs/experiment.toml')

    def test_analyzes_changed_and_context_python_files_only(self):
        """Static analysis runs on changed Python files plus selected context files."""
        from reviewer.retrieval import retrieve
        from reviewer.static_analysis import analyze_snapshot

        context = retrieve(self.inp, self.snap, self.config.retrieval)

        with patch('reviewer.static_analysis.run_bandit') as mock_bandit:
            mock_bandit.return_value = StaticAnalysisResult(
                tool='bandit',
                version='1.7.5',
                head=self.inp['pr']['head'],
                archive_sha256=self.snap.archive_sha256,
                analyzed_files=[],
                config={},
                timeout_seconds=60,
                status='no_findings',
                error=None,
                findings=[],
                metrics=None
            )

            analyze_snapshot(self.inp, self.snap, context, timeout_seconds=60, budget=2000)

            analyzed = mock_bandit.call_args.args[1]
            # Should analyze changed file src/pkg/main.py
            self.assertIn('src/pkg/main.py', analyzed)
            # Should also analyze context files that are Python
            self.assertIn('src/pkg/users.py', analyzed)

    def test_excludes_benchmark_artifacts_and_credentials(self):
        """Static analysis never analyzes benchmark labels or credential files."""
        files_with_artifacts = FILES | {
            'golden/labels.json': '{}',
            'credentials/secrets.py': 'API_KEY = "secret"',
            'results/output.py': 'x = 1',
        }
        snap = snapshot(self.root, files_with_artifacts, self.inp['pr']['head'], 'snap-artifacts')

        from reviewer.retrieval import retrieve
        from reviewer.static_analysis import analyze_snapshot

        context = retrieve(self.inp, snap, self.config.retrieval)

        with patch('reviewer.static_analysis.run_bandit') as mock_bandit:
            mock_bandit.return_value = StaticAnalysisResult(
                tool='bandit', version='1.7.5', head=self.inp['pr']['head'],
                archive_sha256=snap.archive_sha256, analyzed_files=[], config={},
                timeout_seconds=60, status='no_findings', error=None, findings=[], metrics=None
            )

            analyze_snapshot(self.inp, snap, context, timeout_seconds=60, budget=2000)

            analyzed = mock_bandit.call_args.args[1]
            self.assertNotIn('golden/labels.json', analyzed)
            self.assertNotIn('credentials/secrets.py', analyzed)
            self.assertNotIn('results/output.py', analyzed)

    def test_static_analysis_evidence_cannot_expand_finding_anchors(self):
        """Findings must still anchor to diff-represented HEAD lines, not static analysis locations."""
        from reviewer.models import OpenAIReviewer
        from test_baseline import provider as make_provider

        context = SimpleNamespace(
            head=self.inp['pr']['head'],
            diff_sha256='a' * 64,
            chunks=[],
            payload=lambda: []
        )

        static = StaticAnalysisResult(
            tool='bandit',
            version='1.7.5',
            head=self.inp['pr']['head'],
            archive_sha256='b' * 64,
            analyzed_files=['src/pkg/users.py'],
            config={},
            timeout_seconds=60,
            status='success',
            error=None,
            findings=[
                BanditFinding(
                    test_id='B105',
                    test_name='hardcoded_password',
                    issue_severity='HIGH',
                    issue_confidence='HIGH',
                    issue_text='Hardcoded password detected',
                    file='src/pkg/users.py',  # This file is NOT in the diff
                    line_number=1,
                    line_range=[1],
                    more_info='http://example.com'
                )
            ],
            metrics=None
        )

        # Try to create a finding anchored to the static analysis location (not in diff)
        bad_finding = {
            'title': 'Hardcoded password',
            'category': 'security',
            'severity': 'high',
            'file': 'src/pkg/users.py',  # Not in diff
            'start_line': 1,
            'end_line': 1,
            'description': 'Based on static analysis',
            'evidence': ['Bandit B105'],
            'suggested_fix': 'Remove hardcoded password',
            'confidence': 0.9
        }

        provider = make_provider([bad_finding])
        run = run_review(self.inp, provider, mode='tools', context=context,
                        static_analysis=static, static_analysis_latency_seconds=1.0)

        self.assertEqual(run.result.status, 'failed')
        self.assertIn('invalid_reference', run.result.error)


class ToolsModeTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        from reviewer.schemas import ReviewInput, PRMetadata
        pr_meta = PRMetadata.model_validate(pr_data())
        self.inp = ReviewInput(pr=pr_meta, diff=DIFF, diff_complete=True)
        self.snap = snapshot(self.root, FILES, self.inp.pr.head, 'head-snap')
        self.config = load_config('configs/experiment.toml')

    def provider(self, findings=None):
        from test_baseline import provider as make_provider
        return make_provider(findings or [])

    def test_tools_mode_requires_both_context_and_static_analysis(self):
        """Tools mode must receive retrieval context AND static analysis."""
        provider = self.provider()

        with self.assertRaisesRegex(ValueError, 'tools mode requires both'):
            run_review(self.inp, provider, mode='tools', context=None, static_analysis=None)

    def test_tools_mode_receives_same_diff_and_context_as_retrieval(self):
        """Mode B and C get identical diff + retrieval context; only C gets static signals."""
        from reviewer.retrieval import retrieve

        context = retrieve(self.inp, self.snap, self.config.retrieval)
        static = StaticAnalysisResult(
            tool='bandit', version='1.7.5', head=self.inp['pr']['head'],
            archive_sha256=self.snap.archive_sha256, analyzed_files=[],
            config={}, timeout_seconds=60, status='no_findings',
            error=None, findings=[], metrics=None
        )

        provider_b = self.provider()
        provider_c = self.provider()

        run_b = run_review(self.inp, provider_b, mode='retrieval', context=context)
        run_c = run_review(self.inp, provider_c, mode='tools', context=context, static_analysis=static)

        # Both modes get same diff
        payload_b = provider_b._transport.call_args.args[0]
        payload_c = provider_c._transport.call_args.args[0]

        data_b = json.loads(payload_b['input'][0]['content'])
        data_c = json.loads(payload_c['input'][0]['content'])

        self.assertEqual(data_b['diff'], data_c['diff'])
        self.assertEqual(data_b.get('repository_context'), data_c.get('repository_context'))

        # Only tools mode gets static signals
        self.assertNotIn('static_analysis_signals', data_b)
        self.assertIn('static_analysis_signals', data_c)

    def test_tools_mode_persists_static_analysis_provenance(self):
        """Tools mode results include static analysis result and latency."""
        from reviewer.retrieval import retrieve

        context = retrieve(self.inp, self.snap, self.config.retrieval)
        static = StaticAnalysisResult(
            tool='bandit', version='1.7.5', head=self.inp['pr']['head'],
            archive_sha256=self.snap.archive_sha256, analyzed_files=['src/pkg/main.py'],
            config={'format': 'json'}, timeout_seconds=60, status='no_findings',
            error=None, findings=[], metrics={'_totals': {'loc': 50}}
        )

        provider = self.provider()
        run = run_review(self.inp, provider, mode='tools', context=context,
                        static_analysis=static, static_analysis_latency_seconds=2.5)

        self.assertEqual(run.result.mode, 'tools')
        self.assertIsNotNone(run.static_analysis)
        self.assertEqual(run.static_analysis['tool'], 'bandit')
        self.assertEqual(run.static_analysis['version'], '1.7.5')
        self.assertEqual(run.static_analysis_latency_seconds, 2.5)

        # Baseline/retrieval modes have None
        run_baseline = run_review(self.inp, self.provider(), mode='baseline')
        self.assertIsNone(run_baseline.static_analysis)

    def test_prompt_version_updated_for_static_signals(self):
        """Review prompt mentions static_analysis_signals for all modes."""
        prompt = review_prompt()
        self.assertIn('version: 3', prompt)
        self.assertIn('static_analysis_signals', prompt)
        self.assertIn('automated security/defect scanners', prompt)

    def test_cli_tools_mode_integration(self):
        """CLI accepts --mode tools and orchestrates retrieval + static analysis."""
        acquired = SimpleNamespace(
            review_input=self.inp,
            artifact_directory=self.root,
            head_snapshot=self.snap
        )
        provider = self.provider()
        self.config.cache_dir = str(self.root / 'cache')

        with patch('reviewer.static_analysis.get_bandit_version', return_value='1.7.5'), \
             patch('reviewer.static_analysis.subprocess.run') as mock_bandit, \
             patch.dict('os.environ', {'REVIEW_PROVIDER': 'openai', 'REVIEW_MODEL': 'gpt-6-luna'}, clear=True), \
             patch('reviewer.cli.load_config', return_value=self.config), \
             patch('reviewer.cli.GitHubClient') as client, \
             patch('reviewer.cli.OpenAIReviewer', return_value=provider), \
             redirect_stdout(io.StringIO()):

            mock_bandit.return_value = Mock(returncode=0, stdout='{"results": [], "metrics": {}}')
            client.return_value.acquire.return_value = acquired

            code = cli_main(['review', '--repo', 'o/r', '--pr', '1', '--mode', 'tools',
                           '--input-token-limit', '32000', '--results-dir', str(self.root / 'results')])

        self.assertEqual(code, 0)
        self.assertTrue(mock_bandit.called)


if __name__ == '__main__':
    main()
