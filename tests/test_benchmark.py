"""Offline benchmark integration fixtures. No real defects or scores are invented."""
from copy import deepcopy
import io
import json
from pathlib import Path
import socket
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from reviewer.benchmark import (MANIFEST_SHA256, REVISION, Subset, digest, export_predictions,
                                load_benchmark, qualifying_count, read_json, reconstruct)
from reviewer.config import load_config
from reviewer.evaluation import Pricing, cost_estimate, run_development_subset
from reviewer.github import AcquisitionError, DIFF_MEDIA, GitHubClient
from reviewer.models import ModelError, OpenAIReviewer
from reviewer.pipeline import run_baseline
from reviewer.schemas import ReviewInput

ROOT = Path(__file__).resolve().parents[1]
DIFF = 'diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+new\n'
SENTINEL = 'GOLDEN_LABEL_DO_NOT_SEND_76543'


def fixture_bundle(root):
    """Five real task identities; small, explicitly synthetic offline content."""
    subset = json.loads((ROOT / 'configs/reviewbench-dev5.json').read_text())
    manifest, goldens = [], {}
    for i, t in enumerate(subset['tasks']):
        t.update(files_changed=1, lines_added=1, lines_removed=1, diff_sha256=digest(DIFF.encode()))
        row = t['pr'] | {'nwo': t['pr']['repo'].removeprefix('https://github.com/'),
                         'language': 'Python', 'files_changed': 1, 'lines_added': 1, 'lines_removed': 1,
                         'title': SENTINEL, 'body': SENTINEL}
        manifest.append(row)
        golden = {'pr_key': t['task_id'], 'pr': t['pr'], 'findings': [] if i == 1 else [
            {'file': 'src/a.py', 'start_line': 1, 'end_line': 1, 'message': SENTINEL,
             'tp_fp': 'tp', 'category': 'correctness', 'severity': 'high', 'scope': 'introduced-by-pr',
             'source': {'type': 'synthetic-offline-fixture'}}]}
        raw = json.dumps(golden).encode()
        goldens['golden/' + t['task_id'] + '.json'] = raw
        t['golden_sha256'] = digest(raw)
    manifest_raw = json.dumps(manifest).encode()
    subset['manifest_sha256'] = digest(manifest_raw)
    subset_path = root / 'subset.json'
    subset_path.write_text(json.dumps(subset))
    members = {'corpus/manifest.json': manifest_raw} | goldens
    archive = root / 'benchmark.tar.gz'
    write_archive(archive, members)
    return archive, subset_path, members, subset


def write_archive(path, members):
    with tarfile.open(path, 'w:gz') as tar:
        folder = tarfile.TarInfo('ReviewBench-' + REVISION)
        folder.type = tarfile.DIRTYPE
        tar.addfile(folder)
        for name, raw in members.items():
            m = tarfile.TarInfo('ReviewBench-' + REVISION + '/' + name)
            m.size = len(raw)
            tar.addfile(m, io.BytesIO(raw))


def response(findings=None):
    return {'id': 'resp_offline', 'model': 'gpt-6-luna', 'status': 'completed',
            'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': json.dumps({'findings': findings or []})}]}],
            'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
                      'input_tokens_details': {'cached_tokens': 40},
                      'output_tokens_details': {'reasoning_tokens': 5}}}


def finding():
    return {'title': 'Synthetic fixture', 'category': 'correctness', 'severity': 'high',
            'file': 'src/a.py', 'start_line': 1, 'end_line': 1, 'description': 'Fixture description',
            'evidence': ['Fixture evidence'], 'suggested_fix': 'Fixture fix', 'confidence': 0.9}


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.archive, self.subset_path, self.members, self.selection = fixture_bundle(self.root)
        self.benchmark = self.load()
        self.task = self.benchmark.subset.tasks[0]
        self.socket_guard = patch.object(socket, 'create_connection', side_effect=AssertionError('offline tests'))
        self.socket_guard.start()
        self.addCleanup(self.socket_guard.stop)

    def load(self):
        return load_benchmark(self.archive, self.subset_path,
                              expected_manifest_sha256=self.selection['manifest_sha256'])

    def provider(self, responses=None):
        config = load_config(ROOT / 'configs/experiment.toml').review
        config.request_byte_budget = 32000
        transport = Mock(return_value=response([finding()]))
        if responses is not None:
            transport.side_effect = responses
        return OpenAIReviewer('gpt-6-luna', config, api_key='offline-secret', transport=transport)

    def client(self):
        def download(url, **kwargs):
            for task in self.benchmark.subset.tasks:
                if task.pr.head not in url:
                    continue
                if '/commits/' in url:
                    return json.dumps({'sha': task.pr.head}).encode()
                if kwargs['headers']['Accept'] == DIFF_MEDIA:
                    return DIFF.encode()
                return json.dumps({'base_commit': {'sha': task.pr.base},
                                   'merge_base_commit': {'sha': task.pr.base},
                                   'files': [{'filename': 'src/a.py', 'status': 'modified',
                                              'additions': 1, 'deletions': 1}]}).encode()
            raise AssertionError('unexpected endpoint')
        downloader = Mock(side_effect=download)
        return GitHubClient(token='', downloader=downloader), downloader

    def test_real_selection_is_exactly_five_python_tasks(self):
        real = Subset.model_validate_json((ROOT / 'configs/reviewbench-dev5.json').read_bytes())
        self.assertEqual(real.manifest_sha256, MANIFEST_SHA256)
        self.assertEqual([(t.pr.repo.split('github.com/')[1], t.pr.pr_number) for t in real.tasks], [
            ('AA-Factory/aafactory-prototype', 17), ('Tzeusy/butlers', 1021),
            ('allan-mobley-jr/gimmes', 468), ('YasserShkeir/django-smart-ratelimit', 25),
            ('ZhuoZhuoCrayon/throttled-py', 135)])
        self.assertTrue(all(t.source_repository.startswith('review-bench/') for t in real.tasks))

    def test_load_identity_reference_and_zero_qualifying_case(self):
        self.assertEqual(len(self.benchmark.subset.tasks), 5)
        self.assertEqual(self.benchmark.golden[self.task.task_id]['pr'], self.task.pr.model_dump())
        self.assertEqual(qualifying_count(self.benchmark.golden[self.task.task_id]), 1)
        self.assertEqual(qualifying_count(self.benchmark.golden[self.benchmark.subset.tasks[1].task_id]), 0)
        self.assertNotIn(SENTINEL, self.task.model_dump_json())

    def test_missing_bad_hash_and_malformed_data(self):
        for changed in ('missing', 'hash', 'json'):
            with self.subTest(changed=changed):
                members = dict(self.members)
                key = 'golden/' + self.task.task_id + '.json'
                if changed == 'missing': del members[key]
                elif changed == 'hash': members[key] += b' '
                else: members['corpus/manifest.json'] = b'{'
                write_archive(self.archive, members)
                with self.assertRaises(ValueError): self.load()
        write_archive(self.archive, self.members)
        with self.assertRaises(ValueError):
            load_benchmark(self.archive, self.subset_path)  # Synthetic manifest cannot impersonate real corpus.
        with self.assertRaises(ValueError): read_json('{"a":1,"a":2}')
        with self.assertRaises(ValueError): read_json('{"a":NaN}')

    def test_subset_duplicates_and_metadata_mismatch(self):
        selection = deepcopy(self.selection)
        selection['tasks'][1] = selection['tasks'][0]
        with self.assertRaises(ValueError): Subset.model_validate(selection)
        selection = deepcopy(self.selection)
        selection['tasks'][0]['lines_added'] = 2
        self.subset_path.write_text(json.dumps(selection))
        with self.assertRaisesRegex(ValueError, 'metadata'): self.load()

    def test_golden_validation_and_original_labels_are_preserved(self):
        key = 'golden/' + self.task.task_id + '.json'
        original = json.loads(self.members[key])
        negative = deepcopy(original['findings'][0])
        negative['tp_fp'] = 'fp'
        original['findings'].append(negative)
        self.members[key] = json.dumps(original).encode()
        self.selection['tasks'][0]['golden_sha256'] = digest(self.members[key])
        self.subset_path.write_text(json.dumps(self.selection))
        write_archive(self.archive, self.members)
        loaded = self.load()
        self.assertEqual(loaded.golden[self.task.task_id], original)
        self.assertEqual(qualifying_count(original), 1)
        original['findings'][0]['start_line'] = '1'
        self.members[key] = json.dumps(original).encode()
        self.selection['tasks'][0]['golden_sha256'] = digest(self.members[key])
        self.subset_path.write_text(json.dumps(self.selection))
        write_archive(self.archive, self.members)
        with self.assertRaisesRegex(ValueError, 'coordinates'): self.load()

    def test_archive_selected_symlink_rejected_without_extraction(self):
        with tarfile.open(self.archive, 'w:gz') as tar:
            link = tarfile.TarInfo('ReviewBench-' + REVISION + '/corpus/manifest.json')
            link.type = tarfile.SYMTYPE
            link.linkname = '/etc/passwd'
            tar.addfile(link)
        with self.assertRaises(ValueError): self.load()

    def test_frozen_reconstruction_uses_only_shas_and_mirror(self):
        client, transport = self.client()
        inp = reconstruct(self.task, client)
        self.assertIsInstance(inp, ReviewInput)
        self.assertEqual(inp.pr, self.task.pr)
        self.assertEqual(inp.diff_base, self.task.pr.base)
        self.assertEqual(inp.diff, DIFF)
        for call in transport.call_args_list:
            url = call.args[0]
            self.assertIn(self.task.pr.head, url)
            self.assertIn('/repos/' + self.task.source_repository + '/', url)
            self.assertNotIn('/pulls/', url)
        self.assertEqual(len(transport.call_args_list), 3)

    def test_reconstruction_rejects_unavailable_mismatched_and_incomplete(self):
        client, transport = self.client()
        transport.side_effect = AcquisitionError('not_found', 'missing frozen commit')
        with self.assertRaisesRegex(AcquisitionError, 'not_found'): reconstruct(self.task, client)
        for change in ('base', 'head', 'inventory', 'diff'):
            client, transport = self.client()
            original = transport.side_effect
            def altered(url, **kwargs):
                raw = original(url, **kwargs)
                if kwargs['headers']['Accept'] == DIFF_MEDIA:
                    return raw + b' ' if change == 'diff' else raw
                value = json.loads(raw)
                if '/commits/' in url and change == 'head': value['sha'] = 'f' * 40
                if '/compare/' in url:
                    if change == 'base': value['base_commit']['sha'] = 'f' * 40
                    if change == 'inventory': value['files'] = []
                return json.dumps(value).encode()
            transport.side_effect = altered
            with self.subTest(change=change), self.assertRaises(AcquisitionError): reconstruct(self.task, client)

    def test_export_prediction_format_and_zero_finding(self):
        inp = reconstruct(self.task, self.client()[0])
        for count in (0, 1):
            provider = self.provider([response([finding()] if count else [])])
            run = run_baseline(inp, provider)
            exported = export_predictions(run)
            self.assertEqual(exported['pr'], self.task.pr.model_dump())
            self.assertEqual(len(exported['findings']), count)
            self.assertEqual(exported['usage']['total_tokens'], 120)
            if count:
                f = exported['findings'][0]
                self.assertEqual(set(f), {'producer', 'file', 'start_line', 'end_line', 'message'})
                self.assertIn('Fixture evidence', f['message'])
                self.assertNotIn('severity', f)

    def test_five_runs_export_and_label_leakage_guard(self):
        provider = self.provider()
        path, report = run_development_subset(self.benchmark, provider, self.client()[0], self.root / 'results')
        self.assertEqual(report['status'], 'generation_complete_scoring_paused')
        self.assertEqual(report['completed_reviews'], 5)
        self.assertEqual(report['model_attempts'], 5)
        self.assertIsNone(report['quality_metrics'])
        self.assertEqual(len(list((path.parent / 'candidates').glob('*.json'))), 5)
        for call in provider._transport.call_args_list:
            payload = call.args[0]
            self.assertEqual(json.loads(payload['input'][0]['content']), {'diff': DIFF})
            self.assertNotIn(SENTINEL, json.dumps(payload))
            self.assertNotIn('offline-secret', json.dumps(payload))
            self.assertNotIn('tools', payload)
        first = report['tasks'][0]
        self.assertEqual(first['qualifying_tp_count'], 1)
        self.assertIn(SENTINEL, (path.parent / 'reference' / (self.task.task_id + '.json')).read_text())
        raw = json.loads((path.parent / first['result_file']).read_text())
        self.assertEqual(raw['diff_sha256'], self.task.diff_sha256)
        self.assertIsNone(first['approximate_cost_usd'])
        path2, _ = run_development_subset(self.benchmark, provider, self.client()[0], self.root / 'results')
        self.assertNotEqual(path, path2)

    def test_all_preflight_finishes_before_any_model_attempt(self):
        provider = self.provider()
        client, transport = self.client()
        original = transport.side_effect
        def failing(url, **kwargs):
            if self.benchmark.subset.tasks[-1].pr.head in url:
                raise AcquisitionError('not_found', 'frozen state missing')
            return original(url, **kwargs)
        transport.side_effect = failing
        path, report = run_development_subset(self.benchmark, provider, client, self.root / 'results')
        provider._transport.assert_not_called()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['error_code'], 'not_found')
        self.assertTrue(path.exists())

    def test_failure_is_not_exported_as_clean_review_and_no_retry(self):
        provider = self.provider([response(), ModelError('rate_limit', 'offline fixture')])
        path, report = run_development_subset(self.benchmark, provider, self.client()[0], self.root / 'results')
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['completed_reviews'], 1)
        self.assertEqual(provider._transport.call_count, 2)
        self.assertEqual(len(list((path.parent / 'candidates').glob('*.json'))), 1)
        self.assertEqual(report['tasks'][1]['status'], 'failed')

    def test_budget_and_retry_policy_fail_before_model_spend(self):
        provider = self.provider()
        provider.config.request_byte_budget = 1
        _, report = run_development_subset(self.benchmark, provider, self.client()[0], self.root / 'results')
        self.assertEqual(report['error_code'], 'request_byte_budget_exceeded')
        provider._transport.assert_not_called()
        provider.config.max_attempts = 2
        with self.assertRaisesRegex(ValueError, 'max_attempts'):
            run_development_subset(self.benchmark, provider, self.client()[0], self.root / 'other')

    def test_pricing_is_explicit_and_reasoning_not_double_counted(self):
        run = run_baseline(reconstruct(self.task, self.client()[0]), self.provider())
        price = Pricing(model='gpt-6-luna', input_usd_per_million=0.1, output_usd_per_million=0.5,
                        cached_input_usd_per_million=0.01, source='offline-test', as_of='fixture')
        self.assertAlmostEqual(cost_estimate(run, price), (60 * .1 + 40 * .01 + 20 * .5) / 1e6)
        self.assertIsNone(cost_estimate(run, None))
        price.model = 'another-model'
        with self.assertRaises(ValueError): cost_estimate(run, price)

    def test_cli_runs_shared_baseline_and_announces_paused_scoring(self):
        from reviewer.cli import main
        from contextlib import redirect_stdout
        provider = self.provider()
        with patch.dict('os.environ', {'REVIEW_PROVIDER': 'openai', 'REVIEW_MODEL': 'gpt-6-luna'}, clear=True), \
                patch('reviewer.benchmark.load_benchmark', return_value=self.benchmark), \
                patch('reviewer.cli.OpenAIReviewer', return_value=provider), \
                patch('reviewer.cli.GitHubClient', return_value=self.client()[0]), \
                redirect_stdout(io.StringIO()) as out:
            code = main(['evaluate', '--benchmark', 'reviewbench', '--modes', 'baseline',
                         '--benchmark-archive', str(self.archive), '--input-token-limit', '32000',
                         '--results-dir', str(self.root / 'results'), '--config', str(ROOT / 'configs/experiment.toml')])
        self.assertEqual(code, 0)
        self.assertIn('Scoring paused', out.getvalue())
        self.assertEqual(provider._transport.call_count, 5)


if __name__ == '__main__':
    unittest.main()
