"""Offline checks for final manifest, scoped ReviewBench adaptation, and preflight."""
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from reviewer.benchmark import (MANIFEST_SHA256, REVISION, Subset, digest, final_selection,
                                generate_final_subset, load_benchmark, read_json)
from reviewer.config import load_config
from reviewer.evaluation import run_development_subset
from reviewer.preflight import PreflightFailure, _write_atomic, run_preflight
from reviewer.scoped_evaluation import aggregate_scoped, scoped_golden
from reviewer.schemas import RepositoryContext, ReviewInput
from reviewer.static_analysis import StaticAnalysisResult
from reviewer.models import OpenAIReviewer, build_review_payload
import test_benchmark as fixtures
from test_github import diagnosed_benchmark_diff

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / '.cache/reviewbench' / ('ReviewBench-' + REVISION + '.tar.gz')


class FinalEvaluationTests(unittest.TestCase):
    def _preflight_fixture(self, root, *, purpose='development-only', reconstruct_side_effect=None):
        archive, subset_path, _, _ = fixtures.fixture_bundle(root)
        raw_subset = read_json(subset_path.read_bytes())
        benchmark = load_benchmark(archive, subset_path,
                                   expected_manifest_sha256=raw_subset['manifest_sha256'])
        if purpose != benchmark.subset.purpose:
            benchmark = replace(benchmark, subset=benchmark.subset.model_copy(update={'purpose': purpose}))
        config = load_config(ROOT / 'configs/experiment.toml')
        config.review.request_byte_budget = 32000
        config.review.max_output_tokens = 8192
        config.retrieval.context_byte_budget = 6000
        context = Mock()
        context.payload.return_value = [{'file': 'src/helper.py', 'start_line': 1, 'end_line': 1,
                                         'content': 'def helper(): pass'}]
        context.model_dump.return_value = {'ranking': 'structural_lexical_v1', 'chunks': []}
        client = Mock()
        client.frozen_head.side_effect = lambda pr, **kwargs: Mock(commit=pr.head,
                                               archive_sha256='a'*64, repository='review-bench/test')
        def static_for(inp, *args):
            return StaticAnalysisResult(tool='bandit', version='1.7.8',
                head=inp.pr.head, archive_sha256='a'*64, analyzed_files=['src/a.py'],
                config={'signal_byte_budget': 2000}, timeout_seconds=20,
                status='no_findings', error=None, findings=[], metrics={}, signals_sent=[])
        def reconstruct_for(task, client):
            if reconstruct_side_effect is not None:
                return reconstruct_side_effect(task, client)
            return ReviewInput(pr=task.pr, diff=fixtures.DIFF, diff_complete=True)
        return archive, subset_path, benchmark, config, client

    def test_final20_is_mechanical_deterministic_and_excludes_dev5(self):
        raw = generate_final_subset(ARCHIVE, ROOT / 'configs/reviewbench-dev5.json')
        expected = (ROOT / 'configs/reviewbench-final20.json').read_bytes()
        self.assertEqual(raw, expected)
        data = read_json(raw)
        self.assertEqual(len(data['tasks']), 20)
        self.assertEqual(data['benchmark_revision'], REVISION)
        self.assertEqual(data['manifest_sha256'], MANIFEST_SHA256)
        self.assertEqual(digest(raw), '206e5ba692eb31c56980224d2105dff015e1ed3d448b930b5ce0eb090d16b189')
        dev_ids = {task['task_id'] for task in read_json((ROOT / 'configs/reviewbench-dev5.json').read_bytes())['tasks']}
        final_ids = [task['task_id'] for task in data['tasks']]
        self.assertEqual(len(set(final_ids)), 20)
        self.assertTrue(dev_ids.isdisjoint(final_ids))
        benchmark = load_benchmark(ARCHIVE, ROOT / 'configs/reviewbench-final20.json', allow_unpinned_diffs=True)
        self.assertEqual(len(benchmark.golden), 20)
        self.assertTrue(all(task.diff_sha256 is None for task in benchmark.subset.tasks))
        with self.assertRaisesRegex(ValueError, 'not frozen by preflight'):
            load_benchmark(ARCHIVE, ROOT / 'configs/reviewbench-final20.json')

    def test_final_manifest_lock_is_required_and_bound_to_task_shas(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            subset_path = root / 'subset.json'
            raw = (ROOT / 'configs/reviewbench-final20.json').read_bytes()
            subset_path.write_bytes(raw)
            lock = {'schema_version': 1, 'benchmark_revision': REVISION,
                    'subset_sha256': digest(raw),
                    'tasks': [{'task_id': task['task_id'], 'base_sha': task['pr']['base'],
                               'head_sha': task['pr']['head'], 'diff_sha256': 'd'*64,
                               'head_archive_sha256': 'a'*64}
                              for task in read_json(raw)['tasks']]}
            with self.assertRaisesRegex(ValueError, 'not frozen by preflight'):
                load_benchmark(ARCHIVE, subset_path)
            lock_path = subset_path.with_suffix('.lock.json')
            with self.assertRaisesRegex(ValueError, 'exclusion provenance'):
                lock_path.write_text(json.dumps(lock))
                load_benchmark(ARCHIVE, subset_path)
            lock_path.unlink()
            from reviewer.final_exclusions import exclusion_record
            from reviewer.benchmark import Task
            raw_data = read_json(raw)
            excluded = next(task for task in raw_data['tasks']
                            if task['task_id'] == 'nipoppy_nipoppy_964-9421a714')
            lock['tasks'] = [entry for entry in lock['tasks'] if entry['task_id'] != excluded['task_id']]
            lock.update(intended_task_count=20, evaluable_task_count=19,
                        pre_execution_exclusions=[exclusion_record(Task.model_validate(excluded))],
                        request_budget={'ceiling': 73728, 'safety_margin_bytes': 4096,
                                        'method': 'serialized_request_body_utf8_bytes_plus_4096_safety_margin'},
                        context_byte_budget=6000, static_signal_byte_budget=2000,
                        max_output_tokens=8192,
                        benchmark_archive_sha256=digest(ARCHIVE.read_bytes()),
                        benchmark_manifest_sha256=MANIFEST_SHA256)
            from reviewer.preflight import PayloadOnlyReviewer
            config = load_config(ROOT / 'configs/experiment.toml')
            config.review.request_byte_budget = 73728
            config.review.max_output_tokens = 8192
            reviewer = PayloadOnlyReviewer('gpt-6-luna', config.review)
            from reviewer.pipeline import review_prompt
            from reviewer.verification import verification_prompt, PROMPT_VERSION
            from reviewer.archives import sha256
            lock.update(created_at=datetime.now(timezone.utc).isoformat(),
                        review_settings={'request_byte_budget': 73728}, reviewer_provider='openai',
                        reviewer_model='gpt-6-luna', review_prompt_sha256=sha256(review_prompt().encode()),
                        review_prompt_version='v3', verifier_prompt_sha256=sha256(verification_prompt().encode()),
                        verifier_prompt_version=PROMPT_VERSION, verifier_provider='openai',
                        verifier_model='gpt-6-luna', verifier_settings={'request_byte_budget': 73728},
                        retrieval_config=config.retrieval.model_dump(),
                        static_analysis={'tool': 'bandit', 'version': '1.7.8', 'timeout_seconds': 60})
            lock_path.write_text(json.dumps(lock))
            loaded = load_benchmark(ARCHIVE, subset_path)
            self.assertEqual(loaded.diff_lock_sha256, digest(lock_path.read_bytes()))
            self.assertTrue(all(task.diff_sha256 == 'd'*64 for task in loaded.subset.tasks
                                if task.task_id != 'nipoppy_nipoppy_964-9421a714'))
            excluded_task = next(task for task in loaded.subset.tasks
                                 if task.task_id == 'nipoppy_nipoppy_964-9421a714')
            self.assertIsNone(excluded_task.diff_sha256)
            lock['tasks'][0]['head_sha'] = 'e'*40
            lock_path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, 'frozen identity'):
                load_benchmark(ARCHIVE, subset_path)

    def test_mechanical_selection_filters_language_size_and_dev_ids(self):
        rows = [
            {'repo': 'https://github.com/a/b', 'pr_number': 1, 'base': 'a'*40, 'head': 'b'*40,
             'language': 'Python', 'lines_added': 400, 'lines_removed': 600},
            {'repo': 'https://github.com/a/c', 'pr_number': 2, 'base': 'a'*40, 'head': 'c'*40,
             'language': 'Python', 'lines_added': 1001, 'lines_removed': 0},
            {'repo': 'https://github.com/a/d', 'pr_number': 3, 'base': 'a'*40, 'head': 'd'*40,
             'language': 'Java', 'lines_added': 1, 'lines_removed': 1},
        ]
        selected = final_selection(rows, {'unrelated_dev_task'})
        self.assertEqual([row['pr_number'] for row in selected], [1])
        self.assertEqual(final_selection(rows, {'unrelated_dev_task'}), selected)
        self.assertEqual(final_selection(rows, {'a_b_1-bbbbbbbb'}), [])

    def test_subset_schema_accepts_arbitrary_nonempty_size(self):
        dev = read_json((ROOT / 'configs/reviewbench-dev5.json').read_bytes())
        dev['tasks'] = dev['tasks'][:2]
        subset = Subset.model_validate(dev)
        self.assertEqual(len(subset.tasks), 2)
        dev['tasks'] = []
        with self.assertRaises(Exception):
            Subset.model_validate(dev)

    def test_scoped_golden_filters_category_and_scope_without_relabeling(self):
        golden = {'pr_key': 'task', 'findings': [
            {'tp_fp': 'tp', 'category': 'correctness', 'scope': 'introduced-by-pr', 'message': 'in'},
            {'tp_fp': 'tp', 'category': 'maintainability', 'scope': 'introduced-by-pr', 'message': 'category-out'},
            {'tp_fp': 'fp', 'category': 'security', 'scope': 'pre-existing', 'message': 'scope-out'},
            {'tp_fp': 'fp', 'category': 'performance', 'scope': 'exacerbated-by-pr', 'message': 'in-fp'},
        ]}
        filtered = scoped_golden(golden)
        self.assertEqual([f['message'] for f in filtered['findings']], ['in', 'in-fp'])
        self.assertEqual(filtered['findings'][1]['tp_fp'], 'fp')
        self.assertEqual(len(golden['findings']), 4)

    def test_scoped_metrics_unmatched_exclusions_f1_and_denominators(self):
        golden = {'pr_key': 'task', 'findings': [
            {'tp_fp': 'tp', 'category': 'correctness', 'scope': 'introduced-by-pr'},
            {'tp_fp': 'tp', 'category': 'testing', 'scope': 'introduced-by-pr'},
            {'tp_fp': 'fp', 'category': 'security', 'scope': 'pre-existing'},
            {'tp_fp': 'fp', 'category': 'performance', 'scope': 'exacerbated-by-pr'},
        ]}
        details = {'pr_key': 'task', 'candidate_findings': [
            {'status': 'matched_tp', 'matched_golden_index': 0, 'covered_golden_indices': [0],
             'golden_tp_fp': 'tp', 'golden_category': 'correctness', 'golden_scope': 'introduced-by-pr'},
            {'status': 'matched_fp', 'matched_golden_index': 1, 'covered_golden_indices': [],
             'golden_tp_fp': 'fp', 'golden_category': 'performance', 'golden_scope': 'exacerbated-by-pr'},
            {'status': 'novel_fp', 'classifier_tp_fp': 'fp',
             'classifier_category': 'security', 'classifier_scope': 'introduced-by-pr'},
            {'status': 'novel_tp', 'classifier_tp_fp': 'tp',
             'classifier_category': 'correctness', 'classifier_scope': 'pre-existing'},
        ]}
        # A zero-positive task still contributes to PR denominator and is valid.
        empty = {'pr_key': 'clean', 'findings': []}
        empty_details = {'pr_key': 'clean', 'candidate_findings': []}
        score = aggregate_scoped({'task': {'golden': golden, 'details': details},
                                  'clean': {'golden': empty, 'details': empty_details}},
                                 completed_task_ids=['task', 'clean'], failed_task_ids=['failed'])
        self.assertEqual(score['tp'], 1)  # novel out-of-scope TP is excluded
        self.assertEqual(score['fp'], 2)
        self.assertEqual(score['fn'], 0)
        self.assertEqual(score['excluded_golden_count'], 2)
        self.assertEqual(score['eligible_golden_tp'], 1)
        self.assertAlmostEqual(score['precision'], 1/3)
        self.assertEqual(score['grounded_precision'], 0.5)
        self.assertEqual(score['grounded_precision_denominator'], 2)
        self.assertEqual(score['recall'], 1)
        self.assertEqual(score['f1'], 0.5)
        self.assertEqual(score['false_positives_per_pr'], 1)
        self.assertEqual(score['completed_pr_count'], 2)
        self.assertEqual(score['failed_pr_count'], 1)
        self.assertEqual(score['not_attempted_pr_count'], 0)
        self.assertIn('classified unmatched', score['precision_denominator'])

    def test_scope_filter_and_judge_detail_file_adapter_round_trip(self):
        from scripts.reviewbench_scope import aggregate, filter_golden
        with TemporaryDirectory() as temp:
            root = Path(temp)
            task_id, clean_id = 'repo_1-' + 'a'*8, 'clean_2-' + 'b'*8
            pr = {'repo': 'https://github.com/o/r', 'pr_number': 1,
                  'base': 'a'*40, 'head': 'b'*40}
            task_manifest = {'benchmark_revision': REVISION, 'manifest_sha256': MANIFEST_SHA256,
                             'tasks': [{'task_id': task_id, 'pr': pr},
                                       {'task_id': clean_id, 'pr': {**pr, 'pr_number': 2}}]}
            manifest_path = root / 'tasks.json'
            manifest_path.write_text(json.dumps(task_manifest))
            golden_dir = root / 'golden'
            golden_dir.mkdir()
            for identifier, task_pr, findings in (
                    (task_id, pr, [{'file': 'a.py', 'start_line': 1, 'end_line': 1,
                                    'message': 'eligible', 'tp_fp': 'tp', 'category': 'correctness',
                                    'severity': 'high', 'scope': 'introduced-by-pr'},
                                   {'file': 'a.py', 'start_line': 2, 'end_line': 2,
                                    'message': 'excluded', 'tp_fp': 'tp', 'category': 'testing',
                                    'severity': 'low', 'scope': 'introduced-by-pr'}]),
                    (clean_id, {**pr, 'pr_number': 2}, [])):
                (golden_dir / f'{identifier}.json').write_text(json.dumps(
                    {'pr_key': identifier, 'pr': task_pr, 'findings': findings}))
            filtered_dir = root / 'filtered'
            counts = filter_golden(manifest_path, golden_dir, filtered_dir)
            self.assertEqual(counts[task_id]['eligible'], 1)
            self.assertEqual(counts[task_id]['excluded'], 1)
            detail_path, judge_summary_path = root / 'details.json', root / 'judge.json'
            detail_path.write_text(json.dumps([
                {'pr_key': task_id, 'candidate_findings': [
                    {'status': 'matched_tp', 'matched_golden_index': 0,
                     'covered_golden_indices': [0], 'golden_tp_fp': 'tp',
                     'golden_category': 'correctness', 'golden_scope': 'introduced-by-pr'}]},
                {'pr_key': clean_id, 'candidate_findings': []}]))
            judge_summary_path.write_text('{}')
            run_summary_path = root / 'run.json'
            run_summary_path.write_text(json.dumps({'subset_sha256': 'c'*64, 'tasks': [
                {'task_id': task_id, 'mode': 'baseline', 'status': 'success'},
                {'task_id': clean_id, 'mode': 'baseline', 'status': 'success'}]}))
            score = aggregate(detail_path, judge_summary_path, golden_dir, manifest_path,
                              run_summary_path, 'baseline')
            self.assertEqual(score['tp'], 1)
            self.assertEqual(score['fn'], 0)
            self.assertEqual(score['excluded_golden_count'], 1)
            self.assertEqual(score['reviewbench_judge_metrics_sha256'], digest(b'{}'))

    def test_missing_unmatched_scope_fails_closed(self):
        golden = {'pr_key': 'task', 'findings': []}
        details = {'pr_key': 'task', 'candidate_findings': [
            {'status': 'novel_fp', 'classifier_category': 'correctness'}]}
        with self.assertRaisesRegex(ValueError, 'lacks category/scope'):
            aggregate_scoped({'task': {'golden': golden, 'details': details}}, completed_task_ids=['task'])

    def test_zero_eligible_positive_denominator_is_reported_undefined(self):
        golden = {'pr_key': 'clean', 'findings': [
            {'tp_fp': 'tp', 'category': 'testing', 'scope': 'introduced-by-pr'}]}
        details = {'pr_key': 'clean', 'candidate_findings': []}
        score = aggregate_scoped({'clean': {'golden': golden, 'details': details}},
                                 completed_task_ids=['clean'])
        self.assertIsNone(score['recall'])
        self.assertIsNone(score['f1'])
        self.assertEqual(score['eligible_golden_tp'], 0)
        self.assertEqual(score['excluded_golden_count'], 1)

    def test_reviewbench_detail_patch_is_narrow_and_idempotent(self):
        from scripts.patch_reviewbench_scope_details import patch
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'eval.ts'
            member = f'ReviewBench-{REVISION}/scripts/eval/eval.ts'
            with tarfile.open(ARCHIVE, 'r:gz') as archive:
                path.write_bytes(archive.extractfile(member).read())
            self.assertTrue(patch(path))
            changed = path.read_text()
            self.assertIn('classifier_scope: cls.scope', changed)
            self.assertIn('golden_scope: g.scope', changed)
            self.assertFalse(patch(path))
            path.write_text('unrecognized source')
            with self.assertRaisesRegex(ValueError, 'differs from pinned'):
                patch(path)

    def test_preflight_has_no_model_transport_and_records_shared_inputs(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        archive, subset_path, _, _ = fixtures.fixture_bundle(root)
        raw_subset = read_json(subset_path.read_bytes())
        benchmark = load_benchmark(archive, subset_path,
                                   expected_manifest_sha256=raw_subset['manifest_sha256'])
        # Golden annotations may quote legitimate diff/context text. They must
        # never be treated as a blacklist of allowed source-controlled strings.
        for golden in benchmark.golden.values():
            if not golden['findings']:
                continue  # Preserve the fixture's zero-positive case.
            golden['findings'][0]['message'] = 'def helper(): pass'
            golden['findings'].append(golden['findings'][0] | {
                'message': fixtures.DIFF.splitlines()[-1]})
        config = load_config(ROOT / 'configs/experiment.toml')
        config.review.request_byte_budget = 32000
        config.review.max_output_tokens = 8192
        config.retrieval.context_byte_budget = 6000
        context = Mock()
        context.payload.return_value = [{'file': 'src/helper.py', 'start_line': 1, 'end_line': 1,
                                         'content': 'def helper(): pass'}]
        context.model_dump.return_value = {'ranking': 'structural_lexical_v1', 'chunks': []}
        client = Mock()
        client.frozen_head.side_effect = lambda pr, **kwargs: Mock(commit=pr.head,
                                               archive_sha256='a'*64, repository='review-bench/test')
        def static_for(inp, *args):
            return StaticAnalysisResult(tool='bandit', version='1.7.8',
                head=inp.pr.head, archive_sha256='a'*64, analyzed_files=['src/a.py'],
                config={'signal_byte_budget': 2000}, timeout_seconds=20,
                status='no_findings', error=None, findings=[], metrics={}, signals_sent=[])
        def reconstruct_for(task, client):
            return ReviewInput(pr=task.pr, diff=fixtures.DIFF, diff_complete=True)
        with patch('reviewer.preflight.load_benchmark', return_value=benchmark), \
             patch('reviewer.preflight.reconstruct', side_effect=reconstruct_for), \
             patch('reviewer.preflight.prepare_context', return_value=context), \
             patch('reviewer.preflight.prepare_static_analysis', side_effect=static_for), \
             patch('reviewer.preflight.get_bandit_version', return_value='1.7.8'), \
             patch('reviewer.preflight.build_review_payload', wraps=build_review_payload) as payload_builder, \
             patch.object(OpenAIReviewer, 'complete', side_effect=AssertionError('preflight made a model call')) as complete:
            path, report = run_preflight(archive=archive, subset_path=subset_path, config=config,
                client=client, results_dir=root / 'out', review_model='gpt-6-luna', review_provider='openai',
                verify_model='gpt-6-luna', verify_provider='openai', cache_dir=root / 'cache')
        self.assertTrue(path.exists())
        self.assertEqual(report['model_calls'], 0)
        self.assertEqual(len(report['tasks']), 5)
        self.assertTrue(all(task['benchmark_labels_sent'] is False for task in report['tasks']))
        self.assertTrue(all(set(task['request_byte_measurements']) == {'baseline', 'retrieval', 'tools', 'full'}
                            for task in report['tasks']))
        measurement = report['tasks'][0]['request_byte_measurements']['baseline']
        self.assertEqual(measurement['estimator_value'],
                         measurement['serialized_request_body_utf8_bytes'] + 4096)
        self.assertEqual(measurement['configured_estimator_ceiling'], 32000)
        self.assertEqual(measurement['estimator_units'], 'serialized_request_estimator_units')
        self.assertEqual(client.frozen_head.call_count, 5)
        self.assertEqual(payload_builder.call_count, 20)
        complete.assert_not_called()
        self.assertTrue(all(call.args[1].max_output_tokens == 8192 and call.args[2] == fixtures.DIFF
                            for call in payload_builder.call_args_list))

    def test_preflight_first_task_failure_persists_stage_error_and_no_lock(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            def fail_reconstruct(task, client):
                raise ValueError('frozen diff mismatch; api_key=supersecret')
            archive, subset_path, benchmark, config, client = self._preflight_fixture(
                root, purpose='final-evaluation', reconstruct_side_effect=fail_reconstruct)
            with patch('reviewer.preflight.load_benchmark', return_value=benchmark), \
                 patch('reviewer.preflight.reconstruct', side_effect=fail_reconstruct), \
                 patch('reviewer.preflight.get_bandit_version', return_value='1.7.8'), \
                 patch.dict('os.environ', {'OPENAI_API_KEY': 'supersecret'}):
                with self.assertRaises(PreflightFailure) as caught:
                    run_preflight(archive=archive, subset_path=subset_path, config=config,
                        client=client, results_dir=root / 'out', review_model='gpt-6-luna',
                        review_provider='openai', verify_model='gpt-6-luna', verify_provider='openai',
                        cache_dir=root / 'cache')
            failure = caught.exception
            saved = json.loads(failure.report_path.read_text())
            self.assertEqual(failure.task_id, benchmark.subset.tasks[0].task_id)
            self.assertEqual(failure.stage, 'acquisition')
            self.assertEqual(saved['tasks'][0]['status'], 'failed')
            self.assertEqual(saved['tasks'][0]['error'], 'frozen diff mismatch; api_key=[REDACTED]')
            self.assertNotIn('supersecret', failure.report_path.read_text())
            self.assertFalse(saved['tasks'][0]['acquisition_succeeded'])
            self.assertEqual(len(saved['tasks']), 1)
            self.assertFalse(subset_path.with_suffix('.lock.json').exists())

    def test_preflight_failure_preserves_successes_and_leaves_later_tasks_unattempted(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            calls = 0
            def fail_second(task, client):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ValueError('second task reconstruction failed')
                return ReviewInput(pr=task.pr, diff=fixtures.DIFF, diff_complete=True)
            archive, subset_path, benchmark, config, client = self._preflight_fixture(
                root, purpose='final-evaluation', reconstruct_side_effect=fail_second)
            with patch('reviewer.preflight.load_benchmark', return_value=benchmark), \
                 patch('reviewer.preflight.reconstruct', side_effect=fail_second), \
                 patch('reviewer.preflight.prepare_context') as prepare_context, \
                 patch('reviewer.preflight.prepare_static_analysis') as prepare_static, \
                 patch('reviewer.preflight.get_bandit_version', return_value='1.7.8'):
                context = Mock()
                context.payload.return_value = []
                context.model_dump.return_value = {'chunks': []}
                prepare_context.return_value = context
                prepare_static.return_value = StaticAnalysisResult(tool='bandit', version='1.7.8',
                    head=benchmark.subset.tasks[0].pr.head, archive_sha256='a'*64,
                    analyzed_files=['src/a.py'], config={'signal_byte_budget': 2000}, timeout_seconds=20,
                    status='no_findings', error=None, findings=[], metrics={}, signals_sent=[])
                with self.assertRaises(PreflightFailure) as caught:
                    run_preflight(archive=archive, subset_path=subset_path, config=config,
                        client=client, results_dir=root / 'out', review_model='gpt-6-luna',
                        review_provider='openai', verify_model='gpt-6-luna', verify_provider='openai',
                        cache_dir=root / 'cache')
            saved = json.loads(caught.exception.report_path.read_text())
            self.assertEqual([row['status'] for row in saved['tasks']], ['passed', 'failed'])
            self.assertEqual(saved['tasks'][0]['task_id'], benchmark.subset.tasks[0].task_id)
            self.assertEqual(saved['tasks'][1]['task_id'], benchmark.subset.tasks[1].task_id)
            self.assertEqual(saved['tasks'][0]['stage'], 'complete')
            self.assertEqual(saved['tasks'][1]['stage'], 'acquisition')
            self.assertFalse(subset_path.with_suffix('.lock.json').exists())
            self.assertEqual(calls, 2)

    def test_preflight_cli_surfaces_safe_failure_and_report(self):
        from argparse import Namespace
        from io import StringIO
        from types import SimpleNamespace
        from reviewer.cli import _preflight_main
        with TemporaryDirectory() as temp:
            report_path = Path(temp) / 'failed.json'
            error = PreflightFailure(stage='snapshot', task_id='task-1', code='snapshot_unavailable',
                                     message='frozen commit archive is unavailable', report_path=report_path)
            args = Namespace(config='unused', request_byte_budget=10, max_output_tokens=10,
                             context_byte_budget=10, benchmark='reviewbench',
                             benchmark_archive='unused', subset='unused', results_dir=temp)
            stream = StringIO()
            models = SimpleNamespace(review=SimpleNamespace(model='gpt-6-luna', provider='openai'),
                                     verify=SimpleNamespace(model='gpt-6-luna', provider='openai'))
            with patch('reviewer.config.resolve_models', return_value=models), patch('reviewer.cli.load_config'), \
                 patch('reviewer.preflight.run_preflight', side_effect=error), \
                 patch('sys.stderr', stream):
                status = _preflight_main(args, None)
            self.assertEqual(status, 1)
            self.assertIn('Task: task-1', stream.getvalue())
            self.assertIn('Stage: snapshot', stream.getvalue())
            self.assertIn('snapshot_unavailable: frozen commit archive is unavailable', stream.getvalue())
            self.assertIn(f'Report: {report_path}', stream.getvalue())

    def test_preflight_atomic_report_write_replaces_only_complete_json(self):
        with TemporaryDirectory() as temp:
            target = Path(temp) / 'report.json'
            target.write_text('{"previous": true}\n')
            from reviewer import preflight
            real_replace = preflight.os.replace
            def checked_replace(source, destination):
                temp_data = json.loads(Path(source).read_text())
                self.assertEqual(temp_data, {'status': 'ready'})
                self.assertEqual(Path(destination), target)
                real_replace(source, destination)
            with patch('reviewer.preflight.os.replace', side_effect=checked_replace):
                _write_atomic(target, {'status': 'ready'})
            self.assertEqual(json.loads(target.read_text()), {'status': 'ready'})
            self.assertEqual(list(Path(temp).glob('*.tmp')), [])

    def test_empty_selection_preflight_preserves_strict_version_validation(self):
        from reviewer.static_analysis import run_bandit
        for discovered in ('1.7.8', '1.7.5'):
            with self.subTest(version=discovered), TemporaryDirectory() as temp:
                root = Path(temp)
                archive, subset_path, benchmark, config, client = self._preflight_fixture(root)
                context = Mock()
                context.payload.return_value = []
                context.model_dump.return_value = {'chunks': []}
                results = []
                def static_for(inp, snapshot, context, static_config):
                    result = run_bandit(root, [], timeout_seconds=static_config.timeout_seconds,
                                        head=inp.pr.head, archive_sha256='a' * 64)
                    results.append(result)
                    return result
                with patch('reviewer.preflight.load_benchmark', return_value=benchmark), \
                     patch('reviewer.preflight.reconstruct', side_effect=lambda task, client:
                           ReviewInput(pr=task.pr, diff=fixtures.DIFF, diff_complete=True)), \
                     patch('reviewer.preflight.prepare_context', return_value=context), \
                     patch('reviewer.preflight.prepare_static_analysis', side_effect=static_for), \
                     patch('reviewer.preflight.get_bandit_version', return_value='1.7.8'), \
                     patch('reviewer.static_analysis.subprocess.run',
                           return_value=Mock(returncode=0, stdout=f'bandit {discovered}\n')) as process, \
                     patch('reviewer.preflight.platform.platform', return_value='offline-test-platform'), \
                     patch('reviewer.models.OpenAIReviewer.__init__',
                           side_effect=AssertionError('model client created')):
                    kwargs = dict(archive=archive, subset_path=subset_path, config=config,
                        client=client, results_dir=root / 'out', review_model='gpt-6-luna',
                        review_provider='openai', verify_model='gpt-6-luna',
                        verify_provider='openai', cache_dir=root / 'cache')
                    if discovered == '1.7.8':
                        _, report = run_preflight(**kwargs)
                        self.assertEqual(report['status'], 'ready')
                        self.assertTrue(all(row['bandit']['version'] == discovered
                                            and row['bandit']['status'] == 'no_files'
                                            for row in report['tasks']))
                    else:
                        with self.assertRaises(PreflightFailure) as caught:
                            run_preflight(**kwargs)
                        self.assertEqual(caught.exception.stage, 'static_analysis')
                        self.assertEqual(caught.exception.message, 'Bandit result version mismatch')
                        self.assertFalse(subset_path.with_suffix('.lock.json').exists())
                    self.assertTrue(results)
                    self.assertTrue(all(r.version == discovered and r.status == 'no_files'
                                        for r in results))
                    self.assertEqual(process.call_count, len(results))
                    self.assertTrue(all(c.args[0] == ['bandit', '--version']
                                        for c in process.call_args_list))

    def test_final_lock_created_only_after_successful_all_task_preflight(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            archive, subset_path, benchmark, config, client = self._preflight_fixture(
                root, purpose='final-evaluation')
            with patch('reviewer.preflight.load_benchmark', return_value=benchmark), \
                 patch('reviewer.preflight.reconstruct', side_effect=lambda task, client:
                       ReviewInput(pr=task.pr, diff=fixtures.DIFF, diff_complete=True)), \
                 patch('reviewer.preflight.prepare_context') as prepare_context, \
                 patch('reviewer.preflight.prepare_static_analysis') as prepare_static, \
                 patch('reviewer.preflight.get_bandit_version', return_value='1.7.8'), \
                 patch('reviewer.models.OpenAIReviewer.complete', side_effect=AssertionError('model call')) as complete:
                context = Mock()
                context.payload.return_value = []
                context.model_dump.return_value = {'chunks': []}
                prepare_context.return_value = context
                def static_for(inp, *args):
                    return StaticAnalysisResult(tool='bandit', version='1.7.8', head=inp.pr.head,
                        archive_sha256='a'*64, analyzed_files=['src/a.py'],
                        config={'signal_byte_budget': 2000}, timeout_seconds=20,
                        status='no_findings', error=None, findings=[], metrics={}, signals_sent=[])
                prepare_static.side_effect = static_for
                path, report = run_preflight(archive=archive, subset_path=subset_path, config=config,
                    client=client, results_dir=root / 'out', review_model='gpt-6-luna',
                    review_provider='openai', verify_model='gpt-6-luna', verify_provider='openai',
                    cache_dir=root / 'cache')
            lock_path = subset_path.with_suffix('.lock.json')
            lock = json.loads(lock_path.read_text())
            self.assertTrue(path.exists())
            self.assertEqual(report['status'], 'ready')
            self.assertEqual(len(report['tasks']), len(benchmark.subset.tasks))
            self.assertTrue(all(row['status'] == 'passed' for row in report['tasks']))
            self.assertEqual(len(lock['tasks']), len(benchmark.subset.tasks))
            complete.assert_not_called()

    def test_preflight_warning_is_persisted_in_report_and_final_lock(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            subset_path = root / 'final-one.json'
            raw_subset = read_json((ROOT / 'configs/reviewbench-final20.json').read_bytes())
            raw_subset['tasks'] = [next(task for task in raw_subset['tasks']
                                         if task['task_id'] == 'JacobPEvans_ansible-splunk_120-252ddc57')]
            subset_path.write_text(json.dumps(raw_subset, sort_keys=True))
            benchmark = load_benchmark(ARCHIVE, ROOT / 'configs/reviewbench-final20.json',
                                       allow_unpinned_diffs=True)
            task = benchmark.subset.tasks[0]
            subset = benchmark.subset.model_copy(update={'tasks': [task]})
            benchmark = replace(benchmark, subset=subset,
                                subset_sha256=digest(subset_path.read_bytes()),
                                golden={task.task_id: benchmark.golden[task.task_id]})
            diff_bytes, _ = diagnosed_benchmark_diff()
            diff_text = diff_bytes.decode()
            from reviewer.schemas import BenchmarkAggregateMismatch
            warning = BenchmarkAggregateMismatch(
                code='benchmark_aggregate_mismatch', expected_file_count=4,
                expected_additions=94, expected_deletions=131,
                reconstructed_file_count=4, reconstructed_additions=91,
                reconstructed_deletions=131, base_sha=task.pr.base, head_sha=task.pr.head,
                mismatch_fields=['additions'])
            inp = ReviewInput(pr=task.pr, diff=diff_text, diff_complete=True,
                              acquisition_warnings=[warning])
            config = load_config(ROOT / 'configs/experiment.toml')
            config.review.request_byte_budget = 32000
            config.review.max_output_tokens = 8192
            config.retrieval.context_byte_budget = 6000
            client = Mock()
            client.frozen_head.return_value = Mock(commit=task.pr.head, archive_sha256='a'*64)
            context = Mock()
            context.payload.return_value = []
            context.model_dump.return_value = {'chunks': []}
            static = StaticAnalysisResult(tool='bandit', version='1.7.8', head=task.pr.head,
                archive_sha256='a'*64, analyzed_files=[], config={'signal_byte_budget': 2000},
                timeout_seconds=20, status='no_files', error=None, findings=[], metrics={}, signals_sent=[])
            with patch('reviewer.preflight.load_benchmark', return_value=benchmark), \
                 patch('reviewer.preflight.reconstruct', return_value=inp), \
                 patch('reviewer.preflight.prepare_context', return_value=context), \
                 patch('reviewer.preflight.prepare_static_analysis', return_value=static), \
                 patch('reviewer.preflight.get_bandit_version', return_value='1.7.8'):
                path, report = run_preflight(archive=ARCHIVE, subset_path=subset_path, config=config,
                    client=client, results_dir=root / 'out', review_model='gpt-6-luna',
                    review_provider='openai', verify_model='gpt-6-luna', verify_provider='openai',
                    cache_dir=root / 'cache')
            expected = warning.model_dump(mode='json')
            self.assertEqual(report['tasks'][0]['status'], 'passed')
            self.assertEqual(report['tasks'][0]['acquisition_warnings'], [expected])
            lock = json.loads(subset_path.with_suffix('.lock.json').read_text())
            self.assertEqual(lock['tasks'][0]['acquisition_warnings'], [expected])
            # The warning is persisted as provenance; payload construction still
            # receives only diff/context/signals, never ReviewInput metadata.
            from reviewer.preflight import PayloadOnlyReviewer
            payload = PayloadOnlyReviewer('gpt-6-luna', config.review).payload(
                inp.diff, 'prompt', context=context.payload(), static_signals=[])
            self.assertNotIn('benchmark_aggregate_mismatch', json.dumps(payload))
            loaded = load_benchmark(ARCHIVE, subset_path)
            self.assertEqual(loaded.diff_lock_sha256, digest(subset_path.with_suffix('.lock.json').read_bytes()))

    def test_arbitrary_subset_evaluation_count_and_abcd_pairing_contract(self):
        helper = fixtures.BenchmarkTests(methodName='test_five_runs_export_and_label_leakage_guard')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        benchmark = helper.benchmark
        selected = benchmark.subset.tasks[:2]
        subset = benchmark.subset.model_copy(update={'tasks': selected})
        smaller = replace(benchmark, subset=subset,
            golden={task.task_id: benchmark.golden[task.task_id] for task in selected})
        provider = helper.provider()
        path, report = run_development_subset(smaller, provider, helper.client()[0], helper.root / 'out')
        self.assertEqual(report['requested_reviews'], 2)
        self.assertEqual(report['completed_reviews'], 2)
        self.assertEqual(len(list((path.parent / 'candidates').glob('*.json'))), 2)


if __name__ == '__main__':
    unittest.main()
