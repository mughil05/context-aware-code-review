"""Final byte-estimator and fixed Nipoppy exclusion contract tests."""
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from reviewer.archives import sha256
from reviewer.benchmark import REVISION, digest, load_benchmark, read_json
from reviewer.config import load_config
from reviewer.final_exclusions import (TASK_ID, exclusion_record, is_final20,
    REQUEST_BUDGET, CONTEXT_BYTE_BUDGET, STATIC_SIGNAL_BYTE_BUDGET, MAX_OUTPUT_TOKENS)
from reviewer.models import OpenAIReviewer, build_review_payload, ModelError
from reviewer.pipeline import prepare_context, review_prompt
from reviewer.preflight import run_preflight
from reviewer.request_budget import measure_request, SAFETY_MARGIN_BYTES, BUDGET_METHOD
from reviewer.scoped_evaluation import aggregate_scoped
from reviewer.schemas import PRMetadata, ReviewInput
import test_benchmark as fixtures

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / '.cache/reviewbench' / f'ReviewBench-{REVISION}.tar.gz'


class FinalBudgetTests(unittest.TestCase):
    def config(self):
        config = load_config(ROOT / 'configs/experiment.toml')
        config.review.request_byte_budget = REQUEST_BUDGET
        config.review.max_output_tokens = MAX_OUTPUT_TOKENS
        return config

    def test_request_estimator_records_exact_transport_bytes_and_margin(self):
        config = self.config()
        config.review.request_byte_budget = 1_000_000
        payload = build_review_payload('gpt-6-luna', config.review, 'diff', 'prompt')
        measured = measure_request(payload, config.review.request_byte_budget)
        expected = len(json.dumps(payload, allow_nan=False).encode('utf-8'))
        self.assertEqual(measured['serialized_request_body_utf8_bytes'], expected)
        self.assertEqual(measured['safety_margin_bytes'], 4096)
        self.assertEqual(measured['estimator_value'], expected + 4096)
        self.assertEqual(measured['configured_estimator_ceiling'], 1_000_000)
        self.assertEqual(measured['estimator_units'], 'serialized_request_estimator_units')
        self.assertEqual(measured['method'], BUDGET_METHOD)
        config.review.request_byte_budget = expected + 4096
        build_review_payload('gpt-6-luna', config.review, 'diff', 'prompt')
        config.review.request_byte_budget -= 1
        with self.assertRaisesRegex(ModelError, 'serialized-byte estimator ceiling'):
            build_review_payload('gpt-6-luna', config.review, 'diff', 'prompt')

    def test_shared_context_reserves_static_allowance_before_retrieval(self):
        config = self.config()
        pr = PRMetadata(repo='https://github.com/o/r', pr_number=1, base='a'*40, head='b'*40)
        inp = ReviewInput(pr=pr, diff=fixtures.DIFF, diff_complete=True)
        reviewer = OpenAIReviewer('gpt-6-luna', config.review, api_key='test-key')
        context = Mock()
        context.payload.return_value = []
        context.model_copy.return_value = context
        prompt = review_prompt()
        baseline = reviewer.payload(inp.diff, prompt)
        baseline_estimator = measure_request(baseline)['estimator_value']
        with patch('reviewer.retrieval.retrieve', return_value=context) as retrieve:
            result = prepare_context(inp, reviewer, object(), config.retrieval,
                                     reserve_static_signal_byte_cap=STATIC_SIGNAL_BYTE_BUDGET)
        self.assertIs(result, context)
        self.assertEqual(retrieve.call_args.kwargs['available_byte_budget'],
                         REQUEST_BUDGET - baseline_estimator - STATIC_SIGNAL_BYTE_BUDGET * 4 - 256 - 64)
        self.assertEqual(retrieve.call_args.args[2].context_byte_budget, CONTEXT_BYTE_BUDGET)
        self.assertEqual(context.model_copy.call_args.kwargs['update']['overall_request_byte_ceiling'], 73728)
        self.assertEqual(context.model_copy.call_args.kwargs['update']['reserved_static_signal_byte_cap'], 2000)
        self.assertEqual(context.model_copy.call_args.kwargs['update']['reserved_static_signal_estimator_bytes'], 8256)

    def test_final_manifest_keeps_fixed_exclusion_and_lock_rejects_extra_exclusion(self):
        raw = read_json((ROOT / 'configs/reviewbench-final20.json').read_bytes())
        self.assertEqual(len(raw['tasks']), 20)
        selected = [row for row in raw['tasks'] if row['task_id'] == TASK_ID]
        self.assertEqual(len(selected), 1)
        benchmark = load_benchmark(ARCHIVE, ROOT / 'configs/reviewbench-final20.json', allow_unpinned_diffs=True)
        self.assertTrue(is_final20(benchmark.subset.tasks, benchmark.subset.purpose))
        from reviewer.final_exclusions import validate_lock_exclusions
        with self.assertRaisesRegex(ValueError, 'exclusion provenance'):
            validate_lock_exclusions(benchmark.subset, [])
        nipoppy = next(t for t in benchmark.subset.tasks if t.task_id == TASK_ID)
        valid = exclusion_record(nipoppy)
        with self.assertRaisesRegex(ValueError, 'exclusion provenance'):
            validate_lock_exclusions(benchmark.subset, [valid, dict(valid, task_id='undeclared_task')])
        self.assertEqual(valid['status'], 'excluded_pre_execution')
        self.assertFalse(valid['included_in_metric_denominators'])
        self.assertEqual(valid['reason_codes'], ['benchmark_frozen_inventory_mismatch',
                                                   'unsupported_safe_snapshot_symlink'])

    def test_preflight_excludes_nipoppy_without_acquisition_and_writes_final_lock(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            subset_path = root / 'final20.json'
            subset_path.write_bytes((ROOT / 'configs/reviewbench-final20.json').read_bytes())
            benchmark = load_benchmark(ARCHIVE, subset_path, allow_unpinned_diffs=True)
            config = self.config()
            client = Mock()
            client.frozen_head.side_effect = lambda pr, **kwargs: Mock(
                commit=pr.head, archive_sha256='a' * 64, repository='review-bench/test')
            attempted = []
            def reconstruct(task, _client):
                attempted.append(task.task_id)
                return ReviewInput(pr=task.pr, diff=fixtures.DIFF, diff_complete=True)
            context = Mock()
            context.payload.return_value = []
            context.model_dump.return_value = {'chunks': []}
            def static_for(inp, *_args):
                from reviewer.static_analysis import StaticAnalysisResult
                return StaticAnalysisResult(tool='bandit', version='1.7.8', head=inp.pr.head,
                    archive_sha256='a'*64, analyzed_files=[], config={'signal_byte_budget': 2000},
                    timeout_seconds=60, status='no_findings', error=None, findings=[], metrics={})
            with patch('reviewer.preflight.reconstruct', side_effect=reconstruct), \
                 patch('reviewer.preflight.prepare_context', return_value=context) as prepare, \
                 patch('reviewer.preflight.prepare_static_analysis', side_effect=static_for), \
                 patch('reviewer.preflight.get_bandit_version', return_value='1.7.8'), \
                 patch('reviewer.models.OpenAIReviewer.complete', side_effect=AssertionError('model call')) as complete:
                report_path, report = run_preflight(archive=ARCHIVE, subset_path=subset_path,
                    config=config, client=client, results_dir=root / 'reports',
                    review_model='gpt-6-luna', review_provider='openai',
                    verify_model='gpt-6-luna', verify_provider='openai', cache_dir=root / 'cache')
            self.assertEqual(report['status'], 'ready')
            self.assertEqual(report['model_calls'], 0)
            self.assertEqual(report['intended_task_count'], 20)
            self.assertEqual(report['evaluable_task_count'], 19)
            self.assertEqual(report['completed_task_count'], 19)
            self.assertEqual(report['failed_task_count'], 0)
            self.assertEqual(report['excluded_pre_execution_count'], 1)
            self.assertEqual(report['excluded_tasks'][0]['task_id'], TASK_ID)
            self.assertEqual(len(report['tasks']), 19)
            self.assertNotIn(TASK_ID, attempted)
            self.assertEqual(len(attempted), 19)
            self.assertEqual(client.frozen_head.call_count, 19)
            self.assertTrue(all(call.kwargs['reserve_static_signal_byte_cap'] == 2000
                                for call in prepare.call_args_list))
            complete.assert_not_called()
            self.assertTrue(report_path.exists())
            lock_path = subset_path.with_suffix('.lock.json')
            lock = read_json(lock_path.read_bytes())
            self.assertEqual(lock['intended_task_count'], 20)
            self.assertEqual(lock['evaluable_task_count'], 19)
            self.assertEqual(lock['request_budget']['ceiling'], 73728)
            self.assertEqual(lock['pre_execution_exclusions'], report['excluded_tasks'])
            locked = load_benchmark(ARCHIVE, subset_path)
            self.assertIsNone(next(t for t in locked.subset.tasks if t.task_id == TASK_ID).diff_sha256)

    def test_excluded_task_is_not_counted_in_scoped_metric_denominators(self):
        result = aggregate_scoped({}, completed_task_ids=[], excluded_pre_execution_task_ids=[TASK_ID])
        self.assertEqual(result['intended_pr_count'], 1)
        self.assertEqual(result['evaluable_pr_count'], 0)
        self.assertEqual(result['excluded_pre_execution_pr_count'], 1)
        self.assertEqual(result['completed_pr_count'], 0)
        self.assertIsNone(result['false_positives_per_pr'])

    def test_final_evaluation_generation_makes_no_call_for_excluded_task(self):
        from reviewer.evaluation import run_development_subset
        from reviewer.models import OpenAIReviewer
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            benchmark = load_benchmark(ARCHIVE, ROOT / 'configs/reviewbench-final20.json',
                                       allow_unpinned_diffs=True)
            fixed_tasks = [task.model_copy(update={'diff_sha256': sha256(fixtures.DIFF.encode())})
                           for task in benchmark.subset.tasks]
            subset = benchmark.subset.model_copy(update={'tasks': fixed_tasks})
            benchmark = replace(benchmark, subset=subset)
            config = self.config()
            response = {'id': 'offline-response', 'model': 'gpt-6-luna', 'status': 'completed',
                        'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2},
                        'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                                    'content': [{'type': 'output_text', 'text': '{"findings":[]}'}]}]}
            transport = Mock(return_value=response)
            reviewer = OpenAIReviewer('gpt-6-luna', config.review, api_key='test-key', transport=transport)
            reconstructed = []
            def frozen_input(task, _client):
                reconstructed.append(task.task_id)
                return ReviewInput(pr=task.pr, diff=fixtures.DIFF, diff_complete=True)
            with patch('reviewer.evaluation.reconstruct', side_effect=frozen_input):
                summary_path, summary = run_development_subset(
                    benchmark, reviewer, Mock(), root / 'results', modes=['baseline'])
            self.assertEqual(transport.call_count, 19)
            self.assertNotIn(TASK_ID, reconstructed)
            self.assertEqual(summary['intended_task_count'], 20)
            self.assertEqual(summary['evaluable_task_count'], 19)
            self.assertEqual(summary['completed_task_count'], 19)
            self.assertEqual(summary['excluded_pre_execution_count'], 1)
            self.assertEqual(summary['excluded_tasks'][0]['applies_to_modes'],
                             ['baseline', 'retrieval', 'tools', 'full'])
            self.assertFalse(any(row['task_id'] == TASK_ID for row in summary['tasks']))
            sample = summary['tasks'][0]
            raw_run = read_json((summary_path.parent / sample['result_file']).read_bytes())
            self.assertEqual(raw_run['request_byte_measurement'],
                             sample['request_byte_measurements']['baseline'])
            self.assertEqual(read_json((summary_path.parent / 'inputs' /
                                       (sample['task_id'] + '.json')).read_bytes())['diff'], fixtures.DIFF)
            self.assertTrue(summary_path.exists())

    def test_final_scope_golden_filter_omits_nipoppy_from_judge_inputs(self):
        from scripts.reviewbench_scope import filter_golden
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            benchmark = load_benchmark(ARCHIVE, ROOT / 'configs/reviewbench-final20.json',
                                       allow_unpinned_diffs=True)
            golden_dir = root / 'golden'
            golden_dir.mkdir()
            for task_id, golden in benchmark.golden.items():
                (golden_dir / f'{task_id}.json').write_text(json.dumps(golden))
            output_dir = root / 'eligible-goldens'
            counts = filter_golden(ROOT / 'configs/reviewbench-final20.json', golden_dir, output_dir)
            self.assertFalse((output_dir / f'{TASK_ID}.json').exists())
            self.assertEqual(len(list(output_dir.glob('*.json'))), 19)
            self.assertEqual(counts['_pre_execution_exclusions'][0]['task_id'], TASK_ID)


if __name__ == '__main__':
    unittest.main()
