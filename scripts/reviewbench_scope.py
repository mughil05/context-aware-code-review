"""Build scoped golden inputs or aggregate pinned ReviewBench details."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reviewer.benchmark import digest, read_json
from reviewer.benchmark import Task
from reviewer.final_exclusions import TASK_ID as FINAL_EXCLUDED_TASK_ID, exclusion_record
from reviewer.scoped_evaluation import aggregate_scoped, scoped_golden


def exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def filter_golden(task_manifest, golden_dir, output_dir):
    subset = read_json(Path(task_manifest).read_bytes())
    final20 = (subset.get('purpose') == 'final-evaluation' and len(subset.get('tasks', [])) == 20
               and sum(task.get('task_id') == FINAL_EXCLUDED_TASK_ID for task in subset.get('tasks', [])) == 1)
    excluded = []
    if final20:
        fixed = [Task.model_validate(task) for task in subset['tasks']
                 if task.get('task_id') == FINAL_EXCLUDED_TASK_ID]
        if len(fixed) != 1:
            raise ValueError('final manifest lacks the one declared pre-execution exclusion')
        excluded = [exclusion_record(fixed[0])]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for task in subset['tasks']:
        task_id = task['task_id']
        if final20 and task_id == FINAL_EXCLUDED_TASK_ID:
            continue
        source = Path(golden_dir) / f'{task_id}.json'
        original = read_json(source.read_bytes())
        filtered = scoped_golden(original)
        if filtered['pr_key'] != task_id or filtered['pr'] != task['pr']:
            raise ValueError('golden task identity mismatch')
        target = output / f'{task_id}.json'
        exclusive_json(target, filtered)
        counts[task_id] = {'all': len(original['findings']), 'eligible': len(filtered['findings']),
                           'excluded': len(original['findings']) - len(filtered['findings']),
                           'sha256': digest(target.read_bytes())}
    if excluded:
        counts['_pre_execution_exclusions'] = excluded
    return counts


def aggregate(details_path, judge_summary, reference_dir, task_manifest, run_summary, mode):
    subset = read_json(Path(task_manifest).read_bytes())
    summary = read_json(Path(run_summary).read_bytes())
    excluded = []
    final20 = (subset.get('purpose') == 'final-evaluation' and len(subset.get('tasks', [])) == 20
               and sum(task.get('task_id') == FINAL_EXCLUDED_TASK_ID for task in subset.get('tasks', [])) == 1)
    if final20:
        fixed = [Task.model_validate(task) for task in subset['tasks']
                 if task.get('task_id') == FINAL_EXCLUDED_TASK_ID]
        if len(fixed) != 1:
            raise ValueError('final manifest lacks the one declared pre-execution exclusion')
        expected_exclusion = exclusion_record(fixed[0])
        if summary.get('excluded_tasks') != [expected_exclusion]:
            raise ValueError('run summary does not preserve the declared exclusion provenance')
        excluded = [FINAL_EXCLUDED_TASK_ID]
    records = [r for r in summary['tasks'] if r.get('mode') == mode]
    expected_ids = {task['task_id'] for task in subset['tasks']} - set(excluded)
    if len(records) != len(expected_ids) or {r.get('task_id') for r in records} != expected_ids:
        raise ValueError('generation summary task/mode coverage mismatch')
    successful = {r['task_id'] for r in records if r.get('status') == 'success'}
    failed = {r['task_id'] for r in records if r.get('status') == 'failed'}
    not_attempted = {r['task_id'] for r in records if r.get('status') not in ('success', 'failed')}
    detail_array = read_json(Path(details_path).read_bytes())
    detail_by_id = {d.get('pr_key'): d for d in detail_array if isinstance(d, dict)}
    if len(detail_by_id) != len(detail_array):
        raise ValueError('duplicate/malformed ReviewBench detail identity')
    if set(detail_by_id) != successful:
        raise ValueError('ReviewBench detail task set differs from successful generation set')
    results = {}
    for task in subset['tasks']:
        task_id = task['task_id']
        if task_id not in successful:
            continue
        if task_id not in detail_by_id:
            raise ValueError('successful review is missing ReviewBench judge detail')
        golden = read_json((Path(reference_dir) / f'{task_id}.json').read_bytes())
        results[task_id] = {'golden': golden, 'details': detail_by_id[task_id]}
    if successful | failed | not_attempted != expected_ids:
        raise ValueError('generation summary does not cover the selected task manifest')
    output = aggregate_scoped(results, completed_task_ids=successful, failed_task_ids=failed,
                              not_attempted_task_ids=not_attempted,
                              excluded_pre_execution_task_ids=excluded)
    output.update({'mode': mode, 'benchmark_revision': subset['benchmark_revision'],
                   'manifest_sha256': subset['manifest_sha256'],
                   'subset_sha256': summary.get('subset_sha256'),
                   'reviewbench_judge_details_sha256': digest(Path(details_path).read_bytes()),
                   'reviewbench_judge_metrics_sha256': digest(Path(judge_summary).read_bytes()),
                   'generation_summary_sha256': digest(Path(run_summary).read_bytes()),
                   'scoring_status': 'complete'})
    return output


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    filt = sub.add_parser('filter-goldens')
    filt.add_argument('--task-manifest', required=True)
    filt.add_argument('--golden-dir', required=True)
    filt.add_argument('--output-dir', required=True)
    score = sub.add_parser('aggregate')
    score.add_argument('--details', required=True)
    score.add_argument('--judge-summary', required=True,
                       help='Pinned ReviewBench aggregate JSON for provenance; detailed judgments drive adapted metrics')
    score.add_argument('--reference-dir', required=True)
    score.add_argument('--task-manifest', required=True)
    score.add_argument('--run-summary', required=True)
    score.add_argument('--mode', choices=('baseline', 'retrieval', 'tools', 'full'), required=True)
    score.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'filter-goldens':
        counts = filter_golden(args.task_manifest, args.golden_dir, args.output_dir)
        print(json.dumps(counts, indent=2, sort_keys=True))
    else:
        exclusive_json(args.output, aggregate(args.details, args.judge_summary, args.reference_dir,
                                               args.task_manifest, args.run_summary, args.mode))
        print(f'Saved scoped metrics: {args.output}')


if __name__ == '__main__':
    main()
