"""Offline input, completion, checkpoint and comparison checks; never invokes a judge."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ('baseline', 'retrieval', 'tools', 'full')


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_inputs(config, root=ROOT):
    for name, expected in config['file_hashes'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
                f'frozen scoring input changed: {name}')
    run = root / config['generation_run']
    expected = set(config['task_ids'])
    require(len(expected) == 19 and not any(t.startswith('nipoppy_') for t in expected),
            'invalid evaluable task set')
    for mode in MODES:
        paths = list((run / 'modes' / mode / 'candidates').glob('*.json'))
        require({p.stem for p in paths} == expected and len(paths) == 19, 'candidate task mismatch')
    paths = list((run / 'scoring-golden').glob('*.json'))
    require({p.stem for p in paths} == expected and len(paths) == 19, 'golden task mismatch')
    return run


def validate_records(scores, details, config, mode, run, complete):
    ids = set(config['task_ids'])
    score_ids = [x['pr_key'] for x in scores]
    detail_ids = [x['pr_key'] for x in details]
    require(len(score_ids) == len(set(score_ids)) and len(detail_ids) == len(set(detail_ids)),
            'duplicate judged task')
    require(set(score_ids) == set(detail_ids) and set(score_ids) <= ids, 'judged task mismatch')
    if complete:
        require(set(score_ids) == ids, 'not all 19 PRs completed')
    for detail in details:
        candidates = read(run / 'modes' / mode / 'candidates' / (detail['pr_key'] + '.json'))['findings']
        findings = detail['candidate_findings']
        require(len(findings) == len(candidates), 'incomplete candidate judgments')
        for candidate, finding in zip(candidates, findings):
            require(all(candidate[k] == finding[k] for k in ('file', 'start_line', 'end_line', 'message')),
                    'judge detail differs from candidate')
            require(finding['status'] in ('matched_tp', 'matched_fp', 'novel_tp', 'novel_fp'),
                    'missing classification')
            prefix = 'golden' if finding['status'].startswith('matched') else 'classifier'
            require(all(finding.get(prefix + '_' + k) for k in ('scope', 'category', 'tp_fp')),
                    'missing scope/label serialization')
    return len(score_ids)


def check(config, mode, root=ROOT, recovery=False):
    run = validate_inputs(config, root)
    base = run / 'scoring' / mode
    checkpoint = base.with_suffix('.checkpoint.json')
    if recovery:
        require(checkpoint.exists(), 'no checkpoint: retry could repeat all paid work; inspect manually')
        cp = read(checkpoint)
        require(cp['fingerprint'] == config['checkpoint_fingerprints'][mode],
                'checkpoint fingerprint mismatch: upstream would restart; DO NOT RETRY')
        require(all(k == v['pr_key'] for k, v in cp['completed'].items()) and
                all(k == v['pr_key'] for k, v in cp['details'].items()), 'checkpoint key mismatch')
        n = validate_records(list(cp['completed'].values()), list(cp['details'].values()),
                             config, mode, run, False)
        require(n < 19, 'all PRs checkpointed: inspect finalization; do not blindly rerun')
        return {'status': 'compatible_incomplete_checkpoint', 'completed_prs': n,
                'remaining_prs': 19 - n, 'in_flight_paid_work_may_repeat': True}
    require(not checkpoint.exists(), 'checkpoint remains: mode not finalized')
    aggregate, details = read(base.with_suffix('.json')), read(base.with_suffix('.details.json'))
    require(aggregate['eval_config'] == config['eval_config'], 'judge configuration mismatch')
    require(aggregate['pr_count'] == 19, 'aggregate PR count mismatch')
    validate_records(aggregate['per_pr'], details, config, mode, run, True)
    log = base.with_suffix('.log').read_text()
    require(log.rstrip().endswith('JUDGE_EXIT=0'), 'missing successful exit marker')
    require('=== Results (19 PRs, 0 errors) ===' in log, 'missing zero-error completion summary')
    require('FAILED:' not in log and 'Input validation failed:' not in log and
            'Fatal error:' not in log, 'judge error recorded in log')
    return {'status': 'complete', 'mode': mode, 'completed_prs': 19, 'pr_errors': 0}


def comparison(config, root=ROOT):
    rows = []
    for mode in MODES:
        check(config, mode, root)
        run = root / config['generation_run']
        path = run / 'scoring' / f'{mode}.scoped.json'
        data = read(path)
        require(data['mode'] == mode and data['completed_pr_count'] == 19 and
                data['evaluable_pr_count'] == 19 and data['excluded_pre_execution_pr_count'] == 1,
                'scoped completion mismatch')
        for field, source in [('reviewbench_judge_details_sha256', f'scoring/{mode}.details.json'),
                              ('reviewbench_judge_metrics_sha256', f'scoring/{mode}.json'),
                              ('generation_summary_sha256', 'summary.json')]:
            require(data[field] == hashlib.sha256((run / source).read_bytes()).hexdigest(),
                    'scoped provenance mismatch')
        keys = ('augmented_precision', 'augmented_recall', 'f1', 'fixed_golden_recall',
                'grounded_precision', 'false_positives_per_pr', 'eligible_golden_tp',
                'novel_tp', 'excluded_golden_count', 'excluded_candidate_count', 'completed_pr_count')
        rows.append({'mode': mode, **{k: data[k] for k in keys},
                     **{'macro_' + k: data['macro'][k] for k in ('augmented_precision', 'augmented_recall', 'f1')}})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('inputs', 'complete', 'checkpoint', 'compare'))
    p.add_argument('--mode', choices=MODES)
    p.add_argument('--config', default=str(ROOT / 'configs/reviewbench-scoring.json'))
    args = p.parse_args()
    config = read(args.config)
    if args.action == 'inputs':
        validate_inputs(config)
        result = {'status': 'inputs_valid', 'task_count': 19}
    elif args.action == 'compare':
        rows = comparison(config)
        columns = list(rows[0])
        print('\t'.join(columns))
        for row in rows:
            print('\t'.join('null' if row[k] is None else str(row[k]) for k in columns))
        return
    else:
        p.error('--mode required') if not args.mode else None
        result = check(config, args.mode, recovery=args.action == 'checkpoint')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
