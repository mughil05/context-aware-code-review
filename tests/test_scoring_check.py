import hashlib
import json
import pytest
from scripts.reviewbench_scoring_check import check, comparison, validate_inputs


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def ready(tmp_path):
    ids = [f'task{i:02}' for i in range(19)]
    config = {'generation_run': 'run', 'task_ids': ids, 'file_hashes': {},
              'eval_config': {'matcher_model': 'openai/gpt-5.4'},
              'checkpoint_fingerprints': {m: 'frozen' for m in ('baseline','retrieval','tools','full')}}
    for key in ids:
        put(tmp_path/'run/scoring-golden'/f'{key}.json', {})
    put(tmp_path/'run/summary.json', {})
    for mode in config['checkpoint_fingerprints']:
        for key in ids:
            put(tmp_path/f'run/modes/{mode}/candidates/{key}.json', {'findings': []})
        put(tmp_path/f'run/scoring/{mode}.json', {'eval_config': config['eval_config'],
            'pr_count': 19, 'per_pr': [{'pr_key': k} for k in ids]})
        put(tmp_path/f'run/scoring/{mode}.details.json',
            [{'pr_key': k, 'candidate_findings': []} for k in ids])
        (tmp_path/f'run/scoring/{mode}.log').write_text('=== Results (19 PRs, 0 errors) ===\nJUDGE_EXIT=0\n')
    return tmp_path, config


def test_all_modes_complete_without_mutating_files(ready):
    root, c = ready
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    for mode in c['checkpoint_fingerprints']:
        assert check(c, mode, root)['completed_prs'] == 19
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('bad', ['checkpoint', 'missing', 'duplicate', 'wrong_model', 'error', 'exit'])
def test_fail_closed(ready, bad):
    root, c = ready
    path = root/'run/scoring/baseline.json'
    data = json.loads(path.read_text())
    if bad == 'checkpoint': put(root/'run/scoring/baseline.checkpoint.json', {})
    if bad == 'missing': data['per_pr'].pop()
    if bad == 'duplicate': data['per_pr'][-1] = data['per_pr'][0]
    if bad == 'wrong_model': data['eval_config'] = {}
    if bad == 'error': (root/'run/scoring/baseline.log').write_text('FAILED: task\nJUDGE_EXIT=0\n')
    if bad == 'exit': (root/'run/scoring/baseline.log').write_text('output without exit marker')
    put(path, data)
    with pytest.raises(ValueError): check(c, 'baseline', root)


def test_checkpoint_compatibility_and_incomplete_count(ready):
    root, c = ready
    key = c['task_ids'][0]
    cp = {'fingerprint': 'frozen', 'completed': {key: {'pr_key': key}},
          'details': {key: {'pr_key': key, 'candidate_findings': []}}}
    path = root/'run/scoring/baseline.checkpoint.json'
    put(path, cp)
    assert check(c, 'baseline', root, True)['remaining_prs'] == 18
    cp['fingerprint'] = 'changed'; put(path, cp)
    with pytest.raises(ValueError, match='DO NOT RETRY'): check(c, 'baseline', root, True)


def test_frozen_file_hash_and_exclusion_checks(ready):
    root,c=ready
    c['file_hashes']={'run/summary.json':'bad'}
    with pytest.raises(ValueError,match='changed'):validate_inputs(c,root)
    c['file_hashes']={}
    put(root/'run/modes/baseline/candidates/nipoppy_unexpected.json',{})
    with pytest.raises(ValueError,match='task mismatch'):validate_inputs(c,root)


def test_comparison_is_deterministic_and_checks_provenance(ready):
    root,c=ready
    keys=('augmented_precision','augmented_recall','f1','fixed_golden_recall',
          'grounded_precision','false_positives_per_pr','eligible_golden_tp','novel_tp',
          'excluded_golden_count','excluded_candidate_count')
    for mode in c['checkpoint_fingerprints']:
        d={k:None for k in keys}
        d.update(mode=mode,completed_pr_count=19,evaluable_pr_count=19,excluded_pre_execution_pr_count=1,
                 macro={'augmented_precision':None,'augmented_recall':None,'f1':None})
        for field,path in [('reviewbench_judge_details_sha256',f'scoring/{mode}.details.json'),
                           ('reviewbench_judge_metrics_sha256',f'scoring/{mode}.json'),
                           ('generation_summary_sha256','summary.json')]:
            d[field]=hashlib.sha256((root/'run'/path).read_bytes()).hexdigest()
        put(root/f'run/scoring/{mode}.scoped.json',d)
    assert [r['mode'] for r in comparison(c,root)]==list(c['checkpoint_fingerprints'])
    assert comparison(c,root)==comparison(c,root)
    put(root/'run/summary.json',{'changed':True})
    with pytest.raises(ValueError,match='provenance'):comparison(c,root)
