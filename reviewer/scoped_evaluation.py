"""Deterministic category/scope adapter around pinned ReviewBench judge output.

ReviewBench remains responsible for candidate↔golden matching and unmatched
candidate labels. This module only filters reference labels and aggregates its
per-finding judgment output.
"""
from reviewer.benchmark import CATEGORIES, SCOPES


def eligible_finding(finding):
    return finding['category'] in CATEGORIES and finding['scope'] in SCOPES


def scoped_golden(golden):
    """Copy the same PR identity and original finding labels, filtering by contract."""
    return {**golden, 'findings': [dict(f) for f in golden['findings'] if eligible_finding(f)]}


def _task_score(task_id, golden, details):
    selected = scoped_golden(golden)['findings']
    if (golden.get('pr_key') != task_id or details.get('pr_key') != task_id
            or not isinstance(details.get('candidate_findings'), list)):
        raise ValueError('ReviewBench detailed output identity/schema mismatch')
    positives = [i for i, f in enumerate(selected) if f['tp_fp'] == 'tp']
    negatives = [i for i, f in enumerate(selected) if f['tp_fp'] == 'fp']
    covered = set()
    tp = fp = matched_tp = matched_fp = novel_tp = novel_fp = excluded_candidates = 0
    for candidate in details['candidate_findings']:
        status = candidate.get('status')
        if status in ('matched_tp', 'matched_fp'):
            index = candidate.get('matched_golden_index')
            if type(index) is not int or not 0 <= index < len(selected):
                raise ValueError('matched index is outside filtered ReviewBench reference')
            matched_golden = selected[index]
            if (candidate.get('golden_tp_fp') != matched_golden['tp_fp']
                    or candidate.get('golden_category') != matched_golden['category']
                    or candidate.get('golden_scope') != matched_golden['scope']):
                raise ValueError('matched finding labels disagree with filtered ReviewBench reference')
            if matched_golden['tp_fp'] == 'tp':
                tp += 1
                matched_tp += 1
            else:
                fp += 1
                matched_fp += 1
            covered.update(candidate.get('covered_golden_indices', []))
            continue
        if status not in ('novel_tp', 'novel_fp'):
            raise ValueError('ReviewBench candidate lacks an explicit match/classification')
        category, scope = candidate.get('classifier_category'), candidate.get('classifier_scope')
        if not category or not scope:
            raise ValueError('ReviewBench output lacks category/scope for unmatched candidate')
        expected_label = 'tp' if status == 'novel_tp' else 'fp'
        if candidate.get('classifier_tp_fp') != expected_label:
            raise ValueError('unmatched candidate status disagrees with ReviewBench classifier label')
        if category not in CATEGORIES or scope not in SCOPES:
            excluded_candidates += 1
            continue
        if status == 'novel_tp':
            tp += 1
            novel_tp += 1
        else:
            fp += 1
            novel_fp += 1
    if any(type(index) is not int or not 0 <= index < len(selected) for index in covered):
        raise ValueError('covered index is outside filtered ReviewBench reference')
    covered_tp = len({i for i in covered if selected[i]['tp_fp'] == 'tp'})
    return {'task_id': task_id, 'tp': tp, 'fp': fp, 'fn': len(positives) - covered_tp,
            'matched_tp': matched_tp, 'matched_fp': matched_fp,
            'eligible_golden_tp': len(positives), 'eligible_golden_fp': len(negatives),
            'excluded_golden': len(golden['findings']) - len(selected),
            'covered_eligible_golden_tp': covered_tp,
            'novel_tp': novel_tp, 'novel_fp': novel_fp,
            'excluded_candidates': excluded_candidates,
            'candidate_denominator': tp + fp}


def aggregate_scoped(results, *, completed_task_ids, failed_task_ids=(), not_attempted_task_ids=(),
                     excluded_pre_execution_task_ids=()):
    """Adapt ReviewBench details to fixed-golden recall and in-scope FP rates."""
    completed = set(completed_task_ids)
    failed = set(failed_task_ids)
    not_attempted = set(not_attempted_task_ids)
    excluded = set(excluded_pre_execution_task_ids)
    if (completed & failed) or (completed & not_attempted) or (failed & not_attempted) or (
            excluded & (completed | failed | not_attempted)):
        raise ValueError('task cannot have multiple completion states')
    per_task = []
    for task_id in sorted(completed):
        if task_id not in results:
            raise ValueError('completed task has no ReviewBench detail output')
        per_task.append(_task_score(task_id, results[task_id]['golden'], results[task_id]['details']))
    tp = sum(row['tp'] for row in per_task)
    fp = sum(row['fp'] for row in per_task)
    matched_tp = sum(row['matched_tp'] for row in per_task)
    matched_fp = sum(row['matched_fp'] for row in per_task)
    eligible_tp = sum(row['eligible_golden_tp'] for row in per_task)
    covered_tp = sum(row['covered_eligible_golden_tp'] for row in per_task)
    precision = tp / (tp + fp) if tp + fp else None
    grounded_precision = (matched_tp / (matched_tp + matched_fp)
                          if matched_tp + matched_fp else None)
    recall = covered_tp / eligible_tp if eligible_tp else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        'metric_family': 'scope-filtered/adapted ReviewBench; not native ReviewBench metrics',
        'precision': precision, 'recall': recall, 'f1': f1,
        'grounded_precision': grounded_precision,
        'false_positives_per_pr': fp / len(per_task) if per_task else None,
        'tp': tp, 'fp': fp, 'fn': eligible_tp - covered_tp,
        'eligible_golden_tp': eligible_tp,
        'eligible_golden_fp': sum(row['eligible_golden_fp'] for row in per_task),
        'excluded_golden_count': sum(row['excluded_golden'] for row in per_task),
        'covered_eligible_golden_tp': covered_tp,
        'novel_tp': sum(row['novel_tp'] for row in per_task),
        'novel_fp': sum(row['novel_fp'] for row in per_task),
        'excluded_candidate_count': sum(row['excluded_candidates'] for row in per_task),
        'candidate_denominator': tp + fp,
        'grounded_precision_denominator': matched_tp + matched_fp,
        'completed_pr_count': len(per_task), 'failed_pr_count': len(failed),
        'not_attempted_pr_count': len(not_attempted),
        'intended_pr_count': len(completed | failed | not_attempted | excluded),
        'evaluable_pr_count': len(completed | failed | not_attempted),
        'excluded_pre_execution_pr_count': len(excluded),
        'excluded_pre_execution_task_ids': sorted(excluded),
        'precision_denominator': 'in-scope matched TP/FP plus in-scope ReviewBench-classified unmatched TP/FP',
        'grounded_precision_denominator_definition': 'in-scope candidates matched by ReviewBench to eligible golden findings; unmatched candidates excluded',
        'recall_denominator': 'all in-scope tp golden findings in completed PRs; unmatched novel TPs do not expand it',
        'false_positives_per_pr_denominator': 'completed PRs, including zero-positive PRs; failed PRs excluded',
        'per_task': per_task,
    }
