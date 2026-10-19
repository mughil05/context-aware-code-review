"""Synthetic judge outputs only: no matching, models, or final quality results."""
from copy import deepcopy
import pytest
from reviewer.scoped_evaluation import aggregate_scoped
from reviewer.final_exclusions import TASK_ID


def record(key, positives=2, candidates=()):
    return {'golden': {'pr_key': key, 'findings': [
        {'tp_fp': 'tp', 'category': 'correctness', 'scope': 'introduced-by-pr'}
        for _ in range(positives)]},
        'details': {'pr_key': key, 'candidate_findings': list(candidates)}}


def novel(label='tp', category='correctness', scope='introduced-by-pr'):
    return {'status': 'novel_' + label, 'classifier_tp_fp': label,
            'classifier_category': category, 'classifier_scope': scope}


def matched(covered=(0,)):
    return {'status': 'matched_tp', 'matched_golden_index': 0,
            'covered_golden_indices': list(covered), 'golden_tp_fp': 'tp',
            'golden_category': 'correctness', 'golden_scope': 'introduced-by-pr'}


def score(records, **kwargs):
    return aggregate_scoped(records, completed_task_ids=list(records), **kwargs)


def test_novel_tp_augments_both_recall_terms_and_primary_f1():
    r = score({'a': record('a', candidates=[matched(), novel()])})
    assert r['augmented_precision'] == 1
    assert r['recall'] == r['augmented_recall'] == pytest.approx(2/3)
    assert r['fixed_golden_recall'] == .5
    assert r['f1'] == pytest.approx(.8)
    assert r['per_task'][0]['f1'] == pytest.approx(.8)


def test_novel_fp_does_not_increase_recall():
    r = score({'a': record('a', candidates=[matched(), novel('fp')])})
    assert r['recall'] == .5
    assert r['precision'] == .5
    assert r['fp'] == r['per_task'][0]['fp'] == 1


def test_coverage_is_unique_and_duplicate_credit_follows_upstream_output():
    # Upstream classified the duplicate candidate as novel_fp; do not rematch it.
    r = score({'a': record('a', candidates=[matched((0, 0, 1)), novel('fp')])})
    assert r['covered_eligible_golden_tp'] == 2
    assert r['tp'] == 1 and r['fp'] == 1
    assert r['recall'] == 1


@pytest.mark.parametrize('extra', [novel(category='testing'), novel(scope='pre-existing')])
def test_out_of_contract_unmatched_is_excluded(extra):
    r = score({'a': record('a', candidates=[matched(), extra])})
    assert r['novel_tp'] == 0
    assert r['excluded_candidate_count'] == 1
    assert r['recall'] == .5


def test_micro_macro_and_exclusion_denominators():
    r = score({'a': record('a', candidates=[matched(), novel()]),
               'b': record('b', positives=1, candidates=[novel('fp')]),
               'empty': record('empty', positives=0)},
              excluded_pre_execution_task_ids=[TASK_ID])
    assert r['micro']['augmented_precision'] == pytest.approx(2/3)
    assert r['micro']['augmented_recall'] == .5
    assert r['micro']['f1'] == pytest.approx(4/7)
    assert r['macro']['augmented_precision'] == .5
    assert r['macro']['augmented_recall'] == pytest.approx(1/3)
    assert r['macro']['f1'] == pytest.approx(.4)
    assert r['macro_defined_pr_counts']['f1'] == 2
    assert r['false_positives_per_pr'] == pytest.approx(1/3)
    assert r['false_positives_per_evaluable_pr'] == pytest.approx(1/3)
    assert r['intended_pr_count'] == 4 and r['evaluable_pr_count'] == 3


def test_zero_denominators_are_null_and_novel_only_tp_is_defined():
    r = score({'a': record('a', positives=0)})
    assert r['precision'] is r['recall'] is r['f1'] is None
    assert r['macro']['f1'] is None
    assert r['macro_defined_pr_counts']['f1'] == 0
    r = score({'a': record('a', positives=0, candidates=[novel()])})
    assert r['precision'] == r['recall'] == r['f1'] == 1
    assert r['fixed_golden_recall'] is None
    assert score({})['false_positives_per_pr'] is None


def test_incomplete_corpus_does_not_assume_zero_fp_for_unfinished_prs():
    r = score({'a': record('a', candidates=[novel('fp')])}, failed_task_ids=['b'])
    assert r['false_positives_per_pr'] == 1
    assert r['false_positives_per_evaluable_pr'] is None


def test_adapter_never_mutates_inputs_or_calls_generation():
    records = {'a': record('a', candidates=[matched(), novel()])}
    before = deepcopy(records)
    score(records)
    assert records == before
