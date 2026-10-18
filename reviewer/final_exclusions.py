"""Single predeclared, pre-execution exclusion for the pinned final corpus."""
REVISION = '968291d99e82b3e2969d7e0b4acca62585886c13'
FINAL_SUBSET_SHA256 = '206e5ba692eb31c56980224d2105dff015e1ed3d448b930b5ce0eb090d16b189'

TASK_ID = 'nipoppy_nipoppy_964-9421a714'
BASE_SHA = '5575427b46328a907e5c4b15ee3bfa341ba85907'
HEAD_SHA = '9421a7142852552900aa26fa86027ac6cd6cddb9'
EXPECTED = {'file_count': 5, 'additions': 42, 'deletions': 10}
RECONSTRUCTED = {'file_count': 3, 'additions': 38, 'deletions': 0}
REASONS = ['benchmark_frozen_inventory_mismatch', 'unsupported_safe_snapshot_symlink']
SYMLINK = 'tests/data/template_pipeline -> ../../nipoppy/data/template_pipeline/'
REQUEST_BUDGET = 73728
SAFETY_MARGIN_BYTES = 4096
CONTEXT_BYTE_BUDGET = 6000
STATIC_SIGNAL_BYTE_BUDGET = 2000
MAX_OUTPUT_TOKENS = 8192


def matches_declared_task(task):
    return (task.task_id == TASK_ID and task.pr.repo == 'https://github.com/nipoppy/nipoppy'
            and task.pr.pr_number == 964 and task.pr.base == BASE_SHA and task.pr.head == HEAD_SHA
            and task.files_changed == EXPECTED['file_count'] and task.lines_added == EXPECTED['additions']
            and task.lines_removed == EXPECTED['deletions'])


def is_final20(tasks, purpose):
    """Recognize only the frozen 20-task corpus, never arbitrary exclusions."""
    return (purpose == 'final-evaluation' and len(tasks) == 20
            and sum(task.task_id == TASK_ID for task in tasks) == 1)


def exclusion_record(task):
    if not matches_declared_task(task):
        raise ValueError('declared final pre-execution exclusion metadata mismatch')
    return {
        'task_id': TASK_ID,
        'status': 'excluded_pre_execution',
        'applies_to_modes': ['baseline', 'retrieval', 'tools', 'full'],
        'reason_codes': REASONS,
        'benchmark_revision': REVISION,
        'base_sha': BASE_SHA,
        'head_sha': HEAD_SHA,
        'benchmark_expected': EXPECTED,
        'pinned_reconstruction': RECONSTRUCTED,
        'unsupported_snapshot_member': SYMLINK,
        'model_calls': 0,
        'judge_calls': 0,
        'counts_as_completed_pr': False,
        'included_in_metric_denominators': False,
    }


def validate_lock_exclusions(subset, exclusions):
    """Only the one documented final-corpus exclusion is permitted."""
    expected_tasks = [task for task in subset.tasks if task.task_id == TASK_ID]
    if subset.purpose != 'final-evaluation':
        if exclusions not in (None, []):
            raise ValueError('pre-execution exclusions are only valid for final evaluation')
        return []
    if not expected_tasks:
        if exclusions not in (None, []):
            raise ValueError('unexpected pre-execution exclusion in lock')
        return []
    if len(expected_tasks) != 1 or not matches_declared_task(expected_tasks[0]):
        raise ValueError('declared final pre-execution exclusion does not match pinned manifest')
    expected = exclusion_record(expected_tasks[0])
    if exclusions != [expected]:
        raise ValueError('final lock exclusion provenance mismatch')
    return [TASK_ID]


def validate_final_lock_budget(lock):
    expected = {
        'request_budget': {
            'ceiling': REQUEST_BUDGET,
            'safety_margin_bytes': SAFETY_MARGIN_BYTES,
            'method': 'serialized_request_body_utf8_bytes_plus_4096_safety_margin',
        },
        'context_byte_budget': CONTEXT_BYTE_BUDGET,
        'static_signal_byte_budget': STATIC_SIGNAL_BYTE_BUDGET,
        'max_output_tokens': MAX_OUTPUT_TOKENS,
        'intended_task_count': 20,
        'evaluable_task_count': 19,
    }
    if any(lock.get(key) != value for key, value in expected.items()):
        raise ValueError('final lock budget or sample provenance mismatch')
