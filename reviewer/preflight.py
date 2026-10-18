"""No-model reconstruction and A/B/C/D payload preflight for a frozen subset."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
import uuid

import pydantic

from reviewer.archives import sha256
from reviewer.benchmark import REVISION, digest, load_benchmark, reconstruct
from reviewer.diff import parse_diff
from reviewer.models import build_review_payload
from reviewer.payload_validation import validate_reviewer_payloads
from reviewer.request_budget import measure_request, BUDGET_METHOD, SAFETY_MARGIN_BYTES
from reviewer.pipeline import prepare_context, prepare_static_analysis, review_prompt, review_settings
from reviewer.static_analysis import get_bandit_version
from reviewer.verification import PROMPT_VERSION as VERIFY_PROMPT_VERSION, verification_prompt
from reviewer.final_exclusions import (TASK_ID as FINAL_EXCLUDED_TASK_ID, exclusion_record,
                                        REQUEST_BUDGET, CONTEXT_BYTE_BUDGET,
                                        STATIC_SIGNAL_BYTE_BUDGET, MAX_OUTPUT_TOKENS, is_final20)


class PayloadOnlyReviewer:
    """Shares the production payload builder but has no transport/API key."""
    def __init__(self, model, config):
        self.model, self.config = model, config

    def payload(self, diff, prompt, *, context=None, static_signals=None):
        return build_review_payload(self.model, self.config, diff, prompt,
                                    context=context, static_signals=static_signals)


class PreflightFailure(ValueError):
    """A safe preflight failure with its persisted report location."""
    def __init__(self, *, stage, message, code='preflight_error', task_id=None, report_path=None):
        self.stage = stage
        self.message = message
        self.code = code
        self.task_id = task_id
        self.report_path = report_path
        super().__init__(message)


def _safe_error(error):
    """Return a useful error while preventing configured credentials from persisting."""
    code = getattr(error, 'code', None)
    if not isinstance(code, str) or not re.fullmatch(r'[a-zA-Z0-9_.-]{1,80}', code):
        code = type(error).__name__
    message = str(error).strip() or type(error).__name__
    # Provider/GitHub errors should already be sanitized. This is a final guard
    # against accidentally persisting credentials in arbitrary exception text.
    secrets = [os.environ.get(name, '') for name in
               ('OPENAI_API_KEY', 'GITHUB_TOKEN', 'REVIEW_API_KEY')]
    for secret in secrets:
        if secret:
            message = message.replace(secret, '[REDACTED]')
    message = re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*', r'\1[REDACTED]', message)
    message = re.sub(r'(?i)(api[_ -]?key|token)(\s*[=:]\s*)[^\s,;]+', r'\1\2[REDACTED]', message)
    return code, message


def _write_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp',
                                         delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _task_progress(task):
    return {
        'task_id': task.task_id, 'status': 'in_progress', 'stage': 'acquisition',
        'error_code': None, 'error': None,
        'repo': task.pr.repo, 'pr_number': task.pr.pr_number,
        'base_sha': task.pr.base, 'head_sha': task.pr.head,
        'diff_sha256': task.diff_sha256, 'head_archive_sha256': None,
        'acquisition_succeeded': False, 'diff_validation_succeeded': False,
        'snapshot_acquisition_succeeded': False, 'retrieval_succeeded': False,
        'bandit_succeeded': False, 'budget_checks_succeeded': False,
        'leakage_checks_succeeded': False,
    }


def run_preflight(*, archive, subset_path, config, client, results_dir,
                  review_model, review_provider, verify_model=None, verify_provider=None,
                  cache_dir='.cache', expected_bandit='1.7.8'):
    """Reconstruct each task once; no reviewer/verifier instance or model call exists."""
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    target = results / ('reviewbench-preflight-' + uuid.uuid4().hex + '.json')
    report = {
        'schema_version': 1, 'status': 'running', 'model_calls': 0,
        'benchmark_revision': REVISION, 'benchmark_archive_sha256': None,
        'benchmark_manifest_sha256': None, 'subset_sha256': None,
        'created_at': datetime.now(timezone.utc).isoformat(), 'reviewer_model': review_model,
        'reviewer_provider': review_provider, 'verifier_model': verify_model,
        'verifier_provider': verify_provider, 'task_ids': [], 'tasks': [],
        'request_budget': {'ceiling': config.review.request_byte_budget,
                           'safety_margin_bytes': SAFETY_MARGIN_BYTES, 'method': BUDGET_METHOD},
        'context_byte_budget': config.retrieval.context_byte_budget,
        'static_signal_byte_budget': config.static_analysis.signal_byte_budget,
    }
    stage = 'configuration'
    task_id = None
    def persist():
        _write_atomic(target, report)
    def fail(error, *, failed_stage=None, failed_task_id=None):
        code, message = _safe_error(error)
        report.update(status='failed', failure={'task_id': failed_task_id,
                      'stage': failed_stage or stage, 'error_code': code, 'error': message})
        if failed_task_id is not None:
            entry = next((row for row in report['tasks'] if row['task_id'] == failed_task_id), None)
            if entry is not None:
                entry.update(status='failed', stage=failed_stage or stage,
                             error_code=code, error=message)
            report['failed_task_count'] = 1
            report['completed_task_count'] = sum(row.get('status') == 'passed'
                                                 for row in report['tasks'])
        persist()
        return PreflightFailure(stage=failed_stage or stage, task_id=failed_task_id,
                                code=code, message=message, report_path=target)
    persist()
    try:
        if (review_provider != 'openai' or not isinstance(review_model, str)
                or not re.fullmatch(r'[A-Za-z0-9_.:-]+', review_model)):
            raise ValueError('preflight requires configured review model/provider names')
        if (verify_provider != 'openai' or not isinstance(verify_model, str)
                or not re.fullmatch(r'[A-Za-z0-9_.:-]+', verify_model)):
            raise ValueError('A-D preflight requires configured verifier model/provider names')
        if config.review.request_byte_budget <= 0:
            raise ValueError('positive serialized-request byte estimator ceiling required')
        config.review.max_attempts = 1
        stage = 'benchmark_validation'
        benchmark = load_benchmark(archive, subset_path, allow_unpinned_diffs=True)
        if benchmark.subset.benchmark_revision != REVISION:
            raise ValueError('benchmark revision mismatch')
        report.update(benchmark_archive_sha256=benchmark.archive_sha256,
                      benchmark_manifest_sha256=benchmark.subset.manifest_sha256,
                      subset_sha256=benchmark.subset_sha256,
                      task_ids=[task.task_id for task in benchmark.subset.tasks])
        exclusions = []
        final20 = is_final20(benchmark.subset.tasks, benchmark.subset.purpose)
        if final20:
            matches = [task for task in benchmark.subset.tasks if task.task_id == FINAL_EXCLUDED_TASK_ID]
            if len(matches) != 1:
                raise ValueError('final evaluation must contain the one predeclared Nipoppy exclusion')
            exclusions = [exclusion_record(matches[0])]
            if len(benchmark.subset.tasks) != 20:
                raise ValueError('final evaluation intended task count must remain 20')
            if (config.review.request_byte_budget != REQUEST_BUDGET
                    or config.retrieval.context_byte_budget != CONTEXT_BYTE_BUDGET
                    or config.static_analysis.signal_byte_budget != STATIC_SIGNAL_BYTE_BUDGET
                    or config.review.max_output_tokens != MAX_OUTPUT_TOKENS):
                raise ValueError('final evaluation byte budgets differ from the frozen sizing plan')
            report.update(intended_task_count=20, evaluable_task_count=19,
                          excluded_pre_execution_count=1, excluded_tasks=exclusions,
                          completed_task_count=0, failed_task_count=0)
        stage = 'static_analysis'
        reviewer = PayloadOnlyReviewer(review_model, config.review)
        prompt = review_prompt()
        bandit_version = get_bandit_version()
        if bandit_version != expected_bandit:
            raise ValueError(f'expected Bandit {expected_bandit}; found {bandit_version or "unavailable"}')
    except Exception as error:
        raise fail(error) from None
    tasks = []
    updated_tasks = []
    for task in benchmark.subset.tasks:
        if final20 and task.task_id == FINAL_EXCLUDED_TASK_ID:
            # Fixed, predeclared exclusion: do not acquire, snapshot, analyze, or call a model.
            continue
        task_id = task.task_id
        stage = 'acquisition'
        progress = _task_progress(task)
        report['tasks'].append(progress)
        persist()
        started = time.perf_counter()
        try:
            inp = reconstruct(task, client)
            if inp.pr != task.pr:
                raise ValueError('reconstructed PR identity mismatch')
            for warning in inp.acquisition_warnings:
                if (warning.base_sha != task.pr.base or warning.head_sha != task.pr.head
                        or warning.expected_file_count != task.files_changed
                        or warning.expected_additions != task.lines_added
                        or warning.expected_deletions != task.lines_removed):
                    raise ValueError('benchmark aggregate warning does not match pinned task metadata')
            progress['acquisition_succeeded'] = True
            stage = progress['stage'] = 'diff_validation'
            persist()
            parsed = parse_diff(inp.diff)
            diff_hash = sha256(inp.diff.encode('utf-8'))
            if task.diff_sha256 is not None and task.diff_sha256 != diff_hash:
                raise ValueError('frozen diff hash mismatch')
            reconstructed_counts = (
                len(parsed),
                sum(line.kind == 'added' for file in parsed for hunk in file.hunks for line in hunk.lines),
                sum(line.kind == 'deleted' for file in parsed for hunk in file.hunks for line in hunk.lines),
            )
            for warning in inp.acquisition_warnings:
                if (warning.reconstructed_file_count, warning.reconstructed_additions,
                        warning.reconstructed_deletions) != reconstructed_counts:
                    raise ValueError('benchmark aggregate warning does not match parsed frozen diff')
            progress.update(diff_validation_succeeded=True, diff_sha256=diff_hash)
            stage = progress['stage'] = 'snapshot'
            persist()
            snapshot = client.frozen_head(task.pr, source_repository=task.source_repository,
                                          cache_dir=Path(cache_dir) / 'github')
            if snapshot.commit != task.pr.head:
                raise ValueError('frozen head snapshot mismatch')
            progress.update(snapshot_acquisition_succeeded=True,
                            head_archive_sha256=snapshot.archive_sha256)
            persist()
            stage = progress['stage'] = 'retrieval'
            persist()
            context = prepare_context(inp, reviewer, snapshot, config.retrieval,
                                      cache_dir=Path(cache_dir) / 'retrieval',
                                      reserve_static_signal_byte_cap=config.static_analysis.signal_byte_budget)
            progress['retrieval_succeeded'] = True
            stage = progress['stage'] = 'static_analysis'
            persist()
            static = prepare_static_analysis(inp, snapshot, context, config.static_analysis)
            if static.status not in ('success', 'no_findings', 'no_files'):
                raise ValueError('Bandit preflight failed')
            if static.version != expected_bandit:
                raise ValueError('Bandit result version mismatch')
            progress['bandit_succeeded'] = True
            stage = progress['stage'] = 'budget_validation'
            persist()
            context_payload = context.payload()
            signals = static.payload(config.static_analysis.signal_byte_budget)
            trusted_context = deepcopy(context_payload)
            trusted_signals = deepcopy(signals)
            payloads = {
                'baseline': reviewer.payload(inp.diff, prompt),
                'retrieval': reviewer.payload(inp.diff, prompt, context=context_payload),
                'tools': reviewer.payload(inp.diff, prompt, context=context_payload, static_signals=signals),
                'full': reviewer.payload(inp.diff, prompt, context=context_payload, static_signals=signals),
            }
            progress['budget_checks_succeeded'] = True
            stage = progress['stage'] = 'leakage_validation'
            persist()
            validate_reviewer_payloads(
                payloads, frozen_diff=inp.diff, diff_sha256=diff_hash,
                context=trusted_context, signals=trusted_signals,
                prompt=review_prompt(), model=review_model, config=config.review)
            progress['leakage_checks_succeeded'] = True
            task_record = {
            'task_id': task.task_id, 'repo': task.pr.repo, 'pr_number': task.pr.pr_number,
            'base_sha': task.pr.base, 'head_sha': task.pr.head, 'diff_sha256': diff_hash,
            'diff_files': len(parsed), 'diff_added': sum(l.kind == 'added' for f in parsed for h in f.hunks for l in h.lines),
            'diff_removed': sum(l.kind == 'deleted' for f in parsed for h in f.hunks for l in h.lines),
            'head_archive_sha256': snapshot.archive_sha256,
            'request_byte_measurements': {mode: measure_request(payload, config.review.request_byte_budget)
                                         for mode, payload in payloads.items()},
            'retrieval_chunks': context.model_dump(),
            'bandit': static.model_dump(),
            'acquisition_warnings': [warning.model_dump(mode='json')
                                     for warning in inp.acquisition_warnings],
            'preparation_seconds': max(0.0, time.perf_counter() - started),
            'benchmark_labels_sent': False,
            }
            progress.update(task_record, status='passed', stage='complete')
            tasks.append(progress)
            updated_tasks.append(task.model_copy(update={'diff_sha256': diff_hash}))
            report['completed_task_count'] = len(updated_tasks)
            persist()
        except Exception as error:
            raise fail(error, failed_stage=stage, failed_task_id=task.task_id) from None
    # Keep final selection immutable; freeze acquired diffs beside it.
    lock_bytes = b''
    stage = 'finalization'
    try:
      if benchmark.subset.purpose == 'final-evaluation':
        lock = {'schema_version': 1, 'benchmark_revision': REVISION,
                'subset_sha256': benchmark.subset_sha256,
                'benchmark_archive_sha256': benchmark.archive_sha256,
                'benchmark_manifest_sha256': benchmark.subset.manifest_sha256,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'intended_task_count': len(benchmark.subset.tasks),
                'evaluable_task_count': len(updated_tasks),
                'pre_execution_exclusions': exclusions,
                'request_budget': {'ceiling': config.review.request_byte_budget,
                                   'safety_margin_bytes': SAFETY_MARGIN_BYTES,
                                   'method': BUDGET_METHOD},
                'context_byte_budget': config.retrieval.context_byte_budget,
                'static_signal_byte_budget': config.static_analysis.signal_byte_budget,
                'max_output_tokens': config.review.max_output_tokens,
                'review_settings': review_settings(reviewer),
                'reviewer_provider': review_provider, 'reviewer_model': review_model,
                'review_prompt_sha256': digest(prompt.encode()),
                'review_prompt_version': 'v3',
                'verifier_prompt_sha256': digest(verification_prompt().encode()),
                'verifier_prompt_version': VERIFY_PROMPT_VERSION,
                'verifier_provider': verify_provider, 'verifier_model': verify_model,
                'verifier_settings': {'provider': verify_provider, 'model': verify_model,
                                      'max_output_tokens': config.review.max_output_tokens,
                                      'temperature': config.review.temperature,
                                      'request_byte_budget': config.review.request_byte_budget,
                                      'request_budget_method': BUDGET_METHOD},
                'retrieval_config': config.retrieval.model_dump(),
                'static_analysis': {'tool': 'bandit', 'version': bandit_version,
                                    'timeout_seconds': config.static_analysis.timeout_seconds},
                'tasks': []}
        warnings_by_id = {task['task_id']: task.get('acquisition_warnings', []) for task in tasks}
        for task in updated_tasks:
            entry = {'task_id': task.task_id, 'base_sha': task.pr.base,
                     'head_sha': task.pr.head, 'diff_sha256': task.diff_sha256,
                     'head_archive_sha256': next(row['head_archive_sha256'] for row in tasks
                                                 if row['task_id'] == task.task_id)}
            warnings = warnings_by_id.get(task.task_id, [])
            if warnings:
                entry['acquisition_warnings'] = warnings
            lock['tasks'].append(entry)
        lock_path = Path(subset_path).with_suffix('.lock.json')
        lock_bytes = (json.dumps(lock, sort_keys=True, indent=2) + '\n').encode()
        if lock_path.exists():
            if lock_path.read_bytes() != lock_bytes:
                raise ValueError('existing final subset diff lock differs; refusing to overwrite')
        else:
            with lock_path.open('xb') as stream:
                stream.write(lock_bytes)
    except Exception as error:
        raise fail(error, failed_stage='finalization') from None
    subset_hash = benchmark.subset_sha256
    report.update({
        'schema_version': 1, 'status': 'ready', 'model_calls': 0,
        'benchmark_revision': REVISION, 'benchmark_archive_sha256': benchmark.archive_sha256,
        'benchmark_manifest_sha256': benchmark.subset.manifest_sha256,
        'diff_lock_sha256': digest(lock_bytes) if lock_bytes else None,
        'subset_sha256': subset_hash, 'task_ids': [task.task_id for task in benchmark.subset.tasks],
        'request_budget': {'ceiling': config.review.request_byte_budget,
                           'safety_margin_bytes': SAFETY_MARGIN_BYTES, 'method': BUDGET_METHOD},
        'context_byte_budget': config.retrieval.context_byte_budget,
        'static_signal_byte_budget': config.static_analysis.signal_byte_budget,
        'created_at': datetime.now(timezone.utc).isoformat(), 'reviewer_model': review_model,
        'reviewer_provider': review_provider, 'verifier_model': verify_model,
        'verifier_provider': verify_provider,
        'review_settings': review_settings(reviewer),
        'review_prompt_sha256': digest(prompt.encode()),
        'review_prompt_version': 'v3',
        'verifier_prompt_sha256': digest(verification_prompt().encode()),
        'verifier_prompt_version': VERIFY_PROMPT_VERSION,
        'verifier_settings': {'model': verify_model, 'provider': verify_provider,
                              'max_output_tokens': config.review.max_output_tokens,
                              'temperature': config.review.temperature,
                              'request_byte_budget': config.review.request_byte_budget},
        'retrieval_config': config.retrieval.model_dump(),
        'environment': {'python': sys.version.split()[0], 'platform': platform.platform(),
                        'pydantic': pydantic.__version__},
        'bandit': {'version': bandit_version, 'tool': 'bandit',
                   'timeout_seconds': config.static_analysis.timeout_seconds,
                   'signal_byte_budget': config.static_analysis.signal_byte_budget},
        'tasks': tasks,
        'intended_task_count': len(benchmark.subset.tasks),
        'evaluable_task_count': len(updated_tasks),
        'completed_task_count': len(updated_tasks),
        'failed_task_count': 0,
        'excluded_pre_execution_count': len(exclusions),
        'excluded_tasks': exclusions,
        'status': 'ready',
    })
    persist()
    return target, report
