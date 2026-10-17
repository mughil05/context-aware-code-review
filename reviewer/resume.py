"""Offline, fail-closed validation of a prior development run. No network calls."""
from dataclasses import dataclass
import json
from pathlib import Path

from reviewer.benchmark import Subset, digest, export_predictions, read_json
from reviewer.diff import parse_diff
from reviewer.pipeline import review_prompt, review_settings, validate_anchors
from reviewer.schemas import BaselineRun, FindingBatch, ModelAttempt, ReviewInput


class ResumeError(ValueError):
    code = 'incompatible_resume'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')


def read_artifact(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ResumeError('resume artifacts must not use symlinks')
    if not path.is_file():
        raise ResumeError('required resume artifact missing')
    return path.read_bytes()


def compatible_settings(saved, expected, allow_output_budget_change):
    saved = dict(saved)
    before, after = saved.get('max_output_tokens'), expected['max_output_tokens']
    if (allow_output_budget_change and type(before) is int and 0 < before <= after):
        saved['max_output_tokens'] = after
    if canonical(saved) != canonical(expected):
        raise ResumeError('resume model settings differ')


@dataclass
class ResumeTask:
    input: ReviewInput
    run: BaselineRun | None
    raw: bytes | None
    previous_results: list


@dataclass
class ResumePlan:
    source: Path
    summary_sha256: str
    tasks: dict[str, ResumeTask]
    allow_output_budget_change: bool


def validate_resume(source, benchmark, reviewer, *, retry_failed=False,
                    allow_output_budget_change=False):
    """Check every task before any new request; original artifacts stay read-only.

    Only an explicitly allowed output-ceiling increase may differ. Reused runs
    keep their original settings and are never relabeled with the new ceiling.
    """
    try:
        return _validate(source, benchmark, reviewer, retry_failed, allow_output_budget_change)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        if isinstance(error, ResumeError):
            raise
        raise ResumeError('resume artifacts are missing, malformed, or inconsistent') from None


def _validate(source, benchmark, reviewer, retry_failed, allow_output_budget_change):
    source = Path(source).absolute()
    raw_summary = read_artifact(source / 'summary.json')
    summary = read_json(raw_summary)
    expected = {'schema_version': 1, 'benchmark': 'reviewbench',
                'benchmark_revision': benchmark.subset.benchmark_revision,
                'manifest_sha256': benchmark.subset.manifest_sha256,
                'subset_sha256': benchmark.subset_sha256, 'mode': 'baseline',
                'requested_model': reviewer.model, 'scoring_status': 'paused', 'quality_metrics': None}
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ResumeError('resume benchmark, subset, mode, or model differs')
    if Subset.model_validate(read_json(read_artifact(source / 'subset.json'))) != benchmark.subset:
        raise ResumeError('resume subset differs')
    records = summary['tasks']
    if [r['task_id'] for r in records] != [t.task_id for t in benchmark.subset.tasks]:
        raise ResumeError('resume task selection differs')
    settings, prompt = review_settings(reviewer), review_prompt()
    if summary.get('prompt_sha256', digest(prompt.encode())) != digest(prompt.encode()):
        raise ResumeError('resume prompt differs')
    if summary.get('review_settings') is not None:
        compatible_settings(summary['review_settings'], settings, allow_output_budget_change)
    plan = {}
    for task, record in zip(benchmark.subset.tasks, records):
        key = task.task_id
        if (record['pr'] != task.pr.model_dump() or record['mode'] != 'baseline'
                or record['model'] != reviewer.model or record['diff_sha256'] != task.diff_sha256
                or record['golden_sha256'] != task.golden_sha256):
            raise ResumeError('resume task identity differs')
        inp = ReviewInput.model_validate(read_json(read_artifact(source / 'inputs' / (key + '.json'))))
        if (inp.pr != task.pr or digest(inp.diff.encode()) != task.diff_sha256
                or inp.diff_base != record['diff_base']):
            raise ResumeError('resume frozen input differs')
        parsed = parse_diff(inp.diff)
        # Re-run the existing input budget check for every task before any calls.
        payload = reviewer.payload(inp.diff, prompt)
        reference = read_artifact(source / 'reference' / (key + '.json'))
        if (digest(reference) != record['reference_export_sha256']
                or read_json(reference) != benchmark.golden[key]):
            raise ResumeError('resume reference differs')
        status = record['status']
        if status not in ('ready', 'success', 'failed'):
            raise ResumeError('resume task has unresolved preflight state')
        raw = run = None
        previous = list(record.get('previous_results', []))
        for historical in previous:
            history_raw = read_artifact(Path(historical['result_file']))
            history = BaselineRun.model_validate(read_json(history_raw))
            if (digest(history_raw) != historical['result_sha256']
                    or historical['status'] != 'failed' or history.result.status != 'failed'
                    or history.result.pr != task.pr or history.result.mode != 'baseline'
                    or history.requested_model != reviewer.model
                    or history.diff_sha256 != task.diff_sha256
                    or history.prompt_sha256 != digest(prompt.encode())
                    or historical['settings'] != history.settings
                    or [ModelAttempt.model_validate(a) for a in historical['attempts']] != history.attempts):
                raise ResumeError('resume failure history differs')
        relative = f'runs/{key}/baseline.json'
        raw_path = source / relative
        candidate_path = source / 'candidates' / (key + '.json')
        if status == 'ready':
            if raw_path.exists() or candidate_path.exists() or record.get('attempts'):
                raise ResumeError('unrecorded task attempt requires inspection')
        else:
            if record.get('result_file') != relative:
                raise ResumeError('resume result path differs')
            raw = read_artifact(raw_path)
            if record.get('result_sha256', digest(raw)) != digest(raw):
                raise ResumeError('resume result hash differs')
            run = BaselineRun.model_validate(read_json(raw))
            if (run.result.pr != task.pr or run.result.mode != 'baseline'
                    or run.result.status != status or run.provider != 'openai'
                    or run.requested_model != reviewer.model or run.diff_base != inp.diff_base
                    or run.diff_sha256 != task.diff_sha256 or run.prompt_text != prompt
                    or run.prompt_sha256 != digest(prompt.encode())
                    or run.output_schema != FindingBatch.model_json_schema()):
                raise ResumeError('resume result identity, prompt, or schema differs')
            compatible_settings(run.settings, settings, allow_output_budget_change)
            original_payload = dict(payload, max_output_tokens=run.settings['max_output_tokens'])
            if run.request_sha256 != digest(canonical(original_payload)):
                raise ResumeError('resume request hash differs')
            if (len(run.attempts) != 1 or run.attempts[0].number != 1
                    or run.attempts[0].returned_model not in (None, reviewer.model)
                    or [ModelAttempt.model_validate(a) for a in record['attempts']] != run.attempts
                    or record['predictions'] != [f.model_dump() for f in run.result.findings]):
                raise ResumeError('resume attempts or findings are inconsistent')
            if status == 'success':
                attempt = run.attempts[0]
                if (attempt.error_code is not None or attempt.returned_model != reviewer.model
                        or attempt.response_status not in (None, 'completed')):
                    raise ResumeError('successful result contains failed attempt')
                batch = FindingBatch.from_json(run.output_text)
                if batch.findings != run.result.findings:
                    raise ResumeError('resume raw output differs from findings')
                validate_anchors(batch, parsed)
                candidate = read_artifact(candidate_path)
                if (record.get('prediction_file') != f'candidates/{key}.json'
                        or digest(candidate) != record['prediction_sha256']
                        or read_json(candidate) != export_predictions(run)):
                    raise ResumeError('resume candidate differs')
            else:
                if candidate_path.exists():
                    raise ResumeError('failed task has a candidate export')
                previous.append({'result_file': str(raw_path), 'result_sha256': digest(raw),
                                 'status': 'failed', 'settings': run.settings,
                                 'attempts': [a.model_dump() for a in run.attempts],
                                 'model_latency_seconds': run.model_latency_seconds,
                                 'total_latency_seconds': run.total_latency_seconds})
        if (status == 'failed' or (status == 'ready' and previous)) and not retry_failed:
            raise ResumeError('failed tasks require --retry-failed')
        plan[key] = ResumeTask(inp, run, raw, previous)
    if summary.get('review_settings') is None and not any(t.run for t in plan.values()):
        raise ResumeError('legacy run lacks saved model settings')
    return ResumePlan(source, digest(raw_summary), plan, allow_output_budget_change)
