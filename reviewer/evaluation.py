"""Five-task baseline generation and export. Scoring is paused by user decision.

No matcher, judge, or quality metric computation is implemented here.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Annotated
import uuid

from pydantic import Field

from reviewer.benchmark import digest, export_predictions, qualifying_count, reconstruct
from reviewer.pipeline import review_prompt, review_settings, run_baseline, run_review, prepare_context
from reviewer.schemas import StrictModel, Text
from reviewer.storage import save_run


class Pricing(StrictModel):
    model: Text
    input_usd_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    output_usd_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    cached_input_usd_per_million: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    source: Text
    as_of: Text


def cost_estimate(run, pricing):
    if pricing is None or not run.attempts or any(a.usage is None for a in run.attempts):
        return None
    if pricing.model != run.requested_model:
        raise ValueError('pricing model differs from requested review model')
    total = 0.0
    for attempt in run.attempts:
        u = attempt.usage
        cached = u.cached_input_tokens or 0
        total += ((u.input_tokens - cached) * pricing.input_usd_per_million
                  + cached * pricing.cached_input_usd_per_million
                  + u.output_tokens * pricing.output_usd_per_million) / 1_000_000
    return total


def write_json(path, value):
    """Exclusive artifact creation; a run is never merged with previous output."""
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def _persist_result(root, task, run, record, pricing, *, reused_raw=None):
    if reused_raw is None:
        raw_path = save_run(run, root / 'runs', run_id=task.task_id)
    else:
        directory = root / 'runs' / task.task_id
        directory.mkdir(exist_ok=False)
        raw_path = directory / 'baseline.json'
        with raw_path.open('xb') as stream:
            stream.write(reused_raw)
    record.update(status=run.result.status, result_file=str(raw_path.relative_to(root)),
                  result_sha256=digest(raw_path.read_bytes()), settings=run.settings,
                  predictions=[f.model_dump() for f in run.result.findings],
                  attempts=[a.model_dump() for a in run.attempts],
                  model_latency_seconds=run.model_latency_seconds,
                  total_latency_seconds=run.total_latency_seconds,
                  retrieval_latency_seconds=run.retrieval_latency_seconds,
                  approximate_cost_usd=cost_estimate(run, pricing), error=run.result.error)
    if run.result.status == 'success':
        candidate = root / 'candidates' / (task.task_id + '.json')
        write_json(candidate, export_predictions(run))
        record['prediction_sha256'] = digest(candidate.read_bytes())
        record['prediction_file'] = str(candidate.relative_to(root))


def run_development_subset(benchmark, reviewer, client, results_dir, *, pricing=None,
                           resume_from=None, retry_failed=False, allow_output_budget_change=False,
                           modes=None, retrieval_config=None, cache_dir='.cache'):
    """Preflight all five frozen inputs before spending on any review.

    Fail fast on the first model failure, retaining completed and failed raw runs.
    Resume forks a new directory, preserving validated successes byte-for-byte.
    No automatic retries; max_attempts must be one. The three repetitions in
    the future experiment config do not apply to this development smoke run.
    """
    modes = ['baseline'] if modes is None else modes
    if not modes or len(set(modes)) != len(modes) or any(m not in ('baseline', 'retrieval') for m in modes):
        raise ValueError('only unique baseline/retrieval modes are implemented')
    if 'retrieval' in modes and (retrieval_config is None or resume_from is not None):
        raise ValueError('retrieval requires configuration; retrieval resume is not implemented')
    if reviewer.config.max_attempts != 1:
        raise ValueError('development subset requires max_attempts=1')
    if pricing is not None and pricing.model != reviewer.model:
        raise ValueError('pricing model differs from review model')
    if resume_from is None and (retry_failed or allow_output_budget_change):
        raise ValueError('resume flags require --resume-from')
    plan = None
    if resume_from is not None:
        from reviewer.resume import validate_resume
        plan = validate_resume(resume_from, benchmark, reviewer, retry_failed=retry_failed,
                               allow_output_budget_change=allow_output_budget_change)
    parent = Path(results_dir)
    if any(p.is_symlink() for p in (parent, *parent.parents)):
        raise ValueError('results directory cannot contain symlink ancestors')
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / ('reviewbench-dev5-' + uuid.uuid4().hex)
    root.mkdir()
    for name in ('inputs', 'candidates', 'reference', 'runs'):
        (root / name).mkdir()
    report = {'schema_version': 1, 'benchmark': 'reviewbench',
              'benchmark_revision': benchmark.subset.benchmark_revision,
              'manifest_sha256': benchmark.subset.manifest_sha256,
              'subset_sha256': benchmark.subset_sha256, 'mode': ','.join(modes), 'modes': modes,
              'requested_model': reviewer.model, 'created_at': datetime.now(timezone.utc).isoformat(),
              'status': 'preflight', 'scoring_status': 'paused', 'quality_metrics': None,
              'pricing': pricing.model_dump() if pricing else None,
              'tasks': [], 'completed_reviews': 0, 'model_attempts': 0, 'reused_reviews': 0,
              'review_settings': review_settings(reviewer),
              'prompt_sha256': digest(review_prompt().encode()),
              'resume_from': str(plan.source) if plan else None,
              'resume_summary_sha256': plan.summary_sha256 if plan else None,
              'output_budget_change_allowed': allow_output_budget_change}
    write_json(root / 'subset.json', benchmark.subset.model_dump())
    prepared = []
    try:
        for task in benchmark.subset.tasks:
            record = {'task_id': task.task_id, 'pr': task.pr.model_dump(),
                      'status': 'preflight', 'scoring_status': 'paused', 'quality_metrics': None,
                      'golden_sha256': task.golden_sha256,
                      'recorded_tp_count': sum(f['tp_fp'] == 'tp' for f in benchmark.golden[task.task_id]['findings']),
                      'qualifying_tp_count': qualifying_count(benchmark.golden[task.task_id]),
                      'diff_sha256': task.diff_sha256, 'model': reviewer.model, 'mode': modes[0]}
            report['tasks'].append(record)
            started = time.perf_counter()
            prior = plan.tasks[task.task_id] if plan else None
            inp = prior.input if prior else reconstruct(task, client)
            elapsed = 0.0 if prior else time.perf_counter() - started
            record['previous_results'] = prior.previous_results if prior else []
            record['reused'] = bool(prior and prior.run and prior.run.result.status == 'success')
            # Build but never send here. All size/configuration failures precede calls.
            payload = reviewer.payload(inp.diff, review_prompt())
            record['conservative_input_estimate'] = len(json.dumps(payload, ensure_ascii=False).encode()) + 4096
            record['diff_base'] = inp.diff_base
            record['status'] = 'ready'
            write_json(root / 'inputs' / (task.task_id + '.json'), inp.model_dump())
            # Label artifacts are not arguments to run_baseline or the model adapter.
            reference = root / 'reference' / (task.task_id + '.json')
            write_json(reference, benchmark.golden[task.task_id])
            record['reference_export_sha256'] = digest(reference.read_bytes())
            context, retrieval_elapsed = None, 0.0
            if 'retrieval' in modes:
                retrieval_started = time.perf_counter()
                snapshot = client.frozen_head(task.pr, source_repository=task.source_repository,
                                              cache_dir=Path(cache_dir) / 'github')
                context = prepare_context(inp, reviewer, snapshot, retrieval_config,
                                          cache_dir=Path(cache_dir) / 'retrieval')
                retrieval_elapsed = time.perf_counter() - retrieval_started
            report['tasks'].pop()  # Expand this shared frozen input into the requested A/B records.
            for mode in modes:
                mode_record = dict(record, mode=mode)
                if mode == 'retrieval':
                    payload = reviewer.payload(inp.diff, review_prompt(), context=context.payload())
                    mode_record['conservative_input_estimate'] = len(json.dumps(payload, ensure_ascii=False).encode()) + 4096
                report['tasks'].append(mode_record)
                mode_root = root if modes == ['baseline'] else root / 'modes' / mode
                if mode_root != root:
                    for directory in ('runs', 'candidates'):
                        (mode_root / directory).mkdir(parents=True, exist_ok=True)
                prepared.append((task, inp, elapsed, mode_record, mode_root,
                                 context if mode == 'retrieval' else None,
                                 retrieval_elapsed if mode == 'retrieval' else 0.0))
        # Copy every success before attempting new work, including successes that
        # appear after a failed task. A later failure cannot lose reusable work.
        if plan:
            for task, inp, elapsed, record, mode_root, context, retrieval_elapsed in prepared:
                if record['reused']:
                    prior = plan.tasks[task.task_id]
                    _persist_result(root, task, prior.run, record, pricing, reused_raw=prior.raw)
                    record['reused_from'] = str(plan.source / 'runs' / task.task_id / 'baseline.json')
                    report['completed_reviews'] += 1
                    report['reused_reviews'] += 1
        for task, inp, elapsed, record, mode_root, context, retrieval_elapsed in prepared:
            if record['reused']:
                continue
            if record['mode'] == 'baseline':
                run = run_baseline(inp, reviewer, acquisition_latency_seconds=elapsed,
                                   input_artifact_directory=str((root / 'inputs').resolve()))
            else:
                run = run_review(inp, reviewer, mode='retrieval', context=context,
                                 retrieval_latency_seconds=retrieval_elapsed, acquisition_latency_seconds=elapsed,
                                 input_artifact_directory=str((root / 'inputs').resolve()))
            _persist_result(mode_root, task, run, record, pricing)
            if mode_root != root:
                for key in ('result_file', 'prediction_file'):
                    if key in record:
                        record[key] = str(mode_root.relative_to(root) / record[key])
            report['model_attempts'] += len(run.attempts)
            if run.result.status != 'success':
                report['status'] = 'failed'
                break
            report['completed_reviews'] += 1
        else:
            report['status'] = 'generation_complete_scoring_paused'
    except (ValueError, OSError, KeyError, TypeError) as error:
        # Never serialize exception bodies, config values, or credentials.
        report['status'] = 'failed'
        report['error'] = 'benchmark preflight, review, or persistence failed; no quality score produced'
        report['error_code'] = getattr(error, 'code', 'invalid_benchmark_or_artifact')
        if report['tasks']:
            record['status'] = 'failed'
    finally:
        completed = [r for r in report['tasks'] if r.get('status') == 'success']
        costs = [r.get('approximate_cost_usd') for r in completed]
        report['completed_review_cost_usd'] = sum(costs) if costs and all(c is not None for c in costs) else None
        report['recorded_model_attempts_including_history'] = sum(
            len(r.get('attempts', [])) + sum(len(p['attempts']) for p in r.get('previous_results', []))
            for r in report['tasks'])
        write_json(root / 'summary.json', report)
    return root / 'summary.json', report
