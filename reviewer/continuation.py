"""Strict successful-prefix continuation of the frozen final A/B/C/D experiment.

Original directories are read-only. No arbitrary observation selection, setting
changes, automatic retries, or scoring are supported.
"""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from reviewer.archives import sha256
from reviewer.benchmark import export_predictions
from reviewer.diff import parse_diff
from reviewer.final_exclusions import TASK_ID, FINAL_SUBSET_SHA256, exclusion_record, is_final20
from reviewer.models import build_review_payload, ModelReply
from reviewer.pipeline import (review_prompt, review_settings, validate_anchors,
                               prepare_context, prepare_static_analysis, run_baseline,
                               run_review, run_full)
from reviewer.request_budget import measure_request
from reviewer.resume import canonical, read_artifact, ResumeError
from reviewer.schemas import BaselineRun, ReviewInput, FindingBatch
from reviewer.static_analysis import StaticAnalysisResult
from reviewer.verification import (_evidence, verification_prompt, VerifierBatch,
                                   proposal_identifier)

MODES = ['baseline', 'retrieval', 'tools', 'full']
ORIGIN = 'reviewbench-18579bfaf3d04e02bd9d38faebdbd077'
# Bind the initial continuation to the already-audited run, not arbitrary results.
ORIGIN_SUMMARY_SHA256 = '3f51bf0dd6b05863196e322f38481481dc4457a70c39b20554c2021a3b2af43a'


def require(condition, message):
    if not condition:
        raise ResumeError(message)


def read_json(path):
    return json.loads(read_artifact(path))


def request_hash(payload):
    return sha256(canonical(payload))


def reviewer_payload(inp, run, reviewer):
    static = StaticAnalysisResult.model_validate(run.static_analysis) if run.static_analysis else None
    return build_review_payload(reviewer.model, reviewer.config, inp.diff, review_prompt(),
        context=run.retrieval.payload() if run.retrieval else None,
        static_signals=static.payload(static.config['signal_byte_budget']) if static else None)


def observation_path(task_id, mode):
    return f'modes/{mode}/runs/{task_id}/{mode}.json'


def candidate_path(task_id, mode):
    return f'modes/{mode}/candidates/{task_id}.json'


@dataclass
class PrefixPlan:
    source: Path
    summary: dict
    summary_hash: str
    inputs: dict
    runs: dict
    prefix: int
    artifacts: dict
    history: list


def validate_source(source, benchmark, reviewer, verifier, retrieval_config, static_config,
                    *, _seen=None):
    """Read-only audit. Validate ancestry as well as the current successful prefix."""
    source = Path(source).absolute()
    seen = set() if _seen is None else set(_seen)
    require(str(source) not in seen, 'continuation ancestry cycle')
    seen.add(str(source))
    require(is_final20(benchmark.subset.tasks, benchmark.subset.purpose)
            and benchmark.subset_sha256 == FINAL_SUBSET_SHA256,
            'continuation requires the frozen final20 manifest')
    lock = benchmark.lock_provenance
    require(isinstance(lock, dict) and benchmark.diff_lock_sha256, 'continuation requires final lock')
    summary_raw = read_artifact(source / 'summary.json')
    summary = json.loads(summary_raw)
    require(summary['status'] == 'failed', 'source must be a failed final run')
    require(summary['modes'] == MODES and summary['mode'] == ','.join(MODES), 'mode/order mismatch')
    tasks = [t for t in benchmark.subset.tasks if t.task_id != TASK_ID]
    excluded = [exclusion_record(next(t for t in benchmark.subset.tasks if t.task_id == TASK_ID))]
    expected = {'benchmark_revision': benchmark.subset.benchmark_revision,
        'benchmark_archive_sha256': benchmark.archive_sha256,
        'manifest_sha256': benchmark.subset.manifest_sha256,
        'subset_sha256': benchmark.subset_sha256, 'diff_lock_sha256': benchmark.diff_lock_sha256,
        'intended_task_count': 20, 'evaluable_task_count': 19, 'requested_reviews': 76,
        'excluded_pre_execution_count': 1, 'excluded_tasks': excluded,
        'scoring_status': 'paused', 'quality_metrics': None,
        'requested_model': reviewer.model, 'verifier_model': verifier.model,
        'verifier_provider': 'openai', 'review_settings': review_settings(reviewer),
        'verifier_settings': review_settings(verifier),
        'prompt_sha256': sha256(review_prompt().encode()),
        'verifier_prompt_sha256': sha256(verification_prompt().encode()),
        'review_prompt_version': 'v3', 'verifier_prompt_version': '1',
        'context_byte_budget': 6000, 'static_signal_byte_budget': 2000,
        'retrieval_config': retrieval_config.model_dump(),
        'static_analysis_config': static_config.model_dump()}
    require(all(summary.get(k) == v for k, v in expected.items()), 'source frozen experiment mismatch')
    require(reviewer.model == verifier.model == lock['reviewer_model'] == lock['verifier_model']
            and lock['reviewer_provider'] == lock['verifier_provider'] == 'openai', 'locked model/provider mismatch')
    require(review_settings(reviewer) == lock['review_settings']
            and review_settings(verifier) == lock['review_settings'], 'locked settings mismatch')
    require(lock['review_prompt_sha256'] == expected['prompt_sha256']
            and lock['verifier_prompt_sha256'] == expected['verifier_prompt_sha256']
            and lock['review_prompt_version'] == 'v3' and lock['verifier_prompt_version'] == '1',
            'locked prompt mismatch')
    require(lock['retrieval_config'] == retrieval_config.model_dump()
            and lock['static_analysis'] == {'tool': static_config.tool, 'timeout_seconds': static_config.timeout_seconds,
                                            'version': '1.7.8'}
            and static_config.signal_byte_budget == lock['static_signal_byte_budget'] == 2000
            and retrieval_config.context_byte_budget == 6000
            and reviewer.config.request_byte_budget == 73728
            and reviewer.config.max_output_tokens == 8192 and reviewer.config.timeout_seconds == 60
            and reviewer.config.max_attempts == 1, 'locked preparation/settings mismatch')
    require(summary['request_budget'] == lock['request_budget'], 'request budget provenance mismatch')
    require(read_json(source / 'subset.json') == benchmark.subset.model_dump(), 'saved subset mismatch')
    order = [(t.task_id, mode) for t in tasks for mode in MODES]
    rows = summary['tasks']
    require([(r['task_id'], r['mode']) for r in rows] == order, 'observation order mismatch')
    prefix = next((n for n, row in enumerate(rows) if row['status'] != 'success'), len(rows))
    require(13 <= prefix < 76 and summary['completed_reviews'] == prefix, 'invalid successful prefix')
    require(all(r['status'] != 'success' for r in rows[prefix:]), 'non-prefix success is not reusable')
    require(all(r.get('status') in ('ready', 'not_attempted') and not r.get('result_file')
                and not r.get('prediction_file') and not r.get('attempts') and not r.get('verifier_attempts')
                for r in rows[prefix + 1:]), 'unexpected observation after failure')
    artifacts = {'subset.json': read_artifact(source / 'subset.json')}
    inputs, runs = {}, {}
    locked = {t['task_id']: t for t in lock['tasks']}
    for task in tasks:
        name = f'inputs/{task.task_id}.json'
        raw = read_artifact(source / name); artifacts[name] = raw
        inp = ReviewInput.model_validate_json(raw)
        require(inp.pr == task.pr and sha256(inp.diff.encode()) == task.diff_sha256
                == locked[task.task_id]['diff_sha256'], 'frozen input mismatch')
        parse_diff(inp.diff)
        inputs[task.task_id] = inp
        ref = f'reference/{task.task_id}.json'
        artifacts[ref] = read_artifact(source / ref)
        require(json.loads(artifacts[ref]) == benchmark.golden[task.task_id], 'reference artifact mismatch')
    for index, row in enumerate(rows):
        task_id, mode = order[index]
        task = next(t for t in tasks if t.task_id == task_id)
        require(row['pr'] == task.pr.model_dump() and row['diff_sha256'] == task.diff_sha256
                and row['model'] == reviewer.model and row['golden_sha256'] == task.golden_sha256,
                'observation identity mismatch')
        inp = inputs[task_id]
        require(row['diff_base'] == inp.diff_base, 'diff base mismatch')
        if not row.get('result_file'):
            continue
        require(index <= prefix, 'unexpected result after prefix')
        name = observation_path(task_id, mode)
        require(row['result_file'] == name, 'unexpected result path')
        raw = read_artifact(source / name); artifacts[name] = raw
        require(sha256(raw) == row['result_sha256'], 'run artifact hash mismatch')
        run = BaselineRun.model_validate_json(raw); runs[(task_id, mode)] = run
        require(run.result.pr == inp.pr and run.result.mode == mode and run.result.status == row['status']
                and run.diff_sha256 == task.diff_sha256 and run.diff_base == (inp.diff_base or inp.pr.base)
                and run.provider == 'openai' and run.requested_model == reviewer.model
                and run.settings == lock['review_settings']
                and run.prompt_text == review_prompt() and run.prompt_sha256 == expected['prompt_sha256']
                and run.output_schema == FindingBatch.model_json_schema(), 'run provenance mismatch')
        require(row['attempts'] == [a.model_dump() for a in run.attempts]
                and row.get('verifier_attempts', []) == ([a.model_dump() for a in run.verification.attempts] if run.verification else [])
                and row['predictions'] == [f.model_dump() for f in run.result.findings], 'summary/run mismatch')
        require(all(a.returned_model == reviewer.model for a in run.attempts if a.response_id), 'returned model mismatch')
        require(len(run.attempts) == 1, 'unexpected reviewer attempt count')
        if run.result.status == 'success' or mode == 'full':
            require(run.attempts[0].response_id is not None and run.attempts[0].error_code is None,
                    'successful reviewer response missing')
            parsed_output = FindingBatch.from_json(run.output_text)
            require(parsed_output.findings == (run.proposed_findings if mode == 'full' else run.result.findings),
                    'raw output differs from proposals')

        require((run.retrieval is None) == (mode == 'baseline')
                and (run.static_analysis is not None) == (mode in ('tools', 'full')), 'mode evidence mismatch')
        if run.retrieval:
            require(run.retrieval.head == inp.pr.head and run.retrieval.diff_sha256 == task.diff_sha256
                    and run.retrieval.archive_sha256 == locked[task_id]['head_archive_sha256']
                    and run.retrieval.config == retrieval_config.model_dump()
                    and run.retrieval.budget == 6000
                    and run.retrieval.overall_request_byte_ceiling == 73728
                    and run.retrieval.reserved_static_signal_byte_cap == 2000, 'context provenance mismatch')
        if run.static_analysis:
            st = StaticAnalysisResult.model_validate(run.static_analysis)
            require(st.head == inp.pr.head and st.archive_sha256 == locked[task_id]['head_archive_sha256']
                    and st.version == '1.7.8' and st.status in ('success', 'no_findings', 'no_files')
                    and st.config['signal_byte_budget'] == 2000 and st.timeout_seconds == static_config.timeout_seconds
                    and st.signals_sent == st.payload(2000), 'static provenance mismatch')
        payload = reviewer_payload(inp, run, reviewer)
        require(request_hash(payload) == run.request_sha256
                and measure_request(payload, 73728) == run.request_byte_measurement.model_dump(), 'request hash/measurement mismatch')
        if mode in ('tools', 'full'):
            require(run.retrieval == runs[(task_id, 'retrieval')].retrieval, 'B/C/D context mismatch')
        if mode == 'full':
            c = runs[(task_id, 'tools')]
            require(c.result.status == 'success' and run.proposed_findings == c.result.findings
                    and run.attempts == c.attempts and run.request_sha256 == c.request_sha256
                    and run.static_analysis == c.static_analysis, 'C/D pairing mismatch')
            v = run.verification
            require(v is not None and v.provider == 'openai' and v.model == verifier.model
                    and v.prompt_sha256 == expected['verifier_prompt_sha256'] and v.prompt_version == '1',
                    'verifier provenance mismatch')
            expected_vsettings = {k: review_settings(verifier)[k] for k in (
                'api', 'max_output_tokens', 'temperature', 'temperature_sent', 'max_attempts',
                'timeout_seconds', 'request_byte_budget', 'request_budget_method', 'store', 'structured_output')}
            expected_vsettings['proposal_calls'] = len(run.proposed_findings)
            require(v.settings == expected_vsettings
                    and all(a.returned_model == verifier.model for a in v.attempts if a.response_id)
                    and len(v.decisions) <= len(v.attempts) <= len(v.request_hashes), 'verifier settings/attempts mismatch')
            for n, decision in enumerate(v.decisions):
                require(decision.proposal_id == v.proposal_ids[n]
                        and decision.request_sha256 == v.request_hashes[n]
                        and v.attempts[n].error_code is None and v.attempts[n].response_id is not None,
                        'verifier decision provenance mismatch')

            for n, proposal in enumerate(run.proposed_findings):
                require(v.proposal_ids[n] == proposal_identifier(n, proposal), 'proposal identifier mismatch')
                if n >= len(v.request_hashes):
                    continue
                evidence = _evidence(inp, proposal, run.retrieval, st, 2000)
                vp = build_review_payload(verifier.model, verifier.config,
                    json.dumps(evidence, ensure_ascii=False, sort_keys=True), verification_prompt(),
                    schema=VerifierBatch.model_json_schema(), schema_name='verification_decision')
                require(request_hash(vp) == v.request_hashes[n], 'verifier request mismatch')
            if run.result.status == 'success':
                require(v.status == 'success' and run.result.findings == [p for p,d in zip(run.proposed_findings,v.decisions)
                            if d.verdict == 'SUPPORTED'], 'verifier survivor mismatch')
        if row['status'] == 'success':
            validate_anchors(FindingBatch(findings=run.result.findings), parse_diff(inp.diff))
            name = candidate_path(task_id, mode)
            require(row['prediction_file'] == name, 'unexpected candidate path')
            raw = read_artifact(source / name); artifacts[name] = raw
            require(sha256(raw) == row['prediction_sha256'] and json.loads(raw) == export_predictions(run), 'candidate mismatch')
        else:
            require(not row.get('prediction_file') and not (source / candidate_path(task_id, mode)).exists(),
                    'failed observation has candidate')
    # Detect unreferenced raw outputs instead of accidentally paying for them again.
    actual = {p.relative_to(source).as_posix() for p in (source / 'modes').rglob('*.json')}
    expected_files = {k for k in artifacts if k.startswith('modes/')}
    require(actual == expected_files, 'unreferenced observation artifact')
    history = deepcopy(summary.get('historical_failed_observations', []))
    for entry in history:
        name = entry['artifact_file']
        require(name.startswith('history/') and '..' not in Path(name).parts and not Path(name).is_absolute(), 'unsafe history path')
        raw = read_artifact(source / name)
        require(sha256(raw) == entry['artifact_sha256'], 'history hash mismatch')
        old = BaselineRun.model_validate_json(raw)
        require(old.result.status == 'failed', 'history is not a failure')
        artifacts[name] = raw
    parent = summary.get('continuation_source')
    if parent:
        parent_plan = validate_source(parent, benchmark, reviewer, verifier, retrieval_config, static_config, _seen=seen)
        require(parent_plan.summary_hash == summary['continuation_source_summary_sha256']
                and prefix >= parent_plan.prefix, 'continuation ancestry mismatch')
        for row in parent_plan.summary['tasks'][:parent_plan.prefix]:
            for key in ('result_file', 'prediction_file'):
                require(artifacts[row[key]] == parent_plan.artifacts[row[key]], 'reused prefix changed')
        require(history == inherited_history(parent_plan)[0], 'failed history changed')
    else:
        require(source.name == ORIGIN and prefix == 13 and not history
                and sha256(summary_raw) == ORIGIN_SUMMARY_SHA256, 'not the audited origin prefix')
        failed = runs.get(order[13])
        require(order[13] == ('NightHawkATL_ntp-dashboard_96-23782271', 'retrieval')
                and failed is not None and failed.result.status == 'failed'
                and len(failed.attempts) == 1 and failed.attempts[0].error_code == 'timeout'
                and failed.attempts[0].response_id is None and failed.attempts[0].usage is None
                and failed.output_text is None, 'origin timeout differs')
    return PrefixPlan(source, summary, sha256(summary_raw), inputs, runs, prefix, artifacts, history)


def inherited_history(plan):
    history = deepcopy(plan.history)
    artifacts = {k:v for k,v in plan.artifacts.items() if k.startswith('history/')}
    row = plan.summary['tasks'][plan.prefix]
    if row.get('result_file'):
        raw = plan.artifacts[row['result_file']]
        name = f'history/{len(history):04d}-{row["task_id"]}-{row["mode"]}.json'
        history.append({'task_id': row['task_id'], 'mode': row['mode'],
            'artifact_file': name, 'artifact_sha256': sha256(raw),
            'source_run': str(plan.source), 'source_result_file': row['result_file'],
            'completion_billing': 'unknown' if any(a.error_code == 'timeout' and a.usage is None
                for a in (list(plan.runs[(row['task_id'],row['mode'])].attempts) +
                          (list(plan.runs[(row['task_id'],row['mode'])].verification.attempts)
                           if plan.runs[(row['task_id'],row['mode'])].verification else []))) else 'see_recorded_attempts'})
        artifacts[name] = raw
    return history, artifacts


def account(report, runs, history_runs):
    """Count C once and deduplicate successful verifier calls replayed after failure."""
    seen = set()
    counts = {f'{role}_{outcome}': 0 for role in ('reviewer', 'verifier')
              for outcome in ('successful_responses', 'failed_attempts')}
    tokens = {k: 0 for k in ('input_tokens', 'output_tokens', 'total_tokens',
                             'cached_input_tokens', 'reasoning_tokens')}
    unknown = 0
    unknown_timeouts = 0
    historical_failed_attempts = 0
    for run_index, run in enumerate(list(history_runs) + list(runs.values())):
        groups = [('reviewer', run.attempts)] if run.result.mode != 'full' else []
        if run.verification:
            groups.append(('verifier', run.verification.attempts))
        for role, attempts in groups:
            for n, attempt in enumerate(attempts):
                key = (role, attempt.response_id) if attempt.response_id else (
                    role, request_hash(run.model_dump(mode='json')), n)
                if key in seen:
                    continue
                seen.add(key)
                outcome = 'failed_attempts' if attempt.error_code else 'successful_responses'
                counts[f'{role}_{outcome}'] += 1
                if run_index < len(history_runs) and attempt.error_code:
                    historical_failed_attempts += 1
                if attempt.usage:
                    for k in tokens:
                        tokens[k] += getattr(attempt.usage, k) or 0
                else:
                    unknown += 1
                    unknown_timeouts += int(attempt.error_code == 'timeout')
    report.update(counts, recorded_token_usage=tokens, unknown_usage_billing_attempts=unknown,
                  recorded_model_attempts_including_history=len(seen),
                  historical_failed_attempt_count=historical_failed_attempts,
                  unknown_billing_timeout_attempts=unknown_timeouts,
                  reviewer_attempts=counts['reviewer_successful_responses']+counts['reviewer_failed_attempts'],
                  verifier_attempts=counts['verifier_successful_responses']+counts['verifier_failed_attempts'])
    rows = report['tasks']
    report['completed_reviews'] = sum(r['status'] == 'success' for r in rows)
    report['newly_completed_observations'] = report['completed_reviews'] - report['reused_reviews']
    report['failed_current_observations'] = sum(r['status'] == 'failed' for r in rows)
    report['historical_failed_observation_count'] = len(history_runs)
    report['completed_task_count'] = sum(all(r['status'] == 'success' for r in rows[n:n+4])
                                         for n in range(0,76,4))
    report['failed_task_count'] = sum(any(r['status'] == 'failed' for r in rows[n:n+4])
                                      for n in range(0,76,4))
    report['not_attempted_task_count'] = 19-report['completed_task_count']-report['failed_task_count']


class ReplayVerifier:
    """Replay only already successful verifier decisions; new calls use the same adapter.

    This preserves successful paid verifier work if a later proposal timed out.
    Request hashes must match, including for the previously failed request.
    """
    def __init__(self, verifier, prior):
        self.inner = verifier
        self.model, self.config = verifier.model, verifier.config
        self.prior, self.index = prior.verification, 0
        self.replayed_latency_seconds = 0.0

    def payload(self, *args, **kwargs):
        return self.inner.payload(*args, **kwargs)

    def complete(self, payload):
        n = self.index; self.index += 1
        if n < len(self.prior.request_hashes):
            require(request_hash(payload) == self.prior.request_hashes[n], 'continued verifier request mismatch')
        if n < len(self.prior.decisions):
            d = self.prior.decisions[n]
            text = json.dumps({k:getattr(d,k) for k in ('verdict','rationale','evidence_references','confidence')})
            a = self.prior.attempts[n]
            self.replayed_latency_seconds += a.latency_seconds
            return ModelReply(text, (a,), a.latency_seconds, None)
        return self.inner.complete(payload)


def continue_frozen_run(source, benchmark, reviewer, verifier, client, results_dir, *,
                        modes, retrieval_config, static_config, cache_dir='.cache'):
    """Explicit invocation authorizes one new attempt at the first unfinished observation."""
    from reviewer.evaluation import _persist_result
    from reviewer.preflight import _write_atomic
    require(modes == MODES, 'continuation requires baseline,retrieval,tools,full in order')
    plan = validate_source(source, benchmark, reviewer, verifier, retrieval_config, static_config)
    parent = Path(results_dir).absolute()
    require(not any(p.is_symlink() for p in (parent,*parent.parents)), 'unsafe continuation output path')
    ancestor = plan.source
    while ancestor is not None:
        require(not parent.is_relative_to(ancestor), 'continuation cannot write inside source ancestry')
        previous = read_json(ancestor / 'summary.json').get('continuation_source')
        ancestor = Path(previous).absolute() if previous else None
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / ('reviewbench-' + uuid.uuid4().hex); root.mkdir()
    report = deepcopy(plan.summary)
    history, history_artifacts = inherited_history(plan)
    report.update(status='preparing', created_at=datetime.now(timezone.utc).isoformat(),
        continuation_source=str(plan.source), continuation_source_summary_sha256=plan.summary_hash,
        continuation_origin=report.get('continuation_origin', str(plan.source)),
        historical_failed_observations=history, reused_reviews=plan.prefix,
        reused_observation_count=plan.prefix, newly_completed_observations=0, model_attempts=0,
        requested_reviews=76, reused_artifact_hashes={}, error=None, error_code=None)
    for row in report['tasks'][plan.prefix:]:
        # Keep only preparation identity; failed attempt lives in immutable history.
        for key in ('result_file','result_sha256','prediction_file','prediction_sha256','attempts',
                    'verifier_attempts','predictions','error','error_code','settings'):
            row.pop(key,None)
        row.update(status='ready', reused=False)
    copied = {k:v for k,v in plan.artifacts.items() if not k.startswith(('modes/','history/'))}
    copied.update(history_artifacts)
    for row in report['tasks'][:plan.prefix]:
        row['reused'] = True
        for key in ('result_file','prediction_file'):
            name = row[key]; copied[name] = plan.artifacts[name]
            report['reused_artifact_hashes'][name] = sha256(copied[name])
    for name, raw in copied.items():
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
    runs = {key:run for key,run in plan.runs.items() if run.result.status == 'success'}
    history_runs = [BaselineRun.model_validate_json(history_artifacts[e['artifact_file']]) for e in history]
    account(report,runs,history_runs)
    prior_attempt_count = report['recorded_model_attempts_including_history']
    def persist():
        account(report,runs,history_runs)
        report['model_attempts'] = report['recorded_model_attempts_including_history'] - prior_attempt_count
        _write_atomic(root/'summary.json',report)
    persist()
    tasks = {t.task_id:t for t in benchmark.subset.tasks if t.task_id != TASK_ID}
    locked = {t['task_id']:t for t in benchmark.lock_provenance['tasks']}
    prepared = {}
    index = plan.prefix
    try:
        # Prepare all remaining tasks without models, exactly as the original runner.
        for row in report['tasks'][plan.prefix:]:
            key = row['task_id']
            if key in prepared:
                continue
            task, inp = tasks[key], plan.inputs[key]
            started = time.perf_counter()
            snapshot = client.frozen_head(task.pr, source_repository=task.source_repository,
                                          cache_dir=Path(cache_dir)/'github')
            require(snapshot.commit == task.pr.head and snapshot.archive_sha256 == locked[key]['head_archive_sha256'],
                    'continued frozen snapshot mismatch')
            context = prepare_context(inp, reviewer, snapshot, retrieval_config,
                cache_dir=Path(cache_dir)/'retrieval', reserve_static_signal_byte_cap=static_config.signal_byte_budget)
            retrieval_elapsed = time.perf_counter()-started
            started = time.perf_counter()
            static = prepare_static_analysis(inp,snapshot,context,static_config)
            static_elapsed = time.perf_counter()-started
            require(static.status in ('success','no_findings','no_files') and static.version=='1.7.8',
                    'continued static analysis failed')
            for mode in MODES:
                old = plan.runs.get((key,mode))
                if old and old.retrieval:
                    require(old.retrieval == context, 'continued shared context mismatch')
                if old and old.static_analysis:
                    previous = StaticAnalysisResult.model_validate(old.static_analysis)
                    require(previous.analyzed_files == static.analyzed_files
                            and previous.findings == static.findings
                            and previous.payload(2000) == static.payload(2000), 'continued static evidence mismatch')
            prepared[key] = (context, static, retrieval_elapsed, static_elapsed)
        # Check the boundary request BEFORE any new call (including failed D evidence).
        boundary = report['tasks'][plan.prefix]
        old = plan.runs.get((boundary['task_id'],boundary['mode']))
        if old:
            context,static,_,_ = prepared[boundary['task_id']]
            payload = reviewer.payload(plan.inputs[boundary['task_id']].diff,review_prompt(),
                context=context.payload() if boundary['mode']!='baseline' else None,
                static_signals=static.payload(2000) if boundary['mode'] in ('tools','full') else None)
            require(request_hash(payload)==old.request_sha256
                    and measure_request(payload,73728)==old.request_byte_measurement.model_dump(),
                    'continued boundary request hash/measurement mismatch')
        for index in range(plan.prefix,76):
            row=report['tasks'][index];key,mode=row['task_id'],row['mode']
            inp=plan.inputs[key];context,static,rt,st=prepared[key]
            common=dict(retrieval_latency_seconds=rt, input_artifact_directory=str((root/'inputs').resolve()))
            row['status']='running';report['status']='running';persist()
            if mode=='baseline':
                run=run_baseline(inp,reviewer,input_artifact_directory=str((root/'inputs').resolve()))
            elif mode=='retrieval':
                run=run_review(inp,reviewer,mode=mode,context=context,**common)
            elif mode=='tools':
                run=run_review(inp,reviewer,mode=mode,context=context,static_analysis=static,
                               static_analysis_latency_seconds=st,**common)
            else:
                c=runs[(key,'tools')]
                # Use C's exact persisted preparation as well as its proposals.
                prior=plan.runs.get((key,mode))
                verifier_for_run=ReplayVerifier(verifier,prior) if prior and prior.verification else verifier
                run=run_full(inp,reviewer,verifier_for_run,tools_run=c,context=c.retrieval,
                    static_analysis=StaticAnalysisResult.model_validate(c.static_analysis),
                    static_analysis_latency_seconds=c.static_analysis_latency_seconds,
                    retrieval_latency_seconds=c.retrieval_latency_seconds,signal_budget=2000,
                    input_artifact_directory=str((root/'inputs').resolve()))
                if isinstance(verifier_for_run, ReplayVerifier) and run.verification is not None:
                    # Retain latency of paid successful decisions replayed locally.
                    replayed = verifier_for_run.replayed_latency_seconds
                    verification = run.verification.model_copy(update={
                        'latency_seconds': run.verification.latency_seconds + replayed})
                    run = run.model_copy(update={'verification': verification,
                        'verifier_latency_seconds': run.verifier_latency_seconds + replayed,
                        'total_latency_seconds': run.total_latency_seconds + replayed})
            runs[(key,mode)]=run
            mode_root=root/'modes'/mode
            (mode_root/'runs').mkdir(parents=True,exist_ok=True)
            (mode_root/'candidates').mkdir(parents=True,exist_ok=True)
            _persist_result(mode_root,tasks[key],run,row,None)
            for field in ('result_file','prediction_file'):
                if field in row:
                    row[field]=str(Path('modes')/mode/row[field])
            report['status']='failed' if run.result.status!='success' else 'running'
            persist()
            if run.result.status!='success':
                break
        else:
            report['status']='generation_complete_scoring_paused'
    except (ValueError,OSError,KeyError,TypeError) as error:
        report.update(status='failed',error='continuation preparation, control, or persistence failed',
                      error_code=getattr(error,'code','continuation_failed'))
        row=report['tasks'][index]
        if row['status'] not in ('success','failed'):
            row.update(status='failed',error_code=report['error_code'])
    finally:
        persist()
    return root/'summary.json',report
