"""Diff-only experimental control. Later modes deliberately remain unimplemented."""

from datetime import datetime, timezone
from importlib.resources import files
import json
import time

from reviewer.archives import sha256
from reviewer.diff import DiffParseError, parse_diff
from reviewer.models import ModelError, OpenAIReviewer
from reviewer.schemas import BaselineRun, FindingBatch, ReviewInput, ReviewResult, RepositoryContext
from reviewer.request_budget import measure_request, BUDGET_METHOD


def review_prompt() -> str:
    return files("reviewer").joinpath("prompts/review.txt").read_text(encoding="utf-8")


def validate_anchors(batch: FindingBatch, parsed) -> None:
    """Structural checks only. Do not judge the truth of any defect or evidence."""
    by_path = {file.new_path: file for file in parsed if file.new_path is not None}
    for finding in batch.findings:
        file = by_path.get(finding.file)
        if file is None:
            raise ModelError("invalid_reference", "finding path is not a surviving file in the diff")
        represented = {line.new_line for hunk in file.hunks for line in hunk.lines if line.new_line is not None}
        count = sum(finding.start_line <= line <= finding.end_line for line in represented)
        if count != finding.end_line - finding.start_line + 1:
            raise ModelError("invalid_reference", "finding range contains head lines not represented in the diff")
        changed_hunks = [hunk for hunk in file.hunks if any(line.kind != "context" for line in hunk.lines)]
        if not any(line.new_line is not None and finding.start_line <= line.new_line <= finding.end_line
                   for hunk in changed_hunks for line in hunk.lines):
            raise ModelError("invalid_reference", "finding must anchor to a changed hunk")


def review_settings(reviewer: OpenAIReviewer) -> dict:
    """Shared provenance contract for generation and offline resume validation."""
    return {"api": "responses", "max_output_tokens": reviewer.config.max_output_tokens,
                "temperature": reviewer.config.temperature, "temperature_sent": reviewer.config.temperature is not None,
                "max_attempts": reviewer.config.max_attempts, "timeout_seconds": reviewer.config.timeout_seconds,
                "request_byte_budget": reviewer.config.request_byte_budget,
                "request_budget_method": BUDGET_METHOD,
                "truncation": "disabled", "store": False, "structured_output": "json_schema_strict",
                "seed_sent": False, "anchor_policy": "represented_head_lines_in_changed_hunks"}


def run_baseline(review_input: ReviewInput, reviewer: OpenAIReviewer, *,
                 acquisition_latency_seconds: float = 0.0,
                 input_artifact_directory: str | None = None,
                 clock=time.perf_counter) -> BaselineRun:
    """Baseline wrapper: no context argument exists on this interface."""
    return run_review(review_input, reviewer, acquisition_latency_seconds=acquisition_latency_seconds,
                      input_artifact_directory=input_artifact_directory, clock=clock)


def prepare_context(review_input, reviewer, snapshot, config, *, cache_dir=None,
                    reserve_static_signal_byte_cap=0):
    from reviewer.retrieval import retrieve
    baseline_payload = reviewer.payload(review_input.diff, review_prompt())
    baseline_size = measure_request(baseline_payload)["estimator_value"]
    # B/C/D always share one context. Reserve worst-case two-layer JSON escaping
    # of the configured signal cap, plus fixed envelope overhead, for C/D.
    signal_reservation_estimator_bytes = reserve_static_signal_byte_cap * 4 + (256 if reserve_static_signal_byte_cap else 0)
    available = reviewer.config.request_byte_budget - baseline_size - signal_reservation_estimator_bytes - 64
    context = retrieve(review_input, snapshot, config, cache_dir=cache_dir, available_byte_budget=available)
    context = context.model_copy(update={
        'overall_request_byte_ceiling': reviewer.config.request_byte_budget,
        'reserved_static_signal_byte_cap': reserve_static_signal_byte_cap,
        'reserved_static_signal_estimator_bytes': signal_reservation_estimator_bytes,
    })
    reviewer.payload(review_input.diff, review_prompt(), context=context.payload())
    return context


def prepare_static_analysis(review_input, snapshot, context, static_config):
    """Run static analysis on frozen snapshot; never analyze live working tree."""
    from reviewer.static_analysis import analyze_snapshot
    return analyze_snapshot(
        review_input,
        snapshot,
        context,
        timeout_seconds=static_config.timeout_seconds,
        budget=static_config.signal_byte_budget
    )


def run_review(review_input: ReviewInput, reviewer: OpenAIReviewer, *, mode='baseline',
               context: RepositoryContext | None = None, retrieval_latency_seconds=0.0,
               static_analysis=None, static_analysis_latency_seconds=0.0,
               acquisition_latency_seconds=0.0, input_artifact_directory=None,
               clock=time.perf_counter) -> BaselineRun:
    """Shared A/B/C reviewer, schema and anchors; only supplied evidence differs."""
    if mode not in ('baseline', 'retrieval', 'tools') or (mode == 'baseline' and context is not None):
        raise ValueError('unsupported mode or context supplied to baseline')
    if mode == 'retrieval':
        if context is None:
            raise ValueError('retrieval requires explicit frozen context')
        context = RepositoryContext.model_validate(context.model_dump())
        if context.head != review_input.pr.head or context.diff_sha256 != sha256(review_input.diff.encode()):
            raise ValueError('retrieval context belongs to another frozen input')
    if mode == 'tools':
        if context is None or static_analysis is None:
            raise ValueError('tools mode requires both retrieval context and static analysis')
        context = RepositoryContext.model_validate(context.model_dump())
        if context.head != review_input.pr.head or context.diff_sha256 != sha256(review_input.diff.encode()):
            raise ValueError('retrieval context belongs to another frozen input')
        if static_analysis.head != context.head or static_analysis.archive_sha256 != context.archive_sha256:
            raise ValueError('static analysis belongs to another frozen snapshot')
    started = clock()
    review_input = ReviewInput.model_validate(review_input.model_dump())
    prompt = review_prompt()
    schema = FindingBatch.model_json_schema()
    settings = review_settings(reviewer)
    attempts = []
    model_latency = 0.0
    output_text = None
    request_hash = None
    request_measurement = None
    findings = []
    error = None
    try:
        parsed = parse_diff(review_input.diff)
        if static_analysis is not None and mode == 'tools':
            signal_budget = static_analysis.config.get('signal_byte_budget', 0)
            if type(signal_budget) is not int or signal_budget < 0:
                raise ValueError('static analysis signal budget is invalid')
            static_signals = static_analysis.payload(signal_budget)
            static_analysis = static_analysis.model_copy(update={'signals_sent': static_signals})
        else:
            static_signals = None
        if mode == 'tools':
            payload = reviewer.payload(review_input.diff, prompt, context=context.payload(), static_signals=static_signals)
        elif context:
            payload = reviewer.payload(review_input.diff, prompt, context=context.payload())
        else:
            payload = reviewer.payload(review_input.diff, prompt)
        request_hash = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        request_measurement = measure_request(payload, reviewer.config.request_byte_budget)
        reply = reviewer.complete(payload)
        attempts = list(reply.attempts)
        model_latency = reply.latency_seconds
        output_text = reply.text
        if reply.error:
            error = reply.error
        else:
            try:
                batch = FindingBatch.from_json(output_text)
            except (ValueError, TypeError):
                raise ModelError("invalid_output", "model output failed strict FindingBatch validation") from None
            validate_anchors(batch, parsed)
            findings = batch.findings
    except DiffParseError:
        error = "invalid_diff: input diff failed the existing parser"
    except ModelError as failure:
        error = str(failure)
    result = ReviewResult(pr=review_input.pr, mode=mode, status="failed" if error else "success",
                          findings=findings if error is None else [], error=error)
    return BaselineRun(result=result, diff_base=review_input.diff_base or review_input.pr.base,
                       diff_sha256=sha256(review_input.diff.encode("utf-8")),
                       requested_model=reviewer.model, settings=settings,
                       prompt_sha256=sha256(prompt.encode("utf-8")), prompt_text=prompt, output_schema=schema,
                       request_sha256=request_hash, request_byte_measurement=request_measurement,
                       created_at=datetime.now(timezone.utc).isoformat(),
                       attempts=attempts, model_latency_seconds=model_latency,
                       acquisition_latency_seconds=acquisition_latency_seconds,
                       total_latency_seconds=acquisition_latency_seconds + retrieval_latency_seconds + static_analysis_latency_seconds + max(0.0, clock() - started),
                       output_text=output_text, input_artifact_directory=input_artifact_directory,
                       retrieval=context, retrieval_latency_seconds=retrieval_latency_seconds,
                       static_analysis=static_analysis.model_dump() if static_analysis else None,
                       static_analysis_latency_seconds=static_analysis_latency_seconds)


def run_full(review_input: ReviewInput, reviewer: OpenAIReviewer, verifier: OpenAIReviewer, *,
             context: RepositoryContext, static_analysis, retrieval_latency_seconds=0.0,
             static_analysis_latency_seconds=0.0, acquisition_latency_seconds=0.0,
             signal_budget=0, input_artifact_directory=None, clock=time.perf_counter,
             tools_run: BaselineRun | None = None) -> BaselineRun:
    """Mode D: generate the Mode C proposals once, then verify those exact objects."""
    from reviewer.verification import verify_proposals
    if static_analysis.status not in ('success', 'no_findings', 'no_files'):
        prompt = review_prompt()
        started = clock()
        result = ReviewResult(pr=review_input.pr, mode='full', status='failed', findings=[],
                              error='static_analysis_failed: Mode D requires successful Bandit preparation')
        return BaselineRun(result=result, diff_base=review_input.diff_base or review_input.pr.base,
                           diff_sha256=sha256(review_input.diff.encode('utf-8')),
                           requested_model=reviewer.model, settings=review_settings(reviewer),
                           prompt_sha256=sha256(prompt.encode('utf-8')), prompt_text=prompt,
                           output_schema=FindingBatch.model_json_schema(), request_sha256=None,
                           created_at=datetime.now(timezone.utc).isoformat(), attempts=[],
                           model_latency_seconds=0.0, acquisition_latency_seconds=acquisition_latency_seconds,
                           total_latency_seconds=acquisition_latency_seconds + retrieval_latency_seconds +
                           static_analysis_latency_seconds + max(0.0, clock() - started),
                           input_artifact_directory=input_artifact_directory, retrieval=context,
                           retrieval_latency_seconds=retrieval_latency_seconds,
                           static_analysis=static_analysis.model_dump(),
                           static_analysis_latency_seconds=static_analysis_latency_seconds)
    tools_run = tools_run or run_review(review_input, reviewer, mode='tools', context=context,
                           static_analysis=static_analysis,
                           retrieval_latency_seconds=retrieval_latency_seconds,
                           static_analysis_latency_seconds=static_analysis_latency_seconds,
                           acquisition_latency_seconds=acquisition_latency_seconds,
                           input_artifact_directory=input_artifact_directory, clock=clock)
    proposals = list(tools_run.result.findings)
    if tools_run.result.status != 'success':
        failed_result = tools_run.result.model_copy(update={'mode': 'full'})
        return tools_run.model_copy(update={'result': failed_result,
                                            'proposed_findings': proposals})
    verification = verify_proposals(review_input, proposals, verifier, context=context,
                                    static_analysis=static_analysis, signal_budget=signal_budget)
    if verification.status == 'failed':
        result = ReviewResult(pr=review_input.pr, mode='full', status='failed', findings=[],
                              error='verification_failed: verifier did not produce a valid decision for every proposal')
        final_findings = []
    else:
        final_findings = [proposal for proposal, decision in zip(proposals, verification.decisions)
                          if decision.verdict == 'SUPPORTED']
        result = ReviewResult(pr=review_input.pr, mode='full', status='success', findings=final_findings)
    return tools_run.model_copy(update={
        'result': result, 'verification': verification,
        'verifier_latency_seconds': verification.latency_seconds,
        'proposed_findings': proposals,
        'total_latency_seconds': tools_run.total_latency_seconds + verification.latency_seconds,
    })
