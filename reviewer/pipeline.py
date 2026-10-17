"""Diff-only experimental control. Later modes deliberately remain unimplemented."""

from datetime import datetime, timezone
from importlib.resources import files
import json
import time

from reviewer.archives import sha256
from reviewer.diff import DiffParseError, parse_diff
from reviewer.models import ModelError, OpenAIReviewer
from reviewer.schemas import BaselineRun, FindingBatch, ReviewInput, ReviewResult, RepositoryContext


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
                "input_token_limit": reviewer.config.input_token_limit,
                "input_budget_policy": "serialized_request_utf8_bytes_plus_4096_estimate",
                "truncation": "disabled", "store": False, "structured_output": "json_schema_strict",
                "seed_sent": False, "anchor_policy": "represented_head_lines_in_changed_hunks"}


def run_baseline(review_input: ReviewInput, reviewer: OpenAIReviewer, *,
                 acquisition_latency_seconds: float = 0.0,
                 input_artifact_directory: str | None = None,
                 clock=time.perf_counter) -> BaselineRun:
    """Baseline wrapper: no context argument exists on this interface."""
    return run_review(review_input, reviewer, acquisition_latency_seconds=acquisition_latency_seconds,
                      input_artifact_directory=input_artifact_directory, clock=clock)


def prepare_context(review_input, reviewer, snapshot, config, *, cache_dir=None):
    from reviewer.retrieval import retrieve
    baseline_payload = reviewer.payload(review_input.diff, review_prompt())
    baseline_size = len(json.dumps(baseline_payload, ensure_ascii=False).encode()) + 4096
    # Leave room for the additional JSON field; never displace or truncate diff.
    available = reviewer.config.input_token_limit - baseline_size - 64
    context = retrieve(review_input, snapshot, config, cache_dir=cache_dir, available_budget=available)
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
        budget=static_config.signal_token_budget
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
    started = clock()
    review_input = ReviewInput.model_validate(review_input.model_dump())
    prompt = review_prompt()
    schema = FindingBatch.model_json_schema()
    settings = review_settings(reviewer)
    attempts = []
    model_latency = 0.0
    output_text = None
    request_hash = None
    findings = []
    error = None
    try:
        parsed = parse_diff(review_input.diff)
        static_signals = static_analysis.payload(reviewer.config.input_token_limit) if static_analysis and mode == 'tools' else None
        if mode == 'tools':
            payload = reviewer.payload(review_input.diff, prompt, context=context.payload(), static_signals=static_signals)
        elif context:
            payload = reviewer.payload(review_input.diff, prompt, context=context.payload())
        else:
            payload = reviewer.payload(review_input.diff, prompt)
        request_hash = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8"))
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
                       request_sha256=request_hash, created_at=datetime.now(timezone.utc).isoformat(),
                       attempts=attempts, model_latency_seconds=model_latency,
                       acquisition_latency_seconds=acquisition_latency_seconds,
                       total_latency_seconds=acquisition_latency_seconds + retrieval_latency_seconds + static_analysis_latency_seconds + max(0.0, clock() - started),
                       output_text=output_text, input_artifact_directory=input_artifact_directory,
                       retrieval=context, retrieval_latency_seconds=retrieval_latency_seconds,
                       static_analysis=static_analysis.model_dump() if static_analysis else None,
                       static_analysis_latency_seconds=static_analysis_latency_seconds)
