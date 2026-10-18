"""Verify proposed findings without discovering additional defects."""
from importlib.resources import files
import json
import time
from typing import Literal

from pydantic import Field

from reviewer.archives import sha256
from reviewer.diff import parse_diff
from reviewer.models import ModelError, OpenAIReviewer
from reviewer.schemas import (Finding, ModelAttempt, ReviewInput, RepositoryContext,
                              StrictModel, Text, VerificationDecision,
                              VerificationRecord)

PROMPT_VERSION = "1"


class VerifierBatch(StrictModel):
    verdict: Literal['SUPPORTED', 'UNSUPPORTED', 'UNCERTAIN']
    rationale: Text
    evidence_references: list[Text]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


def verification_prompt() -> str:
    return files('reviewer').joinpath('prompts/verify.txt').read_text(encoding='utf-8')


def proposal_identifier(index: int, finding: Finding) -> str:
    canonical = json.dumps({'index': index, 'finding': finding.model_dump()}, sort_keys=True,
                           ensure_ascii=False, allow_nan=False).encode('utf-8')
    return sha256(canonical)


def _evidence(review_input: ReviewInput, finding: Finding, context: RepositoryContext | None,
              static_analysis, signal_budget: int) -> dict:
    """Explicit allowlist of just the target finding's changed hunk and nearby signals."""
    parsed = parse_diff(review_input.diff)
    file = next((f for f in parsed if f.new_path == finding.file), None)
    if file is None:
        raise ValueError('proposal path is not present in the PR diff')
    hunks = []
    for hunk in file.hunks:
        lines = [line for line in hunk.lines if line.new_line is not None]
        if any(line.new_line <= finding.end_line and line.new_line >= finding.start_line
               for line in lines):
            hunks.append({'old_start': hunk.old_start, 'old_count': hunk.old_count,
                          'new_start': hunk.new_start, 'new_count': hunk.new_count,
                          'section': hunk.section, 'lines': [
                {'kind': line.kind, 'head_line': line.new_line, 'text': line.content}
                for line in hunk.lines]})
    if not hunks:
        raise ValueError('proposal has no relevant diff hunk')
    result = {'proposed_finding': finding.model_dump(),
              'diff': {'file': finding.file, 'hunks': hunks}}
    if context is not None:
        references = ' '.join(finding.evidence)
        related = [chunk for chunk in context.chunks if chunk.selected and
                   (chunk.file == finding.file or chunk.file in references)]
        result['repository_context'] = [
            {'file': chunk.file, 'start_line': chunk.start_line, 'end_line': chunk.end_line,
             'content': chunk.content} for chunk in related]
    if static_analysis is not None:
        related_signals = [signal for signal in static_analysis.payload(signal_budget)
                           if signal.get('file') == finding.file and
                           finding.start_line <= signal.get('line', 0) <= finding.end_line]
        result['static_analysis_signals'] = related_signals
    return result


def verify_proposals(review_input: ReviewInput, proposals: list[Finding], verifier: OpenAIReviewer, *,
                     context: RepositoryContext | None = None, static_analysis=None,
                     signal_budget: int = 0, clock=time.perf_counter) -> VerificationRecord:
    """One isolated verifier call per proposal. Empty proposal sets cost no calls."""
    prompt = verification_prompt()
    prompt_hash = sha256(prompt.encode('utf-8'))
    ids = [proposal_identifier(i, proposal) for i, proposal in enumerate(proposals)]
    attempts: list[ModelAttempt] = []
    decisions: list[VerificationDecision] = []
    request_hashes: list[str] = []
    started = clock()
    error = None
    for proposal_id, proposal in zip(ids, proposals):
        try:
            evidence = _evidence(review_input, proposal, context, static_analysis, signal_budget)
            payload = verifier.payload(json.dumps(evidence, ensure_ascii=False, sort_keys=True), prompt,
                                      schema=VerifierBatch.model_json_schema(), schema_name='verification_decision')
            request_hashes.append(sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                                     allow_nan=False).encode('utf-8')))
            reply = verifier.complete(payload)
            attempts.extend(reply.attempts)
            if reply.error:
                raise ModelError('verification_failed', 'verifier request failed')
            answer = VerifierBatch.model_validate_json(reply.text, strict=True)
            decisions.append(VerificationDecision(proposal_id=proposal_id,
                                                  request_sha256=request_hashes[-1],
                                                  **answer.model_dump()))
        except Exception:
            error = 'verification request or output validation failed'
            break
    request_hash = sha256(json.dumps(request_hashes, separators=(',', ':')).encode()) if request_hashes else sha256(b'[]')
    settings = {'api': 'responses', 'max_output_tokens': verifier.config.max_output_tokens,
                'temperature': verifier.config.temperature,
                'temperature_sent': verifier.config.temperature is not None,
                'max_attempts': verifier.config.max_attempts,
                'timeout_seconds': verifier.config.timeout_seconds,
                'input_token_limit': verifier.config.input_token_limit,
                'structured_output': 'json_schema_strict', 'store': False,
                'proposal_calls': len(proposals)}
    return VerificationRecord(status='failed' if error else 'success', prompt_version=PROMPT_VERSION,
                              prompt_sha256=prompt_hash, provider='openai', model=verifier.model,
                              settings=settings, request_sha256=request_hash, proposal_ids=ids,
                              request_hashes=request_hashes, decisions=decisions, attempts=attempts,
                              latency_seconds=max(0.0, clock() - started), error=error)
