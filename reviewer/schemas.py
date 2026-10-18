"""Strict wire contracts. Finding coordinates always refer to the frozen head."""

import json
import re
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


def nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace text")
    return value


def repository_path(value: str) -> str:
    """Reject ambiguous paths instead of silently rewriting model evidence."""
    if (not value or value.startswith("/") or "\\" in value or ":" in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("must be a normalized repository-relative POSIX path")
    return value


Text = Annotated[str, AfterValidator(nonblank)]
RepoPath = Annotated[str, AfterValidator(repository_path)]
CommitSHA = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
PositiveInt = Annotated[int, Field(ge=1)]
Category = Literal["correctness", "security", "reliability", "performance"]
Severity = Literal["low", "medium", "high"]
Mode = Literal["baseline", "retrieval", "tools", "full"]
MODES = ("baseline", "retrieval", "tools", "full")


class PRMetadata(StrictModel):
    repo: str
    pr_number: PositiveInt
    base: CommitSHA
    head: CommitSHA

    @model_validator(mode="after")
    def valid_repo(self) -> Self:
        if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repo):
            raise ValueError("repo must be a canonical https://github.com/owner/repository URL")
        if any(part in (".", "..") for part in self.repo.split("/")[-2:]):
            raise ValueError("invalid repository URL")
        return self


class BenchmarkAggregateMismatch(StrictModel):
    code: Literal["benchmark_aggregate_mismatch"]
    expected_file_count: Annotated[int, Field(ge=0)]
    expected_additions: Annotated[int, Field(ge=0)]
    expected_deletions: Annotated[int, Field(ge=0)]
    reconstructed_file_count: Annotated[int, Field(ge=0)]
    reconstructed_additions: Annotated[int, Field(ge=0)]
    reconstructed_deletions: Annotated[int, Field(ge=0)]
    base_sha: CommitSHA
    head_sha: CommitSHA
    mismatch_fields: Annotated[list[Literal["additions", "deletions"]], Field(min_length=1)]

    @model_validator(mode="after")
    def consistent_totals(self) -> Self:
        if self.expected_file_count != self.reconstructed_file_count:
            raise ValueError("benchmark file-count mismatch is not an aggregate warning")
        expected = []
        if self.expected_additions != self.reconstructed_additions:
            expected.append("additions")
        if self.expected_deletions != self.reconstructed_deletions:
            expected.append("deletions")
        if self.mismatch_fields != expected or not expected:
            raise ValueError("aggregate mismatch fields do not match recorded totals")
        return self


class ReviewInput(StrictModel):
    pr: PRMetadata
    diff: str  # Empty string is a valid, explicitly empty change set.
    diff_complete: Literal[True]
    # Old-side coordinates use this merge base when set; otherwise pr.base.
    diff_base: CommitSHA | None = None
    # Benchmark-only acquisition diagnostics. These are persisted as input
    # provenance and are never included in the reviewer payload.
    acquisition_warnings: list[BenchmarkAggregateMismatch] = Field(default_factory=list)

    @field_validator("diff_complete", mode="before")
    @classmethod
    def complete_is_boolean(cls, value):
        if type(value) is not bool:
            raise ValueError("diff_complete must be the boolean true")
        return value


class Finding(StrictModel):
    title: Text
    category: Category
    severity: Severity
    file: RepoPath
    start_line: PositiveInt
    end_line: PositiveInt
    description: Text
    evidence: Annotated[list[Text], Field(min_length=1)]
    suggested_fix: Text
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

    @model_validator(mode="after")
    def ordered_range(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line (inclusive head coordinates)")
        return self


class FindingBatch(StrictModel):
    """Model output: one JSON object containing a findings array."""
    findings: list[Finding]

    @classmethod
    def from_json(cls, raw: str) -> Self:
        """No fence stripping, JSON repair, duplicate keys, or NaN extensions."""
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def invalid_constant(value):
            raise ValueError(f"invalid JSON constant: {value}")

        return cls.model_validate(json.loads(raw, object_pairs_hook=pairs,
                                             parse_constant=invalid_constant))


class ReviewResult(StrictModel):
    """Minimal result contract; no fabricated timings, costs, or verification data."""
    schema_version: Literal[1] = 1
    pr: PRMetadata
    mode: Mode
    status: Literal["success", "failed"]
    findings: list[Finding]
    error: Text | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def consistent_status(self) -> Self:
        if self.status == "success" and self.error is not None:
            raise ValueError("a successful review cannot contain an error")
        if self.status == "failed" and (self.error is None or self.findings):
            raise ValueError("a failed review requires an error and no final findings")
        return self


class TokenUsage(StrictModel):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)]
    cached_input_tokens: Annotated[int, Field(ge=0)] | None = None
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("inconsistent total token count")
        if self.cached_input_tokens is not None and self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input exceeds input tokens")
        if self.reasoning_tokens is not None and self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning exceeds output tokens")
        return self


class IncompleteDetails(StrictModel):
    # Unknown provider values are redacted, never copied as arbitrary text.
    reason: Literal['max_output_tokens', 'content_filter', 'unknown']


class ModelAttempt(StrictModel):
    number: PositiveInt
    latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    response_id: Text | None = None
    returned_model: Text | None = None
    usage: TokenUsage | None = None
    error_code: Text | None = None
    response_status: Literal['completed', 'failed', 'in_progress', 'cancelled', 'queued', 'incomplete', 'unknown'] | None = None
    incomplete_details: IncompleteDetails | None = None


class ContextChunk(StrictModel):
    file: RepoPath
    start_line: PositiveInt
    end_line: PositiveInt
    content: str  # Unselected text is omitted; metadata/hash remain inspectable.
    content_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    score: Annotated[int, Field(ge=0)]
    reasons: list[Text]
    selected: bool
    selection_reason: Literal['selected', 'budget', 'max_chunks', 'no_signal']


class RepositoryContext(StrictModel):
    head: CommitSHA
    diff_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    archive_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    repository: Text
    ranking: Literal['structural_lexical_v1']
    config: dict[str, object]
    budget_unit: Literal['serialized_utf8_bytes']
    budget: Annotated[int, Field(ge=0)]
    used_budget: Annotated[int, Field(ge=0)]
    overall_request_byte_ceiling: Annotated[int, Field(ge=1)] | None = None
    reserved_static_signal_byte_cap: Annotated[int, Field(ge=0)] = 0
    reserved_static_signal_estimator_bytes: Annotated[int, Field(ge=0)] = 0
    chunks: list[ContextChunk]

    def payload(self) -> list[dict]:
        return [{k: getattr(c, k) for k in ('file', 'start_line', 'end_line', 'content')}
                for c in self.chunks if c.selected]


class VerificationDecision(StrictModel):
    proposal_id: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    request_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    verdict: Literal['SUPPORTED', 'UNSUPPORTED', 'UNCERTAIN']
    rationale: Text
    evidence_references: list[Text]
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class RequestByteMeasurement(StrictModel):
    serialized_request_body_utf8_bytes: Annotated[int, Field(ge=0)]
    safety_margin_bytes: Literal[4096]
    estimator_value: Annotated[int, Field(ge=4096)]
    configured_estimator_ceiling: Annotated[int, Field(ge=1)] | None = None
    estimator_units: Literal['serialized_request_estimator_units']
    method: Literal['serialized_request_body_utf8_bytes_plus_4096_safety_margin']
    fits: bool | None = None

    @model_validator(mode='after')
    def consistent_estimate(self) -> Self:
        if self.estimator_value != self.serialized_request_body_utf8_bytes + self.safety_margin_bytes:
            raise ValueError('request estimator must equal serialized body bytes plus safety margin')
        if self.configured_estimator_ceiling is None:
            if self.fits is not None:
                raise ValueError('fit status requires a configured estimator ceiling')
        elif self.fits != (self.estimator_value <= self.configured_estimator_ceiling):
            raise ValueError('request estimator fit status is inconsistent')
        return self


class VerificationRecord(StrictModel):
    status: Literal['success', 'failed']
    prompt_version: Text
    prompt_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    provider: Literal['openai']
    model: Text
    settings: dict[str, object]
    request_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')] | None
    proposal_ids: list[Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]]
    request_hashes: list[Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]]
    request_byte_measurements: list[RequestByteMeasurement] = Field(default_factory=list)
    decisions: list[VerificationDecision]
    attempts: list[ModelAttempt]
    latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    error: Text | None = None

    @model_validator(mode='after')
    def consistent(self) -> Self:
        if self.status == 'success' and (self.error is not None or len(self.proposal_ids) != len(self.decisions)
                                         or len(self.request_hashes) != len(self.proposal_ids)):
            raise ValueError('successful verification must decide every proposal')
        if self.status == 'failed' and self.error is None:
            raise ValueError('failed verification requires an error')
        decision_ids = [d.proposal_id for d in self.decisions]
        if (decision_ids != self.proposal_ids[:len(decision_ids)]
                or len(self.request_hashes) > len(self.proposal_ids)
                or len(self.request_hashes) < len(self.decisions)):
            raise ValueError('verification decisions must be an ordered proposal prefix')
        if any(decision.request_sha256 != self.request_hashes[index]
               for index, decision in enumerate(self.decisions)):
            raise ValueError('verification request hashes must match their decisions')
        return self


class BaselineRun(StrictModel):
    """Provenance envelope around the existing ReviewResult, not another finding schema."""
    result: ReviewResult
    diff_base: CommitSHA
    diff_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    provider: Literal["openai"] = "openai"
    requested_model: Text
    settings: dict[str, object]
    prompt_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    prompt_text: Text
    output_schema: dict[str, object]
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None
    request_byte_measurement: RequestByteMeasurement | None = None
    created_at: Text
    attempts: list[ModelAttempt]
    model_latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    acquisition_latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    total_latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    output_text: str | None = None
    input_artifact_directory: Text | None = None
    cost_usd: None = None  # Price/version selection is deliberately deferred.
    retrieval: RepositoryContext | None = None
    retrieval_latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    static_analysis: dict[str, object] | None = None  # StaticAnalysisResult for tools/full modes
    static_analysis_latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    verification: VerificationRecord | None = None
    verifier_latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    proposed_findings: list[Finding] = Field(default_factory=list)
