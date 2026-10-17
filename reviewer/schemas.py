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


class ReviewInput(StrictModel):
    pr: PRMetadata
    diff: str  # Empty string is a valid, explicitly empty change set.
    diff_complete: Literal[True]
    # Old-side coordinates use this merge base when set; otherwise pr.base.
    diff_base: CommitSHA | None = None

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
    budget_unit: Literal['serialized_utf8_bytes_conservative_token_estimate']
    budget: Annotated[int, Field(ge=0)]
    used_budget: Annotated[int, Field(ge=0)]
    chunks: list[ContextChunk]

    def payload(self) -> list[dict]:
        return [{k: getattr(c, k) for k in ('file', 'start_line', 'end_line', 'content')}
                for c in self.chunks if c.selected]


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
