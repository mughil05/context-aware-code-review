"""Strict TOML configuration with explicit model resolution; no network or secrets."""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from reviewer.schemas import Category, CommitSHA, Mode, PositiveInt, StrictModel, Text


class BenchmarkConfig(StrictModel):
    name: Literal["reviewbench"]
    repository: Literal["https://github.com/review-bench/ReviewBench"]
    revision: CommitSHA
    manifest: Text
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    language: Literal["Python"]
    categories: list[Category]
    scopes: list[Literal["introduced-by-pr", "exacerbated-by-pr"]]
    split_manifest: str

    @model_validator(mode="after")
    def unique_filters(self) -> Self:
        for values in (self.categories, self.scopes):
            if not values or len(values) != len(set(values)):
                raise ValueError("benchmark filters must be nonempty and unique")
        return self


EnvName = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$")]


class ModelConfig(StrictModel):
    review_model_env: EnvName
    verify_model_env: EnvName
    embedding_model_env: EnvName
    judge_model_env: EnvName


class ReviewConfig(StrictModel):
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] | None = None
    timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 60.0
    max_output_tokens: PositiveInt
    max_attempts: PositiveInt
    input_token_limit: Annotated[int, Field(ge=0)]


class RetrievalConfig(StrictModel):
    context_token_budget: PositiveInt
    max_chunks: PositiveInt
    chunk_target_lines: PositiveInt
    ranking: Literal["structural_lexical_v1"]


class StaticConfig(StrictModel):
    tool: Literal["bandit"]
    timeout_seconds: PositiveInt
    signal_token_budget: PositiveInt


class VerificationConfig(StrictModel):
    uncertain_policy: Literal["drop"]


class EvaluationConfig(StrictModel):
    """Settings only. Evaluation machinery and final methodology remain deferred."""
    protocol: Literal["reviewbench_existing_ground_truth_pending_review"]
    split_seed: Annotated[int, Field(ge=0)]
    development_repository_fraction: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    bootstrap_replicates: PositiveInt
    human_audit_required: Literal[False]

    @field_validator("human_audit_required", mode="before")
    @classmethod
    def boolean_policy(cls, value):
        if type(value) is not bool:
            raise ValueError("human_audit_required must be a boolean")
        return value


class Capabilities(StrictModel):
    retrieval: bool
    static_analysis: bool
    verification: bool


MODE_FLAGS = {
    "baseline": (False, False, False),
    "retrieval": (True, False, False),
    "tools": (True, True, False),
    "full": (True, True, True),
}


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    seed: Annotated[int, Field(ge=0)]
    repetitions: PositiveInt
    results_dir: Text
    cache_dir: Text
    benchmark: BenchmarkConfig
    models: ModelConfig
    review: ReviewConfig
    retrieval: RetrievalConfig
    static_analysis: StaticConfig
    verification: VerificationConfig
    evaluation: EvaluationConfig
    modes: dict[Mode, Capabilities]

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def preserve_ablation(self) -> Self:
        if set(self.modes) != set(MODE_FLAGS):
            raise ValueError("all four experiment modes are required")
        for name, expected in MODE_FLAGS.items():
            flags = self.modes[name]
            if (flags.retrieval, flags.static_analysis, flags.verification) != expected:
                raise ValueError(f"{name} capabilities must preserve the A-D ablation")
        return self


class ModelSelection(StrictModel):
    provider: Text
    model: Text


class ResolvedModels(StrictModel):
    review: ModelSelection | None = None
    verify: ModelSelection | None = None
    embedding: ModelSelection | None = None
    judge: ModelSelection | None = None


def load_config(path: str | Path) -> ExperimentConfig:
    """Validate settings without requiring credentials or experiment readiness."""
    with Path(path).open("rb") as stream:
        return ExperimentConfig.model_validate(tomllib.load(stream))


def resolve_models(config: ExperimentConfig, *, environment: Mapping[str, str] | None = None,
                   overrides: Mapping[str, str] | None = None) -> ResolvedModels:
    """Explicit CLI-style overrides > environment. Read only model/provider names.

    Override keys are review_model/review_provider, verify_*, embedding_*, judge_*.
    Never read or store API keys, and never load .env implicitly.
    """
    env = os.environ if environment is None else environment
    overrides = {} if overrides is None else overrides
    roles = ("review", "verify", "embedding", "judge")
    allowed = {f"{role}_{field}" for role in roles for field in ("model", "provider")}
    if set(overrides) - allowed:
        raise ValueError("unknown model override keys")
    selections = {}
    for role in roles:
        model_env = getattr(config.models, f"{role}_model_env")
        model = overrides.get(f"{role}_model", env.get(model_env, ""))
        provider = overrides.get(f"{role}_provider", env.get(f"{role.upper()}_PROVIDER", ""))
        if not isinstance(model, str) or not isinstance(provider, str):
            raise ValueError("model/provider settings must be strings")
        model, provider = model.strip(), provider.strip()
        if bool(model) != bool(provider):
            raise ValueError(f"{role} requires both model and provider")
        selections[role] = ModelSelection(model=model, provider=provider) if model else None
    return ResolvedModels(**selections)


def validate_ready(config: ExperimentConfig, models: ResolvedModels, mode: Mode,
                   *, final_experiment: bool = False) -> None:
    """Explicit preflight only; does not run a pipeline or inspect remote services."""
    if mode not in MODE_FLAGS:
        raise ValueError("unknown experiment mode")
    # Revalidate to catch mutations inside lists/dicts after initial validation.
    config = ExperimentConfig.model_validate(config.model_dump())
    models = ResolvedModels.model_validate(models.model_dump())
    required = ["review"]
    # Current retrieval is structural/lexical and needs no embedding provider.
    if config.modes[mode].verification:
        required.append("verify")
    if config.review.input_token_limit == 0:
        raise ValueError("set the model-specific input token limit before running")
    optional_tokens = (config.retrieval.context_token_budget if config.modes[mode].retrieval else 0)
    if config.modes[mode].static_analysis:
        optional_tokens += config.static_analysis.signal_token_budget
    if optional_tokens >= config.review.input_token_limit:
        raise ValueError("input token limit must leave room for the diff and review instructions")
    if any(getattr(models, role) is None for role in required):
        raise ValueError("missing required model/provider selection: " + ", ".join(required))
    if final_experiment:
        if not config.benchmark.split_manifest.strip():
            raise ValueError("freeze a split manifest before a final experiment")
        raise ValueError("final evaluation protocol is deferred; review upstream machinery first")
