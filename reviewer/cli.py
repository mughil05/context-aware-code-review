"""Baseline CLI; acquisition and review run only after explicit command invocation."""

import argparse
from collections.abc import Sequence
import sys
import time

from reviewer.config import load_config, resolve_models, validate_ready
from reviewer.github import AcquisitionError, GitHubClient
from reviewer.models import ModelError, OpenAIReviewer
from reviewer.pipeline import run_baseline, run_review, prepare_context
from reviewer.schemas import MODES
from reviewer.storage import save_run


def _display(run, path) -> None:
    result = run.result
    print("Review complete" if result.status == "success" else "Review failed")
    print(f"Mode: {result.mode}\nModel requested: {run.requested_model}")
    if run.attempts and run.attempts[-1].returned_model:
        print(f"Model returned: {run.attempts[-1].returned_model}")
    if result.error:
        print(result.error)
    else:
        print(f"\n{len(result.findings)} findings")
        for finding in result.findings:
            print(f"\n[{finding.severity.upper()}] {finding.title}")
            print(f"File: {finding.file}:{finding.start_line}-{finding.end_line}")
            print(f"Confidence: {finding.confidence:.2f}")
            print(f"\nDescription:\n{finding.description}\n\nEvidence:")
            for evidence in finding.evidence:
                print(f"- {evidence}")
            print(f"\nSuggested fix:\n{finding.suggested_fix}")
    observed = [attempt.usage for attempt in run.attempts if attempt.usage is not None]
    if observed:
        print(f"\nObserved tokens: input={sum(x.input_tokens for x in observed)}, "
              f"output={sum(x.output_tokens for x in observed)}, total={sum(x.total_tokens for x in observed)}")
    else:
        print("\nToken usage: unavailable")
    if len(observed) != len(run.attempts):
        print("Usage is incomplete for one or more attempts; no cost inferred.")
    print(f"Model latency: {run.model_latency_seconds:.3f}s; total: {run.total_latency_seconds:.3f}s")
    print(f"Saved: {path}")


def _mode_group(value):
    if any(mode not in MODES for mode in value.split(',')):
        raise argparse.ArgumentTypeError('invalid choice for --modes')
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reviewer", description="Diff-only AI code reviewer (baseline).")
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="Review a frozen public PR with the diff-only baseline")
    review.add_argument("--repo", required=True)
    review.add_argument("--pr", required=True, type=int)
    review.add_argument("--mode", choices=MODES, default="baseline")
    review.add_argument("--config", default="configs/experiment.toml")
    review.add_argument("--model", help="Override REVIEW_MODEL")
    review.add_argument("--input-token-limit", type=int, help="Explicit conservative request budget")
    review.add_argument("--max-output-tokens", type=int, help="Output ceiling, including reasoning; default from config")
    review.add_argument("--context-token-budget", type=int, help="Conservative serialized context budget; retrieval only")
    review.add_argument("--temperature", type=float, help="Optional; only supported sampling models")
    review.add_argument("--results-dir", help="Override result directory")
    evaluate = commands.add_parser("evaluate", help="Generate five ReviewBench baseline reviews; scoring is paused")
    evaluate.add_argument("--benchmark", choices=("reviewbench",), required=True)
    evaluate.add_argument("--modes", nargs="+", type=_mode_group, default=["baseline"], help="baseline,retrieval or space-separated modes")
    evaluate.add_argument("--config", default="configs/experiment.toml")
    evaluate.add_argument("--benchmark-archive", required=True, help="Pinned ReviewBench source tar.gz")
    evaluate.add_argument("--subset", default="configs/reviewbench-dev5.json")
    evaluate.add_argument("--model", help="Override REVIEW_MODEL")
    evaluate.add_argument("--input-token-limit", type=int)
    evaluate.add_argument("--max-output-tokens", type=int, help="Output ceiling, including reasoning; default from config")
    evaluate.add_argument("--context-token-budget", type=int)
    evaluate.add_argument("--resume-from", help="Validate and reuse a prior run in a new output directory")
    evaluate.add_argument("--retry-failed", action="store_true", help="Explicitly authorize one new attempt on failed tasks")
    evaluate.add_argument("--allow-output-budget-change", action="store_true",
                          help="Allow an increased output ceiling on resume; preserve reused settings")
    evaluate.add_argument("--results-dir")
    evaluate.add_argument("--pricing", help="Optional explicit price provenance JSON; no built-in price table")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        args.modes = [mode for group in args.modes for mode in group.split(',')]
        if any(mode not in MODES for mode in args.modes):
            parser.error('invalid choice for --modes')
        return _benchmark_main(args, parser)
    if args.mode not in ('baseline', 'retrieval'):
        parser.error("only baseline and retrieval review are implemented")
    try:
        config = load_config(args.config)
        if args.input_token_limit is not None:
            config.review.input_token_limit = args.input_token_limit
        if args.max_output_tokens is not None:
            config.review.max_output_tokens = args.max_output_tokens
        if args.context_token_budget is not None:
            config.retrieval.context_token_budget = args.context_token_budget
        if args.temperature is not None:
            config.review.temperature = args.temperature
        if args.results_dir is not None:
            config.results_dir = args.results_dir
        overrides = {"review_model": args.model} if args.model is not None else {}
        models = resolve_models(config, overrides=overrides)
        validate_ready(config, models, args.mode)
        if models.review.provider != "openai":
            raise ModelError("unsupported_configuration", "only REVIEW_PROVIDER=openai is implemented")
        # Credential/config errors precede all GitHub/network access.
        provider = OpenAIReviewer(models.review.model, config.review)
    except ModelError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except (ValueError, OSError):
        # Do not print Pydantic input values or arbitrary config/environment content.
        print("Configuration error: check configuration, REVIEW_PROVIDER=openai, REVIEW_MODEL, "
              "OPENAI_API_KEY, and a positive --input-token-limit. Temperature may be unsupported.", file=sys.stderr)
        return 2
    try:
        started = time.perf_counter()
        acquired = GitHubClient().acquire(args.repo, args.pr, snapshots=args.mode == 'retrieval', cache_dir=config.cache_dir + "/github")
        acquisition_latency = time.perf_counter() - started
        if args.mode == 'baseline':
            run = run_baseline(acquired.review_input, provider, acquisition_latency_seconds=acquisition_latency,
                               input_artifact_directory=str(acquired.artifact_directory))
        else:
            retrieval_started = time.perf_counter()
            context = prepare_context(acquired.review_input, provider, acquired.head_snapshot, config.retrieval,
                                      cache_dir=config.cache_dir + '/retrieval')
            run = run_review(acquired.review_input, provider, mode='retrieval', context=context,
                             retrieval_latency_seconds=time.perf_counter() - retrieval_started,
                             acquisition_latency_seconds=acquisition_latency,
                             input_artifact_directory=str(acquired.artifact_directory))
        path = save_run(run, config.results_dir)
    except AcquisitionError as error:
        print(f"Acquisition failed: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print("Review failed: unable to validate or persist the run; no successful result reported.", file=sys.stderr)
        return 1
    _display(run, path)
    return 0 if run.result.status == "success" else 1


def _benchmark_main(args, parser):
    from pathlib import Path
    from reviewer.benchmark import load_benchmark, read_json, REVISION, MANIFEST_SHA256
    from reviewer.evaluation import Pricing, run_development_subset
    if (not args.modes or len(set(args.modes)) != len(args.modes)
            or any(mode not in ('baseline', 'retrieval') for mode in args.modes)):
        parser.error('only unique baseline/retrieval modes are implemented; scoring is paused')
    failure_code = 'invalid_config_file'
    try:
        config = load_config(args.config)
        failure_code = 'benchmark_pin_mismatch'
        if config.benchmark.revision != REVISION or config.benchmark.manifest_sha256 != MANIFEST_SHA256:
            raise ValueError('unsupported benchmark revision')
        failure_code = 'invalid_input_budget'
        if args.input_token_limit is not None:
            config.review.input_token_limit = args.input_token_limit
        if config.review.input_token_limit <= 0:
            raise ValueError('positive input budget required')
        failure_code = 'invalid_output_budget'
        if args.max_output_tokens is not None:
            config.review.max_output_tokens = args.max_output_tokens
        if args.context_token_budget is not None:
            config.retrieval.context_token_budget = args.context_token_budget
        failure_code = 'invalid_attempt_limit'
        if config.review.max_attempts != 1:
            raise ValueError('five-task development run requires one attempt per task')
        failure_code = 'invalid_model_configuration'
        models = resolve_models(config, overrides={'review_model': args.model} if args.model else {})
        for mode in args.modes:
            validate_ready(config, models, mode)
        if models.review.provider != 'openai':
            raise ValueError('only OpenAI is implemented')
        failure_code = 'invalid_benchmark_data'
        benchmark = load_benchmark(args.benchmark_archive, args.subset)
        failure_code = 'invalid_pricing_configuration'
        pricing = Pricing.model_validate(read_json(Path(args.pricing).read_bytes())) if args.pricing else None
        failure_code = 'invalid_model_configuration'
        reviewer = OpenAIReviewer(models.review.model, config.review)
        failure_code = 'benchmark_run_failed'
        path, report = run_development_subset(benchmark, reviewer, GitHubClient(),
                                              args.results_dir or config.results_dir, pricing=pricing,
                                              resume_from=args.resume_from, retry_failed=args.retry_failed,
                                              allow_output_budget_change=args.allow_output_budget_change,
                                              modes=args.modes, retrieval_config=config.retrieval,
                                              cache_dir=config.cache_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        # Emit only fixed codes, never exception bodies, paths, or Pydantic values.
        if failure_code == 'invalid_benchmark_data':
            if isinstance(error, OSError):
                artifact = ('subset' if error.filename == str(args.subset) else
                            'archive' if error.filename == str(args.benchmark_archive) else 'benchmark')
                failure_code = artifact + ('_missing' if isinstance(error, FileNotFoundError) else '_unreadable')
            else:
                failure_code = {
                    'subset manifest hash differs from pinned manifest': 'subset_manifest_hash_mismatch',
                    'benchmark manifest hash mismatch': 'manifest_hash_mismatch',
                    'golden file hash mismatch': 'golden_hash_mismatch',
                    'unexpected archive root': 'archive_revision_mismatch',
                    'unreadable benchmark archive': 'archive_unreadable',
                    'missing benchmark manifest or selected golden file': 'archive_members_missing',
                    'selected task metadata differs from manifest': 'subset_metadata_mismatch',
                }.get(str(error), failure_code)
        elif isinstance(error, ModelError) and error.code in ('missing_api_key', 'unsupported_configuration'):
            failure_code = error.code
        elif getattr(error, 'code', None) == 'incompatible_resume':
            failure_code = 'incompatible_resume: check saved identities/settings and explicit resume flags'
        print(f'Benchmark configuration failed: {failure_code}. No successful benchmark reported.', file=sys.stderr)
        return 2
    print(f"ReviewBench: {report['status']}; {report['completed_reviews']}/{5 * len(args.modes)} reviews completed")
    print(f"Reused: {report.get('reused_reviews', 0)}; new model attempts: {report['model_attempts']}")
    print('Scoring paused: no matching or precision/recall/F1 computed.')
    print(f'Saved: {path}')
    return 0 if report['status'] == 'generation_complete_scoring_paused' else 1
