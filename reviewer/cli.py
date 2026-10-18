"""Baseline CLI; acquisition and review run only after explicit command invocation."""

import argparse
from collections.abc import Sequence
import sys
import time

from reviewer.config import load_config, resolve_models, validate_ready
from reviewer.github import AcquisitionError, GitHubClient
from reviewer.models import ModelError, OpenAIReviewer
from reviewer.pipeline import run_baseline, run_review, run_full, prepare_context, prepare_static_analysis
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
    if run.static_analysis is not None and run.static_analysis.get('status') not in ('success', 'no_findings', 'no_files'):
        print(f"\nWARNING: Bandit status={run.static_analysis.get('status')}; "
              "this tools review continued without static-analysis evidence.")
    observed = [attempt.usage for attempt in run.attempts if attempt.usage is not None]
    if observed:
        print(f"\nObserved tokens: input={sum(x.input_tokens for x in observed)}, "
              f"output={sum(x.output_tokens for x in observed)}, total={sum(x.total_tokens for x in observed)}")
    else:
        print("\nToken usage: unavailable")
    if len(observed) != len(run.attempts):
        print("Usage is incomplete for one or more attempts; no cost inferred.")
    if run.verification is not None:
        verifier_usage = [attempt.usage for attempt in run.verification.attempts if attempt.usage is not None]
        print(f"Verifier model: {run.verification.model}; status: {run.verification.status}")
        print(f"Verifier tokens: input={sum(x.input_tokens for x in verifier_usage)}, "
              f"output={sum(x.output_tokens for x in verifier_usage)}, "
              f"total={sum(x.total_tokens for x in verifier_usage)}; "
              f"latency={run.verifier_latency_seconds:.3f}s")
    print(f"Model latency: {run.model_latency_seconds:.3f}s; total: {run.total_latency_seconds:.3f}s")
    print(f"Saved: {path}")


def _mode_group(value):
    if any(mode not in MODES for mode in value.split(',')):
        raise argparse.ArgumentTypeError('invalid choice for --modes')
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reviewer",
                                     description="Review frozen GitHub PRs in baseline, retrieval, tools, or full mode.")
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="Review a frozen public PR using an experiment mode")
    review.add_argument("--repo", required=True)
    review.add_argument("--pr", required=True, type=int)
    review.add_argument("--mode", choices=MODES, default="baseline")
    review.add_argument("--config", default="configs/experiment.toml")
    review.add_argument("--model", help="Override REVIEW_MODEL")
    review.add_argument("--request-byte-budget", "--input-token-limit", dest="request_byte_budget", type=int, help="Serialized request byte estimator ceiling (legacy alias: --input-token-limit)")
    review.add_argument("--max-output-tokens", type=int, help="Output ceiling, including reasoning; default from config")
    review.add_argument("--context-byte-budget", "--context-token-budget", dest="context_byte_budget", type=int, help="Serialized UTF-8 context byte cap (legacy alias: --context-token-budget)")
    review.add_argument("--temperature", type=float, help="Optional; only supported sampling models")
    review.add_argument("--results-dir", help="Override result directory")
    evaluate = commands.add_parser("evaluate", help="Generate frozen ReviewBench reviews; scoring is external")
    evaluate.add_argument("--benchmark", choices=("reviewbench",), required=True)
    evaluate.add_argument("--modes", nargs="+", type=_mode_group, default=["baseline"], help="baseline,retrieval or space-separated modes")
    evaluate.add_argument("--config", default="configs/experiment.toml")
    evaluate.add_argument("--benchmark-archive", required=True, help="Pinned ReviewBench source tar.gz")
    evaluate.add_argument("--subset", default="configs/reviewbench-dev5.json")
    evaluate.add_argument("--model", help="Override REVIEW_MODEL")
    evaluate.add_argument("--request-byte-budget", "--input-token-limit", dest="request_byte_budget", type=int,
                          help="Serialized request byte estimator ceiling (legacy alias: --input-token-limit)")
    evaluate.add_argument("--max-output-tokens", type=int, help="Output ceiling, including reasoning; default from config")
    evaluate.add_argument("--context-byte-budget", "--context-token-budget", dest="context_byte_budget", type=int,
                          help="Serialized UTF-8 context byte cap (legacy alias: --context-token-budget)")
    evaluate.add_argument("--resume-from", help="Validate and reuse a prior run in a new output directory")
    evaluate.add_argument("--retry-failed", action="store_true", help="Explicitly authorize one new attempt on failed tasks")
    evaluate.add_argument("--allow-output-budget-change", action="store_true",
                          help="Allow an increased output ceiling on resume; preserve reused settings")
    evaluate.add_argument("--results-dir")
    evaluate.add_argument("--pricing", help="Optional explicit price provenance JSON; no built-in price table")
    preflight = commands.add_parser("preflight", help="Reconstruct and validate a frozen ReviewBench subset without model calls")
    preflight.add_argument("--benchmark", choices=("reviewbench",), required=True)
    preflight.add_argument("--benchmark-archive", required=True)
    preflight.add_argument("--subset", required=True)
    preflight.add_argument("--config", default="configs/experiment.toml")
    preflight.add_argument("--request-byte-budget", "--input-token-limit", dest="request_byte_budget", type=int,
                           required=True, help="Serialized request byte estimator ceiling (legacy alias: --input-token-limit)")
    preflight.add_argument("--max-output-tokens", type=int, required=True)
    preflight.add_argument("--context-byte-budget", "--context-token-budget", dest="context_byte_budget", type=int,
                           required=True, help="Serialized UTF-8 context byte cap (legacy alias: --context-token-budget)")
    preflight.add_argument("--results-dir", default="results/preflight")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        return _preflight_main(args, parser)
    if args.command == "evaluate":
        args.modes = [mode for group in args.modes for mode in group.split(',')]
        if any(mode not in MODES for mode in args.modes):
            parser.error('invalid choice for --modes')
        return _benchmark_main(args, parser)
    if args.mode not in ('baseline', 'retrieval', 'tools', 'full'):
        parser.error("invalid review mode")
    try:
        config = load_config(args.config)
        if args.request_byte_budget is not None:
            config.review.request_byte_budget = args.request_byte_budget
        if args.max_output_tokens is not None:
            config.review.max_output_tokens = args.max_output_tokens
        if args.context_byte_budget is not None:
            config.retrieval.context_byte_budget = args.context_byte_budget
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
        verifier = None
        if args.mode == 'full':
            if models.verify is None or models.verify.provider != 'openai':
                raise ModelError('unsupported_configuration', 'set VERIFY_PROVIDER=openai and VERIFY_MODEL for full mode')
            verifier = OpenAIReviewer(models.verify.model, config.review)
    except ModelError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except (ValueError, OSError):
        # Do not print Pydantic input values or arbitrary config/environment content.
        print("Configuration error: check configuration, REVIEW_PROVIDER=openai, REVIEW_MODEL, "
              "OPENAI_API_KEY, and a positive --request-byte-budget. Temperature may be unsupported.", file=sys.stderr)
        return 2
    try:
        started = time.perf_counter()
        acquired = GitHubClient().acquire(args.repo, args.pr, snapshots=args.mode in ('retrieval', 'tools', 'full'), cache_dir=config.cache_dir + "/github")
        acquisition_latency = time.perf_counter() - started
        if args.mode == 'baseline':
            run = run_baseline(acquired.review_input, provider, acquisition_latency_seconds=acquisition_latency,
                               input_artifact_directory=str(acquired.artifact_directory))
        elif args.mode == 'retrieval':
            retrieval_started = time.perf_counter()
            context = prepare_context(acquired.review_input, provider, acquired.head_snapshot, config.retrieval,
                                      cache_dir=config.cache_dir + '/retrieval',
                                      reserve_static_signal_byte_cap=config.static_analysis.signal_byte_budget)
            run = run_review(acquired.review_input, provider, mode='retrieval', context=context,
                             retrieval_latency_seconds=time.perf_counter() - retrieval_started,
                             acquisition_latency_seconds=acquisition_latency,
                             input_artifact_directory=str(acquired.artifact_directory))
        else:  # tools/full modes
            retrieval_started = time.perf_counter()
            context = prepare_context(acquired.review_input, provider, acquired.head_snapshot, config.retrieval,
                                      cache_dir=config.cache_dir + '/retrieval',
                                      reserve_static_signal_byte_cap=config.static_analysis.signal_byte_budget)
            retrieval_latency = time.perf_counter() - retrieval_started
            static_started = time.perf_counter()
            static_analysis = prepare_static_analysis(acquired.review_input, acquired.head_snapshot, context, config.static_analysis)
            static_latency = time.perf_counter() - static_started
            if args.mode == 'tools':
                run = run_review(acquired.review_input, provider, mode='tools', context=context,
                                 static_analysis=static_analysis, static_analysis_latency_seconds=static_latency,
                                 retrieval_latency_seconds=retrieval_latency,
                                 acquisition_latency_seconds=acquisition_latency,
                                 input_artifact_directory=str(acquired.artifact_directory))
            else:
                run = run_full(acquired.review_input, provider, verifier, context=context,
                               static_analysis=static_analysis,
                               static_analysis_latency_seconds=static_latency,
                               retrieval_latency_seconds=retrieval_latency,
                               acquisition_latency_seconds=acquisition_latency,
                               signal_budget=config.static_analysis.signal_byte_budget,
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
            or any(mode not in ('baseline', 'retrieval', 'tools', 'full') for mode in args.modes)):
        parser.error('invalid or duplicate evaluation mode')
    failure_code = 'invalid_config_file'
    try:
        config = load_config(args.config)
        failure_code = 'benchmark_pin_mismatch'
        if config.benchmark.revision != REVISION or config.benchmark.manifest_sha256 != MANIFEST_SHA256:
            raise ValueError('unsupported benchmark revision')
        failure_code = 'invalid_request_byte_budget'
        if args.request_byte_budget is not None:
            config.review.request_byte_budget = args.request_byte_budget
        if config.review.request_byte_budget <= 0:
            raise ValueError('positive request byte estimator ceiling required')
        failure_code = 'invalid_output_budget'
        if args.max_output_tokens is not None:
            config.review.max_output_tokens = args.max_output_tokens
        if args.context_byte_budget is not None:
            config.retrieval.context_byte_budget = args.context_byte_budget
        failure_code = 'invalid_attempt_limit'
        if config.review.max_attempts != 1:
            raise ValueError('evaluation requires one model attempt per observation')
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
        verifier = None
        if 'full' in args.modes:
            if models.verify is None or models.verify.provider != 'openai':
                raise ModelError('unsupported_configuration', 'set VERIFY_PROVIDER=openai and VERIFY_MODEL for full mode')
            verifier = OpenAIReviewer(models.verify.model, config.review)
        failure_code = 'benchmark_run_failed'
        path, report = run_development_subset(benchmark, reviewer, GitHubClient(),
                                              args.results_dir or config.results_dir, pricing=pricing,
                                              resume_from=args.resume_from, retry_failed=args.retry_failed,
                                              allow_output_budget_change=args.allow_output_budget_change,
                                              modes=args.modes, retrieval_config=config.retrieval,
                                              static_config=config.static_analysis,
                                              cache_dir=config.cache_dir, verifier=verifier,
                                              verify_model=models.verify.model if models.verify else None)
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
    requested_total = len(benchmark.subset.tasks) * len(args.modes)
    print(f"ReviewBench: {report['status']}; {report['completed_reviews']}/{requested_total} requested reviews completed")
    print(f"Reused: {report.get('reused_reviews', 0)}; new model attempts: {report['model_attempts']}")
    print('Scoring paused: no matching or precision/recall/F1 computed.')
    print(f'Saved: {path}')
    return 0 if report['status'] == 'generation_complete_scoring_paused' else 1


def _preflight_main(args, parser):
    """No-model path: config/payload construction plus read-only frozen acquisition."""
    from reviewer.config import resolve_models
    from reviewer.preflight import PreflightFailure, _safe_error, run_preflight
    try:
        config = load_config(args.config)
        config.review.request_byte_budget = args.request_byte_budget
        config.review.max_output_tokens = args.max_output_tokens
        config.retrieval.context_byte_budget = args.context_byte_budget
        models = resolve_models(config)
        if models.review is None:
            raise ValueError('review model is not configured')
        if models.review.provider != 'openai':
            raise ValueError('only OpenAI-compatible payload configuration is supported')
        path, report = run_preflight(
            archive=args.benchmark_archive, subset_path=args.subset, config=config,
            client=GitHubClient(), results_dir=args.results_dir,
            review_model=models.review.model, review_provider=models.review.provider,
            verify_model=models.verify.model if models.verify else None,
            verify_provider=models.verify.provider if models.verify else None,
            cache_dir=config.cache_dir)
    except PreflightFailure as error:
        task = error.task_id or 'none (pre-task validation)'
        print(f'Preflight failed\nTask: {task}\nStage: {error.stage}\n'
              f'Error: {error.code}: {error.message}', file=sys.stderr)
        if error.report_path is not None:
            print(f'Report: {error.report_path}', file=sys.stderr)
        print('Model calls: 0', file=sys.stderr)
        return 1
    except (ValueError, OSError, KeyError, TypeError, AcquisitionError) as error:
        code, message = _safe_error(error)
        print(f'Preflight failed\nTask: none (pre-task validation)\n'
              f'Stage: configuration\nError: {code}: {message}\nModel calls: 0', file=sys.stderr)
        return 1
    if 'intended_task_count' in report:
        print(f"Preflight: ready; intended={report['intended_task_count']}, "
              f"evaluable={report['evaluable_task_count']}, "
              f"excluded={report['excluded_pre_execution_count']}; model calls: 0")
    else:
        print(f"Preflight: ready; {len(report['tasks'])} tasks; model calls: 0")
    print(f"Benchmark archive SHA-256: {report['benchmark_archive_sha256']}")
    print(f"Subset SHA-256: {report['subset_sha256']}")
    print(f"Saved: {path}")
    return 0
