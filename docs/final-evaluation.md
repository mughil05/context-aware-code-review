# Final ReviewBench evaluation runbook

## Frozen sample and pre-execution exclusion

The final manifest was mechanically selected from pinned ReviewBench revision
`968291d99e82b3e2969d7e0b4acca62585886c13`, in corpus order, using only Python
language, at most 1,000 added plus removed lines, and exclusion of the development
subset. Exactly 20 tasks were selected before any final model generation. The frozen
manifest `configs/reviewbench-final20.json` remains unchanged.

One task, `nipoppy_nipoppy_964-9421a714`, was excluded before execution. ReviewBench
metadata records 5 files/+42/-10, while its exact pinned base/head reconstructs
unambiguously to 3 files/+38/-0. Its frozen HEAD archive also contains the symlink
`tests/data/template_pipeline -> ../../nipoppy/data/template_pipeline/`, which the
safe snapshot extractor intentionally rejects. Neither frozen inventory validation
nor archive safety was weakened, and no replacement task was selected. The
exclusion applies identically to A/B/C/D, with zero reviewer and judge calls, and is
reported outside completed-PR and metric denominators. The intended sample is 20;
the evaluable corpus is 19.

Old development and smoke-test artifacts are not final results and must not be
combined with the final runs.

## Frozen request budgets

The application request guard is not a token limit. It measures compact serialized
request-body UTF-8 bytes plus a 4,096-byte safety margin. The fixed common estimator
ceiling is 73,728 units (72 KiB). It was frozen before final model calls using the
rule 66,594 measured maximum × 1.10 = 73,253.4, rounded upward to the next 1 KiB.
The 6,000 retrieval context cap and 2,000 static-signal cap are also serialized-byte
estimator budgets, not tokenizer measurements.

A preserves the complete diff and fails closed if the complete request exceeds the
ceiling. B/C/D use the same deterministic retrieval context; selection reserves the
static-signal allowance for C/D, so B does not receive a larger context. C and D
reviewer payloads are checked after assembly. Each verifier request uses the same
73,728 estimator ceiling and fails closed if its bounded proposal/hunk/context/signal
payload does not fit. No diff is truncated. Preflight and model execution call the
same request builder and estimator.

## No-model preflight

Use the pinned archive and final manifest. This performs read-only GitHub
reconstruction, frozen snapshot/retrieval/Bandit preparation, all four payload
budget checks, and label-leakage checks; it creates no model client/call. Only when
all 19 evaluable tasks pass and the single declared exclusion matches its fixed
provenance does it create the final lock.

```sh
REVIEW_PROVIDER=openai REVIEW_MODEL=gpt-6-luna \
VERIFY_PROVIDER=openai VERIFY_MODEL=gpt-6-luna \
/opt/anaconda3/bin/python -m reviewer preflight \
  --benchmark reviewbench \
  --benchmark-archive .cache/reviewbench/ReviewBench-968291d99e82b3e2969d7e0b4acca62585886c13.tar.gz \
  --subset configs/reviewbench-final20.json \
  --config configs/experiment.toml \
  --request-byte-budget 73728 \
  --max-output-tokens 8192 \
  --context-byte-budget 6000 \
  --results-dir results/preflight
```

The lock records the benchmark/archive/manifest/subset identities, all 19 frozen
diff hashes, Nipoppy exclusion details and reason codes, model/prompt/configuration
provenance, Bandit version, and byte-estimator budgets. A failed or interrupted
preflight never creates or updates a successful lock.

## Final generation and scoring

After preflight succeeds, freeze the lock and config. The generation CLI accepts
comma-separated modes; the four-mode run is:

```sh
REVIEW_PROVIDER=openai REVIEW_MODEL=gpt-6-luna \
VERIFY_PROVIDER=openai VERIFY_MODEL=gpt-6-luna \
/opt/anaconda3/bin/python -m reviewer evaluate \
  --benchmark reviewbench \
  --modes baseline,retrieval,tools,full \
  --benchmark-archive .cache/reviewbench/ReviewBench-968291d99e82b3e2969d7e0b4acca62585886c13.tar.gz \
  --subset configs/reviewbench-final20.json \
  --request-byte-budget 73728 \
  --max-output-tokens 8192 \
  --context-byte-budget 6000
```

Generation persists successful task/mode artifacts incrementally. ReviewBench
candidate exports are produced only for the 19 evaluable tasks. Check the final
summary for intended/evaluable/completed/failed/excluded counts, then judge only the
19-task scoped golden directory. `scripts/reviewbench_scope.py filter-goldens`
excludes the declared Nipoppy task from judge input; its aggregation adapter checks
the same recorded exclusion and excludes it from all denominators. ReviewBench's
pinned matcher/judge remains responsible for semantic matching and unmatched
classification. No model, judge, or quality metrics are run by preflight.


## Primary scope-adapted metrics

Judge against filtered goldens, preserving original TP/FP labels. Unmatched
candidates are classified by ReviewBench; out-of-contract category/scope results
are excluded and counted separately. No custom matcher is used.

Let M be matched candidate TPs, F all eligible candidate FPs, N eligible novel TPs,
G eligible golden TPs, and C distinct eligible golden TPs covered according to
ReviewBench detailed output. Primary augmented precision is (M+N)/(M+N+F);
primary augmented recall is (C+N)/(G+N). Primary F1 is their harmonic mean.
Fixed-golden recall C/G and grounded precision remain separately named diagnostics.
Novel TPs expand both the augmented recall numerator and denominator. These are
scope-filtered/adapted metrics, not unfiltered native ReviewBench scores.

`micro` pools counts before calculating metrics. `macro` averages each defined
per-PR metric; macro F1 averages per-PR F1, not the harmonic mean of macro P/R.
`macro_defined_pr_counts` exposes each denominator. Zero denominators produce null;
F1 is null if either P/R is undefined, and zero if both are defined and sum to zero.
Top-level precision/recall/F1 are aliases of the primary pooled metrics.

FP/PR divides eligible FPs by completed PRs including empty reviews. The separate
FP-per-evaluable-PR value is null until every evaluable PR completes, avoiding an
assumption of zero FPs in failed/unattempted tasks. Nipoppy is outside both PR
denominators. Per-mode aggregation uses identical formulas. Generation artifacts
are read-only; metric changes do not regenerate any reviewer/verifier output.

Judging prerequisites: the pinned upstream model registry must support the chosen
provider/model and have credentials configured. A successful Python reviewer call
does not prove compatibility with ReviewBench's separate TypeScript SDK. Do not
start paid judging until registry compatibility is established offline.


## Predeclared primary judge: OpenAI gpt-5.4

Before any quality scores were observed, the judge was fixed to `openai/gpt-5.4`
for baseline/retrieval/tools/full, concurrency 1. `gpt-6-luna` was initially
preferred but rejected before scoring because it is absent from the pinned SDK
registry. Generation remains OpenAI gpt-6-luna and is unchanged.

`configs/reviewbench-scoring.json` freezes judge choice, registry metadata, upstream
revision/lockfile/patched-source hashes, exact candidate/golden hashes, rendered
prompt identities, and per-mode checkpoint fingerprints calculated using the
pinned upstream fingerprint function. No matcher/classifier semantics changed.
Pinned registry gpt-5.4 contextWindow=272000 and maxTokens=128000; the judge uses
low thinking for matching and medium for classification. These judge settings do
not inherit the product review's 8192 output-token budget. OPENAI_API_KEY is required.

Use `scripts/reviewbench_scoring_check.py inputs` before judging. Run one mode at a
time, in baseline/retrieval/tools/full order. Use upstream strict mode (default),
never --ingest or --no-strict, filtered goldens and the pinned upstream manifest.
Save stdout/stderr to `scoring/<mode>.log`; append `JUDGE_EXIT=0` only when the
judge process exits zero. Then `... complete --mode <mode>` checks 19 unique PRs,
exact candidate/detail identities and labels/scopes, frozen judge/input hashes,
no checkpoint, and a successful error-free log before advancing.

On interruption, stop and preserve logs/checkpoint/outputs. `... checkpoint --mode
<mode>` verifies the exact frozen fingerprint and completed PR records without
judging. A compatible incomplete checkpoint permits retrying only that mode with
the identical output path and settings. Archive the old log before retrying.
An absent, malformed or incompatible checkpoint must be investigated: upstream
can ignore it and restart paid work. An all-19 checkpoint needs finalization
inspection, not blind retry. Completed modes must never be rerun. In-flight PR
work may repeat; upstream checkpoints after each entire PR and writes non-atomically.

Pinned strict mode refuses to write a partial aggregate when PR errors occur.
SDK retries (matcher 2, classifier 3), classifier format repair and tool turns can
add paid requests. Success requires 19 completed PRs and zero errors, not merely
an output file. The upstream classifier may fetch/check out repositories in the
separate judge repo directory; no project Git operation is needed for preparation.

After all four completion checks pass, use `scripts/reviewbench_scope.py aggregate`
with the ORIGINAL reference directory (to retain excluded counts), detailed judged
outputs from FILTERED goldens, and one separate `.scoped.json` per mode. Finally
`scripts/reviewbench_scoring_check.py compare` validates their provenance and prints
a deterministic tab-separated comparison without judging, rematching or scoring.
