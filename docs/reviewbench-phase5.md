# ReviewBench five-task development integration

**Historical phase, closed by owner decision.** Do not continue the interrupted
dev-5 runs as part of Phase 6. Review prompt contract v2 clarifies diff anchoring;
its new hash intentionally makes old-prompt runs incompatible with current resume.
The commands and run discussion below describe Phase 5. Current retrieval and A/B
generation are documented in [Phase 6](retrieval-phase6.md). No quality scores exist.

This phase implements baseline **generation and export**. Scoring integration was
explicitly paused by the owner after inspection showed that upstream semantic
matching requires an LLM. The runner does not match findings, invoke a judge, or
compute TP/FP/FN, precision, recall, or F1. Every summary has `scoring_status:
"paused"` and `quality_metrics: null`, even for zero predictions. There are no
benchmark performance results yet.

## Inspected sources

Pinned ReviewBench revision: `968291d99e82b3e2969d7e0b4acca62585886c13` (package
version 1.0.0). Current main observed during this phase:
`2e45130f1f6bbdd33747a6963727051009754399`. The manifest, matcher, scorer,
`match-types.ts`, and judging-input documentation were byte-identical between
these revisions. The project retains the existing pin.

- [Manifest](https://github.com/review-bench/ReviewBench/blob/968291d99e82b3e2969d7e0b4acca62585886c13/corpus/manifest.json): JSON array, with `repo`, `pr_number`, `base`, `head`, `nwo`,
  `language`, title/body, repository size, and changed-line/file counts.
- [Golden example](https://github.com/review-bench/ReviewBench/blob/968291d99e82b3e2969d7e0b4acca62585886c13/golden/AA-Factory_aafactory-prototype_17-a1978cb7.json):
  `{pr_key, pr, findings}`. Findings include `file`, inclusive head `start_line` /
  `end_line`, `message`, `tp_fp`, severity, category, scope, difficulty,
  context requirements, justifications, and producer/source provenance.
- [Prediction format](https://github.com/review-bench/ReviewBench/blob/968291d99e82b3e2969d7e0b4acca62585886c13/docs/JUDGING_INPUT.md):
  `{pr, agent, findings, usage?}`. Each prediction has producer, file, line range,
  and message. The adapter concatenates our title, description, evidence, and fix
  into `message`. Severity, category, and confidence remain in the original run;
  they are not native prediction fields. Usage names are mapped to upstream names.
- [Runner](https://github.com/review-bench/ReviewBench/blob/968291d99e82b3e2969d7e0b4acca62585886c13/scripts/try-agent.sh):
  constructs three-dot `base...head` diffs from mirrors. We reproduce those frozen
  comparison semantics through GitHub HTTP and our existing parser. We do not
  invoke this script: it requires Git and Docker and defaults to larger sets.

## Why scoring is paused

The [matcher](https://github.com/review-bench/ReviewBench/blob/968291d99e82b3e2969d7e0b4acca62585886c13/scripts/eval/matcher.ts)
uses an LLM, compares findings within the same file, and allows many-to-many semantic
correspondences. Matching is about the underlying defect/fix, not line overlap.
There is no deterministic semantic matcher available to reuse.

[resolveForScoring](https://github.com/review-bench/ReviewBench/blob/968291d99e82b3e2969d7e0b4acca62585886c13/scripts/lib/match-types.ts)
counts coverage of all matched golden indices, but awards each candidate at most
one previously unclaimed golden index in candidate order. Duplicate matches that
lose scoring credit become unmatched candidates. Upstream then uses a classifier
LLM for unmatched predictions. Its standalone `npm run judge` therefore spends API
credit; “offline evaluation” means outside its leaderboard infrastructure, not
without model calls.

The [scorer](https://github.com/review-bench/ReviewBench/blob/968291d99e82b3e2969d7e0b4acca62585886c13/scripts/eval/scorer.ts)
is deterministic **after** correspondence and classification data exist:

- Grounded precision = matched candidates assigned a TP label / matched candidates.
- Grounded recall = covered golden TPs / all golden TPs.
- Augmented precision includes unmatched candidates classified as true positives.
- Augmented recall adds novel true positives to both numerator and denominator.
- Zero denominators yield `null`; macro averages omit null values; micro averages
  pool counts. Upstream does not export a conventional single TP/FP/FN/F1 family.
- Its low-level scorer defaults missing unmatched classifications to FP. We must
  not use that fallback to present unjudged findings as known false positives.

No matching, score importer, alternative metric, or classifier is implemented in
this phase. Scoring tests for TP/FP/FN and aggregate quality metrics are deferred
with the owner's scoring decision. Before later scoring, resolve the mismatch
between our defect-only review contract and upstream labels covering other
categories and pre-existing issues. Original labels are preserved, not relabeled
or filtered for a convenient score.

## Locked development subset

`configs/reviewbench-dev5.json` records all five exact task IDs, original repository
URLs, PR numbers, full base/head SHAs, official mirror names, manifest counts,
golden-file hashes, and measured frozen-diff hashes. These are development cases,
not a representative sample or a test set. Selection used size and label inventory,
never model performance. No prompt was tuned against these findings.

| Task ID | Repository / PR | Files | Added / removed | Qualifying TP inventory |
| --- | --- | ---: | ---: | ---: |
| `AA-Factory_aafactory-prototype_17-a1978cb7` | AA-Factory/aafactory-prototype #17 | 1 | 45 / 0 | 1 |
| `Tzeusy_butlers_1021-403e15fe` | Tzeusy/butlers #1021 | 7 | 278 / 13 | 0 |
| `allan-mobley-jr_gimmes_468-e1477f8b` | allan-mobley-jr/gimmes #468 | 4 | 129 / 30 | 2 |
| `YasserShkeir_django-smart-ratelimit_25-d7c5c7c2` | YasserShkeir/django-smart-ratelimit #25 | 10 | 195 / 24 | 7 |
| `ZhuoZhuoCrayon_throttled-py_135-94c7df60` | ZhuoZhuoCrayon/throttled-py #135 | 7 | 235 / 17 | 2 |

The qualifying inventory predicate is TP + one of correctness/security/reliability/
performance + introduced/exacerbated scope. These inventory counts are not scores.
This small selection covers correctness, reliability, and performance; it does not
provide meaningful coverage of every category. Tzeusy #1021 has 18 upstream TP
labels in testing/maintainability, so zero qualifying defects does not mean zero
upstream positives or a bug-free PR. All selected diffs include Python files.

The initially considered healthstream-rag #34 was excluded: its manifest reports
84 additions / 10 deletions, but the exact frozen comparison gives 71 / 9, and the
changed files are documentation. No revisions or counts were silently substituted.

## Reconstruction and information boundary

The archive loader verifies the pinned manifest SHA-256 and each selected golden
hash, matches identity/count metadata to the subset lock, and reads only selected
regular archive members without extraction. No benchmark scripts are executed.
Golden metadata stays separate from task metadata and `ReviewInput`.

`GitHubClient.frozen_input` uses exact base/head SHAs against the explicitly locked
mirror. It checks the head commit identity, comparison base and merge base, full
file inventory, parsed line totals, and exact raw diff hash. Comparison inventories
at the 300-file limit are unsupported; this subset is far below that bound. Missing
commits, malformed responses, unsupported diffs, and mismatched hashes fail. It
never reads current PR metadata, current branches, PR comments, or later fixes.
There is no automatic repository fallback. Coordinates use the merge base on the
old side and frozen head on the new side, as in the existing acquisition path.

All five tasks reconstruct and pass the request byte estimator check before the first model
call. `run_baseline` receives only the validated `ReviewInput`, the existing model
adapter, timing, and an artifact path. The model receives only the diff, stable
instructions, and schema. Labels, scope metadata, titles/bodies, and task IDs are
not model inputs. An offline test puts a unique label sentinel in both golden data
and metadata and checks every mock model payload for its absence.

## Commands (not executed against models during implementation)

To obtain the pinned source without Git:

```sh
mkdir -p .cache/reviewbench
curl --fail --location \
  https://codeload.github.com/review-bench/ReviewBench/tar.gz/968291d99e82b3e2969d7e0b4acca62585886c13 \
  --output .cache/reviewbench/ReviewBench-968291d99e82b3e2969d7e0b4acca62585886c13.tar.gz
```

Run from the project directory with an exported `OPENAI_API_KEY`:

```sh
REVIEW_PROVIDER=openai REVIEW_MODEL=gpt-6-luna \
/opt/anaconda3/bin/python -m reviewer evaluate \
  --benchmark reviewbench --modes baseline \
  --benchmark-archive .cache/reviewbench/ReviewBench-968291d99e82b3e2969d7e0b4acca62585886c13.tar.gz \
  --subset configs/reviewbench-dev5.json \
  --request-byte-budget 32000
```

On other machines use a Python 3.11+ environment with Pydantic 2. There are no new
runtime dependencies. `max_attempts` must be 1; the future experiment configuration's
three repetitions are deliberately not used. At most five reviewer API calls, one
per task, are made. A model failure stops the batch and is preserved as a failure,
not exported as a clean empty review. No matcher/classifier calls occur. Other
modes and full-dataset selection are rejected.

Measured conservative request estimates using the unchanged prompt/schema were
9,858; 26,395; 21,578; 26,252; and 25,824, in table order. These are serialized UTF-8
bytes plus the adapter's 4,096 allowance, not tokenizer measurements. A 32,000
budget passes all five; oversized inputs fail rather than truncate. The output cap
is 4,096 tokens per call. At published standard GPT-6 Luna pricing of $0.10/M input
and $0.50/M output, budgeting 32,000 input plus maximum output for all five is about
$0.02624. This is a planning estimate, not observed spend or an enforced dollar cap.
[Pricing source](https://developers.openai.com/api/docs/models/gpt-6-luna).

Optional `--pricing PATH` accepts JSON with `model`, `input_usd_per_million`,
`output_usd_per_million`, `cached_input_usd_per_million`, `source`, and `as_of`.
No price table is hardcoded. The rates must match the requested model. With full
usage, cost is calculated from input/output tokens and cached input if reported;
reasoning tokens are already included in output and are not counted twice. Unknown
cached usage is charged as uncached for this estimate. Without prices or complete
usage, cost is `null`.

## Resuming an interrupted development run

`evaluate --resume-from RUN_DIRECTORY` validates the saved run before any request,
then creates a new exclusive result directory. Original artifacts are never
overwritten. Every validated successful raw result is copied byte-for-byte, with
its original model settings and timing, and gets a candidate export in the new
directory. Saved frozen inputs are reused for every task, so resume does not fetch
GitHub again. Never-attempted tasks continue through the existing baseline.

Failed tasks require `--retry-failed`. A resumed invocation makes at most one new
model attempt per pending task and stops at its first new failure. The new summary
records `reused_reviews`, new `model_attempts`, source summary hash, original raw
result hashes, and `previous_results` with failed-attempt usage and provenance.
`completed_review_cost_usd` remains a selected-success cost, not total experiment
spend; historical failures must also be counted when calculating total spend.

Validation covers benchmark revision, manifest/subset identity and hashes, exact
task selection, mode/model, frozen PR base/head and diff hash, saved reference
content, prompt text/hash, output schema, request hash, inference/input settings,
raw-output/findings agreement, local schema and diff anchors, candidate content and
hash, and previous failed-result provenance. Missing or inconsistent artifacts
fail closed. Unrecorded artifacts for a task marked ready require inspection; they
are not silently replayed. Keep ancestor run directories for the failed-attempt
audit trail. Always resume from the **latest** resulting directory; repeating a
command against an older parent intentionally starts another branch from that
parent's saved state and can repeat its pending work.

The output ceiling can be set using `--max-output-tokens N` on `review` or
`evaluate`, or `[review].max_output_tokens` in TOML. The default remains 4096.
It covers reasoning plus visible output; `--request-byte-budget` remains the separate,
unchanged conservative input-size check. Output-ceiling increases during resume
require `--allow-output-budget-change`. This permits only that increase, not other
configuration changes, and applies the new ceiling only to new attempts. A reused
4096-token result stays labeled 4096. The resulting mixed-ceiling development run
is not a uniform-settings final experiment; freeze settings before final A–D runs.

For the interrupted development run, the intended command (not executed during
implementation) is:

```sh
REVIEW_PROVIDER=openai REVIEW_MODEL=gpt-6-luna \
/opt/anaconda3/bin/python -m reviewer evaluate \
  --benchmark reviewbench --modes baseline \
  --benchmark-archive .cache/reviewbench/ReviewBench-968291d99e82b3e2969d7e0b4acca62585886c13.tar.gz \
  --subset configs/reviewbench-dev5.json \
  --request-byte-budget 32000 --max-output-tokens 8192 \
  --resume-from results/reviewbench-dev5-406513990eeb4bc391b587073b04e00e \
  --retry-failed --allow-output-budget-change
```

The API adapter now saves allowlisted `response_status` and
`incomplete_details.reason` in each attempt. Known reasons include
`max_output_tokens` and `content_filter`; other non-null values become `unknown`.
Missing metadata remains null. No arbitrary response detail, authorization data,
or provider error body is copied. Older artifacts remain readable and are not
retroactively assigned an incomplete reason. See the [OpenAI reasoning
documentation](https://developers.openai.com/api/docs/guides/reasoning) for why an
output limit can be reached during reasoning before visible output exists.

## Artifacts and tests

Each invocation uses a new exclusive directory:

```text
results/reviewbench-dev5-<uuid>/
    subset.json
    inputs/<task-id>.json
    candidates/<task-id>.json
    reference/<task-id>.json
    runs/<task-id>/baseline.json
    summary.json
```

Candidates are ready for upstream's normalized input loader; only successful
reviews produce candidate files. Never point that loader at the entire result
root because it recursively reads JSON. Original richer findings, raw response
text, model/settings, hashes, token usage, and timings remain in baseline runs.
Reference exports preserve the entire original JSON object semantically, with
normalized whitespace/key order; source-file and export hashes are both recorded.
Summary records identify generation successes/failures and scoring as paused.
Do not use the exit code for generation completion as evidence that scoring ran.
If stopped partway, some tasks have ready inputs but no predictions. Generated
artifacts and downloaded sources are ignored by the existing `.gitignore`.

Offline tests use synthetic miniature archives and HTTP/model mocks, with socket
connections blocked. They cover loading, the real five-task identity lock,
reconstruction and mismatch failures, output conversion, empty predictions,
original TP/FP label preservation, zero-qualifying metadata, budget/attempt guards,
usage/cost handling, fail-fast persistence, CLI integration, and label isolation.

## Upstream input validation and eventual judge workflow

The normalized candidate export was checked against ReviewBench's actual
`scripts/eval/input-validation.ts` and `scripts/lib/types.ts` at the pinned revision.
Those two unchanged upstream files are included under
`tests/fixtures/reviewbench-validation/` with their upstream MIT license and SHA-256
provenance. An offline Node 24+ test invokes the upstream loader and strict input
checks directly; it does not load or invoke the matcher, scorer, or any provider.
The synthetic five-task export passes validation, resolves all five task identities,
and preserves PR/base/head identifiers, file strings (including spaces and Unicode),
head line coordinates, prediction messages, and usage. The export allowlist excludes
gold labels, raw model responses, and benchmark metadata. No export adapter change
was needed.

ReviewBench loads candidate JSON recursively. Each file has this shape:

```json
{
  "pr": {"repo": "owner/repo", "pr_number": 1, "base": "<sha>", "head": "<sha>"},
  "agent": "context-aware-code-review/baseline",
  "findings": [{
    "producer": "context-aware-code-review/baseline",
    "file": "src/example.py",
    "start_line": 17,
    "end_line": 19,
    "message": "Title, description, evidence, and suggested fix"
  }],
  "usage": {"time_in_ms": 100, "prompt_tokens": 10,
             "completion_tokens": 20, "total_tokens": 30}
}
```

The PR key is content-derived from repository, PR number, and the first eight head
SHA characters; arbitrary JSON filenames and nested folders work. Upstream then
checks all four identity values (repo, PR number, base, head) against the golden
record. Its structural validator checks field types but does not validate that a
path exists or that coordinates are valid head lines; those checks remain the
responsibility of our baseline's existing local diff validation. Multiple candidate
files for one PR are concatenated, so our generated directory must contain exactly
one file per task to avoid accidental duplicate findings. A wholly empty candidate
corpus fails strict input validation unless `--allow-empty` is supplied. Empty
findings for an individual PR are accepted in a mixed corpus.

For the five-task development run, the eventual sequence is:

1. Generate the baseline once. Keep `candidates/` separate from `reference/` and
   `runs/`; the upstream candidate loader recursively reads every JSON file.
2. Run ReviewBench's own judge from the pinned checkout against the candidate
   directory, matching golden directory, and pinned corpus manifest. Use a unique
   output path. The relevant command form is:

   ```sh
   npm run judge -- --candidate /absolute/run/candidates \
     --golden /absolute/ReviewBench/golden \
     --manifest /absolute/ReviewBench/corpus/manifest.json \
     --provider <configured-provider> --model <judge-model> \
     --output /absolute/run/judge-results.json \
     --repo-dir /absolute/ReviewBench --concurrency 1 --limit 5 --allow-empty
   ```

   `--allow-empty` is necessary when all five reviews have zero findings. The
   candidate directory still must contain the intended five task files; before
   judging, compare task identities rather than relying on the flag to establish
   coverage. This CLI performs semantic matching and LLM classification of
   unmatched candidates, so it is a paid model operation. No command above was run.
3. ReviewBench writes the requested aggregate JSON and its companion per-finding
   details JSON (`<output>.details.json`). Aggregate output includes `macro`,
   `micro`, `pr_count`, `per_pr` (including `pr_key` and `metrics.overall`,
   `metrics.by_severity`, `metrics.by_category`), and run/corpus metadata where
   available. Details retain candidate findings, match statuses, covered golden
   indices, and classifications. These are the authoritative judge outputs; our
   project currently has no importer or experiment-summary integration for them.
4. Preserve both raw judge files beside the immutable generation result. A later
   small summary adapter can join `per_pr` to the frozen task IDs and copy upstream
   aggregate values, while keeping ReviewBench's metric names and denominators
   unchanged. It must never infer a judgment when a match/classification is absent.

The reviewer export does not have ReviewBench's native `category` or `severity`
prediction fields. They are folded into neither the native schema nor metadata;
the judge infers labels from the finding message. Upstream's official output is not
one conventional TP/FP/FN/precision/recall/F1 tuple: it reports grounded and
augmented metrics, macro/micro aggregates, severity strata, and category strata.
Keep those names and definitions intact when presenting results.

### Category and scope reporting

ReviewBench's scorer has category strata, but no scope-filter option. Its category
metrics can be reported as an unfiltered diagnostic, clearly stating that they do
not isolate `introduced-by-pr` / `exacerbated-by-pr`. Do not treat golden findings
outside our four categories as false negatives for the declared reviewer contract.

For a defensible contract-aligned result, create a separate deterministic *golden
input view* containing only original findings whose category is one of
correctness/security/reliability/performance and whose scope is
introduced-by-pr/exacerbated-by-pr. Preserve each selected finding and its original
`tp_fp` label exactly. Run the same pinned ReviewBench judge/scorer against this
filtered golden directory and the unchanged candidate files. This reuses its
matcher/scorer rather than adding ours. Report its grounded precision/recall by
category as grounded metrics over that declared reference set, not as conventional
all-prediction precision or a custom F1. Report a separate unfiltered upstream run
only as optional transparency; it requires another judge run and its own output
artifact.

Do not claim scope-filtered augmented precision or F1: the upstream scorer does not
filter unmatched candidate classifications by scope, and its exported candidate
detail does not retain enough scope information for a trustworthy post-hoc scoped
metric. If conventional in-scope precision/F1 later proves necessary, first extend
the upstream evaluation output/classification contract in a separately reviewed
methodology change, then rescore every mode consistently. Until then, show the
contract-aligned grounded metrics and the unfiltered official metrics as distinct
families, with their limitations and judge cost visible.

An empty selected reference set is not evidence that the PR is bug-free. Preserve
null/undefined denominators as ReviewBench reports them and show eligible reference
counts and task coverage beside the metrics.

```sh
python -m unittest discover -s tests -p test_benchmark.py -v
python -m unittest discover -s tests -v
```

The next decision is how to obtain trustworthy semantic matches for evaluation.
Scoring remains paused; no live five-task model run or retrieval work is authorized
by this implementation step.
