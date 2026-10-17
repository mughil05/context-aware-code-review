# Phase 6: frozen repository context

Baseline A and retrieval B use the same OpenAI adapter, model/settings, prompt,
FindingBatch schema, parser, and local HEAD-line validation. A sends only `diff`.
B adds `repository_context`, a list of source excerpts with file, inclusive HEAD
line range, and content. Scores, benchmark metadata, and selection labels are not
model inputs. No real PR or benchmark model run was executed during implementation.

## Review contract v2

`reviewer/prompts/review.txt` now explicitly prohibits findings outside files and
HEAD-side line ranges represented in the PR diff. Context may support evidence but
cannot extend allowed anchors. Every represented line in a proposed range must
still pass the unchanged local validator. Deletion-related findings may anchor to
surviving context in a changed hunk. Both A/B use this identical contract.
The prompt hash is computed from the exact version-controlled text and saved in
each result. Historical Phase 5 results retain their old prompt/hash; do not resume
or combine them as if generated with v2. Phase 5 live development is closed.

## Retrieval implementation

The retriever reads the frozen HEAD archive returned by existing acquisition. It
checks the exact HEAD identity and archive SHA-256, validates archive member paths
and types, and indexes only eligible Python source. It reads archive bytes instead
of a mutable extracted directory. Live review enables the existing snapshot option
(which currently downloads base/diff-base/head snapshots). Benchmark B requests
only the locked task's HEAD from its configured mirror, using the same archive
handling. There is no current-branch fallback and repository code is never executed.

Top-level functions/classes, including decorators, form code units. Module-level
gaps form additional units. Units larger than `chunk_target_lines` (default 80)
are split into line windows. Syntax errors or unsupported syntax fall back to line
windows. Every HEAD line represented in the diff is removed from candidate chunks;
remaining contiguous segments keep their original coordinates. Referenced imports
are derived from changed eligible files; identifiers come from diff text, paths,
and import names. Relative imports are resolved approximately against module paths,
with conventional `src/` layout support.

Structural and lexical signals are added using fixed integer weights:

| Signal | Score |
| --- | ---: |
| Imported module or descendant | 100 |
| Chunk defines a referenced function/class | 60 |
| Same changed file, outside diff | 30 |
| Same directory as a changed file | 20 |
| Distinct shared code/name tokens | 2 each, capped at 40 |
| Distinct shared path tokens | 5 each, capped at 20 |

Same-file and nearby-module scores are mutually exclusive. Keywords and common
boilerplate tokens are removed. Candidates sort by descending score, then path,
start line, end line, and content hash. Zero-score chunks are not selected. Larger
chunks that do not fit are skipped so later smaller chunks can fit. This is hybrid
**structural/lexical** ranking, not lexical/semantic hybrid retrieval.

**Semantic embeddings are deferred.** The project has no embedding adapter or
local embedding model. Adding a provider, vector store/cache, and paid indexing
would expand this phase unnecessarily. No embedding model is required and no
semantic scores, vectors, or cosine similarity are fabricated. The retriever's
`retrieve(...) -> RepositoryContext` boundary allows a later separately evaluated
semantic implementation without changing the reviewer or anchor validation.

## Budgets, cache, and exclusions

Configure `[retrieval]` in `configs/experiment.toml`:

- `context_token_budget = 6000`: conservative UTF-8 byte estimate of serialized
  context, including the second JSON-encoding layer used in the model request.
  This is **not measured tokenizer usage**. `--context-token-budget` overrides it.
- `max_chunks = 20` and `chunk_target_lines = 80`.
- `ranking = "structural_lexical_v1"`.

The available context budget is also limited by space remaining under the existing
whole-request input limit, reserving JSON overhead. The exact same adapter budget
check runs on the final payload before any model call. Diff bytes are never removed
or truncated to make room. Empty context is valid. Output-token settings remain
independent of both input and context budgets.

`.cache/retrieval/` stores AST ranges/import metadata keyed by source content,
algorithm version, and chunk size. Unchanged files reuse parsing work across
snapshots. Changed content or chunk size produces a new key. Cache entries carry a
checksum; malformed entries are ignored and recomputed. Source text is always
sliced from the verified archive, never supplied by a cache entry. No cache hit
state affects ranking, no embeddings are cached, and there are no background calls.

Only UTF-8 `.py` files up to 256 KiB are eligible. Binary/NUL data, dot directories,
vendor/dependency/build/generated directories, benchmark labels/reference/results/
judgment/scoring paths, obvious credential/configuration filenames, generated-code
headers, private keys, and recognizable literal credentials are excluded. Archive
links and special members are rejected. README, diffs, JSON labels, and non-source
files are not indexed. These are conservative exclusions, not a complete secret
scanner or a guarantee against disguised data in ordinary source code.

Each raw review result records HEAD/archive identity, retrieval config, budget and
used estimate, plus ranked chunk paths/ranges/hashes, scores, signals, and selected
status. Only selected chunks retain source text; unselected text is omitted.
Retrieval timing is recorded separately. Baseline results have no context.

## CLI and result layout

A future single-PR smoke test, requiring explicit authorization before execution:

```sh
REVIEW_PROVIDER=openai REVIEW_MODEL=gpt-6-luna \
/opt/anaconda3/bin/python -m reviewer review \
  --repo pallets/itsdangerous --pr 378 --mode retrieval \
  --input-token-limit 32000 --max-output-tokens 8192 \
  --context-token-budget 6000
```

Benchmark generation accepts `--modes baseline,retrieval` or
`--modes baseline retrieval`, with the existing benchmark archive/subset arguments.
All selected task inputs, snapshots, contexts, and input budgets preflight before
any model request. Each task/mode uses the shared review pipeline, with one attempt
and fail-fast behavior. Scoring is still paused; this phase did not run ReviewBench.

```text
results/reviewbench-dev5-<uuid>/
    subset.json
    inputs/<task>.json
    reference/<task>.json
    modes/baseline/runs/<task>/baseline.json
    modes/baseline/candidates/<task>.json
    modes/retrieval/runs/<task>/retrieval.json
    modes/retrieval/candidates/<task>.json
    summary.json
```

Each mode's candidate directory is independently ReviewBench compatible; never
pass the full run root to its recursive loader. Baseline-only runs retain their
previous layout. Existing baseline resume remains available for matching v2 runs;
retrieval/multi-mode resume is explicitly unsupported in this phase.

## Limitations and tests

No embeddings, AST call graph, type inference, cross-language retrieval, static
analysis, verification, or semantic judging are implemented. Dynamic imports,
re-exports, arbitrary package layouts, and aliased call resolution are approximate.
The fixed weights are an initial method, not benchmark-tuned or proven better.
Large definitions may be split mid-body. Security exclusions can omit useful code.
The snapshot archive size/member limits from acquisition still apply.

Offline tests cover imports, symbols, lexical and combined ranking, deterministic
ordering, AST/cache behavior, exclusions, budgets, archive/head hashes, CLI paths,
A/B input isolation and shared instructions/schema, synthetic benchmark label
isolation, and rejection of findings anchored only in supporting context.
No runtime dependency was added; the existing Pydantic dependency is sufficient.
