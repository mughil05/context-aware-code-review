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
