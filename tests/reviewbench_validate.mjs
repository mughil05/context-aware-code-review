// Offline bridge to the unmodified pinned ReviewBench validator, never its judge.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { registerHooks } from 'node:module';

const root = new URL('./fixtures/reviewbench-validation/', import.meta.url);
const validator = new URL('scripts/eval/input-validation.ts', root);
const types = new URL('scripts/lib/types.ts', root);
const provenance = JSON.parse(readFileSync(new URL('provenance.json', root), 'utf8'));
for (const [path, expected] of Object.entries(provenance.sha256)) {
  const actual = createHash('sha256').update(readFileSync(new URL(path, root))).digest('hex');
  if (actual !== expected) throw new Error(`Pinned upstream source changed: ${path}`);
}
// Upstream uses NodeNext .js specifiers for .ts sources. Resolve only that one
// runtime dependency; Node 24 strips types without changing validator logic.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL === validator.href || context.parentURL === types.href) {
      if (context.parentURL === validator.href && specifier === '../lib/types.js') {
        return { url: types.href, shortCircuit: true };
      }
      if (!['fs', 'path'].includes(specifier)) {
        throw new Error(`Unexpected upstream runtime import: ${specifier}`);
      }
    }
    return nextResolve(specifier, context);
  },
});
globalThis.fetch = () => { throw new Error('Network prohibited in offline validation'); };

const { loadCandidateFindings, buildEvalInputSummary, strictInputProblems,
        candidateIdentityProblem } = await import(validator.href);
const [candidatePath, subsetPath, emptyFlag] = process.argv.slice(2);
const subset = JSON.parse(readFileSync(subsetPath, 'utf8'));
const expected = new Map(subset.tasks.map(task => [task.task_id, task.pr]));
const loaded = loadCandidateFindings(candidatePath);
const summary = buildEvalInputSummary(loaded, new Set(expected.keys()), Infinity);
const problems = strictInputProblems(summary, loaded.stats,
  { strict: true, allowEmpty: emptyFlag === '--allow-empty' });
for (const [key, candidate] of loaded.byPR) {
  if (expected.has(key)) {
    const issue = candidateIdentityProblem(candidate.pr, expected.get(key));
    if (issue) problems.push(`${key}: ${issue}`);
  }
}
process.stdout.write(JSON.stringify({ stats: loaded.stats, summary, problems,
  loaded: Object.fromEntries(loaded.byPR) }));
process.exitCode = problems.length ? 1 : 0;
