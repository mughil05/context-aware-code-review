"""Pinned ReviewBench data, frozen inputs, and normalized prediction export.

No judging or model calls. Benchmark descriptions never enter ReviewInput.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tarfile
from typing import Annotated, Literal

from pydantic import Field, model_validator

from reviewer.schemas import BenchmarkAggregateMismatch, CommitSHA, PRMetadata, StrictModel, Text
from reviewer.final_exclusions import (validate_lock_exclusions, validate_final_lock_budget,
                                       is_final20, FINAL_SUBSET_SHA256)

REVISION = '968291d99e82b3e2969d7e0b4acca62585886c13'
MANIFEST_SHA256 = 'e74656b8c1c4bbab5ce85f1fcb742a7aaad550a4d7bd46e1f83a0746807bfa02'
CATEGORIES = {'correctness', 'security', 'reliability', 'performance'}
SCOPES = {'introduced-by-pr', 'exacerbated-by-pr'}
Digest = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
Count = Annotated[int, Field(ge=0)]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(raw: bytes | str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON number')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def task_key(pr: PRMetadata) -> str:
    return pr.repo.removeprefix('https://github.com/').replace('/', '_') + f'_{pr.pr_number}-{pr.head[:8]}'


class Task(StrictModel):
    task_id: Text
    pr: PRMetadata
    language: Literal['Python']
    files_changed: Annotated[int, Field(ge=1, lt=300)]
    lines_added: Count
    lines_removed: Count
    golden_sha256: Digest
    # Final manifests are initially generated from the corpus metadata. The
    # no-model preflight fills this lock after reconstructing the frozen diff.
    diff_sha256: Digest | None = None
    source_repository: Text

    @model_validator(mode='after')
    def identity(self):
        nwo = self.pr.repo.removeprefix('https://github.com/')
        if self.task_id != task_key(self.pr):
            raise ValueError('task ID and PR identity disagree')
        if self.source_repository not in (nwo, 'review-bench/' + nwo.replace('/', '_')):
            raise ValueError('source must be original repository or official mirror')
        return self


class Subset(StrictModel):
    schema_version: Literal[1]
    benchmark_revision: Literal[REVISION]
    manifest_sha256: Digest
    purpose: Literal['development-only', 'final-evaluation']
    selection_rule: Text | None = None
    tasks: Annotated[list[Task], Field(min_length=1)]

    @model_validator(mode='after')
    def unique(self):
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError('subset task IDs must be unique')
        if self.purpose == 'final-evaluation' and not self.selection_rule:
            raise ValueError('final evaluation subset requires a selection rule')
        return self


@dataclass(frozen=True)
class Benchmark:
    subset: Subset
    subset_sha256: str
    archive_sha256: str
    diff_lock_sha256: str | None
    # Kept separate from tasks/ReviewInput, only export/scoring may use these.
    golden: dict[str, dict]
    lock_provenance: dict | None = None


def load_benchmark(archive: str | Path, subset_path: str | Path, *,
                   expected_manifest_sha256: str = MANIFEST_SHA256,
                   allow_unpinned_diffs: bool = False) -> Benchmark:
    raw_subset = Path(subset_path).read_bytes()
    archive_path = Path(archive)
    if archive_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('benchmark archive too large')
    raw_archive = archive_path.read_bytes()
    subset = Subset.model_validate(read_json(raw_subset))
    lock_sha256 = None
    lock_data = None
    lock_path = Path(subset_path).with_suffix('.lock.json')
    if subset.purpose == 'final-evaluation' and lock_path.exists():
        lock_raw = lock_path.read_bytes()
        lock = read_json(lock_raw)
        if (not isinstance(lock, dict) or lock.get('schema_version') != 1
                or lock.get('benchmark_revision') != REVISION
                or lock.get('subset_sha256') != digest(raw_subset)
                or not isinstance(lock.get('tasks'), list)):
            raise ValueError('final subset diff lock mismatch')
        excluded_ids = validate_lock_exclusions(subset, lock.get('pre_execution_exclusions'))
        if is_final20(subset.tasks, subset.purpose):
            if digest(raw_subset) != FINAL_SUBSET_SHA256:
                raise ValueError('final20 subset differs from the frozen predeclared sample')
            validate_final_lock_budget(lock)
            if (lock.get('benchmark_archive_sha256') != digest(raw_archive)
                    or lock.get('benchmark_manifest_sha256') != subset.manifest_sha256):
                raise ValueError('final lock benchmark artifact provenance mismatch')
        by_id = {entry.get('task_id'): entry for entry in lock['tasks'] if isinstance(entry, dict)}
        expected_locked_ids = {task.task_id for task in subset.tasks} - set(excluded_ids)
        if len(by_id) != len(lock['tasks']) or set(by_id) != expected_locked_ids:
            raise ValueError('final subset diff lock task mismatch')
        locked_tasks = []
        for task in subset.tasks:
            if task.task_id in excluded_ids:
                locked_tasks.append(task)
                continue
            entry = by_id[task.task_id]
            if entry.get('base_sha') != task.pr.base or entry.get('head_sha') != task.pr.head:
                raise ValueError('final subset diff lock frozen identity mismatch')
            locked_diff = entry.get('diff_sha256')
            if not isinstance(locked_diff, str) or not re.fullmatch(r'[0-9a-f]{64}', locked_diff):
                raise ValueError('final subset diff lock contains an invalid hash')
            if (is_final20(subset.tasks, subset.purpose)
                    and (not isinstance(entry.get('head_archive_sha256'), str)
                         or not re.fullmatch(r'[0-9a-f]{64}', entry['head_archive_sha256']))):
                raise ValueError('final subset lock contains an invalid frozen HEAD archive hash')
            if task.diff_sha256 is not None and task.diff_sha256 != locked_diff:
                raise ValueError('final subset manifest and diff lock disagree')
            warnings = entry.get('acquisition_warnings', [])
            if not isinstance(warnings, list):
                raise ValueError('final subset lock acquisition warnings are malformed')
            for raw_warning in warnings:
                warning = BenchmarkAggregateMismatch.model_validate(raw_warning)
                if (warning.base_sha != task.pr.base or warning.head_sha != task.pr.head
                        or warning.expected_file_count != task.files_changed
                        or warning.expected_additions != task.lines_added
                        or warning.expected_deletions != task.lines_removed):
                    raise ValueError('final subset lock acquisition warning identity mismatch')
            locked_tasks.append(task.model_copy(update={'diff_sha256': locked_diff}))
        subset = subset.model_copy(update={'tasks': locked_tasks})
        lock_sha256 = digest(lock_raw)
        lock_data = lock
    elif is_final20(subset.tasks, subset.purpose) and digest(raw_subset) != FINAL_SUBSET_SHA256:
        raise ValueError('final20 subset differs from the frozen predeclared sample')
    if subset.manifest_sha256 != expected_manifest_sha256:
        raise ValueError('subset manifest hash differs from pinned manifest')
    locked_excluded = set()
    if is_final20(subset.tasks, subset.purpose) and lock_path.exists():
        locked_excluded = set(validate_lock_exclusions(subset, lock.get('pre_execution_exclusions')))
    if not allow_unpinned_diffs and any(task.diff_sha256 is None and task.task_id not in locked_excluded
                                         for task in subset.tasks):
        raise ValueError('subset contains diffs not frozen by preflight')
    required = {'corpus/manifest.json'} | {f'golden/{t.task_id}.json' for t in subset.tasks}
    contents = {}
    # Read selected regular members only. No archive extraction or executed code.
    total = 0
    try:
        with tarfile.open(archive, 'r|gz') as tar:
            for number, member in enumerate(tar):
                total += member.size
                if member.size < 0 or number > 100000 or total > 256 * 1024 * 1024:
                    raise ValueError('benchmark archive exceeds limits')
                prefix = 'ReviewBench-' + REVISION + '/'
                if member.name.rstrip('/') == prefix.rstrip('/') and member.isdir():
                    continue
                if not member.name.startswith(prefix):
                    raise ValueError('unexpected archive root')
                relative = member.name[len(prefix):]
                if relative not in required:
                    continue
                if not member.isfile() or member.size > 8 * 1024 * 1024 or relative in contents:
                    raise ValueError('unsafe, duplicate, or oversized benchmark member')
                contents[relative] = tar.extractfile(member).read()
    except (tarfile.TarError, EOFError, OSError) as error:
        raise ValueError('unreadable benchmark archive') from error
    if set(contents) != required:
        raise ValueError('missing benchmark manifest or selected golden file')
    manifest_raw = contents['corpus/manifest.json']
    if digest(manifest_raw) != subset.manifest_sha256:
        raise ValueError('benchmark manifest hash mismatch')
    manifest = read_json(manifest_raw)
    if not isinstance(manifest, list):
        raise ValueError('manifest must be an array')
    by_key = {}
    for row in manifest:
        if not isinstance(row, dict):
            raise ValueError('manifest task must be an object')
        pr = PRMetadata.model_validate({k: row[k] for k in ('repo', 'pr_number', 'base', 'head')})
        key = task_key(pr)
        if key in by_key:
            raise ValueError('duplicate manifest task')
        by_key[key] = row
    golden = {}
    for task in subset.tasks:
        row = by_key.get(task.task_id)
        expected = task.pr.model_dump() | {k: getattr(task, k) for k in ('language', 'files_changed', 'lines_added', 'lines_removed')}
        if row is None or any(type(row.get(k)) is not type(v) or row[k] != v for k, v in expected.items()):
            raise ValueError('selected task metadata differs from manifest')
        raw = contents[f'golden/{task.task_id}.json']
        if digest(raw) != task.golden_sha256:
            raise ValueError('golden file hash mismatch')
        g = read_json(raw)
        if not isinstance(g, dict) or g.get('pr_key') != task.task_id or g.get('pr') != task.pr.model_dump() or not isinstance(g.get('findings'), list):
            raise ValueError('golden identity or findings malformed')
        for f in g['findings']:
            validate_golden_finding(f)
        golden[task.task_id] = g
    return Benchmark(subset, digest(raw_subset), digest(raw_archive), lock_sha256, golden, lock_data)


def validate_golden_finding(f):
    from reviewer.schemas import repository_path
    if not isinstance(f, dict):
        raise ValueError('golden finding must be an object')
    repository_path(f['file'])
    if (type(f['start_line']) is not int or type(f['end_line']) is not int
            or f['start_line'] < 1 or f['end_line'] < f['start_line']):
        raise ValueError('invalid golden coordinates')
    if f['tp_fp'] not in ('tp', 'fp') or f['severity'] not in ('low', 'medium', 'high'):
        raise ValueError('invalid golden label')
    for key in ('message', 'category', 'scope'):
        if not isinstance(f[key], str) or not f[key].strip():
            raise ValueError('missing golden text')


def qualifying_count(golden):
    return sum(f['tp_fp'] == 'tp' and f['category'] in CATEGORIES and f['scope'] in SCOPES
               for f in golden['findings'])


def reconstruct(task: Task, client):
    """Only immutable metadata is passed to the existing HTTP/parser foundation."""
    return client.frozen_input(task.pr, source_repository=task.source_repository,
                               expected_counts=(task.files_changed, task.lines_added, task.lines_removed),
                               expected_diff_sha256=task.diff_sha256,
                               allow_benchmark_aggregate_mismatch=True)


def final_selection(manifest_rows: list[dict], dev_task_ids: set[str]) -> list[dict]:
    """Mechanical final subset rule; corpus order is retained for stable output."""
    selected = []
    for row in manifest_rows:
        if (row.get('language') != 'Python'
                or type(row.get('lines_added')) is not int
                or type(row.get('lines_removed')) is not int
                or row['lines_added'] + row['lines_removed'] > 1000):
            continue
        pr = PRMetadata.model_validate({k: row[k] for k in ('repo', 'pr_number', 'base', 'head')})
        task_id = task_key(pr)
        if task_id not in dev_task_ids:
            selected.append(row)
    return selected


def generate_final_subset(archive: str | Path, dev_subset_path: str | Path) -> bytes:
    """Render a manifest-ordered final selection from only the pinned corpus."""
    dev = Subset.model_validate(read_json(Path(dev_subset_path).read_bytes()))
    rows = _read_pinned_corpus_manifest(archive)
    selected = final_selection(rows, {task.task_id for task in dev.tasks})
    if len(selected) != 20:
        raise ValueError(f'mechanical final selection produced {len(selected)} tasks; expected 20')
    tasks = []
    with tarfile.open(archive, 'r:gz') as tar:
        names = tar.getnames()
        for row in selected:
            pr = PRMetadata.model_validate({k: row[k] for k in ('repo', 'pr_number', 'base', 'head')})
            task_id = task_key(pr)
            member = f'ReviewBench-{REVISION}/golden/{task_id}.json'
            if names.count(member) != 1:
                raise ValueError('duplicate golden archive member')
            raw = tar.extractfile(member).read()
            tasks.append(Task(task_id=task_id, pr=pr, language=row['language'],
                              files_changed=row['files_changed'], lines_added=row['lines_added'],
                              lines_removed=row['lines_removed'], golden_sha256=digest(raw),
                              diff_sha256=None,
                              source_repository='review-bench/' + pr.repo.removeprefix('https://github.com/').replace('/', '_')))
    subset = Subset(schema_version=1, benchmark_revision=REVISION, manifest_sha256=MANIFEST_SHA256,
                    purpose='final-evaluation',
                    selection_rule='Pinned corpus order; language=Python; lines_added+lines_removed<=1000; exclude every task ID in configs/reviewbench-dev5.json.',
                    tasks=tasks)
    return (json.dumps(subset.model_dump(), indent=2, sort_keys=True) + '\n').encode()


def _read_pinned_corpus_manifest(archive: str | Path) -> list[dict]:
    wanted = f'ReviewBench-{REVISION}/corpus/manifest.json'
    with tarfile.open(archive, 'r:gz') as tar:
        matches = [m for m in tar.getmembers() if m.name == wanted]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError('pinned corpus manifest missing or duplicated')
        raw = tar.extractfile(matches[0]).read()
    if digest(raw) != MANIFEST_SHA256:
        raise ValueError('benchmark manifest hash mismatch')
    rows = read_json(raw)
    if not isinstance(rows, list):
        raise ValueError('manifest must be an array')
    return rows


def export_predictions(run):
    """ReviewBench docs/JUDGING_INPUT.md; never export failed runs as empty reviews."""
    if run.result.status != 'success':
        raise ValueError('cannot export failed review')
    producer = 'context-aware-code-review/' + run.result.mode
    findings = []
    for f in run.result.findings:
        findings.append({'producer': producer, 'file': f.file, 'start_line': f.start_line,
                         'end_line': f.end_line, 'message': f'{f.title}\n\n{f.description}\n\nEvidence:\n'
                         + '\n'.join('- ' + x for x in f.evidence) + '\n\nSuggested fix:\n' + f.suggested_fix})
    usage = {'time_in_ms': run.total_latency_seconds * 1000}
    if run.attempts and all(a.usage is not None for a in run.attempts):
        usage.update(prompt_tokens=sum(a.usage.input_tokens for a in run.attempts),
                     completion_tokens=sum(a.usage.output_tokens for a in run.attempts),
                     total_tokens=sum(a.usage.total_tokens for a in run.attempts))
        if all(a.usage.cached_input_tokens is not None for a in run.attempts):
            usage['cached_tokens'] = sum(a.usage.cached_input_tokens for a in run.attempts)
    return {'pr': run.result.pr.model_dump(), 'agent': producer, 'findings': findings, 'usage': usage}
