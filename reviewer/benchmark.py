"""Pinned ReviewBench data, frozen inputs, and normalized prediction export.

No judging or model calls. Benchmark descriptions never enter ReviewInput.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Annotated, Literal

from pydantic import Field, model_validator

from reviewer.schemas import CommitSHA, PRMetadata, StrictModel, Text

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
    diff_sha256: Digest
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
    purpose: Literal['development-only']
    tasks: Annotated[list[Task], Field(min_length=5, max_length=5)]

    @model_validator(mode='after')
    def unique(self):
        if len({t.task_id for t in self.tasks}) != 5:
            raise ValueError('subset must contain five unique tasks')
        return self


@dataclass(frozen=True)
class Benchmark:
    subset: Subset
    subset_sha256: str
    # Kept separate from tasks/ReviewInput, only export/scoring may use these.
    golden: dict[str, dict]


def load_benchmark(archive: str | Path, subset_path: str | Path, *,
                   expected_manifest_sha256: str = MANIFEST_SHA256) -> Benchmark:
    raw_subset = Path(subset_path).read_bytes()
    subset = Subset.model_validate(read_json(raw_subset))
    if subset.manifest_sha256 != expected_manifest_sha256:
        raise ValueError('subset manifest hash differs from pinned manifest')
    required = {'corpus/manifest.json'} | {f'golden/{t.task_id}.json' for t in subset.tasks}
    contents = {}
    # Read selected regular members only. No archive extraction or executed code.
    if Path(archive).stat().st_size > 64 * 1024 * 1024:
        raise ValueError('benchmark archive too large')
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
    return Benchmark(subset, digest(raw_subset), golden)


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
                               expected_diff_sha256=task.diff_sha256)


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
