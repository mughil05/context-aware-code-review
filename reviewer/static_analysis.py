"""Run Bandit only against explicitly selected files from a frozen PR snapshot.

Signals are evidence for review, never final findings. Repository source is read
from the hash-verified archive and staged in an isolated temporary directory;
Bandit never receives a checkout or an archive-controlled path.
"""
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import time
from typing import Annotated, Literal

from pydantic import Field

from reviewer.schemas import CommitSHA, PositiveInt, RepoPath, StrictModel, Text


class BanditFinding(StrictModel):
    """Small normalized subset of one Bandit issue."""
    test_id: Text
    test_name: Text
    issue_severity: Text
    issue_confidence: Text
    issue_text: Text
    file: RepoPath
    line_number: PositiveInt
    line_range: list[PositiveInt]
    more_info: Text


class StaticAnalysisResult(StrictModel):
    """Static analysis provenance, including the exact evidence sent to the model."""
    tool: Text
    version: Text | None
    head: CommitSHA
    archive_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    analyzed_files: list[RepoPath]
    config: dict[str, object]
    timeout_seconds: PositiveInt
    status: Literal['success', 'no_findings', 'no_files', 'failed', 'timeout', 'tool_unavailable']
    latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    error: Text | None
    findings: list[BanditFinding]
    metrics: dict[str, object] | None
    signals_sent: list[dict[str, object]] = Field(default_factory=list)

    def payload(self, budget: int) -> list[dict]:
        """Return stable, concise signal records that fit the configured byte budget."""
        if type(budget) is not int or budget < 0:
            raise ValueError('signal budget must be a nonnegative integer')
        items = []
        findings = sorted(self.findings, key=lambda f: (f.file, f.line_number, f.test_id,
                                                         f.issue_severity, f.issue_confidence,
                                                         f.issue_text, f.test_name))
        for finding in findings:
            item = {'test_id': finding.test_id, 'severity': finding.issue_severity,
                    'confidence': finding.issue_confidence, 'file': finding.file,
                    'line': finding.line_number, 'message': finding.issue_text}
            candidate = items + [item]
            size = len(json.dumps({'static_analysis_signals': candidate},
                                  ensure_ascii=False, sort_keys=True).encode('utf-8'))
            if size <= budget:
                items.append(item)
        return items


def get_bandit_version() -> str | None:
    """Return the installed tool version, without leaking command output."""
    try:
        result = subprocess.run(['bandit', '--version'], capture_output=True, text=True,
                                timeout=5, check=False)
        if result.returncode == 0 and result.stdout:
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                return parts[1]
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _base_result(*, version, head, archive_sha256, analyzed_files, config,
                 timeout_seconds, status, error=None, findings=None, metrics=None,
                 latency_seconds=None):
    return StaticAnalysisResult(tool='bandit', version=version, head=head,
                                archive_sha256=archive_sha256,
                                analyzed_files=analyzed_files, config=config,
                                timeout_seconds=timeout_seconds, status=status,
                                latency_seconds=latency_seconds, error=error,
                                findings=findings or [], metrics=metrics)


def run_bandit(snapshot_directory: Path, analyzed_files: list[str], *,
               timeout_seconds: int, head: str, archive_sha256: str) -> StaticAnalysisResult:
    """Run Bandit on an isolated staging directory using explicit files (no -r).

    The caller must create this directory from verified frozen archive bytes.
    Reject links, missing files and paths escaping staging before spawning Bandit.
    """
    files = sorted(set(analyzed_files))
    config = {'format': 'json', 'recursive': False, 'severity': 'low',
              'confidence': 'low', 'signal_budget_unit': 'serialized_json_utf8_bytes',
              'selected_files': files, 'selected_file_count': len(files)}
    if not files:
        return _base_result(version=None, head=head, archive_sha256=archive_sha256,
                            analyzed_files=[], config=config, timeout_seconds=timeout_seconds,
                            status='no_files')
    version = get_bandit_version()
    if version is None:
        return _base_result(version=None, head=head, archive_sha256=archive_sha256,
                            analyzed_files=[], config=config, timeout_seconds=timeout_seconds,
                            status='tool_unavailable', error='bandit command not found or version unavailable')

    try:
        root = Path(snapshot_directory).resolve(strict=True)
        if not root.is_dir():
            raise ValueError
    except (OSError, ValueError):
        return _base_result(version=version, head=head, archive_sha256=archive_sha256,
                            analyzed_files=[], config=config, timeout_seconds=timeout_seconds,
                            status='failed', error='static analysis staging directory is unavailable')
    validated = []
    for name in files:
        pure = PurePosixPath(name)
        if (pure.is_absolute() or not name or any(part in ('', '.', '..') for part in pure.parts)
                or '\\' in name):
            return _base_result(version=version, head=head, archive_sha256=archive_sha256,
                                analyzed_files=[], config=config, timeout_seconds=timeout_seconds,
                                status='failed', error='unsafe static analysis path')
        target = root.joinpath(*pure.parts)
        try:
            ancestors = [root.joinpath(*pure.parts[:index]) for index in range(1, len(pure.parts))]
            if (any(parent.is_symlink() for parent in ancestors) or target.is_symlink()
                    or not target.is_file() or not target.resolve(strict=True).is_relative_to(root)):
                raise ValueError
        except (OSError, ValueError):
            return _base_result(version=version, head=head, archive_sha256=archive_sha256,
                                analyzed_files=[], config=config, timeout_seconds=timeout_seconds,
                                status='failed', error='static analysis input is missing or outside frozen staging')
        validated.append(target)

    command = ['bandit', '-f', 'json', '--severity-level', 'low',
               '--confidence-level', 'low'] + [str(path) for path in validated]
    config['invocation'] = ['bandit', '-f', 'json', '--severity-level', 'low',
                            '--confidence-level', 'low', '<explicit-files>']
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                timeout=timeout_seconds, check=False)
        latency = max(0.0, time.perf_counter() - started)
    except subprocess.TimeoutExpired:
        elapsed = max(0.0, time.perf_counter() - started)
        config['latency_seconds'] = elapsed
        return _base_result(version=version, head=head, archive_sha256=archive_sha256,
                            analyzed_files=files, config=config, timeout_seconds=timeout_seconds,
                            status='timeout', error=f'bandit execution exceeded {timeout_seconds}s timeout',
                            latency_seconds=elapsed)
    except (OSError, ValueError) as error:
        elapsed = max(0.0, time.perf_counter() - started)
        config['latency_seconds'] = elapsed
        return _base_result(version=version, head=head, archive_sha256=archive_sha256,
                            analyzed_files=files, config=config, timeout_seconds=timeout_seconds,
                            status='failed', error=f'bandit execution failed: {type(error).__name__}',
                            latency_seconds=elapsed)

    config['latency_seconds'] = latency
    stderr = result.stderr if isinstance(result.stderr, str) else ''
    stderr_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    config['stderr_line_count'] = len(stderr_lines)
    config['stderr_warning_count'] = sum(line.lower().startswith('warning:') for line in stderr_lines)
    config['stderr_error_count'] = sum(
        line.lower().startswith('error:') or ': error:' in line.lower() for line in stderr_lines)
    config['return_code'] = result.returncode
    if result.returncode not in (0, 1):
        return _base_result(version=version, head=head, archive_sha256=archive_sha256,
                            analyzed_files=files, config=config, timeout_seconds=timeout_seconds,
                            status='failed', error=f'bandit exited with status {result.returncode} before producing a report',
                            latency_seconds=latency)
    try:
        data = json.loads(result.stdout)
        if (not isinstance(data, dict) or not isinstance(data.get('results'), list)
                or not isinstance(data.get('errors', []), list)):
            raise ValueError
    except (ValueError, TypeError):
        return _base_result(version=version, head=head, archive_sha256=archive_sha256,
                            analyzed_files=files, config=config, timeout_seconds=timeout_seconds,
                            status='failed', error='bandit output was not valid expected JSON',
                            latency_seconds=latency)

    config['reported_error_count'] = len(data.get('errors', []))
    # Bandit returns 1 when findings pass its selected thresholds; only 0/1 are
    # normal report statuses. The JSON error list and higher return codes fail closed.
    if data.get('errors'):
        return _base_result(version=version, head=head, archive_sha256=archive_sha256,
                            analyzed_files=files, config=config, timeout_seconds=timeout_seconds,
                            status='failed', error='bandit reported an execution or input error')
    if result.returncode == 1 and not data['results']:
        return _base_result(version=version, head=head, archive_sha256=archive_sha256,
                            analyzed_files=files, config=config, timeout_seconds=timeout_seconds,
                            status='failed', error='bandit exit status disagreed with its JSON findings',
                            latency_seconds=latency)

    allowed = {str(path.resolve()): path.relative_to(root).as_posix() for path in validated}
    findings = []
    try:
        for item in data['results']:
            if not isinstance(item, dict):
                raise ValueError
            raw_path = item.get('filename')
            abs_path = str(Path(raw_path).resolve()) if isinstance(raw_path, str) else ''
            # Do not silently accept an analyzer report for an unrequested file.
            if abs_path not in allowed:
                raise ValueError
            findings.append(BanditFinding(
                test_id=item['test_id'], test_name=item['test_name'],
                issue_severity=item['issue_severity'], issue_confidence=item['issue_confidence'],
                issue_text=item['issue_text'], file=allowed[abs_path],
                line_number=item['line_number'], line_range=item['line_range'],
                more_info=item['more_info']))
    except (KeyError, TypeError, ValueError):
        return _base_result(version=version, head=head, archive_sha256=archive_sha256,
                            analyzed_files=files, config=config, timeout_seconds=timeout_seconds,
                            status='failed', error='bandit returned malformed or out-of-scope findings')

    metrics = data.get('metrics') if isinstance(data.get('metrics'), dict) else None
    return _base_result(version=version, head=head, archive_sha256=archive_sha256,
                        analyzed_files=files, config=config, timeout_seconds=timeout_seconds,
                        status='success' if findings else 'no_findings', findings=findings, metrics=metrics,
                        latency_seconds=latency)


def analyze_snapshot(review_input, snapshot, context, *, timeout_seconds: int, budget: int):
    """Stage only eligible selected Python source from the verified frozen archive."""
    from reviewer.diff import parse_diff
    from reviewer.retrieval import eligible, snapshot_sources

    sources = snapshot_sources(snapshot, review_input.pr.head)
    parsed = parse_diff(review_input.diff)
    changed_py = {f.new_path for f in parsed if f.new_path and f.new_path.endswith('.py')}
    context_py = ({chunk.file for chunk in context.chunks if chunk.selected and chunk.file.endswith('.py')}
                  if context else set())
    selected = sorted((changed_py | context_py) & sources.keys())
    selected = [path for path in selected if eligible(path)]
    staging = tempfile.TemporaryDirectory(prefix='reviewer-bandit-')
    try:
        root = Path(staging.name).resolve()
        for path in selected:
            target = root.joinpath(*PurePosixPath(path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            # New files in an owned temp root; never follow an archive-provided link.
            with target.open('xb') as stream:
                stream.write(sources[path].encode('utf-8'))
        result = run_bandit(root, selected, timeout_seconds=timeout_seconds,
                            head=review_input.pr.head, archive_sha256=snapshot.archive_sha256)
    finally:
        staging.cleanup()
    config = dict(result.config)
    config['signal_token_budget'] = budget
    return result.model_copy(update={'config': config})
