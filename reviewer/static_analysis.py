"""Isolated Bandit execution with fixed configuration and JSON output.

Signals are evidence, not final findings. Record failures and base/head provenance.
Never install or execute dependencies from a target repository.
"""
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Annotated

from pydantic import Field

from reviewer.schemas import CommitSHA, PositiveInt, RepoPath, StrictModel, Text


class BanditFinding(StrictModel):
    """Internal schema for one Bandit issue; not exposed directly to the model."""
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
    """Provenance envelope for static analysis execution."""
    tool: Text
    version: Text | None
    head: CommitSHA
    archive_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    analyzed_files: list[RepoPath]
    config: dict[str, object]
    timeout_seconds: PositiveInt
    status: Text  # 'success', 'no_findings', 'failed', 'timeout', 'tool_unavailable'
    error: Text | None
    findings: list[BanditFinding]
    metrics: dict[str, object] | None  # Bandit's metrics dict if available

    def payload(self, budget: int) -> list[dict]:
        """Return findings serialized for model input within budget."""
        items = []
        for finding in self.findings:
            item = {
                'file': finding.file,
                'line': finding.line_number,
                'severity': finding.issue_severity,
                'confidence': finding.issue_confidence,
                'issue': finding.issue_text,
                'test_id': finding.test_id,
            }
            serialized = json.dumps({'static_analysis_signals': items + [item]}, ensure_ascii=False)
            if len(serialized.encode('utf-8')) > budget:
                break
            items.append(item)
        return items


def get_bandit_version() -> str | None:
    """Attempt to determine Bandit version; return None if unavailable."""
    try:
        result = subprocess.run(
            ['bandit', '--version'],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        if result.returncode == 0 and result.stdout:
            # Extract version from output like "bandit 1.7.5"
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                return parts[1]
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def run_bandit(snapshot_directory: Path, analyzed_files: list[str], *,
               timeout_seconds: int, head: str, archive_sha256: str) -> StaticAnalysisResult:
    """Execute Bandit on selected files from the frozen HEAD snapshot.

    Never analyze live working tree, benchmark artifacts, or credential files.
    """
    version = get_bandit_version()
    config = {'format': 'json', 'analyzed_file_count': len(analyzed_files)}

    if version is None:
        return StaticAnalysisResult(
            tool='bandit',
            version=None,
            head=head,
            archive_sha256=archive_sha256,
            analyzed_files=[],
            config=config,
            timeout_seconds=timeout_seconds,
            status='tool_unavailable',
            error='bandit command not found or version unavailable',
            findings=[],
            metrics=None
        )

    if not analyzed_files:
        return StaticAnalysisResult(
            tool='bandit',
            version=version,
            head=head,
            archive_sha256=archive_sha256,
            analyzed_files=[],
            config=config,
            timeout_seconds=timeout_seconds,
            status='no_findings',
            error=None,
            findings=[],
            metrics=None
        )

    # Create temporary file with paths to analyze
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as targets:
        for path in analyzed_files:
            targets.write(str(snapshot_directory / path) + '\n')
        targets_file = targets.name

    try:
        # Run Bandit with JSON output, reading from targets file
        result = subprocess.run(
            ['bandit', '-f', 'json', '-r'] + [str(snapshot_directory / p) for p in analyzed_files],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False  # Bandit returns non-zero when it finds issues
        )

        # Parse JSON output
        try:
            data = json.loads(result.stdout) if result.stdout else {}
        except json.JSONDecodeError:
            return StaticAnalysisResult(
                tool='bandit',
                version=version,
                head=head,
                archive_sha256=archive_sha256,
                analyzed_files=analyzed_files,
                config=config,
                timeout_seconds=timeout_seconds,
                status='failed',
                error='bandit output was not valid JSON',
                findings=[],
                metrics=None
            )

        # Extract findings
        findings = []
        wrapper_prefix = str(snapshot_directory) + '/'
        for item in data.get('results', []):
            # Convert absolute path back to repository-relative path
            abs_path = item.get('filename', '')
            if not abs_path.startswith(wrapper_prefix):
                continue
            rel_path = abs_path[len(wrapper_prefix):]

            findings.append(BanditFinding(
                test_id=item.get('test_id', 'unknown'),
                test_name=item.get('test_name', 'unknown'),
                issue_severity=item.get('issue_severity', 'UNDEFINED'),
                issue_confidence=item.get('issue_confidence', 'UNDEFINED'),
                issue_text=item.get('issue_text', ''),
                file=rel_path,
                line_number=item.get('line_number', 1),
                line_range=item.get('line_range', [item.get('line_number', 1)]),
                more_info=item.get('more_info', '')
            ))

        metrics = data.get('metrics') if 'metrics' in data else None
        status = 'success' if findings else 'no_findings'

        return StaticAnalysisResult(
            tool='bandit',
            version=version,
            head=head,
            archive_sha256=archive_sha256,
            analyzed_files=analyzed_files,
            config=config,
            timeout_seconds=timeout_seconds,
            status=status,
            error=None,
            findings=findings,
            metrics=metrics
        )

    except subprocess.TimeoutExpired:
        return StaticAnalysisResult(
            tool='bandit',
            version=version,
            head=head,
            archive_sha256=archive_sha256,
            analyzed_files=analyzed_files,
            config=config,
            timeout_seconds=timeout_seconds,
            status='timeout',
            error=f'bandit execution exceeded {timeout_seconds}s timeout',
            findings=[],
            metrics=None
        )
    except (OSError, ValueError) as error:
        return StaticAnalysisResult(
            tool='bandit',
            version=version,
            head=head,
            archive_sha256=archive_sha256,
            analyzed_files=analyzed_files,
            config=config,
            timeout_seconds=timeout_seconds,
            status='failed',
            error=f'bandit execution failed: {type(error).__name__}',
            findings=[],
            metrics=None
        )
    finally:
        try:
            Path(targets_file).unlink()
        except OSError:
            pass


def analyze_snapshot(review_input, snapshot, context, *, timeout_seconds: int, budget: int):
    """Run Bandit on changed Python files and selected context files.

    Never analyze benchmark artifacts, credentials, or files outside the frozen snapshot.
    """
    from reviewer.diff import parse_diff
    from reviewer.retrieval import eligible

    parsed = parse_diff(review_input.diff)

    # Changed Python files
    changed_py = {f.new_path for f in parsed if f.new_path and f.new_path.endswith('.py')}

    # Context Python files (already filtered by retrieval exclusions)
    context_py = set()
    if context:
        context_py = {chunk.file for chunk in context.chunks if chunk.selected and chunk.file.endswith('.py')}

    # Combine and ensure all files pass eligibility check
    analyzed = sorted({f for f in (changed_py | context_py) if eligible(f)})

    result = run_bandit(
        snapshot.directory,
        analyzed,
        timeout_seconds=timeout_seconds,
        head=review_input.pr.head,
        archive_sha256=snapshot.archive_sha256
    )

    return result
