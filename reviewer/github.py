"""Read-only public GitHub PR acquisition. No Git subprocesses or remote writes."""

from dataclasses import dataclass
import json
from http.client import HTTPException
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import ValidationError

from reviewer.archives import ArchiveError, extract_snapshot, sha256
from reviewer.diff import DiffParseError, FileDiff, UnsupportedDiffError, parse_diff
from reviewer.schemas import BenchmarkAggregateMismatch, PRMetadata, ReviewInput, repository_path

API = "https://api.github.com"
JSON_MEDIA = "application/vnd.github+json"
DIFF_MEDIA = "application/vnd.github.diff"


class AcquisitionError(ValueError):
    """A safe-to-display code/message; never includes response bodies or credentials."""
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _download(url: str, *, headers: dict[str, str], timeout: float, max_bytes: int,
              archive: bool = False) -> bytes:
    """GET only. Follow one archive redirect to codeload, dropping authorization."""
    opener = build_opener(_NoRedirect())
    for attempt in range(2):
        try:
            with opener.open(Request(url, headers=headers, method="GET"), timeout=timeout) as response:
                if response.status != 200:
                    raise AcquisitionError("http_error", "unexpected GitHub response status")
                data = response.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise AcquisitionError("size_limit", "download exceeded configured byte limit")
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdecimal() or int(length) != len(data)):
                    raise AcquisitionError("incomplete_response", "download length does not match Content-Length")
                return data
        except HTTPError as error:
            with error:
                status = error.code
                response_headers = error.headers
                if status in (301, 302, 303, 307, 308) and archive and attempt == 0:
                    target = response_headers.get("Location", "")
                    try:
                        parsed = urlsplit(target)
                    except ValueError:
                        raise AcquisitionError("unsafe_redirect", "malformed archive redirect") from None
                    if (parsed.scheme != "https" or parsed.netloc != "codeload.github.com"
                            or parsed.username or parsed.password or parsed.fragment):
                        raise AcquisitionError("unsafe_redirect", "archive redirect was not to GitHub codeload") from None
                    url = target
                    headers = {key: value for key, value in headers.items() if key.lower() != "authorization"}
                    continue
                body = error.read(4096).lower()
                if status == 429 or status == 403 and (
                    response_headers.get("X-RateLimit-Remaining") == "0"
                    or response_headers.get("Retry-After") is not None or b"rate limit" in body
                ):
                    raise AcquisitionError("rate_limited", "GitHub rate limit reached; retry later or configure GITHUB_TOKEN") from None
                if status == 404:
                    raise AcquisitionError("not_found", "requested repository, PR, or frozen artifact was not found") from None
                if status in (401, 403):
                    raise AcquisitionError("access_denied", "GitHub denied access; check public availability or token permissions") from None
                raise AcquisitionError("http_error", f"GitHub returned HTTP {status}") from None
        except (TimeoutError, socket.timeout):
            raise AcquisitionError("timeout", "GitHub request timed out") from None
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise AcquisitionError("timeout", "GitHub request timed out") from None
            raise AcquisitionError("network_error", "GitHub request failed") from None
        except (OSError, HTTPException):
            raise AcquisitionError("network_error", "GitHub request failed or returned an incomplete response") from None
    raise AcquisitionError("unsafe_redirect", "too many archive redirects")


def normalize_repository(repository: str) -> str:
    """Accept owner/name or an HTTPS github.com repository URL, optionally .git/."""
    if not isinstance(repository, str):
        raise AcquisitionError("invalid_input", "repository must be a string")
    name = repository.removeprefix("https://github.com/").rstrip("/")
    if name.endswith(".git"):
        name = name[:-4]
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name)
            or any(part in (".", "..") for part in name.split("/"))):
        raise AcquisitionError("invalid_input", "use owner/name or https://github.com/owner/name")
    return name


def _json(data: bytes):
    try:
        value = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise AcquisitionError("malformed_response", "GitHub returned invalid JSON") from None
    return value


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("expected nonnegative integer")
    return value


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("invalid commit SHA")
    return value


@dataclass(frozen=True)
class ChangedFile:
    filename: str
    previous_filename: str | None
    status: str
    additions: int
    deletions: int


@dataclass(frozen=True)
class Snapshot:
    repository: str
    commit: str
    archive_sha256: str
    archive_path: Path
    directory: Path


@dataclass(frozen=True)
class AcquiredPR:
    review_input: ReviewInput
    title: str  # Provenance only, not part of diff-only model input.
    changed_files: tuple[ChangedFile, ...]
    parsed_diff: tuple[FileDiff, ...]
    diff_sha256: str  # SHA-256 of exact downloaded UTF-8 diff bytes.
    metadata_sha256: str
    comparison_sha256: str
    file_page_sha256: tuple[str, ...]
    base_snapshot: Snapshot | None  # PR base branch tip, not necessarily diff base.
    diff_base_snapshot: Snapshot | None
    head_snapshot: Snapshot | None
    artifact_directory: Path


class GitHubClient:
    def __init__(self, *, token: str | None = None, timeout: float = 30,
                 max_diff_bytes: int = 10 * 1024 * 1024,
                 max_archive_bytes: int = 64 * 1024 * 1024,
                 downloader: Callable = _download):
        # No credentials are serialized. An explicitly empty token disables env auth.
        token = os.environ.get("GITHUB_TOKEN", "") if token is None else token
        if not isinstance(token, str) or any(ord(c) < 32 or ord(c) == 127 for c in token):
            raise AcquisitionError("invalid_input", "invalid token configuration")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not 0 < timeout < float("inf")):
            raise AcquisitionError("invalid_input", "timeout must be positive and finite")
        for limit in (max_diff_bytes, max_archive_bytes):
            if type(limit) is not int or limit <= 0:
                raise AcquisitionError("invalid_input", "download limits must be positive integers")
        self._token = token
        self.timeout = timeout
        self.max_diff_bytes = max_diff_bytes
        self.max_archive_bytes = max_archive_bytes
        self._downloader = downloader

    def _get(self, path: str, *, media=JSON_MEDIA, archive=False) -> bytes:
        headers = {"Accept": media, "User-Agent": "context-aware-code-review/0.0.1",
                   "X-GitHub-Api-Version": "2026-03-10", "Accept-Encoding": "identity"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        limit = self.max_archive_bytes if archive else self.max_diff_bytes if media == DIFF_MEDIA else 8 * 1024 * 1024
        return self._downloader(API + path, headers=headers, timeout=self.timeout,
                                max_bytes=limit, archive=archive)

    def frozen_input(self, pr: PRMetadata, *, source_repository: str,
                     expected_counts: tuple[int, int, int], expected_diff_sha256: str | None,
                     allow_benchmark_aggregate_mismatch: bool = False) -> ReviewInput:
        """Reconstruct an immutable benchmark diff, never consulting live PR state.

        Reuses HTTP, unified-diff parsing and inventory checks. Small subsets only:
        GitHub comparison inventories are capped at 300 files, so reject that bound.
        """
        pr = PRMetadata.model_validate(pr.model_dump())
        source = normalize_repository(source_repository)
        if (len(expected_counts) != 3 or any(type(n) is not int or n < 0 for n in expected_counts)
                or not 0 < expected_counts[0] < 300
                or (expected_diff_sha256 is not None and
                    (not isinstance(expected_diff_sha256, str)
                     or not re.fullmatch(r'[0-9a-f]{64}', expected_diff_sha256)))):
            raise AcquisitionError('invalid_input', 'invalid frozen input expectations')
        path = f'/repos/{source}/compare/{pr.base}...{pr.head}'
        head = _json(self._get(f'/repos/{source}/commits/{pr.head}'))
        comparison = _json(self._get(path + '?per_page=1&page=1'))
        try:
            if _sha(head['sha']) != pr.head or _sha(comparison['base_commit']['sha']) != pr.base:
                raise ValueError('unexpected commit identity')
            diff_base = _sha(comparison['merge_base_commit']['sha'])
            records = comparison['files']
            if not isinstance(records, list) or len(records) != expected_counts[0]:
                raise ValueError('incomplete file inventory')
            files = []
            for record in records:
                status = record['status']
                if status not in ('added', 'removed', 'modified', 'renamed'):
                    raise ValueError('unsupported file status')
                files.append(ChangedFile(repository_path(record['filename']),
                    repository_path(record['previous_filename']) if status == 'renamed' else None,
                    status, _integer(record['additions']), _integer(record['deletions'])))
        except (KeyError, TypeError, ValueError):
            raise AcquisitionError('frozen_state_mismatch', 'frozen commit identity or inventory does not match benchmark') from None
        raw = self._get(path, media=DIFF_MEDIA)
        if expected_diff_sha256 is not None and sha256(raw) != expected_diff_sha256:
            raise AcquisitionError('hash_mismatch', 'frozen diff differs from subset lock')
        try:
            diff = raw.decode('utf-8')
            parsed = parse_diff(diff)
        except (DiffParseError, UnicodeDecodeError):
            raise AcquisitionError('invalid_diff', 'frozen diff is malformed or unsupported') from None
        aggregate_mismatch = self._check_diff(
            parsed, files, expected_counts,
            allow_aggregate_mismatch=allow_benchmark_aggregate_mismatch)
        warnings = []
        if aggregate_mismatch is not None:
            warnings.append(BenchmarkAggregateMismatch(
                code='benchmark_aggregate_mismatch',
                expected_file_count=expected_counts[0], expected_additions=expected_counts[1],
                expected_deletions=expected_counts[2],
                reconstructed_file_count=aggregate_mismatch['reconstructed_file_count'],
                reconstructed_additions=aggregate_mismatch['reconstructed_additions'],
                reconstructed_deletions=aggregate_mismatch['reconstructed_deletions'],
                base_sha=pr.base, head_sha=pr.head,
                mismatch_fields=aggregate_mismatch['mismatch_fields']))
        return ReviewInput(pr=pr, diff=diff, diff_base=diff_base, diff_complete=True,
                           acquisition_warnings=warnings)

    def _metadata(self, name: str, number: int):
        raw = self._get(f"/repos/{name}/pulls/{number}")
        try:
            item = _json(raw)
            if type(item["number"]) is not int or item["number"] != number:
                raise ValueError("wrong PR number")
            base_repo = normalize_repository(item["base"]["repo"]["full_name"])
            if base_repo.casefold() != name.casefold() or item["base"]["repo"]["private"] is not False:
                raise ValueError("unexpected or nonpublic repository")
            pr = PRMetadata(repo="https://github.com/" + base_repo, pr_number=number,
                            base=_sha(item["base"]["sha"]), head=_sha(item["head"]["sha"]))
            head_repo = item["head"]["repo"]
            head_name = normalize_repository(head_repo["full_name"]) if head_repo else base_repo
            title = item["title"]
            if not isinstance(title, str):
                raise ValueError("invalid title")
            totals = tuple(_integer(item[key]) for key in ("changed_files", "additions", "deletions"))
        except (KeyError, TypeError, ValueError, ValidationError):
            raise AcquisitionError("malformed_response", "invalid GitHub PR metadata") from None
        return pr, title, head_name, totals, raw

    def acquire(self, repository: str, pr_number: int, *, snapshots: bool = False,
                cache_dir: str | Path = ".cache/github", expected_diff_sha256: str | None = None) -> AcquiredPR:
        """Acquire a stable, bounded public PR; persist artifacts only on full success.

        PR files are mutable endpoints, so metadata is checked again after download.
        Diff/archives use immutable commit identifiers, never default branches.
        Snapshot downloading is opt-in; baseline acquisition needs only the diff.
        """
        name = normalize_repository(repository)
        if type(pr_number) is not int or pr_number <= 0:
            raise AcquisitionError("invalid_input", "PR number must be a positive integer")
        if type(snapshots) is not bool:
            raise AcquisitionError("invalid_input", "snapshots must be boolean")
        pr, title, head_name, totals, metadata_raw = self._metadata(name, pr_number)
        name = pr.repo.removeprefix("https://github.com/")
        if totals[0] > 3000:
            raise AcquisitionError("size_limit", "PR exceeds GitHub's 3000-file listing limit")
        comparison_path = f"/repos/{name}/compare/{pr.base}...{pr.head}"
        comparison_raw = self._get(comparison_path + "?per_page=1&page=1")
        try:
            comparison = _json(comparison_raw)
            diff_base = _sha(comparison["merge_base_commit"]["sha"])
            if _sha(comparison["base_commit"]["sha"]) != pr.base:
                raise ValueError("base mismatch")
        except (KeyError, TypeError, ValueError):
            raise AcquisitionError("malformed_response", "invalid frozen commit comparison") from None
        diff_raw = self._get(comparison_path, media=DIFF_MEDIA)
        digest = sha256(diff_raw)
        if expected_diff_sha256 is not None and digest != expected_diff_sha256:
            raise AcquisitionError("hash_mismatch", "raw diff SHA-256 does not match expected value")
        try:
            diff_text = diff_raw.decode("utf-8")
            parsed = parse_diff(diff_text)
        except UnsupportedDiffError:
            raise AcquisitionError("unsupported_diff", "PR contains an unsupported diff format") from None
        except (DiffParseError, UnicodeDecodeError):
            raise AcquisitionError("invalid_diff", "PR diff is malformed, incomplete, or not UTF-8") from None
        files = []
        pages = []
        for page in range(1, max(1, (totals[0] + 99) // 100) + 1):
            raw = self._get(f"/repos/{name}/pulls/{pr_number}/files?per_page=100&page={page}")
            pages.append(raw)
            records = _json(raw)
            if not isinstance(records, list) or len(records) != min(100, totals[0] - len(files)):
                raise AcquisitionError("incomplete_diff", "PR file page count does not match metadata")
            for record in records:
                try:
                    status = record["status"]
                    if status not in ("added", "removed", "modified", "renamed"):
                        raise AcquisitionError("unsupported_diff", "unsupported GitHub file change type")
                    filename = record["filename"]
                    previous = record["previous_filename"] if status == "renamed" else None
                    if not isinstance(filename, str) or (status == "renamed" and not isinstance(previous, str)):
                        raise ValueError("invalid path types")
                    filename = repository_path(filename)
                    previous = repository_path(previous) if previous is not None else None
                    files.append(ChangedFile(filename, previous, status,
                                             _integer(record["additions"]), _integer(record["deletions"])))
                except AcquisitionError:
                    raise
                except (KeyError, TypeError, ValueError):
                    raise AcquisitionError("malformed_response", "invalid GitHub changed-file metadata") from None
        self._check_diff(parsed, files, totals)
        after_pr, _, after_head, after_totals, _ = self._metadata(name, pr_number)
        if after_pr != pr or after_totals != totals or after_head != head_name:
            raise AcquisitionError("pr_changed", "PR changed during acquisition; retry to acquire one frozen state")
        review_input = ReviewInput(pr=pr, diff=diff_text, diff_base=diff_base, diff_complete=True)
        cache = Path(cache_dir)
        if any(p.is_symlink() for p in (cache, *cache.parents)):
            raise AcquisitionError("unsafe_cache", "cache path must not contain symlinks")
        cache.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="pr-", dir=cache))
        try:
            (staging / "diff.patch").write_bytes(diff_raw)
            # Only allowlisted metadata is persisted: no bodies, comments, URLs, tokens.
            (staging / "input.json").write_text(review_input.model_dump_json(indent=2), encoding="utf-8")
            base_snapshot = diff_base_snapshot = head_snapshot = None
            if snapshots:
                base_snapshot = self._snapshot(name, pr.base, staging / "base")
                diff_base_snapshot = base_snapshot if diff_base == pr.base else self._snapshot(name, diff_base, staging / "diff-base")
                head_snapshot = self._snapshot(head_name, pr.head, staging / "head")
            result = AcquiredPR(review_input, title, tuple(files), parsed, digest,
                                sha256(metadata_raw), sha256(comparison_raw), tuple(sha256(x) for x in pages),
                                base_snapshot, diff_base_snapshot, head_snapshot, staging)
            manifest = {"schema_version": 1, "pr": pr.model_dump(), "title": title, "diff_base": diff_base,
                        "diff_sha256": digest, "metadata_sha256": result.metadata_sha256,
                        "comparison_sha256": result.comparison_sha256,
                        "file_page_sha256": result.file_page_sha256,
                        "changed_files": [vars(file) for file in files],
                        "snapshots": {role: {"repository": snap.repository, "commit": snap.commit,
                                             "archive_sha256": snap.archive_sha256}
                                      for role, snap in (("base", base_snapshot), ("diff_base", diff_base_snapshot),
                                                         ("head", head_snapshot)) if snap is not None}}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return result
        except BaseException:
            shutil.rmtree(staging)
            raise

    def frozen_head(self, pr: PRMetadata, *, source_repository: str, cache_dir='.cache/github') -> Snapshot:
        """Acquire only the exact HEAD archive for an already reconstructed task."""
        pr = PRMetadata.model_validate(pr.model_dump())
        repository = normalize_repository(source_repository)
        cache = Path(cache_dir)
        if any(p.is_symlink() for p in (cache, *cache.parents)):
            raise AcquisitionError('unsafe_cache', 'cache path must not contain symlinks')
        cache.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='head-', dir=cache))
        try:
            return self._snapshot(repository, pr.head, staging / 'head')
        except BaseException:
            shutil.rmtree(staging)
            raise

    def _snapshot(self, repository: str, commit: str, target: Path) -> Snapshot:
        try:
            data = self._get(f"/repos/{repository}/tarball/{commit}", archive=True)
        except AcquisitionError as error:
            if error.code == "not_found":
                raise AcquisitionError("snapshot_unavailable", "frozen commit archive is unavailable; no branch fallback") from None
            raise
        try:
            digest = extract_snapshot(data, target)
        except ArchiveError:
            raise AcquisitionError("unsafe_archive", "snapshot archive is unsafe, malformed, or exceeds extraction limits") from None
        archive_path = target.with_suffix(".tar.gz")
        archive_path.write_bytes(data)
        return Snapshot(repository, commit, digest, archive_path, target)

    @staticmethod
    def _check_diff(parsed, files, totals, *, allow_aggregate_mismatch=False):
        if len(parsed) != len(files) or len({file.filename for file in files}) != len(files):
            raise AcquisitionError("incomplete_diff", "diff and changed-file inventory disagree")
        by_name = {file.filename: file for file in files}
        for patch in parsed:
            file = by_name.get(patch.new_path or patch.old_path)
            status = "removed" if patch.status == "deleted" else patch.status
            if file is None or file.status != status or (status == "renamed" and file.previous_filename != patch.old_path):
                raise AcquisitionError("incomplete_diff", "diff paths/status disagree with file inventory")
            added = sum(line.kind == "added" for hunk in patch.hunks for line in hunk.lines)
            deleted = sum(line.kind == "deleted" for hunk in patch.hunks for line in hunk.lines)
            if (added, deleted) != (file.additions, file.deletions):
                raise AcquisitionError("incomplete_diff", "diff line counts disagree with file inventory")
        reconstructed = (sum(file.additions for file in files), sum(file.deletions for file in files))
        if reconstructed != totals[1:]:
            if not allow_aggregate_mismatch:
                raise AcquisitionError("incomplete_diff", "changed-file totals disagree with PR metadata")
            mismatch_fields = []
            if reconstructed[0] != totals[1]:
                mismatch_fields.append('additions')
            if reconstructed[1] != totals[2]:
                mismatch_fields.append('deletions')
            return {
                'reconstructed_file_count': len(files),
                'reconstructed_additions': reconstructed[0],
                'reconstructed_deletions': reconstructed[1],
                'mismatch_fields': mismatch_fields,
            }
        return None
