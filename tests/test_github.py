from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead

from reviewer.archives import sha256
from reviewer.benchmark import Task, reconstruct
from reviewer.diff import parse_diff
from reviewer.github import API, DIFF_MEDIA, AcquisitionError, GitHubClient, _download, normalize_repository
from reviewer.schemas import ReviewInput
from test_archives import archive_bytes

BASE, HEAD, MERGE = "a" * 40, "b" * 40, "c" * 40
DIFF = b"diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+new\n"


def metadata():
    return {"number": 7, "title": "Example PR", "changed_files": 1, "additions": 1, "deletions": 1,
            "base": {"sha": BASE, "repo": {"full_name": "owner/repo", "private": False}},
            "head": {"sha": HEAD, "repo": {"full_name": "fork/repo"}}}


def file_record(name="src/a.py"):
    return {"filename": name, "status": "modified", "additions": 1, "deletions": 1}


def diagnosed_benchmark_diff():
    """Text fixture with the pinned PR #120 per-file totals: 4 files, +91/-131."""
    specs = [
        ('.claude/rules/nix-tool-execution.md', 'removed', 0, 15),
        ('CHANGELOG.md', 'modified', 0, 3),
        ('CLAUDE.md', 'modified', 38, 113),
        ('roles/splunk_docker/README.md', 'modified', 53, 0),
    ]
    records, chunks = [], []
    for path, status, additions, deletions in specs:
        records.append({'filename': path, 'previous_filename': None,
                        'status': status, 'additions': additions, 'deletions': deletions})
        chunks.append(f'diff --git a/{path} b/{path}\n')
        if status == 'removed':
            chunks.append('deleted file mode 100644\n')
        chunks.extend((f'--- a/{path}\n', '+++ /dev/null\n' if status == 'removed' else f'+++ b/{path}\n'))
        old_start = 1 if deletions else 0
        new_start = 1 if additions else 0
        chunks.append(f'@@ -{old_start},{deletions} +{new_start},{additions} @@\n')
        chunks.extend('-old line\n' for _ in range(deletions))
        chunks.extend('+new line\n' for _ in range(additions))
    return ''.join(chunks).encode(), records


class FakeGitHub:
    def __init__(self):
        self.metadata = metadata()
        self.after = None
        self.comparison = {"base_commit": {"sha": BASE}, "merge_base_commit": {"sha": MERGE}}
        self.diff = DIFF
        self.files = [file_record()]
        self.archive = archive_bytes()
        self.calls = []
        self.metadata_calls = 0
        self.snapshot_error = None

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if "/tarball/" in url:
            if self.snapshot_error:
                raise self.snapshot_error
            return self.archive
        if kwargs["headers"]["Accept"] == DIFF_MEDIA:
            return self.diff
        if "/compare/" in url:
            return json.dumps(self.comparison).encode()
        if "/files?" in url:
            page = int(url.rsplit("page=", 1)[1])
            return json.dumps(self.files[(page - 1) * 100:page * 100]).encode()
        self.metadata_calls += 1
        return json.dumps(self.after if self.metadata_calls > 1 and self.after is not None else self.metadata).encode()


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name).resolve() / "cache"
        self.fake = FakeGitHub()
        self.client = GitHubClient(token="", downloader=self.fake)
        guard = patch("socket.create_connection", side_effect=AssertionError("tests must not access network"))
        guard.start(); self.addCleanup(guard.stop)

    def acquire(self, **kwargs):
        return self.client.acquire("owner/repo", 7, cache_dir=self.cache, **kwargs)

    def assert_error(self, code, **kwargs):
        with self.assertRaises(AcquisitionError) as caught:
            self.acquire(**kwargs)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_success_frozen_state_parser_and_hashes(self):
        result = self.acquire()
        self.assertIsInstance(result.review_input, ReviewInput)
        self.assertEqual(result.review_input.pr.base, BASE)
        self.assertEqual(result.review_input.pr.head, HEAD)
        self.assertEqual(result.review_input.diff_base, MERGE)
        self.assertTrue(result.review_input.diff_complete)
        self.assertEqual(result.parsed_diff[0].old_path, "src/a.py")
        self.assertEqual(result.parsed_diff[0].hunks[0].lines[0].old_line, 1)
        self.assertEqual(result.parsed_diff[0].hunks[0].lines[1].new_line, 1)
        self.assertEqual(result.diff_sha256, sha256(DIFF))
        self.assertEqual((result.artifact_directory / "diff.patch").read_bytes(), DIFF)
        self.assertEqual(ReviewInput.model_validate_json((result.artifact_directory / "input.json").read_text()), result.review_input)
        self.assertNotIn("title", result.review_input.model_dump())
        self.assertIsNone(result.head_snapshot)
        self.assertEqual(self.fake.metadata_calls, 2)
        comparisons = [url for url, _ in self.fake.calls if "/compare/" in url]
        self.assertTrue(all(BASE + "..." + HEAD in url for url in comparisons))
        self.assertEqual(self.acquire().diff_sha256, result.diff_sha256)

    def test_optional_snapshots_exact_shas_and_fork(self):
        result = self.acquire(snapshots=True)
        self.assertEqual(result.base_snapshot.commit, BASE)
        self.assertEqual(result.diff_base_snapshot.commit, MERGE)
        self.assertEqual(result.head_snapshot.commit, HEAD)
        self.assertEqual(result.head_snapshot.repository, "fork/repo")
        self.assertEqual((result.head_snapshot.directory / "src/a.py").read_bytes(), b"new\n")
        self.assertEqual(result.head_snapshot.archive_sha256, sha256(self.fake.archive))
        urls = [url for url, _ in self.fake.calls if "/tarball/" in url]
        self.assertEqual(urls, [f"{API}/repos/owner/repo/tarball/{BASE}",
                                f"{API}/repos/owner/repo/tarball/{MERGE}",
                                f"{API}/repos/fork/repo/tarball/{HEAD}"])

    def test_same_diff_base_reuses_snapshot(self):
        self.fake.comparison["merge_base_commit"]["sha"] = BASE
        result = self.acquire(snapshots=True)
        self.assertIs(result.base_snapshot, result.diff_base_snapshot)
        self.assertEqual(len([url for url, _ in self.fake.calls if "/tarball/" in url]), 2)

    def test_paginated_file_listing(self):
        self.fake.files = [file_record(f"x{i}.py") for i in range(101)]
        self.fake.diff = b"".join(DIFF.replace(b"src/a.py", f"x{i}.py".encode()) for i in range(101))
        self.fake.metadata.update(changed_files=101, additions=101, deletions=101)
        result = self.acquire()
        self.assertEqual(len(result.changed_files), 101)
        self.assertEqual(len(result.file_page_sha256), 2)

    def test_missing_file_or_hunk_counts_rejected(self):
        for mutate in (lambda: setattr(self.fake, "diff", b""),
                       lambda: self.fake.files[0].update(additions=2),
                       lambda: self.fake.files[0].update(filename="wrong.py"),
                       lambda: setattr(self.fake, "files", []),
                       lambda: self.fake.metadata.update(additions=2)):
            self.fake = FakeGitHub(); self.client = GitHubClient(token="", downloader=self.fake)
            mutate(); self.assert_error("incomplete_diff")

    def test_live_acquisition_still_rejects_aggregate_total_mismatch(self):
        self.fake.metadata.update(additions=94)
        with self.assertRaisesRegex(AcquisitionError, 'changed-file totals disagree'):
            self.acquire()

    def test_benchmark_reconstruction_audits_aggregate_mismatch_but_preserves_frozen_checks(self):
        data = json.loads((Path(__file__).resolve().parents[1] / 'configs/reviewbench-final20.json').read_text())
        task_data = next(item for item in data['tasks']
                         if item['task_id'] == 'JacobPEvans_ansible-splunk_120-252ddc57')
        task = Task.model_validate(task_data)
        diff, records = diagnosed_benchmark_diff()
        base, head = task.pr.base, task.pr.head
        def make_transport(*, returned_head=head, file_records=records, raw_diff=diff):
            def transport(url, **kwargs):
                if '/commits/' in url:
                    return json.dumps({'sha': returned_head}).encode()
                if kwargs['headers']['Accept'] == DIFF_MEDIA:
                    return raw_diff
                return json.dumps({'base_commit': {'sha': base}, 'merge_base_commit': {'sha': base},
                                   'files': file_records}).encode()
            return transport
        client = GitHubClient(token='', downloader=make_transport())
        result = reconstruct(task, client)
        self.assertEqual(result.pr.base, base)
        self.assertEqual(result.pr.head, head)
        self.assertEqual(len(result.acquisition_warnings), 1)
        warning = result.acquisition_warnings[0].model_dump()
        self.assertEqual(warning, {
            'code': 'benchmark_aggregate_mismatch', 'expected_file_count': 4,
            'expected_additions': 94, 'expected_deletions': 131,
            'reconstructed_file_count': 4, 'reconstructed_additions': 91,
            'reconstructed_deletions': 131, 'base_sha': base, 'head_sha': head,
            'mismatch_fields': ['additions'],
        })
        self.assertEqual((len(parse_diff(result.diff)),
                          sum(line.startswith('+') and not line.startswith('+++') for line in result.diff.splitlines()),
                          sum(line.startswith('-') and not line.startswith('---') for line in result.diff.splitlines())),
                         (4, 91, 131))
        matching_task = task.model_copy(update={'lines_added': 91})
        matching = reconstruct(matching_task, GitHubClient(token='', downloader=make_transport()))
        self.assertEqual(matching.acquisition_warnings, [])
        wrong_head = GitHubClient(token='', downloader=make_transport(returned_head='f'*40))
        with self.assertRaisesRegex(AcquisitionError, 'frozen_state_mismatch'):
            reconstruct(task, wrong_head)
        wrong_files = [dict(item) for item in records]
        wrong_files[2]['additions'] += 1
        with self.assertRaisesRegex(AcquisitionError, 'line counts disagree'):
            reconstruct(task, GitHubClient(token='', downloader=make_transport(file_records=wrong_files)))
        with self.assertRaisesRegex(AcquisitionError, 'invalid_diff'):
            reconstruct(task, GitHubClient(token='', downloader=make_transport(raw_diff=diff + b'\n')))

    def test_empty_pr(self):
        self.fake.diff = b""; self.fake.files = []
        self.fake.metadata.update(changed_files=0, additions=0, deletions=0)
        self.assertEqual(self.acquire().parsed_diff, ())

    def test_malformed_metadata_and_comparison(self):
        for change in ({"number": True}, {"changed_files": "1"}, {"base": {}},
                       {"title": None}, {"head": {"sha": "branch", "repo": None}}):
            self.fake.metadata = metadata() | change
            self.assert_error("malformed_response")
        self.fake.metadata = metadata()
        self.fake.comparison = {"base_commit": {"sha": HEAD}, "merge_base_commit": {"sha": MERGE}}
        self.assert_error("malformed_response")

    def test_changed_pr_is_not_returned(self):
        self.fake.after = deepcopy(self.fake.metadata)
        self.fake.after["head"]["sha"] = "d" * 40
        self.assert_error("pr_changed")
        self.assertFalse(self.cache.exists())

    def test_invalid_and_unsupported_diffs(self):
        self.fake.diff = DIFF[:-6]
        self.assert_error("invalid_diff")
        self.fake.diff = b"diff --git a/x b/x\nBinary files a/x and b/x differ\n"
        self.assert_error("unsupported_diff")

    def test_hash_mismatch(self):
        self.assert_error("hash_mismatch", expected_diff_sha256="0" * 64)
        self.assertEqual(self.acquire(expected_diff_sha256=sha256(DIFF)).diff_sha256, sha256(DIFF))

    def test_failed_snapshots_leave_no_partial_artifacts(self):
        self.fake.snapshot_error = AcquisitionError("not_found", "missing")
        self.assert_error("snapshot_unavailable", snapshots=True)
        self.assertEqual(list(self.cache.iterdir()), [])
        self.fake.snapshot_error = None
        self.fake.archive = b"bad archive"
        self.assert_error("unsafe_archive", snapshots=True)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_credentials_are_optional_and_not_serialized(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "test-secret"}):
            self.client = GitHubClient(downloader=self.fake)
            result = self.acquire()
        self.assertTrue(all(call[1]["headers"].get("Authorization") == "Bearer test-secret" for call in self.fake.calls))
        for path in result.artifact_directory.iterdir():
            self.assertNotIn(b"test-secret", path.read_bytes())
        self.assertNotIn("test-secret", repr(self.client))
        self.assertNotIn("test-secret", repr(result))

    def test_file_limits_and_malformed_file_records(self):
        self.fake.metadata["changed_files"] = 3001
        self.assert_error("size_limit")
        self.fake.metadata = metadata()
        for record in ({"filename": None}, {"filename": "../bad"}, {"additions": True},
                       {"status": "renamed"}):
            self.fake.files = [file_record() | record]
            self.assert_error("malformed_response")
        self.fake.files = [file_record() | {"status": "copied"}]
        self.assert_error("unsupported_diff")

    def test_missing_pr_and_malformed_json_fail_before_artifacts(self):
        self.client = GitHubClient(token="", downloader=Mock(side_effect=AcquisitionError("not_found", "missing PR")))
        self.assert_error("not_found")
        self.client = GitHubClient(token="", downloader=Mock(return_value=b"<html>bad gateway</html>"))
        self.assert_error("malformed_response")
        self.assertFalse(self.cache.exists())

    def test_unavailable_diff_does_not_return_input(self):
        normal_get = self.client._get
        def get(path, **kwargs):
            if kwargs.get("media") == DIFF_MEDIA:
                raise AcquisitionError("not_found", "diff unavailable")
            return normal_get(path, **kwargs)
        self.client._get = get
        self.assert_error("not_found")
        self.assertFalse(self.cache.exists())

    def test_deleted_fork_only_uses_same_head_sha(self):
        self.fake.metadata["head"]["repo"] = None
        result = self.acquire(snapshots=True)
        self.assertEqual(result.head_snapshot.repository, "owner/repo")
        self.assertEqual(result.head_snapshot.commit, HEAD)

    def test_cache_symlink_is_rejected(self):
        outside = Path(self.temp.name).resolve() / "outside"
        outside.mkdir()
        self.cache.symlink_to(outside, target_is_directory=True)
        self.assert_error("unsafe_cache")
        self.assertEqual(list(outside.iterdir()), [])

    def test_repository_normalization_and_input_validation(self):
        for repo in ("owner/repo", "https://github.com/owner/repo", "https://github.com/owner/repo.git/"):
            self.assertEqual(normalize_repository(repo), "owner/repo")
        for repo in ("http://github.com/o/r", "https://evil.test/o/r", "o/r?token=secret", "../r", "o/r/pull/7"):
            with self.assertRaises(AcquisitionError):
                normalize_repository(repo)
        for number in (True, 0, -1, "7"):
            with self.assertRaises(AcquisitionError):
                self.client.acquire("owner/repo", number)
        self.assertEqual(self.fake.calls, [])


class Response(io.BytesIO):
    status = 200
    def __init__(self, body=b"ok", headers=None):
        super().__init__(body)
        self.headers = {} if headers is None else headers


class TransportTests(unittest.TestCase):
    def request(self, opener, **kwargs):
        with patch("reviewer.github.build_opener", return_value=opener):
            return _download(API + "/repos/o/r", headers={"Authorization": "Bearer secret"},
                             timeout=1, max_bytes=10, **kwargs)

    def test_get_only(self):
        opener = Mock(); opener.open.return_value = Response()
        self.assertEqual(self.request(opener), b"ok")
        self.assertEqual(opener.open.call_args.args[0].method, "GET")

    def test_http_failures_sanitized(self):
        for status, headers, code in ((404, {}, "not_found"), (429, {}, "rate_limited"),
                (403, {"X-RateLimit-Remaining": "0"}, "rate_limited"),
                (403, {"Retry-After": "60"}, "rate_limited"), (401, {}, "access_denied"),
                (500, {}, "http_error")):
            with self.subTest(status=status, headers=headers):
                opener = Mock()
                opener.open.side_effect = HTTPError(API, status, "secret", headers, io.BytesIO(b"secret"))
                with self.assertRaises(AcquisitionError) as error:
                    self.request(opener)
                self.assertEqual(error.exception.code, code)
                self.assertNotIn("secret", str(error.exception))

    def test_timeout_network_and_partial_response(self):
        for failure, code in ((TimeoutError("secret"), "timeout"), (URLError(TimeoutError()), "timeout"),
                              (URLError("secret"), "network_error"), (IncompleteRead(b"secret"), "network_error")):
            opener = Mock(); opener.open.side_effect = failure
            with self.assertRaises(AcquisitionError) as error:
                self.request(opener)
            self.assertEqual(error.exception.code, code)
            self.assertNotIn("secret", str(error.exception))

    def test_download_bounds_and_content_length(self):
        for response, code in ((Response(b"x" * 11), "size_limit"),
                               (Response(b"ok", {"Content-Length": "3"}), "incomplete_response")):
            opener = Mock(); opener.open.return_value = response
            with self.assertRaises(AcquisitionError) as error:
                self.request(opener)
            self.assertEqual(error.exception.code, code)

    def test_archive_redirect_drops_credentials(self):
        opener = Mock()
        opener.open.side_effect = [HTTPError(API, 302, "redirect", {"Location": "https://codeload.github.com/o/r/tar.gz/sha"}, io.BytesIO()), Response()]
        self.assertEqual(self.request(opener, archive=True), b"ok")
        self.assertNotIn("Authorization", opener.open.call_args.args[0].headers)
        self.assertEqual(opener.open.call_args.args[0].method, "GET")

    def test_unsafe_redirect_is_not_followed(self):
        for location in ("http://codeload.github.com/x", "https://evil.test/x", "https://api.github.com/x",
                         "https://secret@codeload.github.com/x", "https://[bad"):
            opener = Mock()
            opener.open.side_effect = HTTPError(API, 302, "redirect", {"Location": location}, io.BytesIO())
            with self.assertRaises(AcquisitionError) as error:
                self.request(opener, archive=True)
            self.assertEqual(error.exception.code, "unsafe_redirect")
            self.assertEqual(opener.open.call_count, 1)
