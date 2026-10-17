import json
import unittest

from pydantic import ValidationError

from reviewer.schemas import Finding, FindingBatch, PRMetadata, ReviewInput, ReviewResult


def pr_data():
    return dict(repo="https://github.com/example/project", pr_number=123,
                base="a" * 40, head="b" * 40)


def finding_data():
    return dict(title="Lost update", category="correctness", severity="high",
                file="src/accounts.py", start_line=4, end_line=6,
                description="Concurrent writes lose the previous update.",
                evidence=["src/accounts.py:4-6 reads then writes separately."],
                suggested_fix="Use an atomic update.", confidence=0.9)


class SchemaTests(unittest.TestCase):
    def test_valid_finding_json_roundtrip(self):
        batch = FindingBatch.from_json(json.dumps({"findings": [finding_data()]}))
        self.assertEqual(batch.findings[0].start_line, 4)
        self.assertEqual(FindingBatch.from_json(batch.model_dump_json()), batch)
        self.assertFalse(Finding.model_json_schema()["additionalProperties"])

    def test_malformed_finding_fields(self):
        cases = [("title", " "), ("category", "style"), ("severity", "critical"),
                 ("start_line", 0), ("start_line", True), ("start_line", "4"),
                 ("end_line", 3), ("end_line", 6.5), ("description", ""),
                 ("suggested_fix", "\n"), ("confidence", -0.1), ("confidence", 1.1),
                 ("confidence", float("nan")), ("confidence", float("inf")),
                 ("confidence", True), ("confidence", "0.9"), ("evidence", []),
                 ("evidence", [" "]), ("evidence", "source"), ("extra", "unknown")]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                Finding.model_validate(finding_data() | {key: value})
        for key in finding_data():
            data = finding_data(); del data[key]
            with self.subTest(missing=key), self.assertRaises(ValidationError):
                Finding.model_validate(data)

    def test_paths(self):
        for path in ("", "/tmp/x", "../x", "a/../x", "./x", "a//x", "x/",
                     "a\\b", "C:/x", "a\x00b", "a\nb", "a\tb"):
            with self.subTest(path=path), self.assertRaises(ValidationError):
                Finding.model_validate(finding_data() | {"file": path})
        for path in ("src/my file.py", "src/é.py"):
            self.assertEqual(Finding.model_validate(finding_data() | {"file": path}).file, path)

    def test_json_is_not_repaired(self):
        for raw in ('{"findings":', '```json\n{"findings": []}\n```',
                    '{"findings": [], "findings": []}', '{"findings": [], "x": NaN}',
                    '{"findings": [], "x": Infinity}', '[]', '{"findings": null}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                FindingBatch.from_json(raw)
        self.assertEqual(FindingBatch.from_json('{"findings": []}').findings, [])

    def test_pr_metadata(self):
        self.assertEqual(PRMetadata(**pr_data()).pr_number, 123)
        for key, value in (("pr_number", True), ("pr_number", 0), ("pr_number", "123"),
                           ("head", "main"), ("base", "a" * 39), ("base", "A" * 40),
                           ("repo", "https://github.com/a/b?token=secret"),
                           ("repo", "http://github.com/a/b"), ("repo", "https://github.com/../b")):
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                PRMetadata.model_validate(pr_data() | {key: value})

    def test_input_requires_explicit_completeness(self):
        data = dict(pr=pr_data(), diff="", diff_complete=True)
        self.assertEqual(ReviewInput.model_validate(data).diff, "")
        for value in (False, 1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ReviewInput.model_validate(data | {"diff_complete": value})
        with self.assertRaises(ValidationError):
            ReviewInput.model_validate({"pr": pr_data(), "diff": ""})
        with self.assertRaises(ValidationError):
            ReviewInput.model_validate(data | {"golden_findings": []})

    def test_diff_base_is_explicit_and_validated(self):
        data = dict(pr=pr_data(), diff="", diff_complete=True)
        self.assertIsNone(ReviewInput.model_validate(data).diff_base)
        self.assertEqual(ReviewInput.model_validate(data | {"diff_base": "c" * 40}).diff_base, "c" * 40)
        with self.assertRaises(ValidationError):
            ReviewInput.model_validate(data | {"diff_base": "main"})

    def test_result_distinguishes_failure_and_clean_review(self):
        data = dict(pr=pr_data(), mode="baseline", status="success", findings=[])
        self.assertEqual(ReviewResult.model_validate(data).findings, [])
        self.assertEqual(ReviewResult.model_validate(data | {"status": "failed", "error": "timeout"}).error, "timeout")
        for update in ({"status": "failed"}, {"error": "timeout"}, {"mode": "unknown"},
                       {"schema_version": True}, {"schema_version": 1.0},
                       {"status": "failed", "error": "timeout", "findings": [finding_data()]}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                ReviewResult.model_validate(data | update)
