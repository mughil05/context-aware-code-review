from dataclasses import FrozenInstanceError
import unittest

from reviewer.diff import DiffParseError, UnsupportedDiffError, parse_diff


MIXED = """diff --git a/src/a.py b/src/a.py
index 1234567..abcdef0 100644
--- a/src/a.py
+++ b/src/a.py
@@ -1,3 +1,4 @@ example
 same
-old
+new
+extra
 last
@@ -8,2 +9 @@ later
 context
-removed
"""


def plain(body, old="a/x.py", new="b/x.py"):
    return f"--- {old}\n+++ {new}\n{body}"


class DiffTests(unittest.TestCase):
    def test_added_deleted_context_and_multiple_hunks(self):
        file, = parse_diff(MIXED)
        self.assertEqual((file.old_path, file.new_path, file.status), ("src/a.py", "src/a.py", "modified"))
        self.assertEqual(len(file.hunks), 2)
        self.assertEqual(file.hunks[0].section, "example")
        self.assertEqual([(x.kind, x.old_line, x.new_line, x.content) for x in file.hunks[0].lines], [
            ("context", 1, 1, "same"), ("deleted", 2, None, "old"),
            ("added", None, 2, "new"), ("added", None, 3, "extra"),
            ("context", 3, 4, "last")])
        self.assertEqual([(x.old_line, x.new_line) for x in file.hunks[1].lines], [(8, 9), (9, None)])
        self.assertEqual(file.old_to_new(3), 4)
        self.assertEqual(file.new_to_old(4), 3)
        self.assertIsNone(file.old_to_new(2))
        self.assertIsNone(file.new_to_old(2))
        self.assertIsNone(file.old_to_new(6))  # No extrapolation beyond represented context.
        with self.assertRaises(ValueError):
            file.new_to_old(True)
        self.assertEqual(parse_diff(MIXED), parse_diff(MIXED))
        with self.assertRaises(FrozenInstanceError):
            file.old_path = "changed"

    def test_multiple_files(self):
        files = parse_diff(MIXED + plain("@@ -1 +1 @@\n-a\n+b\n", "a/y.py", "b/y.py"))
        self.assertEqual([x.new_path for x in files], ["src/a.py", "y.py"])

    def test_added_file(self):
        patch = "diff --git a/new.py b/new.py\nnew file mode 100644\n" + plain(
            "@@ -0,0 +1,2 @@\n+one\n+two\n", "/dev/null", "b/new.py")
        file, = parse_diff(patch)
        self.assertIsNone(file.old_path)
        self.assertEqual(file.status, "added")
        self.assertEqual([x.new_line for x in file.hunks[0].lines], [1, 2])
        self.assertTrue(all(x.old_line is None for x in file.hunks[0].lines))

    def test_deleted_file(self):
        file, = parse_diff(plain("@@ -1,2 +0,0 @@\n-one\n-two\n", "a/gone.py", "/dev/null"))
        self.assertEqual(file.status, "deleted")
        self.assertIsNone(file.new_path)
        self.assertEqual([x.old_line for x in file.hunks[0].lines], [1, 2])
        self.assertTrue(all(x.new_line is None for x in file.hunks[0].lines))

    def test_rename_with_changes(self):
        file, = parse_diff("diff --git a/old.py b/new.py\nsimilarity index 80%\n"
                           "rename from old.py\nrename to new.py\n" +
                           plain("@@ -1 +1 @@\n-before\n+after\n", "a/old.py", "b/new.py"))
        self.assertEqual((file.old_path, file.new_path, file.status), ("old.py", "new.py", "renamed"))

    def test_metadata_only_rename_and_mode_change(self):
        file, = parse_diff("diff --git a/old.py b/new.py\nsimilarity index 100%\n"
                           "rename from old.py\nrename to new.py\n")
        self.assertEqual(file.hunks, ())
        self.assertEqual(file.status, "renamed")
        file, = parse_diff("diff --git a/x.py b/x.py\nold mode 100644\nnew mode 100755\n")
        self.assertEqual((file.old_mode, file.new_mode), ("100644", "100755"))

    def test_empty_file_addition_and_deletion(self):
        for operation, index, status in [("new", "0000000..e69de29", "added"),
                                         ("deleted", "e69de29..0000000", "deleted")]:
            with self.subTest(status=status):
                file, = parse_diff(f"diff --git a/x b/x\n{operation} file mode 100644\nindex {index}\n")
                self.assertEqual(file.status, status)
                self.assertEqual(file.hunks, ())

    def test_zero_length_insert_delete_ranges(self):
        file, = parse_diff(plain("@@ -3,0 +4,2 @@\n+a\n+b\n"))
        self.assertEqual([x.new_line for x in file.hunks[0].lines], [4, 5])
        file, = parse_diff(plain("@@ -4,2 +3,0 @@\n-a\n-b\n"))
        self.assertEqual([x.old_line for x in file.hunks[0].lines], [4, 5])
        file, = parse_diff(plain("@@ -0,0 +1 @@\n+first\n"))
        self.assertEqual(file.hunks[0].lines[0].new_line, 1)

    def test_no_newline_markers_are_not_code_lines(self):
        file, = parse_diff(plain("@@ -1 +1 @@\n-old\n\\ No newline at end of file\n"
                                 "+new\n\\ No newline at end of file\n"))
        self.assertTrue(all(x.no_newline for x in file.hunks[0].lines))
        self.assertEqual(len(file.hunks[0].lines), 2)
        file, = parse_diff(plain("@@ -1 +1 @@\n same\n\\ No newline at end of file\n"))
        self.assertTrue(file.hunks[0].lines[0].no_newline)

    def test_preserves_payload_whitespace_and_header_like_code(self):
        file, = parse_diff(plain("@@ -1,2 +1,2 @@\n--- text\n+++ text\n \t trailing \r\n"))
        self.assertEqual([x.content for x in file.hunks[0].lines], ["-- text", "++ text", "\t trailing \r"])

    def test_paths_spaces_unicode_and_plain_timestamps(self):
        file, = parse_diff('diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\n'
                           '--- "a/caf\\303\\251.py"\n+++ "b/caf\\303\\251.py"\n@@ -1 +1 @@\n-a\n+b\n')
        self.assertEqual(file.new_path, "café.py")
        file, = parse_diff('diff --git a/my file.py b/my file.py\n' +
                           plain("@@ -1 +1 @@\n-a\n+b\n", "a/my file.py", "b/my file.py"))
        self.assertEqual(file.new_path, "my file.py")
        file, = parse_diff(plain("@@ -1 +1 @@\n-a\n+b\n", "src/x.py\tdate", "src/x.py\tdate"))
        self.assertEqual(file.new_path, "src/x.py")

    def test_empty_input(self):
        self.assertEqual(parse_diff(""), ())
        with self.assertRaises(TypeError):
            parse_diff(b"patch")

    def test_truncated_and_malformed_patches(self):
        cases = ["\n", "@@ -1 +1 @@\n-a\n+b\n", "--- a/x\n", "diff --git a/x b/x\n",
                 plain(""), plain("@@ -1,2 +1,2 @@\n same\n"),
                 plain("@@ -1 +1 @@\n-a\n"), plain("@@ -1 +1 @@\n-a\n+b"),
                 plain("@@ -1 +1 @@\n-a\n+b\n+surplus\n"),
                 plain("@@ -1 +1 @@\n unmarked\ninvalid\n"),
                 plain("@@ -0 +1 @@\n-a\n+b\n"), plain("@@ -0,0 +0,0 @@\n"),
                 plain("@@ -1,-1 +1 @@\n-a\n+b\n"),
                 plain("@@ -1 +1 @@\n-a\n+b\n@@ -1 +1 @@\n-c\n+d\n"),
                 plain("@@ -1 +1 @@\n-a\n+b\n@@ -4 +5 @@\n-c\n+d\n"),
                 plain("@@ -1 +1 @@\n-a\n+b\n", "/dev/null", "b/x"),
                 plain("@@ -1 +1 @@\n-a\n+b\n", "a/x", "/dev/null"),
                 plain("@@ -1 +1 @@\n-a\n+b\n", "/dev/null", "/dev/null"),
                 plain("@@ -1 +1 @@\n\\ No newline at end of file\n-a\n+b\n"),
                 plain("@@ -1 +1 @@\n same\n\\ No newline at end of file\n\\ No newline at end of file\n"),
                 plain("@@ -1,2 +1,2 @@\n one\n\\ No newline at end of file\n two\n"),
                 plain("@@ -1,2 +1 @@\n-a\n+b\n"),
                 plain("@@ -1 +1 @@\n-a\n+b\n") * 2,
                 "diff --git a/x b/x\nnew file mode 100644\nindex 0000000..1234567\n",
                 "diff --git a/x b/y\nsimilarity index 80%\nrename from x\nrename to y\n",
                 "diff --git a/x b/y\nsimilarity index 100%\nrename from x\n",
                 "diff --git a/x b/x\nold mode 100644\n",
                 "diff --git a/x b/x\nold mode 100644\nold mode 100644\nnew mode 100755\n",
                 "diff --git a/x b/x\n" + plain("@@ -1 +1 @@\n-a\n+b\n", "a/y", "b/y"),
                 "diff --git a/x b/y\n" + plain("@@ -1 +1 @@\n-a\n+b\n", "a/x", "b/y")]
        for patch in cases:
            with self.subTest(patch=patch), self.assertRaises(DiffParseError):
                parse_diff(patch)

    def test_metadata_cannot_hide_missing_changes(self):
        cases = [
            "diff --git a/x b/x\nold mode 100644\nnew mode 100755\nindex 1234567..abcdef0\n",
            "diff --git a/x b/x\nsimilarity index 100%\nrename from x\nrename to x\n",
            "diff --git a/x b/x\nnew file mode 100644\nindex 1234567..abcdef0\n" +
            plain("@@ -0,0 +1 @@\n+added\n", "/dev/null", "b/x"),
        ]
        for patch in cases:
            with self.subTest(patch=patch), self.assertRaises(DiffParseError):
                parse_diff(patch)

    def test_unsafe_paths(self):
        for path in ("a/../x", "/etc/passwd", "a/./x", "a/x//y", '"a/x\\t.py"', "a/C:/x"):
            with self.subTest(path=path), self.assertRaises(DiffParseError):
                parse_diff(plain("@@ -1 +1 @@\n-a\n+b\n", path, "b/x"))

    def test_unsupported_formats(self):
        for patch in ("diff --cc x\n", "diff --combined x\n",
                      "diff --git a/x b/x\nGIT binary patch\n",
                      "diff --git a/x b/x\nBinary files a/x and b/x differ\n",
                      "diff --git a/x b/y\ncopy from x\ncopy to y\n",
                      "diff --git a/x b/x\n\x00\n"):
            with self.subTest(patch=patch), self.assertRaises(UnsupportedDiffError):
                parse_diff(patch)
