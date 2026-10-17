import gzip
import hashlib
import io
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

from reviewer.archives import ArchiveError, extract_snapshot, sha256


def archive_bytes(entries=None):
    entries = entries if entries is not None else [("repo-sha/src/a.py", b"new\n", tarfile.REGTYPE)]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, content, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = content.decode()
            elif kind == tarfile.REGTYPE:
                member.size = len(content)
            archive.addfile(member, io.BytesIO(content) if kind == tarfile.REGTYPE else None)
    return gzip.compress(stream.getvalue(), mtime=0)


class ArchiveTests(unittest.TestCase):
    def test_regular_files_hash_and_wrapper_removal(self):
        data = archive_bytes()
        with TemporaryDirectory() as temp:
            target = Path(temp).resolve() / "snapshot"
            self.assertEqual(extract_snapshot(data, target), hashlib.sha256(data).hexdigest())
            self.assertEqual((target / "src/a.py").read_bytes(), b"new\n")
            self.assertFalse((target / "repo-sha").exists())

    def test_unsafe_paths_fail_without_partial_output(self):
        for name in ("../escape", "/tmp/escape", "root/../../escape", "root/./x",
                     "root//x", "C:/escape", "root/a\\b", "root/a\tb"):
            with self.subTest(name=name), TemporaryDirectory() as temp:
                target = Path(temp).resolve() / "snapshot"
                data = archive_bytes([("root/good", b"ok", tarfile.REGTYPE), (name, b"bad", tarfile.REGTYPE)])
                with self.assertRaises(ArchiveError):
                    extract_snapshot(data, target)
                self.assertEqual(list(Path(temp).resolve().iterdir()), [])

    def test_all_links_and_special_members_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE):
            with self.subTest(kind=kind), TemporaryDirectory() as temp:
                data = archive_bytes([("root/link", b"../../outside", kind)])
                with self.assertRaises(ArchiveError):
                    extract_snapshot(data, Path(temp).resolve() / "snapshot")
                self.assertEqual(list(Path(temp).resolve().iterdir()), [])

    def test_malformed_duplicate_and_conflicting_archives(self):
        cases = [b"not gzip", gzip.compress(b"not tar"), archive_bytes()[:15], archive_bytes([]),
                 archive_bytes([("root/a", b"x", tarfile.REGTYPE)] * 2),
                 archive_bytes([("root/a", b"x", tarfile.REGTYPE), ("other/b", b"y", tarfile.REGTYPE)]),
                 archive_bytes([("root/a", b"x", tarfile.REGTYPE), ("root/a/b", b"y", tarfile.REGTYPE)]),
                 archive_bytes([("root", b"x", tarfile.REGTYPE)])]
        for data in cases:
            with self.subTest(size=len(data)), TemporaryDirectory() as temp:
                with self.assertRaises(ArchiveError):
                    extract_snapshot(data, Path(temp).resolve() / "snapshot")
                self.assertEqual(list(Path(temp).resolve().iterdir()), [])

    def test_limits_and_expected_hash(self):
        data = archive_bytes([("root/a", b"x", tarfile.REGTYPE), ("root/b", b"y", tarfile.REGTYPE)])
        for kwargs in ({"max_bytes": 32}, {"max_members": 1}, {"expected_sha256": "0" * 64}):
            with self.subTest(kwargs=kwargs), TemporaryDirectory() as temp:
                with self.assertRaises(ArchiveError):
                    extract_snapshot(data, Path(temp).resolve() / "snapshot", **kwargs)
        with TemporaryDirectory() as temp:
            self.assertEqual(extract_snapshot(data, Path(temp).resolve() / "snapshot", expected_sha256=sha256(data)), sha256(data))

    def test_existing_destinations_and_symlink_parents_rejected(self):
        with TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            existing = parent / "existing"
            existing.mkdir()
            (existing / "sentinel").write_text("keep")
            with self.assertRaises(ArchiveError):
                extract_snapshot(archive_bytes(), existing)
            link = parent / "link"
            link.symlink_to(existing, target_is_directory=True)
            with self.assertRaises(ArchiveError):
                extract_snapshot(archive_bytes(), link / "new-parent" / "snapshot")
            self.assertEqual([x.name for x in existing.iterdir()], ["sentinel"])

    def test_sha256_known_vector(self):
        self.assertEqual(sha256(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
