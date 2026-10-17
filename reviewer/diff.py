"""Strict, deterministic parser for ordinary Git/plain unified text diffs.

No filesystem or network access. Unknown/binary/combined formats fail explicitly.
A syntactically complete patch cannot prove that an upstream producer supplied all
files or hunks; acquisition must separately attest completeness in ReviewInput.
"""

from dataclasses import dataclass, replace
import re
from typing import Literal

from reviewer.schemas import repository_path


class DiffParseError(ValueError):
    """Malformed, inconsistent, or visibly incomplete unified diff."""


class UnsupportedDiffError(DiffParseError):
    """Recognized format outside the text-diff contract."""


@dataclass(frozen=True)
class DiffLine:
    kind: Literal["context", "added", "deleted"]
    content: str  # Prefix and LF removed; other whitespace is preserved.
    old_line: int | None
    new_line: int | None
    no_newline: bool = False


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str
    lines: tuple[DiffLine, ...]


@dataclass(frozen=True)
class FileDiff:
    old_path: str | None
    new_path: str | None
    status: Literal["modified", "added", "deleted", "renamed"]
    hunks: tuple[Hunk, ...]
    old_mode: str | None = None
    new_mode: str | None = None

    def old_to_new(self, line: int) -> int | None:
        """Map a represented context line; deletion or absent context returns None.

        Gaps outside hunks are deliberately not extrapolated without full files.
        """
        if type(line) is not int or line < 1:
            raise ValueError("line must be a positive integer")
        return next((item.new_line for hunk in self.hunks for item in hunk.lines
                     if item.old_line == line), None)

    def new_to_old(self, line: int) -> int | None:
        """Map a represented context line; additions/absent context return None."""
        if type(line) is not int or line < 1:
            raise ValueError("line must be a positive integer")
        return next((item.old_line for hunk in self.hunks for item in hunk.lines
                     if item.new_line == line), None)


_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: (.*))?")
_MODE = r"[0-7]{6}"
_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13,
            "\\": 92, '"': 34}


def _unquote(raw: str) -> str:
    if not raw.startswith('"'):
        return raw
    if not raw.endswith('"') or len(raw) < 2:
        raise DiffParseError("unterminated quoted path")
    data = bytearray()
    body = raw[1:-1]
    i = 0
    while i < len(body):
        char = body[i]
        if char == '"':
            raise DiffParseError("unescaped quote in quoted path")
        if char != "\\":
            data.extend(char.encode("utf-8"))
            i += 1
            continue
        i += 1
        if i == len(body):
            raise DiffParseError("incomplete path escape")
        if body[i] in _ESCAPES:
            data.append(_ESCAPES[body[i]])
            i += 1
        elif re.fullmatch(r"[0-3][0-7]{2}", body[i:i + 3]):
            data.append(int(body[i:i + 3], 8))
            i += 3
        else:
            raise DiffParseError("unsupported path escape")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedDiffError("non-UTF-8 paths are unsupported") from exc


def _path(raw: str, side: str | None = None) -> str | None:
    raw = _unquote(raw)
    if raw == "/dev/null":
        return None
    if side and raw.startswith(side + "/"):
        raw = raw[2:]
    try:
        return repository_path(raw)
    except ValueError as exc:
        raise DiffParseError(str(exc)) from exc


def _git_paths(line: str) -> tuple[str, str]:
    raw = line[len("diff --git "):]
    # Git leaves ordinary spaces unquoted; split at its b/ prefix. Ambiguity fails.
    match = re.fullmatch(r'("(?:[^"\\]|\\.)*"|a/.*?) ("(?:[^"\\]|\\.)*"|b/.*)', raw)
    if not match:
        raise DiffParseError("invalid diff --git header")
    a, b = (_unquote(value) for value in match.groups())
    if not a.startswith("a/") or not b.startswith("b/"):
        raise UnsupportedDiffError("Git headers must use a/ and b/ prefixes")
    old, new = _path(match[1], "a"), _path(match[2], "b")
    if old is None or new is None:
        raise DiffParseError("null path in diff --git header")
    return old, new


def parse_diff(text: str) -> tuple[FileDiff, ...]:
    """Parse a complete LF-delimited patch or raise; never return partial results.

    Empty input is an empty diff. Metadata-only renames/mode changes and empty
    added/deleted files are supported. Each payload line must be LF terminated;
    a source file lacking LF is represented by Git's no-newline marker.
    """
    if not isinstance(text, str):
        raise TypeError("diff must be a string")
    if not text:
        return ()
    if "\x00" in text:
        raise UnsupportedDiffError("NUL/binary content is unsupported")
    if not text.endswith("\n"):
        raise DiffParseError("unterminated patch line; diff may be truncated")
    lines = text[:-1].split("\n")
    result = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(("diff --cc ", "diff --combined ", "@@@")):
            raise UnsupportedDiffError("combined diffs are unsupported")
        git = lines[i].startswith("diff --git ")
        git_paths = _git_paths(lines[i]) if git else None
        if git:
            i += 1
        elif not lines[i].startswith("--- "):
            raise DiffParseError(f"expected file header at patch line {i + 1}")
        metadata = {}
        while git and i < len(lines) and not lines[i].startswith(("--- ", "diff --")):
            line = lines[i]
            if line.startswith(("Binary files ", "GIT binary patch", "copy from ", "copy to ")):
                raise UnsupportedDiffError("binary and copy diffs are unsupported")
            match = re.fullmatch(r"(old mode|new mode|new file mode|deleted file mode) (" + _MODE + r")", line)
            rename = re.fullmatch(r"(rename from|rename to) (.+)", line)
            similarity = re.fullmatch(r"(similarity index|dissimilarity index) (\d{1,3})%", line)
            index = re.fullmatch(r"index ([0-9a-f]+)\.\.([0-9a-f]+)(?: (" + _MODE + r"))?", line)
            if match:
                key, value = match.groups()
            elif rename:
                key, value = rename.groups()
                value = _path(value)
                if value is None:
                    raise DiffParseError("null rename path")
            elif similarity and int(similarity[2]) <= 100:
                key, value = similarity.groups()
            elif index:
                key, value = "index", index.groups()
            else:
                raise DiffParseError(f"unexpected metadata at patch line {i + 1}: {line!r}")
            if key in metadata:
                raise DiffParseError(f"duplicate metadata: {key}")
            metadata[key] = value
            i += 1
        has_headers = i < len(lines) and lines[i].startswith("--- ")
        if has_headers:
            old = _path(lines[i][4:].split("\t", 1)[0], "a")
            i += 1
            if i == len(lines) or not lines[i].startswith("+++ "):
                raise DiffParseError("missing new-file header")
            new = _path(lines[i][4:].split("\t", 1)[0], "b")
            i += 1
        elif git_paths:
            old, new = git_paths
            if "new file mode" in metadata:
                old = None
            if "deleted file mode" in metadata:
                new = None
        else:
            raise DiffParseError("missing file headers")
        if old is None and new is None:
            raise DiffParseError("both file paths are null")
        if git_paths and ((old is not None and old != git_paths[0])
                          or (new is not None and new != git_paths[1])):
            raise DiffParseError("file headers disagree with diff --git paths")
        renamed = "rename from" in metadata or "rename to" in metadata
        if renamed and (metadata.get("rename from") != old or metadata.get("rename to") != new):
            raise DiffParseError("incomplete or contradictory rename metadata")
        if git and old and new and old != new and not renamed:
            raise DiffParseError("changed Git paths require rename metadata")
        if ("new file mode" in metadata and old is not None
                or "deleted file mode" in metadata and new is not None
                or "new file mode" in metadata and "deleted file mode" in metadata):
            raise DiffParseError("contradictory file operation metadata")
        if ("old mode" in metadata) != ("new mode" in metadata):
            raise DiffParseError("mode change requires both old and new modes")
        if (old is None or new is None) and (renamed or "old mode" in metadata):
            raise DiffParseError("rename/mode change conflicts with addition/deletion")
        if renamed and old == new:
            raise DiffParseError("rename must change the file path")
        index = metadata.get("index")
        if index:
            old_null, new_null = set(index[0]) == {"0"}, set(index[1]) == {"0"}
            if old_null != (old is None) or new_null != (new is None):
                raise DiffParseError("index null hashes disagree with file operation")
        hunks = []
        old_end = new_end = 0
        eof_sides = set()
        while i < len(lines) and lines[i].startswith("@@"):
            match = _HUNK.fullmatch(lines[i])
            if not match:
                raise DiffParseError(f"invalid hunk header at patch line {i + 1}")
            os, oc, ns, nc = (int(match[1]), int(match[2] or 1), int(match[3]), int(match[4] or 1))
            if (oc and os == 0) or (nc and ns == 0) or not (oc or nc):
                raise DiffParseError("invalid hunk range")
            if old is None and (os != 0 or oc != 0) or new is None and (ns != 0 or nc != 0):
                raise DiffParseError("hunk consumes lines from a nonexistent file")
            # Empty ranges locate an insertion/deletion after start, not at start.
            old_before, new_before = os - bool(oc), ns - bool(nc)
            if old_before < old_end or new_before < new_end:
                raise DiffParseError("overlapping or out-of-order hunks")
            if old_before - old_end != new_before - new_end:
                raise DiffParseError("inconsistent unchanged gap between hunks")
            i += 1
            payload = []
            old_used = new_used = 0
            while old_used < oc or new_used < nc:
                if i == len(lines):
                    raise DiffParseError("truncated hunk")
                line = lines[i]
                if line == "\\ No newline at end of file":
                    _mark_no_newline(payload, eof_sides)
                    i += 1
                    continue
                if not line or line[0] not in " +-":
                    raise DiffParseError(f"invalid or truncated hunk at patch line {i + 1}")
                kind = {" ": "context", "+": "added", "-": "deleted"}[line[0]]
                uses_old, uses_new = kind != "added", kind != "deleted"
                if uses_old and "old" in eof_sides or uses_new and "new" in eof_sides:
                    raise DiffParseError("content appears after a source EOF marker")
                item = DiffLine(kind, line[1:], os + old_used if uses_old else None,
                                ns + new_used if uses_new else None)
                old_used += uses_old
                new_used += uses_new
                if old_used > oc or new_used > nc:
                    raise DiffParseError("hunk line counts exceeded")
                payload.append(item)
                i += 1
            if i < len(lines) and lines[i] == "\\ No newline at end of file":
                _mark_no_newline(payload, eof_sides)
                i += 1
            hunks.append(Hunk(os, oc, ns, nc, match[5] or "", tuple(payload)))
            old_end, new_end = old_before + oc, new_before + nc
        if has_headers and not hunks:
            raise DiffParseError("file headers without hunks; possibly truncated diff")
        if not hunks:
            if not (renamed or "old mode" in metadata or "new file mode" in metadata
                    or "deleted file mode" in metadata):
                raise DiffParseError("incomplete metadata-only diff")
            index = metadata.get("index")
            if old is not None and new is not None and index and index[0] != index[1]:
                raise DiffParseError("changed blob hashes without hunks; possibly truncated diff")
            if renamed and metadata.get("similarity index") != "100":
                raise DiffParseError("rename without hunks must have 100% similarity")
            if "new file mode" in metadata or "deleted file mode" in metadata:
                # Git's canonical empty blob (SHA-1). Without it, a missing hunk is ambiguous.
                index = metadata.get("index")
                empty_hash = index[1 if old is None else 0] if index else ""
                null_hash = index[0 if old is None else 1] if index else ""
                if (len(empty_hash) < 7 or not "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391".startswith(empty_hash)
                        or not null_hash or set(null_hash) != {"0"}):
                    raise DiffParseError("addition/deletion without hunks must identify an empty blob")
        status = "added" if old is None else "deleted" if new is None else "renamed" if old != new else "modified"
        result.append(FileDiff(old, new, status, tuple(hunks),
                               metadata.get("old mode", metadata.get("deleted file mode")),
                               metadata.get("new mode", metadata.get("new file mode"))))
        # The outer loop accepts only a new file header; surplus payload fails.
    old_paths = [file.old_path for file in result if file.old_path is not None]
    new_paths = [file.new_path for file in result if file.new_path is not None]
    if len(old_paths) != len(set(old_paths)) or len(new_paths) != len(set(new_paths)):
        raise DiffParseError("duplicate file sections on the same side")
    return tuple(result)


def _mark_no_newline(payload: list[DiffLine], eof_sides: set[str]) -> None:
    if not payload or payload[-1].no_newline:
        raise DiffParseError("orphan or duplicate no-newline marker")
    item = payload[-1]
    payload[-1] = replace(item, no_newline=True)
    if item.old_line is not None:
        eof_sides.add("old")
    if item.new_line is not None:
        eof_sides.add("new")
