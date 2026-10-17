"""Bounded extraction of untrusted GitHub tar.gz snapshots; never follows links."""

import gzip
import hashlib
import io
from pathlib import Path
import shutil
import tarfile
import tempfile
import zlib

from reviewer.schemas import repository_path


class ArchiveError(ValueError):
    """Unsafe, malformed, oversized, or hash-mismatched snapshot archive."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_snapshot(data: bytes, destination: Path, *, expected_sha256: str | None = None,
                     max_bytes: int = 256 * 1024 * 1024, max_members: int = 100_000) -> str:
    """Extract into a new destination, atomically on success; return raw archive hash.

    Accept one wrapper directory, then regular files/directories only. Reject every
    link, device, FIFO, duplicate path, sparse member, and noncanonical pathname.
    Permissions/owners from the archive are never applied. Bounds include tar padding
    and metadata via a bounded gzip decompression before any extraction.
    """
    if type(max_bytes) is not int or max_bytes <= 0 or type(max_members) is not int or max_members <= 0:
        raise ValueError("archive limits must be positive integers")
    digest = sha256(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ArchiveError("snapshot SHA-256 mismatch")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ArchiveError("snapshot destination must not already exist")
    if any(p.is_symlink() for p in (destination.parent, *destination.parent.parents)):
        raise ArchiveError("snapshot parent must not contain symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=destination.parent))
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
            unpacked = compressed.read(max_bytes + 1)
        if len(unpacked) > max_bytes:
            raise ArchiveError("snapshot exceeds uncompressed byte limit")
        seen = set()
        root = None
        total = 0
        with tarfile.open(fileobj=io.BytesIO(unpacked), mode="r:") as archive:
            for count, member in enumerate(archive, 1):
                if count > max_members:
                    raise ArchiveError("snapshot exceeds member limit")
                name = member.name[:-1] if member.isdir() and member.name.endswith("/") else member.name
                try:
                    repository_path(name)
                except ValueError:
                    raise ArchiveError("unsafe snapshot path") from None
                if name in seen:
                    raise ArchiveError("duplicate snapshot path")
                seen.add(name)
                parts = name.split("/")
                if root is None:
                    root = parts[0]
                if parts[0] != root:
                    raise ArchiveError("snapshot must contain one wrapper directory")
                if not (member.isdir() or member.isreg()) or member.issparse():
                    raise ArchiveError("snapshot links and special files are unsupported")
                if len(parts) == 1:
                    if not member.isdir():
                        raise ArchiveError("snapshot wrapper must be a directory")
                    continue
                total += member.size
                if member.size < 0 or total > max_bytes:
                    raise ArchiveError("snapshot exceeds extracted byte limit")
                target = staging.joinpath(*parts[1:])
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise ArchiveError("snapshot member has no content")
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    if target.stat().st_size != member.size:
                        raise ArchiveError("truncated snapshot member")
        if not seen:
            raise ArchiveError("empty or malformed snapshot archive")
        staging.rename(destination)
        return digest
    except ArchiveError:
        raise
    except (OSError, EOFError, tarfile.TarError, ValueError, zlib.error):
        raise ArchiveError("malformed or conflicting snapshot archive") from None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
