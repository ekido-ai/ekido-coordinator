"""Content-addressed file store. A blob's name is its hash, so a stored file can't be swapped later."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterable, Iterator

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DAY_PATTERN = re.compile(r"^\d{8}$")
MAX_BLOB_BYTES = 512 * 1024 * 1024
# Bytes one provider may upload that no committed checkpoint references yet. Blobs land before their manifest
# commits, so a little is always pending. Without a cap, any session token could fill the disk.
MAX_PENDING_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


class ArtifactError(ValueError):
    pass


class UploadBudgetExceeded(ArtifactError):
    pass


class ArtifactStore:
    def __init__(self, root: Path, max_bytes: int = MAX_BLOB_BYTES):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, digest: str) -> Path:
        # fullmatch, not match: with match, "$" also accepts a trailing newline. Only 64 hex digits reach the
        # filesystem, and the containment check below holds even if that ever changes.
        if not SHA256_PATTERN.fullmatch(digest):
            raise ArtifactError("not a lowercase SHA-256")
        # Defence in depth: the pattern already keeps names safe, and this holds even if it ever changes.
        root = os.path.realpath(self.root)
        prefix = root if root.endswith(os.sep) else root + os.sep  # a store at "/" must not become "//"
        path = os.path.normpath(os.path.join(root, digest[:2], digest))
        if not path.startswith(prefix):
            raise ArtifactError("artifact path escapes the store")
        return Path(path)

    def exists(self, digest: str) -> bool:
        return self.path_for(digest).is_file()

    @property
    def quarantine_root(self) -> Path:
        """Beside the store, on the same disk, so moving a blob in or out is an atomic rename."""
        return self.root.parent / f"{self.root.name}-quarantine"

    def iter_digests(self) -> Iterator[str]:
        """Every stored blob. Temp files from uploads in progress are skipped by name."""
        if not self.root.is_dir():
            return
        for shard in self.root.iterdir():
            if not shard.is_dir() or len(shard.name) != 2:
                continue
            for entry in shard.iterdir():
                if entry.is_file() and SHA256_PATTERN.fullmatch(entry.name) and entry.name[:2] == shard.name:
                    yield entry.name

    def modified_at(self, digest: str) -> datetime:
        return datetime.fromtimestamp(self.path_for(digest).stat().st_mtime, tz=timezone.utc)

    def size_of(self, digest: str) -> int:
        return self.path_for(digest).stat().st_size

    def quarantine(self, digest: str, now: datetime) -> Path:
        """Moves a blob out of the store into a dated folder. Nothing is deleted here, so it can be put back."""
        source = self.path_for(digest)
        target_dir = self.quarantine_root / now.strftime("%Y%m%d")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / digest
        os.replace(source, target)
        return target

    def restore(self, digest: str) -> bool:
        """Puts a quarantined blob back. Returns False if no quarantined copy exists."""
        if not self.quarantine_root.is_dir():
            return False
        for day in sorted(self.quarantine_root.iterdir(), reverse=True):
            candidate = day / digest
            if SHA256_PATTERN.fullmatch(digest) and candidate.is_file():
                target = self.path_for(digest)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(candidate, target)
                return True
        return False

    def purge_quarantine(self, older_than: timedelta, now: datetime) -> list[str]:
        """Permanently removes quarantine folders older than the retention period. Returns the folders removed."""
        removed = []
        if not self.quarantine_root.is_dir():
            return removed
        cutoff = (now - older_than).strftime("%Y%m%d")
        for day in self.quarantine_root.iterdir():
            if day.is_dir() and DAY_PATTERN.fullmatch(day.name) and day.name < cutoff:
                shutil.rmtree(day)
                removed.append(day.name)
        return sorted(removed)

    async def put(self, digest: str, chunks: AsyncIterable[bytes]) -> int:
        """Writes to a temp file and renames only once the hash matches, so a partial upload is never visible."""
        target = self.path_for(digest)
        if target.is_file():
            async for _ in chunks:  # Identical content is already stored. Drain the request and succeed.
                pass
            # Mark it as just uploaded: a new checkpoint is about to reference it, so retention must not move it
            # in the window before that commit lands.
            os.utime(target)
            return target.stat().st_size
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.parent / f".{uuid.uuid4().hex}.tmp"
        hasher = hashlib.sha256()
        size = 0
        try:
            with open(temp, "wb") as handle:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise ArtifactError(f"upload is over the {self.max_bytes} byte limit")
                    hasher.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if hasher.hexdigest() != digest:
                raise ArtifactError("content doesn't match its hash")
            os.replace(temp, target)
            return size
        finally:
            temp.unlink(missing_ok=True)


class UploadBudget:
    """Caps the bytes each provider has uploaded that no committed checkpoint references yet.

    In memory: a coordinator restart starts every budget at zero, which only ever errs towards allowing.
    A provider's pending bytes clear when one of its checkpoints commits, because that commit references them.
    """

    def __init__(self, limit: int = MAX_PENDING_UPLOAD_BYTES, window_seconds: float = 24 * 3600):
        self.limit = limit
        self.window = window_seconds
        self._pending: dict[str, dict[str, tuple[int, float]]] = {}
        self._lock = threading.Lock()

    def _live(self, owner: str, now: float) -> dict[str, tuple[int, float]]:
        entries = self._pending.setdefault(owner, {})
        for digest in [d for d, (_, at) in entries.items() if now - at > self.window]:
            del entries[digest]
        return entries

    def check(self, owner: str, size: int, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            pending = sum(size for size, _ in self._live(owner, now).values())
            if pending + size > self.limit:
                raise UploadBudgetExceeded(f"too many uploads waiting for a checkpoint commit ({pending + size} of {self.limit} bytes)")

    def record(self, owner: str, digest: str, size: int, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._live(owner, now)[digest] = (size, now)

    def release(self, owner: str) -> None:
        with self._lock:
            self._pending.pop(owner, None)

    def pending_bytes(self, owner: str, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        with self._lock:
            return sum(size for size, _ in self._live(owner, now).values())
