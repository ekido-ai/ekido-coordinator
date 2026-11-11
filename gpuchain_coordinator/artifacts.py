"""Content-addressed file store. A blob's name is its hash, so a stored file can't be swapped later."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import AsyncIterable

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_BLOB_BYTES = 512 * 1024 * 1024


class ArtifactError(ValueError):
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

    async def put(self, digest: str, chunks: AsyncIterable[bytes]) -> int:
        """Writes to a temp file and renames only once the hash matches, so a partial upload is never visible."""
        target = self.path_for(digest)
        if target.is_file():
            async for _ in chunks:  # Identical content is already stored. Drain the request and succeed.
                pass
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
