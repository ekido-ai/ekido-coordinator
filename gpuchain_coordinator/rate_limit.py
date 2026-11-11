"""Per-client rate limits on every public route: token buckets, one for reads and one for writes.

Behind our Caddy every request arrives from loopback with X-Forwarded-For, so the client is the first forwarded
address. A loopback request without that header comes from this machine (the operator, the verifiers, the
monitor) and is not limited. Over the limit, the answer is 429 with Retry-After.

State is in memory, which is right for one coordinator. With several coordinators (docs/system-design.md,
Phase B) the same interface moves to a shared store.
"""

from __future__ import annotations

import ipaddress
import json
import math
import time
from dataclasses import dataclass

from starlette.types import ASGIApp, Receive, Scope, Send

READ_METHODS = {"GET", "HEAD", "OPTIONS"}
MAX_TRACKED = 50_000


@dataclass
class Bucket:
    tokens: float
    updated: float


def _loopback(host: str | None) -> bool:
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def client_key(scope: Scope) -> str | None:
    """The address to limit, or None for this machine's own traffic."""
    client = scope.get("client")
    host = client[0] if client else None
    forwarded = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"x-forwarded-for"), None)
    if _loopback(host):
        if not forwarded:
            return None
        return forwarded.split(",")[0].strip()[:64] or None
    return host or "unknown"


class TokenBuckets:
    def __init__(self, per_minute: int, clock=time.monotonic) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.clock = clock
        self.buckets: dict[str, Bucket] = {}

    def take(self, key: str) -> float:
        """0 if allowed, else the seconds until a token is free."""
        now = self.clock()
        bucket = self.buckets.get(key)
        if bucket is None:
            if len(self.buckets) >= MAX_TRACKED:
                # Bounded memory: forget the quarter that went quiet longest.
                for stale in sorted(self.buckets, key=lambda k: self.buckets[k].updated)[: MAX_TRACKED // 4]:
                    del self.buckets[stale]
            bucket = self.buckets[key] = Bucket(self.capacity, now)
        bucket.tokens = min(self.capacity, bucket.tokens + (now - bucket.updated) * self.rate)
        bucket.updated = now
        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return 0.0
        return (1 - bucket.tokens) / self.rate


class RateLimiter:
    def __init__(self, app: ASGIApp, *, reads_per_minute: int, writes_per_minute: int, enabled: bool = True, clock=time.monotonic) -> None:
        self.app = app
        self.enabled = enabled
        self.reads = TokenBuckets(reads_per_minute, clock)
        self.writes = TokenBuckets(writes_per_minute, clock)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self.enabled or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        key = client_key(scope)
        if key is None:
            await self.app(scope, receive, send)
            return
        is_read = scope["type"] == "websocket" or scope.get("method", "GET") in READ_METHODS
        wait = (self.reads if is_read else self.writes).take(key)
        if wait <= 0:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1013})
            return
        retry = str(max(1, math.ceil(wait)))
        body = json.dumps({"error": "RATE_LIMITED", "detail": f"Too many requests. Try again in {retry} s."}).encode()
        await send({"type": "http.response.start", "status": 429, "headers": [(b"content-type", b"application/json"), (b"retry-after", retry.encode())]})
        await send({"type": "http.response.body", "body": body})
