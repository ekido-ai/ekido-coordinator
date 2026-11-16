"""Who may use the operator API: this machine, or anyone holding API_TOKEN.

Without this, anyone on the same network could start jobs, cancel them or inject faults. Workers and
verifiers from other machines keep working, because their routes prove identity in their own way:
signed agent envelopes, short-lived artifact tokens, and signed verifier sessions.
"""

from __future__ import annotations

import ipaddress
import re
import secrets

import anyio

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .public_view import Redactor

OWN_AUTH_PREFIXES = ("/agent/ws", "/artifacts/", "/verifier/", "/providers/join")
# What a public server shows anyone as it is.
PUBLIC_READ_PATHS = frozenset({"/healthz", "/readyz", "/fees"})
# What a public server shows anyone with every provider identity replaced by a label (public_view.py): job pages and
# the job event stream, so the public demo works. The provider list and everything else stay behind the operator
# token: no public list or count of who is connected (decision 8).
PUBLIC_JOB_READ = re.compile(r"^/jobs(/[^/]+(/(timeline|checkpoints|settlement-preview|recoveries|settlement))?)?$")
PUBLIC_JOB_STREAM = re.compile(r"^/ws/jobs/[^/]+$")
LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}


def _is_loopback_ip(host: str | None) -> bool:
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    return address.is_loopback or bool(getattr(address, "ipv4_mapped", None) and address.ipv4_mapped.is_loopback)


def _host_name(value: str) -> str:
    if value.startswith("["):
        return value[1:].split("]", 1)[0].lower()
    return value.rsplit(":", 1)[0].lower() if ":" in value else value.lower()


class ApiGuard:
    def __init__(self, app: ASGIApp, *, token: str, allowed_origins: list[str], open_api: bool = False, public_read: bool = False, redactor: Redactor | None = None) -> None:
        self.app = app
        self.token = token.encode()
        self.allowed_origins = set(allowed_origins)
        self.open_api = open_api
        # A public server answers anyone only on PUBLIC_READ_PATHS. Everything else, reads included, needs the token.
        self.public_read = public_read
        # Without a redactor, job pages stay behind the token even on a public server.
        self.redactor = redactor

    def _public_job_view(self, scope: Scope) -> bool:
        if not (self.public_read and self.redactor is not None):
            return False
        path = scope["path"]
        if scope["type"] == "websocket":
            return bool(PUBLIC_JOB_STREAM.match(path))
        return scope.get("method") in ("GET", "HEAD") and bool(PUBLIC_JOB_READ.match(path))

    def _refusal(self, scope: Scope) -> tuple[int, str] | None:
        path = scope["path"]
        if path == "/healthz" or path.startswith(OWN_AUTH_PREFIXES):
            return None
        headers = {name.decode("latin-1"): value.decode("latin-1") for name, value in scope["headers"]}

        # Browsers don't apply CORS to WebSockets, and a form post needs no preflight, so the origin is checked here.
        origin = headers.get("origin")
        if origin is not None and origin not in self.allowed_origins:
            return 403, f"The origin {origin} isn't allowed to use this API."
        if self.open_api:
            return None
        if self.public_read and scope.get("method") in ("GET", "HEAD") and path in PUBLIC_READ_PATHS:
            return None

        scheme, _, presented = headers.get("authorization", "").partition(" ")
        if self.token and scheme.lower() == "bearer" and secrets.compare_digest(presented.encode(), self.token):
            return None

        # A web page can point its own domain at 127.0.0.1, so the Host header must be local too.
        client = scope.get("client")
        forwarded = "x-forwarded-for" in headers or "forwarded" in headers
        if client and _is_loopback_ip(client[0]) and not forwarded and _host_name(headers.get("host", "")) in LOOPBACK_NAMES:
            return None
        if self._public_job_view(scope):
            return "PUBLIC"  # allowed, through the redactor
        return 401, "This API only answers this machine. From another machine, send Authorization: Bearer with API_TOKEN."

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        refusal = self._refusal(scope)
        if refusal is None:
            await self.app(scope, receive, send)
        elif refusal == "PUBLIC":
            await anyio.to_thread.run_sync(self.redactor.refresh)
            await self.app(scope, receive, self._redacting(scope, send))
        elif scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            await JSONResponse({"error": "API_LOCKED", "detail": refusal[1]}, status_code=refusal[0])(scope, receive, send)

    def _redacting(self, scope: Scope, send: Send) -> Send:
        """Wraps send so JSON bodies and WebSocket messages leave without provider identities."""
        redactor = self.redactor
        if scope["type"] == "websocket":
            async def send_ws(message: Message) -> None:
                if message["type"] == "websocket.send":
                    if message.get("text") is not None:
                        message = {**message, "text": redactor.scrub_text(message["text"])}
                    elif message.get("bytes") is not None:  # never pass raw bytes through unchecked
                        message = {**message, "bytes": None, "text": redactor.scrub_text(message["bytes"].decode("utf-8", "replace"))}
                await send(message)
            return send_ws

        start: dict[str, Message] = {}
        chunks: list[bytes] = []

        async def send_http(message: Message) -> None:
            if message["type"] == "http.response.start":
                start["message"] = message
                return
            if message["type"] != "http.response.body":
                await send(message)
                return
            chunks.append(message.get("body", b""))
            if message.get("more_body"):
                return
            head = start["message"]
            body = b"".join(chunks)
            content_type = dict(head.get("headers", [])).get(b"content-type", b"")
            if not body:  # HEAD, 204 and the like: nothing to hide
                await send(head)
                await send({"type": "http.response.body", "body": b""})
                return
            if content_type.startswith(b"application/json"):
                try:
                    body = redactor.scrub_json_bytes(body)
                except ValueError:
                    body = b'{"error":"HIDDEN","detail":"This response could not be shown publicly."}'
            else:  # anything that isn't JSON is not shown publicly
                body = b'{"error":"HIDDEN","detail":"This response could not be shown publicly."}'
                head = {**head, "headers": [(k, v) for k, v in head.get("headers", []) if k != b"content-type"] + [(b"content-type", b"application/json")]}
            headers = [(k, v) for k, v in head.get("headers", []) if k != b"content-length"] + [(b"content-length", str(len(body)).encode())]
            await send({**head, "headers": headers})
            await send({"type": "http.response.body", "body": body})
        return send_http
