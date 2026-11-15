"""Who may use the operator API: this machine, or anyone holding API_TOKEN.

Without this, anyone on the same network could start jobs, cancel them or inject faults. Workers and
verifiers from other machines keep working, because their routes prove identity in their own way:
signed agent envelopes, short-lived artifact tokens, and signed verifier sessions.
"""

from __future__ import annotations

import ipaddress
import secrets

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# /v1/ checks sessions and API keys itself (v1_accounts.py), including the Origin of cookie-authenticated changes.
OWN_AUTH_PREFIXES = ("/agent/ws", "/artifacts/", "/verifier/", "/providers/join", "/v1/")
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
    def __init__(self, app: ASGIApp, *, token: str, allowed_origins: list[str], open_api: bool = False, public_read: bool = False) -> None:
        self.app = app
        self.token = token.encode()
        self.allowed_origins = set(allowed_origins)
        self.open_api = open_api
        # A public server lets anyone watch: GET requests and the job event stream. Changing anything still needs the token.
        self.public_read = public_read

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
        if self.public_read and (scope.get("method") in ("GET", "HEAD") or (scope["type"] == "websocket" and path.startswith("/ws/jobs/"))):
            return None

        scheme, _, presented = headers.get("authorization", "").partition(" ")
        if self.token and scheme.lower() == "bearer" and secrets.compare_digest(presented.encode(), self.token):
            return None

        # A web page can point its own domain at 127.0.0.1, so the Host header must be local too.
        client = scope.get("client")
        forwarded = "x-forwarded-for" in headers or "forwarded" in headers
        if client and _is_loopback_ip(client[0]) and not forwarded and _host_name(headers.get("host", "")) in LOOPBACK_NAMES:
            return None
        return 401, "This API only answers this machine. From another machine, send Authorization: Bearer with API_TOKEN."

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        refusal = self._refusal(scope)
        if refusal is None:
            await self.app(scope, receive, send)
        elif scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            await JSONResponse({"error": "API_LOCKED", "detail": refusal[1]}, status_code=refusal[0])(scope, receive, send)
