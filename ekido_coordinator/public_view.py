"""What a public visitor sees of a job: everything except who the providers are.

Decision 8 (16 Nov): no public list or count of who is connected. Job pages still show progress, recoveries and
settlement to anyone, so the public demo works. For a visitor without the operator token, every provider identity
in a response, whether a provider id, its name, its wallet, a signing key or a worker key, is replaced by a stable
label such as "provider-3fa2c1". It is replaced as a value, as a dict key, or inside free text.

Labels come from a keyed hash of the provider id, so they stay the same across responses (a recovery still reads
"provider-3fa2c1 dropped, provider-9b04de took over") but can't be reversed by hashing known ids, and they don't
reveal how many providers exist. Values under identity keys that match no known provider are hidden the same way,
so a provider removed since, or a raw address, never shows through.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from collections.abc import Callable
from typing import Any

# Keys whose value always identifies a provider, even if it matches nothing known.
IDENTITY_KEYS = frozenset({
    "provider_id", "target_provider_id", "provider", "provider_name", "signer", "expected_signer",
    "wallet_address", "worker_key", "worker_id", "enrolled_key", "session_pubkey", "payees",
})
# Not "name": recipes, models and datasets have names too. Provider names are caught as known identities instead.
# Inside these containers a provider signed something; anyone can recover the signer's key from a signature, so
# signatures there are hidden too (a provider's quote in a job's provider_terms).
PROVIDER_SIGNED = frozenset({"provider_terms"})
SIGNATURE_KEYS = frozenset({"signature", "sig", "quote_signature"})
# Identity strings shorter than this are matched only as whole values, never inside text, to avoid false hits.
MIN_TEXT_MATCH = 4


class Redactor:
    """Holds the current identities (refreshed at most every `ttl` seconds) and scrubs JSON-like data."""

    def __init__(self, load: Callable[[], dict[str, str]], *, secret: str = "", ttl: float = 30.0) -> None:
        # load() returns {identity string: provider id}; the label is derived from the provider id.
        self._load = load
        self._secret = (secret or secrets.token_hex(32)).encode()
        self._ttl = ttl
        self._lock = threading.Lock()
        self._loaded_at = 0.0
        self._labels: dict[str, str] = {}
        self._pattern: re.Pattern[str] | None = None

    def label(self, value: str) -> str:
        return "provider-" + hmac.new(self._secret, value.lower().encode(), hashlib.sha256).hexdigest()[:6]

    def refresh(self, force: bool = False) -> None:
        with self._lock:
            if not force and time.monotonic() - self._loaded_at < self._ttl:
                return
            identities = self._load()
            labels = {k.lower(): self.label(pid) for k, pid in identities.items() if k}
            long = sorted((k for k in labels if len(k) >= MIN_TEXT_MATCH), key=len, reverse=True)
            self._pattern = re.compile("(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, long)) + ")(?![A-Za-z0-9_])", re.I) if long else None
            self._labels = labels
            self._loaded_at = time.monotonic()

    def _text(self, value: str) -> str:
        known = self._labels.get(value.lower())
        if known:
            return known
        if self._pattern is None:
            return value
        return self._pattern.sub(lambda m: self._labels[m.group(1).lower()], value)

    def _forced(self, value: Any) -> Any:
        """A value under an identity key: always hidden, known or not."""
        if isinstance(value, str):
            return self._labels.get(value.lower()) or self.label(value)
        if isinstance(value, list):
            return [self._forced(v) for v in value]
        if isinstance(value, dict):  # e.g. "provider": {...}: scrub it field by field, keeping non-identity fields
            return self.scrub(value)
        return value

    def _key(self, key: Any) -> Any:
        return self._text(key) if isinstance(key, str) else key

    def scrub(self, data: Any, signed: bool = False) -> Any:
        if isinstance(data, dict):
            out = {}
            for k, v in data.items():
                if k in IDENTITY_KEYS or (signed and k in SIGNATURE_KEYS):
                    out[self._key(k)] = self._forced(v)
                else:
                    out[self._key(k)] = self.scrub(v, signed or k in PROVIDER_SIGNED)
            return out
        if isinstance(data, list):
            return [self.scrub(v, signed) for v in data]
        if isinstance(data, str):
            return self._text(data)
        return data

    def scrub_json_bytes(self, body: bytes) -> bytes:
        return json.dumps(self.scrub(json.loads(body)), separators=(",", ":")).encode()

    def scrub_text(self, text: str) -> str:
        try:
            return json.dumps(self.scrub(json.loads(text)), separators=(",", ":"))
        except ValueError:
            return self._text(text)


def load_identities(engine) -> dict[str, str]:
    """Every string that identifies a provider, mapped to that provider's id."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from .models import Provider, ProviderKey, ProviderKeyProof, Wallet, Worker

    out: dict[str, str] = {}
    with Session(engine) as session:
        for p, address in session.execute(select(Provider, Wallet.address).join(Wallet, Wallet.id == Provider.wallet_id, isouter=True)):
            for value in (p.id, p.name, p.enrolled_key, address):
                if value:
                    out[value] = p.id
        for model in (ProviderKey, ProviderKeyProof):
            for key, pid in session.execute(select(model.key, model.provider_id)):
                out[key] = pid
        for wid, pubkey, pid in session.execute(select(Worker.id, Worker.session_pubkey, Worker.provider_id)):
            out[wid] = pid
            if pubkey:
                out[pubkey] = pid
    return out
