"""A small chain indexer: copies the escrow's and registry's raw logs into the database, incrementally.

It exists so the dashboard can load the whole history in one request instead of scanning a million blocks
through rate-limited public RPCs. It stores logs exactly as the RPC returned them; the browser decodes them
itself and checks the result against the contracts' own totals, falling back to a full scan if they disagree.
So the index makes the page faster without asking anyone to trust it.

Only blocks with CONFIRMATIONS on top are indexed, and inserts are idempotent, so restarts and overlaps are safe.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ChainLog, IndexCursor

log = logging.getLogger("gpuchain.chain_index")
CONFIRMATIONS = 5


class RpcLogs:
    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url
        self.timeout = timeout

    def call(self, method: str, params: list) -> Any:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"content-type": "application/json", "user-agent": "gpuchain-indexer/1"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            reply = json.loads(r.read())
        if "error" in reply:
            raise RuntimeError(str(reply["error"].get("message", reply["error"])))
        return reply["result"]


class ChainIndexer:
    def __init__(self, engine, rpc, addresses: list[str], from_block: int, step: int = 50_000, name: str = "escrow+registry") -> None:
        self.engine = engine
        self.rpc = rpc
        self.addresses = [a.lower() for a in addresses]
        self.from_block = from_block
        self.step = step
        self.name = name
        self._stop = threading.Event()

    def cursor(self, session: Session) -> int:
        row = session.get(IndexCursor, self.name)
        return row.block if row else self.from_block - 1

    def sync_once(self) -> int:
        """Indexes up to `step` blocks. Returns how many blocks it advanced (0 when caught up)."""
        latest = int(self.rpc.call("eth_blockNumber", []), 16) - CONFIRMATIONS
        with Session(self.engine) as session:
            start = self.cursor(session) + 1
            if start > latest:
                return 0
            end = min(start + self.step - 1, latest)
            logs = self.rpc.call("eth_getLogs", [{"address": self.addresses, "fromBlock": hex(start), "toBlock": hex(end)}])
            for item in logs:
                block, index = int(item["blockNumber"], 16), int(item["logIndex"], 16)
                key = f"{block}:{index}"
                if session.get(ChainLog, key) is None:
                    session.add(ChainLog(key=key, block_number=block, log_index=index, tx_hash=item["transactionHash"], address=item["address"].lower(), topics=item["topics"], data=item["data"]))
            row = session.get(IndexCursor, self.name)
            if row is None:
                session.add(IndexCursor(name=self.name, block=end))
            else:
                row.block = end
            session.commit()
            return end - start + 1

    def run_forever(self, idle: float = 15.0) -> None:
        while not self._stop.is_set():
            try:
                advanced = self.sync_once()
            except Exception:
                log.exception("chain index sync failed; retrying")
                advanced = 0
            # Fewer blocks than a full step means we reached the head. A new block arrives every ~2 s, so without
            # this pause the loop would chase each one and call the RPC several times a second, forever.
            if advanced < self.step:
                self._stop.wait(idle)

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_forever, daemon=True, name="chain-index")
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()


def logs_since(session: Session, name: str, from_block: int, limit: int) -> dict:
    rows = session.scalars(select(ChainLog).where(ChainLog.block_number >= from_block).order_by(ChainLog.block_number, ChainLog.log_index).limit(limit))
    cursor = session.get(IndexCursor, name)
    return {
        "synced_to": cursor.block if cursor else None,
        "logs": [
            {"address": r.address, "topics": r.topics, "data": r.data, "blockNumber": hex(r.block_number), "logIndex": hex(r.log_index), "transactionHash": r.tx_hash}
            for r in rows
        ],
    }
