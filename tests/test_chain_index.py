import unittest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from ekido_coordinator.chain_index import ChainIndexer, logs_since
from ekido_coordinator.main import Settings, create_app
from ekido_coordinator.models import Base

ESCROW = "0x1BF65f755457a92850009c88AFb337Ec57745a22"


def raw(block, index, tx="0x" + "a" * 64):
    return {"address": ESCROW, "topics": ["0x" + "1" * 64], "data": "0x", "blockNumber": hex(block), "logIndex": hex(index), "transactionHash": tx}


class FakeRpc:
    def __init__(self, head, logs):
        self.head, self.logs, self.calls = head, logs, []

    def call(self, method, params):
        self.calls.append((method, params))
        if method == "eth_blockNumber":
            return hex(self.head)
        q = params[0]
        lo, hi = int(q["fromBlock"], 16), int(q["toBlock"], 16)
        return [l for l in self.logs if lo <= int(l["blockNumber"], 16) <= hi]


def engine():
    e = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(e)
    return e


class ChainIndexTest(unittest.TestCase):
    def test_backfills_in_steps_and_stops_at_confirmations(self):
        e = engine()
        rpc = FakeRpc(head=1000, logs=[raw(105, 0), raw(450, 2), raw(990, 1), raw(998, 0)])
        ix = ChainIndexer(e, rpc, [ESCROW], from_block=100, step=300)
        while ix.sync_once():
            pass
        from sqlalchemy.orm import Session

        with Session(e) as s:
            out = logs_since(s, "escrow+registry", 0, 100)
        # Head 1000 minus 5 confirmations: block 998 is not indexed yet.
        self.assertEqual(out["synced_to"], 995)
        self.assertEqual([int(l["blockNumber"], 16) for l in out["logs"]], [105, 450, 990])
        self.assertEqual(len([c for c in rpc.calls if c[0] == "eth_getLogs"]), 3)

    def test_restarts_and_overlaps_never_duplicate(self):
        e = engine()
        rpc = FakeRpc(head=200, logs=[raw(150, 0)])
        ChainIndexer(e, rpc, [ESCROW], from_block=100).sync_once()
        rpc.head = 400
        rpc.logs.append(raw(300, 1))
        ix = ChainIndexer(e, rpc, [ESCROW], from_block=100)
        ix.sync_once()
        ix.sync_once()
        from sqlalchemy.orm import Session

        with Session(e) as s:
            logs = logs_since(s, "escrow+registry", 0, 100)["logs"]
        self.assertEqual([int(l["blockNumber"], 16) for l in logs], [150, 300])

    def test_rests_at_the_head_instead_of_chasing_each_new_block(self):
        e = engine()
        rpc = FakeRpc(head=1000, logs=[])
        ix = ChainIndexer(e, rpc, [ESCROW], from_block=900, step=50)
        waits = []
        ix._stop.wait = lambda seconds: (waits.append(seconds), rpc.__setattr__("head", rpc.head + 1), len(waits) >= 3 and ix.stop())[0]
        ix.run_forever(idle=15)
        # Two full steps to catch up, then a rest after every short sync, never a tight loop.
        self.assertEqual(waits, [15, 15, 15])
        self.assertLessEqual(len([c for c in rpc.calls if c[0] == "eth_getLogs"]), 6)

    def test_public_endpoint_serves_raw_logs(self):
        app = create_app(Settings(database_url="sqlite://", _env_file=None))
        r = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 1)).get("/chain/logs?from_block=0")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"synced_to": None, "logs": []})


if __name__ == "__main__":
    unittest.main()
