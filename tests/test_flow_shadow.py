"""End-to-end SHADOW flow with live order books and stubbed LLM (no token cost).

Forces a marketable entry signal, then checks: graph -> limit order -> fill -> position ->
stop-loss exit -> trade + decision logs -> end-of-challenge liquidation.
Run: python -m tests.test_flow_shadow
"""
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path

# isolate all state BEFORE importing src modules (never touch the live bot's DB/logs)
tmp = Path(os.getenv("TEST_STATE_DIR") or tempfile.mkdtemp())
tmp.mkdir(parents=True, exist_ok=True)
os.environ["TRADER_DB"] = str(tmp / "t.db")
os.environ["TRADER_DECISION_LOG"] = str(tmp / "decisions.jsonl")
os.environ["TRADER_SCAN_LOG"] = str(tmp / "scan.jsonl")

import src.db as dbmod  # noqa: E402
import src.engine as eng  # noqa: E402
from src.agents import nodes  # noqa: E402
from src.strategy import Signal  # noqa: E402


def fake_call(state, role, system, user, model=None):
    out = {"bull": {"stance": "BUY", "confidence": 0.7, "argument": "تست: سیگنال معتبر"},
           "bear": {"veto": False, "severity": 0.2, "argument": "تست: ریسک پایین"},
           "arbiter": {"verdict": "APPROVE", "size_factor": 0.5, "argument": "تست"}}[role]
    return out, {"model": "stub", "cost": 0.0, "role": role}


nodes._call = fake_call
e = eng.Engine()
assert str(e.db.conn.execute("PRAGMA database_list").fetchone()[2]).endswith("t.db")
e.db.put("shadow_balances", {"TMN": 5_000_000})
e.db.put("peak_equity", 5_000_000)

ask = e.broker.book("BTCTMN")["ask"]
sig = Signal("BTCTMN", "oversold", ask * 1.001, ask * 0.985, ask * 1.03, 2.9, 0.55, "forced test signal")
eng.latest_signal = lambda sym, *a, **k: (sig, "ok") if sym == "BTCTMN" else (None, "no_setup")

e.scan()
pos = e.db.positions()
assert "BTCTMN" in pos, f"position not opened; orders={e.db.rows('SELECT * FROM orders')}"
print("1) opened:", {k: pos["BTCTMN"][k] for k in ("qty", "entry", "stop", "target")})
print("   balances:", e.broker.balances(), "| portfolio:", e.portfolio())

p = pos["BTCTMN"]
p["stop"] = p["entry"] * 1.05                 # force stop-loss on next tick
e.db.upsert_position(p)
e.tick()
assert not e.db.positions(), "stop-loss did not close the position"
tr = e.db.rows("SELECT * FROM trades")
print("2) closed:", {k: tr[-1][k] for k in ("exit_reason", "pnl_tmn", "pnl_pct")})

assert e.db.get("cooldown_BTCTMN"), "loss cooldown not set"
e.db.put("cooldown_BTCTMN", None)              # skip the post-loss cooldown for the test
ask = e.broker.book("BTCTMN")["ask"]            # fresh marketable signal
sig = Signal("BTCTMN", "oversold", ask * 1.001, ask * 0.985, ask * 1.03, 2.9, 0.55, "forced test signal")
e.scan()                                       # second entry, then liquidation
assert "BTCTMN" in e.db.positions(), "second entry failed"
e.end = eng.utcnow() + timedelta(minutes=10)
e.tick()
assert not e.db.positions() and e.db.get("liquidated"), "liquidation failed"
bal = e.broker.balances()
dust = {a: q for a, q in bal.items() if a != "TMN"}
assert all(q * e.broker.book(f"{a}TMN")["bid"] < 1000 for a, q in dust.items()), bal   # rounding dust only
print("3) liquidated -> balances:", bal)

decs = [json.loads(l) for l in open(dbmod.DECISION_LOG, encoding="utf-8")]
orders = e.db.rows("SELECT client_id FROM orders")
logged = {d["execution"]["client_order_id"] for d in decs if d.get("execution")}
assert all(o["client_id"] in logged for o in orders), "an order has no decision record"
assert len(orders) == 4, orders
print(f"4) {len(orders)} orders, {len(decs)} decisions; every order has a decision record:",
      [d["action"] for d in decs])
print("ALL OK")
