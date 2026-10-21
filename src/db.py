"""SQLite persistence: positions, orders, trades, decisions, equity snapshots, broker state.

Everything the engine needs to resume after a crash/restart lives here.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from src.config import DB_PATH, DECISION_LOG

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY, qty REAL, entry REAL, stop REAL, target REAL, atr REAL,
    opened TEXT, kind TEXT, cost_tmn REAL, trailing INTEGER DEFAULT 0, decision_id TEXT);
CREATE TABLE IF NOT EXISTS orders (
    client_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, type TEXT, price REAL, qty REAL,
    tmn REAL, status TEXT, created TEXT, filled_qty REAL DEFAULT 0, avg_price REAL,
    purpose TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, kind TEXT, opened TEXT, closed TEXT,
    entry REAL, exit REAL, qty REAL, pnl_tmn REAL, pnl_pct REAL, exit_reason TEXT, decision_id TEXT);
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY, ts TEXT, symbol TEXT, action TEXT, confidence REAL, payload TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, equity REAL, cash REAL, exposure REAL, dd REAL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, msg TEXT);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class DB:
    def __init__(self, path=DB_PATH):
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(positions)")}
        for col, typ in (("hold_h", "REAL"), ("trail_act", "REAL"), ("trail_mult", "REAL")):
            if col not in cols:   # migrate DBs created before per-position exit profiles
                self.conn.execute(f"ALTER TABLE positions ADD COLUMN {col} {typ}")

    def x(self, sql: str, args: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, args)

    def rows(self, sql: str, args: tuple | dict = ()) -> list[dict]:
        return [dict(r) for r in self.x(sql, args).fetchall()]

    # ------------------------------------------------------------- kv
    def get(self, k: str, default: Any = None) -> Any:
        r = self.x("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else default

    def put(self, k: str, v: Any) -> None:
        self.x("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))

    # ------------------------------------------------------- positions
    def positions(self) -> dict[str, dict]:
        return {r["symbol"]: r for r in self.rows("SELECT * FROM positions")}

    def upsert_position(self, p: dict) -> None:
        cols = ",".join(p)
        self.x(f"INSERT OR REPLACE INTO positions({cols}) VALUES({','.join('?' * len(p))})", tuple(p.values()))

    def delete_position(self, symbol: str) -> None:
        self.x("DELETE FROM positions WHERE symbol=?", (symbol,))

    # ---------------------------------------------------------- orders
    def save_order(self, o: dict) -> None:
        cols = ",".join(o)
        self.x(f"INSERT OR REPLACE INTO orders({cols}) VALUES({','.join('?' * len(o))})", tuple(o.values()))

    def open_orders(self) -> list[dict]:
        return self.rows("SELECT * FROM orders WHERE status IN ('NEW','PARTIAL')")

    def update_order(self, client_id: str, **fields) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        self.x(f"UPDATE orders SET {sets} WHERE client_id=?", (*fields.values(), client_id))

    # ------------------------------------------------- trades/decisions
    def add_trade(self, t: dict) -> None:
        cols = ",".join(t)
        self.x(f"INSERT INTO trades({cols}) VALUES({','.join('?' * len(t))})", tuple(t.values()))

    def add_decision(self, d: dict) -> None:
        """Persist to SQLite and append to logs/decision_logs.jsonl (the judges' artifact)."""
        self.x("INSERT OR REPLACE INTO decisions(id,ts,symbol,action,confidence,payload) VALUES(?,?,?,?,?,?)",
               (d["decision_id"], d["timestamp"], d["symbol"], d["action"], d.get("confidence"),
                json.dumps(d, ensure_ascii=False)))
        with open(DECISION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def snapshot_equity(self, equity: float, cash: float, exposure: float, dd: float) -> None:
        self.x("INSERT OR REPLACE INTO equity VALUES(?,?,?,?,?)", (now_iso(), equity, cash, exposure, dd))

    def event(self, level: str, msg: str) -> None:
        self.x("INSERT INTO events(ts,level,msg) VALUES(?,?,?)", (now_iso(), level, msg[:1000]))
