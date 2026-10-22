"""One-screen status report of the running engine (read-only).

Usage: python -m scripts.status [--hours 12]
"""
import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone

from src.config import DB_PATH, settings

TEHRAN = timezone(timedelta(hours=3, minutes=30))


def t(ts: str) -> str:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(TEHRAN).strftime("%m-%d %H:%M")


def main(hours: int) -> None:
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    kv = {r["k"]: json.loads(r["v"]) for r in db.execute("SELECT k, v FROM kv")}
    scan = kv.get("last_scan", {})
    pf = scan.get("portfolio", {})
    eq = pf.get("equity", 0)
    print(f"=== {settings.mode} | now {datetime.now(TEHRAN):%m-%d %H:%M} Tehran | last scan {t(scan['ts']) if scan else '-'}")
    print(f"equity {eq:,.0f} TMN ({(eq / settings.risk.initial_capital_tmn - 1) * 100:+.2f}% vs 5M) | "
          f"drawdown now {pf.get('dd_pct', 0):.2f}% | exposure {pf.get('exposure_pct', 0):.1f}%")
    conn = kv.get("conn", {})
    print(f"connection: {'OK' if conn.get('ok') else 'DOWN'} | last check {t(conn['last_check']) if conn.get('last_check') else '-'}"
          f" | outages total {conn.get('outages', 0)} | MCP reconcile {(scan.get('mcp') or {}).get('ok')}")

    print("\n--- open positions")
    for p in db.execute("SELECT * FROM positions"):
        print(f"{p['symbol']:8s} {p['kind']:15s} opened {t(p['opened'])} entry {p['entry']:,.0f} stop {p['stop']:,.0f} "
              f"target {p['target']:,.0f} trailing={bool(p['trailing'])}")

    print(f"\n--- closed trades (last {hours}h)")
    tot = 0.0
    for r in db.execute("SELECT * FROM trades WHERE closed >= ? ORDER BY closed", (since,)):
        tot += r["pnl_tmn"]
        print(f"{t(r['closed'])} {r['symbol']:8s} {r['kind'] or '':15s} {r['pnl_pct']:+6.2f}% {r['pnl_tmn']:+10,.0f} TMN  {r['exit_reason']}")
    print(f"realized in window: {tot:+,.0f} TMN")

    print(f"\n--- agent decisions (last {hours}h)")
    for r in db.execute("SELECT ts, symbol, action, payload FROM decisions WHERE ts >= ? ORDER BY ts", (since,)):
        d = json.loads(r["payload"])
        print(f"{t(r['ts'])} {r['symbol']:8s} {r['action']:6s} {d['setup']:18s} {d['agent_debate']['risk_assessment'][:70]}")

    print(f"\n--- warnings / events (last {hours}h)")
    for r in db.execute("SELECT ts, level, msg FROM events WHERE ts >= ? AND level != 'INFO' ORDER BY id", (since,)):
        print(f"{t(r['ts'])} {r['level']:6s} {r['msg'][:110]}")
    print(f"\nLLM: {scan.get('llm_budget')}")

    # whole-challenge summary (numbers for the demo video)
    tr = list(db.execute("SELECT pnl_tmn, pnl_pct FROM trades"))
    eqs = [r[0] for r in db.execute("SELECT equity FROM equity ORDER BY ts")]
    peak, max_dd = settings.risk.initial_capital_tmn, 0.0
    for e in eqs:
        peak = max(peak, e)
        max_dd = max(max_dd, 1 - e / peak)
    n_dec = db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    n_agent = sum(1 for (p,) in db.execute("SELECT payload FROM decisions")
                  if json.loads(p).get("setup") not in ("risk_rule", "connectivity_test", "INCIDENT_test_run"))
    print("\n=== SINCE START (for the video)")
    print(f"return {(eq / settings.risk.initial_capital_tmn - 1) * 100:+.2f}% | max drawdown {max_dd * 100:.2f}% | "
          f"closed trades {len(tr)} (wins {sum(1 for t in tr if t[0] > 0)}) | realized {sum(t[0] for t in tr):+,.0f} TMN")
    print(f"decision records {n_dec} (agent-debated {n_agent}) | connection outages {conn.get('outages', 0)} | "
          f"LLM cost ${scan.get('llm_budget', {}).get('total_usd', 0):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=12)
    main(ap.parse_args().hours)
