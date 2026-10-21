"""Point-in-time replay of the REAL agent graph on recent historical signals.

For every signal the live engine would have produced in the last days:
  1. the agents (real LLM calls) see only data available at that moment and decide BUY/SKIP + size;
  2. the actual outcome is simulated on the bars that followed (limit fill, stop/target/trailing/time,
     maker fee on entry, taker fee + half-spread on exit);
  3. we measure whether the agents' decisions beat "take every signal".

Usage: python -m scripts.agent_replay [--days 3] [--min-signals 20]
Outputs: reports/agent_replay.csv, reports/agent_replay_decisions.jsonl
"""
from __future__ import annotations

import argparse
import json
import warnings

import pandas as pd

from src.agents.graph import get_graph
from src.config import ROOT, settings
from src.llm.client import get_llm
from src.market_data import add_indicators, load_history, resample, snapshot
from src.strategy import HALF_SPREAD, SETUP_BASE_RATES, Signal, StrategyParams, ensemble_signals, usd_frame

warnings.filterwarnings("ignore")
OUT = ROOT / "reports"
CAPITAL = settings.risk.initial_capital_tmn
MAKER, TAKER = settings.fees.maker, settings.fees.taker


def build_frames():
    syms = list(settings.symbols)
    m1 = {s: load_history(s) for s in syms + ["USDTTMN"]}
    u15, u60, u240 = (add_indicators(resample(m1["USDTTMN"], tf)) for tf in ("15min", "60min", "240min"))
    px15 = {s: add_indicators(resample(m1[s], "15min")) for s in syms}
    sig15 = {s: usd_frame(px15[s], u15) for s in syms}
    sig60 = {s: usd_frame(add_indicators(resample(m1[s], "60min")), u60) for s in syms}
    sig240 = {s: usd_frame(add_indicators(resample(m1[s], "240min")), u240) for s in syms}
    return px15, sig15, sig60, sig240, u15


def collect_signals(px15, sig15, sig60, sig240, u15, start, end):
    out = []
    for T in pd.date_range(start, end, freq="15min"):
        p15 = {s: f[f.index < T] for s, f in px15.items()}
        s15 = {s: f[f.index < T] for s, f in sig15.items()}
        s60 = {s: f[f.index + pd.Timedelta("60min") <= T] for s, f in sig60.items()}
        s240 = {s: f[f.index + pd.Timedelta("240min") <= T] for s, f in sig240.items()}
        u = u15[u15.index < T]
        sigs, diag = ensemble_signals(p15, s15, s60, u, T + pd.Timedelta(seconds=20), sig240=s240)
        usdt_1h = round(float(u["close"].iloc[-1] / u["close"].iloc[-5] - 1) * 100, 3)
        for x in sigs:
            out.append((T, x, diag[x.symbol], s15[x.symbol], usdt_1h))
    return out


def outcome(px: pd.DataFrame, s: Signal, T: pd.Timestamp) -> dict:
    """Simulate the live engine's handling of this signal on the bars that followed."""
    after = px[px.index >= T]
    if not len(after):
        return {"filled": False}
    first = after.iloc[0]
    if first["low"] > s.price:                     # limit not reached within one bar -> cancelled
        return {"filled": False}
    fill = min(first["open"], s.price)
    scale = fill / s.price
    stop, target, atr = s.stop * scale, s.target * scale, s.atr_tmn
    hs = HALF_SPREAD.get(s.symbol, 0.001)
    rt_cost = MAKER + TAKER + hs
    trailing = False
    max_bars = int(s.hold_h * 4)
    exit_px, reason, bars = None, "open", 0
    for i, (t, b) in enumerate(after.iterrows()):
        bars = i + 1
        if b["low"] <= stop:
            exit_px, reason = min(b["open"], stop) * (1 - hs), "trailing_stop" if trailing else "stop_loss"
            break
        if b["high"] >= target:
            exit_px, reason = max(b["open"], target), "take_profit"
            break
        if s.trail_act and not trailing and b["high"] >= fill + s.trail_act * atr:
            trailing = True
        if trailing:
            stop = max(stop, b["high"] - (s.trail_atr or 2.0) * atr, min(fill * (1 + rt_cost), b["close"] * 0.999))
        if bars >= max_bars:
            exit_px, reason = b["close"] * (1 - hs), "time_exit"
            break
    if exit_px is None:                            # still open at the end of the data: mark to market
        exit_px, reason = after["close"].iloc[-1] * (1 - hs), "open_mtm"
    ret = (1 - TAKER) * (1 - MAKER) * exit_px / fill - 1
    return {"filled": True, "ret_pct": round(ret * 100, 3), "exit_reason": reason, "bars": bars}


def main(days: int, min_signals: int):
    px15, sig15, sig60, sig240, u15 = build_frames()
    end = u15.index[-1] + pd.Timedelta("15min")
    d = days
    while True:
        sigs = collect_signals(px15, sig15, sig60, sig240, u15, end - pd.Timedelta(days=d), end)
        if len(sigs) >= min_signals or d >= 7:
            break
        d += 1
    print(f"window: last {d} days -> {len(sigs)} signals (evaluated independently, fresh portfolio)", flush=True)

    graph = get_graph()
    rows, decisions = [], []
    for k, (T, s, diag, s15, usdt_1h) in enumerate(sigs, 1):
        snap = snapshot(s15)
        snap.pop("price", None)
        snap.update(spread_pct=round(HALF_SPREAD.get(s.symbol, 0.001) * 200, 3), ob_imbalance=None,
                    r4h_usd_pct=diag.get("r4h_usd"), z_1h=diag.get("z_1h"), r24h_usd_pct=diag.get("r24h_usd"),
                    supertrend_4h=diag.get("st_4h"), max_hold_h=s.hold_h, trailing_stop=bool(s.trail_act))
        state = {"symbol": s.symbol, "setup": s.kind, "signal": s.to_dict(), "market": snap,
                 "portfolio": {"equity": CAPITAL, "cash": CAPITAL, "dd_pct": 0.0, "open_positions": 0,
                               "exposure_pct": 0.0, "exposure_tmn": 0.0, "pending_entries": 0},
                 "context": {"usdt_1h_pct": usdt_1h, "probe": False,
                             "setup_base_rate": SETUP_BASE_RATES.get(s.kind, ""), "recent_trades_symbol": "none"},
                 "llm_calls": []}
        out_state = graph.invoke(state, config={"configurable": {"execute": None, "db": None}})
        dec = out_state["decision"]
        # no broker in the replay: an approved entry is reported as SKIP by the executor, so use Risk Guard's verdict
        if out_state["risk"]["approved"]:
            dec["action"] = "BUY"
        size = out_state["risk"]["size_tmn"]
        res = outcome(px15[s.symbol], s, T)
        row = {"signal_time": T, "symbol": s.symbol, "setup": s.kind, "agent_action": dec["action"],
               "confidence": dec["confidence"], "size_tmn": size, **res,
               "llm_cost": dec["llm"]["cost_usd"], "arbiter": bool(dec["agent_debate"].get("arbiter_argument"))}
        rows.append(row)
        dec["signal_time"] = str(T)
        dec["outcome"] = res
        decisions.append(dec)
        print(f"[{k:2d}/{len(sigs)}] {T:%m-%d %H:%M} {s.symbol:8s} {s.kind:15s} -> {dec['action']:4s} "
              f"conf={dec['confidence']:.2f} | outcome: {res.get('ret_pct', 'no fill')} {res.get('exit_reason', '')}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "agent_replay.csv", index=False)
    with open(OUT / "agent_replay_decisions.jsonl", "w", encoding="utf-8") as f:
        for dd in decisions:
            f.write(json.dumps(dd, ensure_ascii=False, default=str) + "\n")

    f = df[df.filled == True].copy()  # noqa: E712
    f["win"] = f.ret_pct > 0
    buy, skip = f[f.agent_action == "BUY"], f[f.agent_action == "SKIP"]
    correct = ((f.agent_action == "BUY") & f.win) | ((f.agent_action == "SKIP") & ~f.win)
    print("\n================ RESULT ================")
    print(f"signals: {len(df)} | filled (limit reached): {len(f)} | agent BUY {len(buy)} / SKIP {len(skip)}")
    print(f"take EVERY signal : win {f.win.mean() * 100:.0f}% | avg {f.ret_pct.mean():+.2f}%/trade | "
          f"PnL at 20% size {f.ret_pct.sum() * 0.2:+.2f}% of capital")
    if len(buy):
        print(f"agent-APPROVED    : win {buy.win.mean() * 100:.0f}% | avg {buy.ret_pct.mean():+.2f}%/trade | "
              f"PnL with agent sizes {(buy.ret_pct / 100 * buy.size_tmn).sum() / CAPITAL * 100:+.2f}% of capital")
    if len(skip):
        print(f"agent-REJECTED    : win {skip.win.mean() * 100:.0f}% | avg {skip.ret_pct.mean():+.2f}%/trade "
              f"(losses avoided if negative)")
    print(f"decision accuracy (BUY->win, SKIP->loss): {correct.mean() * 100:.0f}%")
    print(f"mean confidence: winners {f[f.win].confidence.mean():.2f} vs losers {f[~f.win].confidence.mean():.2f}")
    print("by setup:\n" + f.groupby("setup").agg(n=("ret_pct", "size"), win=("win", "mean"), avg=("ret_pct", "mean"),
                                                   agent_buy=("agent_action", lambda x: (x == "BUY").sum())).round(2).to_string())
    print(f"LLM cost of replay: ${df.llm_cost.sum():.4f} | budget: {get_llm().budget.summary()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--min-signals", type=int, default=20)
    a = ap.parse_args()
    main(a.days, a.min_signals)
