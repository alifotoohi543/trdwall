"""Main engine: fast rule loop + agentic decision graph.

* Fast loop (every `loop_interval_s`): sync orders, fill/cancel entries, stop-loss / take-profit /
  trailing / time exits, USDT shock stop-tightening, equity snapshots, end-of-challenge liquidation.
* Scan (every closed 15m bar): refresh candles, compute USD-basis signals for all symbols, and
  wake the LangGraph agents only when a signal passes the rule filters (token economy).
* Every order has a matching record in decision_logs.jsonl (entries from the graph, exits from rules).

Usage:
    python -m src.engine            # run forever in TRADING_MODE from .env
    python -m src.engine --once     # one scan + one fast tick, then exit (smoke test)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.agents.graph import get_graph
from src.agents.state import AgentDebate, DecisionLog, Execution
from src.broker.base import BaseBroker
from src.config import LOG_DIR, SCAN_LOG, settings
from src.db import DB, now_iso
from src.llm.client import get_llm
from src.market_data import add_indicators, load_history, resample, snapshot
from src.strategy import (SETUP_BASE_RATES, StrategyParams, ensemble_signals, make_signal, round_trip_cost,
                          usd_frame)
from src.wallex_client import WallexClient, WallexError

log = logging.getLogger("engine")
TF = "15min"
BAR = pd.Timedelta(TF)
ENTRY_TTL = timedelta(minutes=15)        # unfilled limit entries are cancelled after one bar
PROBE_AFTER = timedelta(hours=20)        # activity probe if no entry for this long
PROBE_RSI = 32.0
HISTORY_DAYS = 14                     # 1m history kept in memory (4h indicators need ~2 weeks)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def make_broker(db: DB, client: WallexClient) -> BaseBroker:
    if settings.mode == "LIVE":
        from src.broker.live_broker import LiveBroker
        return LiveBroker(db, client)
    from src.broker.shadow_broker import ShadowBroker
    return ShadowBroker(db, client)


class Engine:
    def __init__(self):
        self.db = DB()
        self.client = WallexClient()
        self.broker = make_broker(self.db, self.client)
        self.p = StrategyParams()
        self.symbols = list(settings.symbols)
        self.m1: dict[str, pd.DataFrame] = {}
        self.usdt_mid: deque[tuple[datetime, float]] = deque(maxlen=180)
        self.last_bar: pd.Timestamp | None = None
        self.last_equity_snap = utcnow() - timedelta(minutes=10)
        self.end = (datetime.fromisoformat(settings.challenge_end_utc.replace("Z", "+00:00"))
                    if settings.challenge_end_utc else None)
        if self.db.get("started_at") is None:
            self.db.put("started_at", now_iso())
            self.db.put("peak_equity", settings.risk.initial_capital_tmn)
        self.db.event("INFO", f"engine start mode={settings.mode} end={self.end}")

    # ================================================================ data
    def refresh_candles(self) -> None:
        now = int(time.time())
        for sym in self.symbols + ["USDTTMN"]:
            df = self.m1.get(sym)
            if df is None:
                df = load_history(sym)
                df = df[df.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=HISTORY_DAYS)]
            start = int(df.index[-1].timestamp()) - 1800 if len(df) else now - HISTORY_DAYS * 86400
            try:
                new = self.client.candles_1m(sym, start, now)
            except WallexError as e:
                log.warning("candles %s failed: %s", sym, e)
                new = pd.DataFrame()
            if len(new):
                df = pd.concat([df, new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            self.m1[sym] = df[df.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=HISTORY_DAYS)]

    def all_frames(self):
        """Closed bars only: 15m TMN/USD frames, 1h USD frames, 15m USDT frame."""
        now = pd.Timestamp.now(tz="UTC")
        u15 = add_indicators(resample(self.m1["USDTTMN"], "15min"))
        u15 = u15[u15.index < now.floor("15min")]
        u60 = add_indicators(resample(self.m1["USDTTMN"], "60min"))
        u60 = u60[u60.index < now.floor("60min")]
        u240 = add_indicators(resample(self.m1["USDTTMN"], "240min"))
        u240 = u240[u240.index + pd.Timedelta("240min") <= now]
        px15, sig15, sig60, sig240 = {}, {}, {}, {}
        for s in self.symbols:
            try:
                p15 = add_indicators(resample(self.m1[s], "15min"))
                px15[s] = p15[p15.index < now.floor("15min")]
                sig15[s] = usd_frame(px15[s], u15)
                p60 = add_indicators(resample(self.m1[s], "60min"))
                sig60[s] = usd_frame(p60[p60.index < now.floor("60min")], u60)
                p240 = add_indicators(resample(self.m1[s], "240min"))
                sig240[s] = usd_frame(p240[p240.index + pd.Timedelta("240min") <= now], u240)
            except Exception as e:  # one bad market must not block the others
                log.warning("frames %s: %s", s, e)
        return px15, sig15, sig60, u15, sig240

    def usdt_1h_pct(self) -> float | None:
        if len(self.usdt_mid) < 2:
            return None
        t_now, m_now = self.usdt_mid[-1]
        past = [m for t, m in self.usdt_mid if t <= t_now - timedelta(minutes=55)]
        ref = past[-1] if past else self.usdt_mid[0][1]
        return round((m_now / ref - 1) * 100, 3)

    # ============================================================ portfolio
    def portfolio(self) -> dict:
        bal = self.broker.balances()
        cash = bal.get("TMN", 0.0)
        reserved = sum(o["tmn"] for o in self.db.open_orders() if o["side"] == "BUY")
        exposure = 0.0
        for asset, qty in bal.items():
            if asset == "TMN" or qty <= 0:
                continue
            sym = f"{asset}TMN"
            try:
                bid = self.broker.book(sym)["bid"] or 0.0
            except (WallexError, KeyError):
                bid = 0.0
            exposure += qty * bid
        # reserved TMN of resting buys is locked (LIVE) / debited (SHADOW) but still ours
        equity = cash + reserved + exposure
        peak = max(self.db.get("peak_equity", equity), equity)
        self.db.put("peak_equity", peak)
        dd = 1 - equity / peak if peak else 0.0
        return {"equity": equity, "cash": cash, "exposure_tmn": exposure,
                "exposure_pct": round(exposure / equity * 100, 2) if equity else 0.0,
                "dd_pct": round(dd * 100, 3), "open_positions": len(self.db.positions()),
                "pending_entries": len([o for o in self.db.open_orders() if o["side"] == "BUY"])}

    # ============================================================== logging
    def log_rule_decision(self, symbol: str, action: str, reason: str, price: float, qty: float,
                          client_id: str | None, metrics: dict | None = None) -> None:
        dec = DecisionLog(
            timestamp=now_iso(), mode=settings.mode, symbol=symbol, action=action, confidence=1.0,
            setup="risk_rule", market_metrics={"price_tmn": price, **(metrics or {})},
            agent_debate=AgentDebate(bull_argument="—", bear_argument="—", risk_assessment=reason),
            execution=Execution(order_type="LIMIT", side="SELL", allocated_tmn=round(price * qty),
                                entry_price=price, client_order_id=client_id) if action == "SELL" else None,
            llm={"calls": 0, "cost_usd": 0.0})
        self.db.add_decision(dec.model_dump())

    # ============================================================== entries
    def execute_entry(self, symbol: str, tmn: float, sig: dict) -> str | None:
        qty = tmn / sig["price"]
        o = self.broker.place_limit_buy(symbol, qty, sig["price"], "entry",
                                        {"stop": sig["stop"], "target": sig["target"], "kind": sig["kind"],
                                         "ref_price": sig["price"], "atr": sig["atr_tmn"], "hold_h": sig["hold_h"],
                                         "trail_act": sig["trail_act"], "trail_mult": sig["trail_atr"]})
        log.info("ENTRY order %s %s %.0f TMN @ %.0f -> %s", o["client_id"], symbol, tmn, sig["price"], o["status"])
        if o["status"] == "FILLED":
            self.open_position_from_order(o)
        return o["client_id"]

    def open_position_from_order(self, o: dict) -> None:
        meta = json.loads(o["meta"])
        fill = o.get("avg_price") or o["price"]
        scale = fill / meta["ref_price"]
        qty = o["filled_qty"]
        if not qty:
            return
        self.db.upsert_position({"symbol": o["symbol"], "qty": qty, "entry": fill, "stop": meta["stop"] * scale,
                                 "target": meta["target"] * scale, "atr": meta["atr"], "opened": now_iso(),
                                 "kind": meta["kind"], "cost_tmn": qty * fill / (1 - settings.fees.taker),
                                 "trailing": 0, "decision_id": o["client_id"], "hold_h": meta.get("hold_h", 4),
                                 "trail_act": meta.get("trail_act"), "trail_mult": meta.get("trail_mult")})
        self.db.put("last_entry_at", now_iso())
        self.db.event("TRADE", f"opened {o['symbol']} qty={qty} @ {fill:.0f}")
        log.info("POSITION opened %s qty=%s @ %.0f", o["symbol"], qty, fill)

    # =============================================================== orders
    def manage_orders(self) -> None:
        for o in self.db.open_orders():
            try:
                o = self.broker.sync_order(o)
            except WallexError as e:
                log.warning("sync %s failed: %s", o["client_id"], e)
                continue
            age = utcnow() - datetime.fromisoformat(o["created"].replace("Z", "+00:00"))
            if o["side"] == "BUY":
                if o["status"] == "FILLED":
                    self.open_position_from_order(o)
                elif age > ENTRY_TTL:
                    self.broker.cancel(o)
                    log.info("entry %s %s expired unfilled -> cancelled", o["client_id"], o["symbol"])
                    if o.get("filled_qty"):
                        self.open_position_from_order(o)    # keep the partial fill as a position
                    else:
                        self.log_rule_decision(o["symbol"], "CANCEL",
                                               "سفارش لیمیت ورود در مدت یک کندل ۱۵ دقیقه‌ای پر نشد و لغو شد (قیمت به سطح ورود برنگشت).",
                                               o["price"], 0, o["client_id"])
            elif o["side"] == "SELL":
                if o["status"] == "FILLED":
                    self.finalize_exit(o)
                elif age > timedelta(seconds=90):
                    self.broker.cancel(o)                   # remaining qty retried on next tick
                    if o.get("filled_qty"):
                        self.finalize_exit(o, partial=True)

    # ================================================================ exits
    def close_position(self, pos: dict, reason_code: str, reason_fa: str, price: float) -> None:
        if any(o["side"] == "SELL" and o["symbol"] == pos["symbol"] for o in self.db.open_orders()):
            return                                          # exit already in flight
        try:
            o = self.broker.sell_now(pos["symbol"], pos["qty"], reason_code, {"reason": reason_code})
        except (WallexError, ValueError) as e:
            log.error("exit %s failed: %s", pos["symbol"], e)
            self.db.event("ERROR", f"exit {pos['symbol']} failed: {e}")
            return
        self.log_rule_decision(pos["symbol"], "SELL", reason_fa, price, pos["qty"], o["client_id"],
                               {"entry": pos["entry"], "stop": pos["stop"], "target": pos["target"]})
        if o["status"] == "FILLED":
            self.finalize_exit(o)

    def finalize_exit(self, o: dict, partial: bool = False) -> None:
        pos = self.db.positions().get(o["symbol"])
        if not pos:
            return
        sold = o["filled_qty"] or 0.0
        px = o.get("avg_price") or o["price"]
        proceeds = sold * px * (1 - settings.fees.taker)
        frac = min(sold / pos["qty"], 1.0) if pos["qty"] else 1.0
        cost = pos["cost_tmn"] * frac
        self.db.add_trade({"symbol": pos["symbol"], "kind": pos["kind"], "opened": pos["opened"], "closed": now_iso(),
                           "entry": pos["entry"], "exit": px, "qty": sold, "pnl_tmn": proceeds - cost,
                           "pnl_pct": (proceeds / cost - 1) * 100 if cost else 0.0,
                           "exit_reason": json.loads(o.get("meta") or "{}").get("reason", o.get("purpose")),
                           "decision_id": pos["decision_id"]})
        remaining = pos["qty"] - sold
        if remaining * px < settings.risk.min_order_tmn * 0.5:
            self.db.delete_position(pos["symbol"])
        else:
            pos.update(qty=remaining, cost_tmn=pos["cost_tmn"] - cost)
            self.db.upsert_position(pos)
        if proceeds < cost:
            self.db.put(f"cooldown_{pos['symbol']}",
                        (utcnow() + timedelta(minutes=settings.risk.cooldown_after_loss_min)).isoformat())
        log.info("EXIT %s sold=%s @ %.0f pnl=%.0f TMN (%s)", o["symbol"], sold, px, proceeds - cost,
                 "partial" if partial else "full")

    def manage_positions(self, shock: bool) -> None:
        for sym, pos in self.db.positions().items():
            try:
                book = self.broker.book(sym)
            except WallexError as e:
                log.warning("book %s failed: %s", sym, e)
                continue
            bid = book["bid"]
            if bid is None:
                continue
            opened = datetime.fromisoformat(pos["opened"].replace("Z", "+00:00"))
            held = utcnow() - opened
            if bid <= pos["stop"]:
                code = "trailing_stop" if pos["trailing"] else "stop_loss"
                fa = (f"حد ضرر متحرک فعال شد: قیمت خرید {bid:,.0f} ≤ {pos['stop']:,.0f}؛ قفل سود."
                      if pos["trailing"] else
                      f"حد ضرر فعال شد: قیمت خرید {bid:,.0f} ≤ حد ضرر {pos['stop']:,.0f} (ورود {pos['entry']:,.0f}).")
                self.close_position(pos, code, fa, bid)
                continue
            if bid >= pos["target"]:
                self.close_position(pos, "take_profit",
                                    f"حد سود محقق شد: قیمت {bid:,.0f} ≥ هدف {pos['target']:,.0f} (ورود {pos['entry']:,.0f}).", bid)
                continue
            hold_h = pos.get("hold_h") or 4
            if held >= timedelta(hours=hold_h):
                self.close_position(pos, "time_exit",
                                    f"خروج زمانی: {hold_h:g} ساعت از ورود گذشت و لبه آماری setup ({pos['kind']}) مصرف شد.", bid)
                continue
            changed = False
            if pos.get("trail_act") and not pos["trailing"] and bid >= pos["entry"] + pos["trail_act"] * pos["atr"]:
                pos["trailing"] = 1
                changed = True
            if pos["trailing"]:
                floor = min(pos["entry"] * (1 + round_trip_cost(sym)), bid * 0.999)
                new_stop = max(pos["stop"], bid - (pos.get("trail_mult") or 2.0) * pos["atr"], floor)
                changed |= new_stop > pos["stop"]
                pos["stop"] = new_stop
            if shock:
                new_stop = max(pos["stop"], bid - self.p.shock_stop_atr * pos["atr"])
                if new_stop > pos["stop"]:
                    pos["stop"] = new_stop
                    changed = True
                    self.db.event("RISK", f"USDT shock: tightened {sym} stop to {new_stop:.0f}")
            if changed:
                self.db.upsert_position(pos)

    # ================================================================= scan
    def scan(self) -> None:
        self.refresh_candles()
        pf = self.portfolio()
        now = pd.Timestamp.now(tz="UTC")
        px15, sig15, sig60, u15, sig240 = self.all_frames()
        signals, diag = ensemble_signals(px15, sig15, sig60, u15, now, self.p, sig240=sig240)

        # activity probe: no entry for PROBE_AFTER -> allow one small, debated entry on the most oversold coin
        last_entry = self.db.get("last_entry_at") or self.db.get("started_at")
        if not signals and utcnow() - datetime.fromisoformat(last_entry.replace("Z", "+00:00")) > PROBE_AFTER:
            cands = [(sig15[s]["rsi"].iloc[-1], s) for s in sig15 if sig15[s]["rsi"].iloc[-1] < PROBE_RSI]
            if cands:
                _, sym = min(cands)
                signals = [make_signal(sym, "probe_oversold", sig15[sym].iloc[-1], float(px15[sym]["close"].iloc[-1]),
                                       "mr", sig15[sym].index[-1])]
                diag[sym]["status"] = "probe"

        seen = set(self.db.get("seen_signals", []))
        for s in signals:
            key = f"{s.kind}|{s.symbol}|{s.bar}"
            if key in seen:
                continue                                    # each signal bar is debated once, even after restarts
            seen.add(key)
            self.db.put("seen_signals", sorted(seen)[-500:])
            if s.symbol in self.db.positions() or any(o["symbol"] == s.symbol for o in self.db.open_orders()):
                continue
            cd = self.db.get(f"cooldown_{s.symbol}")
            if cd and utcnow() < datetime.fromisoformat(cd):
                continue
            if pf["dd_pct"] / 100 >= settings.risk.hard_stop_dd or pf["open_positions"] >= settings.risk.max_open_positions:
                continue
            if self.end and utcnow() >= self.end - timedelta(minutes=settings.liquidate_minutes_before_end + 60):
                continue
            try:
                ob = self.broker.book(s.symbol)
            except WallexError:
                ob = {}
            snap_usd = snapshot(sig15[s.symbol])
            snap_usd.pop("price", None)
            snap_usd.update(spread_pct=round(ob.get("spread_pct") or 0, 3),
                            ob_imbalance=round(ob["ob_imbalance"], 2) if ob.get("ob_imbalance") else None,
                            r4h_usd_pct=diag[s.symbol].get("r4h_usd"), z_1h=diag[s.symbol].get("z_1h"),
                            r24h_usd_pct=diag[s.symbol].get("r24h_usd"), supertrend_4h=diag[s.symbol].get("st_4h"),
                            max_hold_h=s.hold_h, trailing_stop=bool(s.trail_act))
            recent = self.db.rows("SELECT pnl_pct, exit_reason FROM trades WHERE symbol=? ORDER BY id DESC LIMIT 3",
                                  (s.symbol,))
            state = {"symbol": s.symbol, "setup": s.kind, "signal": s.to_dict(), "market": snap_usd, "portfolio": pf,
                     "context": {"usdt_1h_pct": self.usdt_1h_pct(), "probe": s.kind == "probe_oversold",
                                 "setup_base_rate": SETUP_BASE_RATES.get(s.kind, "relaxed activity probe, no proven edge"),
                                 "recent_trades_symbol": [f"{r['pnl_pct']:+.2f}% {r['exit_reason']}" for r in recent] or "none"},
                     "llm_calls": []}
            log.info("SIGNAL %s %s -> waking agents", s.symbol, s.reason)
            out = get_graph().invoke(state, config={"configurable": {"execute": self.execute_entry, "db": self.db}})
            d = out["decision"]
            log.info("DECISION %s %s conf=%.2f | %s", s.symbol, d["action"], d["confidence"],
                     d["agent_debate"]["risk_assessment"][:160])
            pf = self.portfolio()

        rows = [{"symbol": s, "price": float(px15[s]["close"].iloc[-1]), "rsi_usd": diag[s].get("rsi_usd"),
                 "z_1h": diag[s].get("z_1h"), "r4h_usd": diag[s].get("r4h_usd"),
                 "st_4h": diag[s].get("st_4h"), "r24h_usd": diag[s].get("r24h_usd"),
                 "status": diag[s].get("status", "no_setup")} for s in px15]
        scan = {"ts": now_iso(), "rows": rows, "portfolio": pf, "llm_budget": get_llm().budget.summary()}
        self.db.put("last_scan", scan)
        with open(SCAN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(scan, ensure_ascii=False) + "\n")
        log.info("SCAN %s | equity=%.0f dd=%.2f%% pos=%d | %s", scan["ts"], pf["equity"], pf["dd_pct"],
                 pf["open_positions"], " ".join(f"{r['symbol'][:-3]}:{r['rsi_usd']}/{r['z_1h']}/{r['status'][:6]}"
                                                for r in rows))

    # ========================================================== liquidation
    def liquidate_all(self, why: str) -> None:
        log.warning("LIQUIDATION: %s", why)
        for o in self.db.open_orders():
            if o["side"] == "BUY":
                self.broker.cancel(o)
                if o.get("filled_qty"):
                    self.open_position_from_order(o)
        for sym, pos in self.db.positions().items():
            bid = (self.broker.book(sym)["bid"] or pos["entry"])
            self.close_position(pos, "final_liquidation",
                                f"نقد کردن خودکار پایان چالش ({why}): تبدیل کامل دارایی به تومان برای محاسبه سود محقق‌شده.", bid)
        # sweep leftovers (dust from partial fills, manual transfers) that are above the minimum notional
        for asset, qty in self.broker.balances().items():
            sym = f"{asset}TMN"
            if asset == "TMN" or sym not in self.symbols or sym in self.db.positions():
                continue
            bid = self.broker.book(sym)["bid"] or 0
            if qty * bid >= 60_000:
                o = self.broker.sell_now(sym, qty, "final_liquidation", {"reason": "final_liquidation_sweep"})
                self.log_rule_decision(sym, "SELL", "فروش باقی‌مانده دارایی در نقد کردن پایان چالش.", bid, qty, o["client_id"])

    # ================================================================= loop
    def tick(self) -> None:
        try:
            self.usdt_mid.append((utcnow(), self.broker.book("USDTTMN")["mid"]))
        except (WallexError, TypeError):
            pass
        u = self.usdt_1h_pct()
        shock = u is not None and abs(u) > self.p.shock_pct
        self.manage_orders()
        if self.end and utcnow() >= self.end - timedelta(minutes=settings.liquidate_minutes_before_end):
            if not self.db.get("liquidated"):
                self.liquidate_all("زمان‌بند ۳۰ دقیقه قبل از پایان")
                self.db.put("liquidated", now_iso())
            elif self.db.positions():
                self.liquidate_all("تلاش مجدد")
            return
        self.manage_positions(shock)
        if utcnow() - self.last_equity_snap >= timedelta(minutes=5):
            pf = self.portfolio()
            self.db.snapshot_equity(pf["equity"], pf["cash"], pf["exposure_tmn"], pf["dd_pct"])
            self.last_equity_snap = utcnow()

    def run(self, once: bool = False) -> None:
        while True:
            t0 = time.monotonic()
            try:
                self.tick()
                bar = pd.Timestamp.now(tz="UTC").floor(TF)
                if (once or bar != self.last_bar) and pd.Timestamp.now(tz="UTC") - bar >= pd.Timedelta(seconds=20):
                    if not self.db.get("liquidated"):
                        self.scan()
                    self.last_bar = bar
            except Exception as e:  # keep running whatever happens; stability is scored
                log.exception("loop error: %s", e)
                self.db.event("ERROR", f"loop error: {e}")
            if once:
                return
            if self.end and utcnow() > self.end + timedelta(minutes=5):
                log.info("challenge over; final portfolio %s", self.portfolio())
                return
            time.sleep(max(1.0, settings.loop_interval_s - (time.monotonic() - t0)))


def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "engine.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False))
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    for noisy in ("httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    setup_logging()
    Engine().run(once=args.once)
