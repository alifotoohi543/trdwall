"""Event-driven multi-symbol backtester for the rule layer.

Conservative assumptions:
  * signal on bar t close; LIMIT entry at that price is filled on bar t+1 only if
    the bar trades through it (else cancelled); MARKET entry fills at t+1 open + spread;
  * if stop and target are both inside a bar, the stop is assumed to hit first;
  * stop exits pay half-spread, take-profit is a LIMIT;
  * trailing stop is updated only after the bar's stop check;
  * everything is liquidated to TMN at the end (realized PnL, like the challenge).
Each trade also records its PnL in USD terms (prices / USDTTMN) to separate our
edge from the toman's drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import settings
from src.market_data import add_indicators, load_history, resample
from src.strategy import (HALF_SPREAD, StrategyParams, build_signal, entry_flags, passes_edge_filter,
                          position_size_tmn, round_trip_cost, usd_frame, usdt_shock)

FEE = settings.fees.taker


@dataclass
class Position:
    symbol: str
    qty: float
    entry: float
    stop: float
    target: float
    atr: float
    opened: pd.Timestamp
    kind: str
    cost_tmn: float
    usdt_in: float
    bars: int = 0
    trailing: bool = False


@dataclass
class BTResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict = field(default_factory=dict)


def load_universe(symbols, timeframe: str, usd_signals: bool = True):
    """Return (price frames in TMN, signal frames, USDTTMN frame)."""
    usdt = add_indicators(resample(load_history("USDTTMN"), timeframe))
    px = {s: add_indicators(resample(load_history(s), timeframe)) for s in symbols}
    sig = {s: usd_frame(df, usdt) for s, df in px.items()} if usd_signals else px
    return px, sig, usdt


def run_backtest(px: dict[str, pd.DataFrame], sig: dict[str, pd.DataFrame], usdt: pd.DataFrame,
                 p: StrategyParams, start=None, end=None, capital: float | None = None) -> BTResult:
    r = settings.risk
    capital = capital or r.initial_capital_tmn
    idx = usdt.index
    for s in px:
        idx = idx.intersection(px[s].index).intersection(sig[s].index)
    idx = idx[(idx >= (start or idx[0])) & (idx <= (end or idx[-1]))]
    flags = {s: entry_flags(df, p) for s, df in sig.items()}
    shock = usdt_shock(usdt, p)

    cash, peak, fees_paid = capital, capital, 0.0
    positions: dict[str, Position] = {}
    trades, equity_pts = [], []
    pending: list[tuple[str, str, pd.Timestamp]] = []
    cooldown: dict[str, pd.Timestamp] = {}

    def close_pos(pos: Position, px_exit: float, t, why: str, pay_spread: bool):
        nonlocal cash, fees_paid
        if pay_spread:
            px_exit *= 1 - HALF_SPREAD.get(pos.symbol, 0.001)
        proceeds = pos.qty * px_exit * (1 - FEE)
        fees_paid += pos.qty * px_exit * FEE
        cash += proceeds
        u_out = float(usdt.at[t, "close"])
        trades.append({"symbol": pos.symbol, "kind": pos.kind, "opened": pos.opened, "closed": t,
                       "entry": pos.entry, "exit": px_exit, "pnl_tmn": proceeds - pos.cost_tmn,
                       "pnl_pct": (proceeds / pos.cost_tmn - 1) * 100,
                       "pnl_usd_pct": ((proceeds / u_out) / (pos.cost_tmn / pos.usdt_in) - 1) * 100,
                       "exit_reason": why, "bars": pos.bars})
        if proceeds < pos.cost_tmn:
            cooldown[pos.symbol] = t + pd.Timedelta(minutes=r.cooldown_after_loss_min)

    for i, t in enumerate(idx):
        # 1) fill pending entries on this bar
        for sym, kind, t_sig in pending:
            if sym in positions or len(positions) >= r.max_open_positions:
                continue
            ref = float(px[sym].at[t_sig, "close"])
            s = build_signal(sym, sig[sym].loc[t_sig], kind, p, ref)
            eq_now = equity_pts[-1][1]
            size = position_size_tmn(eq_now, s.price, s.stop, 1 - eq_now / peak)
            exposure = sum(ps.qty * float(px[ps.symbol].at[t, "open"]) for ps in positions.values())
            size = min(size, eq_now * r.max_total_exposure_frac - exposure, cash)
            if size < r.min_order_tmn:
                continue
            bar = px[sym].loc[t]
            if p.limit_entry:
                if float(bar["low"]) > ref:
                    continue                          # limit not reached -> cancelled
                fill = min(float(bar["open"]), ref)
            else:
                fill = float(bar["open"]) * (1 + HALF_SPREAD.get(sym, 0.001))
            qty = size * (1 - FEE) / fill
            fees_paid += size * FEE
            cash -= size
            scale = fill / ref
            positions[sym] = Position(sym, qty, fill, s.stop * scale, s.target * scale, (ref - s.stop) / p.atr_stop,
                                      t, kind, size, float(usdt.at[t, "close"]))
        pending = []

        in_shock = bool(shock.at[t])
        # 2) manage open positions with this bar's range
        for sym in list(positions):
            pos, bar = positions[sym], px[sym].loc[t]
            pos.bars += 1
            o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
            if l <= pos.stop:
                close_pos(positions.pop(sym), min(o, pos.stop), t,
                          "trailing_stop" if pos.trailing else "stop_loss", pay_spread=True)
                continue
            if h >= pos.target:
                close_pos(positions.pop(sym), max(o, pos.target), t, "take_profit", pay_spread=False)
                continue
            if h >= pos.entry + p.trail_activate_atr * pos.atr:
                pos.trailing = True
            if pos.trailing:
                floor = min(pos.entry * (1 + round_trip_cost(sym)), c * 0.999)
                pos.stop = max(pos.stop, h - p.trail_atr * pos.atr, floor)
            if in_shock:
                pos.stop = max(pos.stop, c - p.shock_stop_atr * pos.atr)
            if pos.bars >= p.max_hold_bars:
                close_pos(positions.pop(sym), c, t, "time_exit", pay_spread=True)

        # 3) mark to market
        eq = cash + sum(ps.qty * float(px[ps.symbol].at[t, "close"]) for ps in positions.values())
        peak = max(peak, eq)
        equity_pts.append((t, eq))

        # 4) new signals on this closed bar -> limit orders for the next bar
        if i < len(idx) - 1 and not in_shock and (1 - eq / peak) < r.hard_stop_dd:
            for sym in px:
                kind = flags[sym].at[t]
                if kind and sym not in positions and cooldown.get(sym, t) <= t:
                    s = build_signal(sym, sig[sym].loc[t], kind, p, float(px[sym].at[t, "close"]))
                    if passes_edge_filter(s, p):
                        pending.append((sym, kind, t))

    t_end = idx[-1]
    for sym in list(positions):
        close_pos(positions.pop(sym), float(px[sym].at[t_end, "close"]), t_end, "final_liquidation", pay_spread=True)
    equity_pts[-1] = (t_end, cash)

    eq = pd.Series(dict(equity_pts))
    tr = pd.DataFrame(trades)
    days = max((idx[-1] - idx[0]).total_seconds() / 86400, 1e-9)
    losers = tr[tr.pnl_tmn <= 0] if len(tr) else tr
    stats = {
        "return_pct": round((cash / capital - 1) * 100, 2),
        "max_dd_pct": round(float((1 - eq / eq.cummax()).max()) * 100, 2),
        "trades": len(tr),
        "trades_per_day": round(len(tr) / days, 2),
        "win_rate": round((tr.pnl_tmn > 0).mean() * 100, 1) if len(tr) else None,
        "profit_factor": round(tr[tr.pnl_tmn > 0].pnl_tmn.sum() / -losers.pnl_tmn.sum(), 2)
        if len(losers) and losers.pnl_tmn.sum() < 0 else None,
        "avg_trade_pct": round(tr.pnl_pct.mean(), 3) if len(tr) else None,
        "avg_trade_usd_pct": round(tr.pnl_usd_pct.mean(), 3) if len(tr) else None,
        "fees_tmn": round(fees_paid),
    }
    return BTResult(eq, tr, stats)
