"""Event-driven multi-symbol backtester for the rule layer.

Conservative assumptions:
  * signal on bar t close, fill at bar t+1 open + half-spread slippage + taker fee;
  * if stop and target are both inside a bar, the stop is assumed to hit first;
  * trailing stop is updated only after the bar's stop check;
  * everything is liquidated to TMN at the end (realized PnL, like the challenge).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import settings
from src.market_data import add_indicators, load_history, resample
from src.strategy import (HALF_SPREAD, round_trip_cost, StrategyParams, build_signal, entry_flags, passes_edge_filter,
                          position_size_tmn, usdt_regime)

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
    bars: int = 0
    trailing: bool = False


@dataclass
class BTResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict = field(default_factory=dict)


def load_universe(symbols, timeframe: str) -> dict[str, pd.DataFrame]:
    return {s: add_indicators(resample(load_history(s), timeframe)) for s in symbols}


def _buy_px(sym, px):
    return px * (1 + HALF_SPREAD.get(sym, 0.001))


def _sell_px(sym, px):
    return px * (1 - HALF_SPREAD.get(sym, 0.001))


def run_backtest(data: dict[str, pd.DataFrame], usdt: pd.DataFrame | None, p: StrategyParams,
                 use_usdt: bool = False, trade_alts: bool = True,
                 start=None, end=None, capital: float | None = None) -> BTResult:
    r = settings.risk
    capital = capital or r.initial_capital_tmn
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    if usdt is not None:
        idx = idx.intersection(usdt.index)
    idx = idx[(idx >= (start or idx[0])) & (idx <= (end or idx[-1]))]
    flags = {s: entry_flags(df, p) for s, df in data.items()}
    regime = usdt_regime(usdt, p) if (use_usdt and usdt is not None) else None

    cash, usdt_qty, peak = capital, 0.0, capital
    positions: dict[str, Position] = {}
    trades, equity_pts = [], []
    pending: list[tuple[str, str]] = []
    cooldown: dict[str, pd.Timestamp] = {}
    fees_paid = 0.0

    def usdt_px(t):
        return float(usdt.at[t, "close"]) if usdt is not None else 0.0

    def raise_cash(need, t):
        nonlocal cash, usdt_qty, fees_paid
        if need <= cash or usdt_qty <= 0:
            return
        px = _sell_px("USDTTMN", float(usdt.at[t, "open"]))
        q = min(usdt_qty, (need - cash) / (px * (1 - FEE)) * 1.001)
        cash += q * px * (1 - FEE)
        fees_paid += q * px * FEE
        usdt_qty -= q

    def close_pos(pos, px_raw, t, why):
        nonlocal cash, fees_paid
        px = _sell_px(pos.symbol, px_raw)
        proceeds = pos.qty * px * (1 - FEE)
        fees_paid += pos.qty * px * FEE
        cash += proceeds
        trades.append({"symbol": pos.symbol, "kind": pos.kind, "opened": pos.opened, "closed": t,
                       "entry": pos.entry, "exit": px, "pnl_tmn": proceeds - pos.cost_tmn,
                       "pnl_pct": (proceeds / pos.cost_tmn - 1) * 100, "exit_reason": why, "bars": pos.bars})
        if proceeds < pos.cost_tmn:
            cooldown[pos.symbol] = t + pd.Timedelta(minutes=r.cooldown_after_loss_min)

    for i, t in enumerate(idx):
        # 1) fill pending entries at this bar's open
        for sym, kind in pending:
            if sym in positions or len(positions) >= r.max_open_positions:
                continue
            row_prev = data[sym].loc[idx[i - 1]]
            sig = build_signal(sym, row_prev, kind, p)
            eq_now = equity_pts[-1][1] if equity_pts else capital
            dd = 1 - eq_now / peak
            size = position_size_tmn(eq_now, sig.price, sig.stop, dd)
            exposure = sum(ps.qty * float(data[ps.symbol].at[t, "open"]) for ps in positions.values())
            size = min(size, eq_now * r.max_total_exposure_frac - exposure)
            if size < r.min_order_tmn:
                continue
            raise_cash(size, t)
            size = min(size, cash)
            if size < r.min_order_tmn:
                continue
            px = _buy_px(sym, float(data[sym].at[t, "open"]))
            qty = size * (1 - FEE) / px
            fees_paid += size * FEE
            cash -= size
            a = float(row_prev["atr"])
            positions[sym] = Position(sym, qty, px, px - p.atr_stop * a, px + p.atr_tp * a, a, t, kind, size)
        pending = []

        # 2) manage open positions with this bar's range
        for sym in list(positions):
            pos, bar = positions[sym], data[sym].loc[t]
            pos.bars += 1
            o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
            if l <= pos.stop:
                close_pos(positions.pop(sym), min(o, pos.stop), t, "trailing_stop" if pos.trailing else "stop_loss")
                continue
            if h >= pos.target:
                close_pos(positions.pop(sym), max(o, pos.target), t, "take_profit")
                continue
            if h >= pos.entry + p.trail_activate_atr * pos.atr:
                pos.trailing = True
            if pos.trailing:
                # lock in at least the round-trip cost, but never above the current close
                floor = min(pos.entry * (1 + round_trip_cost(sym)), c * 0.999)
                pos.stop = max(pos.stop, h - p.trail_atr * pos.atr, floor)
            if pos.bars >= p.max_hold_bars:
                close_pos(positions.pop(sym), c, t, "time_exit")

        # 3) USDT parking for idle cash
        if regime is not None:
            on = bool(regime.at[t])
            eq_now = cash + usdt_qty * usdt_px(t) + sum(ps.qty * float(data[ps.symbol].at[t, "close"])
                                                         for ps in positions.values())
            want = eq_now * p.usdt_alloc - usdt_qty * usdt_px(t)
            if on and want > r.min_order_tmn * 2 and cash > r.min_order_tmn * 2:
                px = _buy_px("USDTTMN", float(usdt.at[t, "close"]))
                spend = min(want, cash * 0.98)
                usdt_qty += spend * (1 - FEE) / px
                fees_paid += spend * FEE
                cash -= spend
            elif not on and usdt_qty > 0:
                raise_cash(cash + usdt_qty * usdt_px(t) * 2, t)

        # 4) mark to market
        eq = cash + usdt_qty * usdt_px(t) + sum(ps.qty * float(data[ps.symbol].at[t, "close"]) for ps in positions.values())
        peak = max(peak, eq)
        equity_pts.append((t, eq))

        # 5) new signals on this closed bar -> filled next bar
        if trade_alts and i < len(idx) - 1 and (1 - eq / peak) < r.hard_stop_dd:
            for sym in data:
                kind = flags[sym].at[t]
                if kind and sym not in positions and cooldown.get(sym, t) <= t:
                    sig = build_signal(sym, data[sym].loc[t], kind, p)
                    if passes_edge_filter(sig):
                        pending.append((sym, kind))

    # final liquidation
    t_end = idx[-1]
    for sym in list(positions):
        close_pos(positions.pop(sym), float(data[sym].at[t_end, "close"]), t_end, "final_liquidation")
    if usdt_qty > 0:
        px = _sell_px("USDTTMN", usdt_px(t_end))
        cash += usdt_qty * px * (1 - FEE)
        fees_paid += usdt_qty * px * FEE
        usdt_qty = 0
    equity_pts[-1] = (t_end, cash)

    eq = pd.Series(dict(equity_pts))
    tr = pd.DataFrame(trades)
    dd = (1 - eq / eq.cummax()).max()
    wins = tr[tr.pnl_tmn > 0] if len(tr) else tr
    stats = {
        "return_pct": round((cash / capital - 1) * 100, 2),
        "max_dd_pct": round(dd * 100, 2),
        "trades": len(tr),
        "win_rate": round(len(wins) / len(tr) * 100, 1) if len(tr) else None,
        "profit_factor": round(wins.pnl_tmn.sum() / -tr[tr.pnl_tmn <= 0].pnl_tmn.sum(), 2)
        if len(tr) and (tr.pnl_tmn <= 0).any() else None,
        "avg_trade_pct": round(tr.pnl_pct.mean(), 3) if len(tr) else None,
        "fees_tmn": round(fees_paid),
    }
    return BTResult(eq, tr, stats)
