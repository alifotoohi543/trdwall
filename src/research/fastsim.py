"""Fast per-symbol trade simulator used for strategy research.

Same execution assumptions as src/backtest.py (the portfolio backtester):
  * signal at bar i close -> LIMIT buy at that TMN price, filled on bar i+1 only if low <= limit;
  * stop checked before target inside a bar (pessimistic); stop/time exits pay half-spread;
  * trailing stop (optional) activates after `trail_act` ATR of profit, floors at break-even + costs;
  * one position per symbol at a time; fee on both sides.
Returns one row per trade with net return, so strategies can be ranked without portfolio effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import settings

FEE = settings.fees.taker


def _arrays(px: pd.DataFrame, atr_pct: pd.Series):
    o, h, l, c = (px[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    return o, h, l, c, atr_pct.reindex(px.index).to_numpy(float)


def _trade(i, o, h, l, c, ap, half_spread, stop_atr, tp_atr, hold_bars, trail_act, trail_atr, min_stop=0.0):
    """Simulate one limit entry signalled at bar i. Returns (exit_bar, net_ret, reason) or None."""
    n = len(c)
    if i >= n - 1 or not np.isfinite(ap[i]) or ap[i] <= 0:
        return None
    limit = c[i]
    if l[i + 1] > limit:
        return None
    fill = min(o[i + 1], limit)
    a = ap[i] * limit
    stop, tp = fill - max(stop_atr * a, min_stop * fill), fill + tp_atr * a
    rt_cost = 2 * FEE + half_spread
    trailing = False
    last = min(i + hold_bars, n - 1)
    for j in range(i + 1, last + 1):
        if l[j] <= stop:
            return j, (1 - FEE) ** 2 * min(o[j], stop) * (1 - half_spread) / fill - 1, "trail" if trailing else "stop"
        if h[j] >= tp:
            return j, (1 - FEE) ** 2 * max(o[j], tp) / fill - 1, "tp"
        if trail_act is not None:
            if not trailing and h[j] >= fill + trail_act * a:
                trailing = True
            if trailing:
                stop = max(stop, h[j] - trail_atr * a, min(fill * (1 + rt_cost), c[j] * 0.999))
    return last, (1 - FEE) ** 2 * c[last] * (1 - half_spread) / fill - 1, "time"


def simulate(px: pd.DataFrame, atr_pct: pd.Series, entries: pd.Series, half_spread: float,
             stop_atr: float, tp_atr: float, hold_bars: int,
             trail_act: float | None = None, trail_atr: float | None = None, min_stop: float = 0.0) -> pd.DataFrame:
    arr = _arrays(px, atr_pct)
    sig = entries.reindex(px.index).fillna(False).to_numpy(bool)
    out, i, n = [], 0, len(px)
    while i < n - 1:
        if sig[i]:
            r = _trade(i, *arr, half_spread, stop_atr, tp_atr, hold_bars, trail_act, trail_atr, min_stop)
            if r is not None:
                j, ret, reason = r
                out.append((px.index[i], px.index[j], ret, reason, j - i))
                i = j + 1                              # one position per symbol
                continue
        i += 1
    return pd.DataFrame(out, columns=["t_signal", "t_exit", "ret", "reason", "bars"])


def label_every_bar(px: pd.DataFrame, atr_pct: pd.Series, half_spread: float, stop_atr: float, tp_atr: float,
                    hold_bars: int, trail_act=None, trail_atr=None) -> pd.Series:
    """Net return of a hypothetical entry at every bar (overlapping) - the ML training target."""
    arr = _arrays(px, atr_pct)
    y = np.full(len(px), np.nan)
    for i in range(len(px) - hold_bars - 1):
        r = _trade(i, *arr, half_spread, stop_atr, tp_atr, hold_bars, trail_act, trail_atr)
        if r is not None:
            y[i] = r[1]
    return pd.Series(y, index=px.index)
