"""Rule-based signal layer (the "Trigger Filter").

The same functions drive the backtest and the live engine: they decide *when*
the agents are woken up and propose concrete entry/stop/target levels. The LLM
agents can only veto or downsize these proposals, never invent trades.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import settings

# Typical half-spread per market (fraction), measured from live books; used as slippage.
HALF_SPREAD = {"BTCTMN": 0.0001, "ETHTMN": 0.0004, "SOLTMN": 0.001, "XRPTMN": 0.0012,
               "DOGETMN": 0.0007, "USDTTMN": 0.0002}


@dataclass
class StrategyParams:
    name: str = "trend_pullback"
    rsi_dip: float = 40.0          # pullback threshold
    breakout_bars: int = 48        # lookback for breakout high
    atr_stop: float = 1.5
    atr_tp: float = 3.0
    trail_atr: float = 1.2
    trail_activate_atr: float = 1.5   # trailing starts only after this much profit
    max_hold_bars: int = 48
    min_vol_z: float = 0.5         # breakout volume confirmation
    usdt_fast: int = 48            # USDT trend EMAs (in bars)
    usdt_slow: int = 192
    usdt_alloc: float = 0.5        # fraction of idle cash parked in USDT when the regime is on


@dataclass
class Signal:
    symbol: str
    kind: str
    price: float
    stop: float
    target: float
    expected_move_pct: float
    cost_pct: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def round_trip_cost(symbol: str) -> float:
    return 2 * settings.fees.taker + 2 * HALF_SPREAD.get(symbol, 0.001)


def entry_flags(df: pd.DataFrame, p: StrategyParams) -> pd.Series:
    """Return a string Series: '' = no signal, otherwise the setup name."""
    c = df["close"]
    uptrend = (c > df["ema_trend"]) & (df["ema_fast"] > df["ema_slow"])
    out = pd.Series("", index=df.index, dtype=object)

    if p.name in ("trend_pullback", "combo"):
        dipped = (df["rsi"].rolling(4).min() < p.rsi_dip)
        recover = (df["rsi"] > p.rsi_dip) & (df["rsi"].shift() <= df["rsi"]) & (df["macd_hist"] > df["macd_hist"].shift())
        out[uptrend & dipped & recover & (c > df["bb_lo"])] = "trend_pullback"

    if p.name in ("breakout", "combo"):
        hh = df["high"].rolling(p.breakout_bars).max().shift()
        brk = (c > hh) & (df["vol_z"] > p.min_vol_z) & (df["rsi"] < 78) & (c > df["ema_trend"])
        out[brk & (out == "")] = "breakout"
    return out


def usdt_regime(usdt: pd.DataFrame, p: StrategyParams) -> pd.Series:
    """True when the toman is weakening (USDTTMN uptrend) -> park idle cash in USDT."""
    c = usdt["close"]
    return (c.ewm(span=p.usdt_fast, adjust=False).mean() > c.ewm(span=p.usdt_slow, adjust=False).mean()) \
        & (c > c.ewm(span=p.usdt_slow, adjust=False).mean())


def build_signal(symbol: str, row: pd.Series, kind: str, p: StrategyParams) -> Signal:
    price = float(row["close"])
    a = float(row["atr"])
    stop = price - p.atr_stop * a
    target = price + p.atr_tp * a
    move = (target / price - 1)
    cost = round_trip_cost(symbol)
    reason = (f"{kind}: RSI {row['rsi']:.0f}, close {((price / row['ema_trend']) - 1) * 100:+.2f}% vs EMA96, "
              f"ATR {row['atr_pct'] * 100:.2f}%, expected {move * 100:.2f}% vs cost {cost * 100:.2f}%")
    return Signal(symbol, kind, price, stop, target, move * 100, cost * 100, reason)


def passes_edge_filter(sig: Signal) -> bool:
    return sig.expected_move_pct >= settings.fees.edge_multiple * sig.cost_pct


def latest_signal(symbol: str, df: pd.DataFrame, p: StrategyParams) -> Signal | None:
    """Signal on the last *closed* bar, or None."""
    flags = entry_flags(df, p)
    kind = flags.iloc[-1]
    if not kind:
        return None
    sig = build_signal(symbol, df.iloc[-1], kind, p)
    return sig if passes_edge_filter(sig) else None


def position_size_tmn(equity: float, price: float, stop: float, dd: float) -> float:
    """Risk-based sizing capped by max position fraction; halved past the circuit breaker."""
    r = settings.risk
    if dd >= r.hard_stop_dd:
        return 0.0
    risk_frac = max((price - stop) / price, 1e-4)
    size = equity * r.risk_per_trade_frac / risk_frac
    size = min(size, equity * r.max_position_frac)
    if dd >= r.circuit_breaker_dd:
        size *= 0.5
    return float(np.floor(size)) if size >= r.min_order_tmn else 0.0
