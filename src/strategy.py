"""Rule-based signal layer (the "Trigger Filter").

The same functions drive the backtest and the live engine: they decide *when*
the agents are woken up and propose concrete entry/stop/target levels. The LLM
agents can only veto or downsize these proposals, never invent trades.

Design decisions (see README):
  * Signals are computed on the USD-denominated price (TMN price / USDTTMN) so that
    politically driven toman moves do not create fake technical signals; orders
    are still executed on the TMN market.
  * A USDTTMN shock filter blocks new entries and tightens stops when the toman
    moves sharply (usually news).
  * Entries are LIMIT orders at the signal price (no spread paid); take-profit is
    a LIMIT; stop-loss is a market exit (pays half-spread).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import settings
from src.market_data import add_indicators

# Typical half-spread per market (fraction), measured from live books; used as slippage.
HALF_SPREAD = {"BTCTMN": 0.0001, "ETHTMN": 0.0004, "SOLTMN": 0.001, "XRPTMN": 0.0012,
               "DOGETMN": 0.0007, "USDTTMN": 0.0002}


@dataclass
class StrategyParams:
    name: str = "oversold"          # oversold | pullback | mean_rev | both
    usd_signals: bool = True        # compute indicators on TMN/USDT price
    limit_entry: bool = True
    rsi_dip: float = 40.0           # pullback: RSI dipped below this inside an uptrend
    mr_rsi: float = 30.0            # mean reversion: oversold threshold
    capit_rsi: float = 25.0         # oversold/capitulation: buy the extreme (best edge in event study)
    atr_stop: float = 2.5
    atr_tp: float = 4.0
    trail_atr: float = 1.2
    trail_activate_atr: float = 3.0
    max_hold_bars: int = 16
    edge_multiple: float = settings.fees.edge_multiple
    shock_pct: float = 1.0          # |USDTTMN move| over shock_window that counts as news
    shock_window: str = "60min"
    shock_stop_atr: float = 0.75    # tightened stop distance during a shock


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


# ------------------------------------------------------------------- frames
def usd_frame(px: pd.DataFrame, usdt: pd.DataFrame) -> pd.DataFrame:
    """Divide a TMN OHLCV frame by the USDTTMN close of the same bar and recompute indicators."""
    u = usdt["close"].reindex(px.index).ffill()
    f = px[["open", "high", "low", "close"]].div(u, axis=0)
    f["volume"] = px["volume"]
    return add_indicators(f.dropna())


def usdt_shock(usdt: pd.DataFrame, p: StrategyParams) -> pd.Series:
    bars = max(1, int(pd.Timedelta(p.shock_window) / (usdt.index[1] - usdt.index[0])))
    return usdt["close"].pct_change(bars).abs() * 100 > p.shock_pct


# ------------------------------------------------------------------ signals
def entry_flags(df: pd.DataFrame, p: StrategyParams) -> pd.Series:
    """'' = no signal, otherwise the setup name. `df` is the signal frame (USD or TMN)."""
    c = df["close"]
    out = pd.Series("", index=df.index, dtype=object)
    if p.name in ("pullback", "both"):
        uptrend = (c > df["ema_trend"]) & (df["ema_fast"] > df["ema_slow"])
        dipped = df["rsi"].rolling(4).min() < p.rsi_dip
        recover = (df["rsi"] > p.rsi_dip) & (df["rsi"].diff() >= 0) & (df["macd_hist"].diff() > 0)
        out[uptrend & dipped & recover & (c > df["bb_lo"])] = "pullback"
    if p.name == "oversold":
        # capitulation: first bar that closes deeply oversold, not already in a free-fall day
        hit = df["rsi"] < p.capit_rsi
        first = hit & ~hit.shift(1, fill_value=False)
        not_freefall = c > c.shift(96) * 0.92
        out[first & not_freefall] = "oversold"
    if p.name in ("mean_rev", "both"):
        oversold = df["rsi"].rolling(3).min() < p.mr_rsi
        reclaim = (c > df["bb_lo"]) & (c.shift() <= df["bb_lo"].shift())
        not_crash = c > df["ema_trend"] * 0.97
        out[oversold & reclaim & not_crash & (out == "")] = "mean_rev"
    return out


def round_trip_cost(symbol: str, limit_entry: bool = True) -> float:
    spreads = 1 if limit_entry else 2         # stop exit always pays the spread
    return 2 * settings.fees.taker + spreads * HALF_SPREAD.get(symbol, 0.001)


def build_signal(symbol: str, sig_row: pd.Series, kind: str, p: StrategyParams, price: float) -> Signal:
    """Levels are set in TMN using the volatility (ATR%) measured on the signal frame."""
    a = float(sig_row["atr_pct"]) * price
    stop, target = price - p.atr_stop * a, price + p.atr_tp * a
    move = target / price - 1
    cost = round_trip_cost(symbol, p.limit_entry)
    reason = (f"{kind}: RSI(usd) {sig_row['rsi']:.0f}, {((sig_row['close'] / sig_row['ema_trend']) - 1) * 100:+.2f}% "
              f"vs EMA96, ATR {sig_row['atr_pct'] * 100:.2f}%, target {move * 100:.2f}% vs cost {cost * 100:.2f}%")
    return Signal(symbol, kind, price, stop, target, move * 100, cost * 100, reason)


def passes_edge_filter(sig: Signal, p: StrategyParams) -> bool:
    return sig.expected_move_pct >= p.edge_multiple * sig.cost_pct


def latest_signal(symbol: str, px: pd.DataFrame, sig: pd.DataFrame, usdt: pd.DataFrame,
                  p: StrategyParams) -> tuple[Signal | None, str]:
    """Signal on the last closed bar. Returns (signal, status) where status explains a None."""
    if bool(usdt_shock(usdt, p).iloc[-1]):
        return None, "usdt_shock"
    kind = entry_flags(sig, p).iloc[-1]
    if not kind:
        return None, "no_setup"
    s = build_signal(symbol, sig.iloc[-1], kind, p, float(px["close"].iloc[-1]))
    return (s, "ok") if passes_edge_filter(s, p) else (None, "edge_too_small")


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
