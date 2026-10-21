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

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import ROOT, settings
from src.market_data import add_indicators

# Typical half-spread per market (fraction), measured from live books; used as slippage.
HALF_SPREAD = {"BTCTMN": 0.0001, "ETHTMN": 0.0004, "SOLTMN": 0.001, "XRPTMN": 0.0012,
               "DOGETMN": 0.0007, "USDTTMN": 0.0002}
_hs_file = ROOT / "reports" / "half_spreads.json"
if _hs_file.exists():   # measured for the whole research universe by scripts/research.py
    HALF_SPREAD.update({k: max(v, HALF_SPREAD.get(k, 0.0)) for k, v in json.loads(_hs_file.read_text()).items()})


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


# Base rates from scripts/run_backtest.py + event study (27 days of Wallex 15m data), shown to the agents.
SETUP_BASE_RATES = {
    "zscore_1h": "1h close >3 std below its 48h mean (USD basis), walk-forward OUT-OF-SAMPLE: win-rate 71%, "
                 "avg +1.4%/trade net (n=17 over 11 coins); also positive in training. Mean-reversion, exits within 4h",
    "xs_reversal": "coin with the worst 4h USD return in the 14-coin universe and below -5%: OUT-OF-SAMPLE win-rate 54%, "
                   "avg +2.3%/trade net (n=13), positive in 78% of parameter variants; gains mostly from volatile "
                   "meme coins; trailing stop, holds up to 24h",
    "oversold_rsi20": "first 15m bar with USD-basis RSI<20: OUT-OF-SAMPLE win-rate 62%, avg +0.9%/trade (n=8); "
                      "trailing stop, holds up to 24h",
    "oversold": "first 15m bar with USD-basis RSI<25: mean fwd 4h return +1.34% gross (n=120 bars), "
                "backtest net win-rate 56%; edge weakens when 24h drop exceeds ~6% or during USDT shocks",
    "pullback": "RSI dip inside uptrend: historically weak after fees (few samples)",
    "mean_rev": "Bollinger reclaim after oversold: ~0% fwd 4h, weak",
}


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
    hold_h: float = 4.0                    # time exit
    trail_act: float | None = None         # trailing activation (ATR multiples of profit)
    trail_atr: float | None = None         # trailing distance (ATR)
    atr_tmn: float = 0.0                   # ATR of the signal timeframe, in TMN
    bar: str = ""                          # signal bar timestamp (dedupe key)

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


# ====================================================================== ensemble
# Chosen by scripts/research.py (walk-forward) + scripts/ensemble_check.py (portfolio level).
EXIT_PROFILES = {
    "mr":    dict(stop_atr=2.5, tp_atr=4.0, hold_h=4, trail_act=None, trail_atr=None),
    "trend": dict(stop_atr=2.0, tp_atr=10.0, hold_h=24, trail_act=2.0, trail_atr=2.0),
}
ENSEMBLE_PRIORITY = ["xs_reversal", "zscore_1h", "oversold_rsi20"]
XS_DROP = 0.05


def make_signal(symbol: str, kind: str, sig_row: pd.Series, price: float, exit_name: str, bar) -> Signal:
    ex = EXIT_PROFILES[exit_name]
    a = float(sig_row["atr_pct"]) * price
    stop, target = price - ex["stop_atr"] * a, price + ex["tp_atr"] * a
    cost = round_trip_cost(symbol, True)
    move = target / price - 1
    reason = (f"{kind}: RSI(usd) {sig_row['rsi']:.0f}, {((sig_row['close'] / sig_row['ema_trend']) - 1) * 100:+.2f}% vs EMA96, "
              f"ATR {sig_row['atr_pct'] * 100:.2f}%, stop -{ex['stop_atr'] * a / price * 100:.2f}% "
              f"target +{move * 100:.2f}% hold<= {ex['hold_h']}h, cost {cost * 100:.2f}%")
    return Signal(symbol, kind, price, stop, target, move * 100, cost * 100, reason,
                  hold_h=ex["hold_h"], trail_act=ex["trail_act"], trail_atr=ex["trail_atr"], atr_tmn=a, bar=str(bar))


def ensemble_signals(px15: dict, sig15: dict, sig60: dict, usdt15: pd.DataFrame,
                     now: pd.Timestamp, p: StrategyParams | None = None) -> tuple[list[Signal], dict]:
    """Signals on the last closed bars + per-symbol diagnostics for the scan log.

    The 1h z-score only fires right after its hour closes (fresh), never on a stale bar.
    """
    from src.research.strategies import s_rsi_oversold, s_zscore   # same code as the research

    p = p or StrategyParams()
    diag = {s: {} for s in px15}
    shock = bool(usdt_shock(usdt15, p).iloc[-1])
    out: list[Signal] = []

    # cross-sectional reversal (15m, needs the whole universe on a common index)
    idx = None
    for d in sig15.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    r4 = pd.DataFrame({s: d["close"].reindex(idx).pct_change(16) for s, d in sig15.items()})
    worst = r4.eq(r4.min(axis=1), axis=0) & (r4 < -XS_DROP)
    for s in sig15:
        diag[s]["r4h_usd"] = round(float(r4[s].iloc[-1]) * 100, 2) if pd.notna(r4[s].iloc[-1]) else None
        fl = worst[s] & ~worst[s].shift(1, fill_value=False)
        if bool(fl.iloc[-1]) and idx[-1] == sig15[s].index[-1]:
            out.append(make_signal(s, "xs_reversal", sig15[s].iloc[-1], float(px15[s]["close"].iloc[-1]), "trend", idx[-1]))

    for s in sig15:
        d15 = sig15[s]
        diag[s]["rsi_usd"] = round(float(d15["rsi"].iloc[-1]), 1)
        if bool(s_rsi_oversold(d15, rsi_th=20).iloc[-1]):
            out.append(make_signal(s, "oversold_rsi20", d15.iloc[-1], float(px15[s]["close"].iloc[-1]), "trend", d15.index[-1]))
        d60 = sig60.get(s)
        if d60 is not None and len(d60) > 60:
            z = (d60["close"] - d60["close"].rolling(48).mean()) / d60["close"].rolling(48).std()
            diag[s]["z_1h"] = round(float(z.iloc[-1]), 2)
            fresh = now - (d60.index[-1] + pd.Timedelta("60min")) < pd.Timedelta("15min")
            if fresh and bool(s_zscore(d60, z0=3.0).iloc[-1]):
                out.append(make_signal(s, "zscore_1h", d60.iloc[-1], float(px15[s]["close"].iloc[-1]), "mr", d60.index[-1]))

    rank = {k: i for i, k in enumerate(ENSEMBLE_PRIORITY)}
    out.sort(key=lambda x: rank[x.kind])
    if shock:
        for x in out:
            diag[x.symbol]["status"] = f"usdt_shock({x.kind})"
        return [], diag
    kept = []
    for x in out:
        ok = x.expected_move_pct >= p.edge_multiple * x.cost_pct
        diag[x.symbol]["status"] = x.kind if ok else f"edge_too_small({x.kind})"
        if ok and x.symbol not in {k.symbol for k in kept}:
            kept.append(x)
    return kept, diag
