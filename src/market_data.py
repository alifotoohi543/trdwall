"""Market data: candle storage/resampling, technical indicators, orderbook metrics.

Pure pandas/numpy — no LLM involvement. Used identically by backtest, shadow and live.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import HIST_DIR
from src.wallex_client import WallexClient

OHLCV = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


# ------------------------------------------------------------------ candles
def resample(df_1m: pd.DataFrame, timeframe: str = "15min") -> pd.DataFrame:
    """Resample 1m bars; Wallex omits minutes with no trades, so forward-fill closes."""
    out = df_1m.resample(timeframe, label="left", closed="left").agg(OHLCV)
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    return out.dropna()


def hist_path(symbol: str) -> Path:
    return HIST_DIR / f"{symbol}_1m.csv"


def load_history(symbol: str) -> pd.DataFrame:
    p = hist_path(symbol)
    if not p.exists():
        return pd.DataFrame(columns=list(OHLCV))
    return pd.read_csv(p, index_col="time", parse_dates=["time"])


def update_history(client: WallexClient, symbol: str, days: int = 30, chunk_days: int = 2) -> pd.DataFrame:
    """Download 1m candles in chunks and merge into data/historical/<symbol>_1m.csv."""
    old = load_history(symbol)
    end = int(time.time())
    start = end - days * 86400
    if not old.empty:
        start = max(start, int(old.index[-1].timestamp()) - 3600)
    frames = [old] if not old.empty else []
    t = start
    while t < end:
        chunk = client.candles_1m(symbol, t, min(t + chunk_days * 86400, end))
        if not chunk.empty:
            frames.append(chunk)
        t += chunk_days * 86400
    if not frames:
        return old
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "time"
    df.to_csv(hist_path(symbol))
    return df


# --------------------------------------------------------------- indicators
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["ema_fast"] = ema(c, 12)
    df["ema_slow"] = ema(c, 26)
    df["ema_trend"] = ema(c, 96)                     # ~1 day on 15m bars
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["rsi"] = rsi(c, 14)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    df["bb_mid"], df["bb_up"], df["bb_lo"] = mid, mid + 2 * sd, mid - 2 * sd
    df["bb_pctb"] = (c - df["bb_lo"]) / (df["bb_up"] - df["bb_lo"])
    df["atr"] = atr(df, 14)
    df["atr_pct"] = df["atr"] / c
    tp = (df["high"] + df["low"] + c) / 3
    vol = df["volume"].replace(0, np.nan)
    df["vwap"] = (tp * vol).rolling(32, min_periods=8).sum() / vol.rolling(32, min_periods=8).sum()
    df["vwap"] = df["vwap"].fillna(c)
    df["ret_1"] = c.pct_change()
    df["ret_4"] = c.pct_change(4)
    df["vol_z"] = (df["volume"] - df["volume"].rolling(48).mean()) / df["volume"].rolling(48).std()
    return df


# ---------------------------------------------------------------- orderbook
def orderbook_metrics(depth: dict, band_pct: float = 0.5) -> dict:
    """Spread and bid/ask value imbalance within ±band_pct of mid.

    Wallex books are fragmented by bots quoting dust at 1-toman steps, so a
    "top N levels" imbalance is noise; a price band is far more stable.
    """
    bids = sorted(depth.get("bid", []), key=lambda x: -float(x["price"]))
    asks = sorted(depth.get("ask", []), key=lambda x: float(x["price"]))
    if not bids or not asks:
        return {"bid": None, "ask": None, "mid": None, "spread_pct": None, "ob_imbalance": None}
    bb, ba = float(bids[0]["price"]), float(asks[0]["price"])
    mid = (bb + ba) / 2
    lo, hi = mid * (1 - band_pct / 100), mid * (1 + band_pct / 100)
    bid_val = sum(float(b["price"]) * float(b["quantity"]) for b in bids if float(b["price"]) >= lo)
    ask_val = sum(float(a["price"]) * float(a["quantity"]) for a in asks if float(a["price"]) <= hi)
    return {"bid": bb, "ask": ba, "mid": mid, "spread_pct": (ba - bb) / mid * 100,
            "ob_imbalance": bid_val / ask_val if ask_val else None,
            "bid_depth_tmn": bid_val, "ask_depth_tmn": ask_val}


def snapshot(df: pd.DataFrame, ob: dict | None = None) -> dict:
    """Compact, rounded feature dict for the latest bar (kept small to save LLM tokens)."""
    r = df.iloc[-1]
    snap = {
        "price": float(r["close"]),
        "rsi": round(float(r["rsi"]), 1),
        "macd_hist_pct": round(float(r["macd_hist"] / r["close"] * 100), 3),
        "bb_pctb": round(float(r["bb_pctb"]), 2),
        "atr_pct": round(float(r["atr_pct"] * 100), 2),
        "above_vwap_pct": round(float((r["close"] / r["vwap"] - 1) * 100), 2),
        "trend_ema96_pct": round(float((r["close"] / r["ema_trend"] - 1) * 100), 2),
        "ret_1h_pct": round(float(r["ret_4"] * 100), 2),
        "ret_24h_pct": round(float((df["close"].iloc[-1] / df["close"].iloc[-97] - 1) * 100), 2) if len(df) > 97 else None,
        "vol_z": round(float(r["vol_z"]), 1) if pd.notna(r["vol_z"]) else None,
    }
    if ob:
        snap["spread_pct"] = round(ob["spread_pct"], 3) if ob.get("spread_pct") is not None else None
        snap["ob_imbalance"] = round(ob["ob_imbalance"], 2) if ob.get("ob_imbalance") else None
    return snap
