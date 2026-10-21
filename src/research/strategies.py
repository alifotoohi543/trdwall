"""Strategy library for research: well-known technical algorithms as entry-flag generators.

Every function takes a USD-basis OHLCV frame (with `add_indicators` columns) and returns a
boolean Series "enter long at this bar's close". Exits are handled uniformly by the simulator
using an exit profile, so strategies are compared on entry quality alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.market_data import atr, ema, rsi


# --------------------------------------------------------------- extra indicators
def supertrend_dir(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """+1 uptrend / -1 downtrend (classic SuperTrend)."""
    a = atr(df, n).values
    hl2 = ((df["high"] + df["low"]) / 2).values
    c = df["close"].values
    up, dn = hl2 - mult * a, hl2 + mult * a
    fu, fd = up.copy(), dn.copy()
    d = np.ones(len(c))
    for i in range(1, len(c)):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        d[i] = 1 if c[i] > fd[i - 1] else (-1 if c[i] < fu[i - 1] else d[i - 1])
    return pd.Series(d, index=df.index)


def adx(df: pd.DataFrame, n: int = 14):
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / tr
    ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, ndi


def cross_up(a: pd.Series, b) -> pd.Series:
    b_prev = b.shift() if isinstance(b, pd.Series) else b
    return (a > b) & (a.shift() <= b_prev)


# ------------------------------------------------------------------- strategies
def s_rsi_oversold(d, rsi_th=25):
    hit = d["rsi"] < rsi_th
    return hit & ~hit.shift(1, fill_value=False) & (d["close"] > d["close"].shift(96) * 0.92)


def s_bb_reversion(d, k=2.0):
    m, sd = d["close"].rolling(20).mean(), d["close"].rolling(20).std()
    lo = m - k * sd
    return cross_up(d["close"], lo) & (d["rsi"].rolling(4).min() < 35)


def s_zscore(d, z0=2.5, n=48):
    z = (d["close"] - d["close"].rolling(n).mean()) / d["close"].rolling(n).std()
    hit = z < -z0
    return hit & ~hit.shift(1, fill_value=False)


def s_connors_rsi2(d, th=5):
    r2 = rsi(d["close"], 2)
    return (r2 < th) & (r2.shift() >= th) & (d["close"] > d["close"].rolling(200).mean())


def s_stoch(d, th=15):
    ll, hh = d["low"].rolling(14).min(), d["high"].rolling(14).max()
    k = 100 * (d["close"] - ll) / (hh - ll).replace(0, np.nan)
    dd = k.rolling(3).mean()
    return cross_up(k, dd) & (k.shift() < th)


def s_vwap_dev(d, dev=1.5):
    hit = d["close"] < d["vwap"] * (1 - dev / 100)
    return hit & ~hit.shift(1, fill_value=False)


def s_ema_cross(d, f=12, s=26):
    return cross_up(ema(d["close"], f), ema(d["close"], s)) & (d["close"] > d["ema_trend"])


def s_macd(d, below_zero=True):
    sig = cross_up(d["macd"], d["macd_signal"])
    return sig & (d["macd"] < 0) if below_zero else sig & (d["close"] > d["ema_trend"])


def s_supertrend(d, n=10, mult=3.0):
    st = supertrend_dir(d, n, mult)
    return (st == 1) & (st.shift() == -1)


def s_donchian(d, n=48):
    return cross_up(d["close"], d["high"].rolling(n).max().shift())


def s_ichimoku(d, t=9, k=26, s=52):
    hh, ll = d["high"], d["low"]
    tenkan = (hh.rolling(t).max() + ll.rolling(t).min()) / 2
    kijun = (hh.rolling(k).max() + ll.rolling(k).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(k)
    span_b = ((hh.rolling(s).max() + ll.rolling(s).min()) / 2).shift(k)
    return cross_up(tenkan, kijun) & (d["close"] > np.maximum(span_a, span_b))


def s_squeeze(d, kc_mult=1.5):
    m, sd = d["close"].rolling(20).mean(), d["close"].rolling(20).std()
    a = atr(d, 20)
    squeeze = ((m + 2 * sd) < (m + kc_mult * a)) & ((m - 2 * sd) > (m - kc_mult * a))
    released = squeeze.shift().fillna(False).astype(bool) & ~squeeze
    return released & (d["close"] > m) & (d["macd_hist"] > 0)


def s_adx(d, th=25):
    ax, pdi, ndi = adx(d)
    return cross_up(pdi, ndi) & (ax > th)


def s_heikin(d, n_red=3):
    ha_c = (d["open"] + d["high"] + d["low"] + d["close"]) / 4
    ha_o = ha_c.copy()
    vals_o, vals_c = ha_o.values, ha_c.values
    o0 = (d["open"].values + d["close"].values) / 2
    vals_o[0] = o0[0]
    for i in range(1, len(vals_o)):
        vals_o[i] = (vals_o[i - 1] + vals_c[i - 1]) / 2
    green = pd.Series(vals_c > vals_o, index=d.index)
    reds = (~green).astype(int).rolling(n_red).sum().shift() == n_red
    return green & reds & (d["close"] > ema(d["close"], 50))


STRATEGIES = {
    # name: (fn, family, param grid)
    "rsi_oversold":  (s_rsi_oversold, "mr", [dict(rsi_th=20), dict(rsi_th=25), dict(rsi_th=30)]),
    "bb_reversion":  (s_bb_reversion, "mr", [dict(k=2.0), dict(k=2.5)]),
    "zscore":        (s_zscore, "mr", [dict(z0=2.0), dict(z0=2.5), dict(z0=3.0)]),
    "connors_rsi2":  (s_connors_rsi2, "mr", [dict(th=5), dict(th=10)]),
    "stochastic":    (s_stoch, "mr", [dict(th=10), dict(th=20)]),
    "vwap_dev":      (s_vwap_dev, "mr", [dict(dev=1.0), dict(dev=1.5), dict(dev=2.5)]),
    "ema_cross":     (s_ema_cross, "trend", [dict(f=9, s=21), dict(f=12, s=26), dict(f=20, s=50)]),
    "macd":          (s_macd, "trend", [dict(below_zero=True), dict(below_zero=False)]),
    "supertrend":    (s_supertrend, "trend", [dict(n=10, mult=2.0), dict(n=10, mult=3.0), dict(n=14, mult=3.0)]),
    "donchian":      (s_donchian, "trend", [dict(n=20), dict(n=48), dict(n=96)]),
    "ichimoku":      (s_ichimoku, "trend", [dict()]),
    "ttm_squeeze":   (s_squeeze, "trend", [dict(kc_mult=1.5), dict(kc_mult=2.0)]),
    "adx_di":        (s_adx, "trend", [dict(th=20), dict(th=25)]),
    "heikin_ashi":   (s_heikin, "trend", [dict(n_red=3), dict(n_red=5)]),
}

# exit profiles in hours; both are tried for every strategy and chosen on the training window
EXITS = {
    "mr":    dict(stop_atr=2.5, tp_atr=4.0, hold_h=4, trail_act=None, trail_atr=None),
    "swing": dict(stop_atr=2.0, tp_atr=6.0, hold_h=12, trail_act=2.0, trail_atr=1.5),
    "trend": dict(stop_atr=2.0, tp_atr=10.0, hold_h=24, trail_act=2.0, trail_atr=2.0),
}
