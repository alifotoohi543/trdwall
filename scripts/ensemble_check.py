"""Portfolio-level check of an ensemble of research winners (max concurrent positions, sizing).

Usage: python -m scripts.ensemble_check
"""
import json
import warnings

import numpy as np
import pandas as pd

from src.config import ROOT, settings
from src.market_data import add_indicators, load_history, resample
from src.research.fastsim import simulate
from src.research.strategies import EXITS, s_rsi_oversold, s_supertrend, s_zscore
from src.strategy import usd_frame

warnings.filterwarnings("ignore")
hs = json.load(open(ROOT / "reports" / "half_spreads.json"))
SYMS = sorted(hs)


def frames(tf):
    usdt = add_indicators(resample(load_history("USDTTMN"), tf))
    px = {s: add_indicators(resample(load_history(s), tf)) for s in SYMS}
    return px, {s: usd_frame(px[s], usdt) for s in SYMS}, usdt


def xs_rev(sig, bph, drop):
    idx = None
    for d in sig.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    r4 = pd.DataFrame({s: d["close"].reindex(idx).pct_change(4 * bph) for s, d in sig.items()})
    w = r4.eq(r4.min(axis=1), axis=0) & (r4 < -drop)
    return {s: w[s] & ~w[s].shift(1, fill_value=False) for s in sig}


def trades_for(name, tf, flags, px, sig, exn, bph):
    ex = EXITS[exn]
    out = []
    for s, fl in flags.items():
        t = simulate(px[s], sig[s]["atr_pct"], fl.fillna(False), hs[s], ex["stop_atr"], ex["tp_atr"],
                     int(ex["hold_h"] * bph), ex["trail_act"], ex["trail_atr"])
        # signal is known at bar close; entry happens during the next bar
        t["t_entry"] = t.t_signal + pd.Timedelta(tf)
        t["t_exit"] = t.t_exit + pd.Timedelta(tf)
        out.append(t.assign(symbol=s, strategy=name))
    return pd.concat(out)


def portfolio(tr, max_pos=3, size=0.2, start=None, end=None):
    tr = tr[(tr.t_entry >= start) & (tr.t_entry < end)].sort_values("t_entry")
    open_, taken, eq = [], [], 1.0
    for _, t in tr.iterrows():
        open_ = [o for o in open_ if o[0] > t.t_entry]
        if len(open_) >= max_pos or any(o[1] == t.symbol for o in open_):
            continue
        open_.append((t.t_exit, t.symbol))
        taken.append(t)
    tk = pd.DataFrame(taken)
    if not len(tk):
        return 0.0, 0.0, 0
    tk = tk.sort_values("t_exit")
    curve = 1 + (tk.ret * size).cumsum()            # realized equity, fixed sizing on initial capital
    dd = float((1 - curve / curve.cummax().clip(lower=1)).max())
    return float((curve.iloc[-1] - 1) * 100), dd * 100, len(tk)


px15, sig15, usdt15 = frames("15min")
px60, sig60, _ = frames("60min")
px240, sig240, _ = frames("240min")


def xs_mom(sig, bars, gain):
    idx = None
    for d in sig.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    r = pd.DataFrame({s: d["close"].reindex(idx).pct_change(bars) for s, d in sig.items()})
    b = r.eq(r.max(axis=1), axis=0) & (r > gain)
    return {s: b[s] & ~b[s].shift(1, fill_value=False) for s in sig}
t0 = usdt15.index[0] + pd.Timedelta(days=3)
t_end = usdt15.index[-1]
t_split = t0 + (t_end - t0) * 0.67

parts = {
    "zscore_1h":     trades_for("zscore_1h", "60min", {s: s_zscore(sig60[s], z0=3.0) for s in SYMS}, px60, sig60, "mr", 1),
    "xs_rev_15m":    trades_for("xs_rev_15m", "15min", xs_rev(sig15, 4, 0.05), px15, sig15, "trend", 4),
    "rsi20_15m":     trades_for("rsi20_15m", "15min", {s: s_rsi_oversold(sig15[s], rsi_th=20) for s in SYMS}, px15, sig15, "trend", 4),
    "supertrend_4h": trades_for("supertrend_4h", "240min", {s: s_supertrend(sig240[s], 14, 3.0) for s in SYMS}, px240, sig240, "swing", 0.25),
    "momentum_4h":   trades_for("momentum_4h", "240min", xs_mom(sig240, 6, 0.06), px240, sig240, "swing", 0.25),
    "rsi25_15m_cur": trades_for("rsi25_15m_cur", "15min", {s: s_rsi_oversold(sig15[s], rsi_th=25) for s in SYMS}, px15, sig15, "mr", 4),
}
combos = {
    "current bot (rsi25 15m, 5 coins)": parts["rsi25_15m_cur"][parts["rsi25_15m_cur"].symbol.isin(["BTCTMN", "ETHTMN", "SOLTMN", "XRPTMN", "DOGETMN"])],
    "current bot on 14 coins": parts["rsi25_15m_cur"],
    "zscore_1h": parts["zscore_1h"],
    "xs_rev_15m": parts["xs_rev_15m"],
    "zscore_1h + xs_rev_15m": pd.concat([parts["zscore_1h"], parts["xs_rev_15m"]]),
    "zscore_1h + xs_rev_15m + rsi20_15m": pd.concat([parts["zscore_1h"], parts["xs_rev_15m"], parts["rsi20_15m"]]),
    "trend only (supertrend_4h + momentum_4h)": pd.concat([parts["supertrend_4h"], parts["momentum_4h"]]),
    "ALL 5 (dip + trend)": pd.concat([parts[k] for k in ("zscore_1h", "xs_rev_15m", "rsi20_15m", "supertrend_4h", "momentum_4h")]),
}
d_tr, d_te = (t_split - t0).days + (t_split - t0).seconds / 86400, (t_end - t_split).days + (t_end - t_split).seconds / 86400
rows = []
for name, tr in combos.items():
    r_tr, dd_tr, n_tr = portfolio(tr, start=t0, end=t_split)
    r_te, dd_te, n_te = portfolio(tr, start=t_split, end=t_end)
    wins = []
    for s in pd.date_range(t0, t_end - pd.Timedelta(days=3), freq="12h"):
        wins.append(portfolio(tr, start=s, end=s + pd.Timedelta(days=3))[0])
    w = np.array(wins)
    rows.append({"ensemble": name, "train_ret%": round(r_tr, 2), "train_maxdd%": round(dd_tr, 2), "train_n": n_tr,
                 "TEST_ret%": round(r_te, 2), "TEST_maxdd%": round(dd_te, 2), "TEST_n": n_te,
                 "trades/day": round((n_tr + n_te) / (d_tr + d_te), 2),
                 "3d_mean%": round(w.mean(), 2), "3d_worst%": round(w.min(), 2), "3d_pos%": round((w > 0).mean() * 100)})
res = pd.DataFrame(rows)
res.to_csv(ROOT / "reports" / "ensemble_check.csv", index=False)
pd.set_option("display.width", 250)
print(f"train {d_tr:.1f}d | test {d_te:.1f}d | max 3 positions, 20% each, all costs included")
print(res.to_string(index=False))
