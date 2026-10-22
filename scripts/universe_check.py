"""Does a bigger coin universe improve the 5-algorithm ensemble?

Compares the current 14-coin universe with 14 + next-tier coins (data/universe_extra.txt) on the same
walk-forward split, rolling 3-day windows and the last 3 days, with 3 or 4 concurrent positions.
Extra coins get a conservative spread floor (their books are thinner).

Usage: python -m scripts.universe_check
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
BASE = list(settings.symbols)
EXTRA = (ROOT / "data" / "universe_extra.txt").read_text().split()
HS = json.load(open(ROOT / "reports" / "half_spreads.json"))
HS.update({k: max(v, 0.001) for k, v in json.load(open(ROOT / "reports" / "half_spreads_extra.json")).items()})


def frames(syms, tf):
    usdt = add_indicators(resample(load_history("USDTTMN"), tf))
    px = {s: add_indicators(resample(load_history(s), tf)) for s in syms}
    return px, {s: usd_frame(px[s], usdt) for s in syms}, usdt


def xs(sig, bars, thr, worst=True):
    idx = None
    for d in sig.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    r = pd.DataFrame({s: d["close"].reindex(idx).pct_change(bars) for s, d in sig.items()})
    pick = r.eq(r.min(axis=1) if worst else r.max(axis=1), axis=0) & ((r < -thr) if worst else (r > thr))
    return {s: pick[s] & ~pick[s].shift(1, fill_value=False) for s in sig}


def trades(name, tf, flags, px, sig, exn, bph, min_stop=0.0):
    ex, out = EXITS[exn], []
    for s, fl in flags.items():
        t = simulate(px[s], sig[s]["atr_pct"], fl.fillna(False), HS.get(s, 0.002), ex["stop_atr"], ex["tp_atr"],
                     max(1, int(ex["hold_h"] * bph)), ex["trail_act"], ex["trail_atr"], min_stop)
        t["t_entry"], t["t_exit"] = t.t_signal + pd.Timedelta(tf), t.t_exit + pd.Timedelta(tf)
        out.append(t.assign(symbol=s, strategy=name))
    return pd.concat(out)


def ensemble(syms):
    p15, s15, _ = frames(syms, "15min")
    p60, s60, _ = frames(syms, "60min")
    p240, s240, _ = frames(syms, "240min")
    return pd.concat([
        trades("zscore_1h", "60min", {s: s_zscore(s60[s], z0=3.0) for s in syms}, p60, s60, "mr", 1),
        trades("xs_reversal", "15min", xs(s15, 16, 0.05, worst=True), p15, s15, "trend", 4),
        trades("oversold_rsi20", "15min", {s: s_rsi_oversold(s15[s], 20) for s in syms}, p15, s15, "trend", 4),
        trades("supertrend_4h", "240min", {s: s_supertrend(s240[s], 14, 3.0) for s in syms}, p240, s240, "swing", 0.25),
        trades("momentum_4h", "240min", xs(s240, 6, 0.06, worst=False), p240, s240, "swing", 0.25),
    ])


def portfolio(tr, start, end, max_pos, size=0.2):
    tr = tr[(tr.t_entry >= start) & (tr.t_entry < end)].sort_values("t_entry")
    open_, taken = [], []
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
    curve = 1 + (tk.ret * size).cumsum()
    return float((curve.iloc[-1] - 1) * 100), float((1 - curve / curve.cummax().clip(lower=1)).max() * 100), len(tk)


usdt_idx = add_indicators(resample(load_history("USDTTMN"), "15min")).index
t0, t_end = usdt_idx[0] + pd.Timedelta(days=3), usdt_idx[-1]
t_split = t0 + (t_end - t0) * 0.67
d_all = (t_end - t0) / pd.Timedelta("1D")
universes = {"current 14": BASE, f"14 + {len(EXTRA)} extra = {len(BASE) + len(EXTRA)}": BASE + EXTRA}
rows = []
for uname, syms in universes.items():
    tr = ensemble(syms)
    for max_pos, size in ((3, 0.20), (4, 0.18)):
        r_tr, dd_tr, n_tr = portfolio(tr, t0, t_split, max_pos, size)
        r_te, dd_te, n_te = portfolio(tr, t_split, t_end, max_pos, size)
        r_l3, dd_l3, n_l3 = portfolio(tr, t_end - pd.Timedelta(days=3), t_end, max_pos, size)
        w = np.array([portfolio(tr, s, s + pd.Timedelta(days=3), max_pos, size)[0]
                      for s in pd.date_range(t0, t_end - pd.Timedelta(days=3), freq="12h")])
        rows.append({"universe": uname, "max_pos": max_pos, "size": size,
                     "train%": round(r_tr, 2), "TEST%": round(r_te, 2), "TEST_dd%": round(dd_te, 2),
                     "last3d%": round(r_l3, 2), "trades/day": round((n_tr + n_te) / d_all, 2),
                     "signals/day": round(len(tr[tr.t_entry >= t0]) / d_all, 1),
                     "3d_mean%": round(w.mean(), 2), "3d_worst%": round(w.min(), 2), "3d_pos%": round((w > 0).mean() * 100)})
    by = tr[tr.t_entry >= t_split].groupby("symbol").ret.agg(["count", "mean"]).round(4)
    if uname != "current 14":
        print("extra coins in TEST period (n, avg ret):")
        print(by[by.index.isin(EXTRA)].to_string())
res = pd.DataFrame(rows)
res.to_csv(ROOT / "reports" / "universe_check.csv", index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))
