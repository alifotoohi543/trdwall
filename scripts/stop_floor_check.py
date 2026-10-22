"""Minimum stop-distance floor: does a stop that never sits inside the noise improve the ensemble?

Live finding (2026-10-22 night): 2xATR(15m) stops were 0.4-0.7% (quiet hours) and two trades were
stopped within ~40 s. Same walk-forward split / portfolio rules (14 coins, 4 x 18%) as universe_check.

Usage: python -m scripts.stop_floor_check
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
SYMS = list(settings.symbols)
HS = json.load(open(ROOT / "reports" / "half_spreads.json"))


def frames(tf):
    usdt = add_indicators(resample(load_history("USDTTMN"), tf))
    px = {s: add_indicators(resample(load_history(s), tf)) for s in SYMS}
    return px, {s: usd_frame(px[s], usdt) for s in SYMS}


def xs(sig, bars, thr, worst):
    idx = None
    for d in sig.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    r = pd.DataFrame({s: d["close"].reindex(idx).pct_change(bars) for s, d in sig.items()})
    pick = r.eq(r.min(axis=1) if worst else r.max(axis=1), axis=0) & ((r < -thr) if worst else (r > thr))
    return {s: pick[s] & ~pick[s].shift(1, fill_value=False) for s in sig}


F = {tf: frames(tf) for tf in ("15min", "60min", "240min")}
SIGNALS = {  # name: (tf, exit, bars-per-hour, flags)
    "zscore_1h": ("60min", "mr", 1, {s: s_zscore(F["60min"][1][s], z0=3.0) for s in SYMS}),
    "xs_reversal": ("15min", "trend", 4, xs(F["15min"][1], 16, 0.05, True)),
    "oversold_rsi20": ("15min", "trend", 4, {s: s_rsi_oversold(F["15min"][1][s], 20) for s in SYMS}),
    "supertrend_4h": ("240min", "swing", 0.25, {s: s_supertrend(F["240min"][1][s], 14, 3.0) for s in SYMS}),
    "momentum_4h": ("240min", "swing", 0.25, xs(F["240min"][1], 6, 0.06, False)),
}


def ensemble(min_stop):
    out = []
    for name, (tf, exn, bph, flags) in SIGNALS.items():
        px, sig = F[tf]
        ex = EXITS[exn]
        for s, fl in flags.items():
            t = simulate(px[s], sig[s]["atr_pct"], fl.fillna(False), HS.get(s, 0.002), ex["stop_atr"], ex["tp_atr"],
                         max(1, int(ex["hold_h"] * bph)), ex["trail_act"], ex["trail_atr"], min_stop)
            t["t_entry"], t["t_exit"] = t.t_signal + pd.Timedelta(tf), t.t_exit + pd.Timedelta(tf)
            out.append(t.assign(symbol=s, strategy=name))
    return pd.concat(out)


def portfolio(tr, start, end, max_pos=4, size=0.18):
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
        return 0.0, 0.0, 0, tk
    tk = tk.sort_values("t_exit")
    curve = 1 + (tk.ret * size).cumsum()
    return float((curve.iloc[-1] - 1) * 100), float((1 - curve / curve.cummax().clip(lower=1)).max() * 100), len(tk), tk


idx = F["15min"][0]["BTCTMN"].index
t0, t_end = idx[0] + pd.Timedelta(days=3), idx[-1]
t_split = t0 + (t_end - t0) * 0.67
rows = []
for floor in (0.0, 0.008, 0.012, 0.015, 0.02):
    tr = ensemble(floor)
    r_tr, _, n_tr, _ = portfolio(tr, t0, t_split)
    r_te, dd_te, n_te, tk_te = portfolio(tr, t_split, t_end)
    r_l3, dd_l3, _, _ = portfolio(tr, t_end - pd.Timedelta(days=3), t_end)
    w = np.array([portfolio(tr, s, s + pd.Timedelta(days=3))[0]
                  for s in pd.date_range(t0, t_end - pd.Timedelta(days=3), freq="12h")])
    fast_stops = int(((tk_te.reason == "stop") & (tk_te.bars <= 1)).sum()) if len(tk_te) else 0
    rows.append({"min_stop": f"{floor * 100:.1f}%", "train%": round(r_tr, 2), "TEST%": round(r_te, 2),
                 "TEST_dd%": round(dd_te, 2), "last3d%": round(r_l3, 2), "last3d_dd%": round(dd_l3, 2),
                 "test_win%": round((tk_te.ret > 0).mean() * 100) if len(tk_te) else None,
                 "stopped_in_1_bar(test)": fast_stops, "trades": n_tr + n_te,
                 "3d_mean%": round(w.mean(), 2), "3d_worst%": round(w.min(), 2), "3d_pos%": round((w > 0).mean() * 100)})
res = pd.DataFrame(rows)
res.to_csv(ROOT / "reports" / "stop_floor_check.csv", index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))
