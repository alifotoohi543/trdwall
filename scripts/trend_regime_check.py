"""Can trend-following work with a bigger timeframe and/or a market-regime filter?

Same walk-forward split and cost model as scripts/research.py. Regime filter: BTC (USD basis)
above its EMA96 with EMA12>EMA26 on the same timeframe, i.e. only ride trends in a risk-on market.

Usage: python -m scripts.trend_regime_check
"""
import json
import warnings

import numpy as np
import pandas as pd

from src.config import ROOT
from src.market_data import add_indicators, load_history, resample
from src.research.fastsim import simulate
from src.research.strategies import EXITS, STRATEGIES, adx
from src.strategy import usd_frame

warnings.filterwarnings("ignore")
hs = json.load(open(ROOT / "reports" / "half_spreads.json"))
SYMS = sorted(hs)
TREND = [k for k, v in STRATEGIES.items() if v[1] == "trend"]
SIZE, MIN_N = 0.20, 6

rows = []
for tf in ("60min", "240min"):
    usdt = add_indicators(resample(load_history("USDTTMN"), tf))
    px = {s: add_indicators(resample(load_history(s), tf)) for s in SYMS}
    sig = {s: usd_frame(px[s], usdt) for s in SYMS}
    bph = pd.Timedelta("1h") / pd.Timedelta(tf)
    b = sig["BTCTMN"]
    ax, _, _ = adx(b)
    regimes = {"none": pd.Series(True, index=b.index),
               "btc_uptrend": (b["close"] > b["ema_trend"]) & (b["ema_fast"] > b["ema_slow"]),
               "btc_uptrend_adx": (b["close"] > b["ema_trend"]) & (b["ema_fast"] > b["ema_slow"]) & (ax > 20)}
    t0 = usdt.index[0] + pd.Timedelta(days=3)
    t_end = usdt.index[-1]
    t_split = t0 + (t_end - t0) * 0.67
    d_tr, d_te = (t_split - t0) / pd.Timedelta("1D"), (t_end - t_split) / pd.Timedelta("1D")
    for name in TREND + ["xs_momentum"]:
        if name == "xs_momentum":
            idx = None
            for d in sig.values():
                idx = d.index if idx is None else idx.intersection(d.index)
            r24 = pd.DataFrame({s: d["close"].reindex(idx).pct_change(max(1, int(24 * bph))) for s, d in sig.items()})
            configs = []
            for g in (0.03, 0.06):
                best = r24.eq(r24.max(axis=1), axis=0) & (r24 > g)
                configs.append(({"gain": g}, {s: best[s] & ~best[s].shift(1, fill_value=False) for s in SYMS}))
        else:
            fn, _, grid = STRATEGIES[name]
            configs = [(p, {s: fn(sig[s], **p) for s in SYMS}) for p in grid]
        for rname, reg in regimes.items():
            evals = []
            for params, flags in configs:
                for exn in ("swing", "trend"):
                    ex = EXITS[exn]
                    tr = []
                    for s in SYMS:
                        fl = flags[s].fillna(False) & reg.reindex(flags[s].index).fillna(False)
                        t = simulate(px[s], sig[s]["atr_pct"], fl, hs[s], ex["stop_atr"], ex["tp_atr"],
                                     max(1, int(ex["hold_h"] * bph)), ex["trail_act"], ex["trail_atr"])
                        tr.append(t)
                    tr = pd.concat(tr)
                    tr = tr[tr.t_signal >= t0]
                    evals.append((params, exn, tr[tr.t_signal < t_split], tr[tr.t_signal >= t_split]))
            ok = [e for e in evals if len(e[2]) >= MIN_N]
            if not ok:
                continue
            params, exn, trn, tst = max(ok, key=lambda e: e[2].ret.sum())
            test_all = [e[3].ret.sum() * SIZE * 100 / d_te for e in evals]
            rows.append({"tf": tf, "strategy": name, "regime": rname, "params": json.dumps(params), "exit": exn,
                         "train_n": len(trn), "train_%/day": round(trn.ret.sum() * SIZE * 100 / d_tr, 3),
                         "test_n": len(tst), "test_win%": round((tst.ret > 0).mean() * 100, 1) if len(tst) else None,
                         "TEST_%/day": round(tst.ret.sum() * SIZE * 100 / d_te, 3) if len(tst) else 0.0,
                         "all_cfg_median": round(float(np.median(test_all)), 3),
                         "all_cfg_pos%": round(float(np.mean([x > 0 for x in test_all])) * 100)})
    print(f"[{tf}] done", flush=True)

res = pd.DataFrame(rows).sort_values("TEST_%/day", ascending=False)
res.to_csv(ROOT / "reports" / "trend_regime_check.csv", index=False)
pd.set_option("display.width", 250)
print(res.head(20).to_string(index=False))
robust = res[(res["train_%/day"] > 0) & (res["TEST_%/day"] > 0) & (res["all_cfg_pos%"] >= 50) & (res.test_n >= 5)]
print("\nROBUST (train>0, test>0, >=50% of configs positive, >=5 test trades):")
print(robust.to_string(index=False) if len(robust) else "  none")
