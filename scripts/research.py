"""Strategy research with walk-forward validation.

For every algorithm in the library (+ cross-sectional, BTC-lead and a gradient-boosting model):
  1. choose parameters + exit profile on the TRAIN window only,
  2. report the chosen config on the unseen TEST window (out-of-sample),
  3. report how robust the algorithm is across all of its configs.
Everything is net of 0.25%/side fees and each market's real spread, on USD-basis signals.

Usage: python -m scripts.research [--tf 15min 60min]
Outputs: reports/research_results.csv, reports/research_trades.csv
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import ROOT, settings
from src.market_data import add_indicators, load_history, resample
from src.research.fastsim import label_every_bar, simulate
from src.market_data import rsi
from src.research.strategies import EXITS, STRATEGIES, adx
from src.strategy import usd_frame
from src.wallex_client import WallexClient

warnings.filterwarnings("ignore")
OUT = ROOT / "reports"
SIZE = 0.20               # capital fraction per trade, to express results as % of account
MIN_TRAIN_TRADES = 8


def load(symbols, tf):
    usdt = add_indicators(resample(load_history("USDTTMN"), tf))
    px, sig = {}, {}
    for s in symbols:
        h = load_history(s)
        if len(h) < 10_000:
            continue
        px[s] = add_indicators(resample(h, tf))
        sig[s] = usd_frame(px[s], usdt)
    return px, sig, usdt


def half_spreads(symbols) -> dict[str, float]:
    m = WallexClient().markets()
    out = {}
    for s in symbols:
        st = m[s]["stats"]
        try:
            bid, ask = float(st["bidPrice"]), float(st["askPrice"])
            out[s] = max((ask - bid) / (ask + bid), 0.0001)
        except (TypeError, ValueError, KeyError):
            out[s] = 0.002
    return out


def stats(tr: pd.DataFrame, days: float) -> dict:
    if not len(tr):
        return {"n": 0, "win%": None, "avg%": None, "pct_per_day": 0.0}
    return {"n": len(tr), "win%": round((tr.ret > 0).mean() * 100, 1), "avg%": round(tr.ret.mean() * 100, 3),
            "pct_per_day": round(tr.ret.sum() * SIZE * 100 / days, 3)}


# ------------------------------------------------------ extra (panel) strategies
def panel_flags(sig: dict[str, pd.DataFrame], bph: int) -> dict[str, list[tuple[dict, dict[str, pd.Series]]]]:
    idx = None
    for d in sig.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    r4 = pd.DataFrame({s: d["close"].reindex(idx).pct_change(4 * bph) for s, d in sig.items()})
    r24 = pd.DataFrame({s: d["close"].reindex(idx).pct_change(24 * bph) for s, d in sig.items()})
    r1 = pd.DataFrame({s: d["close"].reindex(idx).pct_change(bph) for s, d in sig.items()})
    out = {"xs_reversal": [], "xs_momentum": [], "btc_lead": []}
    for x in (0.02, 0.03, 0.05):
        worst = r4.eq(r4.min(axis=1), axis=0) & (r4 < -x)
        out["xs_reversal"].append(({"drop": x}, {s: worst[s] & ~worst[s].shift(1, fill_value=False) for s in sig}))
    for x in (0.03, 0.06):
        best = r24.eq(r24.max(axis=1), axis=0) & (r24 > x)
        out["xs_momentum"].append(({"gain": x}, {s: best[s] & ~best[s].shift(1, fill_value=False) for s in sig}))
    if "BTCTMN" in r1:
        for x, lag in ((0.008, 0.006), (0.012, 0.008)):
            btc_up = r1["BTCTMN"] > x
            fl = {s: btc_up & (r1[s] < r1["BTCTMN"] - lag) & ~(btc_up.shift(1, fill_value=False))
                  for s in sig if s != "BTCTMN"}
            out["btc_lead"].append(({"btc_1h": x, "lag": lag}, fl))
    return out


# --------------------------------------------------------------------- ML model
def features(d: pd.DataFrame, btc: pd.DataFrame, usdt: pd.DataFrame, bph: int) -> pd.DataFrame:
    c = d["close"]
    ax, pdi, ndi = adx(d)
    ll, hh = d["low"].rolling(14).min(), d["high"].rolling(14).max()
    f = pd.DataFrame({
        "rsi": d["rsi"], "rsi2": rsi(c, 2), "macd_h": d["macd_hist"] / c, "pctb": d["bb_pctb"],
        "atr": d["atr_pct"], "vwap": c / d["vwap"] - 1, "trend": c / d["ema_trend"] - 1,
        "z48": (c - c.rolling(48).mean()) / c.rolling(48).std(), "stoch": (c - ll) / (hh - ll),
        "adx": ax, "di": pdi - ndi, "vol_z": d["vol_z"],
        "r1": c.pct_change(), "r1h": c.pct_change(bph), "r4h": c.pct_change(4 * bph), "r24h": c.pct_change(24 * bph),
        "btc_r1h": btc["close"].reindex(d.index).pct_change(bph),
        "btc_r4h": btc["close"].reindex(d.index).pct_change(4 * bph),
        "usdt_r1h": usdt["close"].reindex(d.index).pct_change(bph),
        "usdt_r4h": usdt["close"].reindex(d.index).pct_change(4 * bph),
        "hour": d.index.hour,
    })
    return f.replace([np.inf, -np.inf], np.nan)


def run_ml(px, sig, usdt, hs, bph, t_split, t_end, embargo):
    btc = sig.get("BTCTMN", next(iter(sig.values())))
    rows = []
    for exn in ("mr", "swing"):
        ex = EXITS[exn]
        X, Y = [], []
        for s in sig:
            f = features(sig[s], btc, usdt, bph)
            y = label_every_bar(px[s], sig[s]["atr_pct"], hs[s], ex["stop_atr"], ex["tp_atr"],
                                int(ex["hold_h"] * bph), ex["trail_act"], ex["trail_atr"])
            f["sym"] = s
            X.append(f)
            Y.append(y)
        X, Y = pd.concat(X), pd.concat(Y)
        ok = Y.notna() & X.drop(columns="sym").notna().all(axis=1)
        X, Y = X[ok], Y[ok]
        train = X.index < (t_split - embargo)
        val_cut = X.index[train].sort_values()[int(train.sum() * 0.75)]
        tr_core, val = train & (X.index < val_cut), train & (X.index >= val_cut)
        feats = [c for c in X.columns if c != "sym"]
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0)
        m.fit(X.loc[tr_core, feats], (Y[tr_core] > 0).astype(int))
        pv = pd.Series(m.predict_proba(X.loc[val, feats])[:, 1], index=X.index[val])
        best_th, best = None, -1e9
        for th in (0.5, 0.55, 0.6, 0.65, 0.7):
            sel = pv > th
            if sel.sum() >= MIN_TRAIN_TRADES and Y[val][sel.values].mean() > best:
                best_th, best = th, Y[val][sel.values].mean()
        if best_th is None:
            continue
        m.fit(X.loc[train, feats], (Y[train] > 0).astype(int))
        for split, mask_fn in (("train", lambda ix: ix < t_split), ("test", lambda ix: (ix >= t_split) & (ix <= t_end))):
            trades = []
            for s in sig:
                xs = X[(X.sym == s)]
                xs = xs[mask_fn(xs.index)]
                if not len(xs):
                    continue
                p = pd.Series(m.predict_proba(xs[feats])[:, 1], index=xs.index)
                tr = simulate(px[s], sig[s]["atr_pct"], p > best_th, hs[s], ex["stop_atr"], ex["tp_atr"],
                              int(ex["hold_h"] * bph), ex["trail_act"], ex["trail_atr"])
                trades.append(tr.assign(symbol=s))
            rows.append((exn, best_th, split, pd.concat(trades) if trades else pd.DataFrame(columns=["ret"])))
    return rows


# ------------------------------------------------------------------------- main
def main(tfs):
    symbols = sorted(set((ROOT / "data" / "universe.txt").read_text().split()) | set(settings.symbols))
    hs = half_spreads(symbols)
    json.dump(hs, open(OUT / "half_spreads.json", "w"), indent=1)
    results, all_trades = [], []
    for tf in tfs:
        px, sig, usdt = load(symbols, tf)
        bph = int(pd.Timedelta("1h") / pd.Timedelta(tf))
        idx = usdt.index
        t0 = idx[0] + pd.Timedelta(days=3)
        t_end = idx[-1]
        t_split = t0 + (t_end - t0) * 0.67
        d_train, d_test = (t_split - t0).total_seconds() / 86400, (t_end - t_split).total_seconds() / 86400
        print(f"\n[{tf}] {len(sig)} markets | train {t0:%m-%d}->{t_split:%m-%d %H:%M} ({d_train:.1f}d) "
              f"| test ->{t_end:%m-%d %H:%M} ({d_test:.1f}d)", flush=True)

        catalog = {name: [(p, {s: fn(sig[s], **p) for s in sig}) for p in grid]
                   for name, (fn, fam, grid) in STRATEGIES.items()}
        catalog.update(panel_flags(sig, bph))

        for name, configs in catalog.items():
            evals = []
            for params, flags in configs:
                for exn, ex in EXITS.items():
                    tr = []
                    for s, fl in flags.items():
                        t = simulate(px[s], sig[s]["atr_pct"], fl.fillna(False), hs[s], ex["stop_atr"], ex["tp_atr"],
                                     int(ex["hold_h"] * bph), ex["trail_act"], ex["trail_atr"])
                        tr.append(t.assign(symbol=s))
                    tr = pd.concat(tr)
                    tr = tr[tr.t_signal >= t0]
                    trn, tst = tr[tr.t_signal < t_split], tr[tr.t_signal >= t_split]
                    evals.append((params, exn, trn, tst))
            scored = [(e, e[2].ret.sum() if len(e[2]) >= MIN_TRAIN_TRADES else -np.inf) for e in evals]
            (params, exn, trn, tst), sc = max(scored, key=lambda x: x[1])
            test_all = [stats(e[3], d_test)["pct_per_day"] for e in evals]
            results.append({"tf": tf, "strategy": name, "params": json.dumps(params), "exit": exn,
                            **{f"train_{k}": v for k, v in stats(trn, d_train).items()},
                            **{f"test_{k}": v for k, v in stats(tst, d_test).items()},
                            "test_all_configs_median": round(float(np.median(test_all)), 3),
                            "test_all_configs_pos%": round(float(np.mean([x > 0 for x in test_all])) * 100)})
            all_trades.append(pd.concat([trn.assign(split="train"), tst.assign(split="test")])
                              .assign(tf=tf, strategy=name, exit=exn))
            r = results[-1]
            print(f"  {name:14s} {r['params']:28s} {exn:5s} | train n={r['train_n']:4d} "
                  f"{r['train_pct_per_day']:+.3f}%/d | TEST n={r['test_n']:3d} win={r['test_win%']} "
                  f"{r['test_pct_per_day']:+.3f}%/d | all-configs median {r['test_all_configs_median']:+.3f}", flush=True)

        print(f"  [ML] gradient boosting walk-forward ({tf}) ...", flush=True)
        for exn, th, split, tr in run_ml(px, sig, usdt, hs, bph, t_split, t_end, pd.Timedelta(days=1)):
            tr = tr[tr.t_signal >= t0] if len(tr) else tr
            days = d_train if split == "train" else d_test
            st = stats(tr, days)
            if split == "train":
                row = {"tf": tf, "strategy": "ml_gbm", "params": json.dumps({"threshold": th}), "exit": exn,
                       **{f"train_{k}": v for k, v in st.items()}}
            else:
                row.update({f"test_{k}": v for k, v in st.items()})
                results.append(row)
                print(f"  ml_gbm         th={th:<24} {exn:5s} | train n={row['train_n']:4d} "
                      f"{row['train_pct_per_day']:+.3f}%/d | TEST n={row['test_n']:3d} win={row['test_win%']} "
                      f"{row['test_pct_per_day']:+.3f}%/d", flush=True)
            if len(tr):
                all_trades.append(tr.assign(split=split, tf=tf, strategy="ml_gbm", exit=exn))

    res = pd.DataFrame(results).sort_values("test_pct_per_day", ascending=False)
    res.to_csv(OUT / "research_results.csv", index=False)
    pd.concat(all_trades).to_csv(OUT / "research_trades.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== RANKED BY OUT-OF-SAMPLE RETURN (% of capital per day, 20% sizing, net of all costs) ===")
    cols = ["tf", "strategy", "params", "exit", "train_n", "train_pct_per_day", "test_n", "test_win%",
            "test_avg%", "test_pct_per_day", "test_all_configs_median", "test_all_configs_pos%"]
    print(res[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", nargs="+", default=["15min", "60min"])
    main(ap.parse_args().tf)
