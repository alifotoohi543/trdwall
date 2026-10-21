"""Compare strategy variants on full history and rolling 3-day windows.

Usage: python -m scripts.run_backtest
Writes reports/backtest_summary.csv, reports/backtest_windows.csv, reports/backtest_trades.csv
"""
import warnings

import pandas as pd

from src.backtest import load_universe, run_backtest
from src.config import ROOT
from src.strategy import StrategyParams

warnings.filterwarnings("ignore")
ALTS = ["BTCTMN", "ETHTMN", "SOLTMN", "XRPTMN", "DOGETMN"]
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

VARIANTS = {
    "A_usdt_100":         dict(p=StrategyParams(usdt_alloc=1.0), use_usdt=True, trade_alts=False),
    "A_usdt_50":          dict(p=StrategyParams(usdt_alloc=0.5), use_usdt=True, trade_alts=False),
    "B_breakout":         dict(p=StrategyParams(name="breakout"), use_usdt=False),
    "B_pullback":         dict(p=StrategyParams(name="trend_pullback"), use_usdt=False),
    "C_pullback+usdt50":  dict(p=StrategyParams(name="trend_pullback", usdt_alloc=0.5), use_usdt=True),
}


def benchmarks(data, usdt, start, end):
    def bh(df):
        s = df.close[(df.index >= start) & (df.index <= end)]
        return round((s.iloc[-1] / s.iloc[0] * (1 - 0.0025) ** 2 - 1) * 100, 2)
    return {"hold_USDT": bh(usdt), "hold_BTC": bh(data["BTCTMN"])}


if __name__ == "__main__":
    rows, wrows, all_trades = [], [], []
    for tf in ["15min", "60min"]:
        data = load_universe(ALTS, tf)
        usdt = load_universe(["USDTTMN"], tf)["USDTTMN"]
        idx = usdt.index
        warm = idx[0] + pd.Timedelta(days=3)          # indicator warm-up
        full_start, full_end = warm, idx[-1]
        rows.append({"tf": tf, "variant": "BENCH", **benchmarks(data, usdt, full_start, full_end)})
        starts = pd.date_range(warm, idx[-1] - pd.Timedelta(days=3), freq="12h")
        for name, v in VARIANTS.items():
            res = run_backtest(data, usdt, v["p"], use_usdt=v["use_usdt"], trade_alts=v.get("trade_alts", True),
                               start=full_start, end=full_end)
            rows.append({"tf": tf, "variant": name, **res.stats})
            if len(res.trades):
                all_trades.append(res.trades.assign(tf=tf, variant=name))
            for s in starts:
                w = run_backtest(data, usdt, v["p"], use_usdt=v["use_usdt"], trade_alts=v.get("trade_alts", True),
                                 start=s, end=s + pd.Timedelta(days=3))
                wrows.append({"tf": tf, "variant": name, "start": s, **w.stats})
        for s in starts:
            wrows.append({"tf": tf, "variant": "hold_USDT", "start": s,
                          "return_pct": benchmarks(data, usdt, s, s + pd.Timedelta(days=3))["hold_USDT"]})

    summ = pd.DataFrame(rows)
    win = pd.DataFrame(wrows)
    agg = win.groupby(["tf", "variant"]).agg(
        w3d_mean=("return_pct", "mean"), w3d_median=("return_pct", "median"),
        w3d_worst=("return_pct", "min"), w3d_best=("return_pct", "max"),
        w3d_pos=("return_pct", lambda x: (x > 0).mean() * 100), w3d_maxdd=("max_dd_pct", "mean")).round(2)
    summ.to_csv(OUT / "backtest_summary.csv", index=False)
    win.to_csv(OUT / "backtest_windows.csv", index=False)
    if all_trades:
        pd.concat(all_trades).to_csv(OUT / "backtest_trades.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== FULL PERIOD (~27 days) ===")
    print(summ.to_string(index=False))
    print("\n=== ROLLING 3-DAY WINDOWS (every 12h) ===")
    print(agg.to_string())
