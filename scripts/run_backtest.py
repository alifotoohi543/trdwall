"""Compare strategy variants (frequency vs. profitability) on full history and rolling 3-day windows.

Usage: python -m scripts.run_backtest
Writes reports/backtest_summary.csv, reports/backtest_windows.csv, reports/backtest_trades.csv
"""
import warnings

import pandas as pd

from src.backtest import load_universe, run_backtest
from src.config import ROOT
from src.strategy import StrategyParams as P

warnings.filterwarnings("ignore")
ALTS = ["BTCTMN", "ETHTMN", "SOLTMN", "XRPTMN", "DOGETMN"]
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

# (timeframe, params) ordered roughly from most selective to most active
VARIANTS = {
    "15m_oversold_rsi22":     ("15min", P(name="oversold", capit_rsi=22)),
    "15m_oversold_rsi25":     ("15min", P(name="oversold", capit_rsi=25)),
    "15m_oversold_rsi30":     ("15min", P(name="oversold", capit_rsi=30)),
    "1h_pullback_strict":     ("60min", P(name="pullback", atr_stop=1.5, atr_tp=3, max_hold_bars=48, trail_activate_atr=1.5)),
    "1h_both":                ("60min", P(name="both", max_hold_bars=24)),
    "15m_pullback":           ("15min", P(name="pullback", atr_stop=1.5, atr_tp=3, max_hold_bars=48, trail_activate_atr=1.5)),
    "15m_both":               ("15min", P(name="both", atr_stop=1.5, atr_tp=3, max_hold_bars=48, trail_activate_atr=1.5)),
    "15m_both_loose":         ("15min", P(name="both", atr_stop=1.5, atr_tp=3, max_hold_bars=48, trail_activate_atr=1.5, rsi_dip=45, mr_rsi=35, edge_multiple=1.5)),
    "15m_both_very_loose":    ("15min", P(name="both", atr_stop=1.5, atr_tp=3, max_hold_bars=48, trail_activate_atr=1.5, rsi_dip=50, mr_rsi=40, edge_multiple=1.0)),
    # ablations: what each design decision is worth
    "15m_oversold_TMNsignals":    ("15min", P(name="oversold", usd_signals=False)),
    "15m_oversold_marketorders":  ("15min", P(name="oversold", limit_entry=False)),
}

if __name__ == "__main__":
    cache, rows, wrows, all_trades = {}, [], [], []
    for name, (tf, p) in VARIANTS.items():
        key = (tf, p.usd_signals)
        if key not in cache:
            cache[key] = load_universe(ALTS, tf, usd_signals=p.usd_signals)
        px, sig, usdt = cache[key]
        start = usdt.index[0] + pd.Timedelta(days=3)          # indicator warm-up
        res = run_backtest(px, sig, usdt, p, start=start)
        rows.append({"variant": name, **res.stats})
        if len(res.trades):
            all_trades.append(res.trades.assign(variant=name))
        for s in pd.date_range(start, usdt.index[-1] - pd.Timedelta(days=3), freq="12h"):
            w = run_backtest(px, sig, usdt, p, start=s, end=s + pd.Timedelta(days=3))
            wrows.append({"variant": name, "start": s, **w.stats})

    summ = pd.DataFrame(rows)
    win = pd.DataFrame(wrows)
    agg = win.groupby("variant", sort=False).agg(
        w3d_mean=("return_pct", "mean"), w3d_median=("return_pct", "median"),
        w3d_worst=("return_pct", "min"), w3d_best=("return_pct", "max"),
        w3d_positive_pct=("return_pct", lambda x: (x > 0).mean() * 100),
        w3d_maxdd=("max_dd_pct", "mean")).round(2)
    summ.to_csv(OUT / "backtest_summary.csv", index=False)
    win.to_csv(OUT / "backtest_windows.csv", index=False)
    if all_trades:
        pd.concat(all_trades).to_csv(OUT / "backtest_trades.csv", index=False)
    pd.set_option("display.width", 220)
    print("=== FULL PERIOD (~27 days, 5M TMN, fee 0.25%/side) ===")
    print(summ.to_string(index=False))
    print("\n=== ROLLING 3-DAY WINDOWS (every 12h) ===")
    print(agg.to_string())
