"""Replay the last days bar-by-bar through the LIVE signal function and compare with the research flags.

Run: python -m tests.test_signal_parity
"""
import warnings

import pandas as pd

from src.config import settings
from src.market_data import add_indicators, load_history, resample
from src.research.strategies import s_rsi_oversold, s_supertrend, s_zscore
from src.strategy import MOM_GAIN, XS_DROP, ensemble_signals, usd_frame

warnings.filterwarnings("ignore")
SYMS = list(settings.symbols)
m1 = {s: load_history(s) for s in SYMS + ["USDTTMN"]}
u15, u60, u240 = (add_indicators(resample(m1["USDTTMN"], tf)) for tf in ("15min", "60min", "240min"))
px15 = {s: add_indicators(resample(m1[s], "15min")) for s in SYMS}
sig15 = {s: usd_frame(px15[s], u15) for s in SYMS}
sig60 = {s: usd_frame(add_indicators(resample(m1[s], "60min")), u60) for s in SYMS}
sig240 = {s: usd_frame(add_indicators(resample(m1[s], "240min")), u240) for s in SYMS}

# research flags on full history
expected = set()
idx = None
for d in sig15.values():
    idx = d.index if idx is None else idx.intersection(d.index)
r4 = pd.DataFrame({s: d["close"].reindex(idx).pct_change(16) for s, d in sig15.items()})
w = r4.eq(r4.min(axis=1), axis=0) & (r4 < -XS_DROP)
for s in SYMS:
    for t in (w[s] & ~w[s].shift(1, fill_value=False))[lambda x: x].index:
        expected.add(("xs_reversal", s, t + pd.Timedelta("15min")))
    for t in s_rsi_oversold(sig15[s], 20)[lambda x: x].index:
        expected.add(("oversold_rsi20", s, t + pd.Timedelta("15min")))
    for t in s_zscore(sig60[s], 3.0)[lambda x: x].index:
        expected.add(("zscore_1h", s, t + pd.Timedelta("60min")))
    for t in s_supertrend(sig240[s], 14, 3.0)[lambda x: x].index:
        expected.add(("supertrend_4h", s, t + pd.Timedelta("240min")))
idx4 = None
for d in sig240.values():
    idx4 = d.index if idx4 is None else idx4.intersection(d.index)
r24 = pd.DataFrame({s: d["close"].reindex(idx4).pct_change(6) for s, d in sig240.items()})
b4 = r24.eq(r24.max(axis=1), axis=0) & (r24 > MOM_GAIN)
for s in SYMS:
    for t in (b4[s] & ~b4[s].shift(1, fill_value=False))[lambda x: x].index:
        expected.add(("momentum_4h", s, t + pd.Timedelta("240min")))

# live replay: at each 15m boundary T, the engine sees bars that closed by T
end = u15.index[-1]
steps = pd.date_range(end - pd.Timedelta(days=6), end, freq="15min")
got = set()
for T in steps:
    p15 = {s: f[f.index < T] for s, f in px15.items()}
    s15 = {s: f[f.index < T] for s, f in sig15.items()}
    s60 = {s: f[f.index + pd.Timedelta("60min") <= T] for s, f in sig60.items()}
    s240 = {s: f[f.index + pd.Timedelta("240min") <= T] for s, f in sig240.items()}
    sigs, _ = ensemble_signals(p15, s15, s60, u15[u15.index < T], T + pd.Timedelta(seconds=20), sig240=s240)
    got |= {(x.kind, x.symbol, T) for x in sigs}
    # the engine keeps one signal per symbol per scan (highest priority); mirror that for the comparison
    for e in [e for e in expected if e[2] == T and e[1] in {x.symbol for x in sigs}]:
        if e not in got:
            expected.discard(e)
exp_window = {e for e in expected if steps[0] <= e[2] <= steps[-1]}
print(f"replayed {len(steps)} scans | research signals {len(exp_window)} | live signals {len(got)}")
missing, extra = exp_window - got, got - exp_window
print("by kind:", pd.Series([g[0] for g in got]).value_counts().to_dict())
print("missing in live:", sorted(missing)[:10])
print("extra in live  :", sorted(extra)[:10])
