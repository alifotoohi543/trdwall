"""Live dashboard for the demo video and for monitoring.

Run:  .\\.venv\\Scripts\\streamlit run dashboard/app.py
Reads the engine's SQLite/JSONL state read-only; fetches public candles from Wallex for charts.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DB_PATH, settings  # noqa: E402
from src.market_data import add_indicators, resample  # noqa: E402
from src.strategy import StrategyParams, usd_frame  # noqa: E402
from src.wallex_client import WallexClient  # noqa: E402

# reference palette (dataviz skill): series-1 blue; status colors reserved for buy/sell/loss
BLUE, BLUE_FILL = "#2a78d6", "rgba(42,120,214,0.10)"
GOOD, CRITICAL, WARNING, GRAY = "#0ca30c", "#d03b3b", "#fab219", "#8a8984"
P = StrategyParams()
CAPITAL = settings.risk.initial_capital_tmn

st.set_page_config(page_title="Wallex Agentic Trader", page_icon="📈", layout="wide")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"], .stMarkdown, .stMetric, button, input { font-family: 'Vazirmatn', sans-serif; }
  .rtl { direction: rtl; text-align: right; }
  .debate { direction: rtl; text-align: right; line-height: 1.9; padding: 4px 0; }
  .tag { display:inline-block; padding:1px 8px; border-radius:6px; font-size:0.8rem; margin-left:6px; }
  .buy { background:#0ca30c22; color:#0a7d0a; } .sell { background:#d03b3b22; color:#b02f2f; }
  .skip { background:#8a898422; color:#5f5e5a; } .cancel { background:#fab21933; color:#8a6100; }
  div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data access
def q(sql: str, args: tuple = ()) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con, params=args)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def kv(key: str, default=None):
    df = q("SELECT v FROM kv WHERE k=?", (key,))
    return json.loads(df.v.iloc[0]) if len(df) else default


@st.cache_data(ttl=60, show_spinner=False)
def candles(symbol: str, hours: int = 48) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = WallexClient()
    now = int(datetime.now(timezone.utc).timestamp())
    m1 = c.candles_1m(symbol, now - hours * 3600 - 2 * 86400, now)
    u1 = c.candles_1m("USDTTMN", now - hours * 3600 - 2 * 86400, now)
    px = add_indicators(resample(m1, "15min"))
    sig = usd_frame(px, add_indicators(resample(u1, "15min")))
    cut = px.index[-1] - pd.Timedelta(hours=hours)
    return px[px.index >= cut], sig[sig.index >= cut]


def fmt_tmn(x: float) -> str:
    return f"{x:,.0f}"


def base_layout(fig: go.Figure, height: int, ytitle: str = "") -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      hovermode="x unified", yaxis_title=ytitle,
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(128,128,128,0.15)", zeroline=False))
    return fig


# ======================================================================= header
end = datetime.fromisoformat(settings.challenge_end_utc.replace("Z", "+00:00")) if settings.challenge_end_utc else None
hl, hr = st.columns([3, 2])
with hl:
    st.markdown("## 📈 Wallex Agentic Trader")
    st.markdown('<div class="rtl">موتور معاملاتی چندعامله: لایه کمّی پایتون + مناظره ایجنت‌ها با LangGraph</div>',
                unsafe_allow_html=True)


@st.fragment(run_every=30)
def status_bar():
    scan = kv("last_scan") or {}
    left = end - datetime.now(timezone.utc) if end else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Mode", settings.mode)
    c2.metric("زمان باقی‌مانده", f"{int(left.total_seconds() // 3600)}h {int(left.total_seconds() % 3600 // 60)}m"
              if left and left.total_seconds() > 0 else "پایان")
    c3.metric("آخرین اسکن (UTC)", (scan.get("ts") or "—")[11:19])


with hr:
    status_bar()


# ========================================================================= KPIs
@st.fragment(run_every=30)
def kpis():
    scan = kv("last_scan") or {}
    pf = scan.get("portfolio", {})
    eq = pf.get("equity", CAPITAL)
    eqdf = q("SELECT * FROM equity ORDER BY ts")
    max_dd = float(eqdf.dd.max()) if len(eqdf) else 0.0
    tr = q("SELECT * FROM trades")
    realized = float(tr.pnl_tmn.sum()) if len(tr) else 0.0
    wins = int((tr.pnl_tmn > 0).sum()) if len(tr) else 0
    budget = scan.get("llm_budget", {})
    cols = st.columns(6)
    cols[0].metric("ارزش کل حساب (تومان)", fmt_tmn(eq), f"{(eq / CAPITAL - 1) * 100:+.2f}%")
    cols[1].metric("سود محقق‌شده (تومان)", fmt_tmn(realized))
    cols[2].metric("حداکثر افت حساب", f"{max_dd:.2f}%")
    cols[3].metric("معاملات بسته‌شده", f"{len(tr)}")
    cols[4].metric("پوزیشن باز", f"{pf.get('open_positions', 0)}")
    cols[5].metric("هزینه LLM", f"${budget.get('total_usd', 0):.4f}")
    st.caption(f"افت فعلی {pf.get('dd_pct', 0):.2f}% · برد {wins}/{len(tr)} · درگیری سرمایه {pf.get('exposure_pct', 0):.1f}% · "
               f"بودجه باقی‌مانده LLM ${budget.get('remaining_usd', 0):.2f} از {settings.llm.budget_total_usd:.1f}$ · "
               f"{budget.get('calls', 0)} فراخوانی")


kpis()

tab_live, tab_decisions, tab_arch, tab_bt = st.tabs(
    ["🟢 بازار و سرمایه", "🧠 تصمیم‌ها و مناظره ایجنت‌ها", "🏗️ معماری", "🧪 بک‌تست"])

# ====================================================================== live tab
with tab_live:
    @st.fragment(run_every=60)
    def live():
        a, b = st.columns([5, 4])
        with a:
            st.markdown("#### ارزش حساب (تومان)")
            eqdf = q("SELECT * FROM equity ORDER BY ts")
            if len(eqdf) >= 2:
                eqdf["ts"] = pd.to_datetime(eqdf.ts)
                fig = go.Figure(go.Scatter(x=eqdf.ts, y=eqdf.equity, mode="lines", line=dict(color=BLUE, width=2),
                                           fill="tozeroy", fillcolor=BLUE_FILL,
                                           hovertemplate="%{y:,.0f} TMN<extra></extra>"))
                fig.add_hline(y=CAPITAL, line=dict(color=GRAY, width=1, dash="dot"),
                              annotation_text="سرمایه اولیه", annotation_position="bottom left")
                lo, hi = eqdf.equity.min(), eqdf.equity.max()
                pad = max((hi - lo) * 0.2, CAPITAL * 0.002)
                fig.update_yaxes(range=[min(lo, CAPITAL) - pad, max(hi, CAPITAL) + pad])
                st.plotly_chart(base_layout(fig, 260), width="stretch")
                st.markdown("#### افت حساب از سقف (%)")
                fig = go.Figure(go.Scatter(x=eqdf.ts, y=-eqdf.dd, mode="lines", line=dict(color=CRITICAL, width=2),
                                           fill="tozeroy", fillcolor="rgba(208,59,59,0.10)",
                                           hovertemplate="%{y:.2f}%<extra></extra>"))
                fig.add_hline(y=-settings.risk.circuit_breaker_dd * 100, line=dict(color=WARNING, width=1, dash="dot"),
                              annotation_text="Circuit breaker −3%", annotation_position="bottom left")
                fig.update_yaxes(range=[-max(4.0, float(eqdf.dd.max()) * 1.3), 0.2])
                st.plotly_chart(base_layout(fig, 170), width="stretch")
            else:
                st.info("داده سرمایه هنوز کافی نیست (هر ۵ دقیقه ثبت می‌شود).")
        with b:
            st.markdown("#### اسکن آخر بازار (RSI روی قیمت دلاری)")
            scan = kv("last_scan") or {}
            rows = pd.DataFrame(scan.get("rows", []))
            if len(rows):
                status_fa = {"no_setup": "بدون سیگنال", "ok": "✅ سیگنال", "probe": "🧪 ورود آزمایشی",
                             "usdt_shock": "⚠️ شوک تتر", "edge_too_small": "سود کمتر از هزینه"}
                rows["وضعیت"] = rows.status.map(status_fa).fillna(rows.status)
                rows["فاصله تا ماشه"] = (rows.rsi_usd - P.capit_rsi).round(1)
                st.dataframe(rows.rename(columns={"symbol": "نماد", "price": "قیمت (تومان)", "rsi_usd": "RSI دلاری",
                                                  "ret_24h_usd": "تغییر ۲۴س دلاری٪"})
                             [["نماد", "قیمت (تومان)", "RSI دلاری", "فاصله تا ماشه", "تغییر ۲۴س دلاری٪", "وضعیت"]],
                             hide_index=True, width="stretch",
                             column_config={"قیمت (تومان)": st.column_config.NumberColumn(format="localized")})
                st.caption(f"ماشه ورود: RSI دلاری < {P.capit_rsi:.0f} · ایجنت‌ها فقط در صورت سیگنال بیدار می‌شوند")
            st.markdown("#### پوزیشن‌های باز")
            pos = q("SELECT symbol, qty, entry, stop, target, opened, kind, trailing FROM positions")
            if len(pos):
                st.dataframe(pos, hide_index=True, width="stretch")
            else:
                st.caption("پوزیشن بازی وجود ندارد.")

        st.markdown("#### نمودار قیمت و نقاط معامله")
        sym = st.selectbox("نماد", list(settings.symbols), label_visibility="collapsed", key="sym")
        try:
            px, sig = candles(sym)
        except Exception as e:
            st.warning(f"دریافت کندل ناموفق: {e}")
            return
        fig = go.Figure(go.Candlestick(x=px.index, open=px.open, high=px.high, low=px.low, close=px.close,
                                       increasing_line_color=GOOD, decreasing_line_color=CRITICAL, name=sym))
        orders = q("SELECT * FROM orders WHERE symbol=? AND status='FILLED'", (sym,))
        for side, color, sym_mark, label in (("BUY", GOOD, "triangle-up", "خرید"), ("SELL", CRITICAL, "triangle-down", "فروش")):
            o = orders[orders.side == side] if len(orders) else orders
            if len(o):
                fig.add_trace(go.Scatter(x=pd.to_datetime(o.created), y=o.avg_price.fillna(o.price), mode="markers+text",
                                         marker=dict(symbol=sym_mark, size=13, color=color, line=dict(color="white", width=2)),
                                         text=[label] * len(o), textposition="top center",
                                         hovertemplate=f"{label} %{{y:,.0f}}<extra></extra>"))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(base_layout(fig, 340, "TMN"), width="stretch")
        fig = go.Figure(go.Scatter(x=sig.index, y=sig.rsi, mode="lines", line=dict(color=BLUE, width=2),
                                   hovertemplate="RSI %{y:.1f}<extra></extra>"))
        fig.add_hrect(y0=0, y1=P.capit_rsi, fillcolor="rgba(12,163,12,0.08)", line_width=0)
        fig.add_hline(y=P.capit_rsi, line=dict(color=GOOD, width=1, dash="dot"),
                      annotation_text=f"ماشه خرید RSI<{P.capit_rsi:.0f}", annotation_position="bottom left")
        fig.update_yaxes(range=[0, 100])
        st.markdown("##### RSI روی قیمت دلاری (حذف نوسان تتر)")
        st.plotly_chart(base_layout(fig, 170), width="stretch")

    live()

# ================================================================ decisions tab
with tab_decisions:
    @st.fragment(run_every=30)
    def decisions():
        d = q("SELECT payload FROM decisions ORDER BY ts DESC LIMIT 50")
        if not len(d):
            st.info("هنوز تصمیمی ثبت نشده. ایجنت‌ها فقط وقتی سیگنال معتبر باشد بیدار می‌شوند (صرفه‌جویی در توکن).")
        else:
            recs = [json.loads(p) for p in d.payload]
            agent = [r for r in recs if r["setup"] != "risk_rule"]
            c = st.columns(4)
            c[0].metric("تصمیم‌های ایجنتی", len(agent))
            c[1].metric("ورود تأییدشده", sum(r["action"] == "BUY" for r in agent))
            c[2].metric("رد شده", sum(r["action"] == "SKIP" for r in agent))
            c[3].metric("ارجاع به داور ارشد", sum(bool(r["agent_debate"].get("arbiter_argument")) for r in agent))
            for r in recs:
                tag = {"BUY": "buy", "SELL": "sell", "SKIP": "skip", "CANCEL": "cancel"}[r["action"]]
                title = f"{r['timestamp'][5:16].replace('T', ' ')} · {r['symbol']} · {r['action']} · {r['setup']}"
                with st.expander(title, expanded=False):
                    db_ = r["agent_debate"]
                    html = f'<div class="debate"><span class="tag {tag}">{r["action"]}</span>'
                    if r["setup"] != "risk_rule":
                        html += (f"<b>🐂 تحلیل‌گر صعودی:</b> {db_['bull_argument']}<br>"
                                 f"<b>🐻 منتقد:</b> {db_['bear_argument']}<br>")
                        if db_.get("arbiter_argument"):
                            html += f"<b>⚖️ داور ارشد:</b> {db_['arbiter_argument']}<br>"
                    html += f"<b>🛡️ نگهبان ریسک:</b> {db_['risk_assessment']}</div>"
                    st.markdown(html, unsafe_allow_html=True)
                    l, rr = st.columns(2)
                    l.json(r["market_metrics"], expanded=False)
                    rr.json({"execution": r.get("execution"), "llm": r.get("llm"), "confidence": r["confidence"]},
                            expanded=False)
        st.markdown("#### معاملات بسته‌شده")
        tr = q("SELECT symbol, kind, opened, closed, entry, exit, pnl_tmn, pnl_pct, exit_reason FROM trades ORDER BY id DESC")
        if len(tr):
            st.dataframe(tr, hide_index=True, width="stretch")
        else:
            st.caption("هنوز معامله‌ای بسته نشده.")
        st.markdown("#### رویدادهای سیستم")
        ev = q("SELECT ts, level, msg FROM events ORDER BY id DESC LIMIT 30")
        st.dataframe(ev, hide_index=True, width="stretch")

    decisions()

# ============================================================= architecture tab
with tab_arch:
    st.markdown('<div class="rtl"><b>دو لایه مجزا:</b> لایه سریع پایتون هر دقیقه ریسک را مدیریت می‌کند و هر ۱۵ دقیقه '
                'سیگنال می‌سازد؛ ایجنت‌ها فقط وقتی سیگنال معتبر باشد بیدار می‌شوند و فقط می‌توانند حجم را کم یا ورود را رد کنند.</div>',
                unsafe_allow_html=True)
    st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#8a8984", fillcolor="#f4f3f0"];
  edge [color="#8a8984", fontname="Helvetica", fontsize=9];
  subgraph cluster_fast { label="Fast layer (Python, no LLM)"; fontname="Helvetica"; color="#cfcdc6";
    data [label="Wallex public API\\ncandles + order book"]; usd [label="USD-basis indicators\\n(TMN price / USDTTMN)"];
    trig [label="Trigger filter\\nRSI<25 + edge>2.5x cost\\n+ USDT shock filter"]; risk_loop [label="1-min risk loop\\nstop / TP / trailing\\nshock tightening\\nauto-liquidation"]; }
  subgraph cluster_agents { label="Agentic layer (LangGraph)"; fontname="Helvetica"; color="#cfcdc6";
    bull [label="Bull analyst\\ngpt-6-luna", fillcolor="#e3f2e3"]; bear [label="Bear critic\\ngpt-6-luna", fillcolor="#f8e1e1"];
    arb [label="Arbiter\\ngpt-6-sol\\n(only on disagreement)", fillcolor="#fff3d6"]; rg [label="Risk Guard\\n(deterministic)", fillcolor="#e1ecf9"];
    ex [label="Executor\\nPydantic decision log"]; }
  broker [label="Broker adapter\\nSHADOW | LIVE", fillcolor="#e1ecf9"]; logs [label="decision_logs.jsonl\\nSQLite"];
  data -> usd -> trig; trig -> bull [label="signal"]; bull -> bear; bear -> arb [label="disagree"]; bear -> rg [label="agree"];
  arb -> rg; rg -> ex; ex -> broker; ex -> logs; risk_loop -> broker; data -> risk_loop;
}""", width="stretch")
    st.markdown("""
<div class="rtl">

| اصل طراحی | پیاده‌سازی |
|---|---|
| حذف نوسان سیاسی تتر از سیگنال | اندیکاتورها روی قیمت دلاری (قیمت تومانی ÷ تتر)، معامله در بازار تومانی |
| پرهیز از تله کارمزد | سود هدف ≥ ۲.۵ برابر هزینه رفت‌وبرگشت، سفارش Limit برای ورود |
| ریسک | حداکثر ۲۵٪ در هر پوزیشن، ۷۵٪ کل، توقف ورود در افت ۶٪، نصف شدن حجم در افت ۳٪ |
| صرفه‌جویی توکن | ایجنت‌ها فقط روی سیگنال؛ مدل ارزان برای مناظره، مدل قوی فقط برای اختلاف؛ سقف بودجه |
| پایداری | Retry/backoff، ذخیره وضعیت در SQLite، ادامه پس از ری‌استارت، fallback بدون LLM |

</div>""", unsafe_allow_html=True)

# ================================================================= backtest tab
with tab_bt:
    summ_p, win_p = ROOT / "reports" / "backtest_summary.csv", ROOT / "reports" / "backtest_windows.csv"
    if summ_p.exists():
        summ = pd.read_csv(summ_p)
        st.markdown('<div class="rtl">بک‌تست ۲۷ روزه روی کندل‌های والکس با کارمزد ۰.۲۵٪، اسپرد واقعی، ورود Limit در کندل بعد، '
                    'و فرض بدبینانه برخورد حد ضرر پیش از حد سود. <b>نتیجه کلیدی: هرچه تعداد معامله بیشتر، ضرر بیشتر.</b></div>',
                    unsafe_allow_html=True)
        s = summ.sort_values("trades_per_day")
        fig = go.Figure(go.Bar(x=s.variant, y=s.return_pct,
                               marker=dict(color=[GOOD if v > 0 else CRITICAL for v in s.return_pct], cornerradius=4),
                               text=[f"{v:+.1f}%" for v in s.return_pct], textposition="outside",
                               customdata=s.trades_per_day,
                               hovertemplate="%{x}<br>return %{y:+.2f}%<br>%{customdata} trades/day<extra></extra>"))
        fig.update_xaxes(title="variants sorted by trades per day →", tickangle=-30)
        st.plotly_chart(base_layout(fig, 360, "return over 27 days (%)"), width="stretch")
        st.dataframe(summ, hide_index=True, width="stretch")
        if win_p.exists():
            st.markdown("##### پنجره‌های ۳ روزه (هم‌اندازه با چالش)")
            w = pd.read_csv(win_p)
            agg = w.groupby("variant", sort=False).return_pct.agg(
                mean="mean", worst="min", best="max", positive=lambda x: (x > 0).mean() * 100).round(2)
            st.dataframe(agg, width="stretch")
    else:
        st.info("ابتدا بک‌تست را اجرا کنید: python -m scripts.run_backtest")
