"""LangGraph nodes: Bull analyst -> Bear critic -> (Arbiter on disagreement) -> Risk Guard -> Executor.

Token economy: inputs are compact JSON (~200 tokens), outputs are short JSON, the cheap
model handles Bull/Bear, and the strong model is only consulted when they disagree.
Every LLM failure degrades to a conservative rule-based default instead of crashing.
"""
from __future__ import annotations

import json
import logging

from langchain_core.runnables import RunnableConfig

from src.config import settings
from src.db import now_iso
from src.llm.client import get_llm
from src.strategy import position_size_tmn

from .state import AgentDebate, ArbiterView, BearView, BullView, DecisionLog, Execution, TradingState

log = logging.getLogger(__name__)

FA = "Write the 'argument' field in Persian (Farsi), at most 35 words, citing the numbers you rely on."

BULL_SYS = (
    "You are the Bull analyst of a crypto spot trading desk on Wallex (prices in toman, indicators "
    "computed on the USD price to strip out toman/USDT noise). A rule-based trigger proposes a LONG entry. "
    "Judge it against the setup's historical base rate given in context.setup_base_rate (this setup is "
    "contrarian by design: buying weakness is expected, so a falling price alone is not a reason to skip). "
    "Skip only if this instance looks clearly worse than the typical case. Round-trip cost is ~0.55%. "
    "confidence = your probability (0..1) that price reaches the target before the stop. "
    'Reply JSON only: {"stance":"BUY"|"SKIP","confidence":0..1,"argument":"..."}. ' + FA)

BEAR_SYS = (
    "You are the Bear critic (risk skeptic) of the same desk. Find concrete reasons this long entry may fail: "
    "falling-knife momentum, weak 24h trend, wide spread, thin bids, toman/USDT shock, recent losses on this "
    "coin, target not covering costs. The setup is contrarian by design (see context.setup_base_rate): do not "
    "veto merely because price fell; veto only when this case is clearly worse than the base rate. "
    'Reply JSON only: {"veto":true|false,"severity":0..1,"argument":"..."}. ' + FA)

ARBITER_SYS = (
    "You are the senior Portfolio Manager. The Bull and Bear disagree on a proposed long entry. Weigh both "
    "arguments against the data and decide. Prefer capital preservation: scoring rewards realized return "
    "but heavily penalizes drawdown. "
    'Reply JSON only: {"verdict":"APPROVE"|"REJECT","size_factor":0..1,"argument":"..."}. ' + FA)


def _payload(state: TradingState, extra: dict | None = None) -> str:
    sig = state["signal"]
    d = {"symbol": state["symbol"], "setup": state["setup"],
         "entry": round(sig["price"]), "stop_pct": round((sig["stop"] / sig["price"] - 1) * 100, 2),
         "target_pct": round(sig["expected_move_pct"], 2), "cost_pct": round(sig["cost_pct"], 2),
         "market_usd_basis": state["market"], "context": state["context"],
         "portfolio": {k: state["portfolio"][k] for k in ("dd_pct", "open_positions", "exposure_pct")}}
    if extra:
        d.update(extra)
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def _call(state: TradingState, role: str, system: str, user: str, model: str | None = None):
    out, meta = get_llm().chat_json(system, user, model=model)
    meta["role"] = role
    return out, meta


# ------------------------------------------------------------------ nodes
def bull_node(state: TradingState) -> dict:
    out, meta = _call(state, "bull", BULL_SYS, _payload(state))
    if out is None:
        view = BullView(stance="BUY", confidence=0.55,
                        argument="LLM در دسترس نیست؛ سیگنال قانون‌محور (اشباع فروش روی قیمت دلاری) از فیلتر لبه سود عبور کرده است.")
    else:
        try:
            view = BullView(**out)
        except Exception:
            view = BullView(stance="SKIP", confidence=0.0, argument=f"invalid bull output: {str(out)[:80]}")
    return {"bull": view.model_dump(), "llm_calls": state.get("llm_calls", []) + [meta]}


def bear_node(state: TradingState) -> dict:
    out, meta = _call(state, "bear", BEAR_SYS, _payload(state, {"bull_view": state["bull"]}))
    if out is None:
        view = BearView(veto=False, severity=0.3, argument="LLM در دسترس نیست؛ ریسک پیش‌فرض متوسط در نظر گرفته شد.")
    else:
        try:
            view = BearView(**out)
        except Exception:
            view = BearView(veto=True, severity=0.8, argument=f"invalid bear output: {str(out)[:80]}")
    return {"bear": view.model_dump(), "llm_calls": state["llm_calls"] + [meta]}


def route_after_debate(state: TradingState) -> str:
    """Conditional edge: escalate to the Arbiter only on genuine disagreement."""
    bull, bear = state["bull"], state["bear"]
    if bull["stance"] != "BUY":
        return "risk"
    disagree = bear["veto"] or bear["severity"] >= 0.5
    clear_veto = bear["veto"] and bear["severity"] >= 0.85
    return "arbiter" if disagree and not clear_veto and bull["confidence"] >= 0.6 else "risk"


def arbiter_node(state: TradingState) -> dict:
    out, meta = _call(state, "arbiter", ARBITER_SYS,
                      _payload(state, {"bull_view": state["bull"], "bear_view": state["bear"]}),
                      model=settings.llm.model_strong)
    if out is None:
        view = ArbiterView(verdict="REJECT", size_factor=0.0, argument="داور در دسترس نبود؛ در صورت اختلاف، ورود انجام نمی‌شود.")
    else:
        try:
            view = ArbiterView(**out)
        except Exception:
            view = ArbiterView(verdict="REJECT", size_factor=0.0, argument=f"invalid arbiter output: {str(out)[:80]}")
    return {"arbiter": view.model_dump(), "llm_calls": state["llm_calls"] + [meta]}


def risk_node(state: TradingState) -> dict:
    """Deterministic Risk Guard: the LLMs can only shrink or veto, never enlarge risk."""
    r, pf, sig = settings.risk, state["portfolio"], state["signal"]
    bull, bear, arb = state["bull"], state["bear"], state.get("arbiter")
    dd = pf["dd_pct"] / 100
    reasons = []
    approved = True
    if bull["stance"] != "BUY" or bull["confidence"] < 0.55:
        approved = False
        reasons.append(f"تحلیل‌گر صعودی قانع نشد (اطمینان {bull['confidence']:.2f})")
    if arb is not None:
        if arb["verdict"] != "APPROVE":
            approved = False
            reasons.append("داور ارشد ورود را رد کرد")
    elif bear["veto"] and approved:
        approved = False
        reasons.append(f"منتقد وتو کرد (شدت {bear['severity']:.2f})")
    if dd >= r.hard_stop_dd:
        approved = False
        reasons.append(f"افت حساب {dd:.1%} به سقف توقف {r.hard_stop_dd:.0%} رسیده")
    if pf["open_positions"] >= r.max_open_positions:
        approved = False
        reasons.append("سقف تعداد پوزیشن باز پر است")

    size = 0.0
    if approved:
        base = position_size_tmn(pf["equity"], sig["price"], sig["stop"], dd)
        factor = max(0.3, min(1.0, bull["confidence"] - 0.5 * bear["severity"] + 0.3))
        if arb is not None:
            factor = min(factor, arb["size_factor"])
        if state["context"].get("probe"):
            factor = min(factor, 0.4)                       # activity probe trades stay small
        room = pf["equity"] * r.max_total_exposure_frac - pf["exposure_tmn"]
        size = float(int(min(base * factor, room, pf["cash"] * 0.995)))
        if size < r.min_order_tmn:
            approved = False
            reasons.append(f"حجم مجاز ({size:,.0f} تومان) کمتر از حداقل سفارش است")
        else:
            reasons.append(f"مجوز ورود با {size:,.0f} تومان ({size / pf['equity']:.1%} سرمایه)، "
                           f"ضریب اطمینان {factor:.2f}، افت فعلی حساب {dd:.2%}، "
                           f"ریسک تا حد ضرر {(1 - sig['stop'] / sig['price']) * size:,.0f} تومان")
    return {"risk": {"approved": approved, "size_tmn": size if approved else 0.0, "assessment": "؛ ".join(reasons)}}


def executor_node(state: TradingState, config: RunnableConfig) -> dict:
    """Validate the decision with Pydantic, send the order through the broker, persist the log."""
    cfg = config.get("configurable", {})
    execute = cfg.get("execute")          # callable(symbol, tmn, signal) -> client_order_id | None
    sig, risk = state["signal"], state["risk"]
    calls = state.get("llm_calls", [])
    action = "BUY" if risk["approved"] else "SKIP"
    conf = state["bull"]["confidence"] * (1 - 0.5 * state["bear"]["severity"])
    execution = None
    if action == "BUY" and execute is not None:
        try:
            cid = execute(state["symbol"], risk["size_tmn"], sig)
        except Exception as e:  # never let a broker error kill the graph; record it instead
            log.exception("execution failed")
            cid = None
            risk["assessment"] += f"؛ خطای اجرا: {str(e)[:120]}"
        if cid:
            execution = Execution(order_type="LIMIT", side="BUY", allocated_tmn=risk["size_tmn"],
                                  entry_price=sig["price"], stop_loss=sig["stop"], take_profit=sig["target"],
                                  client_order_id=cid)
        else:
            action = "SKIP"
    dec = DecisionLog(
        timestamp=now_iso(), mode=settings.mode, symbol=state["symbol"], action=action,
        confidence=round(conf, 3), setup=state["setup"],
        market_metrics={"price_tmn": sig["price"], **state["market"], "usdt_1h_pct": state["context"].get("usdt_1h_pct")},
        agent_debate=AgentDebate(bull_argument=state["bull"]["argument"], bear_argument=state["bear"]["argument"],
                                 arbiter_argument=(state.get("arbiter") or {}).get("argument"),
                                 risk_assessment=risk["assessment"]),
        execution=execution,
        llm={"calls": len(calls), "cost_usd": round(sum(c.get("cost", 0) for c in calls), 6),
             "models": sorted({c["model"] for c in calls}),
             "fallbacks": [c["role"] for c in calls if c.get("skipped") or c.get("error")]},
    )
    payload = dec.model_dump()
    if cfg.get("db") is not None:
        cfg["db"].add_decision(payload)
    return {"decision": payload}
