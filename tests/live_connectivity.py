"""Zero-cost LIVE order-lifecycle check on the real Wallex account.

Places one small LIMIT buy far below the market (cannot fill), reads its status, cancels it and
verifies the cancel and the unlocked balance. Both actions are recorded in the decision log so
every order on the account has a matching record.

Run: python -m tests.live_connectivity   (requires WALLEX_API_KEY; TRADING_MODE is ignored)
"""
from __future__ import annotations

import time

from src.agents.state import AgentDebate, DecisionLog, Execution
from src.broker.live_broker import LiveBroker
from src.db import DB, now_iso

SYMBOL = "BTCTMN"
TMN = 100_000                 # above Wallex minNotional (50k)
BELOW = 0.30                  # 30% under the best bid: cannot fill


def log(db: DB, action: str, price: float, cid: str, text: str) -> None:
    db.add_decision(DecisionLog(
        timestamp=now_iso(), mode="LIVE", symbol=SYMBOL, action=action, confidence=1.0, setup="connectivity_test",
        market_metrics={"price_tmn": price},
        agent_debate=AgentDebate(bull_argument="—", bear_argument="—", risk_assessment=text),
        execution=Execution(order_type="LIMIT", side="BUY", allocated_tmn=TMN, entry_price=price, client_order_id=cid),
        llm={"calls": 0, "cost_usd": 0.0}).model_dump())


def main() -> None:
    db = DB()
    br = LiveBroker(db)
    tmn0 = br.balances().get("TMN", 0.0)
    bid = br.book(SYMBOL)["bid"]
    price = bid * (1 - BELOW)
    print(f"free TMN before: {tmn0:,.0f} | best bid {bid:,.0f} -> test limit {price:,.0f}")

    o = br.place_limit_buy(SYMBOL, TMN / price, price, "connectivity_test", {"test": True})
    log(db, "BUY", o["price"], o["client_id"],
        "تست اتصال سیستم: سفارش لیمیت ۳۰٪ زیر بازار (غیرقابل اجرا) برای آزمودن چرخه ثبت/پیگیری/لغو سفارش در حساب واقعی.")
    print("placed:", o["client_id"], o["status"])
    time.sleep(2)
    o = br.sync_order(o)
    print("status after 2s:", o["status"], "| filled:", o.get("filled_qty"))
    locked = br.client.balances().get("TMN", {}).get("locked")
    print("TMN locked while open:", locked)

    br.cancel(o)
    log(db, "CANCEL", o["price"], o["client_id"], "تست اتصال سیستم: لغو سفارش آزمایشی (بدون اجرا و بدون هزینه).")
    time.sleep(2)
    st = br.client.order_status(o["client_id"])
    tmn1 = br.balances().get("TMN", 0.0)
    print("exchange status after cancel:", st.get("status"), "| executedQty:", st.get("executedQty"))
    print(f"free TMN after: {tmn1:,.0f} (diff {tmn1 - tmn0:+,.0f})")
    ok = str(st.get("status", "")).upper() in ("CANCELED", "CANCELLED") and abs(tmn1 - tmn0) < 1
    print("LIVE CONNECTIVITY:", "OK" if ok else "CHECK OUTPUT")


if __name__ == "__main__":
    main()
