"""Paper-trading broker: real live order books, virtual balances, real fee model.

* LIMIT buys rest until the live best ask trades at/below the limit, then fill at the limit.
* Exits fill at the live best bid (walking the book is ignored: our size is tiny vs. depth).
* Fees (taker rate, conservative) are deducted from the received asset, like Wallex.
"""
from __future__ import annotations

import json

from src.config import settings
from src.db import DB, now_iso

from .base import BaseBroker

FEE = settings.fees.taker


class ShadowBroker(BaseBroker):
    mode = "SHADOW"

    def __init__(self, db: DB, client=None):
        super().__init__(db, client)
        if self.db.get("shadow_balances") is None:
            self.db.put("shadow_balances", {"TMN": settings.risk.initial_capital_tmn})

    def _bal(self) -> dict[str, float]:
        return self.db.get("shadow_balances")

    def _adj(self, asset: str, delta: float) -> None:
        b = self._bal()
        b[asset] = b.get(asset, 0.0) + delta
        if abs(b[asset]) < 1e-12:
            b[asset] = 0.0
        self.db.put("shadow_balances", b)

    def balances(self) -> dict[str, float]:
        return {k: v for k, v in self._bal().items() if v > 0}

    def place_limit_buy(self, symbol, qty, price, purpose, meta):
        qty, price = self.round_qty(symbol, qty), self.round_px(symbol, price)
        cost = qty * price
        if cost > self._bal().get("TMN", 0) + 1e-6:
            raise ValueError(f"insufficient TMN for {symbol}: need {cost:.0f}")
        self._adj("TMN", -cost)                           # reserve, like a real exchange
        o = {"client_id": self.new_client_id("sb"), "symbol": symbol, "side": "BUY", "type": "LIMIT",
             "price": price, "qty": qty, "tmn": cost, "status": "NEW", "created": now_iso(),
             "purpose": purpose, "meta": json.dumps(meta, ensure_ascii=False)}
        self.db.save_order(o)
        return self.sync_order(o)                         # may fill instantly if marketable

    def sync_order(self, order):
        if order["status"] not in ("NEW", "PARTIAL"):
            return order
        ask = self.book(order["symbol"])["ask"]
        if ask is not None and ask <= order["price"]:
            base = order["symbol"].removesuffix("TMN")
            self._adj(base, order["qty"] * (1 - FEE))
            order.update(status="FILLED", filled_qty=order["qty"] * (1 - FEE), avg_price=order["price"])
            self.db.update_order(order["client_id"], status="FILLED", filled_qty=order["filled_qty"],
                                 avg_price=order["price"])
        return order

    def cancel(self, order):
        if order["status"] in ("NEW", "PARTIAL"):
            self._adj("TMN", order["tmn"])
            self.db.update_order(order["client_id"], status="CANCELED")
            order["status"] = "CANCELED"

    def sell_now(self, symbol, qty, purpose, meta):
        base = symbol.removesuffix("TMN")
        qty = min(self.round_qty(symbol, qty), self._bal().get(base, 0.0))
        bid = self.book(symbol)["bid"]
        if bid is None or qty <= 0:
            raise ValueError(f"cannot sell {symbol}: bid={bid} qty={qty}")
        self._adj(base, -qty)
        self._adj("TMN", qty * bid * (1 - FEE))
        o = {"client_id": self.new_client_id("ss"), "symbol": symbol, "side": "SELL", "type": "MARKET",
             "price": bid, "qty": qty, "tmn": qty * bid, "status": "FILLED", "created": now_iso(),
             "filled_qty": qty, "avg_price": bid, "purpose": purpose,
             "meta": json.dumps(meta, ensure_ascii=False)}
        self.db.save_order(o)
        return o
