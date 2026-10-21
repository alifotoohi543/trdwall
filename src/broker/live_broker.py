"""LIVE broker: real orders on the Wallex trade-credit account via the REST API.

Endpoints (developers.wallex.ir): POST /v1/account/orders, GET/DELETE /v1/account/orders/{client_id},
GET /v1/account/balances. The challenge's trade-credit account has its own API key.

Exits use an aggressive LIMIT sell (best bid minus a protection band) instead of a raw MARKET
order: it executes immediately like a market order but can never fill at an absurd price.
"""
from __future__ import annotations

import json
import logging

from src.db import DB, now_iso
from src.wallex_client import WallexError

from .base import BaseBroker

log = logging.getLogger(__name__)
EXIT_PROTECTION = 0.003      # accept up to 0.3% below best bid when exiting


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class LiveBroker(BaseBroker):
    mode = "LIVE"

    def __init__(self, db: DB, client=None):
        super().__init__(db, client)
        if not self.client.api_key:
            raise RuntimeError("LIVE mode requires WALLEX_API_KEY in .env")

    def balances(self) -> dict[str, float]:
        out = {}
        for asset, b in self.client.balances().items():
            free = _f(b.get("value")) - _f(b.get("locked"))
            if free > 0:
                out[asset] = free
        return out

    def _fmt(self, symbol: str, qty: float, price: float) -> tuple[str, str]:
        r = self.rules(symbol)
        return f"{self.round_qty(symbol, qty):.{r['qty_dp']}f}", f"{self.round_px(symbol, price):.{r['px_dp']}f}"

    def _submit(self, symbol, side, qty, price, purpose, meta) -> dict:
        q, p = self._fmt(symbol, qty, price)
        cid = self.new_client_id("lb" if side == "BUY" else "ls")
        o = {"client_id": cid, "symbol": symbol, "side": side, "type": "LIMIT", "price": float(p), "qty": float(q),
             "tmn": float(p) * float(q), "status": "NEW", "created": now_iso(), "purpose": purpose,
             "meta": json.dumps(meta, ensure_ascii=False)}
        self.db.save_order(o)                       # persist before sending: survives crashes
        try:
            self.client.place_order(symbol, side, q, p, "LIMIT", client_id=cid)
        except WallexError as e:
            # A timeout may hide a successful placement; client_id is unique, so just look it up.
            try:
                self.client.order_status(cid)
                log.warning("place_order raised but order %s exists: %s", cid, e)
            except WallexError:
                self.db.update_order(cid, status="REJECTED")
                o["status"] = "REJECTED"
                raise
        return self.sync_order(o)

    def place_limit_buy(self, symbol, qty, price, purpose, meta):
        return self._submit(symbol, "BUY", qty, price, purpose, meta)

    def sell_now(self, symbol, qty, purpose, meta):
        # fees are taken from the received coin, so never ask to sell more than the free balance
        free = self.balances().get(symbol.removesuffix("TMN"), 0.0)
        qty = min(qty, free)
        if qty <= 0:
            raise ValueError(f"no free {symbol} balance to sell")
        bid = self.book(symbol)["bid"]
        return self._submit(symbol, "SELL", qty, bid * (1 - EXIT_PROTECTION), purpose, meta)

    def sync_order(self, order):
        if order["status"] not in ("NEW", "PARTIAL"):
            return order
        r = self.client.order_status(order["client_id"])
        status = str(r.get("status", "")).upper()
        executed = _f(r.get("executedQty"))
        avg = _f(r.get("executedPrice")) or (_f(r.get("executedSum")) / executed if executed else None)
        base = order["symbol"].removesuffix("TMN")
        fee_in_base = sum(_f(f.get("fee")) for f in r.get("fills") or [] if f.get("feeAsset") == base)
        net_qty = executed - fee_in_base if order["side"] == "BUY" else executed
        if status in ("FILLED", "DONE", "FINISHED") or (executed and executed >= order["qty"] * 0.999):
            new = "FILLED"
        elif status in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED"):
            new = "CANCELED"
        else:
            new = "PARTIAL" if executed else "NEW"
        order.update(status=new, filled_qty=net_qty, avg_price=avg)
        self.db.update_order(order["client_id"], status=new, filled_qty=net_qty, avg_price=avg)
        return order

    def cancel(self, order):
        try:
            self.client.cancel_order(order["client_id"])
        except WallexError as e:
            log.warning("cancel %s failed (maybe already filled): %s", order["client_id"], e)
        self.sync_order(order)
        if order["status"] in ("NEW", "PARTIAL"):
            order["status"] = "CANCELED"
            self.db.update_order(order["client_id"], status="CANCELED")
