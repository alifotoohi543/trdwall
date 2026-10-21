"""Exchange adapter interface shared by SHADOW and LIVE modes.

The engine only talks to this interface, so the exact same decision/risk code runs
in paper trading and with real money.
"""
from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod

from src.db import DB
from src.market_data import orderbook_metrics
from src.wallex_client import WallexClient


class BaseBroker(ABC):
    mode = "BASE"

    def __init__(self, db: DB, client: WallexClient | None = None):
        self.db = db
        self.client = client or WallexClient()
        self._rules: dict[str, dict] = {}

    # ----------------------------------------------------------- helpers
    def rules(self, symbol: str) -> dict:
        if not self._rules:
            self._rules = {s: {"qty_dp": int(m["stepSize"]), "px_dp": int(m["tickSize"]),
                               "min_qty": float(m["minQty"]), "min_notional": float(m["minNotional"])}
                           for s, m in self.client.markets().items()}
        return self._rules[symbol]

    def round_qty(self, symbol: str, qty: float) -> float:
        dp = self.rules(symbol)["qty_dp"]
        return math.floor(qty * 10 ** dp) / 10 ** dp

    def round_px(self, symbol: str, px: float) -> float:
        dp = self.rules(symbol)["px_dp"]
        return round(px, dp) if dp else float(int(px))

    def book(self, symbol: str) -> dict:
        return orderbook_metrics(self.client.depth(symbol))

    @staticmethod
    def new_client_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:16]}"   # Wallex allows [A-Za-z0-9_] only

    # --------------------------------------------------------- interface
    @abstractmethod
    def balances(self) -> dict[str, float]:
        """Free balances by asset, e.g. {'TMN': 5e6, 'BTC': 0.001}."""

    @abstractmethod
    def place_limit_buy(self, symbol: str, qty: float, price: float, purpose: str, meta: dict) -> dict:
        """Place a resting LIMIT buy; returns the order record (status NEW)."""

    @abstractmethod
    def sell_now(self, symbol: str, qty: float, purpose: str, meta: dict) -> dict:
        """Exit immediately (market). Returns order record with avg_price and filled_qty."""

    @abstractmethod
    def sync_order(self, order: dict) -> dict:
        """Refresh an open order's status/fills from the venue; returns updated record."""

    @abstractmethod
    def cancel(self, order: dict) -> None:
        ...
