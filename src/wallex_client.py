"""Thin Wallex REST client with retry/backoff and client-side rate limiting.

Public endpoints need no key. Private endpoints (account/orders) use the
`x-api-key` header and are only used by the LIVE broker.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx
import pandas as pd

from src.config import settings

log = logging.getLogger(__name__)


class WallexError(RuntimeError):
    pass


class WallexClient:
    def __init__(self, api_key: str | None = None, min_interval_s: float = 0.25, max_retries: int = 4):
        self.api_key = api_key if api_key is not None else settings.wallex_api_key
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self._last_call = 0.0
        self._lock = threading.Lock()
        self._http = httpx.Client(base_url=settings.wallex_base_url, timeout=15.0,
                                  headers={"User-Agent": "wallex-agentic-trader/0.1"})

    # ------------------------------------------------------------------ core
    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _request(self, method: str, path: str, private: bool = False, **kw) -> Any:
        headers = kw.pop("headers", {})
        if private:
            if not self.api_key:
                raise WallexError("WALLEX_API_KEY is not set")
            headers["x-api-key"] = self.api_key
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                r = self._http.request(method, path, headers=headers, **kw)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
                data = r.json()
                if r.status_code >= 400 or data.get("success") is False:
                    raise WallexError(f"{method} {path} -> {r.status_code}: {str(data)[:300]}")
                return data.get("result", data)
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as e:
                if attempt == self.max_retries:
                    raise WallexError(f"{method} {path} failed after {attempt} tries: {e}") from e
                log.warning("Wallex %s %s failed (%s), retry %d in %.1fs", method, path, e, attempt, delay)
                time.sleep(delay)
                delay *= 2

    # --------------------------------------------------------------- public
    def markets(self) -> dict[str, dict]:
        return self._request("GET", "/v1/markets")["symbols"]

    def depth(self, symbol: str) -> dict[str, list[dict]]:
        return self._request("GET", "/v1/depth", params={"symbol": symbol})

    def trades(self, symbol: str) -> list[dict]:
        return self._request("GET", "/v1/trades", params={"symbol": symbol})["latestTrades"]

    def candles_1m(self, symbol: str, start: int, end: int) -> pd.DataFrame:
        """1-minute OHLCV between unix seconds [start, end].

        Quirk: Wallex's udf/history returns 1-minute bars for resolution=15 and
        nothing for resolution=1/5, so we request 15 and resample locally.
        """
        # udf/history has no "result" wrapper, so it bypasses _request's unwrapping but keeps retries.
        params = {"symbol": symbol, "resolution": "15", "from": start, "to": end}
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                j = self._http.get("/v1/udf/history", params=params).json()
                break
            except (httpx.TransportError, ValueError) as e:
                if attempt == self.max_retries:
                    raise WallexError(f"udf/history {symbol} failed: {e}") from e
                time.sleep(2 ** attempt)
        if j.get("s") != "ok" or not j.get("t"):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame({"time": pd.to_datetime(j["t"], unit="s", utc=True),
                           "open": j["o"], "high": j["h"], "low": j["l"],
                           "close": j["c"], "volume": j["v"]}).set_index("time")
        return df.astype(float).sort_index()

    # -------------------------------------------------------------- private
    def balances(self) -> dict[str, dict]:
        """{asset: {"value": total, "locked": frozen, ...}} for the account owning the API key."""
        return self._request("GET", "/v1/account/balances", private=True)["balances"]

    def fee_levels(self) -> Any:
        return self._request("GET", "/v1/account/fee", private=True)

    def place_order(self, symbol: str, side: str, qty: str, price: str, order_type: str = "LIMIT",
                    client_id: str | None = None) -> dict:
        """POST /v1/account/orders. qty/price must already be rounded strings.
        client_id may only contain letters, digits and underscore (Wallex rule)."""
        body: dict[str, Any] = {"symbol": symbol, "side": side.upper(), "type": order_type.upper(),
                                "quantity": qty, "price": price}
        if client_id:
            body["client_id"] = client_id
        return self._request("POST", "/v1/account/orders", private=True, json=body)

    def cancel_order(self, client_id: str) -> dict:
        return self._request("DELETE", f"/v1/account/orders/{client_id}", private=True)

    def order_status(self, client_id: str) -> dict:
        return self._request("GET", f"/v1/account/orders/{client_id}", private=True)

    def open_orders(self, symbol: str | None = None) -> Any:
        params = {"symbol": symbol} if symbol else None
        return self._request("GET", "/v1/account/openOrders", private=True, params=params)
