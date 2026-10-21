"""Read-only client for the official Wallex MCP server (https://api.wallex.ir/mcp, Streamable HTTP).

Role in the system ("eyes", never "hands"):
  * account reconciliation: balances + open orders read through MCP are compared with the
    engine's own state every scan, so a drift (manual trade, lost fill, API glitch) is detected;
  * market intelligence for the agents: recent-trade order flow via MCP when a signal fires.
Order placement stays on the tested REST path (idempotent client_id, persisted before sending),
and this module refuses every state-changing tool by design (Wallex marks them STATE-CHANGING).
Any MCP failure returns None and never interrupts trading.
"""
from __future__ import annotations

import asyncio
import json
import logging

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.config import settings

log = logging.getLogger(__name__)
MCP_URL = "https://api.wallex.ir/mcp"
READ_ONLY_TOOLS = {
    "wallex_basic_get_account_balances", "wallex_basic_get_account_fees", "wallex_basic_get_latest_trades",
    "wallex_basic_get_market_depth", "wallex_basic_get_markets", "wallex_basic_candle_history",
    "wallex_spot_get_open_order", "wallex_spot_get_orders", "wallex_spot_get_orders_client",
    "wallex_spot_get_depth", "wallex_spot_get_markets", "wallex_credit_sub_account_balances_details",
}


async def _call_many(calls: list[tuple[str, dict]], timeout_s: float) -> list[dict | None]:
    headers = {"X-API-Key": settings.wallex_api_key} if settings.wallex_api_key else {}
    out: list[dict | None] = []
    async with httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(15, read=timeout_s)) as client:
        async with streamable_http_client(MCP_URL, http_client=client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                for tool, args in calls:
                    r = await session.call_tool(tool, args)
                    if getattr(r, "is_error", False) or not r.content:
                        out.append(None)
                        continue
                    try:
                        out.append(json.loads(r.content[0].text))
                    except (ValueError, AttributeError):
                        out.append(None)
    return out


def call_many(calls: list[tuple[str, dict]], timeout_s: float = 30.0) -> list[dict | None]:
    """Run several read-only MCP tool calls in one session. Returns None per failed call."""
    for tool, _ in calls:
        if tool not in READ_ONLY_TOOLS:
            raise PermissionError(f"{tool} is not a read-only tool; orders go through the REST broker")
    try:
        return asyncio.run(asyncio.wait_for(_call_many(calls, timeout_s), timeout_s + 15))
    except Exception as e:  # network / protocol errors must never stop the engine
        log.warning("Wallex MCP call failed: %s", str(e)[:200])
        return [None] * len(calls)


def account_snapshot() -> dict | None:
    """Balances (value/locked) and open orders as seen by Wallex through MCP."""
    bal, oo = call_many([("wallex_basic_get_account_balances", {}), ("wallex_spot_get_open_order", {})])
    if bal is None:
        return None
    balances = {a: (float(b.get("value") or 0), float(b.get("locked") or 0))
                for a, b in (bal.get("result", {}).get("balances") or {}).items()
                if float(b.get("value") or 0) > 0}
    orders = (oo or {}).get("result", {}).get("orders") if oo else None
    return {"balances": balances, "open_orders": len(orders) if isinstance(orders, list) else None}


def trade_flow(symbol: str) -> dict | None:
    """Aggressor balance of the latest public trades (who is hitting whom right now)."""
    (res,) = call_many([("wallex_basic_get_latest_trades", {"symbol": symbol})])
    trades = (res or {}).get("result", {}).get("latestTrades") or []
    if not trades:
        return None
    buy = sum(float(t["sum"]) for t in trades if t.get("isBuyOrder"))
    tot = sum(float(t["sum"]) for t in trades)
    ts = sorted(t["timestamp"] for t in trades)
    return {"n_trades": len(trades), "buy_value_ratio": round(buy / tot, 2) if tot else None,
            "first": ts[0], "last": ts[-1], "value_tmn": round(tot)}
