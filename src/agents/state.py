"""Graph state and the validated Decision Log schema (Pydantic)."""
from __future__ import annotations

import uuid
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


# ------------------------------------------------------------ agent outputs
class BullView(BaseModel):
    stance: Literal["BUY", "SKIP"] = "SKIP"
    confidence: float = Field(0.0, ge=0, le=1)
    argument: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _clip(cls, v):
        return min(max(float(v or 0), 0.0), 1.0)


class BearView(BaseModel):
    veto: bool = False
    severity: float = Field(0.0, ge=0, le=1)
    argument: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def _clip(cls, v):
        return min(max(float(v or 0), 0.0), 1.0)


class ArbiterView(BaseModel):
    verdict: Literal["APPROVE", "REJECT"] = "REJECT"
    size_factor: float = Field(0.5, ge=0, le=1)
    argument: str = ""

    @field_validator("size_factor", mode="before")
    @classmethod
    def _clip(cls, v):
        return min(max(float(v if v is not None else 0.5), 0.0), 1.0)


# ------------------------------------------------------------- decision log
class AgentDebate(BaseModel):
    bull_argument: str
    bear_argument: str
    arbiter_argument: Optional[str] = None
    risk_assessment: str


class Execution(BaseModel):
    order_type: Literal["LIMIT", "MARKET"]
    side: Literal["BUY", "SELL"]
    allocated_tmn: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    client_order_id: Optional[str] = None


class DecisionLog(BaseModel):
    decision_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str
    mode: str
    symbol: str
    action: Literal["BUY", "SELL", "SKIP", "CANCEL"]
    confidence: float
    setup: str
    market_metrics: dict[str, Any]
    agent_debate: AgentDebate
    execution: Optional[Execution] = None
    llm: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------- graph state
class TradingState(TypedDict, total=False):
    symbol: str
    setup: str
    signal: dict          # Signal.to_dict()
    market: dict          # compact USD-based snapshot + orderbook
    portfolio: dict       # equity, cash, dd, open_positions, exposure_pct
    context: dict         # usdt move, recent outcomes on this symbol, probe flag
    bull: dict
    bear: dict
    arbiter: Optional[dict]
    risk: dict
    decision: dict
    llm_calls: list
