"""Budget-guarded LLM client (OpenAI-compatible BoxAPI).

Total budget is tiny ($3), so every call:
  * goes through `BudgetGuard`, which persists cumulative spend (from `usage.cost`)
    to data/llm_spend.json and refuses calls past the daily/total caps;
  * asks for compact JSON output with a hard max token limit;
  * returns None on budget exhaustion or error so callers fall back to rule-based logic.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from src.config import DATA_DIR, settings

log = logging.getLogger(__name__)
SPEND_FILE = DATA_DIR / "llm_spend.json"

# Fallback prices (USD per 1M tokens) if the response lacks usage.cost.
PRICES = {"gpt-6-luna": (0.10, 0.50), "gpt-5.6-luna": (0.20, 1.20), "gpt-5-mini": (0.25, 2.0),
          "gpt-6-sol": (2.0, 10.0), "gpt-6.1-sol": (2.0, 10.0)}


class BudgetGuard:
    def __init__(self, total_usd: float, daily_usd: float):
        self.total_usd, self.daily_usd = total_usd, daily_usd
        self._lock = threading.Lock()
        self.state = {"total": 0.0, "calls": 0, "daily": {}}
        if SPEND_FILE.exists():
            self.state.update(json.loads(SPEND_FILE.read_text()))

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def can_spend(self) -> bool:
        return (self.state["total"] < self.total_usd
                and self.state["daily"].get(self._today(), 0.0) < self.daily_usd)

    def record(self, cost: float) -> None:
        with self._lock:
            d = self._today()
            self.state["total"] += cost
            self.state["calls"] += 1
            self.state["daily"][d] = self.state["daily"].get(d, 0.0) + cost
            SPEND_FILE.write_text(json.dumps(self.state, indent=1))

    def summary(self) -> dict:
        return {"total_usd": round(self.state["total"], 5), "calls": self.state["calls"],
                "today_usd": round(self.state["daily"].get(self._today(), 0.0), 5),
                "remaining_usd": round(self.total_usd - self.state["total"], 4)}


class LLMClient:
    def __init__(self):
        cfg = settings.llm
        self.cfg = cfg
        self.budget = BudgetGuard(cfg.budget_total_usd, cfg.budget_daily_usd)
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout_s,
                              max_retries=1) if cfg.api_key else None

    def chat_json(self, system: str, user: str, model: str | None = None,
                  max_tokens: int | None = None) -> tuple[dict | None, dict]:
        """Return (parsed_json | None, meta). meta has model, tokens, cost, latency."""
        model = model or self.cfg.model_fast
        meta: dict[str, Any] = {"model": model, "cost": 0.0}
        if self._client is None or not self.budget.can_spend():
            meta["skipped"] = "no_key" if self._client is None else "budget_exhausted"
            return None, meta
        t0 = time.monotonic()
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=max_tokens or self.cfg.max_output_tokens,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
        except Exception as e:  # network / provider errors must never crash trading
            log.warning("LLM call failed: %s", e)
            meta["error"] = str(e)[:200]
            return None, meta
        u = resp.usage
        cost = getattr(u, "cost", None)
        if cost is None and u is not None:
            pin, pout = PRICES.get(model, (2.0, 10.0))
            cost = (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6
        cost = float(cost or 0.0)
        self.budget.record(cost)
        meta.update(cost=cost, latency_s=round(time.monotonic() - t0, 2),
                    prompt_tokens=getattr(u, "prompt_tokens", None),
                    completion_tokens=getattr(u, "completion_tokens", None))
        text = resp.choices[0].message.content or ""
        try:
            return json.loads(text[text.find("{"): text.rfind("}") + 1]), meta
        except ValueError:
            meta["error"] = f"bad_json: {text[:120]}"
            return None, meta


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
