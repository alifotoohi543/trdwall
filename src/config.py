"""Central configuration: environment, risk limits, fees, symbols."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
HIST_DIR = DATA_DIR / "historical"
LOG_DIR = ROOT / "logs"
# TRADER_DB / TRADER_DECISION_LOG let tests and demos use separate state without touching the live bot
DB_PATH = Path(os.getenv("TRADER_DB") or DATA_DIR / "trading_state.db")
DECISION_LOG = Path(os.getenv("TRADER_DECISION_LOG") or LOG_DIR / "decision_logs.jsonl")
SCAN_LOG = Path(os.getenv("TRADER_SCAN_LOG") or LOG_DIR / "scan_log.jsonl")


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = os.getenv("LLM_API_KEY", "")
    base_url: str = os.getenv("LLM_BASE_URL", "https://ai.boxapi.ir/ai/v1")
    model_fast: str = os.getenv("LLM_MODEL_FAST", "gpt-6-luna")
    model_strong: str = os.getenv("LLM_MODEL_STRONG", "gpt-6-sol")
    budget_total_usd: float = _f("LLM_BUDGET_USD_TOTAL", 0.40)
    budget_daily_usd: float = _f("LLM_BUDGET_USD_DAILY", 0.20)
    max_output_tokens: int = 400
    timeout_s: float = 45.0


@dataclass(frozen=True)
class FeeConfig:
    # Wallex spot fees per side (fraction). Update from the challenge account's fee tier.
    maker: float = _f("FEE_MAKER", 0.0025)
    taker: float = _f("FEE_TAKER", 0.0030)
    # Minimum expected move must exceed this multiple of round-trip cost (fees + spread).
    edge_multiple: float = 2.5


@dataclass(frozen=True)
class RiskConfig:
    initial_capital_tmn: float = 5_000_000
    max_position_frac: float = 0.18        # never more than 18% of equity in one position (4 slots)
    max_total_exposure_frac: float = 0.75  # at most 75% of equity in crypto at once
    max_open_positions: int = 4            # universe_check: 4x18% beat 3x25% with equal drawdown
    risk_per_trade_frac: float = 0.01      # equity lost if stop-loss hits
    circuit_breaker_dd: float = 0.03       # drawdown that halves position sizes
    hard_stop_dd: float = 0.06             # drawdown that halts new entries
    atr_stop_mult: float = 1.5
    atr_tp_mult: float = 3.0
    trailing_atr_mult: float = 1.2
    min_order_tmn: float = 100_000         # Wallex minNotional is ~50k; keep margin
    cooldown_after_loss_min: int = 30


@dataclass(frozen=True)
class Settings:
    mode: str = os.getenv("TRADING_MODE", "SHADOW").upper()
    wallex_base_url: str = "https://api.wallex.ir"
    wallex_api_key: str = os.getenv("WALLEX_API_KEY", "")
    # liquid allowed markets (24h volume >= 4B TMN, spread <= 0.3%); challenge-excluded list in data/excluded_bases.txt
    symbols: tuple[str, ...] = ("BTCTMN", "ETHTMN", "SOLTMN", "XRPTMN", "DOGETMN", "BNBTMN", "ADATMN", "LTCTMN",
                                "PEPETMN", "PUMPTMN", "HYPETMN", "SANDTMN", "XAUTTMN", "USOONTMN")
    reference_symbols: tuple[str, ...] = ("USDTTMN", "BTCUSDT", "ETHUSDT")
    timeframe: str = "15min"
    loop_interval_s: int = 60              # fast loop (stops, triggers)
    liquidate_minutes_before_end: int = 30
    challenge_end_utc: str = os.getenv("CHALLENGE_END_UTC", "")  # e.g. 2026-10-27T20:30:00Z
    llm: LLMConfig = field(default_factory=LLMConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


settings = Settings()

for _d in (DATA_DIR, HIST_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
