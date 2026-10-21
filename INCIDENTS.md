# Incident log

## 2026-10-21 23:11 (Tehran) — integration test executed on the real account

**What happened.** While adding the Wallex MCP integration, the end-to-end test `tests/test_flow_shadow.py` was run
after `.env` had already been switched to `TRADING_MODE=LIVE`. The test isolated its database (`TRADER_DB`) but not
the broker selection, so the engine built a `LiveBroker` and executed the test's forced scenario on the real
trade-credit account: two BTCTMN round trips (~1.25M TMN each) within 20 seconds.

| time (UTC) | side | client order id | value (TMN) |
|---|---|---|---|
| 19:41:33 | BUY  | lb_d4a45b35d1ea4172 | 1,248,750 |
| 19:41:39 | SELL | ls_cff9a7e14686460d | 1,244,868 |
| 19:41:47 | BUY  | lb_b364fe624bbb4439 | 1,246,975 |
| 19:41:52 | SELL | ls_3db2d635220047c0 | 1,241,843 |

**Impact.** −16,474 TMN (−0.33% of capital), fees and spread only. No position left open.

**Detection.** The first LIVE scan after restart showed equity 4,983,526 with no positions; the new MCP account
reconciliation and the exchange order history (`wallex_spot_get_orders`) identified the four orders within minutes.

**Fix (defence in depth).**
1. The test now forces `TRADING_MODE=SHADOW` before importing the engine and asserts the broker is a `ShadowBroker`.
2. The engine refuses to start in LIVE when a custom `TRADER_DB` is set (the marker of every test/demo run),
   unless `ALLOW_LIVE_WITH_CUSTOM_DB=1` is set explicitly.
Both guards were verified with `.env` still set to LIVE: the test ran in SHADOW and the real order count did not change.

**Records.** The four decisions were copied into `logs/decision_logs.jsonl` and the `decisions` table with
`setup = "INCIDENT_test_run"` and an explanatory note, so every order on the account has a matching record.
