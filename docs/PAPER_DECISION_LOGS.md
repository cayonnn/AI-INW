# 🧪 Alpha vs Guardian Decision Logging System

> **Production-Grade Separation for Audit & Analysis**

---

## 📁 Log File Structure

```
logs/
├── decisions/
│   ├── alpha_decisions.log      ← Alpha layer decisions
│   ├── guardian_decisions.log   ← Guardian layer decisions
│   ├── execution.log            ← Final execution outcomes
│   └── hybrid_trace.log         ← Combined JSON traces
├── live_loop_v3.log             ← Main system log
└── guardian_metrics.csv         ← Time-series metrics
```

---

## 📝 Log Formats

### 1️⃣ Alpha Decisions (`alpha_decisions.log`)

Purpose: Track all signal generation from the Alpha layer.

```log
2026-01-19 14:20:42 | CYCLE=17 | SYMBOL=XAUUSD
  RULE_SIGNAL=BUY | EMA20=2032.10 | EMA50=2029.80
  AI_SIGNAL=BUY | AI_CONFIDENCE=0.72
  MATCH=True | CANDIDATE=BUY
------------------------------------------------------------
2026-01-19 14:21:12 | CYCLE=18 | SYMBOL=XAUUSD
  RULE_SIGNAL=BUY | EMA20=2032.15 | EMA50=2029.85
  AI_SIGNAL=HOLD | AI_CONFIDENCE=0.45
  MATCH=False | CANDIDATE=HOLD
------------------------------------------------------------
```

Fields:
| Field | Type | Description |
|-------|------|-------------|
| RULE_SIGNAL | BUY/SELL/HOLD | EMA crossover result |
| AI_SIGNAL | BUY/SELL/HOLD | XGBoost prediction |
| AI_CONFIDENCE | Float [0,1] | Prediction probability |
| MATCH | Bool | Rule == AI? |
| CANDIDATE | BUY/SELL/HOLD | Final Alpha output |

---

### 2️⃣ Guardian Decisions (`guardian_decisions.log`)

Purpose: Track all Guardian governance evaluations.

```log
2026-01-19 14:20:42 | CYCLE=17
  EQUITY=672.00 | FREE_MARGIN=-66.00 | DAILY_DD=3.40%
  RULE_POLICY=MARGIN_CRITICAL | RULE_ACTION=FORCE_HOLD
  PPO_ACTION=EMERGENCY_FREEZE | PPO_CONF=0.99
  FINAL=BLOCK | REASON=[RULE_HARD] Margin Critical
------------------------------------------------------------
2026-01-19 14:30:15 | CYCLE=25
  EQUITY=680.00 | FREE_MARGIN=45.00 | DAILY_DD=2.10%
  RULE_POLICY=None | RULE_ACTION=ALLOW
  PPO_ACTION=ALLOW | PPO_CONF=0.85
  FINAL=ALLOW | REASON=All checks passed
------------------------------------------------------------
```

Fields:
| Field | Type | Description |
|-------|------|-------------|
| EQUITY | Float | Current account equity |
| FREE_MARGIN | Float | Available margin |
| DAILY_DD | Percent | Today's drawdown |
| RULE_POLICY | String | Triggered policy name |
| RULE_ACTION | Enum | Rule Guardian action |
| PPO_ACTION | Enum | PPO advisor suggestion |
| PPO_CONF | Float | PPO confidence score |
| FINAL | ALLOW/BLOCK | Ultimate decision |
| REASON | String | Human-readable reason |

---

### 3️⃣ Execution Log (`execution.log`)

Purpose: Track actual order execution or blocking.

```log
2026-01-19 14:20:42 | CYCLE=17 | SYMBOL=XAUUSD
  ACTION=BLOCKED | DIRECTION=BUY
  LOT=None | SL=None | TP=None | TICKET=None
  ALPHA_WANTED=BUY | GUARDIAN_SAID=BLOCK
  REASON=[RULE_HARD] Margin Critical | BLOCK_COUNT=4
------------------------------------------------------------
2026-01-19 14:45:00 | CYCLE=32 | SYMBOL=XAUUSD
  ACTION=EXECUTED | DIRECTION=BUY
  LOT=0.05 | SL=2028.50 | TP=2042.00 | TICKET=12345678
  ALPHA_WANTED=BUY | GUARDIAN_SAID=ALLOW
  REASON=Order successful | BLOCK_COUNT=0
------------------------------------------------------------
```

---

### 4️⃣ Hybrid Trace (`hybrid_trace.log`)

Purpose: Full JSON trace for each decision cycle (for post-analysis).

```json
{
  "alpha": {
    "timestamp": "2026-01-19 14:20:42",
    "cycle": 17,
    "symbol": "XAUUSD",
    "rule_signal": "BUY",
    "ema20": 2032.1,
    "ema50": 2029.8,
    "ema_slope": 0.0,
    "ai_signal": "BUY",
    "ai_confidence": 0.72,
    "match": true,
    "candidate": "BUY"
  },
  "guardian": {
    "timestamp": "2026-01-19 14:20:42",
    "cycle": 17,
    "equity": 672.0,
    "free_margin": -66.0,
    "daily_dd": 0.034,
    "open_positions": 2,
    "rule_policy_triggered": "MARGIN_CRITICAL",
    "rule_action": "FORCE_HOLD",
    "ppo_action": "EMERGENCY_FREEZE",
    "ppo_confidence": 0.99,
    "final_decision": "BLOCK",
    "reason": "[RULE_HARD] Margin Critical"
  },
  "execution": {
    "timestamp": "2026-01-19 14:20:42",
    "cycle": 17,
    "symbol": "XAUUSD",
    "action_taken": "BLOCKED",
    "direction": "BUY",
    "lot_size": null,
    "sl": null,
    "tp": null,
    "ticket": null,
    "block_reason": "[RULE_HARD] Margin Critical",
    "alpha_wanted": "BUY",
    "guardian_said": "BLOCK",
    "block_count": 4
  }
}
------------------------------------------------------------
```

---

## 🔧 Implementation Reference

The logging is implemented in `live_loop_v3.py` via:

```python
# Decision Logger (src/utils/decision_logger.py)
from src.utils.decision_logger import DecisionLogger

logger = DecisionLogger(log_dir="logs/decisions")

# Log Alpha decision
logger.log_alpha(
    cycle=cycle,
    symbol="XAUUSD",
    rule_signal=rule_signal,
    ai_signal=ai_signal,
    ai_confidence=confidence,
    candidate=candidate
)

# Log Guardian decision
logger.log_guardian(
    cycle=cycle,
    equity=equity,
    free_margin=free_margin,
    daily_dd=daily_dd,
    rule_policy=policy,
    rule_action=rule_action,
    ppo_action=ppo_action,
    ppo_conf=ppo_conf,
    final=final_decision,
    reason=reason
)

# Log Execution
logger.log_execution(
    cycle=cycle,
    symbol="XAUUSD",
    action=action_taken,
    direction=direction,
    lot=lot,
    sl=sl,
    tp=tp,
    ticket=ticket,
    alpha_wanted=alpha_signal,
    guardian_said=guardian_decision,
    reason=reason
)

# Write combined JSON trace
logger.write_hybrid_trace()
```

---

## 📊 Analysis Benefits

| Use Case | Log File |
|----------|----------|
| Alpha accuracy analysis | `alpha_decisions.log` |
| Guardian block patterns | `guardian_decisions.log` |
| Trade execution audit | `execution.log` |
| Full decision replay | `hybrid_trace.log` |
| ML training (PPO) | `guardian_metrics.csv` |

---

## 🏆 Auditor Statement

> *"Complete separation of Alpha and Guardian decision logs enables transparent auditability. Each layer's rationale is independently traceable, satisfying regulatory and competition requirements for explainability."*
