# 🔍 Trade Trace: Tick → Order (Debug Level Analysis)

> **Cycle #17 — BUY Signal BLOCKED by Guardian**

---

## 📊 Market State at Tick

```yaml
Timestamp: 2026-01-19 14:20:42
Symbol: XAUUSD
Cycle: 17

Price Data:
  Price: 2034.20
  EMA20: 2032.10
  EMA50: 2029.80
  EMA Slope: 0.0

Account State:
  Equity: $672.00
  Free Margin: -$66.00
  Balance: $700.00
  Daily DD: 3.4%
  Open Positions: 2
```

---

## ⚡ Step-by-Step Trace

### ① Feature Extraction
```
Input: MT5 H1 bars (500), H4 bars (125)
Output: DataFrame with 20+ features
├── EMA20: 2032.1
├── EMA50: 2029.8
├── ATR14: 2.5
├── RSI14: 55.2
└── Hour: 14, Day: Sunday
```

### ② Rule Strategy (Alpha Layer 1)
```python
# Decision Logic
if ema20 > ema50:
    rule_signal = "BUY"
elif ema20 < ema50:
    rule_signal = "SELL"
else:
    rule_signal = "HOLD"

# Result
EMA20 (2032.1) > EMA50 (2029.8)
✅ rule_signal = "BUY"
```

### ③ Alpha AI Model (Alpha Layer 2)
```python
# XGBoost Imitation Model
model = load("models/xgb_imitation.pkl")
prediction = model.predict(features)
probabilities = model.predict_proba(features)

# Result
ai_signal = "BUY"
ai_confidence = 0.72
```

### ④ Signal Fusion
```python
# Confirmation Logic
if rule_signal == ai_signal:
    candidate = rule_signal
    match = True
else:
    candidate = "HOLD"
    match = False

# Result
Rule = BUY, AI = BUY
✅ MATCH = True
✅ candidate = "BUY"
```

### ⑤ Risk Filters (Pre-Guardian)
```python
# Basic Checks
max_positions_ok = (open_positions < 3)  # True
risk_per_trade_ok = True
spread_ok = True

# Result
✅ All Risk Filters PASSED
```

### ⑥ Guardian Rule Engine (Hard Safety)
```python
# Policy Table Evaluation
state = {
    "daily_dd": 0.034,
    "margin_ratio": -66/672,  # NEGATIVE!
    "margin_block_count": 3,
    "error_detected": False
}

# Applicable Policies (sorted by priority)
MARGIN_CRITICAL (ratio < 0.20) → Priority 100 → FORCE_HOLD
SPAM_BLOCK (blocks >= 3)       → Priority 90  → FORCE_HOLD

# Result
🛑 rule_action = FORCE_HOLD
🛑 policy_triggered = "MARGIN_CRITICAL"
```

### ⑦ Guardian PPO Advisor
```python
# PPO Model Inference
ppo_state = [
    daily_dd,       # 0.034
    chaos,          # 0 
    margin_ratio,   # -0.098
    open_positions, # 2
    step/1000,      # 0.017
    free_margin_ratio  # -0.098
]

ppo_action, confidence = ppo_model.predict(ppo_state)

# Result
🧠 ppo_action = EMERGENCY_FREEZE
🧠 ppo_confidence = 0.99
```

### ⑧ Hybrid Arbitration
```python
# Hybrid Decision Logic
if rule_action in [FORCE_HOLD, EMERGENCY_FREEZE]:
    final = "BLOCK"
    reason = "[RULE_HARD] Margin Critical"
elif ppo_action == EMERGENCY_FREEZE and ppo_conf > 0.9:
    final = "BLOCK"
    reason = "[PPO_HIGH_CONF] Emergency Freeze"
else:
    final = "ALLOW"

# Result
🔒 final_decision = "BLOCK"
🔒 reason = "[RULE_HARD] Margin Critical"
```

### ⑨ Execution (BLOCKED)
```
❌ ORDER NOT SENT
├── Alpha Wanted: BUY
├── Guardian Said: BLOCK
├── Block Reason: Margin Critical
├── Block Count: 4 (cumulative)
└── Action Taken: SKIP + Shadow Record
```

---

## 📝 Decision Log Output

### Alpha Decision Log
```json
{
  "timestamp": "2026-01-19 14:20:42",
  "cycle": 17,
  "symbol": "XAUUSD",
  "rule_signal": "BUY",
  "ema20": 2032.1,
  "ema50": 2029.8,
  "ai_signal": "BUY",
  "ai_confidence": 0.72,
  "match": true,
  "candidate": "BUY"
}
```

### Guardian Decision Log
```json
{
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
}
```

### Execution Log
```json
{
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
```

---

## 🎯 Key Insight

> **Alpha identified a valid BUY opportunity (EMA crossover + AI confirmation), but Guardian correctly blocked execution due to negative free margin, preventing a certain OrderSend failure (retcode=10019).**

---

## 📈 Diagram: Decision Flow for Cycle #17

```
┌─────────────────┐
│  Market Tick    │
│  Price: 2034.20 │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Rule Strategy  │
│  EMA20 > EMA50  │
│  → BUY ✓        │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Alpha AI (XGB) │
│  predict: BUY   │
│  conf: 0.72     │
│  → BUY ✓        │
└────────┬────────┘
         ↓
┌─────────────────┐
│  CANDIDATE: BUY │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Guardian Rule  │
│  margin: -$66   │
│  → FORCE_HOLD ✗ │ ← VETO
└────────┬────────┘
         ↓
┌─────────────────┐
│  Guardian PPO   │
│  action: FREEZE │
│  conf: 0.99     │
│  → AGREE ✗      │
└────────┬────────┘
         ↓
┌─────────────────┐
│  FINAL: BLOCKED │
│  Reason: Margin │
└─────────────────┘
```
