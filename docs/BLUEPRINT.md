# 🧠 AI Trading Fund – Full Blueprint

> ระดับ Hedge Fund / Quant Competition / Autonomous AI CIO

---

## 0. System Philosophy (Non-Negotiable)

* **Capital Protection > Alpha**
* **Explainability > Black Box**
* **Portfolio of Strategies > Single Strategy**
* **Regime Awareness > Static Logic**
* **Self-Pruning > Overfitting**

---

## 1. Global Architecture Overview

```
Market Data
   ↓
Feature & Regime Engine
   ↓
AI Signal Generation (ML / Rule / Hybrid)
   ↓
Signal Fusion Engine
   ↓
Decision Gate (Risk + Confidence)
   ↓
Execution (MT5 EA)
   ↓
Alpha Attribution
   ↓
Decay Detection
   ↓
Self-Pruning Strategy Pool
   ↓
Meta-Portfolio Optimizer
   ↓
Meta-RL Capital Allocation
   ↓
Risk Committee Voting
   ↓
Capital Protection Layer
   ↺
```

---

## 2. Core Engines

| Engine | File | Function |
|--------|------|----------|
| Signal Fusion | `signal_fusion.py` | Combine ML + rules |
| Decision Gate | `decision_gate.py` | Risk filtering |
| Position Sizer | `position_sizer.py` | Kelly + vol scaling |
| Stop Loss | `stop_loss.py` | Structure + ATR |
| Auto Disable | `auto_disable.py` | 5-layer kill-switch |
| Crisis Mode | `crisis_mode.py` | NORMAL→DEFENSIVE→SURVIVAL |
| Recovery | `recovery_engine.py` | Gradual re-entry |
| Alpha Attribution | `alpha_attribution.py` | Factor decomposition |
| Decay Detection | `strategy_decay.py` | Edge monitoring |
| Strategy Pool | `strategy_pool.py` | Lifecycle management |
| Meta Portfolio | `meta_portfolio.py` | Strategy allocation |
| Meta-RL | `meta_rl_allocator.py` | Self-learning CIO |
| Risk Committee | `ai_risk_committee.py` | Multi-member voting |
| Capital Protection | `capital_protection.py` | 4-layer defense |
| Crowding Detection | `crowding_detection.py` | Global alpha risk |
| Stress Simulator | `stress_simulator.py` | Black swan testing |
| Explainability | `explainability.py` | XAI audit trail |
| Genetic Strategy | `genetic_strategy.py` | Auto evolution |
| Global Regime | `global_regime_forecast.py` | Macro detection |

---

## 3. Risk Hierarchy

```
Trade Level
   └─ Vol-adaptive SL, risk cap

Strategy Level
   └─ Max DD per strategy, auto freeze

Portfolio Level
   └─ Portfolio DD trigger, correlation cap

System Level
   └─ Kill switch, crisis mode, cash bias
```

---

## 4. Strategy Lifecycle

```
Candidate → Active → At-Risk → Frozen → Retired
```

---

## 5. Crisis State Machine

```
NORMAL
   ↓ shock
DEFENSIVE (30% risk)
   ↓ severe
SURVIVAL (5% risk)
   ↓ stable
DEFENSIVE
   ↓ recovered
NORMAL
```

---

## 6. Recovery State Machine

```
SURVIVAL → STABILIZATION → RE-ENTRY → NORMAL
```

---

## 7. Risk Committee Voting

| Member | Focus |
|--------|-------|
| Risk Engine | DD / Vol |
| Stress Sim | Tail risk |
| Crowding | Unwind risk |
| Regime | Macro shift |
| Alpha Monitor | Edge decay |

**Decision**: `NORMAL → CAUTION → RISK_OFF → EMERGENCY_STOP`

---

## 8. MT5 EA (Zero Logic)

EA หน้าที่เดียว:
- รับคำสั่ง JSON
- ตรวจความถูกต้อง
- ส่งออเดอร์
- รายงานผล

**ไม่มี**: Strategy / Indicator / Decision Logic

---

## 9. File Structure

```
src/
├── features/          # Feature engineering
├── models/            # AI models + genetic
├── signals/           # Fusion + decision + regime
├── risk/              # All risk management
├── execution/         # MT5 connector
└── utils/             # Analytics + XAI

mt5_ea/
└── AI_Trading_EA.mq5  # Zero-logic executor
```

---

## 10. Final Truth

> A strategy can fail.
> A system that protects capital survives.
> A system that explains itself earns trust.

---

**✅ This Blueprint = Hedge Fund Core Architecture**
