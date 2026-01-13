# 🧠 AI Signal Fusion EA – Detailed Project Workflow

> Global Competition Level | Prop Firm / Quant Fund / Algo Contest

---

## 1️⃣ System Architecture

```
┌──────────────┐
│ Market Data  │  (Tick / OHLC / Volume / Session)
└──────┬───────┘
       ↓
┌──────────────┐
│ Feature Eng. │  (Indicators / Price Action / Context)
└──────┬───────┘
       ↓
┌─────────────────────────────┐
│ AI Signal Fusion Engine      │
│ - Rule-based Signals         │
│ - ML / DL Models (ONNX)      │
│ - Strategy Weighting         │
└──────┬──────────────────────┘
       ↓
┌──────────────┐
│ Risk Engine  │  (Position Sizing / DD Control)
└──────┬───────┘
       ↓
┌──────────────┐
│ Trade Exec   │  (Order / Slippage / Spread)
└──────┬───────┘
       ↓
┌──────────────┐
│ Monitoring   │  (Logs / Metrics / Alerts)
└──────────────┘
```

---

## 2️⃣ Phase 1: Research & Strategy Design (Offline)

### Market Hypothesis
- Trend continuation
- Mean reversion
- Breakout volatility

### Signal Candidates
| Category | Examples |
|----------|----------|
| Technical | RSI, MACD, BB, ATR |
| Price Action | HH/HL, LL/LH, Candle patterns |
| Context | Session, Volatility regime |

### Dataset Construction
- Input: OHLCV
- Label: Future return / Win probability
- Split: Train / Validation / Test (Walk-Forward)

---

## 3️⃣ Phase 2: Model Development (Offline Only)

### Model Types
- **ML**: XGBoost, LightGBM, RandomForest
- **DL**: LSTM, Transformer (optional)

### Training Rules
| ❌ Forbidden | ✅ Required |
|-------------|------------|
| Live training | Walk-forward validation |
| Runtime optimization | Out-of-sample testing |
| Parameter tuning in EA | Offline-only training |

### Evaluation Metrics
- Sharpe Ratio
- Max Drawdown
- Stability Score

### Model Export
```
artifacts/models/
├── lstm_direction_v1.onnx
├── xgb_timing_v1.json
└── model_metadata.json
```

---

## 4️⃣ Phase 3: EA Architecture (MQL5)

### Module Structure
```
AI_Trading_EA.mq5
├── DataFeed.mqh
├── FeatureEngine.mqh
├── SignalEngine.mqh
│   ├── RuleSignals
│   ├── AIModelWrapper (ONNX)
│   └── FusionLogic
├── RegimeDetector.mqh
├── RiskManager.mqh
├── TradeExecutor.mqh
├── Logger.mqh
└── Config.mqh
```

### Deterministic Rule
> Same input → Same output (No random seed)

---

## 5️⃣ Phase 4: Signal Fusion Logic

### Signal Sources
| Source | Output |
|--------|--------|
| Rule-based | -1 / 0 / +1 |
| ML Model | Probability |
| Context | Weight modifier |

### Fusion Formula
```
FinalScore = (RuleSignal × W1) + (AI_Prob × W2) + (ContextWeight × W3)
```

### Decision Gate
- Threshold entry
- Cooldown period
- Max trades per session

---

## 6️⃣ Phase 5: Risk Management

### Position Sizing
- Fixed % risk per trade
- ATR-based stop loss

### Global Risk Rules
| Rule | Action |
|------|--------|
| Daily DD > 3% | Pause trading |
| 5 consecutive losses | Cooldown 4 hours |
| Max exposure > 15% | Block new trades |

### Trade Filters
- Spread filter
- News filter
- Low liquidity filter

---

## 7️⃣ Phase 6: Backtest & Validation

### Backtest Levels
1. Single symbol
2. Multi-symbol
3. Monte Carlo simulation

### Acceptance Criteria
| Metric | Threshold |
|--------|-----------|
| Profit Factor | > 1.5 |
| Max Drawdown | < 15-20% |
| Sharpe Ratio | > 1.0 |

---

## 8️⃣ Phase 7: Deployment

### Deployment Flow
```
Research → Train → Validate → Export ONNX
                      ↓
                EA Load Model
                      ↓
           Demo → Small Live → Scale
```

### Model Swap
- Hot-swap ONNX without recompile
- Version tracking in config

---

## 9️⃣ Phase 8: Live Monitoring

### Metrics Tracking
- Win rate
- Expectancy
- Drawdown slope

### Model Decay Detection
- Performance drift alerts
- Regime mismatch detection

### Retraining Cycle
- Monthly or event-based
- Offline only

---

## 🔟 Competition-Grade Checklist

| ✅ Requirement | Status |
|---------------|--------|
| Deterministic EA | ✓ |
| Explainable logic | ✓ |
| Modular architecture | ✓ |
| Offline AI lifecycle | ✓ |
| Strict risk governance | ✓ |

---

## Summary

> **EA** = Execution Engine  
> **AI** = Decision Support  
> **Training** = Offline only  
> **Adaptation** = Strategy selection, not weight mutation
