# 📁 AI Trading System - Complete Project Structure

> สำหรับ Review โดย OpenChat / Quant Team / Auditor

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI TRADING SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│  .agent/prompts/     │  Claude AI Engineer Prompts             │
├──────────────────────┼──────────────────────────────────────────┤
│  src/                │  Python Brain (Research / Decision)     │
├──────────────────────┼──────────────────────────────────────────┤
│  mt5_ea/             │  MQL5 Execution Engine (Hands)          │
├──────────────────────┼──────────────────────────────────────────┤
│  config/             │  Configuration (No Recompile Change)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Complete File Tree (38 Files)

```
f:\Mobile App\AI trade แขงขัน\
│
├── 📄 README.md                          # Project overview
├── 📄 requirements.txt                   # Python dependencies
│
├── 📁 .agent/prompts/                    # 🧠 Claude AI Engineer
│   ├── README.md
│   ├── 01_research_model_development.txt
│   ├── 02_signal_fusion_decision.txt
│   ├── 03_mt5_execution_ea.txt
│   ├── 04_refactor_improve.txt
│   ├── 05_audit_stress_test.txt
│   └── 06_competition_optimization.txt
│
├── 📁 config/                            # ⚙️ Configuration
│   ├── settings.yaml                     # Global settings
│   ├── symbols.yaml                      # Symbol configs
│   └── risk_params.yaml                  # Risk management rules
│
├── 📁 docs/
│   └── PROJECT_WORKFLOW.md               # 8-Phase workflow
│
├── 📁 src/                               # 🧠 Python Brain
│   │
│   ├── orchestrator.py                   # 🎯 Central Controller
│   │
│   ├── 📁 data/                          # Data Layer
│   │   ├── __init__.py
│   │   ├── mt5_connector.py              # MT5 data fetch
│   │   ├── data_processor.py             # Data cleaning
│   │   └── data_validator.py             # Data quality check
│   │
│   ├── 📁 features/                      # Feature Engineering
│   │   ├── __init__.py
│   │   ├── price_action.py               # Candles, swings, S/R
│   │   ├── trend_momentum.py             # EMA, RSI, MACD, ADX
│   │   ├── volatility.py                 # ATR, Bollinger, Vol%
│   │   └── regime_detector.py            # Market regime
│   │
│   ├── 📁 models/                        # AI Models (Offline)
│   │   ├── __init__.py
│   │   ├── lstm_direction.py             # LSTM for direction
│   │   ├── xgb_timing.py                 # XGBoost for timing
│   │   ├── signal_schema.py              # Standard signal format
│   │   └── model_hotswap.py              # Model hot-reload
│   │
│   ├── 📁 signals/                       # Signal Processing
│   │   ├── __init__.py
│   │   ├── signal_fusion.py              # Multi-model fusion
│   │   ├── decision_gate.py              # Final trade gate
│   │   ├── multi_timeframe.py            # MTF analysis
│   │   └── multi_symbol_manager.py       # Correlation control
│   │
│   ├── 📁 risk/                          # Risk Management
│   │   ├── __init__.py
│   │   ├── position_sizer.py             # Fixed fractional
│   │   ├── stop_loss.py                  # ATR-based SL/TP
│   │   └── auto_disable.py               # Kill switch
│   │
│   └── 📁 utils/                         # Utilities
│       ├── __init__.py
│       ├── logger.py                     # Structured logging
│       ├── config_loader.py              # YAML loader
│       ├── validators.py                 # Input validation
│       └── analytics.py                  # Performance tracking
│
├── 📁 training/                          # 🎓 Offline Training
│   └── train_models.py                   # Model training script
│
├── 📁 backtesting/                       # 🧪 Validation
│   └── backtest_engine.py                # Backtesting framework
│
└── 📁 mt5_ea/                            # 🤖 MT5 Execution
    └── AI_Trading_EA.mq5                 # Production EA
```

---

## 🔍 Module Responsibilities

### Core Pipeline Flow
```
Market Data → Features → AI Models → Signal Fusion → Decision Gate → Risk → Execute
```

### Module Details

| Module | Role | Train Live? |
|--------|------|-------------|
| `data/` | Fetch & validate OHLCV | ❌ |
| `features/` | Technical indicators | ❌ |
| `models/` | AI prediction (LSTM/XGB) | ❌ Offline only |
| `signals/` | Fuse & filter signals | ❌ |
| `risk/` | Position sizing, SL/TP | ❌ Deterministic |
| `mt5_ea/` | Order execution | ❌ No ML |

---

## ✅ Competition-Grade Checklist

| Requirement | Status | File |
|-------------|--------|------|
| Deterministic EA | ✅ | `AI_Trading_EA.mq5` |
| Offline training only | ✅ | `training/train_models.py` |
| Signal fusion (not single model) | ✅ | `signal_fusion.py` |
| Risk governance | ✅ | `auto_disable.py` |
| Model hot-swap | ✅ | `model_hotswap.py` |
| Multi-TF support | ✅ | `multi_timeframe.py` |
| Correlation control | ✅ | `multi_symbol_manager.py` |
| Audit-ready prompts | ✅ | `.agent/prompts/` |

---

## 🧠 Key Design Principles

1. **EA ไม่คิดเอง** — แค่รับคำสั่งและ execute
2. **AI ไม่ train live** — Offline only
3. **Risk ปิดระบบได้** — Auto-disable on drawdown
4. **เปลี่ยนโมเดลไม่ต้อง recompile** — Hot-swap ONNX
5. **ทุกอย่าง explain ได้** — No black box

---

## 📊 File Count Summary

| Category | Files |
|----------|-------|
| Config | 3 |
| Data Layer | 4 |
| Features | 5 |
| Models | 5 |
| Signals | 5 |
| Risk | 4 |
| Utils | 5 |
| Training | 1 |
| Backtesting | 1 |
| MT5 EA | 1 |
| Docs | 2 |
| Prompts | 7 |
| **Total** | **43** |

---

> **สรุป**: โครงสร้างนี้เป็นระดับ **Prop Firm / Quant Fund / Algo Contest**  
> พร้อมสำหรับ production และ audit
