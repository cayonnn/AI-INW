# AI-Assisted Trading System for MetaTrader 5

Professional Quant-style algorithmic trading platform with AI signal generation and deterministic risk management.

## 🎯 Core Philosophy

- **AI Role**: Market intelligence & signal generation ONLY
- **Risk Management**: Deterministic, rule-based, explainable (NO ML)
- **Prediction Target**: Probabilities, confidence, regimes — NOT raw prices
- **Architecture**: Modular, replaceable, testable components

## 📁 Project Structure

```
├── config/                 # Configuration files
│   ├── settings.yaml      # Global settings
│   ├── symbols.yaml       # Symbol configurations
│   └── risk_params.yaml   # Risk parameters
├── src/                    # Python source code
│   ├── data/              # MT5 data layer
│   ├── features/          # Feature engineering
│   ├── models/            # AI models (LSTM, XGBoost)
│   ├── signals/           # Signal fusion & decision
│   ├── risk/              # Risk management
│   └── utils/             # Utilities
├── training/              # Model training scripts
├── backtesting/           # Backtesting framework
├── mt5_ea/                # MetaTrader 5 Expert Advisor
└── artifacts/             # Saved models
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure MT5 Connection

Set environment variables:
```bash
export MT5_LOGIN=your_login
export MT5_PASSWORD=your_password
export MT5_SERVER=your_broker_server
```

### 3. Train Models

```bash
python training/train_models.py --model all --symbol EURUSD
```

### 4. Deploy EA

1. Copy `mt5_ea/AI_Trading_EA.mq5` to MT5 `Experts` folder
2. Copy ONNX model to `MQL5/Files/Models/`
3. Compile and attach to chart

## 📊 AI Signal Schema

```python
AISignal:
  direction: LONG | SHORT | NEUTRAL
  probability: 0.0 - 1.0
  confidence: 0.0 - 1.0
  volatility_state: LOW | NORMAL | HIGH
  regime: TRENDING | RANGING | VOLATILE | QUIET
  expected_rr: float
```

## ⚠️ Risk Management Rules (Deterministic)

| Rule | Default |
|------|---------|
| Risk per trade | 1% |
| ATR SL multiplier | 2.0x |
| Min R:R | 1.5:1 |
| Max exposure | 15% |
| Max drawdown | 10% |

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📈 Performance Metrics

- Max Drawdown
- Profit Factor
- Sharpe Ratio
- Expectancy
- Win Rate

## ⚡ Competition Ready

- Walk-forward validation
- Regime-aware strategy switching
- Model hot-swap capability
- Auto-disable on abnormal conditions

---

**Warning**: This system is for educational and research purposes. Trading involves significant risk.
