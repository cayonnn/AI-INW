# ⚠️ LEGACY MODULES – DO NOT USE

This folder contains deprecated logic from the old architecture.

## ❌ DEPRECATED MODULES

The following modules are NO LONGER in active use:
- `position_sizer.py` → Use `RiskManager.calc_lot()` instead
- `stoploss_calculator.py` → Use AI SL/TP models instead
- `crisis_controller.py` → Integrated into LiveLoopV3

## ✅ CURRENT STACK (Fund-Grade)

All execution MUST go through:

| Component | Location |
|-----------|----------|
| **Entry Point** | `live_loop_v3.py` |
| **Risk Management** | `src/risk/risk_manager.py` |
| **Trailing/BE** | `src/risk/trailing.py` |
| **AI SL/TP** | `models/xgb_sl.pkl`, `models/xgb_tp.pkl` |
| **Execution Contract** | `src/core/execution_contract.py` |
| **Parity Checker** | `src/core/parity_checker.py` |

## 🚫 IMPORT BLOCK

Attempting to import from this folder will raise an `ImportError`:

```python
from legacy import anything  # ❌ WILL FAIL
```

This is intentional to prevent accidental usage of deprecated logic.

## Migration Date

Deprecated: 2026-01-10
