"""
inference_xgb.py
================
XGBoost AI Inference Module with MT5 Integration

Pipeline:
1. Load trained XGBoost model
2. Generate AI signal from features
3. Apply confidence threshold (low confidence → HOLD)
4. Check Risk Guard (max positions, cooldown, anti-dup)
5. Execute trade via MT5 command

Mental Model:
- XGBoost = Student (makes predictions)
- Risk Guard = Principal (final authority)
- Rule-Based = Fallback (if AI not confident)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# =========================
# CONFIG
# =========================

MODEL_PATH = "models/xgb_signal.model"
CONFIDENCE_THRESHOLD = 0.6  # Below this → HOLD

# Signal mapping
CLASS_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
REVERSE_MAP = {"SELL": 0, "HOLD": 1, "BUY": 2}

# Feature columns (must match training)
FEATURE_COLUMNS = [
    "ema_fast",
    "ema_slow", 
    "ema_spread",
    "ema_slope",
    "atr",
    "atr_ratio",
    "atr_threshold",
    "htf_trend",
    "volatility_ok",
    "hour",
    "day_of_week",
]


# =========================
# AI SIGNAL RESULT
# =========================

@dataclass
class AIInferenceResult:
    """AI inference result."""
    signal: str           # BUY / SELL / HOLD
    confidence: float     # 0.0 - 1.0
    raw_proba: Dict       # Probability for each class
    reason: str           # Explanation
    passed_guard: bool    # Passed risk guard check
    guard_reason: str     # Why blocked (if any)


# =========================
# RISK GUARD
# =========================

class InferenceRiskGuard:
    """Risk guard for AI inference."""
    
    def __init__(
        self,
        max_positions: int = 3,
        cooldown_sec: int = 30,
        anti_dup: bool = True
    ):
        self.max_positions = max_positions
        self.cooldown_sec = cooldown_sec
        self.anti_dup = anti_dup
        
        self.last_trade: Dict[str, dict] = {}  # {symbol: {time, dir}}
        self.positions: Dict[str, int] = {}    # {symbol: count}
    
    def check(self, symbol: str, signal: str) -> Tuple[bool, str]:
        """Check if trade is allowed."""
        now = datetime.now().timestamp()
        
        # HOLD always allowed
        if signal == "HOLD":
            return True, "HOLD signal"
        
        # Max positions check
        pos_count = self.positions.get(symbol, 0)
        if pos_count >= self.max_positions:
            return False, f"Max positions reached ({pos_count}/{self.max_positions})"
        
        # Anti-duplicate and cooldown
        last = self.last_trade.get(symbol)
        if last:
            elapsed = now - last["time"]
            
            # Cooldown check
            if elapsed < self.cooldown_sec:
                return False, f"Cooldown ({self.cooldown_sec - elapsed:.0f}s remaining)"
            
            # Anti-dup check
            if self.anti_dup and last["dir"] == signal:
                return False, f"Duplicate direction blocked (last={last['dir']})"
        
        return True, "OK"
    
    def record_trade(self, symbol: str, signal: str):
        """Record successful trade."""
        now = datetime.now().timestamp()
        self.last_trade[symbol] = {"time": now, "dir": signal}
        self.positions[symbol] = self.positions.get(symbol, 0) + 1
    
    def update_positions(self, symbol: str, count: int):
        """Update position count (from MT5)."""
        self.positions[symbol] = count


# =========================
# XGB INFERENCE ENGINE
# =========================

class XGBInferenceEngine:
    """XGBoost inference engine with risk guard."""
    
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.loaded = False
        
        self.risk_guard = InferenceRiskGuard()
    
    def load_model(self) -> bool:
        """Load XGBoost model."""
        if not HAS_XGB:
            print("❌ XGBoost not installed")
            return False
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            return False
        
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            self.loaded = True
            print(f"✅ Model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def infer(
        self,
        features: Dict,
        symbol: str = "XAUUSD"
    ) -> AIInferenceResult:
        """
        Run inference on features.
        
        Args:
            features: Dict with feature values
            symbol: Trading symbol
            
        Returns:
            AIInferenceResult
        """
        # Fallback if model not loaded
        if not self.loaded:
            return AIInferenceResult(
                signal="HOLD",
                confidence=0,
                raw_proba={"SELL": 0, "HOLD": 1, "BUY": 0},
                reason="Model not loaded",
                passed_guard=True,
                guard_reason=""
            )
        
        try:
            # Prepare features
            df = self._prepare_features(features)
            
            # Get prediction
            proba = self.model.predict_proba(df)[0]
            class_idx = int(np.argmax(proba))
            confidence = float(proba[class_idx])
            
            raw_proba = {
                "SELL": float(proba[0]),
                "HOLD": float(proba[1]),
                "BUY": float(proba[2])
            }
            
            # Determine signal
            if confidence < self.confidence_threshold:
                signal = "HOLD"
                reason = f"Low confidence ({confidence:.0%} < {self.confidence_threshold:.0%})"
            else:
                signal = CLASS_MAP[class_idx]
                reason = f"AI predicts {signal} with {confidence:.0%} confidence"
            
            # Risk guard check
            passed, guard_reason = self.risk_guard.check(symbol, signal)
            
            if not passed:
                final_signal = "HOLD"
                reason = f"{reason} → Blocked by Risk Guard: {guard_reason}"
            else:
                final_signal = signal
            
            return AIInferenceResult(
                signal=final_signal,
                confidence=confidence,
                raw_proba=raw_proba,
                reason=reason,
                passed_guard=passed,
                guard_reason=guard_reason if not passed else ""
            )
            
        except Exception as e:
            return AIInferenceResult(
                signal="HOLD",
                confidence=0,
                raw_proba={"SELL": 0, "HOLD": 1, "BUY": 0},
                reason=f"Inference error: {e}",
                passed_guard=False,
                guard_reason=str(e)
            )
    
    def _prepare_features(self, features: Dict) -> pd.DataFrame:
        """Prepare features as DataFrame."""
        # Ensure all required columns
        row = {}
        for col in FEATURE_COLUMNS:
            row[col] = features.get(col, 0)
        
        return pd.DataFrame([row])
    
    def record_trade(self, symbol: str, signal: str):
        """Record executed trade."""
        self.risk_guard.record_trade(symbol, signal)


# =========================
# FACTORY FUNCTION
# =========================

_inference_engine: Optional[XGBInferenceEngine] = None


def get_inference_engine() -> XGBInferenceEngine:
    """Get singleton inference engine."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = XGBInferenceEngine()
        _inference_engine.load_model()
    return _inference_engine


def infer_signal(features: Dict, symbol: str = "XAUUSD") -> Tuple[str, float]:
    """
    Quick inference function.
    
    Args:
        features: Feature dict
        symbol: Trading symbol
        
    Returns:
        (signal, confidence)
    """
    engine = get_inference_engine()
    result = engine.infer(features, symbol)
    return result.signal, result.confidence


# =========================
# TEST
# =========================

if __name__ == "__main__":
    print("🧪 Testing XGBoost Inference Engine")
    print("=" * 50)
    
    # Create engine
    engine = XGBInferenceEngine()
    
    # Load model
    if engine.load_model():
        # Test features
        test_features = {
            "ema_fast": 4454.64,
            "ema_slow": 4449.64,
            "ema_spread": 5.0,
            "ema_slope": 0.0,
            "atr": 16.09,
            "atr_ratio": 0.0036,
            "atr_threshold": 11.07,
            "htf_trend": 1,
            "volatility_ok": 1,
            "hour": 9,
            "day_of_week": 4,
        }
        
        result = engine.infer(test_features, "XAUUSD")
        
        print(f"\n📊 Result:")
        print(f"   Signal: {result.signal}")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   Reason: {result.reason}")
        print(f"   Passed Guard: {result.passed_guard}")
        print(f"\n   Probabilities:")
        for cls, prob in result.raw_proba.items():
            print(f"      {cls}: {prob:.1%}")
    else:
        print("\n⚠️ Model not available. Train first:")
        print("   python src/ai/train_xgboost.py")
