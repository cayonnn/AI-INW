"""
ai_signal.py
============
AI Signal Engine (Student)

Mental Model:
- Rule-Based SignalEngine V2 = Teacher (provides training data)
- AI Signal = Student (learns to imitate teacher decisions)
- Risk Guard = Principal (supervises and controls both)

The AI Signal NEVER bypasses Risk Guard.
It only replaces the signal generation logic, not execution safety.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path
import pickle


@dataclass
class AISignalResult:
    """AI Signal output."""
    action: str           # BUY / SELL / HOLD
    confidence: float     # Model prediction probability
    reason: str           # Explanation
    indicators: Dict      # Input features used
    model_type: str       # Model name
    teacher_action: str   # What rule-based would say (for comparison)


class AISignal:
    """
    AI-based signal generator (Student).
    
    Learns from Rule-Based SignalEngine V2 (Teacher).
    Always subject to Risk Guard (Principal).
    
    Usage:
        ai = AISignal()
        ai.load_model("models/imitation_xgb.pkl")
        result = ai.predict(indicators)
    """
    
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Path to trained model (optional, load later)
        """
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.model_type = "unknown"
        self.loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path: str = "models/imitation_xgb.pkl"):
        """Load trained imitation model."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_columns = model_data["feature_columns"]
        self.model_type = model_data.get("model_type", "xgboost")
        self.loaded = True
        
        print(f"✅ AI Signal loaded: {self.model_type}")
    
    def predict(
        self,
        indicators: Dict,
        teacher_action: str = "UNKNOWN"
    ) -> AISignalResult:
        """
        Predict action from indicators.
        
        Args:
            indicators: Dict with feature values (from SignalEngine V2)
            teacher_action: What rule-based would say (for comparison/logging)
            
        Returns:
            AISignalResult with prediction
        """
        if not self.loaded:
            return AISignalResult(
                action="HOLD",
                confidence=0,
                reason="Model not loaded",
                indicators=indicators,
                model_type="none",
                teacher_action=teacher_action
            )
        
        try:
            # Prepare features
            features = self._prepare_features(indicators)
            X = pd.DataFrame([features])[self.feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Predict
            pred = self.model.predict(X_scaled)[0]
            proba = self.model.predict_proba(X_scaled)[0]
            
            # Decode prediction
            action_code = self.label_encoder.inverse_transform([pred])[0]
            confidence = float(proba.max())
            
            # Map to action string
            action_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
            action = action_map.get(action_code, "HOLD")
            
            # Build reason
            reason = f"AI predicts {action} with {confidence:.0%} confidence"
            if action != teacher_action and teacher_action != "UNKNOWN":
                reason += f" (Teacher: {teacher_action})"
            
            return AISignalResult(
                action=action,
                confidence=confidence,
                reason=reason,
                indicators=indicators,
                model_type=self.model_type,
                teacher_action=teacher_action
            )
            
        except Exception as e:
            return AISignalResult(
                action="HOLD",
                confidence=0,
                reason=f"Prediction error: {e}",
                indicators=indicators,
                model_type=self.model_type,
                teacher_action=teacher_action
            )
    
    def _prepare_features(self, indicators: Dict) -> Dict:
        """Prepare features from SignalEngine indicators."""
        # Map SignalEngine V2 indicators to feature format
        close = indicators.get("close", 1)
        atr = indicators.get("atr", 0)
        
        return {
            "ema_fast": indicators.get("ema_fast", 0),
            "ema_slow": indicators.get("ema_slow", 0),
            "ema_spread": indicators.get("ema_spread", 0),
            "ema_slope": 0,  # Will be calculated from history
            "atr": atr,
            "atr_ratio": atr / close if close > 0 else 0,
            "atr_threshold": indicators.get("atr_threshold", 0),
            "htf_trend": self._encode_htf(indicators.get("htf_trend", "FLAT")),
            "volatility_ok": 1 if indicators.get("volatility_ok", False) else 0,
            "hour": pd.Timestamp.utcnow().hour,
            "day_of_week": pd.Timestamp.utcnow().weekday(),
        }
    
    def _encode_htf(self, htf_trend: str) -> int:
        """Encode HTF trend string to numeric."""
        if htf_trend == "BULL":
            return 1
        elif htf_trend == "BEAR":
            return -1
        return 0


# =========================
# Hybrid Signal (Teacher + Student + Principal)
# =========================

class HybridSignal:
    """
    Hybrid signal combining Rule-Based (Teacher) and AI (Student).
    
    Modes:
    - "teacher": Only use rule-based (default, safe)
    - "student": Only use AI (requires trained model)
    - "consensus": Trade only when both agree
    - "ai_override": AI can override teacher (risky)
    
    All modes still go through Risk Guard (Principal)!
    """
    
    def __init__(
        self,
        mode: str = "teacher",
        ai_model_path: str = None,
        min_ai_confidence: float = 0.6
    ):
        """
        Args:
            mode: Signal mode
            ai_model_path: Path to AI model (required for student/consensus/ai_override)
            min_ai_confidence: Minimum AI confidence to act
        """
        self.mode = mode
        self.min_ai_confidence = min_ai_confidence
        
        self.ai_signal = None
        if mode != "teacher" and ai_model_path:
            self.ai_signal = AISignal(ai_model_path)
    
    def compute(
        self,
        teacher_action: str,
        teacher_confidence: float,
        indicators: Dict
    ) -> Tuple[str, float, str]:
        """
        Compute final signal based on mode.
        
        Args:
            teacher_action: Rule-based signal action
            teacher_confidence: Rule-based confidence
            indicators: Feature indicators
            
        Returns:
            (action, confidence, reason)
        """
        # Teacher-only mode (safest)
        if self.mode == "teacher":
            return teacher_action, teacher_confidence, "Teacher signal"
        
        # Get AI prediction
        if self.ai_signal is None or not self.ai_signal.loaded:
            return teacher_action, teacher_confidence, "AI not available, using Teacher"
        
        ai_result = self.ai_signal.predict(indicators, teacher_action)
        
        # Student-only mode
        if self.mode == "student":
            return ai_result.action, ai_result.confidence, ai_result.reason
        
        # Consensus mode (both must agree)
        if self.mode == "consensus":
            if ai_result.action == teacher_action:
                combined_conf = (teacher_confidence + ai_result.confidence) / 2
                return teacher_action, combined_conf, f"Consensus: Teacher + AI agree on {teacher_action}"
            else:
                return "HOLD", 0.5, f"No consensus: Teacher={teacher_action}, AI={ai_result.action}"
        
        # AI Override mode (AI can override if confident)
        if self.mode == "ai_override":
            if ai_result.confidence >= self.min_ai_confidence:
                return ai_result.action, ai_result.confidence, f"AI override: {ai_result.reason}"
            else:
                return teacher_action, teacher_confidence, f"AI not confident ({ai_result.confidence:.0%}), using Teacher"
        
        # Default: Teacher
        return teacher_action, teacher_confidence, "Default to Teacher"


# =========================
# Factory
# =========================

def get_ai_signal(model_path: str = "models/imitation_xgb.pkl") -> AISignal:
    """Get AI Signal instance."""
    return AISignal(model_path)


def get_hybrid_signal(
    mode: str = "teacher",
    model_path: str = "models/imitation_xgb.pkl"
) -> HybridSignal:
    """Get Hybrid Signal instance."""
    return HybridSignal(mode=mode, ai_model_path=model_path)
