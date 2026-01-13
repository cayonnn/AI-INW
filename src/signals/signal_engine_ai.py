"""
signal_engine_ai.py
====================
SignalEngine V2 + AI Student Integration

Features:
- Rule-Based Teacher (EMA + ATR)
- AI Student (XGBoost Multi-class)
- Higher Timeframe Filter (H1 → H4)
- Risk Guard (max positions, cooldown, anti-dup)

Mental Model:
- Teacher = Rule-Based (explainable, stable)
- Student = AI Model (adaptive, data-driven)
- Principal = Risk Guard (final authority)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# =========================
# SIGNAL RESULT
# =========================

@dataclass
class AISignalResult:
    """Combined signal result."""
    final_signal: str          # BUY / SELL / HOLD
    teacher_signal: str        # Rule-based signal
    student_signal: str        # AI signal
    htf_trend: str             # Higher timeframe trend
    confidence: float          # AI confidence
    reason: str                # Explanation
    blocked: bool              # Blocked by Risk Guard
    block_reason: str          # Why blocked


# =========================
# INDICATOR FUNCTIONS
# =========================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, 0.001)
    return 100 - (100 / (1 + rs))


# =========================
# SIGNAL ENGINE AI
# =========================

class SignalEngineAI:
    """
    Hybrid Signal Engine with Teacher + Student + Risk Guard.
    """
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "H1",
        htf: str = "H4",
        model_path: str = "models/xgb_signal_model.pkl",
        max_positions: int = 3,
        cooldown_seconds: int = 30,
        mode: str = "hybrid"  # "teacher", "student", "hybrid", "consensus"
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.htf = htf
        self.mode = mode
        
        # Risk Guard state
        self.max_positions = max_positions
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.positions = 0
        self.last_signal = None
        self.last_time = datetime.min
        
        # Load AI Model
        self.model = None
        self.model_features = None
        self.ai_loaded = False
        self._load_model(model_path)
        
        # Label mapping
        self.label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    
    def _load_model(self, path: str):
        """Load AI model from file."""
        if not HAS_JOBLIB:
            print("⚠️ joblib not installed")
            return
            
        if not os.path.exists(path):
            print(f"⚠️ Model not found: {path}")
            return
        
        try:
            data = joblib.load(path)
            if isinstance(data, dict):
                self.model = data.get("model")
                self.model_features = data.get("feature_names", [])
            else:
                self.model = data
            self.ai_loaded = True
            print(f"✅ AI Model loaded: {path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    
    # -----------------------------
    # Rule-Based Teacher
    # -----------------------------
    def rule_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Generate rule-based signal (Teacher).
        
        Returns:
            (signal, confidence)
        """
        close = df['close']
        
        # Calculate indicators
        ema12 = ema(close, 12).iloc[-1]
        ema26 = ema(close, 26).iloc[-1]
        atr_val = atr(df, 14).iloc[-1]
        atr_mean = atr(df, 14).rolling(50).mean().iloc[-1]
        
        if pd.isna(atr_mean):
            atr_mean = atr_val
        
        atr_threshold = atr_mean * 0.7
        vol_ok = atr_val > atr_threshold
        
        # Generate signal
        if ema12 > ema26 and vol_ok:
            signal = "BUY"
            confidence = min(1.0, (ema12 - ema26) / close.iloc[-1] * 100)
        elif ema12 < ema26 and vol_ok:
            signal = "SELL"
            confidence = min(1.0, (ema26 - ema12) / close.iloc[-1] * 100)
        else:
            signal = "HOLD"
            confidence = 0.3
        
        return signal, round(confidence, 2)
    
    # -----------------------------
    # AI Student Prediction
    # -----------------------------
    def ai_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Generate AI signal (Student).
        
        Returns:
            (signal, confidence)
        """
        if not self.ai_loaded or self.model is None:
            return "HOLD", 0.0
        
        try:
            # Prepare features
            close = df['close']
            
            features = {
                "ema20": ema(close, 20).iloc[-1],
                "ema50": ema(close, 50).iloc[-1],
                "ema_spread": ema(close, 20).iloc[-1] - ema(close, 50).iloc[-1],
                "atr14": atr(df, 14).iloc[-1],
                "atr_ratio": atr(df, 14).iloc[-1] / close.iloc[-1],
                "rsi14": rsi(close, 14).iloc[-1],
                "hour": datetime.now().hour,
                "day_of_week": datetime.now().weekday(),
            }
            
            # Use model features if available
            if self.model_features:
                X = pd.DataFrame([{k: features.get(k, 0) for k in self.model_features}])
            else:
                X = pd.DataFrame([features])
            
            X = X.fillna(0)
            
            # Predict
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba.max())
            
            signal = self.label_map.get(pred, "HOLD")
            
            return signal, round(confidence, 2)
            
        except Exception as e:
            print(f"⚠️ AI prediction error: {e}")
            return "HOLD", 0.0
    
    # -----------------------------
    # Higher Timeframe Filter
    # -----------------------------
    def htf_filter(self, df_htf: pd.DataFrame) -> str:
        """
        Get higher timeframe trend direction.
        
        Returns:
            "UP", "DOWN", or "FLAT"
        """
        if df_htf.empty or len(df_htf) < 30:
            return "FLAT"
        
        close = df_htf['close']
        ema12 = ema(close, 12).iloc[-1]
        ema26 = ema(close, 26).iloc[-1]
        
        if ema12 > ema26:
            return "UP"
        elif ema12 < ema26:
            return "DOWN"
        else:
            return "FLAT"
    
    # -----------------------------
    # Risk Guard
    # -----------------------------
    def risk_check(self, signal: str) -> Tuple[bool, str]:
        """
        Check if signal passes Risk Guard.
        
        Returns:
            (allowed, reason)
        """
        now = datetime.now()
        
        # HOLD always allowed
        if signal == "HOLD":
            return True, "HOLD signal"
        
        # Max positions
        if self.positions >= self.max_positions:
            return False, f"Max positions ({self.positions}/{self.max_positions})"
        
        # Cooldown
        if now - self.last_time < self.cooldown:
            remaining = (self.cooldown - (now - self.last_time)).seconds
            return False, f"Cooldown ({remaining}s remaining)"
        
        # Anti-duplicate
        if self.last_signal == signal:
            return False, f"Duplicate direction ({signal})"
        
        return True, "OK"
    
    # -----------------------------
    # Generate Combined Signal
    # -----------------------------
    def generate_signal(
        self,
        df: pd.DataFrame,
        df_htf: pd.DataFrame = None
    ) -> AISignalResult:
        """
        Generate combined signal from Teacher + Student + HTF + Risk Guard.
        
        Args:
            df: Main timeframe OHLC data
            df_htf: Higher timeframe OHLC data (optional)
            
        Returns:
            AISignalResult with full details
        """
        # Get Teacher signal
        teacher_sig, teacher_conf = self.rule_signal(df)
        
        # Get Student signal
        student_sig, student_conf = self.ai_signal(df)
        
        # Get HTF trend
        htf_trend = self.htf_filter(df_htf) if df_htf is not None else "FLAT"
        
        # Determine final signal based on mode
        if self.mode == "teacher":
            final_signal = teacher_sig
            confidence = teacher_conf
            reason = f"Teacher: {teacher_sig}"
            
        elif self.mode == "student":
            final_signal = student_sig
            confidence = student_conf
            reason = f"Student: {student_sig} ({student_conf:.0%})"
            
        elif self.mode == "consensus":
            # Both must agree
            if teacher_sig == student_sig and teacher_sig != "HOLD":
                final_signal = teacher_sig
                confidence = (teacher_conf + student_conf) / 2
                reason = f"Consensus: Teacher={teacher_sig}, Student={student_sig}"
            else:
                final_signal = "HOLD"
                confidence = 0.5
                reason = f"No consensus: Teacher={teacher_sig}, Student={student_sig}"
                
        else:  # hybrid (default)
            # Use HTF + Student for final decision
            if htf_trend == "UP" and student_sig == "BUY":
                final_signal = "BUY"
                confidence = student_conf
                reason = f"HTF UP + AI BUY"
            elif htf_trend == "DOWN" and student_sig == "SELL":
                final_signal = "SELL"
                confidence = student_conf
                reason = f"HTF DOWN + AI SELL"
            elif teacher_sig != "HOLD" and htf_trend == "FLAT":
                final_signal = teacher_sig
                confidence = teacher_conf
                reason = f"Fallback to Teacher: {teacher_sig}"
            else:
                final_signal = "HOLD"
                confidence = 0.3
                reason = f"No alignment: HTF={htf_trend}, AI={student_sig}, Teacher={teacher_sig}"
        
        # Risk Guard check
        allowed, block_reason = self.risk_check(final_signal)
        
        if not allowed:
            blocked_signal = final_signal
            final_signal = "HOLD"
            reason = f"Blocked: {block_reason} (wanted {blocked_signal})"
        
        # Update state if trade allowed
        if final_signal != "HOLD" and allowed:
            self.last_signal = final_signal
            self.last_time = datetime.now()
            self.positions += 1
        
        return AISignalResult(
            final_signal=final_signal,
            teacher_signal=teacher_sig,
            student_signal=student_sig,
            htf_trend=htf_trend,
            confidence=confidence,
            reason=reason,
            blocked=not allowed,
            block_reason=block_reason if not allowed else ""
        )
    
    def reset_positions(self):
        """Reset position count (call when position closed)."""
        self.positions = max(0, self.positions - 1)


# =========================
# FACTORY FUNCTION
# =========================

def get_signal_engine_ai(
    symbol: str = "XAUUSD",
    mode: str = "hybrid"
) -> SignalEngineAI:
    """Get SignalEngine AI instance."""
    return SignalEngineAI(symbol=symbol, mode=mode)


# =========================
# TEST
# =========================

if __name__ == "__main__":
    print("🧪 Testing SignalEngine AI")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n = 100
    price = 2000 + np.cumsum(np.random.randn(n) * 5)
    
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="H"),
        "open": price,
        "high": price + np.abs(np.random.randn(n) * 3),
        "low": price - np.abs(np.random.randn(n) * 3),
        "close": price + np.random.randn(n) * 2,
        "volume": np.random.randint(100, 1000, n)
    })
    
    df_htf = df.resample("4H", on="time").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()
    
    # Test engine
    engine = SignalEngineAI(mode="teacher")
    
    result = engine.generate_signal(df, df_htf)
    
    print(f"\n📊 Signal Result:")
    print(f"   Final: {result.final_signal}")
    print(f"   Teacher: {result.teacher_signal}")
    print(f"   Student: {result.student_signal}")
    print(f"   HTF Trend: {result.htf_trend}")
    print(f"   Confidence: {result.confidence:.0%}")
    print(f"   Reason: {result.reason}")
    print(f"   Blocked: {result.blocked}")
