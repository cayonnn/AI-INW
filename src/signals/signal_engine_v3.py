# src/signals/signal_engine_v3.py
"""
SignalEngine V3: Hybrid Rule + AI
=================================
Full integration of Rule-Based Teacher + AI Student + Risk Guard + HTF Filter

Mental Model:
- Rule-Based = Teacher (explainable, stable)
- XGBoost AI = Student (adaptive, data-driven)
- Risk Guard = Principal (final authority)
- HTF Filter = Context (trend confirmation)

Usage:
    engine = SignalEngineV3()
    signal, info = engine.generate_signal(df_h1, df_h4, "XAUUSD")
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# =========================
# SIGNAL RESULT
# =========================

@dataclass
class SignalResultV3:
    """Result from SignalEngine V3."""
    signal: str             # BUY / SELL / HOLD
    rule_signal: str        # Teacher signal
    ai_signal: str          # Student signal  
    htf_trend: str          # UP / DOWN / FLAT
    confidence: float       # 0-1
    reason: str             # Explanation
    allowed: bool           # Passed Risk Guard
    block_reason: str       # Why blocked


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


# =========================
# SIGNAL ENGINE V3
# =========================

class SignalEngineV3:
    """
    Hybrid Signal Engine:
    - Rule-Based = Teacher
    - XGBoost AI = Student
    - Risk Guard = Principal
    - Higher Timeframe Filter (H4)
    """
    
    def __init__(
        self,
        ai_model_path: str = "models/xgb_imitation.pkl",
        max_positions: int = 3,
        cooldown_sec: int = 30
    ):
        self.ai_model_path = ai_model_path
        self.max_positions = max_positions
        self.cooldown_sec = cooldown_sec
        
        # State
        self.last_signal: Dict[str, str] = {}
        self.last_trade_time: Dict[str, datetime] = {}
        self.positions: Dict[str, int] = {}
        
        # Load AI model
        self.ai_model = None
        self.ai_loaded = False
        self._load_ai_model()
        
        # Label mapping
        self.label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    
    def _load_ai_model(self):
        """Load XGBoost AI model."""
        if not os.path.exists(self.ai_model_path):
            print(f"⚠️ AI model not found: {self.ai_model_path}")
            return
        
        try:
            import joblib
            self.ai_model = joblib.load(self.ai_model_path)
            self.ai_loaded = True
            print(f"✅ AI model loaded: {self.ai_model_path}")
        except Exception as e:
            print(f"⚠️ Failed to load AI model: {e}")
    
    # -------------------------
    # Rule-Based Teacher
    # -------------------------
    def _rule_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Generate Rule-Based signal (Teacher)."""
        close = df['close']
        
        ema20 = ema(close, 20).iloc[-1]
        ema50 = ema(close, 50).iloc[-1]
        atr_val = atr(df, 14).iloc[-1]
        atr_mean = atr(df, 14).rolling(50).mean().iloc[-1]
        
        if pd.isna(atr_mean):
            atr_mean = atr_val
        
        trend_up = ema20 > ema50
        vol_ok = atr_val > atr_mean * 0.7
        
        if trend_up and vol_ok:
            return "BUY", 0.7
        elif not trend_up and vol_ok:
            return "SELL", 0.7
        else:
            return "HOLD", 0.3
    
    # -------------------------
    # AI Student
    # -------------------------
    def _ai_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Generate AI signal (Student)."""
        if not self.ai_loaded or self.ai_model is None:
            return "HOLD", 0.0
        
        try:
            close = df['close']
            
            # Prepare features (must match training)
            features = np.array([[
                ema(close, 20).iloc[-1],
                ema(close, 50).iloc[-1],
                ema(close, 20).iloc[-1] - ema(close, 50).iloc[-1],
                atr(df, 14).iloc[-1],
                0,  # rsi placeholder
                datetime.now().hour,
                datetime.now().weekday()
            ]])
            
            pred = self.ai_model.predict(features)[0]
            proba = self.ai_model.predict_proba(features)[0]
            confidence = float(proba.max())
            
            signal = self.label_map.get(pred, "HOLD")
            
            return signal, confidence
            
        except Exception as e:
            return "HOLD", 0.0
    
    # -------------------------
    # Higher Timeframe Filter
    # -------------------------
    def _htf_filter(self, df_h4: pd.DataFrame) -> str:
        """Get Higher Timeframe trend."""
        if df_h4 is None or len(df_h4) < 50:
            return "FLAT"
        
        close = df_h4['close']
        ema20 = ema(close, 20).iloc[-1]
        ema50 = ema(close, 50).iloc[-1]
        
        if ema20 > ema50:
            return "UP"
        elif ema20 < ema50:
            return "DOWN"
        else:
            return "FLAT"
    
    def _htf_agrees(self, signal: str, htf_trend: str) -> bool:
        """Check if signal agrees with HTF trend."""
        if signal == "BUY" and htf_trend == "UP":
            return True
        if signal == "SELL" and htf_trend == "DOWN":
            return True
        if signal == "HOLD":
            return True
        return False
    
    # -------------------------
    # Risk Guard
    # -------------------------
    def _has_open_position(self, symbol: str, direction: str) -> bool:
        """Check if there's an open position for symbol in given direction."""
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(symbol=symbol)
            if positions is None or len(positions) == 0:
                return False
            
            for pos in positions:
                pos_dir = "BUY" if pos.type == 0 else "SELL"
                if pos_dir == direction:
                    return True
            return False
        except:
            return False  # Can't check, assume no position
    
    def _risk_guard(self, symbol: str, signal: str) -> Tuple[bool, str]:
        """Apply Risk Guard rules."""
        now = datetime.now()
        
        if signal == "HOLD":
            return True, ""
        
        # Max positions
        pos = self.positions.get(symbol, 0)
        if pos >= self.max_positions:
            return False, f"Max positions ({pos}/{self.max_positions})"
        
        # Cooldown
        last_time = self.last_trade_time.get(symbol, datetime.min)
        elapsed = (now - last_time).total_seconds()
        if elapsed < self.cooldown_sec:
            return False, f"Cooldown ({self.cooldown_sec - elapsed:.0f}s)"
        
        # 🔧 FIX: Duplicate direction - NOW WITH MT5 POSITION CHECK!
        last_sig = self.last_signal.get(symbol)
        if last_sig == signal:
            # Only block if position is actually still open
            if self._has_open_position(symbol, signal):
                return False, f"Duplicate direction ({signal}) - position open"
            # Position closed (TP/SL hit) - allow re-entry!
        
        return True, ""
    
    # -------------------------
    # Main Generate Signal
    # -------------------------
    def generate_signal(
        self,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame = None,
        symbol: str = "XAUUSD"
    ) -> Tuple[str, str]:
        """
        Generate combined signal.
        
        Returns:
            (signal, info)
        """
        # Rule Teacher
        rule_sig, rule_conf = self._rule_signal(df_h1)
        
        # AI Student
        ai_sig, ai_conf = self._ai_signal(df_h1)
        
        # HTF Filter
        htf_trend = self._htf_filter(df_h4)
        
        # Combine: Consensus mode
        if rule_sig == ai_sig and rule_sig != "HOLD":
            final_signal = rule_sig
            confidence = (rule_conf + ai_conf) / 2
            reason = f"Consensus: Rule={rule_sig}, AI={ai_sig}"
        elif self._htf_agrees(rule_sig, htf_trend) and rule_sig != "HOLD":
            final_signal = rule_sig
            confidence = rule_conf
            reason = f"Rule+HTF: {rule_sig} (HTF={htf_trend})"
        else:
            final_signal = "HOLD"
            confidence = 0.3
            reason = f"No consensus: Rule={rule_sig}, AI={ai_sig}, HTF={htf_trend}"
        
        # Risk Guard
        allowed, block_reason = self._risk_guard(symbol, final_signal)
        
        if not allowed:
            info = f"HOLD (blocked: {block_reason})"
            return "HOLD", info
        
        # Update state
        if final_signal != "HOLD":
            self.last_signal[symbol] = final_signal
            self.last_trade_time[symbol] = datetime.now()
            self.positions[symbol] = self.positions.get(symbol, 0) + 1
        
        info = f"{final_signal} | {reason} | Conf={confidence:.0%}"
        return final_signal, info
    
    def generate_signal_full(
        self,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame = None,
        symbol: str = "XAUUSD"
    ) -> SignalResultV3:
        """Generate signal with full details."""
        rule_sig, rule_conf = self._rule_signal(df_h1)
        ai_sig, ai_conf = self._ai_signal(df_h1)
        htf_trend = self._htf_filter(df_h4)
        
        # Consensus
        if rule_sig == ai_sig and rule_sig != "HOLD":
            final = rule_sig
            conf = (rule_conf + ai_conf) / 2
            reason = "Consensus"
        elif self._htf_agrees(rule_sig, htf_trend):
            final = rule_sig
            conf = rule_conf
            reason = "Rule+HTF"
        else:
            final = "HOLD"
            conf = 0.3
            reason = "No alignment"
        
        allowed, block = self._risk_guard(symbol, final)
        
        if not allowed:
            final = "HOLD"
        else:
            if final != "HOLD":
                self.last_signal[symbol] = final
                self.last_trade_time[symbol] = datetime.now()
        
        return SignalResultV3(
            signal=final,
            rule_signal=rule_sig,
            ai_signal=ai_sig,
            htf_trend=htf_trend,
            confidence=conf,
            reason=reason,
            allowed=allowed,
            block_reason=block
        )


# =========================
# TEST
# =========================

if __name__ == "__main__":
    print("🧪 Testing SignalEngine V3")
    print("=" * 50)
    
    import numpy as np
    
    # Sample data
    np.random.seed(42)
    n = 100
    price = 2000 + np.cumsum(np.random.randn(n) * 5)
    
    df_h1 = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": price,
        "high": price + np.abs(np.random.randn(n) * 3),
        "low": price - np.abs(np.random.randn(n) * 3),
        "close": price + np.random.randn(n) * 2,
    })
    
    df_h4 = df_h1.copy()
    
    engine = SignalEngineV3()
    
    for i in range(5):
        signal, info = engine.generate_signal(df_h1, df_h4, "XAUUSD")
        print(f"Cycle {i+1}: {signal} | {info}")
