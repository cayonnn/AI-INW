"""
signal_engine_v2.py
===================
Fund-Grade Signal Engine V2

Unified signal generation with:
- EMA Trend-Following + ATR Volatility Filter
- HTF Structure Gate (H4 trend confirmation)
- Signal Cooldown (Fund Discipline)
- Quality Tracking (Win/MAE/MFE)

Compatible with:
- Live trading (MT5Adapter)
- Backtesting (BacktestAdapter)
- AI enhancement (plug-in ready)
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .adapters import MarketDataAdapter
from .htf_filter import HTFTrendFilter
from .cooldown import SignalCooldown
from .quality_report import SignalQualityReport


# =========================
# Signal Result Structure
# =========================

@dataclass
class SignalResultV2:
    """Fund-Grade signal result."""
    action: str           # BUY / SELL / HOLD
    confidence: float     # 0.0 - 1.0
    reason: str           # Human-readable explanation
    indicators: Dict      # Raw indicator values
    htf_trend: str        # Higher timeframe trend
    blocked_by: str       # What blocked it (if HOLD)


# =========================
# Indicator Helpers
# =========================

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# =========================
# Signal Engine V2
# =========================

class SignalEngineV2:
    """
    Fund-Grade Signal Engine with full pipeline.
    
    Pipeline:
    1. Compute signal from H1 data (EMA + ATR)
    2. Filter by HTF trend (H4)
    3. Check cooldown (discipline)
    4. Record for AI training
    5. Return explainable result
    """
    
    def __init__(
        self,
        data_adapter: MarketDataAdapter,
        ema_fast: int = 20,
        ema_slow: int = 50,
        atr_period: int = 14,
        atr_lookback: int = 50,
        atr_threshold: float = 0.7,
        cooldown_seconds: int = 60,
        use_htf_filter: bool = True,
        record_for_ai: bool = True
    ):
        """
        Args:
            data_adapter: MarketDataAdapter for data access
            ema_fast: Fast EMA period
            ema_slow: Slow EMA period
            atr_period: ATR period
            atr_lookback: ATR mean lookback
            atr_threshold: ATR threshold multiplier
            cooldown_seconds: Minimum seconds between signals
            use_htf_filter: Enable HTF trend filter
            record_for_ai: Enable dataset recording for AI training
        """
        self.data = data_adapter
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.atr_lookback = atr_lookback
        self.atr_threshold = atr_threshold
        
        # Sub-components
        self.htf_filter = HTFTrendFilter(data_adapter) if use_htf_filter else None
        self.cooldown = SignalCooldown(min_interval=cooldown_seconds)
        self.quality = SignalQualityReport()
        
        self.use_htf_filter = use_htf_filter
        self.record_for_ai = record_for_ai
        
        # AI Dataset Recorder
        self.dataset_recorder = None
        if record_for_ai:
            try:
                from src.ai.dataset_recorder import get_dataset_recorder
                self.dataset_recorder = get_dataset_recorder()
            except ImportError:
                pass
    
    def compute(self, symbol: str, timeframe: str = "H1") -> SignalResultV2:
        """
        Compute signal for symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Signal timeframe (default H1)
            
        Returns:
            SignalResultV2 with action, confidence, and explanation
        """
        # Get data
        candles = self.data.get_candles(symbol, timeframe, 100)
        
        if candles.empty or len(candles) < 60:
            return SignalResultV2(
                action="HOLD",
                confidence=0,
                reason="Insufficient data",
                indicators={},
                htf_trend="UNKNOWN",
                blocked_by="data"
            )
        
        close = candles['close']
        
        # Calculate indicators
        ema_fast_val = ema(close, self.ema_fast).iloc[-1]
        ema_slow_val = ema(close, self.ema_slow).iloc[-1]
        atr_val = atr(candles, self.atr_period).iloc[-1]
        atr_mean = atr(candles, self.atr_period).rolling(self.atr_lookback).mean().iloc[-1]
        
        # Handle NaN
        if pd.isna(atr_mean):
            atr_mean = atr_val
        
        vol_threshold = atr_mean * self.atr_threshold
        
        indicators = {
            "ema_fast": round(ema_fast_val, 2),
            "ema_slow": round(ema_slow_val, 2),
            "ema_spread": round(ema_fast_val - ema_slow_val, 2),
            "atr": round(atr_val, 2),
            "atr_mean": round(atr_mean, 2),
            "atr_threshold": round(vol_threshold, 2),
            "close": round(close.iloc[-1], 2),
        }
        
        # Get HTF trend
        htf_trend = "FLAT"
        if self.htf_filter:
            htf_result = self.htf_filter.get_trend(symbol)
            htf_trend = htf_result.trend
        
        # Check volatility
        vol_ok = atr_val > vol_threshold
        if not vol_ok:
            return SignalResultV2(
                action="HOLD",
                confidence=0.2,
                reason=f"Low volatility (ATR {atr_val:.2f} < {vol_threshold:.2f})",
                indicators=indicators,
                htf_trend=htf_trend,
                blocked_by="volatility"
            )
        
        # Determine raw direction
        trend_up = ema_fast_val > ema_slow_val
        trend_down = ema_fast_val < ema_slow_val
        
        if not trend_up and not trend_down:
            return SignalResultV2(
                action="HOLD",
                confidence=0.3,
                reason="No clear trend (EMAs equal)",
                indicators=indicators,
                htf_trend=htf_trend,
                blocked_by="trend"
            )
        
        direction = "BUY" if trend_up else "SELL"
        
        # HTF filter
        if self.use_htf_filter and self.htf_filter:
            allowed, htf_reason = self.htf_filter.allows_direction(symbol, direction)
            if not allowed:
                return SignalResultV2(
                    action="HOLD",
                    confidence=0.4,
                    reason=htf_reason,
                    indicators=indicators,
                    htf_trend=htf_trend,
                    blocked_by="htf_filter"
                )
        
        # Cooldown check
        cooldown_status = self.cooldown.check(symbol, direction, indicators)
        if not cooldown_status.allowed:
            return SignalResultV2(
                action="HOLD",
                confidence=0.5,
                reason=cooldown_status.reason,
                indicators=indicators,
                htf_trend=htf_trend,
                blocked_by="cooldown"
            )
        
        # Calculate confidence
        atr_strength = min(atr_val / atr_mean, 2.0) / 2.0 if atr_mean > 0 else 0.5
        ema_strength = abs(ema_fast_val - ema_slow_val) / close.iloc[-1] * 50
        confidence = round(min(1.0, atr_strength * 0.5 + ema_strength * 0.5), 2)
        
        # Build reason
        htf_note = f", aligned with HTF {htf_trend}" if htf_trend != "FLAT" else ""
        trend_name = "Uptrend" if direction == "BUY" else "Downtrend"
        reason = f"{trend_name} (EMA{self.ema_fast} vs EMA{self.ema_slow}) + Volatility OK{htf_note}"
        
        result = SignalResultV2(
            action=direction,
            confidence=confidence,
            reason=reason,
            indicators=indicators,
            htf_trend=htf_trend,
            blocked_by=""
        )
        
        # Record for AI training
        self._record_decision(symbol, result)
        
        return result
    
    def _record_decision(self, symbol: str, result: SignalResultV2):
        """Record decision for AI imitation learning."""
        if self.dataset_recorder:
            try:
                self.dataset_recorder.record(
                    symbol=symbol,
                    action=result.action,
                    confidence=result.confidence,
                    reason=result.reason,
                    indicators=result.indicators,
                    blocked_by=result.blocked_by,
                    htf_trend=result.htf_trend
                )
            except Exception:
                pass  # Don't let recording errors affect trading
    
    def record_signal(self, symbol: str, direction: str, indicators: Dict):
        """Record that a signal was executed."""
        self.cooldown.record(symbol, direction, indicators)
    
    def get_quality_summary(self):
        """Get quality report summary."""
        return self.quality.get_summary()
    
    def get_dataset_stats(self) -> Dict:
        """Get AI dataset statistics."""
        if self.dataset_recorder:
            return self.dataset_recorder.get_stats()
        return {"total": 0}


# =========================
# Factory Function
# =========================

def get_signal_engine_v2(
    data_adapter: MarketDataAdapter,
    config: Dict[str, Any] = None
) -> SignalEngineV2:
    """
    Factory function for Signal Engine V2.
    
    Args:
        data_adapter: MarketDataAdapter instance
        config: Optional configuration overrides
        
    Returns:
        SignalEngineV2 instance
    """
    config = config or {}
    
    return SignalEngineV2(
        data_adapter=data_adapter,
        ema_fast=config.get("ema_fast", 20),
        ema_slow=config.get("ema_slow", 50),
        atr_period=config.get("atr_period", 14),
        atr_lookback=config.get("atr_lookback", 50),
        atr_threshold=config.get("atr_threshold", 0.7),
        cooldown_seconds=config.get("cooldown_seconds", 60),
        use_htf_filter=config.get("use_htf_filter", True),
        record_for_ai=config.get("record_for_ai", True)
    )
