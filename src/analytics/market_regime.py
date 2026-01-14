# src/analytics/market_regime.py
"""
Market Regime Detector - Competition Grade
===========================================

Detects market regime using multiple indicators:
- ADX (trend strength)
- EMA slope (direction)
- ATR expansion (volatility)
- RSI chop zone detection

Regime Types:
- STRONG_TREND: Clear directional move
- WEAK_TREND: Mild trend
- CHOP: Ranging/choppy market

Each regime maps to a trading mode (Alpha/Neutral/Defensive).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("MARKET_REGIME")


class Regime(str, Enum):
    """Market regime types."""
    STRONG_TREND = "STRONG_TREND"
    WEAK_TREND = "WEAK_TREND"
    CHOP = "CHOP"


@dataclass
class RegimeResult:
    """Regime detection result."""
    regime: Regime
    trend_score: float      # 0-100, higher = stronger trend
    direction: str          # UP, DOWN, or NEUTRAL
    adx_value: float
    ema_slope: float
    atr_expansion: float
    rsi_chop: bool
    confidence: float       # 0-1


# Regime to Mode mapping
REGIME_MODE_MAP = {
    Regime.STRONG_TREND: "ALPHA",
    Regime.WEAK_TREND: "NEUTRAL",
    Regime.CHOP: "DEFENSIVE",
}


class MarketRegimeDetector:
    """
    Market Regime Detector.
    
    Uses composite scoring to detect regime:
    
    TrendScore = 
      + ADX(14) > 25        (+30)
      + EMA slope strong    (+25)
      + ATR expansion       (+25)
      - RSI chop zone       (-20)
    
    Regime mapping:
    ┌───────────────┬─────────────┬─────────────┐
    │ TrendScore    │ Regime      │ Mode        │
    ├───────────────┼─────────────┼─────────────┤
    │ > 60          │ STRONG_TREND│ ALPHA       │
    │ 40-60         │ WEAK_TREND  │ NEUTRAL     │
    │ < 40          │ CHOP        │ DEFENSIVE   │
    └───────────────┴─────────────┴─────────────┘
    """
    
    # Thresholds
    STRONG_TREND_THRESHOLD = 60
    WEAK_TREND_THRESHOLD = 40
    
    ADX_TREND_THRESHOLD = 25
    RSI_CHOP_LOW = 45
    RSI_CHOP_HIGH = 55
    
    def __init__(
        self,
        adx_period: int = 14,
        ema_fast: int = 50,
        ema_slow: int = 200,
        atr_period: int = 14,
        rsi_period: int = 14
    ):
        """
        Initialize Regime Detector.
        
        Args:
            adx_period: ADX calculation period
            ema_fast: Fast EMA for slope
            ema_slow: Slow EMA for trend
            atr_period: ATR period
            rsi_period: RSI period
        """
        self.adx_period = adx_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        
        self.last_regime: Optional[RegimeResult] = None
        
        logger.info(
            f"MarketRegimeDetector initialized: "
            f"ADX={adx_period}, EMA={ema_fast}/{ema_slow}"
        )
    
    def detect(self, df: pd.DataFrame) -> RegimeResult:
        """
        Detect current market regime.
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            
        Returns:
            RegimeResult with regime and scores
        """
        if len(df) < max(self.ema_slow, 50):
            # Not enough data
            return RegimeResult(
                regime=Regime.CHOP,
                trend_score=0,
                direction="NEUTRAL",
                adx_value=0,
                ema_slope=0,
                atr_expansion=0,
                rsi_chop=True,
                confidence=0
            )
        
        # Calculate indicators
        adx_value = self._calc_adx(df)
        ema_slope, direction = self._calc_ema_slope(df)
        atr_expansion = self._calc_atr_expansion(df)
        rsi_value = self._calc_rsi(df)
        rsi_chop = self.RSI_CHOP_LOW <= rsi_value <= self.RSI_CHOP_HIGH
        
        # Calculate trend score
        trend_score = 0
        
        # ADX component (+30)
        if adx_value > self.ADX_TREND_THRESHOLD:
            adx_bonus = min(30, (adx_value - 20) * 1.5)
            trend_score += adx_bonus
        
        # EMA slope component (+25)
        slope_score = min(25, abs(ema_slope) * 50)
        trend_score += slope_score
        
        # ATR expansion component (+25)
        if atr_expansion > 1.2:
            atr_bonus = min(25, (atr_expansion - 1) * 25)
            trend_score += atr_bonus
        
        # RSI chop penalty (-20)
        if rsi_chop:
            trend_score -= 20
        
        # Clamp score
        trend_score = max(0, min(100, trend_score))
        
        # Determine regime
        if trend_score >= self.STRONG_TREND_THRESHOLD:
            regime = Regime.STRONG_TREND
        elif trend_score >= self.WEAK_TREND_THRESHOLD:
            regime = Regime.WEAK_TREND
        else:
            regime = Regime.CHOP
        
        # Calculate confidence
        confidence = trend_score / 100
        
        result = RegimeResult(
            regime=regime,
            trend_score=round(trend_score, 1),
            direction=direction,
            adx_value=round(adx_value, 1),
            ema_slope=round(ema_slope, 4),
            atr_expansion=round(atr_expansion, 2),
            rsi_chop=rsi_chop,
            confidence=round(confidence, 2)
        )
        
        self.last_regime = result
        
        return result
    
    def _calc_adx(self, df: pd.DataFrame) -> float:
        """Calculate ADX indicator."""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            n = self.adx_period
            
            # True Range
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    abs(high[1:] - close[:-1]),
                    abs(low[1:] - close[:-1])
                )
            )
            
            # +DM, -DM
            plus_dm = np.where(
                (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                np.maximum(high[1:] - high[:-1], 0),
                0
            )
            minus_dm = np.where(
                (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                np.maximum(low[:-1] - low[1:], 0),
                0
            )
            
            # Smoothed values
            tr_smooth = self._ema(tr, n)
            plus_dm_smooth = self._ema(plus_dm, n)
            minus_dm_smooth = self._ema(minus_dm, n)
            
            # +DI, -DI
            plus_di = 100 * plus_dm_smooth / np.where(tr_smooth > 0, tr_smooth, 1)
            minus_di = 100 * minus_dm_smooth / np.where(tr_smooth > 0, tr_smooth, 1)
            
            # DX, ADX
            dx = 100 * abs(plus_di - minus_di) / np.where(
                (plus_di + minus_di) > 0, plus_di + minus_di, 1
            )
            adx = self._ema(dx, n)
            
            return float(adx[-1]) if len(adx) > 0 else 0
        except:
            return 0
    
    def _calc_ema_slope(self, df: pd.DataFrame) -> tuple[float, str]:
        """Calculate EMA slope and direction."""
        try:
            close = df['close'].values
            
            ema_fast = self._ema(close, self.ema_fast)
            ema_slow = self._ema(close, self.ema_slow)
            
            # Slope = percentage change over last 5 bars
            if len(ema_fast) >= 5:
                slope = (ema_fast[-1] - ema_fast[-5]) / ema_fast[-5]
            else:
                slope = 0
            
            # Direction
            if ema_fast[-1] > ema_slow[-1] and slope > 0.001:
                direction = "UP"
            elif ema_fast[-1] < ema_slow[-1] and slope < -0.001:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"
            
            return slope, direction
        except:
            return 0, "NEUTRAL"
    
    def _calc_atr_expansion(self, df: pd.DataFrame) -> float:
        """Calculate ATR expansion ratio (current / average)."""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # True Range
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    abs(high[1:] - close[:-1]),
                    abs(low[1:] - close[:-1])
                )
            )
            
            atr = self._ema(tr, self.atr_period)
            
            if len(atr) >= 20:
                current_atr = atr[-1]
                avg_atr = np.mean(atr[-20:])
                return current_atr / avg_atr if avg_atr > 0 else 1.0
            return 1.0
        except:
            return 1.0
    
    def _calc_rsi(self, df: pd.DataFrame) -> float:
        """Calculate RSI."""
        try:
            close = df['close'].values
            delta = np.diff(close)
            
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = self._ema(gain, self.rsi_period)
            avg_loss = self._ema(loss, self.rsi_period)
            
            rs = avg_gain[-1] / avg_loss[-1] if avg_loss[-1] > 0 else 100
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
        except:
            return 50
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        if len(data) < period:
            return data
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def get_recommended_mode(self) -> str:
        """Get recommended trading mode based on last regime."""
        if self.last_regime:
            return REGIME_MODE_MAP.get(self.last_regime.regime, "NEUTRAL")
        return "NEUTRAL"


# Singleton instance
_detector: Optional[MarketRegimeDetector] = None


def get_regime_detector() -> MarketRegimeDetector:
    """Get or create singleton MarketRegimeDetector."""
    global _detector
    if _detector is None:
        _detector = MarketRegimeDetector()
    return _detector


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MarketRegimeDetector Test")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 300
    
    # Trending data
    trend = np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    noise = np.random.randn(n) * 0.3
    
    df = pd.DataFrame({
        'open': 2600 + trend + noise,
        'high': 2600 + trend + noise + abs(np.random.randn(n)),
        'low': 2600 + trend + noise - abs(np.random.randn(n)),
        'close': 2600 + trend + noise,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    detector = MarketRegimeDetector()
    result = detector.detect(df)
    
    print(f"\n--- Regime Detection ---")
    print(f"Regime: {result.regime.value}")
    print(f"Trend Score: {result.trend_score}")
    print(f"Direction: {result.direction}")
    print(f"ADX: {result.adx_value}")
    print(f"EMA Slope: {result.ema_slope}")
    print(f"ATR Expansion: {result.atr_expansion}")
    print(f"RSI Chop: {result.rsi_chop}")
    print(f"Confidence: {result.confidence}")
    print(f"\nRecommended Mode: {detector.get_recommended_mode()}")
