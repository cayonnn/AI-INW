"""
AI Trading System - Market Regime Detector
============================================
Classifies market conditions: TRENDING, RANGING, VOLATILE, QUIET.
"""

from typing import Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class MarketRegime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET = "QUIET"


class RegimeDetector:
    """Detects market regime using ADX, volatility, and price action."""
    
    def __init__(self, lookback: int = 50, trend_threshold: float = 0.6, vol_high: int = 75, vol_low: int = 25):
        config = get_config()
        self.lookback = lookback or config.features.regime_lookback
        self.trend_threshold = trend_threshold or config.features.trend_threshold
        self.vol_threshold_high = vol_high or config.features.volatility_threshold_high
        self.vol_threshold_low = vol_low or config.features.volatility_threshold_low
    
    def detect_regime(self, df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect current market regime with confidence."""
        data = df.tail(self.lookback).copy()
        
        # ADX-based trend strength
        adx = self._calculate_adx(data)
        is_trending = adx > 25
        
        # Volatility percentile
        vol_pct = self._calculate_volatility_percentile(data)
        is_high_vol = vol_pct > self.vol_threshold_high
        is_low_vol = vol_pct < self.vol_threshold_low
        
        # EMA slope for trend direction
        ema_slope = self._calculate_ema_slope(data)
        strong_trend = abs(ema_slope) > 0.001
        
        # Classify regime
        if is_high_vol:
            regime = MarketRegime.VOLATILE
            confidence = min(vol_pct / 100, 0.95)
        elif is_trending and strong_trend:
            regime = MarketRegime.TRENDING
            confidence = min(adx / 50, 0.95)
        elif is_low_vol:
            regime = MarketRegime.QUIET
            confidence = 1 - (vol_pct / 100)
        else:
            regime = MarketRegime.RANGING
            confidence = 1 - (adx / 50)
        
        return regime, confidence
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        tr = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)
        
        plus_dm = np.where((df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
                          np.maximum(df["high"] - df["high"].shift(1), 0), 0)
        minus_dm = np.where((df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
                           np.maximum(df["low"].shift(1) - df["low"], 0), 0)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return float(adx.iloc[-1]) if len(adx) > 0 else 0
    
    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        returns = df["close"].pct_change().dropna()
        if len(returns) < 10:
            return 50.0
        current_vol = returns.tail(10).std()
        percentile = (returns.rolling(10).std() < current_vol).mean() * 100
        return float(percentile) if not np.isnan(percentile) else 50.0
    
    def _calculate_ema_slope(self, df: pd.DataFrame, period: int = 20) -> float:
        ema = df["close"].ewm(span=period, adjust=False).mean()
        if len(ema) < 5:
            return 0.0
        slope = (ema.iloc[-1] - ema.iloc[-5]) / ema.iloc[-5]
        return float(slope)
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime classification features to DataFrame."""
        result = df.copy()
        
        regimes = []
        confidences = []
        
        for i in range(len(df)):
            if i < self.lookback:
                regimes.append(MarketRegime.RANGING.value)
                confidences.append(0.5)
            else:
                regime, conf = self.detect_regime(df.iloc[:i+1])
                regimes.append(regime.value)
                confidences.append(conf)
        
        result["regime"] = regimes
        result["regime_confidence"] = confidences
        
        # One-hot encode
        for regime in MarketRegime:
            result[f"regime_{regime.value.lower()}"] = (result["regime"] == regime.value).astype(int)
        
        logger.debug(f"Added regime features")
        return result
