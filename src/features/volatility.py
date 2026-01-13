"""
AI Trading System - Volatility Features
=========================================
ATR, Bollinger Bands, and volatility metrics.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class VolatilityFeatures:
    """Volatility feature engineering: ATR, Bollinger Bands, historical volatility."""
    
    def __init__(self, atr_period: int = 14, bb_period: int = 20, bb_std: float = 2.0, vol_lookback: int = 100):
        config = get_config()
        self.atr_period = atr_period or config.features.atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.vol_lookback = vol_lookback
    
    def add_atr(self, df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Add ATR and related features."""
        result = df.copy()
        periods = periods or [self.atr_period]
        
        tr = pd.concat([
            result["high"] - result["low"],
            abs(result["high"] - result["close"].shift(1)),
            abs(result["low"] - result["close"].shift(1))
        ], axis=1).max(axis=1)
        result["true_range"] = tr
        
        for period in periods:
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            result[f"atr_{period}"] = atr
            result[f"atr_{period}_pct"] = atr / result["close"] * 100
        
        primary_atr = result[f"atr_{periods[0]}"]
        result["atr_slope"] = primary_atr.diff(5) / primary_atr * 100
        result["atr_percentile"] = primary_atr.rolling(self.vol_lookback).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() * 100 if len(x) > 1 else 50
        )
        result["range_to_atr"] = (result["high"] - result["low"]) / primary_atr
        
        return result
    
    def add_bollinger_bands(self, df: pd.DataFrame, period: Optional[int] = None, std_dev: Optional[float] = None) -> pd.DataFrame:
        """Add Bollinger Bands and related features."""
        result = df.copy()
        period = period or self.bb_period
        std_dev = std_dev or self.bb_std
        
        sma = result["close"].rolling(period).mean()
        std = result["close"].rolling(period).std()
        
        result["bb_middle"] = sma
        result["bb_upper"] = sma + (std * std_dev)
        result["bb_lower"] = sma - (std * std_dev)
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / sma * 100
        
        bb_range = result["bb_upper"] - result["bb_lower"]
        result["bb_percent_b"] = np.where(bb_range > 0, (result["close"] - result["bb_lower"]) / bb_range, 0.5)
        result["bb_squeeze"] = (result["bb_width"] < result["bb_width"].rolling(50).quantile(0.2)).astype(int)
        
        return result
    
    def add_historical_volatility(self, df: pd.DataFrame, periods: List[int] = [10, 20]) -> pd.DataFrame:
        """Add historical volatility metrics."""
        result = df.copy()
        log_returns = np.log(result["close"] / result["close"].shift(1))
        result["log_return"] = log_returns
        
        for period in periods:
            result[f"volatility_{period}"] = log_returns.rolling(period).std() * np.sqrt(252) * 100
        
        result["vol_percentile"] = result["volatility_20"].rolling(self.vol_lookback).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() * 100 if len(x) > 1 else 50
        ) if "volatility_20" in result.columns else 50
        
        return result
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all volatility features."""
        result = self.add_atr(df)
        result = self.add_bollinger_bands(result)
        result = self.add_historical_volatility(result)
        logger.debug(f"Added volatility features: {len(result.columns) - len(df.columns)} new columns")
        return result
