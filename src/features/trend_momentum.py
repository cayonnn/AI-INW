"""
AI Trading System - Trend & Momentum Features
===============================================
EMA, RSI, MACD, and other trend/momentum indicators.

Usage:
    from src.features.trend_momentum import TrendMomentumFeatures
    
    tm = TrendMomentumFeatures()
    df = tm.add_all_features(ohlcv_df)
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config_loader import get_config


logger = get_logger(__name__)


class TrendMomentumFeatures:
    """
    Trend and momentum feature engineering.
    
    Extracts features based on:
    - Moving averages (EMA, SMA)
    - RSI and derivatives
    - MACD and derivatives
    - ADX trend strength
    - Momentum oscillators
    """
    
    def __init__(
        self,
        ema_periods: Optional[List[int]] = None,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        adx_period: int = 14,
    ):
        """
        Initialize trend/momentum feature extractor.
        
        Args:
            ema_periods: List of EMA periods
            rsi_period: RSI calculation period
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            adx_period: ADX calculation period
        """
        config = get_config()
        
        self.ema_periods = ema_periods or config.features.ema_periods
        self.rsi_period = rsi_period or config.features.rsi_period
        self.macd_fast = macd_fast or config.features.macd["fast_period"]
        self.macd_slow = macd_slow or config.features.macd["slow_period"]
        self.macd_signal = macd_signal or config.features.macd["signal_period"]
        self.adx_period = adx_period
    
    # ─────────────────────────────────────────────────────────────────────────
    # MOVING AVERAGES
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_ema(
        self,
        df: pd.DataFrame,
        periods: Optional[List[int]] = None,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Add Exponential Moving Averages.
        
        Args:
            df: OHLCV DataFrame
            periods: List of EMA periods
            column: Column to calculate EMA on
        
        Returns:
            DataFrame with EMA columns
        """
        result = df.copy()
        periods = periods or self.ema_periods
        
        for period in periods:
            ema = result[column].ewm(span=period, adjust=False).mean()
            result[f"ema_{period}"] = ema
            
            # Price distance from EMA (normalized)
            result[f"price_dist_ema_{period}"] = (
                (result[column] - ema) / ema
            )
            
            # EMA slope (rate of change)
            result[f"ema_{period}_slope"] = ema.diff(3) / ema * 100
        
        # EMA relationships
        if len(periods) >= 2:
            # Fast EMA above/below slow EMA
            fastest = min(periods)
            slowest = max(periods)
            result["ema_trend"] = np.where(
                result[f"ema_{fastest}"] > result[f"ema_{slowest}"],
                1,
                -1
            )
            
            # EMA spread (distance between fast and slow)
            result["ema_spread"] = (
                (result[f"ema_{fastest}"] - result[f"ema_{slowest}"]) /
                result[f"ema_{slowest}"]
            )
        
        # EMA alignment (all in order = strong trend)
        if len(periods) >= 3:
            sorted_periods = sorted(periods)
            bullish_aligned = pd.Series(True, index=result.index)
            bearish_aligned = pd.Series(True, index=result.index)
            
            for i in range(len(sorted_periods) - 1):
                faster = sorted_periods[i]
                slower = sorted_periods[i + 1]
                bullish_aligned &= (result[f"ema_{faster}"] > result[f"ema_{slower}"])
                bearish_aligned &= (result[f"ema_{faster}"] < result[f"ema_{slower}"])
            
            result["ema_bullish_aligned"] = bullish_aligned.astype(int)
            result["ema_bearish_aligned"] = bearish_aligned.astype(int)
        
        return result
    
    def add_sma(
        self,
        df: pd.DataFrame,
        periods: List[int],
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Add Simple Moving Averages.
        
        Args:
            df: OHLCV DataFrame
            periods: List of SMA periods
            column: Column to calculate SMA on
        
        Returns:
            DataFrame with SMA columns
        """
        result = df.copy()
        
        for period in periods:
            result[f"sma_{period}"] = result[column].rolling(period).mean()
            result[f"price_dist_sma_{period}"] = (
                (result[column] - result[f"sma_{period}"]) / result[f"sma_{period}"]
            )
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # RSI
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_rsi(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Add RSI and related features.
        
        Args:
            df: OHLCV DataFrame
            period: RSI period
            column: Column to calculate RSI on
        
        Returns:
            DataFrame with RSI features
        """
        result = df.copy()
        period = period or self.rsi_period
        
        # Calculate RSI
        delta = result[column].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        result["rsi"] = rsi
        
        # RSI zones
        result["rsi_oversold"] = (rsi < 30).astype(int)
        result["rsi_overbought"] = (rsi > 70).astype(int)
        result["rsi_neutral"] = ((rsi >= 30) & (rsi <= 70)).astype(int)
        
        # RSI distance from 50 (normalized)
        result["rsi_deviation"] = (rsi - 50) / 50
        
        # RSI slope/momentum
        result["rsi_slope"] = rsi.diff(3)
        result["rsi_acceleration"] = result["rsi_slope"].diff()
        
        # RSI divergence setup (price makes new high/low, RSI doesn't)
        price_new_high = result[column] == result[column].rolling(20).max()
        price_new_low = result[column] == result[column].rolling(20).min()
        rsi_new_high = rsi == rsi.rolling(20).max()
        rsi_new_low = rsi == rsi.rolling(20).min()
        
        result["rsi_bearish_div"] = (price_new_high & ~rsi_new_high).astype(int)
        result["rsi_bullish_div"] = (price_new_low & ~rsi_new_low).astype(int)
        
        # Stochastic RSI
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        stoch_rsi = np.where(
            (rsi_max - rsi_min) > 0,
            (rsi - rsi_min) / (rsi_max - rsi_min),
            0.5
        )
        result["stoch_rsi"] = stoch_rsi
        result["stoch_rsi_signal"] = pd.Series(stoch_rsi, index=result.index).rolling(3).mean()
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # MACD
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_macd(
        self,
        df: pd.DataFrame,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Add MACD and related features.
        
        Args:
            df: OHLCV DataFrame
            fast: MACD fast period
            slow: MACD slow period
            signal: MACD signal period
            column: Column to calculate MACD on
        
        Returns:
            DataFrame with MACD features
        """
        result = df.copy()
        fast = fast or self.macd_fast
        slow = slow or self.macd_slow
        signal = signal or self.macd_signal
        
        # Calculate MACD
        ema_fast = result[column].ewm(span=fast, adjust=False).mean()
        ema_slow = result[column].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_histogram"] = histogram
        
        # Normalized MACD (relative to price)
        result["macd_normalized"] = macd_line / result[column] * 100
        
        # MACD direction
        result["macd_above_signal"] = (macd_line > signal_line).astype(int)
        result["macd_above_zero"] = (macd_line > 0).astype(int)
        
        # MACD crossovers
        result["macd_bullish_cross"] = (
            (macd_line > signal_line) &
            (macd_line.shift(1) <= signal_line.shift(1))
        ).astype(int)
        result["macd_bearish_cross"] = (
            (macd_line < signal_line) &
            (macd_line.shift(1) >= signal_line.shift(1))
        ).astype(int)
        
        # Zero line crossovers
        result["macd_cross_zero_up"] = (
            (macd_line > 0) & (macd_line.shift(1) <= 0)
        ).astype(int)
        result["macd_cross_zero_down"] = (
            (macd_line < 0) & (macd_line.shift(1) >= 0)
        ).astype(int)
        
        # Histogram momentum
        result["macd_hist_slope"] = histogram.diff()
        result["macd_hist_increasing"] = (result["macd_hist_slope"] > 0).astype(int)
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # ADX (TREND STRENGTH)
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_adx(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Add ADX trend strength indicator.
        
        Args:
            df: OHLCV DataFrame
            period: ADX period
        
        Returns:
            DataFrame with ADX features
        """
        result = df.copy()
        period = period or self.adx_period
        
        # True Range
        tr = pd.concat([
            result["high"] - result["low"],
            abs(result["high"] - result["close"].shift(1)),
            abs(result["low"] - result["close"].shift(1))
        ], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = np.where(
            (result["high"] - result["high"].shift(1)) > 
            (result["low"].shift(1) - result["low"]),
            np.maximum(result["high"] - result["high"].shift(1), 0),
            0
        )
        minus_dm = np.where(
            (result["low"].shift(1) - result["low"]) > 
            (result["high"] - result["high"].shift(1)),
            np.maximum(result["low"].shift(1) - result["low"], 0),
            0
        )
        
        # Smoothed values
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=result.index).ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=result.index).ewm(alpha=1/period, adjust=False).mean() / atr
        
        # ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        result["plus_di"] = plus_di
        result["minus_di"] = minus_di
        result["adx"] = adx
        
        # ADX interpretation
        result["trend_strong"] = (adx > 25).astype(int)
        result["trend_very_strong"] = (adx > 50).astype(int)
        result["trend_weak"] = (adx < 20).astype(int)
        
        # DI crossovers
        result["di_bullish"] = (plus_di > minus_di).astype(int)
        result["di_bearish"] = (minus_di > plus_di).astype(int)
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # MOMENTUM
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_momentum(
        self,
        df: pd.DataFrame,
        periods: List[int] = [5, 10, 20],
    ) -> pd.DataFrame:
        """
        Add momentum indicators.
        
        Args:
            df: OHLCV DataFrame
            periods: Momentum calculation periods
        
        Returns:
            DataFrame with momentum features
        """
        result = df.copy()
        
        for period in periods:
            # Rate of Change (ROC)
            result[f"roc_{period}"] = (
                (result["close"] - result["close"].shift(period)) /
                result["close"].shift(period) * 100
            )
            
            # Momentum (simple difference)
            result[f"momentum_{period}"] = result["close"].diff(period)
        
        # Williams %R
        highest_high = result["high"].rolling(14).max()
        lowest_low = result["low"].rolling(14).min()
        result["williams_r"] = np.where(
            (highest_high - lowest_low) > 0,
            (highest_high - result["close"]) / (highest_high - lowest_low) * -100,
            -50
        )
        
        # CCI (Commodity Channel Index)
        typical_price = (result["high"] + result["low"] + result["close"]) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        result["cci"] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # COMBINED FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_all_features(
        self,
        df: pd.DataFrame,
        include_ema: bool = True,
        include_rsi: bool = True,
        include_macd: bool = True,
        include_adx: bool = True,
        include_momentum: bool = True,
    ) -> pd.DataFrame:
        """
        Add all trend and momentum features.
        
        Args:
            df: OHLCV DataFrame
            include_ema: Include EMA features
            include_rsi: Include RSI features
            include_macd: Include MACD features
            include_adx: Include ADX features
            include_momentum: Include momentum features
        
        Returns:
            DataFrame with all features
        """
        result = df.copy()
        
        if include_ema:
            result = self.add_ema(result)
        
        if include_rsi:
            result = self.add_rsi(result)
        
        if include_macd:
            result = self.add_macd(result)
        
        if include_adx:
            result = self.add_adx(result)
        
        if include_momentum:
            result = self.add_momentum(result)
        
        logger.debug(f"Added trend/momentum features: {len(result.columns) - len(df.columns)} new columns")
        return result
