"""
AI Trading System - Price Action Features
===========================================
Candlestick patterns, swing points, and support/resistance levels.

Usage:
    from src.features.price_action import PriceActionFeatures
    
    pa = PriceActionFeatures()
    df = pa.add_all_features(ohlcv_df)
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


class PriceActionFeatures:
    """
    Price action feature engineering.
    
    Extracts features based on:
    - Candlestick properties (body size, wick ratios)
    - Swing points (local highs/lows)
    - Support/Resistance levels
    - Price patterns
    """
    
    def __init__(
        self,
        swing_lookback: int = 20,
        sr_sensitivity: float = 0.002,
        sr_min_touches: int = 2,
    ):
        """
        Initialize price action feature extractor.
        
        Args:
            swing_lookback: Lookback for swing point detection
            sr_sensitivity: Price clustering sensitivity for S/R
            sr_min_touches: Minimum touches for valid S/R level
        """
        self.swing_lookback = swing_lookback
        self.sr_sensitivity = sr_sensitivity
        self.sr_min_touches = sr_min_touches
    
    # ─────────────────────────────────────────────────────────────────────────
    # CANDLESTICK FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add candlestick-based features.
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            DataFrame with candle features added
        """
        result = df.copy()
        
        # Basic candle properties
        result["body_size"] = abs(result["close"] - result["open"])
        result["candle_range"] = result["high"] - result["low"]
        result["upper_wick"] = result["high"] - result[["open", "close"]].max(axis=1)
        result["lower_wick"] = result[["open", "close"]].min(axis=1) - result["low"]
        
        # Normalized versions (0-1 scale)
        result["body_ratio"] = np.where(
            result["candle_range"] > 0,
            result["body_size"] / result["candle_range"],
            0.5
        )
        result["upper_wick_ratio"] = np.where(
            result["candle_range"] > 0,
            result["upper_wick"] / result["candle_range"],
            0
        )
        result["lower_wick_ratio"] = np.where(
            result["candle_range"] > 0,
            result["lower_wick"] / result["candle_range"],
            0
        )
        
        # Candle direction
        result["is_bullish"] = (result["close"] > result["open"]).astype(int)
        result["is_bearish"] = (result["close"] < result["open"]).astype(int)
        
        # Gap detection
        result["gap_up"] = (result["low"] > result["high"].shift(1)).astype(int)
        result["gap_down"] = (result["high"] < result["low"].shift(1)).astype(int)
        result["gap_size"] = np.where(
            result["gap_up"] == 1,
            result["low"] - result["high"].shift(1),
            np.where(
                result["gap_down"] == 1,
                result["low"].shift(1) - result["high"],
                0
            )
        )
        
        # Relative candle size (vs recent average)
        avg_range = result["candle_range"].rolling(20).mean()
        result["relative_size"] = np.where(
            avg_range > 0,
            result["candle_range"] / avg_range,
            1
        )
        
        return result
    
    def add_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect common candlestick patterns.
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            DataFrame with pattern signals
        """
        result = df.copy()
        
        # Need candle features first
        if "body_ratio" not in result.columns:
            result = self.add_candle_features(result)
        
        # Doji (small body, 10-30% of range)
        result["is_doji"] = (result["body_ratio"] < 0.15).astype(int)
        
        # Hammer/Hanging Man (small upper wick, long lower wick)
        result["is_hammer"] = (
            (result["lower_wick_ratio"] > 0.6) &
            (result["upper_wick_ratio"] < 0.1) &
            (result["body_ratio"] > 0.1) &
            (result["body_ratio"] < 0.35)
        ).astype(int)
        
        # Inverted Hammer / Shooting Star
        result["is_shooting_star"] = (
            (result["upper_wick_ratio"] > 0.6) &
            (result["lower_wick_ratio"] < 0.1) &
            (result["body_ratio"] > 0.1) &
            (result["body_ratio"] < 0.35)
        ).astype(int)
        
        # Marubozu (strong body, minimal wicks)
        result["is_marubozu"] = (
            (result["body_ratio"] > 0.9) &
            (result["upper_wick_ratio"] < 0.05) &
            (result["lower_wick_ratio"] < 0.05)
        ).astype(int)
        
        # Engulfing patterns (2-candle)
        prev_bullish = result["is_bullish"].shift(1).fillna(0).astype(bool)
        prev_bearish = result["is_bearish"].shift(1).fillna(0).astype(bool)
        prev_open = result["open"].shift(1)
        prev_close = result["close"].shift(1)
        
        result["bullish_engulfing"] = (
            prev_bearish &
            result["is_bullish"].astype(bool) &
            (result["open"] < prev_close) &
            (result["close"] > prev_open)
        ).astype(int)
        
        result["bearish_engulfing"] = (
            prev_bullish &
            result["is_bearish"].astype(bool) &
            (result["open"] > prev_close) &
            (result["close"] < prev_open)
        ).astype(int)
        
        # Inside bar
        result["is_inside_bar"] = (
            (result["high"] < result["high"].shift(1)) &
            (result["low"] > result["low"].shift(1))
        ).astype(int)
        
        # Outside bar (engulfing range)
        result["is_outside_bar"] = (
            (result["high"] > result["high"].shift(1)) &
            (result["low"] < result["low"].shift(1))
        ).astype(int)
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # SWING POINTS
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect_swing_points(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Detect swing high and swing low points.
        
        Args:
            df: OHLCV DataFrame
            lookback: Number of bars to look on each side
        
        Returns:
            DataFrame with swing point columns
        """
        result = df.copy()
        lookback = lookback or self.swing_lookback
        
        # Swing High: high is highest in lookback window on both sides
        result["swing_high"] = np.nan
        result["swing_low"] = np.nan
        
        for i in range(lookback, len(result) - lookback):
            # Check swing high
            high_val = result.iloc[i]["high"]
            is_swing_high = True
            for j in range(1, lookback + 1):
                if result.iloc[i - j]["high"] >= high_val or result.iloc[i + j]["high"] >= high_val:
                    is_swing_high = False
                    break
            if is_swing_high:
                result.iloc[i, result.columns.get_loc("swing_high")] = high_val
            
            # Check swing low
            low_val = result.iloc[i]["low"]
            is_swing_low = True
            for j in range(1, lookback + 1):
                if result.iloc[i - j]["low"] <= low_val or result.iloc[i + j]["low"] <= low_val:
                    is_swing_low = False
                    break
            if is_swing_low:
                result.iloc[i, result.columns.get_loc("swing_low")] = low_val
        
        # Binary indicators
        result["is_swing_high"] = result["swing_high"].notna().astype(int)
        result["is_swing_low"] = result["swing_low"].notna().astype(int)
        
        # Fill forward for distance calculation
        result["last_swing_high"] = result["swing_high"].ffill()
        result["last_swing_low"] = result["swing_low"].ffill()
        
        # Distance from last swing points
        result["dist_from_swing_high"] = (result["close"] - result["last_swing_high"]) / result["close"]
        result["dist_from_swing_low"] = (result["close"] - result["last_swing_low"]) / result["close"]
        
        # Bars since last swing
        result["bars_since_swing_high"] = (
            result.groupby((result["is_swing_high"] == 1).cumsum()).cumcount()
        )
        result["bars_since_swing_low"] = (
            result.groupby((result["is_swing_low"] == 1).cumsum()).cumcount()
        )
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUPPORT & RESISTANCE
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect_sr_levels(
        self,
        df: pd.DataFrame,
        lookback: int = 100,
    ) -> Tuple[List[float], List[float]]:
        """
        Detect support and resistance levels using price clustering.
        
        Args:
            df: OHLCV DataFrame
            lookback: Number of bars to analyze
        
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        data = df.tail(lookback).copy()
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(data) - 5):
            if data.iloc[i]["high"] == data.iloc[i-5:i+6]["high"].max():
                swing_highs.append(data.iloc[i]["high"])
            if data.iloc[i]["low"] == data.iloc[i-5:i+6]["low"].min():
                swing_lows.append(data.iloc[i]["low"])
        
        # Cluster levels
        resistance = self._cluster_levels(swing_highs)
        support = self._cluster_levels(swing_lows)
        
        return support, resistance
    
    def _cluster_levels(self, prices: List[float]) -> List[float]:
        """Cluster nearby price levels into zones."""
        if not prices:
            return []
        
        prices = sorted(prices)
        clusters = []
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            if abs(price - current_cluster[-1]) / current_cluster[-1] < self.sr_sensitivity:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= self.sr_min_touches:
                    clusters.append(np.mean(current_cluster))
                current_cluster = [price]
        
        if len(current_cluster) >= self.sr_min_touches:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def add_sr_features(
        self,
        df: pd.DataFrame,
        lookback: int = 100,
    ) -> pd.DataFrame:
        """
        Add support/resistance distance features.
        
        Args:
            df: OHLCV DataFrame
            lookback: Bars to analyze for S/R
        
        Returns:
            DataFrame with S/R features
        """
        result = df.copy()
        
        support, resistance = self.detect_sr_levels(df, lookback)
        current_price = result["close"].iloc[-1]
        
        # Distance to nearest support/resistance
        if support:
            nearest_support = min(support, key=lambda x: abs(x - current_price) if x < current_price else float('inf'))
            result["dist_to_support"] = (result["close"] - nearest_support) / result["close"]
        else:
            result["dist_to_support"] = 0.0
        
        if resistance:
            nearest_resistance = min(resistance, key=lambda x: abs(x - current_price) if x > current_price else float('inf'))
            result["dist_to_resistance"] = (nearest_resistance - result["close"]) / result["close"]
        else:
            result["dist_to_resistance"] = 0.0
        
        # Position within range (0 = at support, 1 = at resistance)
        if support and resistance:
            price_range = max(resistance) - min(support)
            if price_range > 0:
                result["price_position"] = (result["close"] - min(support)) / price_range
            else:
                result["price_position"] = 0.5
        else:
            result["price_position"] = 0.5
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # HIGHER TIMEFRAME CONTEXT
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_daily_context(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add daily high/low context features.
        
        Args:
            df: Intraday OHLCV DataFrame with DatetimeIndex
        
        Returns:
            DataFrame with daily context features
        """
        result = df.copy()
        
        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping daily context")
            return result
        
        # Get daily high/low
        daily_high = result.groupby(result.index.date)["high"].transform("max")
        daily_low = result.groupby(result.index.date)["low"].transform("min")
        daily_open = result.groupby(result.index.date)["open"].transform("first")
        
        result["daily_high"] = daily_high
        result["daily_low"] = daily_low
        result["daily_open"] = daily_open
        result["daily_range"] = daily_high - daily_low
        
        # Position within daily range
        result["position_in_daily_range"] = np.where(
            result["daily_range"] > 0,
            (result["close"] - daily_low) / result["daily_range"],
            0.5
        )
        
        # Distance from daily high/low
        result["dist_from_daily_high"] = (daily_high - result["close"]) / result["close"]
        result["dist_from_daily_low"] = (result["close"] - daily_low) / result["close"]
        
        # Previous day levels
        result["prev_day_high"] = result["daily_high"].shift(1)
        result["prev_day_low"] = result["daily_low"].shift(1)
        result["prev_day_close"] = result["close"].shift(1)
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # COMBINED FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_all_features(
        self,
        df: pd.DataFrame,
        include_patterns: bool = True,
        include_swings: bool = True,
        include_sr: bool = True,
        include_daily: bool = True,
    ) -> pd.DataFrame:
        """
        Add all price action features.
        
        Args:
            df: OHLCV DataFrame
            include_patterns: Include candlestick patterns
            include_swings: Include swing points
            include_sr: Include S/R features
            include_daily: Include daily context
        
        Returns:
            DataFrame with all features added
        """
        result = self.add_candle_features(df)
        
        if include_patterns:
            result = self.add_candle_patterns(result)
        
        if include_swings:
            result = self.detect_swing_points(result)
        
        if include_sr:
            result = self.add_sr_features(result)
        
        if include_daily and isinstance(result.index, pd.DatetimeIndex):
            result = self.add_daily_context(result)
        
        logger.debug(f"Added price action features: {len(result.columns) - len(df.columns)} new columns")
        return result
