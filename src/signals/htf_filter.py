"""
htf_filter.py
=============
Higher Timeframe Trend Filter (Market Structure Gate)

Uses H4/D1 to confirm overall trend direction.
Prevents trading against the main trend.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class HTFTrendResult:
    """HTF trend analysis result."""
    trend: str          # BULL / BEAR / FLAT
    ema50: float
    ema200: float
    strength: float     # 0.0 - 1.0
    reason: str


class HTFTrendFilter:
    """
    Higher Timeframe Trend Filter.
    
    Uses EMA50 vs EMA200 on H4 to determine market structure.
    Prevents signals against the main trend.
    """
    
    def __init__(self, data_adapter, timeframe: str = "H4"):
        """
        Args:
            data_adapter: MarketDataAdapter instance
            timeframe: HTF timeframe (H4 or D1)
        """
        self.data = data_adapter
        self.timeframe = timeframe
    
    def get_trend(self, symbol: str) -> HTFTrendResult:
        """
        Analyze HTF trend for symbol.
        
        Returns:
            HTFTrendResult with trend direction and strength
        """
        candles = self.data.get_candles(symbol, self.timeframe, 250)
        
        if candles.empty or len(candles) < 200:
            return HTFTrendResult(
                trend="FLAT",
                ema50=0,
                ema200=0,
                strength=0,
                reason="Insufficient data for HTF analysis"
            )
        
        close = candles['close']
        
        # Calculate EMAs
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        
        # Calculate strength (spread as % of price)
        spread = abs(ema50 - ema200)
        strength = min(1.0, (spread / close.iloc[-1]) * 50)
        
        # Determine trend
        if ema50 > ema200 * 1.001:  # 0.1% buffer
            trend = "BULL"
            reason = f"EMA50 ({ema50:.2f}) above EMA200 ({ema200:.2f})"
        elif ema50 < ema200 * 0.999:
            trend = "BEAR"
            reason = f"EMA50 ({ema50:.2f}) below EMA200 ({ema200:.2f})"
        else:
            trend = "FLAT"
            reason = f"EMAs roughly equal (50={ema50:.2f}, 200={ema200:.2f})"
        
        return HTFTrendResult(
            trend=trend,
            ema50=round(ema50, 2),
            ema200=round(ema200, 2),
            strength=round(strength, 2),
            reason=reason
        )
    
    def allows_direction(self, symbol: str, direction: str) -> tuple[bool, str]:
        """
        Check if direction is allowed by HTF trend.
        
        Args:
            symbol: Trading symbol
            direction: "BUY" or "SELL"
            
        Returns:
            (allowed: bool, reason: str)
        """
        result = self.get_trend(symbol)
        
        # FLAT allows both directions
        if result.trend == "FLAT":
            return True, "HTF FLAT - both directions allowed"
        
        # BUY only in BULL
        if direction == "BUY" and result.trend == "BULL":
            return True, f"BUY aligned with HTF BULL trend"
        
        # SELL only in BEAR
        if direction == "SELL" and result.trend == "BEAR":
            return True, f"SELL aligned with HTF BEAR trend"
        
        # Otherwise blocked
        return False, f"Direction {direction} blocked by HTF {result.trend} trend"
