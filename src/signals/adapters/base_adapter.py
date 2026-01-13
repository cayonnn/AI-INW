"""
base_adapter.py
================
Abstract Base for Market Data Adapters

Supports both Live (MT5) and Backtest modes transparently.
SignalEngine doesn't need to know the data source.
"""

from abc import ABC, abstractmethod
import pandas as pd


class MarketDataAdapter(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        """
        Get OHLCV candle data.
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            timeframe: Timeframe (e.g., "H1", "H4", "D1")
            n: Number of candles
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, time
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> dict:
        """
        Get current bid/ask prices.
        
        Returns:
            {"bid": float, "ask": float, "spread": float}
        """
        raise NotImplementedError
