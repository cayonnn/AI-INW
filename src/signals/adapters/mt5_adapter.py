"""
mt5_adapter.py
==============
MT5 Data Adapter for Live Trading

Wraps MT5Connector with MarketDataAdapter interface.
"""

import pandas as pd
from .base_adapter import MarketDataAdapter


class MT5Adapter(MarketDataAdapter):
    """MT5 data adapter for live trading."""
    
    def __init__(self, mt5_connector):
        """
        Args:
            mt5_connector: MT5Connector instance
        """
        self.mt5 = mt5_connector
    
    def get_candles(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        """Get OHLCV data from MT5."""
        rates = self.mt5.get_rates(symbol, timeframe, n)
        if rates is None:
            return pd.DataFrame()
        return pd.DataFrame(rates)
    
    def get_current_price(self, symbol: str) -> dict:
        """Get current price from MT5."""
        tick = self.mt5.get_tick(symbol)
        if tick is None:
            return {"bid": 0, "ask": 0, "spread": 0}
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.ask - tick.bid
        }
