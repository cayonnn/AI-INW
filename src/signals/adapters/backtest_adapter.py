"""
backtest_adapter.py
===================
Backtest Data Adapter

Provides historical data for backtesting with same interface as live.
"""

import pandas as pd
from .base_adapter import MarketDataAdapter


class BacktestAdapter(MarketDataAdapter):
    """Backtest data adapter for historical simulation."""
    
    def __init__(self, data: pd.DataFrame = None, csv_path: str = None):
        """
        Args:
            data: DataFrame with OHLCV data
            csv_path: Path to CSV file with historical data
        """
        if data is not None:
            self.df = data.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = pd.DataFrame()
        
        self.current_index = len(self.df) - 1
    
    def set_index(self, index: int):
        """Set current simulation index."""
        self.current_index = min(index, len(self.df) - 1)
    
    def get_candles(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        """Get historical candles up to current index."""
        # Filter by symbol if column exists
        if 'symbol' in self.df.columns:
            df = self.df[self.df['symbol'] == symbol]
        else:
            df = self.df
        
        # Get data up to current index
        end_idx = self.current_index + 1
        start_idx = max(0, end_idx - n)
        
        return df.iloc[start_idx:end_idx].copy()
    
    def get_current_price(self, symbol: str) -> dict:
        """Get simulated current price."""
        candles = self.get_candles(symbol, "H1", 1)
        if candles.empty:
            return {"bid": 0, "ask": 0, "spread": 0}
        
        close = candles.iloc[-1]['close']
        spread = 0.5  # Simulated spread
        
        return {
            "bid": close,
            "ask": close + spread,
            "spread": spread
        }
    
    def step(self):
        """Move to next candle."""
        self.current_index = min(self.current_index + 1, len(self.df) - 1)
    
    def reset(self):
        """Reset to beginning."""
        self.current_index = 0
