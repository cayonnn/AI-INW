"""Adapters package for unified data access."""

from .base_adapter import MarketDataAdapter
from .mt5_adapter import MT5Adapter
from .backtest_adapter import BacktestAdapter

__all__ = ['MarketDataAdapter', 'MT5Adapter', 'BacktestAdapter']
