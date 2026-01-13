"""
AI Trading System - Utility Package
====================================
Common utilities for logging, metrics, validation, and configuration.
Includes Fund-Grade Core Loop components.
"""

from .logger import get_logger, setup_logging
from .config_loader import ConfigLoader, get_config
from .validators import validate_signal, validate_trade_params
from .analytics import AnalyticsDashboard
from .alpha_attribution import AlphaAttributionEngine
from .strategy_decay import StrategyDecayDetector
from .strategy_pool import SelfPruningStrategyPool
from .meta_portfolio import MetaPortfolioOptimizer
from .alpha_dashboard import ExplainableAlphaDashboard

__all__ = [
    # Core utils
    "get_logger",
    "setup_logging",
    "ConfigLoader",
    "get_config",
    "validate_signal",
    "validate_trade_params",
    "AnalyticsDashboard",
    # Fund-Grade Core Loop
    "AlphaAttributionEngine",
    "StrategyDecayDetector",
    "SelfPruningStrategyPool",
    "MetaPortfolioOptimizer",
    "ExplainableAlphaDashboard",
]

