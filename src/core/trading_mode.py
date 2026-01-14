# src/core/trading_mode.py
"""
Trading Mode Definitions - Production Grade
============================================

Runtime trading modes that control system behavior:
- ALPHA: Aggressive mode when performing well
- NEUTRAL: Balanced mode for normal conditions  
- DEFENSIVE: Conservative mode when underperforming

Mode is calculated every loop from:
- LiveScore
- Current Drawdown
- Market Volatility
"""

from enum import Enum


class TradingMode(str, Enum):
    """
    Trading mode enumeration.
    
    Modes:
    ┌─────────────┬──────────────────────────────────────┐
    │ Mode        │ Description                          │
    ├─────────────┼──────────────────────────────────────┤
    │ ALPHA       │ High performance, increase risk      │
    │ NEUTRAL     │ Normal operation, standard risk      │
    │ DEFENSIVE   │ Underperforming, reduce risk         │
    └─────────────┴──────────────────────────────────────┘
    """
    ALPHA = "ALPHA"
    NEUTRAL = "NEUTRAL"  
    DEFENSIVE = "DEFENSIVE"


# Mode Profiles - Central Truth for all mode effects
MODE_PROFILES = {
    "ALPHA": {
        "risk_mult": 1.3,        # Increase risk 30%
        "max_pyramid": 3,        # Allow full pyramid
        "allow_counter": False,  # No counter-trend trades
        "min_confidence": 0.5,   # Lower confidence threshold
        "description": "Aggressive - performing well"
    },
    "NEUTRAL": {
        "risk_mult": 1.0,        # Standard risk
        "max_pyramid": 2,        # Limited pyramid
        "allow_counter": False,  # No counter-trend
        "min_confidence": 0.6,   # Standard threshold
        "description": "Balanced - normal operation"
    },
    "DEFENSIVE": {
        "risk_mult": 0.6,        # Reduce risk 40%
        "max_pyramid": 0,        # No pyramid entries
        "allow_counter": False,  # No counter-trend
        "min_confidence": 0.7,   # Higher threshold required
        "description": "Conservative - protecting capital"
    }
}


def get_mode_profile(mode: TradingMode) -> dict:
    """Get profile for a trading mode."""
    return MODE_PROFILES.get(mode.value, MODE_PROFILES["NEUTRAL"])
