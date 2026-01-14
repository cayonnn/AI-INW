# src/safety/__init__.py
"""
Safety Module
==============

Emergency controls and risk limits.
"""

from src.safety.kill_switch import (
    KillSwitch,
    KillReason,
    SafetyMetrics,
    get_kill_switch,
    close_all_positions,
    disable_trading,
    is_trading_disabled
)

__all__ = [
    "KillSwitch",
    "KillReason",
    "SafetyMetrics",
    "get_kill_switch",
    "close_all_positions",
    "disable_trading",
    "is_trading_disabled",
]
