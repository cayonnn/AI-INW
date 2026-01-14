# src/shadow/__init__.py
"""
Shadow Mode Module - Competition Grade
======================================

Multi-mode simulation for performance comparison.

Components:
- ShadowModeSimulator: Simulates multiple trading modes
- ShadowAccount: Virtual account for each mode
- ShadowTrade: Individual shadow trade

Usage:
    from src.shadow import get_shadow_simulator
    simulator = get_shadow_simulator()
    simulator.simulate_signal("BUY", 2650, 2640, 2670)
"""

from src.shadow.shadow_mode import (
    ShadowModeSimulator,
    ShadowMode,
    ShadowTrade,
    ShadowAccountState,
    get_shadow_simulator,
    MODE_RISK_MULT
)


__all__ = [
    "ShadowModeSimulator",
    "ShadowMode",
    "ShadowTrade",
    "ShadowAccountState",
    "get_shadow_simulator",
    "MODE_RISK_MULT",
]
