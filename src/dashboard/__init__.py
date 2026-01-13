# src/dashboard/__init__.py
"""
Trader Dashboard Package
========================

Fund-Grade dashboard components for explainability and monitoring.
"""

from .trader_dashboard import TraderDashboard
from .effective_risk import EffectiveRiskStack, RiskStackData

__all__ = [
    "TraderDashboard",
    "EffectiveRiskStack",
    "RiskStackData",
]

