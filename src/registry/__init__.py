# src/registry/__init__.py
"""
Registry Module - Profile Management
=====================================

Manages profile states and promotion governance.
"""

from src.registry.promotion_engine import (
    PromotionEngine,
    ProfileState,
    ProfileMetrics,
    get_promotion_engine
)

__all__ = [
    "PromotionEngine",
    "ProfileState",
    "ProfileMetrics",
    "get_promotion_engine",
]
