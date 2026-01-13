# src/config/__init__.py
"""
Configuration Module
====================
Trading profiles and system configuration.
"""

from .trading_profiles import (
    TradingMode,
    TradingProfile,
    RiskConfig,
    EntryConfig,
    SLTPConfig,
    TrailingConfig,
    ModelConfig,
    get_profile,
    get_profile_by_name,
    get_active_profile,
    set_active_profile,
    ACTIVE_PROFILE,
    CONSERVATIVE_PROFILE,
    BALANCED_PROFILE,
    AGGRESSIVE_PROFILE,
)

from .profile_fingerprint import (
    profile_checksum,
    validate_checksum,
)

__all__ = [
    "TradingMode",
    "TradingProfile",
    "RiskConfig",
    "EntryConfig",
    "SLTPConfig",
    "TrailingConfig",
    "ModelConfig",
    "get_profile",
    "get_profile_by_name",
    "get_active_profile",
    "set_active_profile",
    "ACTIVE_PROFILE",
    "CONSERVATIVE_PROFILE",
    "BALANCED_PROFILE",
    "AGGRESSIVE_PROFILE",
    # Fingerprint
    "profile_checksum",
    "validate_checksum",
]

