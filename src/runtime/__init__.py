# src/runtime/__init__.py
"""
Runtime Module - System Health & Monitoring
============================================

Production-grade runtime components:
- Profile Registry & Drift Detection
- System Health Monitoring
"""

from .profile_registry import (
    ProfileRegistry,
    ModuleRegistration,
    guard_profile_drift,
)

__all__ = [
    "ProfileRegistry",
    "ModuleRegistration",
    "guard_profile_drift",
]
