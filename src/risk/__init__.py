"""
AI Trading System - Risk Management Package
=============================================
Deterministic risk management + CIO Brain architecture.
Now includes PositionManager and Win-Streak Booster for competition.
"""

from .position_sizer import PositionSizer
from .stop_loss import StopLossCalculator, StopLossEngine
from .auto_disable import AutoDisableManager, AutoDisableGuard
from .crisis_mode import CrisisModeController
from .recovery_engine import PostCrisisRecoveryEngine
from .capital_allocator import DynamicCapitalAllocator
from .meta_rl_allocator import MetaRLAllocator
from .ai_risk_committee import AIRiskCommittee
from .capital_protection import CapitalProtectionSystem
from .risk_manager import RiskManager
from .trailing import TrailingManager
from .position_manager import (
    PositionManager,
    PositionAction,
    PositionState,
    ManagementDecision,
    get_position_manager,
)
from .win_streak_booster import (
    WinStreakRiskBooster,
    RMultipleBooster,
    TradeResult,
    get_win_streak_booster,
)

__all__ = [
    "PositionSizer", 
    "StopLossCalculator", 
    "StopLossEngine",
    "AutoDisableManager", 
    "AutoDisableGuard",
    "CrisisModeController",
    "PostCrisisRecoveryEngine",
    "DynamicCapitalAllocator",
    "MetaRLAllocator",
    "AIRiskCommittee",
    "CapitalProtectionSystem",
    "RiskManager",
    "TrailingManager",
    # Position Manager
    "PositionManager",
    "PositionAction",
    "PositionState",
    "ManagementDecision",
    "get_position_manager",
    # Win-Streak Booster 🔥
    "WinStreakRiskBooster",
    "RMultipleBooster",
    "TradeResult",
    "get_win_streak_booster",
]


