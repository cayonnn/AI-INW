# src/rl/guardian_agent.py
"""
Guardian Agent - Risk Intelligence Advisor
============================================

Guardian Agent is an ADVISOR, not an executor.
- Observes system state
- Suggests risk adjustments via actions
- CANNOT override ProgressiveGuard

Architecture:
    Alpha Agent → GuardianAgent → ProgressiveGuard → Live Loop
    (propose)      (advise)        (enforce)         (execute)

Guardian suggestions are SOFT - can be ignored
ProgressiveGuard is HARD - cannot be bypassed
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("GUARDIAN_AGENT")


class GuardianAction(Enum):
    """Actions Guardian can suggest."""
    ALLOW = 0              # No change, proceed normally
    REDUCE_RISK = 1        # Lower lot size by 50%
    FORCE_HOLD = 2         # Override signal to HOLD
    WIDEN_SL = 3           # Increase SL distance
    EMERGENCY_FREEZE = 4   # Block all trading


# ===============================
# 🧠 GUARDIAN POLICY TABLE (RL-Ready)
# ===============================
GUARDIAN_POLICY_TABLE = [
    {
        "state": "DD_LIMIT",
        "condition": lambda s: s.get("daily_dd", 0) >= 0.10,
        "action": GuardianAction.EMERGENCY_FREEZE,
        "priority": 120
    },
    {
        "state": "ERROR_DETECTED",
        "condition": lambda s: s.get("error_detected", False),
        "action": GuardianAction.FORCE_HOLD,
        "priority": 110  # High priority for errors
    },
    {
        "state": "MARGIN_CRITICAL",
        "condition": lambda s: s.get("margin_ratio", 2.0) < 0.20,
        "action": GuardianAction.FORCE_HOLD,
        "priority": 100
    },
    {
        "state": "SPAM_BLOCK",
        "condition": lambda s: s.get("margin_block_count", 0) >= 3,
        "action": GuardianAction.FORCE_HOLD,
        "priority": 90
    },
    {
        "state": "MARGIN_LOW",
        "condition": lambda s: s.get("margin_ratio", 2.0) < 0.50,
        "action": GuardianAction.REDUCE_RISK,
        "priority": 80
    },
    {
        "state": "DD_WARNING",
        "condition": lambda s: s.get("daily_dd", 0) > 0.08,
        "action": GuardianAction.REDUCE_RISK,
        "priority": 70
    },
]


@dataclass
class GuardianSuggestion:
    """Guardian's risk adjustment suggestion."""
    action: GuardianAction = GuardianAction.ALLOW
    risk_multiplier: float = 1.0
    freeze_entries: bool = False
    reason: str = ""
    urgency: int = 0  # 0=normal, 1=warning, 2=urgent


class GuardianAgent:
    """
    Guardian Agent - Risk Intelligence Advisor.
    
    Responsibilities:
    - Observe system state (DD, margin, volatility)
    - Suggest risk adjustments to Alpha agent
    - Learn optimal defensive strategies via RL (future)
    
    Constraints:
    - Cannot override ProgressiveGuard
    - Cannot reset kill latch
    - Suggestions are advisory only
    """
    
    def __init__(
        self,
        mode: str = "advisor",
        model_path: str = "models/guardian_agent.npz"
    ):
        """Initialize Guardian agent."""
        self.mode = mode
        self.model_path = model_path
        self.daily_dd = 0.0
        
        logger.info("GuardianAgent initialized (advisor mode)")

    def reset_daily(self):
        """Reset daily tracking."""
        self.daily_dd = 0.0
        logger.info("GuardianAgent daily reset")

    def decide(self, state: Dict) -> GuardianAction:
        """
        Main decision interface using POLICY TABLE.
        
        Args:
            state: Dictionary with:
                - daily_dd: float (percentage as decimal)
                - margin_ratio: float
                - margin_block_count: int
                - open_positions: int
        
        Returns:
            GuardianAction enum
        """
        # Find all applicable policies
        applicable = [
            p for p in GUARDIAN_POLICY_TABLE 
            if p["condition"](state)
        ]
        
        if not applicable:
            return GuardianAction.ALLOW
        
        # Return highest priority action
        best = sorted(applicable, key=lambda x: x["priority"], reverse=True)[0]
        logger.info(f"Guardian Policy: {best['state']} → {best['action'].name}")
        return best["action"]

    def evaluate(self, signal: str, account_state: Dict, guardian_state: Dict) -> str:
        """
        Legacy interface for Guardian Check.
        Returns: "ALLOW" or "BLOCK"
        """
        # Check if latched
        if isinstance(guardian_state, dict) and guardian_state.get("is_latched", False):
            return "BLOCK"
        
        # Calculate margin ratio
        equity = account_state.get('equity', 0)
        free_margin = account_state.get('margin_free', 0)
        
        # Critical: margin under 5% of equity
        if equity > 0 and free_margin < (equity * 0.05):
            logger.warning(f"Guardian BLOCK: free_margin={free_margin:.0f} < 5% of equity={equity:.0f}")
            return "BLOCK"
        
        # Build state for decide()
        margin_ratio = (free_margin / max(equity - free_margin, 1)) if equity > 0 else 0
        
        state = {
            "daily_dd": self.daily_dd,
            "margin_ratio": margin_ratio,
            "margin_block_count": guardian_state.get("margin_block_count", 0),
            "open_positions": 0,
        }
        
        action = self.decide(state)
        
        if action in [GuardianAction.EMERGENCY_FREEZE, GuardianAction.FORCE_HOLD]:
            return "BLOCK"
        
        return "ALLOW"

    def observe_trade(self, trade_result):
        """Observe trade result for RL update."""
        loss_pct = getattr(trade_result, 'loss_pct', 0.0)
        profit = getattr(trade_result, 'profit', 0.0)
        
        if profit < 0:
            self.daily_dd += abs(loss_pct)


# Singleton
_guardian: Optional[GuardianAgent] = None


def get_guardian_agent() -> GuardianAgent:
    """Get singleton Guardian agent."""
    global _guardian
    if _guardian is None:
        _guardian = GuardianAgent()
    return _guardian
