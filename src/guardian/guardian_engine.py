# src/guardian/guardian_engine.py
"""
Guardian Engine
=================

Unified Guardian system with hard authority.

Features:
    - Rule-based checks
    - PPO advisor
    - Hard authority mode
    - Decision logging
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("GUARDIAN_ENGINE")


@dataclass
class GuardianResult:
    """Result of Guardian evaluation."""
    allowed: bool
    decision: str  # "ALLOW", "BLOCK", "OVERRIDE"
    reason: str
    risk_level: str = "NORMAL"  # "LOW", "NORMAL", "HIGH", "CRITICAL"


class GuardianEngine:
    """
    Unified Guardian Engine.
    
    Combines:
        - Rule-based safety checks
        - PPO advisor
        - Margin monitoring
        - DD tracking
    """
    
    def __init__(
        self,
        profile: dict = None,
        ppo_model_path: str = "models/guardian_ppo_v3_20260115_1857.zip",
        hard_authority: bool = True
    ):
        self.profile = profile or {}
        self.ppo_model_path = ppo_model_path
        self.hard_authority = hard_authority
        
        self.ppo_model = None
        self._load_ppo()
        
        # Limits
        self.max_daily_dd = self.profile.get("daily_limits", {}).get("max_daily_drawdown_pct", 14.0)
        self.max_consecutive_losses = self.profile.get("daily_limits", {}).get("max_consecutive_losses", 4)
        self.min_margin_pct = self.profile.get("margin", {}).get("min_free_margin_pct", 250)
        
        # State
        self.daily_dd = 0.0
        self.consecutive_losses = 0
        self.hard_latch = False
        
        logger.info(f"🛡️ GuardianEngine initialized (hard_authority={hard_authority})")
    
    def _load_ppo(self):
        """Load Guardian PPO model."""
        try:
            from stable_baselines3 import PPO
            
            if os.path.exists(self.ppo_model_path):
                self.ppo_model = PPO.load(self.ppo_model_path)
                logger.info(f"✅ Guardian PPO loaded: {self.ppo_model_path}")
        except Exception as e:
            logger.warning(f"⚠️ Guardian PPO not loaded: {e}")
    
    def evaluate(
        self,
        action: str,
        market_state: dict,
        account_state: dict
    ) -> GuardianResult:
        """
        Evaluate proposed action.
        
        Args:
            action: Proposed action (BUY/SELL/HOLD)
            market_state: Current market conditions
            account_state: Account status
            
        Returns:
            GuardianResult
        """
        # Hard latch check
        if self.hard_latch:
            return GuardianResult(
                allowed=False,
                decision="BLOCK",
                reason="HARD_LATCH_ACTIVE",
                risk_level="CRITICAL"
            )
        
        # HOLD always allowed
        if action == "HOLD":
            return GuardianResult(
                allowed=True,
                decision="ALLOW",
                reason="HOLD_ACTION",
                risk_level="LOW"
            )
        
        # DD Check
        current_dd = account_state.get("daily_dd", 0)
        if current_dd >= self.max_daily_dd:
            self.hard_latch = True
            return GuardianResult(
                allowed=False,
                decision="BLOCK",
                reason=f"DD_LIMIT_EXCEEDED: {current_dd:.1f}% >= {self.max_daily_dd}%",
                risk_level="CRITICAL"
            )
        
        # Margin Check
        margin_pct = account_state.get("free_margin_pct", 100)
        if margin_pct < self.min_margin_pct:
            return GuardianResult(
                allowed=False,
                decision="BLOCK",
                reason=f"LOW_MARGIN: {margin_pct:.1f}% < {self.min_margin_pct}%",
                risk_level="HIGH"
            )
        
        # Consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return GuardianResult(
                allowed=False,
                decision="BLOCK",
                reason=f"LOSS_STREAK: {self.consecutive_losses} >= {self.max_consecutive_losses}",
                risk_level="HIGH"
            )
        
        # PPO Advisor check
        if self.ppo_model is not None:
            try:
                ppo_result = self._check_ppo(market_state, account_state)
                if not ppo_result["allow"]:
                    return GuardianResult(
                        allowed=False,
                        decision="BLOCK",
                        reason=f"PPO_ADVISE: {ppo_result['reason']}",
                        risk_level="HIGH"
                    )
            except:
                pass
        
        # All checks passed
        return GuardianResult(
            allowed=True,
            decision="ALLOW",
            reason="ALL_CHECKS_PASSED",
            risk_level="NORMAL"
        )
    
    def _check_ppo(self, market_state: dict, account_state: dict) -> dict:
        """Check with PPO advisor."""
        # Simplified PPO check
        # Real implementation would use full observation
        if account_state.get("daily_dd", 0) > self.max_daily_dd * 0.8:
            return {"allow": False, "reason": "HIGH_DD_RISK"}
        return {"allow": True, "reason": ""}
    
    def update_trade_result(self, is_win: bool):
        """Update after trade result."""
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
    
    def reset_daily(self):
        """Reset daily state."""
        self.daily_dd = 0.0
        self.consecutive_losses = 0
        self.hard_latch = False
        logger.info("🔄 Guardian daily reset")


# Singleton
_engine: Optional[GuardianEngine] = None

def get_guardian_engine(**kwargs) -> GuardianEngine:
    global _engine
    if _engine is None:
        _engine = GuardianEngine(**kwargs)
    return _engine


if __name__ == "__main__":
    engine = GuardianEngine()
    
    result = engine.evaluate(
        action="BUY",
        market_state={"price": 2650},
        account_state={"daily_dd": 5, "free_margin_pct": 300}
    )
    
    print(f"Allowed: {result.allowed}")
    print(f"Decision: {result.decision}")
    print(f"Reason: {result.reason}")
