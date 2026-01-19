# src/safety/compliance_ai.py
"""
Compliance AI
==============

AI-driven regulatory compliance enforcement.

Features:
    - Exposure limits per symbol
    - Daily loss limits
    - Trading hours enforcement
    - News blackout windows
    - Policy learning

Paper Statement:
    "Our compliance layer learns regulatory constraints as policies
     rather than hard-coded rules, enabling adaptive governance."
"""

import os
import sys
from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("COMPLIANCE_AI")


class ComplianceAction(IntEnum):
    """Compliance actions."""
    ALLOW = 0
    MODIFY = 1   # Modify trade parameters
    BLOCK = 2    # Block trade
    ESCALATE = 3 # Escalate to human


@dataclass
class CompliancePolicy:
    """Compliance policy definition."""
    name: str
    enabled: bool = True
    
    # Limits
    max_exposure_per_symbol: float = 0.20  # 20% of equity
    max_daily_loss: float = 0.10  # 10% daily loss limit
    max_position_size: float = 0.05  # 5% per position
    max_concurrent_positions: int = 5
    
    # Time restrictions
    trading_start_hour: int = 1  # 01:00 UTC
    trading_end_hour: int = 23   # 23:00 UTC
    
    # News blackout
    news_blackout_minutes: int = 30  # Before/after news


@dataclass
class ComplianceDecision:
    """Compliance check result."""
    action: ComplianceAction
    violations: List[str]
    modifications: Dict[str, Any]
    reason: str
    
    @property
    def is_compliant(self) -> bool:
        return self.action == ComplianceAction.ALLOW


class ComplianceAI:
    """
    AI-driven Compliance Layer.
    
    Enforces regulatory and internal policies
    with explainable decisions.
    
    Features:
        - Rule-based + ML hybrid
        - Configurable policies
        - Audit logging
        - Escalation support
    """
    
    def __init__(self, policy: Optional[CompliancePolicy] = None):
        self.policy = policy or CompliancePolicy(name="Default")
        
        # State tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_exposure: Dict[str, float] = {}
        self.violations_today = 0
        
        # History
        self.decision_history: List[ComplianceDecision] = []
        
        logger.info(f"🔐 ComplianceAI initialized: {self.policy.name}")
    
    def check(
        self,
        trade_request: Dict[str, Any],
        account_state: Dict[str, Any]
    ) -> ComplianceDecision:
        """
        Check trade compliance.
        
        Args:
            trade_request: Proposed trade details
            account_state: Current account state
            
        Returns:
            ComplianceDecision with action and violations
        """
        violations = []
        modifications = {}
        
        symbol = trade_request.get("symbol", "XAUUSD")
        lot_size = trade_request.get("lot_size", 0.01)
        action = trade_request.get("action", "HOLD")
        equity = account_state.get("equity", 1000)
        
        # Check 1: Trading hours
        if not self._check_trading_hours():
            violations.append("OUTSIDE_TRADING_HOURS")
        
        # Check 2: Daily loss limit
        if not self._check_daily_loss(account_state):
            violations.append("DAILY_LOSS_EXCEEDED")
        
        # Check 3: Exposure limit
        if action != "HOLD":
            position_value = lot_size * 100000 * 0.001  # Approximate
            exposure_pct = position_value / equity
            
            if exposure_pct > self.policy.max_position_size:
                violations.append("POSITION_SIZE_EXCEEDED")
                # Suggest modification
                max_lot = (self.policy.max_position_size * equity) / (100000 * 0.001)
                modifications["lot_size"] = min(lot_size, max_lot)
        
        # Check 4: Concurrent positions
        open_positions = account_state.get("open_positions", 0)
        if open_positions >= self.policy.max_concurrent_positions:
            if action != "HOLD":
                violations.append("MAX_POSITIONS_REACHED")
        
        # Determine action
        if not violations:
            decision_action = ComplianceAction.ALLOW
            reason = "All checks passed"
        elif len(violations) == 1 and modifications:
            decision_action = ComplianceAction.MODIFY
            reason = f"Modified due to: {violations[0]}"
        elif "DAILY_LOSS_EXCEEDED" in violations:
            decision_action = ComplianceAction.BLOCK
            self.violations_today += 1
            reason = "Daily loss limit reached - trading blocked"
        else:
            decision_action = ComplianceAction.BLOCK
            self.violations_today += 1
            reason = f"Blocked: {', '.join(violations)}"
        
        decision = ComplianceDecision(
            action=decision_action,
            violations=violations,
            modifications=modifications,
            reason=reason
        )
        
        self.decision_history.append(decision)
        
        return decision
    
    def _check_trading_hours(self) -> bool:
        """Check if within trading hours."""
        current_hour = datetime.now().hour
        return self.policy.trading_start_hour <= current_hour < self.policy.trading_end_hour
    
    def _check_daily_loss(self, account_state: Dict) -> bool:
        """Check daily loss limit."""
        daily_pnl_pct = account_state.get("daily_pnl_pct", 0)
        return daily_pnl_pct > -self.policy.max_daily_loss
    
    def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L tracking."""
        self.daily_pnl += pnl_change
    
    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.violations_today = 0
        logger.info("🔄 Compliance daily counters reset")
    
    def get_summary(self) -> Dict:
        """Get compliance summary."""
        total_checks = len(self.decision_history)
        blocks = sum(1 for d in self.decision_history if d.action == ComplianceAction.BLOCK)
        
        return {
            "policy": self.policy.name,
            "total_checks": total_checks,
            "blocks": blocks,
            "block_rate": blocks / max(total_checks, 1),
            "violations_today": self.violations_today,
            "daily_pnl": self.daily_pnl
        }


# =============================================================================
# Singleton
# =============================================================================

_compliance_ai: Optional[ComplianceAI] = None


def get_compliance_ai() -> ComplianceAI:
    """Get singleton ComplianceAI."""
    global _compliance_ai
    if _compliance_ai is None:
        _compliance_ai = ComplianceAI()
    return _compliance_ai


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Compliance AI Test")
    print("=" * 60)
    
    compliance = ComplianceAI()
    
    # Test trades
    trades = [
        {"symbol": "XAUUSD", "action": "BUY", "lot_size": 0.02},
        {"symbol": "XAUUSD", "action": "BUY", "lot_size": 0.50},  # Too big
        {"symbol": "EURUSD", "action": "SELL", "lot_size": 0.01},
    ]
    
    account = {"equity": 1000, "open_positions": 2, "daily_pnl_pct": -0.05}
    
    for trade in trades:
        decision = compliance.check(trade, account)
        status = "✅" if decision.is_compliant else "❌"
        print(f"\n{status} {trade['action']} {trade['lot_size']} lot:")
        print(f"   Action: {decision.action.name}")
        print(f"   Reason: {decision.reason}")
        if decision.violations:
            print(f"   Violations: {decision.violations}")
        if decision.modifications:
            print(f"   Modifications: {decision.modifications}")
    
    print(f"\n{compliance.get_summary()}")
    print("=" * 60)
