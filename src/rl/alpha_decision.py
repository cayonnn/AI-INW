# src/rl/alpha_decision.py
"""
Alpha Decision Contract
=======================

Defines the decision object that Alpha PPO outputs.
This enables explainability and Guardian oversight.

Architecture:
    Alpha PPO → AlphaDecision → Guardian Review → Execution

The decision is NOT just a signal - it includes confidence,
risk assessment, and regime context for intelligent governance.
"""

from dataclasses import dataclass, asdict
from enum import Enum, IntEnum
from typing import Optional, Dict, Any
from datetime import datetime


class AlphaAction(IntEnum):
    """Alpha action types."""
    HOLD = 0
    BUY = 1
    SELL = 2


class MarketRegime(Enum):
    """Market regime classification."""
    TREND = "TREND"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class AlphaMode(Enum):
    """Alpha operating modes."""
    SHADOW = "shadow"    # PPO thinks, Rule executes
    HYBRID = "hybrid"    # PPO executes if confident, else Rule
    FULL_PPO = "full"    # PPO executes, Rule disabled
    

@dataclass
class AlphaDecision:
    """
    Decision object from Alpha PPO.
    
    Contains not just the action, but the reasoning context.
    This enables Guardian to make informed override decisions.
    
    Attributes:
        action: BUY/SELL/HOLD
        confidence: Model confidence [0.0 - 1.0]
        risk_score: Estimated trade risk [0.0 - 1.0] (higher = riskier)
        regime: Current market regime
        reason: Human-readable explanation
        timestamp: Decision timestamp
        
    Paper Statement:
        "The Alpha agent outputs a structured decision object rather than
        a simple signal, enabling hierarchical governance and full
        explainability of trading decisions."
    """
    action: AlphaAction
    confidence: float
    risk_score: float = 0.5
    regime: MarketRegime = MarketRegime.UNKNOWN
    reason: str = ""
    timestamp: Optional[str] = None
    
    # Extra context for Guardian
    ema_signal: Optional[str] = None  # What Rule would say
    guardian_compatible: bool = True  # Pre-check against soft constraints
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    @property
    def action_name(self) -> str:
        """Get action as string."""
        return AlphaAction(self.action).name
    
    @property
    def is_trade(self) -> bool:
        """Check if action is a trade (not HOLD)."""
        return self.action in (AlphaAction.BUY, AlphaAction.SELL)
    
    @property
    def is_confident(self) -> bool:
        """Check if confidence meets threshold (0.6)."""
        return self.confidence >= 0.60
    
    @property
    def is_low_risk(self) -> bool:
        """Check if risk is acceptable (< 0.7)."""
        return self.risk_score < 0.70
    
    def should_execute(self, mode: AlphaMode) -> bool:
        """
        Check if decision should be executed in given mode.
        
        Args:
            mode: Current operating mode
            
        Returns:
            True if PPO decision should be used
        """
        if mode == AlphaMode.SHADOW:
            return False  # Never execute in shadow
        
        if mode == AlphaMode.FULL_PPO:
            return True  # Always use PPO
        
        # HYBRID mode - use PPO only if confident
        return self.is_confident and self.is_low_risk
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "action": self.action_name,
            "confidence": round(self.confidence, 3),
            "risk_score": round(self.risk_score, 3),
            "regime": self.regime.value,
            "reason": self.reason,
            "ema_signal": self.ema_signal,
            "guardian_compatible": self.guardian_compatible,
            "timestamp": self.timestamp
        }
    
    def __repr__(self) -> str:
        return (
            f"AlphaDecision({self.action_name}, "
            f"conf={self.confidence:.0%}, "
            f"risk={self.risk_score:.0%}, "
            f"regime={self.regime.value})"
        )


@dataclass
class AlphaTradeResult:
    """
    Result of an Alpha-initiated trade.
    
    Used for RL reward assignment.
    """
    decision: AlphaDecision
    executed: bool
    blocked_by: Optional[str] = None  # "GUARDIAN", "RULE_FALLBACK", etc.
    order_ticket: Optional[int] = None
    
    # Outcome (filled after trade closes)
    profit: float = 0.0
    pnl_pct: float = 0.0
    dd_avoided: bool = False
    
    def calculate_reward(self) -> float:
        """
        Calculate RL reward for this trade.
        
        Reward function (Production):
            +profit for winning trades
            -0.1 for Guardian blocks
            +0.3 for DD avoided
            -0.2 for overtrade
            +0.05 for HOLD during risk
        """
        reward = 0.0
        
        if self.executed:
            # Trade was executed
            if self.profit > 0:
                reward += self.profit * 0.1  # Scaled profit
            else:
                reward += self.profit * 0.15  # Penalize losses more
        else:
            # Trade was blocked
            if self.blocked_by == "GUARDIAN":
                if self.dd_avoided:
                    reward += 0.3  # DD avoided - good decision
                else:
                    reward -= 0.1  # Blocked but unclear benefit
            elif self.blocked_by == "RULE_FALLBACK":
                reward -= 0.05  # Minor penalty for low confidence
        
        # Bonus for HOLD during risk
        if self.decision.action == AlphaAction.HOLD and self.decision.risk_score > 0.5:
            reward += 0.05
        
        return reward


# =============================================================================
# Factory Functions
# =============================================================================

def create_alpha_decision(
    action: int,
    confidence: float,
    risk_score: float = 0.5,
    regime: str = "UNKNOWN",
    reason: str = "",
    ema_signal: str = None
) -> AlphaDecision:
    """
    Factory function to create AlphaDecision.
    
    Args:
        action: 0=HOLD, 1=BUY, 2=SELL
        confidence: Model confidence [0-1]
        risk_score: Trade risk [0-1]
        regime: Market regime string
        reason: Explanation string
        ema_signal: What the Rule would say
        
    Returns:
        AlphaDecision object
    """
    return AlphaDecision(
        action=AlphaAction(action),
        confidence=float(confidence),
        risk_score=float(risk_score),
        regime=MarketRegime(regime) if regime in [r.value for r in MarketRegime] else MarketRegime.UNKNOWN,
        reason=reason,
        ema_signal=ema_signal
    )


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alpha Decision Contract Test")
    print("=" * 60)
    
    # Test decision creation
    decision = AlphaDecision(
        action=AlphaAction.BUY,
        confidence=0.82,
        risk_score=0.35,
        regime=MarketRegime.TREND,
        reason="Trend continuation + low DD",
        ema_signal="BUY"
    )
    
    print(f"\nDecision: {decision}")
    print(f"Should execute (SHADOW): {decision.should_execute(AlphaMode.SHADOW)}")
    print(f"Should execute (HYBRID): {decision.should_execute(AlphaMode.HYBRID)}")
    print(f"Should execute (FULL):   {decision.should_execute(AlphaMode.FULL_PPO)}")
    print(f"\nDict: {decision.to_dict()}")
    
    # Test low confidence decision
    low_conf = AlphaDecision(
        action=AlphaAction.SELL,
        confidence=0.45,
        risk_score=0.60,
        regime=MarketRegime.VOLATILE,
        reason="Weak signal in volatile market"
    )
    
    print(f"\nLow confidence: {low_conf}")
    print(f"Should execute (HYBRID): {low_conf.should_execute(AlphaMode.HYBRID)}")
    
    # Test reward calculation
    result = AlphaTradeResult(
        decision=decision,
        executed=True,
        profit=15.5
    )
    print(f"\nTrade result reward: {result.calculate_reward():.3f}")
    
    blocked_result = AlphaTradeResult(
        decision=decision,
        executed=False,
        blocked_by="GUARDIAN",
        dd_avoided=True
    )
    print(f"Blocked (DD avoided) reward: {blocked_result.calculate_reward():.3f}")
    
    print("=" * 60)
