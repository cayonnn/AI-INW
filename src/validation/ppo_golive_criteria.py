# src/validation/ppo_golive_criteria.py
"""
PPO Alpha Go-Live Criteria
===========================

Defines strict criteria for enabling PPO live trading.

Stage 1: Shadow Validation (REQUIRED)
Stage 2: Soft Control (10-20%)
Stage 3: Partial Authority

NO PPO LIVE without passing ALL criteria!
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("PPO_GOLIVE")


# =============================================================================
# Go-Live Criteria
# =============================================================================

@dataclass
class Stage1Criteria:
    """Shadow Validation (MINIMUM REQUIRED)"""
    min_trading_days: int = 20
    min_winrate_advantage: float = 0.05  # PPO >= Rule + 5%
    dd_avoided_positive: bool = True     # DD avoided > Freeze cost
    max_guardian_conflicts: float = 0.10  # < 10%
    max_catastrophic_trades: int = 0


@dataclass
class Stage2Criteria:
    """Soft Control (10-20% of trades)"""
    min_confidence: float = 0.85
    require_guardian_safe: bool = True
    max_spread_multiplier: float = 1.2
    require_no_news: bool = True
    max_control_ratio: float = 0.20


@dataclass
class KillConditions:
    """Conditions that DISABLE PPO immediately"""
    max_ppo_dd: float = 0.02          # 2% DD caused by PPO
    max_loss_streak: int = 2          # PPO-specific losses
    confidence_collapse: bool = False  # Sudden confidence drop


@dataclass
class GoLiveStatus:
    """Current go-live status"""
    stage1_passed: bool = False
    stage2_passed: bool = False
    stage3_passed: bool = False
    current_stage: int = 0
    blocking_reasons: List[str] = None
    last_check: str = None
    
    def __post_init__(self):
        if self.blocking_reasons is None:
            self.blocking_reasons = []
        if self.last_check is None:
            self.last_check = datetime.now().isoformat()


class PPOGoLiveCriteria:
    """
    Manages PPO go-live criteria and stage progression.
    
    Stages:
        0: Shadow Only (PPO thinks, Rule executes)
        1: Shadow Validated (metrics pass)
        2: Soft Control (10-20% of trades)
        3: Partial Authority (more control)
    """
    
    def __init__(self):
        self.stage1 = Stage1Criteria()
        self.stage2 = Stage2Criteria()
        self.kill_conditions = KillConditions()
        self.status = GoLiveStatus()
        
        logger.info("🎯 PPOGoLiveCriteria initialized")
    
    def check_stage1(
        self,
        trading_days: int,
        ppo_winrate: float,
        rule_winrate: float,
        dd_avoided: float,
        freeze_cost: float,
        guardian_conflicts: float,
        catastrophic_trades: int
    ) -> bool:
        """Check Stage 1 (Shadow Validation) criteria."""
        blocking = []
        
        if trading_days < self.stage1.min_trading_days:
            blocking.append(f"Need {self.stage1.min_trading_days} days (have {trading_days})")
        
        winrate_diff = ppo_winrate - rule_winrate
        if winrate_diff < self.stage1.min_winrate_advantage:
            blocking.append(f"PPO winrate advantage {winrate_diff:.1%} < {self.stage1.min_winrate_advantage:.1%}")
        
        if dd_avoided <= freeze_cost:
            blocking.append(f"DD avoided ${dd_avoided:.2f} <= Freeze cost ${freeze_cost:.2f}")
        
        if guardian_conflicts > self.stage1.max_guardian_conflicts:
            blocking.append(f"Guardian conflicts {guardian_conflicts:.1%} > {self.stage1.max_guardian_conflicts:.1%}")
        
        if catastrophic_trades > self.stage1.max_catastrophic_trades:
            blocking.append(f"Catastrophic trades: {catastrophic_trades}")
        
        self.status.stage1_passed = len(blocking) == 0
        self.status.blocking_reasons = blocking
        self.status.last_check = datetime.now().isoformat()
        
        if self.status.stage1_passed:
            self.status.current_stage = max(1, self.status.current_stage)
            logger.info("✅ Stage 1 PASSED: Shadow Validation complete")
        else:
            logger.warning(f"❌ Stage 1 BLOCKED: {blocking}")
        
        return self.status.stage1_passed
    
    def check_trade_allowed(
        self,
        confidence: float,
        guardian_state: str,
        spread_ratio: float,
        is_news_window: bool
    ) -> tuple:
        """Check if PPO trade is allowed (Stage 2 criteria)."""
        if self.status.current_stage < 1:
            return False, "Stage 1 not passed"
        
        if confidence < self.stage2.min_confidence:
            return False, f"Confidence {confidence:.2f} < {self.stage2.min_confidence}"
        
        if self.stage2.require_guardian_safe and guardian_state != "SAFE":
            return False, f"Guardian not SAFE ({guardian_state})"
        
        if spread_ratio > self.stage2.max_spread_multiplier:
            return False, f"Spread ratio {spread_ratio:.2f} > {self.stage2.max_spread_multiplier}"
        
        if self.stage2.require_no_news and is_news_window:
            return False, "News window active"
        
        return True, "All criteria met"
    
    def check_kill_conditions(
        self,
        ppo_dd: float,
        ppo_loss_streak: int
    ) -> bool:
        """Check if PPO should be killed."""
        should_kill = False
        reason = None
        
        if ppo_dd >= self.kill_conditions.max_ppo_dd:
            should_kill = True
            reason = f"PPO DD {ppo_dd:.1%} >= {self.kill_conditions.max_ppo_dd:.1%}"
        
        if ppo_loss_streak >= self.kill_conditions.max_loss_streak:
            should_kill = True
            reason = f"PPO loss streak {ppo_loss_streak} >= {self.kill_conditions.max_loss_streak}"
        
        if should_kill:
            logger.warning(f"🚨 PPO KILL CONDITION: {reason}")
        
        return should_kill
    
    def get_current_control_ratio(self) -> float:
        """Get current PPO control ratio based on stage."""
        if self.status.current_stage < 1:
            return 0.0  # Shadow only
        elif self.status.current_stage == 1:
            return 0.10  # 10%
        elif self.status.current_stage == 2:
            return 0.20  # 20%
        else:
            return 0.30  # 30% max
    
    def summary(self) -> str:
        """Generate status summary."""
        lines = [
            "=" * 50,
            "🎯 PPO GO-LIVE STATUS",
            "=" * 50,
            f"Current Stage: {self.status.current_stage}",
            f"Stage 1 (Shadow): {'✅' if self.status.stage1_passed else '❌'}",
            f"Stage 2 (Soft):   {'✅' if self.status.stage2_passed else '❌'}",
            f"Control Ratio: {self.get_current_control_ratio():.0%}",
        ]
        
        if self.status.blocking_reasons:
            lines.append("\nBlocking Reasons:")
            for reason in self.status.blocking_reasons:
                lines.append(f"  ❌ {reason}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


# =============================================================================
# Singleton
# =============================================================================

_criteria: Optional[PPOGoLiveCriteria] = None


def get_golive_criteria() -> PPOGoLiveCriteria:
    """Get singleton criteria."""
    global _criteria
    if _criteria is None:
        _criteria = PPOGoLiveCriteria()
    return _criteria


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PPO Go-Live Criteria Test")
    print("=" * 60)
    
    criteria = PPOGoLiveCriteria()
    
    # Test Stage 1
    passed = criteria.check_stage1(
        trading_days=25,
        ppo_winrate=0.58,
        rule_winrate=0.52,
        dd_avoided=150.0,
        freeze_cost=50.0,
        guardian_conflicts=0.08,
        catastrophic_trades=0
    )
    
    print(criteria.summary())
    
    # Test trade allowed
    allowed, reason = criteria.check_trade_allowed(
        confidence=0.88,
        guardian_state="SAFE",
        spread_ratio=1.1,
        is_news_window=False
    )
    print(f"\nTrade allowed: {allowed} ({reason})")
    
    # Test kill conditions
    killed = criteria.check_kill_conditions(ppo_dd=0.01, ppo_loss_streak=1)
    print(f"Kill condition: {killed}")
