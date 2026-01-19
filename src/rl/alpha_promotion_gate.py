# src/rl/alpha_promotion_gate.py
"""
Alpha PPO Promotion Gate
========================

Defines criteria for promoting Alpha PPO from Shadow to Live mode.

Safety-first approach:
- PPO must prove itself before going live
- Guardian remains unchanged (frozen)
- Conservative thresholds

Usage:
    gate = PromotionGate()
    ready, blockers = gate.evaluate(shadow_stats)
    if ready:
        # Consider enabling PPO Alpha in live mode
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("PROMOTION_GATE")


@dataclass
class PromotionCriteria:
    """
    Criteria for promoting Alpha PPO to live trading.
    
    All criteria must be met for promotion.
    Conservative defaults to ensure safety.
    """
    # Minimum data requirements
    min_shadow_comparisons: int = 500
    min_shadow_days: int = 3
    
    # Performance requirements
    min_shadow_win_rate: float = 0.52
    max_shadow_dd: float = 0.12
    min_shadow_pnl: float = 0.0  # Must be positive
    
    # Behavioral requirements
    min_agreement_rate: float = 0.35  # PPO should agree reasonably
    max_trade_rate_delta: float = 0.20  # PPO shouldn't be too different
    
    # Guardian compatibility
    max_guardian_block_increase: float = 0.10  # PPO shouldn't increase blocks
    
    # Consistency requirements
    min_consecutive_positive_days: int = 2


@dataclass
class PromotionEvaluation:
    """Result of promotion gate evaluation."""
    timestamp: str
    ready: bool
    score: float  # 0-100 overall readiness score
    criteria_met: int
    criteria_total: int
    blockers: List[str]
    warnings: List[str]
    recommendation: str


class PromotionGate:
    """
    Evaluate if Alpha PPO is ready for live trading.
    
    This is a CONSERVATIVE gate - better to delay promotion
    than risk capital with an unproven model.
    """
    
    def __init__(self, criteria: Optional[PromotionCriteria] = None):
        self.criteria = criteria or PromotionCriteria()
    
    def evaluate(self, stats: Dict[str, Any]) -> PromotionEvaluation:
        """
        Evaluate promotion readiness.
        
        Args:
            stats: Dictionary with shadow performance stats
            
        Returns:
            PromotionEvaluation with detailed results
        """
        blockers = []
        warnings = []
        criteria_met = 0
        criteria_total = 0
        
        # 1. Minimum comparisons
        criteria_total += 1
        comparisons = stats.get("total_comparisons", 0)
        if comparisons >= self.criteria.min_shadow_comparisons:
            criteria_met += 1
        else:
            blockers.append(
                f"Need {self.criteria.min_shadow_comparisons} comparisons "
                f"(have {comparisons})"
            )
        
        # 2. Shadow win rate
        criteria_total += 1
        win_rate = stats.get("shadow_win_rate", 0)
        if win_rate >= self.criteria.min_shadow_win_rate:
            criteria_met += 1
        else:
            blockers.append(
                f"Shadow win rate {win_rate:.1%} < "
                f"{self.criteria.min_shadow_win_rate:.1%}"
            )
        
        # 3. Shadow PnL positive
        criteria_total += 1
        shadow_pnl = stats.get("shadow_pnl", 0)
        if shadow_pnl >= self.criteria.min_shadow_pnl:
            criteria_met += 1
        else:
            blockers.append(f"Shadow PnL ${shadow_pnl:.2f} is negative")
        
        # 4. Agreement rate
        criteria_total += 1
        agreement_rate = stats.get("agreement_rate", 0)
        if agreement_rate >= self.criteria.min_agreement_rate:
            criteria_met += 1
        else:
            warnings.append(
                f"Low agreement rate {agreement_rate:.1%} may indicate "
                f"different trading style"
            )
        
        # 5. Trade rate similarity
        criteria_total += 1
        rule_trade_rate = stats.get("rule_trade_rate", 0)
        ppo_trade_rate = stats.get("ppo_trade_rate", 0)
        delta = abs(rule_trade_rate - ppo_trade_rate)
        if delta <= self.criteria.max_trade_rate_delta:
            criteria_met += 1
        else:
            warnings.append(
                f"Trade rate delta {delta:.1%} > {self.criteria.max_trade_rate_delta:.1%}"
            )
        
        # Calculate score
        score = (criteria_met / criteria_total) * 100 if criteria_total > 0 else 0
        ready = len(blockers) == 0
        
        # Generate recommendation
        if ready:
            recommendation = "✅ Alpha PPO is ready for live trading promotion."
        elif score >= 60:
            recommendation = (
                "⚠️ Alpha PPO is close to ready. "
                f"Address {len(blockers)} blocker(s) before promotion."
            )
        else:
            recommendation = (
                f"❌ Alpha PPO needs more shadow testing. "
                f"Score: {score:.0f}/100"
            )
        
        return PromotionEvaluation(
            timestamp=datetime.now().isoformat(),
            ready=ready,
            score=score,
            criteria_met=criteria_met,
            criteria_total=criteria_total,
            blockers=blockers,
            warnings=warnings,
            recommendation=recommendation
        )
    
    def generate_report(self, stats: Dict[str, Any]) -> str:
        """Generate promotion gate report."""
        eval_result = self.evaluate(stats)
        
        lines = [
            "=" * 60,
            "🚀 ALPHA PPO PROMOTION GATE EVALUATION",
            "=" * 60,
            f"Timestamp: {eval_result.timestamp}",
            f"Readiness Score: {eval_result.score:.0f}/100",
            f"Criteria Met: {eval_result.criteria_met}/{eval_result.criteria_total}",
            "",
            f"Status: {'✅ READY' if eval_result.ready else '❌ NOT READY'}",
            "",
        ]
        
        if eval_result.blockers:
            lines.append("🚫 BLOCKERS:")
            for blocker in eval_result.blockers:
                lines.append(f"  • {blocker}")
            lines.append("")
        
        if eval_result.warnings:
            lines.append("⚠️ WARNINGS:")
            for warning in eval_result.warnings:
                lines.append(f"  • {warning}")
            lines.append("")
        
        lines.append(f"📋 {eval_result.recommendation}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alpha PPO Promotion Gate Test")
    print("=" * 60)
    
    gate = PromotionGate()
    
    # Test with sample stats
    test_stats = {
        "total_comparisons": 250,  # Not enough
        "shadow_win_rate": 0.55,
        "shadow_pnl": 15.5,
        "agreement_rate": 0.45,
        "rule_trade_rate": 0.25,
        "ppo_trade_rate": 0.30
    }
    
    print("\n--- Test Case 1: Insufficient Data ---")
    print(gate.generate_report(test_stats))
    
    # Test with good stats
    good_stats = {
        "total_comparisons": 800,
        "shadow_win_rate": 0.58,
        "shadow_pnl": 45.0,
        "agreement_rate": 0.52,
        "rule_trade_rate": 0.25,
        "ppo_trade_rate": 0.28
    }
    
    print("\n--- Test Case 2: Good Performance ---")
    print(gate.generate_report(good_stats))
