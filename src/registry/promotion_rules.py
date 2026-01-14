# src/registry/promotion_rules.py
"""
Chaos-Aware Promotion Rules
============================

Hard gates for profile promotion:
1. Chaos test ≥ 95%
2. No DD-kill in 7 days
3. Shadow > Live for 3 consecutive days
4. No manual override

Leaderboard prefers "doesn't die" over "profits fast but crashes".
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger

logger = get_logger("PROMOTION_RULES")


@dataclass
class PromotionGate:
    """Single promotion gate."""
    name: str
    passed: bool
    value: Any
    required: Any
    reason: str


class PromotionRules:
    """
    Hard Gates for Promotion.
    
    Cannot override manually - system enforced.
    
    Gates:
    1. Chaos success rate ≥ 95%
    2. Zero DD-kills in past 7 days
    3. Shadow score > Live score for 3+ days
    4. Shadow max DD ≤ Live max DD
    5. Resilience score positive
    """
    
    # Gate thresholds
    MIN_CHAOS_RATE = 0.95
    MIN_DAYS_NO_KILL = 7
    MIN_CONSECUTIVE_WINS = 3
    
    def __init__(self):
        """Initialize promotion rules."""
        self.kill_history: List[datetime] = []
        self.score_history: Dict[str, List[float]] = {
            "live": [],
            "shadow": [],
        }
        logger.info("PromotionRules initialized (hard gates enforced)")
    
    def evaluate(
        self,
        shadow_id: str,
        live_id: str,
        chaos_rate: float,
        shadow_scores: List[float],
        live_scores: List[float],
        shadow_dd: float,
        live_dd: float
    ) -> Dict[str, Any]:
        """
        Evaluate all promotion gates.
        
        Returns dict with:
        - approved: bool
        - gates: List[PromotionGate]
        - reason: str
        """
        gates = []
        
        # Gate 1: Chaos success rate
        chaos_passed = chaos_rate >= self.MIN_CHAOS_RATE
        gates.append(PromotionGate(
            name="chaos_rate",
            passed=chaos_passed,
            value=f"{chaos_rate:.0%}",
            required=f"≥{self.MIN_CHAOS_RATE:.0%}",
            reason="Chaos test success rate"
        ))
        
        # Gate 2: No DD-kills recently
        days_since_kill = self._days_since_last_kill()
        kill_passed = days_since_kill >= self.MIN_DAYS_NO_KILL
        gates.append(PromotionGate(
            name="no_dd_kill",
            passed=kill_passed,
            value=f"{days_since_kill} days",
            required=f"≥{self.MIN_DAYS_NO_KILL} days",
            reason="Days since DD-triggered kill"
        ))
        
        # Gate 3: Consecutive wins
        consecutive_wins = self._count_consecutive_wins(shadow_scores, live_scores)
        wins_passed = consecutive_wins >= self.MIN_CONSECUTIVE_WINS
        gates.append(PromotionGate(
            name="consecutive_wins",
            passed=wins_passed,
            value=f"{consecutive_wins} days",
            required=f"≥{self.MIN_CONSECUTIVE_WINS} days",
            reason="Shadow beats live consecutively"
        ))
        
        # Gate 4: DD comparison
        dd_passed = shadow_dd <= live_dd
        gates.append(PromotionGate(
            name="dd_comparison",
            passed=dd_passed,
            value=f"{shadow_dd:.1%}",
            required=f"≤{live_dd:.1%}",
            reason="Shadow DD must be ≤ Live DD"
        ))
        
        # Gate 5: Resilience (no kills during shadow period)
        resilience = self._calculate_resilience(shadow_scores)
        resilience_passed = resilience > 0
        gates.append(PromotionGate(
            name="resilience",
            passed=resilience_passed,
            value=f"{resilience:.2f}",
            required=">0",
            reason="Positive resilience score"
        ))
        
        # All gates must pass
        all_passed = all(g.passed for g in gates)
        
        if all_passed:
            reason = "✅ All gates passed - PROMOTION APPROVED"
        else:
            failed = [g.name for g in gates if not g.passed]
            reason = f"❌ Failed gates: {', '.join(failed)}"
        
        logger.info(reason)
        
        return {
            "approved": all_passed,
            "shadow_id": shadow_id,
            "live_id": live_id,
            "gates": [
                {
                    "name": g.name,
                    "passed": g.passed,
                    "value": g.value,
                    "required": g.required,
                }
                for g in gates
            ],
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
    
    def record_kill(self) -> None:
        """Record a DD-kill event."""
        self.kill_history.append(datetime.now())
        logger.warning("DD-kill recorded for promotion tracking")
    
    def _days_since_last_kill(self) -> int:
        """Calculate days since last kill."""
        if not self.kill_history:
            return 999  # No kills ever
        
        last_kill = max(self.kill_history)
        delta = datetime.now() - last_kill
        return delta.days
    
    def _count_consecutive_wins(
        self,
        shadow: List[float],
        live: List[float]
    ) -> int:
        """Count consecutive days shadow > live."""
        if not shadow or not live:
            return 0
        
        min_len = min(len(shadow), len(live))
        consecutive = 0
        
        # Check from most recent
        for i in range(min_len - 1, -1, -1):
            if shadow[i] > live[i]:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _calculate_resilience(self, scores: List[float]) -> float:
        """Calculate resilience score."""
        if len(scores) < 2:
            return 0
        
        # Resilience = consistency + recovery
        deltas = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        
        # Positive = more recoveries than drops
        positive = sum(1 for d in deltas if d > 0)
        negative = sum(1 for d in deltas if d < 0)
        
        return (positive - negative) / len(deltas)


def create_promotion_report(result: Dict) -> str:
    """Create markdown promotion report."""
    status = "✅ APPROVED" if result["approved"] else "❌ BLOCKED"
    
    report = f"""# Promotion Evaluation

## Status: {status}

**Shadow:** {result['shadow_id']}
**Live:** {result['live_id']}

## Gate Results

| Gate | Status | Value | Required |
|------|--------|-------|----------|
"""
    
    for gate in result["gates"]:
        icon = "✅" if gate["passed"] else "❌"
        report += f"| {gate['name']} | {icon} | {gate['value']} | {gate['required']} |\n"
    
    report += f"""
## Reason
{result['reason']}

---
*Evaluated: {result['timestamp']}*
*Override: NOT ALLOWED*
"""
    
    return report


# Singleton
_rules: Optional[PromotionRules] = None


def get_promotion_rules() -> PromotionRules:
    """Get singleton PromotionRules."""
    global _rules
    if _rules is None:
        _rules = PromotionRules()
    return _rules
