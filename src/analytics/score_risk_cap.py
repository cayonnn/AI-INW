# src/analytics/score_risk_cap.py
"""
Score-driven Risk Cap - Leaderboard Protection
===============================================

Soft clamp on risk scaling based on LiveScore:
- Low score -> Reduce risk multiplier
- Low score -> Disable pyramid entries
- Does NOT kill trades, just reduces scaling

Prevents over-leveraging when underperforming.
"""

from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("SCORE_RISK_CAP")


@dataclass
class RiskCapResult:
    """Risk cap calculation result."""
    risk_mult: float
    max_pyramid: Optional[int]
    reason: str


class ScoreRiskCap:
    """
    Score-based Risk Cap Engine.
    
    Cap Rules:
    ┌─────────────┬───────────┬─────────────┐
    │ Score       │ Risk Mult │ Max Pyramid │
    ├─────────────┼───────────┼─────────────┤
    │ < 30        │ 0.5x      │ 0           │
    │ 30-45       │ 0.8x      │ 1           │
    │ > 45        │ 1.0x      │ None (full) │
    └─────────────┴───────────┴─────────────┘
    """
    
    # Score thresholds
    CRITICAL_THRESHOLD = 30
    WARNING_THRESHOLD = 45
    
    def __init__(
        self,
        critical_threshold: float = 30,
        warning_threshold: float = 45,
        critical_mult: float = 0.5,
        warning_mult: float = 0.8
    ):
        """
        Initialize Score Risk Cap.
        
        Args:
            critical_threshold: Score below this = critical cap
            warning_threshold: Score below this = warning cap
            critical_mult: Risk multiplier in critical
            warning_mult: Risk multiplier in warning
        """
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        self.critical_mult = critical_mult
        self.warning_mult = warning_mult
        
        logger.info(
            f"ScoreRiskCap initialized: "
            f"critical<{critical_threshold} ({critical_mult}x), "
            f"warning<{warning_threshold} ({warning_mult}x)"
        )
    
    def cap(self, score: float) -> RiskCapResult:
        """
        Calculate risk cap based on score.
        
        Args:
            score: Current LiveScore (0-100)
            
        Returns:
            RiskCapResult with multiplier and pyramid limit
        """
        if score < self.critical_threshold:
            return RiskCapResult(
                risk_mult=self.critical_mult,
                max_pyramid=0,
                reason=f"Critical score ({score:.1f} < {self.critical_threshold})"
            )
        
        if score < self.warning_threshold:
            return RiskCapResult(
                risk_mult=self.warning_mult,
                max_pyramid=1,
                reason=f"Warning score ({score:.1f} < {self.warning_threshold})"
            )
        
        # Normal - no cap
        return RiskCapResult(
            risk_mult=1.0,
            max_pyramid=None,  # No limit
            reason="Score OK - no cap"
        )
    
    def get_effective_pyramid_limit(
        self,
        score: float,
        mode_limit: int,
        base_limit: int = 3
    ) -> int:
        """
        Get effective pyramid limit considering all factors.
        
        Args:
            score: Current LiveScore
            mode_limit: Limit from trading mode
            base_limit: Base pyramid limit
            
        Returns:
            Effective maximum pyramid entries
        """
        cap_result = self.cap(score)
        
        limits = [base_limit, mode_limit]
        if cap_result.max_pyramid is not None:
            limits.append(cap_result.max_pyramid)
        
        return min(limits)


# Singleton instance
_cap: Optional[ScoreRiskCap] = None


def get_score_risk_cap() -> ScoreRiskCap:
    """Get or create singleton ScoreRiskCap."""
    global _cap
    if _cap is None:
        _cap = ScoreRiskCap()
    return _cap


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ScoreRiskCap Test")
    print("=" * 60)
    
    cap = ScoreRiskCap()
    
    # Test various scores
    test_scores = [20, 30, 40, 50, 60, 70, 80]
    
    print("\n--- Score Cap Results ---")
    for score in test_scores:
        result = cap.cap(score)
        pyramid_str = str(result.max_pyramid) if result.max_pyramid is not None else "unlimited"
        print(
            f"Score {score:3d}: Risk={result.risk_mult}x, "
            f"Pyramid={pyramid_str}, {result.reason}"
        )
