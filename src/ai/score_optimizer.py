# src/ai/score_optimizer.py
"""
Score Optimizer - Leaderboard-First Engine
==========================================

Optimizes competition score directly, not just P/L.

Integrates with:
- LiveScoreEstimator
- RiskManager
- PyramidManager
- ModeController

Logic:
- Low score → Reduce pyramid, cap risk
- Score improving → Allow boost
- Focus on Sharpe, WinRate, not just profit
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("SCORE_OPTIMIZER")


class ScoreTrend(str, Enum):
    """Score trend direction."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"


@dataclass
class OptimizationResult:
    """Score optimization result."""
    risk_cap: float             # Risk multiplier cap
    max_pyramid: int            # Maximum pyramid depth
    confidence_boost: float     # Confidence multiplier
    allow_alpha: bool           # Allow ALPHA mode
    reason: str
    trend: ScoreTrend
    score: float
    target_score: float


class ScoreOptimizer:
    """
    Score Optimizer - Leaderboard-First Engine.
    
    Optimization Rules:
    ┌───────────────┬───────────┬───────────┬─────────────┐
    │ Score         │ Risk Cap  │ Pyramid   │ Alpha OK    │
    ├───────────────┼───────────┼───────────┼─────────────┤
    │ < 40          │ 0.6x      │ 0         │ No          │
    │ 40-60         │ 0.8x      │ 1         │ No          │
    │ 60-80         │ 1.0x      │ 2         │ Maybe       │
    │ > 80          │ 1.2x      │ 3         │ Yes         │
    └───────────────┴───────────┴───────────┴─────────────┘
    
    Score Trend Effects:
    - IMPROVING: Boost confidence, allow more pyramid
    - DECLINING: Reduce risk, disable pyramid
    """
    
    # Score thresholds
    CRITICAL_SCORE = 40
    WARNING_SCORE = 60
    GOOD_SCORE = 80
    
    # Target score for competition
    TARGET_SCORE = 75
    
    def __init__(
        self,
        target_score: float = 75,
        score_history_size: int = 10
    ):
        """
        Initialize Score Optimizer.
        
        Args:
            target_score: Target competition score
            score_history_size: History size for trend detection
        """
        self.target_score = target_score
        self.score_history: list[float] = []
        self.score_history_size = score_history_size
        
        logger.info(
            f"ScoreOptimizer initialized: target={target_score}"
        )
    
    def optimize(self, current_score: float) -> OptimizationResult:
        """
        Get optimization parameters based on current score.
        
        Args:
            current_score: Current LiveScore (0-100)
            
        Returns:
            OptimizationResult with caps and recommendations
        """
        # Update history
        self.score_history.append(current_score)
        if len(self.score_history) > self.score_history_size:
            self.score_history.pop(0)
        
        # Detect trend
        trend = self._detect_trend()
        
        # Calculate base parameters from score
        risk_cap, max_pyramid, allow_alpha, reason = self._calc_params(current_score)
        
        # Adjust by trend
        confidence_boost = 1.0
        
        if trend == ScoreTrend.IMPROVING:
            confidence_boost = 1.1
            max_pyramid = min(max_pyramid + 1, 3)
            reason += " [IMPROVING +boost]"
        elif trend == ScoreTrend.DECLINING:
            confidence_boost = 0.9
            risk_cap *= 0.9
            max_pyramid = max(max_pyramid - 1, 0)
            allow_alpha = False
            reason += " [DECLINING -risk]"
        
        return OptimizationResult(
            risk_cap=round(risk_cap, 2),
            max_pyramid=max_pyramid,
            confidence_boost=round(confidence_boost, 2),
            allow_alpha=allow_alpha,
            reason=reason,
            trend=trend,
            score=current_score,
            target_score=self.target_score
        )
    
    def _calc_params(
        self, score: float
    ) -> tuple[float, int, bool, str]:
        """Calculate base parameters from score."""
        
        if score < self.CRITICAL_SCORE:
            return 0.6, 0, False, f"Critical ({score:.1f} < {self.CRITICAL_SCORE})"
        
        if score < self.WARNING_SCORE:
            return 0.8, 1, False, f"Warning ({score:.1f} < {self.WARNING_SCORE})"
        
        if score < self.GOOD_SCORE:
            return 1.0, 2, score > 70, f"Normal ({score:.1f})"
        
        # score >= GOOD_SCORE
        return 1.2, 3, True, f"Strong ({score:.1f} >= {self.GOOD_SCORE})"
    
    def _detect_trend(self) -> ScoreTrend:
        """Detect score trend from history."""
        if len(self.score_history) < 3:
            return ScoreTrend.STABLE
        
        recent = self.score_history[-3:]
        
        # Check if consistently improving
        if recent[-1] > recent[-2] > recent[-3]:
            return ScoreTrend.IMPROVING
        
        # Check if consistently declining
        if recent[-1] < recent[-2] < recent[-3]:
            return ScoreTrend.DECLINING
        
        return ScoreTrend.STABLE
    
    def get_distance_to_target(self) -> float:
        """Get distance from current to target score."""
        if not self.score_history:
            return self.target_score
        return self.target_score - self.score_history[-1]
    
    def should_be_aggressive(self) -> bool:
        """Check if we should be aggressive (ahead of target)."""
        if not self.score_history:
            return False
        return self.score_history[-1] > self.target_score
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "current_score": self.score_history[-1] if self.score_history else 0,
            "target_score": self.target_score,
            "distance_to_target": self.get_distance_to_target(),
            "trend": self._detect_trend().value,
            "history_size": len(self.score_history),
            "should_be_aggressive": self.should_be_aggressive(),
        }


# Singleton instance
_optimizer: Optional[ScoreOptimizer] = None


def get_score_optimizer() -> ScoreOptimizer:
    """Get or create singleton ScoreOptimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ScoreOptimizer()
    return _optimizer


# =============================================================================
# INTEGRATED OPTIMIZATION (Combining all components)
# =============================================================================

@dataclass
class FullOptimizationResult:
    """Full optimization result with all parameters."""
    # From ScoreOptimizer
    score_risk_cap: float
    score_max_pyramid: int
    score_allow_alpha: bool
    score_trend: str
    
    # Final computed values
    final_risk_cap: float
    final_max_pyramid: int
    final_allow_alpha: bool
    
    # Context
    current_score: float
    regime: str
    mode: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": {
                "risk_cap": self.score_risk_cap,
                "max_pyramid": self.score_max_pyramid,
                "allow_alpha": self.score_allow_alpha,
                "trend": self.score_trend,
            },
            "final": {
                "risk_cap": self.final_risk_cap,
                "max_pyramid": self.final_max_pyramid,
                "allow_alpha": self.final_allow_alpha,
            },
            "context": {
                "score": self.current_score,
                "regime": self.regime,
                "mode": self.mode,
            }
        }


def optimize_full_stack(
    current_score: float,
    regime: str = "WEAK_TREND",
    current_mode: str = "NEUTRAL"
) -> FullOptimizationResult:
    """
    Run full optimization across all components.
    
    Args:
        current_score: Current LiveScore
        regime: Current market regime
        current_mode: Current trading mode
        
    Returns:
        FullOptimizationResult with final parameters
    """
    optimizer = get_score_optimizer()
    result = optimizer.optimize(current_score)
    
    # Apply regime effects
    regime_risk_mult = {
        "STRONG_TREND": 1.2,
        "WEAK_TREND": 1.0,
        "CHOP": 0.7,
    }.get(regime, 1.0)
    
    regime_pyramid_mod = {
        "STRONG_TREND": 1,
        "WEAK_TREND": 0,
        "CHOP": -1,
    }.get(regime, 0)
    
    # Calculate final values
    final_risk_cap = result.risk_cap * regime_risk_mult
    final_max_pyramid = max(0, result.max_pyramid + regime_pyramid_mod)
    final_allow_alpha = result.allow_alpha and regime != "CHOP"
    
    return FullOptimizationResult(
        score_risk_cap=result.risk_cap,
        score_max_pyramid=result.max_pyramid,
        score_allow_alpha=result.allow_alpha,
        score_trend=result.trend.value,
        final_risk_cap=round(final_risk_cap, 2),
        final_max_pyramid=final_max_pyramid,
        final_allow_alpha=final_allow_alpha,
        current_score=current_score,
        regime=regime,
        mode=current_mode,
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ScoreOptimizer Test")
    print("=" * 60)
    
    optimizer = ScoreOptimizer(target_score=75)
    
    # Test various scores
    test_scores = [30, 45, 55, 65, 75, 85]
    
    print("\n--- Score Optimization ---")
    for score in test_scores:
        result = optimizer.optimize(score)
        print(
            f"Score {score:3d}: RiskCap={result.risk_cap}x, "
            f"Pyramid={result.max_pyramid}, Alpha={result.allow_alpha}, "
            f"{result.reason}"
        )
    
    print(f"\n--- Trend after history: {optimizer._detect_trend().value} ---")
    
    print("\n--- Full Stack Optimization ---")
    full = optimize_full_stack(65, "STRONG_TREND", "NEUTRAL")
    print(f"Score: {full.current_score}, Regime: {full.regime}")
    print(f"Final: RiskCap={full.final_risk_cap}x, Pyramid={full.final_max_pyramid}, Alpha={full.final_allow_alpha}")
