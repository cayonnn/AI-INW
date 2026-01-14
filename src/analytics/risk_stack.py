# src/analytics/risk_stack.py
"""
Effective Risk Stack - Central Risk Aggregator
===============================================

Single source of truth for all risk calculations.
Combines all risk modifiers into one effective risk.

Components:
- Base Risk (from profile)
- Win Streak Multiplier
- AI Confidence Multiplier
- Mode Multiplier (ALPHA/NEUTRAL/DEFENSIVE)
- Score Cap Multiplier
- Pyramid Multiplier

Dashboard-ready output showing exactly what the system uses.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger("RISK_STACK")


@dataclass
class RiskStackSnapshot:
    """Complete risk stack snapshot for dashboard."""
    # Inputs
    base_risk: float
    win_streak_mult: float
    confidence_mult: float
    mode_mult: float
    score_cap_mult: float
    pyramid_mult: float
    
    # Outputs
    effective_risk: float
    max_pyramid: int
    
    # Context
    mode_name: str
    score: float
    drawdown: float
    confidence: float
    streak_level: int
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/dashboard."""
        return {
            "base_risk": self.base_risk,
            "multipliers": {
                "win_streak": self.win_streak_mult,
                "confidence": self.confidence_mult,
                "mode": self.mode_mult,
                "score_cap": self.score_cap_mult,
                "pyramid": self.pyramid_mult,
            },
            "effective_risk": self.effective_risk,
            "max_pyramid": self.max_pyramid,
            "context": {
                "mode": self.mode_name,
                "score": self.score,
                "drawdown": self.drawdown,
                "confidence": self.confidence,
                "streak_level": self.streak_level,
            },
            "timestamp": self.timestamp.isoformat(),
        }


class EffectiveRiskStack:
    """
    Effective Risk Stack Calculator.
    
    Formula:
    effective_risk = base × streak × confidence × mode × score_cap × pyramid
    
    All modules read from this, none set their own risk.
    """
    
    # Maximum effective risk cap (safety)
    MAX_RISK_CAP = 5.0  # 5% max per trade
    
    def __init__(self, max_risk_cap: float = 5.0):
        """
        Initialize Risk Stack.
        
        Args:
            max_risk_cap: Maximum effective risk percentage
        """
        self.max_risk_cap = max_risk_cap
        self.last_snapshot: Optional[RiskStackSnapshot] = None
        
        logger.info(f"EffectiveRiskStack initialized: max_cap={max_risk_cap}%")
    
    def calculate(
        self,
        base_risk: float,
        win_streak_mult: float = 1.0,
        confidence_mult: float = 1.0,
        mode_mult: float = 1.0,
        score_cap_mult: float = 1.0,
        pyramid_mult: float = 1.0,
        max_pyramid: int = 3,
        mode_name: str = "NEUTRAL",
        score: float = 50.0,
        drawdown: float = 0.0,
        confidence: float = 0.5,
        streak_level: int = 0
    ) -> RiskStackSnapshot:
        """
        Calculate effective risk from all components.
        
        Args:
            base_risk: Base risk percentage
            win_streak_mult: Win streak multiplier
            confidence_mult: AI confidence multiplier
            mode_mult: Trading mode multiplier
            score_cap_mult: Score cap multiplier
            pyramid_mult: Pyramid entry multiplier
            max_pyramid: Maximum pyramid entries allowed
            mode_name: Current mode name
            score: Current LiveScore
            drawdown: Current drawdown %
            confidence: AI confidence
            streak_level: Current streak level
            
        Returns:
            RiskStackSnapshot with all details
        """
        # Calculate raw effective risk
        raw_risk = (
            base_risk
            * win_streak_mult
            * confidence_mult
            * mode_mult
            * score_cap_mult
            * pyramid_mult
        )
        
        # Apply safety cap
        effective_risk = min(raw_risk, self.max_risk_cap)
        
        # Create snapshot
        snapshot = RiskStackSnapshot(
            base_risk=base_risk,
            win_streak_mult=win_streak_mult,
            confidence_mult=confidence_mult,
            mode_mult=mode_mult,
            score_cap_mult=score_cap_mult,
            pyramid_mult=pyramid_mult,
            effective_risk=round(effective_risk, 4),
            max_pyramid=max_pyramid,
            mode_name=mode_name,
            score=score,
            drawdown=drawdown,
            confidence=confidence,
            streak_level=streak_level
        )
        
        self.last_snapshot = snapshot
        
        return snapshot
    
    def get_last_snapshot(self) -> Optional[RiskStackSnapshot]:
        """Get last calculated snapshot."""
        return self.last_snapshot
    
    def get_display_string(self, snapshot: Optional[RiskStackSnapshot] = None) -> str:
        """
        Get formatted display string for logging/dashboard.
        
        Example:
        MODE: ALPHA
        --------------------------------
        Base Risk        : 2.0%
        Win Streak       : x1.3
        AI Confidence    : x1.2
        Mode Multiplier  : x1.3
        Score Cap        : x0.8
        --------------------------------
        Effective Risk   : 4.05%
        Max Pyramid      : 2
        """
        s = snapshot or self.last_snapshot
        if not s:
            return "No risk stack calculated yet"
        
        lines = [
            f"MODE: {s.mode_name}",
            "-" * 32,
            f"Base Risk        : {s.base_risk:.1f}%",
            f"Win Streak       : x{s.win_streak_mult:.2f}",
            f"AI Confidence    : x{s.confidence_mult:.2f}",
            f"Mode Multiplier  : x{s.mode_mult:.2f}",
            f"Score Cap        : x{s.score_cap_mult:.2f}",
            f"Pyramid          : x{s.pyramid_mult:.2f}",
            "-" * 32,
            f"Effective Risk   : {s.effective_risk:.2f}%",
            f"Max Pyramid      : {s.max_pyramid}",
        ]
        
        return "\n".join(lines)


# Singleton instance
_stack: Optional[EffectiveRiskStack] = None


def get_effective_risk_stack() -> EffectiveRiskStack:
    """Get or create singleton EffectiveRiskStack."""
    global _stack
    if _stack is None:
        _stack = EffectiveRiskStack()
    return _stack


def get_risk_stack_snapshot() -> Optional[Dict[str, Any]]:
    """Get last risk stack snapshot as dictionary for API."""
    stack = get_effective_risk_stack()
    snapshot = stack.get_last_snapshot()
    return snapshot.to_dict() if snapshot else None


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EffectiveRiskStack Test")
    print("=" * 60)
    
    stack = EffectiveRiskStack()
    
    # Test calculation
    snapshot = stack.calculate(
        base_risk=2.0,
        win_streak_mult=1.3,
        confidence_mult=1.2,
        mode_mult=1.3,
        score_cap_mult=0.8,
        pyramid_mult=1.0,
        max_pyramid=2,
        mode_name="ALPHA",
        score=75,
        drawdown=3.5,
        confidence=0.72,
        streak_level=2
    )
    
    print("\n--- Risk Stack Display ---")
    print(stack.get_display_string())
    
    print("\n--- API Output ---")
    import json
    print(json.dumps(snapshot.to_dict(), indent=2, default=str))
