# src/ai/confidence_engine.py
"""
AI Confidence-Weighted Risk Engine - Killer Feature
====================================================

Aggregates confidence from multiple AI sources:
- Signal probability (XGB/LSTM)
- SL/TP model variance
- Regime model stability

Risk Scaling:
- High confidence → Increase risk
- Low confidence → Decrease risk
- No hardcoding, pure data-driven

Confidence Sources:
┌─────────────────┬────────────┬────────┐
│ Source          │ Weight     │ Output │
├─────────────────┼────────────┼────────┤
│ Signal P(BUY)   │ 0.50       │ 0-1    │
│ SL/TP Variance  │ 0.30       │ 0-1    │
│ Regime Score    │ 0.20       │ 0-1    │
└─────────────────┴────────────┴────────┘
"""

from dataclasses import dataclass
from typing import Optional
import math

from src.utils.logger import get_logger

logger = get_logger("CONFIDENCE_ENGINE")


@dataclass
class ConfidenceResult:
    """Confidence calculation result."""
    score: float               # Overall confidence 0-1
    signal_conf: float         # Signal probability
    sl_conf: float             # SL/TP confidence (1 - variance)
    regime_conf: float         # Regime stability
    risk_multiplier: float     # Final risk multiplier


class ConfidenceEngine:
    """
    AI Confidence Aggregator.
    
    Combines multiple AI outputs into a single confidence score
    that modulates risk dynamically.
    
    Risk Multiplier Curve:
    ┌────────────┬─────────────┐
    │ Confidence │ Multiplier  │
    ├────────────┼─────────────┤
    │ > 0.80     │ 1.4x        │
    │ > 0.65     │ 1.2x        │
    │ > 0.50     │ 1.0x        │
    │ > 0.35     │ 0.8x        │
    │ < 0.35     │ 0.6x        │
    └────────────┴─────────────┘
    """
    
    # Weights for confidence sources
    WEIGHTS = {
        "signal": 0.50,
        "sl_tp": 0.30,
        "regime": 0.20,
    }
    
    # Risk multiplier thresholds
    RISK_CURVE = [
        (0.80, 1.4),
        (0.65, 1.2),
        (0.50, 1.0),
        (0.35, 0.8),
        (0.00, 0.6),
    ]
    
    def __init__(
        self,
        min_confidence: float = 0.3,   # Minimum to allow trade
        max_multiplier: float = 1.5,   # Cap multiplier
        min_multiplier: float = 0.5,   # Floor multiplier
    ):
        """
        Initialize Confidence Engine.
        
        Args:
            min_confidence: Minimum confidence to allow trade
            max_multiplier: Maximum risk multiplier
            min_multiplier: Minimum risk multiplier
        """
        self.min_confidence = min_confidence
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        
        logger.info(
            f"ConfidenceEngine initialized: "
            f"min_conf={min_confidence}, mult_range=[{min_multiplier}, {max_multiplier}]"
        )
    
    def compute(
        self,
        signal_probability: float = 0.5,
        sl_variance: float = 0.0,
        regime_score: float = 0.5
    ) -> ConfidenceResult:
        """
        Compute aggregated confidence and risk multiplier.
        
        Args:
            signal_probability: P(signal direction) 0-1
            sl_variance: SL/TP model variance (lower = more confident)
            regime_score: Market regime stability 0-1
            
        Returns:
            ConfidenceResult with score and multiplier
        """
        # Convert variance to confidence (inverse)
        sl_confidence = 1.0 - min(sl_variance, 1.0)
        
        # Weighted aggregation
        score = (
            signal_probability * self.WEIGHTS["signal"]
            + sl_confidence * self.WEIGHTS["sl_tp"]
            + regime_score * self.WEIGHTS["regime"]
        )
        
        # Get risk multiplier from curve
        multiplier = self._get_risk_multiplier(score)
        
        return ConfidenceResult(
            score=round(score, 3),
            signal_conf=signal_probability,
            sl_conf=sl_confidence,
            regime_conf=regime_score,
            risk_multiplier=multiplier
        )
    
    def _get_risk_multiplier(self, confidence: float) -> float:
        """Get risk multiplier from confidence."""
        for threshold, multiplier in self.RISK_CURVE:
            if confidence >= threshold:
                return min(self.max_multiplier, max(self.min_multiplier, multiplier))
        return self.min_multiplier
    
    def should_trade(self, confidence: float) -> bool:
        """Check if confidence is high enough to trade."""
        return confidence >= self.min_confidence
    
    def get_effective_risk(
        self,
        base_risk: float,
        signal_probability: float,
        sl_variance: float = 0.0,
        regime_score: float = 0.5
    ) -> tuple[float, ConfidenceResult]:
        """
        Calculate effective risk with confidence scaling.
        
        Args:
            base_risk: Base risk percentage
            signal_probability: Signal confidence
            sl_variance: SL/TP variance
            regime_score: Regime stability
            
        Returns:
            (effective_risk, confidence_result)
        """
        result = self.compute(signal_probability, sl_variance, regime_score)
        effective_risk = base_risk * result.risk_multiplier
        
        return round(effective_risk, 4), result


# Singleton instance
_engine: Optional[ConfidenceEngine] = None


def get_confidence_engine() -> ConfidenceEngine:
    """Get or create singleton ConfidenceEngine."""
    global _engine
    if _engine is None:
        _engine = ConfidenceEngine()
    return _engine


def confidence_risk_multiplier(confidence: float) -> float:
    """
    Quick confidence to risk multiplier conversion.
    
    Args:
        confidence: Confidence score 0-1
        
    Returns:
        Risk multiplier (0.6 - 1.4)
    """
    if confidence > 0.80:
        return 1.4
    elif confidence > 0.65:
        return 1.2
    elif confidence > 0.50:
        return 1.0
    elif confidence > 0.35:
        return 0.8
    else:
        return 0.6


# =============================================================================
# FULL RISK STACK (Combined)
# =============================================================================

@dataclass
class EffectiveRiskResult:
    """Full effective risk calculation result."""
    base_risk: float
    win_streak_mult: float
    confidence_mult: float
    pyramid_mult: float
    effective_risk: float
    components: dict


def calculate_effective_risk(
    base_risk: float,
    win_streak_mult: float = 1.0,
    confidence: float = 0.5,
    pyramid_mult: float = 1.0,
    capital_mult: float = 1.0,
) -> EffectiveRiskResult:
    """
    Calculate full effective risk from all components.
    
    Formula:
    effective = base × win_streak × confidence × pyramid × capital
    
    Example:
    base=0.5, streak=1.4, conf=1.3, pyramid=0.7
    effective = 0.5 × 1.4 × 1.3 × 0.7 = 0.64%
    """
    conf_mult = confidence_risk_multiplier(confidence)
    
    effective = base_risk * win_streak_mult * conf_mult * pyramid_mult * capital_mult
    
    # Cap effective risk
    effective = min(effective, 3.0)  # Max 3% per trade
    
    return EffectiveRiskResult(
        base_risk=base_risk,
        win_streak_mult=win_streak_mult,
        confidence_mult=conf_mult,
        pyramid_mult=pyramid_mult,
        effective_risk=round(effective, 4),
        components={
            "base": base_risk,
            "win_streak": win_streak_mult,
            "confidence": conf_mult,
            "pyramid": pyramid_mult,
            "capital": capital_mult,
        }
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ConfidenceEngine Test")
    print("=" * 60)
    
    engine = ConfidenceEngine()
    
    # Test various confidence scenarios
    scenarios = [
        {"signal_p": 0.85, "sl_var": 0.1, "regime": 0.9},  # High confidence
        {"signal_p": 0.70, "sl_var": 0.2, "regime": 0.7},  # Medium-high
        {"signal_p": 0.55, "sl_var": 0.3, "regime": 0.5},  # Medium
        {"signal_p": 0.40, "sl_var": 0.5, "regime": 0.4},  # Low
        {"signal_p": 0.30, "sl_var": 0.7, "regime": 0.3},  # Very low
    ]
    
    print("\n--- Confidence Calculations ---")
    for s in scenarios:
        result = engine.compute(s["signal_p"], s["sl_var"], s["regime"])
        print(
            f"Signal={s['signal_p']:.2f}, SL_var={s['sl_var']:.2f}, Regime={s['regime']:.2f} "
            f"→ Conf={result.score:.2f}, Mult={result.risk_multiplier}"
        )
    
    print("\n--- Full Risk Stack ---")
    full_result = calculate_effective_risk(
        base_risk=0.5,
        win_streak_mult=1.4,
        confidence=0.75,
        pyramid_mult=0.7,
        capital_mult=1.0
    )
    
    print(f"Base Risk:        {full_result.base_risk}%")
    print(f"Win Streak Mult:  x{full_result.win_streak_mult}")
    print(f"Confidence Mult:  x{full_result.confidence_mult}")
    print(f"Pyramid Mult:     x{full_result.pyramid_mult}")
    print(f"------------------------")
    print(f"Effective Risk:   {full_result.effective_risk}%")
