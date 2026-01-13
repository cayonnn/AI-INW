"""
strategy_decay.py
==================
Strategy Decay Detection

ตรวจว่า "Alpha กำลังตาย" หรือยัง

แยกให้ได้ว่า:
- ขาดทุนเพราะ Drawdown ปกติ
- หรือ Edge หายจริง

Fund จริง ไม่รอให้ขาดทุนหนัก
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import math
from src.utils.logger import get_logger

logger = get_logger("STRATEGY_DECAY")


class DecayStatus(str, Enum):
    """Decay status levels."""
    HEALTHY = "HEALTHY"       # No decay detected
    MONITORING = "MONITORING" # Early warning signs
    DECAYING = "DECAYING"     # Confirmed decay
    CRITICAL = "CRITICAL"     # Urgent action needed


@dataclass
class DecayMetrics:
    """Metrics for decay detection."""
    alpha_half_life: float = 30.0      # Days
    win_rate_stability: float = 1.0    # 1.0 = stable, 0.0 = unstable
    signal_entropy: float = 0.0        # 0 = clear, 1 = random
    regime_sensitivity: float = 0.0    # 0 = universal, 1 = regime-specific
    factor_crowding: float = 0.0       # 0 = unique, 1 = crowded


@dataclass
class DecayResult:
    """Result of decay detection."""
    strategy: str
    decay_score: float                 # 0-1, higher = more decay
    status: DecayStatus
    metrics: DecayMetrics
    consecutive_decay_windows: int
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)


class StrategyDecayDetector:
    """
    Strategy Decay Detection Engine.
    
    ตรวจจับ edge ที่กำลังหายไป ก่อนที่จะขาดทุนหนัก
    """

    def __init__(self):
        self.decay_history: Dict[str, List[float]] = {}
        self.consecutive_windows: Dict[str, int] = {}
        
        # Thresholds
        self.decay_threshold = 0.6          # Score > this = decaying
        self.critical_threshold = 0.8       # Score > this = critical
        self.consecutive_threshold = 3      # Windows before confirmation
        
        # Scoring weights
        self.weights = {
            "alpha_half_life": 0.35,
            "entropy": 0.25,
            "regime": 0.20,
            "crowding": 0.20,
        }

    # -------------------------------------------------
    # Main detection
    # -------------------------------------------------
    def detect(self, strategy: str, metrics: DecayMetrics) -> DecayResult:
        """
        Detect decay for a strategy.
        
        Args:
            strategy: Strategy name
            metrics: Current decay metrics
            
        Returns:
            DecayResult with score and recommendation
        """
        # Calculate decay score
        decay_score = self._calculate_decay_score(metrics)
        
        # Track history
        if strategy not in self.decay_history:
            self.decay_history[strategy] = []
            self.consecutive_windows[strategy] = 0
        
        self.decay_history[strategy].append(decay_score)
        
        # Keep last 20 windows
        if len(self.decay_history[strategy]) > 20:
            self.decay_history[strategy] = self.decay_history[strategy][-20:]
        
        # Track consecutive decay
        if decay_score > self.decay_threshold:
            self.consecutive_windows[strategy] += 1
        else:
            self.consecutive_windows[strategy] = max(0, self.consecutive_windows[strategy] - 1)
        
        # Determine status
        status = self._determine_status(decay_score, self.consecutive_windows[strategy])
        
        # Generate recommendation
        recommendation = self._get_recommendation(status, decay_score)
        
        result = DecayResult(
            strategy=strategy,
            decay_score=decay_score,
            status=status,
            metrics=metrics,
            consecutive_decay_windows=self.consecutive_windows[strategy],
            recommendation=recommendation,
        )
        
        if status != DecayStatus.HEALTHY:
            logger.warning(f"{strategy}: Decay={decay_score:.2f}, Status={status.value}")
        
        return result

    def _calculate_decay_score(self, m: DecayMetrics) -> float:
        """
        Calculate decay score (0-1).
        
        decay_score = (
            alpha_half_life_score * 0.35 +
            entropy_increase * 0.25 +
            regime_breakdown * 0.2 +
            factor_correlation * 0.2
        )
        """
        # Alpha half-life score (shorter = worse)
        half_life_score = max(0, 1 - (m.alpha_half_life / 30))
        
        # Entropy score (higher = worse)
        entropy_score = m.signal_entropy
        
        # Regime sensitivity (higher = worse - means only works in some regimes)
        regime_score = m.regime_sensitivity
        
        # Factor crowding (higher = worse)
        crowding_score = m.factor_crowding
        
        # Win rate instability
        stability_penalty = max(0, 1 - m.win_rate_stability)
        
        decay_score = (
            half_life_score * self.weights["alpha_half_life"] +
            entropy_score * self.weights["entropy"] +
            regime_score * self.weights["regime"] +
            crowding_score * self.weights["crowding"] +
            stability_penalty * 0.1  # Bonus penalty
        )
        
        return min(1.0, decay_score)

    def _determine_status(self, score: float, consecutive: int) -> DecayStatus:
        """Determine decay status."""
        if score > self.critical_threshold:
            return DecayStatus.CRITICAL
        
        if score > self.decay_threshold and consecutive >= self.consecutive_threshold:
            return DecayStatus.DECAYING
        
        if score > self.decay_threshold:
            return DecayStatus.MONITORING
        
        return DecayStatus.HEALTHY

    def _get_recommendation(self, status: DecayStatus, score: float) -> str:
        """Get action recommendation."""
        if status == DecayStatus.CRITICAL:
            return "FREEZE_IMMEDIATELY"
        elif status == DecayStatus.DECAYING:
            return "REDUCE_CAPITAL_50%"
        elif status == DecayStatus.MONITORING:
            return "MONITOR_CLOSELY"
        return "NO_ACTION"

    # -------------------------------------------------
    # Batch detection
    # -------------------------------------------------
    def detect_all(self, 
                   strategy_metrics: Dict[str, DecayMetrics]) -> Dict[str, DecayResult]:
        """Detect decay for all strategies."""
        results = {}
        for strategy, metrics in strategy_metrics.items():
            results[strategy] = self.detect(strategy, metrics)
        return results

    def get_at_risk_strategies(self) -> List[str]:
        """Get list of strategies at risk."""
        at_risk = []
        for strategy, windows in self.consecutive_windows.items():
            if windows >= self.consecutive_threshold:
                at_risk.append(strategy)
        return at_risk

    def get_decay_trend(self, strategy: str) -> Optional[str]:
        """Get decay trend for a strategy."""
        history = self.decay_history.get(strategy, [])
        if len(history) < 5:
            return None
        
        recent = sum(history[-5:]) / 5
        older = sum(history[-10:-5]) / 5 if len(history) >= 10 else recent
        
        if recent > older + 0.1:
            return "WORSENING"
        elif recent < older - 0.1:
            return "IMPROVING"
        return "STABLE"
