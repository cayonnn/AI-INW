"""
crowding_detection.py
======================
Crowding Detection - Global Alpha Risk

ป้องกัน "แพ้พร้อมตลาดทั้งโลก"

Alpha ตายไม่ใช่เพราะผิด
แต่เพราะ คนใช้เยอะเกิน

Detection Signals:
- Factor Correlation Spike
- Volatility Asymmetry
- Liquidity Slippage
- Time-of-Day Clustering
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from src.utils.logger import get_logger

logger = get_logger("CROWDING_DETECTION")


class CrowdingLevel(str, Enum):
    """Crowding risk levels."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class CrowdingMetrics:
    """Metrics for crowding detection."""
    # Factor overlap
    factor_correlation: float = 0.0      # How much alpha correlates with market factors
    factor_r_squared: float = 0.0        # Explained by common factors
    
    # Liquidity pressure
    avg_slippage: float = 0.0            # Average slippage in pips
    expected_slippage: float = 0.0       # Expected slippage
    fill_rate: float = 1.0               # Order fill rate
    bid_ask_spread: float = 0.0          # Average spread
    
    # Execution degradation
    execution_time_ms: float = 0.0       # Avg execution time
    partial_fills_pct: float = 0.0       # Partial fill percentage
    
    # Volatility asymmetry
    entry_volatility: float = 0.0        # Vol when entering
    exit_volatility: float = 0.0         # Vol when exiting (usually higher in crowding)
    
    # Timing clustering
    entry_hour_concentration: float = 0.0  # How concentrated are entry times
    exit_hour_concentration: float = 0.0   # How concentrated are exit times


@dataclass
class CrowdingResult:
    """Result of crowding analysis."""
    strategy: str
    crowding_score: float               # 0-1, higher = more crowded
    level: CrowdingLevel
    primary_factors: List[str]          # Top contributing factors
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class CrowdingDetector:
    """
    Crowding Detection Engine.
    
    Detects when strategies are becoming crowded globally.
    """

    def __init__(self):
        self.history: Dict[str, List[CrowdingResult]] = {}
        
        # Thresholds
        self.thresholds = {
            "factor_correlation_high": 0.7,
            "slippage_multiplier": 2.0,  # x times expected
            "fill_rate_low": 0.9,
            "vol_asymmetry": 1.5,       # Exit vol / Entry vol
            "clustering_high": 0.7,
        }
        
        # Weights for scoring
        self.weights = {
            "factor_overlap": 0.4,
            "liquidity_pressure": 0.3,
            "execution_degradation": 0.2,
            "volatility_skew": 0.1,
        }

    # -------------------------------------------------
    # Main detection
    # -------------------------------------------------
    def detect(self, strategy: str, metrics: CrowdingMetrics) -> CrowdingResult:
        """
        Detect crowding for a strategy.
        
        Args:
            strategy: Strategy name
            metrics: Current crowding metrics
            
        Returns:
            CrowdingResult with score and recommendations
        """
        # Calculate component scores
        factor_score = self._score_factor_overlap(metrics)
        liquidity_score = self._score_liquidity_pressure(metrics)
        execution_score = self._score_execution_degradation(metrics)
        vol_score = self._score_volatility_skew(metrics)
        
        # Combined crowding score
        crowding_score = (
            factor_score * self.weights["factor_overlap"] +
            liquidity_score * self.weights["liquidity_pressure"] +
            execution_score * self.weights["execution_degradation"] +
            vol_score * self.weights["volatility_skew"]
        )
        
        # Determine level
        level = self._determine_level(crowding_score)
        
        # Identify primary factors
        factors = self._identify_factors(metrics, factor_score, liquidity_score, 
                                         execution_score, vol_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(level, factors)
        
        result = CrowdingResult(
            strategy=strategy,
            crowding_score=crowding_score,
            level=level,
            primary_factors=factors,
            recommendations=recommendations,
        )
        
        # Track history
        if strategy not in self.history:
            self.history[strategy] = []
        self.history[strategy].append(result)
        
        if level in [CrowdingLevel.HIGH, CrowdingLevel.CRITICAL]:
            logger.warning(f"{strategy}: Crowding={crowding_score:.2f}, Level={level.value}")
        
        return result

    # -------------------------------------------------
    # Component scoring
    # -------------------------------------------------
    def _score_factor_overlap(self, m: CrowdingMetrics) -> float:
        """Score factor overlap (0-1)."""
        # High correlation with market factors = crowded
        corr_score = min(1.0, m.factor_correlation / self.thresholds["factor_correlation_high"])
        r2_score = m.factor_r_squared
        
        return (corr_score * 0.6 + r2_score * 0.4)

    def _score_liquidity_pressure(self, m: CrowdingMetrics) -> float:
        """Score liquidity pressure (0-1)."""
        # Slippage pressure
        if m.expected_slippage > 0:
            slip_ratio = m.avg_slippage / m.expected_slippage
            slip_score = min(1.0, (slip_ratio - 1) / (self.thresholds["slippage_multiplier"] - 1))
        else:
            slip_score = 0.0
        
        # Fill rate pressure
        fill_score = max(0, 1 - (m.fill_rate / self.thresholds["fill_rate_low"]))
        
        return max(slip_score, fill_score)

    def _score_execution_degradation(self, m: CrowdingMetrics) -> float:
        """Score execution quality degradation (0-1)."""
        partial_score = m.partial_fills_pct
        time_score = min(1.0, m.execution_time_ms / 500)  # 500ms as high threshold
        
        return (partial_score * 0.6 + time_score * 0.4)

    def _score_volatility_skew(self, m: CrowdingMetrics) -> float:
        """Score volatility asymmetry (0-1)."""
        if m.entry_volatility > 0:
            vol_ratio = m.exit_volatility / m.entry_volatility
            if vol_ratio > self.thresholds["vol_asymmetry"]:
                return min(1.0, (vol_ratio - 1) / 2)
        return 0.0

    # -------------------------------------------------
    # Level determination
    # -------------------------------------------------
    def _determine_level(self, score: float) -> CrowdingLevel:
        """Determine crowding level from score."""
        if score >= 0.8:
            return CrowdingLevel.CRITICAL
        elif score >= 0.6:
            return CrowdingLevel.HIGH
        elif score >= 0.4:
            return CrowdingLevel.MODERATE
        return CrowdingLevel.LOW

    def _identify_factors(self, m: CrowdingMetrics, 
                         factor: float, liquidity: float,
                         execution: float, vol: float) -> List[str]:
        """Identify primary crowding factors."""
        factors = []
        
        if factor > 0.5:
            factors.append("FACTOR_OVERLAP")
        if liquidity > 0.5:
            factors.append("LIQUIDITY_PRESSURE")
        if execution > 0.5:
            factors.append("EXECUTION_DEGRADATION")
        if vol > 0.5:
            factors.append("VOLATILITY_ASYMMETRY")
        if m.entry_hour_concentration > self.thresholds["clustering_high"]:
            factors.append("TIME_CLUSTERING")
        
        return factors or ["NONE"]

    # -------------------------------------------------
    # Recommendations
    # -------------------------------------------------
    def _generate_recommendations(self, level: CrowdingLevel, 
                                  factors: List[str]) -> List[str]:
        """Generate action recommendations."""
        recommendations = []
        
        if level == CrowdingLevel.CRITICAL:
            recommendations.append("🚨 REDUCE_EXPOSURE_IMMEDIATELY")
            recommendations.append("WIDEN_EXIT_LOGIC")
            recommendations.append("CONSIDER_STRATEGY_PAUSE")
        
        elif level == CrowdingLevel.HIGH:
            recommendations.append("⚠️ REDUCE_POSITION_SIZE")
            recommendations.append("DIVERSIFY_ENTRY_TIMING")
            recommendations.append("MONITOR_SLIPPAGE_CLOSELY")
        
        elif level == CrowdingLevel.MODERATE:
            recommendations.append("MONITOR_LIQUIDITY")
            recommendations.append("CONSIDER_ALTERNATIVE_EXECUTION")
        
        # Factor-specific recommendations
        if "FACTOR_OVERLAP" in factors:
            recommendations.append("REDUCE_FACTOR_EXPOSURE")
        if "LIQUIDITY_PRESSURE" in factors:
            recommendations.append("USE_LIMIT_ORDERS")
        if "TIME_CLUSTERING" in factors:
            recommendations.append("RANDOMIZE_ENTRY_TIMES")
        
        return recommendations

    # -------------------------------------------------
    # Alerts and actions
    # -------------------------------------------------
    def should_reduce_exposure(self, strategy: str) -> Tuple[bool, float]:
        """Check if should reduce exposure and by how much."""
        if strategy not in self.history or not self.history[strategy]:
            return False, 1.0
        
        latest = self.history[strategy][-1]
        
        if latest.level == CrowdingLevel.CRITICAL:
            return True, 0.3  # Reduce to 30%
        elif latest.level == CrowdingLevel.HIGH:
            return True, 0.5  # Reduce to 50%
        
        return False, 1.0

    def get_trend(self, strategy: str) -> Optional[str]:
        """Get crowding trend."""
        history = self.history.get(strategy, [])
        if len(history) < 5:
            return None
        
        recent = sum(h.crowding_score for h in history[-5:]) / 5
        older = sum(h.crowding_score for h in history[-10:-5]) / 5 if len(history) >= 10 else recent
        
        if recent > older + 0.1:
            return "INCREASING"
        elif recent < older - 0.1:
            return "DECREASING"
        return "STABLE"

    def get_status(self) -> Dict:
        """Get detector status."""
        return {
            "strategies_monitored": len(self.history),
            "high_risk_count": sum(
                1 for h in self.history.values()
                if h and h[-1].level in [CrowdingLevel.HIGH, CrowdingLevel.CRITICAL]
            ),
        }
