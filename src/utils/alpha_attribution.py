"""
alpha_attribution.py
======================
Alpha Attribution Engine (AAE)

แยกให้ชัด: กำไรที่ได้ = Alpha จริง หรือ แค่ตลาดพาไป

Core Philosophy:
- Alpha ≠ PnL
- Strategy ที่กำไร ≠ Strategy ที่ควรเพิ่มทุน
- Alpha ต้อง อธิบายได้ และ ทำซ้ำได้
- สิ่งที่อธิบายไม่ได้ → เสี่ยงสูง

กำไรที่อธิบายไม่ได้ = ความเสี่ยงที่มองไม่เห็น
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
from src.utils.logger import get_logger

logger = get_logger("ALPHA_ATTRIBUTION")


class MarketFactor(str, Enum):
    """Common market factors for decomposition."""
    MARKET = "MARKET"           # Beta
    TREND = "TREND"             # Trend factor
    VOLATILITY = "VOLATILITY"   # Vol factor
    MOMENTUM = "MOMENTUM"       # Momentum factor
    CARRY = "CARRY"             # Carry/yield factor
    LIQUIDITY = "LIQUIDITY"     # Liquidity factor
    TIME_OF_DAY = "TIME_OF_DAY" # Intraday patterns


class AlphaStatus(str, Enum):
    """Alpha health status."""
    STRONG = "STRONG"           # High confidence, stable
    STABLE = "STABLE"           # Acceptable
    DECAYING = "DECAYING"       # Half-life shortening
    WEAK = "WEAK"               # Low confidence
    CROWDED = "CROWDED"         # Factor overlap


@dataclass
class TradeRecord:
    """Single trade record for attribution."""
    trade_id: str
    strategy: str
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    regime: str = "UNKNOWN"


@dataclass
class FactorExposure:
    """Exposure to a market factor."""
    factor: MarketFactor
    beta: float               # Factor loading
    contribution: float       # PnL contribution from this factor
    r_squared: float          # Explanatory power


@dataclass
class AlphaResult:
    """Result of alpha attribution analysis."""
    strategy: str
    total_return: float
    alpha: float              # Pure alpha after factor removal
    alpha_pct: float          # Alpha as percentage of total
    factor_exposures: List[FactorExposure]
    unexplained: float        # Residual
    alpha_confidence: float   # 0-1 confidence score
    half_life_days: float     # Alpha persistence
    status: AlphaStatus
    regime_breakdown: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AttributionMetrics:
    """Metrics for alpha scoring."""
    persistence: float = 0.5          # Alpha stability over time
    regime_consistency: float = 0.5   # Consistent across regimes
    factor_independence: float = 0.5  # Not explained by factors
    execution_quality: float = 0.5    # Good fills, low slippage


class AlphaAttributionEngine:
    """
    Alpha Attribution Engine.
    
    แยก: กำไรจาก Alpha จริง vs ตลาดพาไป
    """

    def __init__(self):
        self.history: List[AlphaResult] = []
        self.factor_betas: Dict[str, Dict[MarketFactor, float]] = {}
        
        # Thresholds
        self.min_alpha_confidence = 0.3
        self.decay_half_life_threshold = 10  # days
        self.factor_threshold = 0.7  # R-squared for factor crowding

    # -------------------------------------------------
    # Main analysis
    # -------------------------------------------------
    def analyze(self, trades: List[TradeRecord], 
                market_factors: Dict[MarketFactor, List[float]]) -> Dict[str, AlphaResult]:
        """
        Analyze trades and decompose returns into alpha vs factors.
        
        Args:
            trades: List of trade records
            market_factors: Time series of factor returns
            
        Returns:
            Alpha results by strategy
        """
        if not trades:
            return {}
        
        # Group by strategy
        by_strategy = self._group_by_strategy(trades)
        
        results = {}
        for strategy, strategy_trades in by_strategy.items():
            # Normalize returns
            returns = self._normalize_returns(strategy_trades)
            
            # Factor decomposition
            exposures = self._decompose_factors(returns, market_factors)
            
            # Extract alpha
            total_return = sum(t.pnl_pct for t in strategy_trades)
            factor_return = sum(e.contribution for e in exposures)
            alpha = total_return - factor_return
            
            # Calculate metrics
            metrics = self._calculate_metrics(strategy, strategy_trades, exposures)
            
            # Alpha confidence
            confidence = self._calculate_confidence(metrics)
            
            # Persistence (half-life)
            half_life = self._estimate_half_life(strategy)
            
            # Status
            status = self._determine_status(confidence, half_life, exposures)
            
            # Regime breakdown
            regime_breakdown = self._breakdown_by_regime(strategy_trades)
            
            result = AlphaResult(
                strategy=strategy,
                total_return=total_return,
                alpha=alpha,
                alpha_pct=alpha / total_return if total_return != 0 else 0,
                factor_exposures=exposures,
                unexplained=alpha,
                alpha_confidence=confidence,
                half_life_days=half_life,
                status=status,
                regime_breakdown=regime_breakdown,
            )
            
            results[strategy] = result
            self.history.append(result)
            
            logger.info(f"{strategy}: Alpha={alpha:.4f} ({result.alpha_pct:.1%}), "
                       f"Confidence={confidence:.2f}, Status={status.value}")
        
        return results

    # -------------------------------------------------
    # Factor Decomposition
    # -------------------------------------------------
    def _decompose_factors(self, returns: List[float], 
                          factors: Dict[MarketFactor, List[float]]) -> List[FactorExposure]:
        """
        Decompose returns into factor contributions.
        
        R = α + β₁·Market + β₂·Trend + β₃·Vol + ε
        """
        exposures = []
        
        for factor, factor_returns in factors.items():
            if len(factor_returns) < len(returns):
                continue
            
            # Simple OLS beta estimation
            beta, r_squared = self._estimate_beta(returns, factor_returns[:len(returns)])
            contribution = beta * sum(factor_returns[:len(returns)])
            
            exposures.append(FactorExposure(
                factor=factor,
                beta=beta,
                contribution=contribution,
                r_squared=r_squared,
            ))
        
        return exposures

    def _estimate_beta(self, y: List[float], x: List[float]) -> Tuple[float, float]:
        """Estimate beta and R-squared using simple regression."""
        n = len(y)
        if n < 2:
            return 0.0, 0.0
        
        # Means
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Covariance and variance
        cov_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        var_x = sum((xi - mean_x) ** 2 for xi in x) / n
        var_y = sum((yi - mean_y) ** 2 for yi in y) / n
        
        if var_x == 0:
            return 0.0, 0.0
        
        beta = cov_xy / var_x
        
        # R-squared
        if var_y == 0:
            r_squared = 0.0
        else:
            correlation = cov_xy / (math.sqrt(var_x * var_y) + 1e-10)
            r_squared = correlation ** 2
        
        return beta, r_squared

    # -------------------------------------------------
    # Metrics & Confidence
    # -------------------------------------------------
    def _calculate_metrics(self, strategy: str, trades: List[TradeRecord],
                          exposures: List[FactorExposure]) -> AttributionMetrics:
        """Calculate attribution metrics for confidence scoring."""
        # Persistence: based on history
        persistence = self._calculate_persistence(strategy)
        
        # Regime consistency
        regime_consistency = self._calculate_regime_consistency(trades)
        
        # Factor independence (less factor exposure = more independent)
        total_factor_r2 = sum(e.r_squared for e in exposures)
        factor_independence = max(0, 1 - total_factor_r2 / max(len(exposures), 1))
        
        # Execution quality (placeholder - would need execution data)
        execution_quality = 0.7
        
        return AttributionMetrics(
            persistence=persistence,
            regime_consistency=regime_consistency,
            factor_independence=factor_independence,
            execution_quality=execution_quality,
        )

    def _calculate_confidence(self, metrics: AttributionMetrics) -> float:
        """
        Calculate alpha confidence score.
        
        alpha_confidence = (
            persistence * 0.4 +
            regime_consistency * 0.3 +
            factor_independence * 0.2 +
            execution_quality * 0.1
        )
        """
        return (
            metrics.persistence * 0.4 +
            metrics.regime_consistency * 0.3 +
            metrics.factor_independence * 0.2 +
            metrics.execution_quality * 0.1
        )

    def _calculate_persistence(self, strategy: str) -> float:
        """Calculate alpha persistence from history."""
        strategy_history = [h for h in self.history if h.strategy == strategy]
        
        if len(strategy_history) < 5:
            return 0.5  # Not enough data
        
        # Check if alpha stays positive
        positive_count = sum(1 for h in strategy_history[-10:] if h.alpha > 0)
        return positive_count / min(10, len(strategy_history))

    def _calculate_regime_consistency(self, trades: List[TradeRecord]) -> float:
        """Calculate consistency across regimes."""
        by_regime = self._breakdown_by_regime(trades)
        
        if len(by_regime) <= 1:
            return 0.5
        
        # Check if profitable in multiple regimes
        positive_regimes = sum(1 for pnl in by_regime.values() if pnl > 0)
        return positive_regimes / len(by_regime)

    def _estimate_half_life(self, strategy: str) -> float:
        """Estimate alpha half-life in days."""
        strategy_history = [h for h in self.history if h.strategy == strategy]
        
        if len(strategy_history) < 10:
            return 30.0  # Default assumption
        
        # Simple decay estimation based on alpha trend
        recent = [h.alpha for h in strategy_history[-20:]]
        if len(recent) < 5:
            return 30.0
        
        # Calculate decay rate
        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        
        if first_half <= 0:
            return 30.0
        
        decay_rate = (first_half - second_half) / first_half
        
        if decay_rate <= 0:
            return 60.0  # No decay
        
        # Convert to half-life
        half_life = 0.693 / decay_rate if decay_rate > 0.01 else 60.0
        return max(1.0, min(60.0, half_life))

    def _determine_status(self, confidence: float, half_life: float,
                         exposures: List[FactorExposure]) -> AlphaStatus:
        """Determine alpha status."""
        # Check for crowding
        max_r2 = max((e.r_squared for e in exposures), default=0)
        if max_r2 > self.factor_threshold:
            return AlphaStatus.CROWDED
        
        # Check decay
        if half_life < self.decay_half_life_threshold:
            return AlphaStatus.DECAYING
        
        # Check confidence
        if confidence < self.min_alpha_confidence:
            return AlphaStatus.WEAK
        
        if confidence > 0.7:
            return AlphaStatus.STRONG
        
        return AlphaStatus.STABLE

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _group_by_strategy(self, trades: List[TradeRecord]) -> Dict[str, List[TradeRecord]]:
        """Group trades by strategy."""
        by_strategy = {}
        for trade in trades:
            if trade.strategy not in by_strategy:
                by_strategy[trade.strategy] = []
            by_strategy[trade.strategy].append(trade)
        return by_strategy

    def _normalize_returns(self, trades: List[TradeRecord]) -> List[float]:
        """Normalize returns."""
        return [t.pnl_pct for t in sorted(trades, key=lambda x: x.entry_time)]

    def _breakdown_by_regime(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """Break down PnL by regime."""
        by_regime = {}
        for trade in trades:
            if trade.regime not in by_regime:
                by_regime[trade.regime] = 0.0
            by_regime[trade.regime] += trade.pnl
        return by_regime

    # -------------------------------------------------
    # Kill & Feedback
    # -------------------------------------------------
    def should_freeze_strategy(self, strategy: str) -> bool:
        """Check if strategy should be frozen based on alpha."""
        strategy_results = [h for h in self.history if h.strategy == strategy]
        
        if len(strategy_results) < 5:
            return False
        
        latest = strategy_results[-1]
        
        # Kill criteria
        return (
            latest.alpha_confidence < self.min_alpha_confidence and
            latest.status in [AlphaStatus.DECAYING, AlphaStatus.WEAK]
        )

    def get_allocation_feedback(self, strategy: str) -> Dict:
        """Get feedback for capital allocator."""
        strategy_results = [h for h in self.history if h.strategy == strategy]
        
        if not strategy_results:
            return {"action": "HOLD", "multiplier": 1.0}
        
        latest = strategy_results[-1]
        
        if latest.status == AlphaStatus.STRONG:
            return {"action": "INCREASE", "multiplier": 1.2}
        elif latest.status == AlphaStatus.STABLE:
            return {"action": "HOLD", "multiplier": 1.0}
        elif latest.status == AlphaStatus.DECAYING:
            return {"action": "DECREASE", "multiplier": 0.7}
        elif latest.status in [AlphaStatus.WEAK, AlphaStatus.CROWDED]:
            return {"action": "FREEZE", "multiplier": 0.0}
        
        return {"action": "HOLD", "multiplier": 1.0}

    def get_summary(self) -> Dict:
        """Get attribution summary."""
        if not self.history:
            return {"status": "NO_DATA"}
        
        strategies = set(h.strategy for h in self.history)
        
        return {
            "total_strategies": len(strategies),
            "history_count": len(self.history),
            "latest_by_strategy": {
                s: self.history[-1].status.value
                for s in strategies
                if any(h.strategy == s for h in self.history[-10:])
            },
        }
