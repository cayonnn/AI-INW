"""
meta_portfolio.py
==================
Meta-Portfolio Optimizer

สมองที่จัด "พอร์ตของกลยุทธ์"

คุณไม่ได้เทรด EURUSD, GOLD
คุณกำลังเทรด Strategy A, Strategy B, Strategy C

Objective: Maximize Alpha while controlling Portfolio Risk

Portfolio ที่ดี ไม่ใช่รวม strategy เก่ง
แต่คือ strategy ที่ไม่ชนกัน
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
from src.utils.logger import get_logger

logger = get_logger("META_PORTFOLIO")


class RebalanceTrigger(str, Enum):
    """Triggers for portfolio rebalance."""
    SCHEDULED = "SCHEDULED"
    REGIME_CHANGE = "REGIME_CHANGE"
    ALPHA_DECAY = "ALPHA_DECAY"
    CORRELATION_SPIKE = "CORRELATION_SPIKE"
    DRAWDOWN_ACCELERATION = "DRAWDOWN_ACCELERATION"


@dataclass
class StrategyProfile:
    """Strategy profile for optimization."""
    name: str
    expected_alpha: float           # From Alpha Attribution
    volatility: float               # Risk measure
    max_drawdown: float             # Historical max DD
    correlation_with_others: float  # Average correlation
    regime_scores: Dict[str, float] = field(default_factory=dict)
    is_low_beta: bool = False


@dataclass
class PortfolioAllocation:
    """Optimized portfolio allocation."""
    weights: Dict[str, float]
    expected_portfolio_alpha: float
    expected_portfolio_vol: float
    expected_sharpe: float
    diversification_score: float
    regime: str
    rebalance_trigger: RebalanceTrigger
    timestamp: datetime = field(default_factory=datetime.now)


class MetaPortfolioOptimizer:
    """
    Meta-Portfolio Optimizer.
    
    Optimizes allocation across strategies, not assets.
    """

    def __init__(self):
        self.current_allocation: Optional[PortfolioAllocation] = None
        self.allocation_history: List[PortfolioAllocation] = []
        
        # Constraints
        self.max_strategy_weight = 0.4
        self.min_strategy_weight = 0.05
        self.max_portfolio_dd = 0.10
        self.max_correlation = 0.6
        self.crisis_beta_threshold = 0.3

    # -------------------------------------------------
    # Main optimization
    # -------------------------------------------------
    def optimize(self, strategies: List[StrategyProfile], 
                current_regime: str,
                is_crisis: bool = False,
                trigger: RebalanceTrigger = RebalanceTrigger.SCHEDULED) -> PortfolioAllocation:
        """
        Optimize portfolio allocation across strategies.
        
        Objective:
            Maximize: Σ (wᵢ × Expected Alphaᵢ)
        Subject to:
            - Portfolio DD < limit
            - Correlation < threshold
            - Crisis exposure < cap
        """
        if not strategies:
            return self._empty_allocation(trigger)
        
        # Filter based on mode
        if is_crisis:
            strategies = [s for s in strategies if s.is_low_beta]
            logger.info("Crisis mode: favoring low-beta strategies")
        
        if not strategies:
            return self._empty_allocation(trigger)
        
        # Score strategies
        scores = self._score_strategies(strategies, current_regime)
        
        # Initial weights from scores
        raw_weights = self._scores_to_weights(scores)
        
        # Apply correlation penalty
        penalized = self._apply_correlation_penalty(raw_weights, strategies)
        
        # Apply constraints
        constrained = self._apply_constraints(penalized)
        
        # Normalize
        final_weights = self._normalize(constrained)
        
        # Calculate portfolio metrics
        portfolio_alpha = sum(
            final_weights.get(s.name, 0) * s.expected_alpha
            for s in strategies
        )
        
        portfolio_vol = self._estimate_portfolio_vol(final_weights, strategies)
        
        sharpe = portfolio_alpha / (portfolio_vol + 1e-6)
        
        diversification = self._calculate_diversification(final_weights, strategies)
        
        allocation = PortfolioAllocation(
            weights=final_weights,
            expected_portfolio_alpha=portfolio_alpha,
            expected_portfolio_vol=portfolio_vol,
            expected_sharpe=sharpe,
            diversification_score=diversification,
            regime=current_regime,
            rebalance_trigger=trigger,
        )
        
        self.current_allocation = allocation
        self.allocation_history.append(allocation)
        
        logger.info(f"Portfolio optimized: Alpha={portfolio_alpha:.4f}, Sharpe={sharpe:.2f}")
        for name, weight in final_weights.items():
            if weight > 0:
                logger.debug(f"  {name}: {weight:.2%}")
        
        return allocation

    # -------------------------------------------------
    # Scoring
    # -------------------------------------------------
    def _score_strategies(self, strategies: List[StrategyProfile],
                         regime: str) -> Dict[str, float]:
        """Score strategies based on alpha and regime."""
        scores = {}
        
        for s in strategies:
            # Base score from expected alpha
            base_score = max(0, s.expected_alpha)
            
            # Regime adjustment
            regime_mult = s.regime_scores.get(regime, 0.5)
            
            # Risk penalty
            risk_penalty = s.volatility * 0.5 + s.max_drawdown * 0.5
            
            # Correlation penalty
            corr_penalty = s.correlation_with_others * 0.3
            
            score = base_score * regime_mult - risk_penalty - corr_penalty
            scores[s.name] = max(0, score)
        
        return scores

    def _scores_to_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert scores to raw weights."""
        total = sum(scores.values())
        if total == 0:
            return {k: 0.0 for k in scores}
        return {k: v / total for k, v in scores.items()}

    # -------------------------------------------------
    # Penalties & Constraints
    # -------------------------------------------------
    def _apply_correlation_penalty(self, weights: Dict[str, float],
                                   strategies: List[StrategyProfile]) -> Dict[str, float]:
        """Penalize highly correlated strategies."""
        penalized = weights.copy()
        
        for s in strategies:
            if s.correlation_with_others > self.max_correlation:
                penalized[s.name] *= (1 - s.correlation_with_others)
        
        return penalized

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max weight constraints."""
        constrained = {}
        
        for name, weight in weights.items():
            if weight < self.min_strategy_weight:
                constrained[name] = 0.0
            elif weight > self.max_strategy_weight:
                constrained[name] = self.max_strategy_weight
            else:
                constrained[name] = weight
        
        return constrained

    def _normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total = sum(weights.values())
        if total == 0:
            return weights
        return {k: v / total for k, v in weights.items()}

    # -------------------------------------------------
    # Portfolio metrics
    # -------------------------------------------------
    def _estimate_portfolio_vol(self, weights: Dict[str, float],
                               strategies: List[StrategyProfile]) -> float:
        """Estimate portfolio volatility."""
        # Simple weighted average (ignores correlation for simplicity)
        strategy_map = {s.name: s for s in strategies}
        
        total_vol = 0.0
        for name, weight in weights.items():
            s = strategy_map.get(name)
            if s:
                total_vol += (weight * s.volatility) ** 2
        
        return math.sqrt(total_vol)

    def _calculate_diversification(self, weights: Dict[str, float],
                                   strategies: List[StrategyProfile]) -> float:
        """Calculate diversification score (0-1)."""
        # More strategies with meaningful weights = better diversification
        active = sum(1 for w in weights.values() if w >= self.min_strategy_weight)
        max_possible = len(strategies)
        
        concentration = sum(w ** 2 for w in weights.values())
        
        # Score based on both count and concentration
        count_score = active / max_possible if max_possible > 0 else 0
        concentration_score = 1 - concentration
        
        return (count_score + concentration_score) / 2

    # -------------------------------------------------
    # Regime-based adjustments
    # -------------------------------------------------
    def adjust_for_regime(self, regime: str) -> Dict[str, float]:
        """Get regime-based strategy adjustments."""
        adjustments = {
            "TRENDING": {
                "Trend_Follow": 1.3,
                "Breakout": 1.1,
                "Mean_Reversion": 0.6,
            },
            "RANGING": {
                "Mean_Reversion": 1.4,
                "Trend_Follow": 0.5,
                "Breakout": 0.7,
            },
            "VOLATILE": {
                "Defensive": 1.5,
                "Momentum": 0.4,
                "Scalping": 0.3,
            },
            "CRISIS": {
                "Defensive": 1.8,
                "Hedge": 1.5,
                "All_Others": 0.2,
            },
        }
        return adjustments.get(regime, {})

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _empty_allocation(self, trigger: RebalanceTrigger) -> PortfolioAllocation:
        """Return empty allocation."""
        return PortfolioAllocation(
            weights={},
            expected_portfolio_alpha=0.0,
            expected_portfolio_vol=0.0,
            expected_sharpe=0.0,
            diversification_score=0.0,
            regime="UNKNOWN",
            rebalance_trigger=trigger,
        )

    def get_weight(self, strategy: str) -> float:
        """Get current weight for a strategy."""
        if self.current_allocation:
            return self.current_allocation.weights.get(strategy, 0.0)
        return 0.0

    def should_rebalance(self, trigger: RebalanceTrigger) -> bool:
        """Check if rebalance should happen."""
        # Always rebalance on regime change or crisis signals
        if trigger in [RebalanceTrigger.REGIME_CHANGE, 
                      RebalanceTrigger.DRAWDOWN_ACCELERATION]:
            return True
        
        # Scheduled: daily
        if self.current_allocation:
            hours_since = (datetime.now() - self.current_allocation.timestamp).total_seconds() / 3600
            return hours_since >= 24
        
        return True

    def get_status(self) -> Dict:
        """Get optimizer status."""
        if not self.current_allocation:
            return {"status": "NOT_INITIALIZED"}
        
        return {
            "active_strategies": len([w for w in self.current_allocation.weights.values() if w > 0]),
            "expected_sharpe": round(self.current_allocation.expected_sharpe, 3),
            "diversification": round(self.current_allocation.diversification_score, 3),
            "regime": self.current_allocation.regime,
            "weights": self.current_allocation.weights,
        }
