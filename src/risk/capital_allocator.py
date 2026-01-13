"""
capital_allocator.py
=====================
Dynamic Capital Re-Allocation Engine (DCRA)

ไม่ใช่แค่ "เพิ่ม-ลด lot"
แต่คือ จัดสรรทุนใหม่ตลอดเวลา ตาม Alpha + Risk

Core Philosophy:
- Capital flows to strength
- Risk leaves decay
- No strategy is permanent
- Allocation is a living system

กองทุนไม่ได้ถามว่า "เคยดีไหม"
แต่ถามว่า "ตอนนี้ยังควรถือทุนไหม"
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger("CAPITAL_ALLOCATOR")


class RebalanceMode(str, Enum):
    """Rebalance frequency modes."""
    NORMAL = "NORMAL"         # Daily
    DEFENSIVE = "DEFENSIVE"   # Every 2-3 days
    CRISIS = "CRISIS"         # Frozen
    RECOVERY = "RECOVERY"     # Gradual step-up


class CapitalBucket(str, Enum):
    """Capital bucket types."""
    CORE = "CORE"             # Long-term stable alpha (50%)
    TACTICAL = "TACTICAL"     # Short-term opportunity (30%)
    DEFENSIVE = "DEFENSIVE"   # Hedge / carry (15%)
    RESERVE = "RESERVE"       # Dry powder (5%)


@dataclass
class StrategyMetrics:
    """Metrics for strategy scoring."""
    strategy_name: str
    rolling_sharpe: float = 0.0           # EWMA Sharpe ratio
    alpha_decay_factor: float = 1.0       # 1.0 = no decay, 0.0 = fully decayed
    pnl_stability: float = 0.5            # PnL consistency
    max_drawdown: float = 0.0             # Current max drawdown
    drawdown_recovery_days: int = 0       # Days since drawdown low
    signal_confidence: float = 0.5        # Average signal confidence
    volatility: float = 0.01              # Strategy volatility
    correlation_with_portfolio: float = 0.0
    regime_compatibility: float = 1.0     # 1.0 = supports current regime
    trade_count: int = 0                  # Recent trade count


@dataclass
class AllocationResult:
    """Capital allocation result for a strategy."""
    strategy_name: str
    weight: float                         # 0.0 to 1.0
    bucket: CapitalBucket
    capital_amount: float
    score: float
    is_frozen: bool = False


class DynamicCapitalAllocator:
    """
    Dynamic Capital Re-Allocation Engine.
    
    กองทุนไม่ "ฆ่า strategy"
    เขา ปล่อยให้มันอดทุนจนหายไปเอง
    """

    def __init__(self, total_capital: float = 10000.0):
        self.total_capital = total_capital
        self.strategy_weights: Dict[str, float] = {}
        self.strategy_scores: Dict[str, float] = {}
        self.strategy_buckets: Dict[str, CapitalBucket] = {}
        
        self.mode = RebalanceMode.NORMAL
        self.is_frozen = False
        self.last_rebalance: Optional[datetime] = None
        
        # Constraints
        self.min_weight = 0.05
        self.max_weight = 0.60
        self.correlation_threshold = 0.7
        
        # Capital bucket splits
        self.bucket_splits = {
            CapitalBucket.CORE: 0.50,
            CapitalBucket.TACTICAL: 0.30,
            CapitalBucket.DEFENSIVE: 0.15,
            CapitalBucket.RESERVE: 0.05,
        }
        
        # Scoring weights
        self.score_weights = {
            "sharpe": 0.30,
            "alpha_decay": 0.25,
            "pnl_stability": 0.20,
            "drawdown_inverse": 0.15,
            "confidence": 0.10,
        }

    # -------------------------------------------------
    # Main rebalance
    # -------------------------------------------------
    def rebalance(self, strategy_metrics: List[StrategyMetrics], 
                  current_regime: str = None) -> Dict[str, AllocationResult]:
        """
        Rebalance capital across strategies.
        
        Args:
            strategy_metrics: List of strategy performance metrics
            current_regime: Current market regime for filtering
            
        Returns:
            Dictionary of strategy allocations
        """
        if self.is_frozen:
            logger.warning("Capital allocation is FROZEN - skipping rebalance")
            return self._get_current_allocations()
        
        if not self._should_rebalance():
            return self._get_current_allocations()
        
        logger.info("Rebalancing capital allocation...")
        
        # 1. Compute scores
        scores = self._compute_scores(strategy_metrics, current_regime)
        self.strategy_scores = scores
        
        # 2. Compute raw weights
        raw_weights = self._scores_to_weights(scores)
        
        # 3. Apply correlation penalty
        penalized_weights = self._apply_correlation_penalty(
            raw_weights, strategy_metrics
        )
        
        # 4. Apply constraints
        constrained_weights = self._apply_constraints(penalized_weights)
        
        # 5. Normalize
        final_weights = self._normalize_weights(constrained_weights)
        self.strategy_weights = final_weights
        
        # 6. Assign buckets
        allocations = self._create_allocations(final_weights, scores)
        
        self.last_rebalance = datetime.now()
        
        logger.info(f"Rebalance complete: {len(allocations)} strategies")
        for name, alloc in allocations.items():
            logger.debug(f"  {name}: {alloc.weight:.2%} ({alloc.bucket.value})")
        
        return allocations

    # -------------------------------------------------
    # Scoring
    # -------------------------------------------------
    def _compute_scores(self, metrics_list: List[StrategyMetrics], 
                        regime: str = None) -> Dict[str, float]:
        """Compute strategy scores."""
        scores = {}
        
        for m in metrics_list:
            # Base score components
            sharpe_score = max(0, m.rolling_sharpe) / 3.0  # Normalize to ~0-1
            alpha_score = m.alpha_decay_factor
            pnl_score = m.pnl_stability
            
            # Drawdown inverse (lower drawdown = higher score)
            dd_score = max(0, 1 - abs(m.max_drawdown) * 10)
            
            conf_score = m.signal_confidence
            
            # Weighted sum
            score = (
                sharpe_score * self.score_weights["sharpe"] +
                alpha_score * self.score_weights["alpha_decay"] +
                pnl_score * self.score_weights["pnl_stability"] +
                dd_score * self.score_weights["drawdown_inverse"] +
                conf_score * self.score_weights["confidence"]
            )
            
            # Risk adjustment
            epsilon = 0.001
            risk_adjusted = score / (m.volatility + epsilon)
            
            # Regime filter
            if m.regime_compatibility < 0.5:
                risk_adjusted *= 0.2
            
            scores[m.strategy_name] = max(0, risk_adjusted)
        
        return scores

    def _scores_to_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Convert scores to raw weights."""
        total = sum(scores.values())
        if total == 0:
            return {k: 0.0 for k in scores}
        
        return {k: v / total for k, v in scores.items()}

    # -------------------------------------------------
    # Constraints & Penalties
    # -------------------------------------------------
    def _apply_correlation_penalty(self, weights: Dict[str, float],
                                   metrics_list: List[StrategyMetrics]) -> Dict[str, float]:
        """Penalize correlated strategies."""
        metrics_map = {m.strategy_name: m for m in metrics_list}
        penalized = weights.copy()
        
        # Simple pairwise correlation check
        strategies = list(weights.keys())
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                m1 = metrics_map.get(s1)
                m2 = metrics_map.get(s2)
                
                if m1 and m2:
                    # If both highly correlated with portfolio
                    if (m1.correlation_with_portfolio > self.correlation_threshold and
                        m2.correlation_with_portfolio > self.correlation_threshold):
                        # Penalize lower Sharpe
                        if m1.rolling_sharpe < m2.rolling_sharpe:
                            penalized[s1] *= 0.5
                        else:
                            penalized[s2] *= 0.5
        
        return penalized

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max weight constraints."""
        constrained = {}
        
        for name, weight in weights.items():
            if weight < self.min_weight:
                constrained[name] = 0.0  # Below minimum = exclude
            elif weight > self.max_weight:
                constrained[name] = self.max_weight
            else:
                constrained[name] = weight
        
        return constrained

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total == 0:
            return weights
        
        return {k: v / total for k, v in weights.items()}

    # -------------------------------------------------
    # Allocation & Buckets
    # -------------------------------------------------
    def _create_allocations(self, weights: Dict[str, float],
                            scores: Dict[str, float]) -> Dict[str, AllocationResult]:
        """Create final allocations with buckets."""
        allocations = {}
        
        # Sort by score for bucket assignment
        sorted_strategies = sorted(
            weights.items(),
            key=lambda x: scores.get(x[0], 0),
            reverse=True
        )
        
        # Assign buckets (highest scores = core)
        for i, (name, weight) in enumerate(sorted_strategies):
            if weight == 0:
                continue
            
            # Assign bucket based on rank
            if i == 0:
                bucket = CapitalBucket.CORE
            elif i <= 2:
                bucket = CapitalBucket.TACTICAL
            else:
                bucket = CapitalBucket.DEFENSIVE
            
            self.strategy_buckets[name] = bucket
            
            allocations[name] = AllocationResult(
                strategy_name=name,
                weight=weight,
                bucket=bucket,
                capital_amount=self.total_capital * weight,
                score=scores.get(name, 0),
                is_frozen=self.is_frozen,
            )
        
        return allocations

    # -------------------------------------------------
    # Alpha rotation
    # -------------------------------------------------
    def rotate_alpha(self, decaying_strategy: str, 
                    strategy_metrics: List[StrategyMetrics]) -> Optional[str]:
        """
        Shift capital from decaying strategy to best peer.
        
        ไม่ปิด strategy แต่ "อดอาหาร"
        """
        if decaying_strategy not in self.strategy_weights:
            return None
        
        # Find best peer
        best_peer = None
        best_score = 0
        
        for m in strategy_metrics:
            if m.strategy_name == decaying_strategy:
                continue
            score = self.strategy_scores.get(m.strategy_name, 0)
            if score > best_score:
                best_score = score
                best_peer = m.strategy_name
        
        if best_peer:
            # Transfer half of decaying weight
            transfer_amount = self.strategy_weights[decaying_strategy] * 0.5
            self.strategy_weights[decaying_strategy] -= transfer_amount
            self.strategy_weights[best_peer] = self.strategy_weights.get(best_peer, 0) + transfer_amount
            
            logger.info(f"Alpha rotation: {decaying_strategy} → {best_peer} ({transfer_amount:.2%})")
            
        return best_peer

    # -------------------------------------------------
    # Mode & Freeze control
    # -------------------------------------------------
    def set_mode(self, mode: RebalanceMode):
        """Set rebalance mode."""
        self.mode = mode
        if mode == RebalanceMode.CRISIS:
            self.freeze()
        logger.info(f"Capital allocator mode: {mode.value}")

    def freeze(self):
        """Freeze all reallocation."""
        self.is_frozen = True
        logger.warning("⛔ Capital allocation FROZEN")

    def unfreeze(self):
        """Unfreeze reallocation."""
        self.is_frozen = False
        logger.info("✅ Capital allocation UNFROZEN")

    def _should_rebalance(self) -> bool:
        """Check if should rebalance based on mode."""
        if self.last_rebalance is None:
            return True
        
        days_since = (datetime.now() - self.last_rebalance).days
        
        if self.mode == RebalanceMode.NORMAL:
            return days_since >= 1
        elif self.mode == RebalanceMode.DEFENSIVE:
            return days_since >= 2
        elif self.mode == RebalanceMode.CRISIS:
            return False
        elif self.mode == RebalanceMode.RECOVERY:
            return days_since >= 3
        
        return True

    # -------------------------------------------------
    # Getters
    # -------------------------------------------------
    def get_budget(self, strategy: str) -> float:
        """Get capital budget for a strategy."""
        weight = self.strategy_weights.get(strategy, 0.0)
        return self.total_capital * weight

    def get_weight(self, strategy: str) -> float:
        """Get weight for a strategy."""
        return self.strategy_weights.get(strategy, 0.0)

    def get_bucket(self, strategy: str) -> Optional[CapitalBucket]:
        """Get bucket for a strategy."""
        return self.strategy_buckets.get(strategy)

    def _get_current_allocations(self) -> Dict[str, AllocationResult]:
        """Get current allocations without rebalancing."""
        allocations = {}
        for name, weight in self.strategy_weights.items():
            allocations[name] = AllocationResult(
                strategy_name=name,
                weight=weight,
                bucket=self.strategy_buckets.get(name, CapitalBucket.TACTICAL),
                capital_amount=self.total_capital * weight,
                score=self.strategy_scores.get(name, 0),
                is_frozen=self.is_frozen,
            )
        return allocations

    def get_status(self) -> Dict:
        """Get allocator status."""
        return {
            "mode": self.mode.value,
            "is_frozen": self.is_frozen,
            "total_capital": self.total_capital,
            "strategy_count": len(self.strategy_weights),
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "weights": self.strategy_weights,
        }

    def update_capital(self, new_capital: float):
        """Update total capital."""
        self.total_capital = new_capital
        logger.info(f"Capital updated: {new_capital:.2f}")
