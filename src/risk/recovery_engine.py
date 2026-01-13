"""
recovery_engine.py
===================
Post-Crisis Recovery Engine (PCRE)

"สมองฟื้นตัว" หลังรอดจากพายุ

เป้าหมายไม่ใช่ "เอาคืนเร็ว"
เป้าหมายคือ กลับสู่ Alpha อย่างมีวินัย

State Machine:
SURVIVAL → STABILIZATION → RE-ENTRY → NORMAL

Core Philosophy:
1. Capital First – ฟื้นความเสถียรของทุนก่อน
2. Confidence Second – ฟื้นความน่าเชื่อถือของสัญญาณ
3. Alpha Last – ค่อยเพิ่มความเสี่ยง

Recovery ≠ Revenge
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from src.utils.logger import get_logger

logger = get_logger("RECOVERY_ENGINE")


class RecoveryState(str, Enum):
    """Recovery state levels."""
    SURVIVAL = "SURVIVAL"           # Still in crisis
    STABILIZATION = "STABILIZATION" # Market stabilizing
    RE_ENTRY = "RE_ENTRY"          # Gradual re-entry
    NORMAL = "NORMAL"              # Fully recovered


class StrategyType(str, Enum):
    """Strategy types for recovery ordering."""
    MEAN_REVERSION = "MEAN_REVERSION"
    HTF_TREND = "HTF_TREND"
    BREAKOUT = "BREAKOUT"
    SCALPING = "SCALPING"


@dataclass
class RecoveryMetrics:
    """Metrics for recovery evaluation."""
    # Market Stability
    atr_zscore: float = 0.0
    spread_normalized: bool = True
    liquidity_shock_events: int = 0
    
    # Equity Behavior
    drawdown_flat_bars: int = 0
    new_equity_low: bool = False
    equity_variance: float = 0.0
    
    # Signal Quality
    mean_confidence: float = 0.5
    signal_entropy: float = 0.5
    signal_agreement: float = 0.5
    
    # Recovery Progress
    consecutive_profitable_days: int = 0
    recovery_pnl_stability: float = 0.0


@dataclass
class RecoveryConfig:
    """Configuration for each recovery state."""
    risk_budget_multiplier: float
    max_positions: int
    allowed_strategy_count: int
    correlation_limit: float
    allow_pyramiding: bool
    min_days_to_next: int


class PostCrisisRecoveryEngine:
    """
    Post-Crisis Recovery Engine.
    
    กองทุนไม่ฟื้นพร้อมทุก strategy
    เขาฟื้นทีละ "Alpha Source"
    """

    def __init__(self):
        self.state = RecoveryState.SURVIVAL
        self.recovery_start: Optional[datetime] = None
        self.state_entered_at: Optional[datetime] = None
        self.days_in_state: int = 0
        self.state_history: List[Dict] = []
        
        # State configurations
        self.state_configs = {
            RecoveryState.SURVIVAL: RecoveryConfig(
                risk_budget_multiplier=0.05,
                max_positions=0,
                allowed_strategy_count=0,
                correlation_limit=0.0,
                allow_pyramiding=False,
                min_days_to_next=1,
            ),
            RecoveryState.STABILIZATION: RecoveryConfig(
                risk_budget_multiplier=0.15,
                max_positions=1,
                allowed_strategy_count=1,
                correlation_limit=0.2,
                allow_pyramiding=False,
                min_days_to_next=3,
            ),
            RecoveryState.RE_ENTRY: RecoveryConfig(
                risk_budget_multiplier=0.40,
                max_positions=3,
                allowed_strategy_count=2,
                correlation_limit=0.4,
                allow_pyramiding=False,
                min_days_to_next=5,
            ),
            RecoveryState.NORMAL: RecoveryConfig(
                risk_budget_multiplier=1.0,
                max_positions=5,
                allowed_strategy_count=4,
                correlation_limit=0.7,
                allow_pyramiding=True,
                min_days_to_next=0,
            ),
        }
        
        # Strategy re-enable order (most conservative first)
        self.strategy_enable_order = [
            StrategyType.MEAN_REVERSION,
            StrategyType.HTF_TREND,
            StrategyType.BREAKOUT,
            StrategyType.SCALPING,
        ]
        
        # Thresholds
        self.thresholds = {
            # Market Stability
            "atr_zscore_stable": 1.8,
            "max_liquidity_shocks": 0,
            
            # Equity Behavior
            "min_flat_bars": 20,
            "max_equity_variance": 0.02,
            
            # Signal Quality
            "min_confidence": 0.6,
            "max_entropy": 0.4,
            "min_agreement": 0.6,
            
            # Recovery Progress
            "min_profitable_days": 3,
            "min_pnl_stability": 0.7,
        }

    # -------------------------------------------------
    # Main evaluation
    # -------------------------------------------------
    def evaluate(self, metrics: RecoveryMetrics):
        """Evaluate and transition recovery state."""
        # Update days in state
        if self.state_entered_at:
            self.days_in_state = (datetime.now() - self.state_entered_at).days
        
        # State transitions
        if self.state == RecoveryState.SURVIVAL:
            if self._can_stabilize(metrics):
                self._transition(RecoveryState.STABILIZATION)
        
        elif self.state == RecoveryState.STABILIZATION:
            if self._can_reenter(metrics):
                self._transition(RecoveryState.RE_ENTRY)
        
        elif self.state == RecoveryState.RE_ENTRY:
            if self._can_normalize(metrics):
                self._transition(RecoveryState.NORMAL)

    def _can_stabilize(self, m: RecoveryMetrics) -> bool:
        """Check if can move from SURVIVAL to STABILIZATION."""
        config = self.state_configs[RecoveryState.SURVIVAL]
        
        if self.days_in_state < config.min_days_to_next:
            return False
        
        return (
            m.atr_zscore < self.thresholds["atr_zscore_stable"] and
            m.spread_normalized and
            m.liquidity_shock_events <= self.thresholds["max_liquidity_shocks"]
        )

    def _can_reenter(self, m: RecoveryMetrics) -> bool:
        """Check if can move from STABILIZATION to RE-ENTRY."""
        config = self.state_configs[RecoveryState.STABILIZATION]
        
        if self.days_in_state < config.min_days_to_next:
            return False
        
        return (
            m.atr_zscore < self.thresholds["atr_zscore_stable"] and
            m.drawdown_flat_bars >= self.thresholds["min_flat_bars"] and
            not m.new_equity_low and
            m.mean_confidence >= self.thresholds["min_confidence"] and
            m.signal_agreement >= self.thresholds["min_agreement"]
        )

    def _can_normalize(self, m: RecoveryMetrics) -> bool:
        """Check if can move from RE-ENTRY to NORMAL."""
        config = self.state_configs[RecoveryState.RE_ENTRY]
        
        if self.days_in_state < config.min_days_to_next:
            return False
        
        return (
            m.consecutive_profitable_days >= self.thresholds["min_profitable_days"] and
            m.recovery_pnl_stability >= self.thresholds["min_pnl_stability"] and
            m.equity_variance < self.thresholds["max_equity_variance"] and
            m.signal_entropy < self.thresholds["max_entropy"]
        )

    def _transition(self, new_state: RecoveryState):
        """Transition to new recovery state."""
        old_state = self.state
        self.state = new_state
        self.state_entered_at = datetime.now()
        self.days_in_state = 0
        
        if self.recovery_start is None:
            self.recovery_start = datetime.now()
        
        self.state_history.append({
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": datetime.now().isoformat(),
        })
        
        emoji = {
            RecoveryState.SURVIVAL: "🔴",
            RecoveryState.STABILIZATION: "🟡",
            RecoveryState.RE_ENTRY: "🟢",
            RecoveryState.NORMAL: "🔵",
        }
        
        logger.info(f"{emoji[new_state]} RECOVERY STATE → {new_state.value}")

    # -------------------------------------------------
    # Portfolio adjustments
    # -------------------------------------------------
    def get_config(self) -> RecoveryConfig:
        """Get current state configuration."""
        return self.state_configs[self.state]

    def get_risk_budget(self, base_risk: float) -> float:
        """Get adjusted risk budget."""
        return base_risk * self.get_config().risk_budget_multiplier

    def get_allowed_strategies(self) -> List[StrategyType]:
        """Get list of allowed strategies in current state."""
        config = self.get_config()
        return self.strategy_enable_order[:config.allowed_strategy_count]

    def is_strategy_allowed(self, strategy: StrategyType) -> bool:
        """Check if strategy is allowed."""
        allowed = self.get_allowed_strategies()
        return strategy in allowed

    def can_open_position(self, current_positions: int) -> bool:
        """Check if can open new position."""
        config = self.get_config()
        return current_positions < config.max_positions

    def get_correlation_limit(self) -> float:
        """Get correlation limit for current state."""
        return self.get_config().correlation_limit

    def can_pyramid(self) -> bool:
        """Check if pyramiding is allowed."""
        return self.get_config().allow_pyramiding

    # -------------------------------------------------
    # Confidence rebuild
    # -------------------------------------------------
    def calculate_confidence_score(self, m: RecoveryMetrics) -> float:
        """
        Calculate overall recovery confidence.
        
        confidence_score = (
            signal_consistency * 0.4 +
            pnl_stability * 0.3 +
            volatility_normality * 0.3
        )
        """
        signal_consistency = m.signal_agreement * (1 - m.signal_entropy)
        pnl_stability = m.recovery_pnl_stability
        volatility_normality = max(0, 1 - (m.atr_zscore / 3))
        
        return (
            signal_consistency * 0.4 +
            pnl_stability * 0.3 +
            volatility_normality * 0.3
        )

    # -------------------------------------------------
    # Anti-overconfidence guards
    # -------------------------------------------------
    def should_disable_aggressive_entries(self) -> bool:
        """Disable aggressive entries during recovery."""
        return self.state != RecoveryState.NORMAL

    def get_trade_frequency_cap(self, normal_cap: int) -> int:
        """Get capped trade frequency."""
        caps = {
            RecoveryState.SURVIVAL: 0,
            RecoveryState.STABILIZATION: max(1, normal_cap // 4),
            RecoveryState.RE_ENTRY: max(2, normal_cap // 2),
            RecoveryState.NORMAL: normal_cap,
        }
        return caps[self.state]

    def get_daily_loss_limit(self, normal_limit: float) -> float:
        """Get tighter daily loss limit during recovery."""
        multipliers = {
            RecoveryState.SURVIVAL: 0.1,
            RecoveryState.STABILIZATION: 0.3,
            RecoveryState.RE_ENTRY: 0.5,
            RecoveryState.NORMAL: 1.0,
        }
        return normal_limit * multipliers[self.state]

    # -------------------------------------------------
    # Status
    # -------------------------------------------------
    def get_status(self) -> Dict:
        """Get current status."""
        config = self.get_config()
        return {
            "state": self.state.value,
            "days_in_state": self.days_in_state,
            "recovery_start": self.recovery_start.isoformat() if self.recovery_start else None,
            "risk_budget_pct": config.risk_budget_multiplier * 100,
            "max_positions": config.max_positions,
            "allowed_strategies": [s.value for s in self.get_allowed_strategies()],
            "correlation_limit": config.correlation_limit,
        }

    def is_recovered(self) -> bool:
        """Check if fully recovered."""
        return self.state == RecoveryState.NORMAL

    def is_recovering(self) -> bool:
        """Check if in recovery process."""
        return self.state in [RecoveryState.STABILIZATION, RecoveryState.RE_ENTRY]

    def reset(self):
        """Reset to survival state (for entering new crisis)."""
        self.state = RecoveryState.SURVIVAL
        self.recovery_start = None
        self.state_entered_at = datetime.now()
        self.days_in_state = 0
        logger.warning("🔴 RECOVERY ENGINE RESET - Entering SURVIVAL")
