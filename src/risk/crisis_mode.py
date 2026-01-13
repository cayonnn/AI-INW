"""
crisis_mode.py
===============
Crisis Mode Controller – Risk-Off Portfolio Shift

ไม่ใช่แค่ "หยุดเทรด" แต่คือ เปลี่ยนโหมดสมองของทั้งระบบ
เป้าหมาย: รักษาทุน + อยู่รอด

State Machine:
NORMAL → DEFENSIVE → SURVIVAL → DEFENSIVE → NORMAL

กองทุนไม่พยายาม "ชนะตลาด" ตอนตลาดบ้า
เขา "ไม่แพ้" แล้วรอวันถัดไป
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from src.utils.logger import get_logger

logger = get_logger("CRISIS_MODE")


class CrisisMode(str, Enum):
    """Crisis mode levels."""
    NORMAL = "NORMAL"         # Full operation
    DEFENSIVE = "DEFENSIVE"   # Reduced risk
    SURVIVAL = "SURVIVAL"     # Capital preservation only


class StrategyType(str, Enum):
    """Strategy types for re-weighting."""
    SCALPING = "SCALPING"
    BREAKOUT = "BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"
    HEDGE = "HEDGE"
    HTF_TREND = "HTF_TREND"


@dataclass
class CrisisMetrics:
    """Metrics for crisis mode evaluation."""
    volatility_zscore: float = 0.0
    atr_zscore: float = 0.0
    vix_level: float = 20.0
    correlation_score: float = 0.0
    drawdown_flat_days: int = 0
    signal_confidence: float = 0.5
    kill_events_24h: int = 0


@dataclass
class ModeConfig:
    """Configuration for each crisis mode."""
    risk_budget_multiplier: float
    max_leverage: float
    max_positions: int
    allowed_strategies: Set[StrategyType]
    allow_new_entries: bool
    order_type: str
    slippage_tolerance: float
    cooldown_multiplier: float


class CrisisModeController:
    """
    Crisis Mode Controller - Risk-Off Portfolio Shift.
    
    Changes behavior of the entire system based on market conditions.
    """

    def __init__(self):
        self.mode = CrisisMode.NORMAL
        self.entered_at: Optional[datetime] = None
        self.mode_history: List[Dict] = []
        
        # Kill reasons that trigger crisis mode
        self.crisis_triggers = {
            "VOLATILITY_SHOCK",
            "LIQUIDITY_CRISIS",
            "CORRELATION_BREAKDOWN",
            "CONFIDENCE_COLLAPSE",
        }
        
        # Mode configurations
        self.mode_configs = {
            CrisisMode.NORMAL: ModeConfig(
                risk_budget_multiplier=1.0,
                max_leverage=1.0,
                max_positions=5,
                allowed_strategies={s for s in StrategyType},
                allow_new_entries=True,
                order_type="MARKET",
                slippage_tolerance=1.0,
                cooldown_multiplier=1.0,
            ),
            CrisisMode.DEFENSIVE: ModeConfig(
                risk_budget_multiplier=0.3,
                max_leverage=0.3,
                max_positions=2,
                allowed_strategies={StrategyType.HTF_TREND, StrategyType.MEAN_REVERSION, StrategyType.HEDGE},
                allow_new_entries=True,
                order_type="LIMIT",
                slippage_tolerance=0.3,
                cooldown_multiplier=2.0,
            ),
            CrisisMode.SURVIVAL: ModeConfig(
                risk_budget_multiplier=0.05,
                max_leverage=0.1,
                max_positions=1,
                allowed_strategies={StrategyType.HEDGE},
                allow_new_entries=False,
                order_type="LIMIT",
                slippage_tolerance=0.1,
                cooldown_multiplier=3.0,
            ),
        }
        
        # Strategy weights by mode
        self.strategy_weights = {
            CrisisMode.NORMAL: {
                StrategyType.SCALPING: 1.0,
                StrategyType.BREAKOUT: 1.0,
                StrategyType.MEAN_REVERSION: 0.7,
                StrategyType.HEDGE: 0.3,
                StrategyType.HTF_TREND: 0.8,
            },
            CrisisMode.DEFENSIVE: {
                StrategyType.SCALPING: 0.0,
                StrategyType.BREAKOUT: 0.2,
                StrategyType.MEAN_REVERSION: 1.0,
                StrategyType.HEDGE: 0.8,
                StrategyType.HTF_TREND: 0.6,
            },
            CrisisMode.SURVIVAL: {
                StrategyType.SCALPING: 0.0,
                StrategyType.BREAKOUT: 0.0,
                StrategyType.MEAN_REVERSION: 0.5,
                StrategyType.HEDGE: 1.0,
                StrategyType.HTF_TREND: 0.0,
            },
        }
        
        # Thresholds
        self.thresholds = {
            "vol_zscore_defensive": 2.5,
            "vol_zscore_survival": 4.0,
            "vol_zscore_exit": 1.5,
            "correlation_defensive": 0.7,
            "correlation_survival": 0.9,
            "vix_defensive": 25,
            "vix_survival": 35,
            "min_stable_days": 3,
            "min_exit_confidence": 0.6,
        }

    # -------------------------------------------------
    # Main evaluation
    # -------------------------------------------------
    def evaluate(self, metrics: CrisisMetrics, kill_reasons: List[str] = None):
        """
        Evaluate conditions and update mode.
        """
        kill_reasons = kill_reasons or []
        
        # Check for crisis triggers
        has_crisis_trigger = any(r in self.crisis_triggers for r in kill_reasons)
        
        if self._should_enter_survival(metrics, has_crisis_trigger):
            self._activate(CrisisMode.SURVIVAL, metrics)
        elif self._should_enter_defensive(metrics, has_crisis_trigger):
            self._activate(CrisisMode.DEFENSIVE, metrics)
        elif self._can_exit_to_normal(metrics):
            self._activate(CrisisMode.NORMAL, metrics)

    def _should_enter_survival(self, m: CrisisMetrics, has_trigger: bool) -> bool:
        """Check if should enter SURVIVAL mode."""
        if self.mode == CrisisMode.SURVIVAL:
            return False
        
        return (
            m.volatility_zscore > self.thresholds["vol_zscore_survival"] or
            m.correlation_score > self.thresholds["correlation_survival"] or
            m.vix_level > self.thresholds["vix_survival"] or
            m.kill_events_24h >= 3
        )

    def _should_enter_defensive(self, m: CrisisMetrics, has_trigger: bool) -> bool:
        """Check if should enter DEFENSIVE mode."""
        if self.mode in [CrisisMode.DEFENSIVE, CrisisMode.SURVIVAL]:
            return False
        
        return (
            m.volatility_zscore > self.thresholds["vol_zscore_defensive"] or
            m.correlation_score > self.thresholds["correlation_defensive"] or
            m.vix_level > self.thresholds["vix_defensive"] or
            has_trigger or
            m.kill_events_24h >= 1
        )

    def _can_exit_to_normal(self, m: CrisisMetrics) -> bool:
        """Check if can return to NORMAL mode."""
        if self.mode == CrisisMode.NORMAL:
            return False
        
        # Must pass all conditions
        return (
            m.volatility_zscore < self.thresholds["vol_zscore_exit"] and
            m.drawdown_flat_days >= self.thresholds["min_stable_days"] and
            m.signal_confidence > self.thresholds["min_exit_confidence"] and
            m.kill_events_24h == 0
        )

    def _activate(self, new_mode: CrisisMode, metrics: CrisisMetrics):
        """Activate a new mode."""
        if self.mode == new_mode:
            return
        
        old_mode = self.mode
        self.mode = new_mode
        self.entered_at = datetime.now()
        
        self.mode_history.append({
            "from": old_mode.value,
            "to": new_mode.value,
            "timestamp": self.entered_at.isoformat(),
            "metrics": {
                "vol_z": metrics.volatility_zscore,
                "vix": metrics.vix_level,
            },
        })
        
        if new_mode == CrisisMode.SURVIVAL:
            logger.critical(f"🚨 CRISIS MODE → SURVIVAL (from {old_mode.value})")
        elif new_mode == CrisisMode.DEFENSIVE:
            logger.warning(f"⚠️ CRISIS MODE → DEFENSIVE (from {old_mode.value})")
        else:
            logger.info(f"✅ CRISIS MODE → NORMAL (from {old_mode.value})")

    # -------------------------------------------------
    # Portfolio adjustments
    # -------------------------------------------------
    def get_config(self) -> ModeConfig:
        """Get current mode configuration."""
        return self.mode_configs[self.mode]

    def get_risk_budget(self, base_risk: float) -> float:
        """Get adjusted risk budget for current mode."""
        config = self.get_config()
        return base_risk * config.risk_budget_multiplier

    def get_strategy_weight(self, strategy: StrategyType) -> float:
        """Get weight for a strategy in current mode."""
        return self.strategy_weights[self.mode].get(strategy, 0.0)

    def is_strategy_allowed(self, strategy: StrategyType) -> bool:
        """Check if strategy is allowed in current mode."""
        config = self.get_config()
        return strategy in config.allowed_strategies

    def can_open_position(self, current_positions: int) -> bool:
        """Check if can open new position."""
        config = self.get_config()
        if not config.allow_new_entries:
            return False
        return current_positions < config.max_positions

    def get_order_type(self) -> str:
        """Get order type for current mode."""
        return self.get_config().order_type

    def get_slippage_tolerance(self, base_tolerance: float) -> float:
        """Get adjusted slippage tolerance."""
        config = self.get_config()
        return base_tolerance * config.slippage_tolerance

    def get_cooldown_multiplier(self) -> float:
        """Get trade cooldown multiplier."""
        return self.get_config().cooldown_multiplier

    # -------------------------------------------------
    # Status
    # -------------------------------------------------
    def get_status(self) -> Dict:
        """Get current status."""
        config = self.get_config()
        return {
            "mode": self.mode.value,
            "entered_at": self.entered_at.isoformat() if self.entered_at else None,
            "risk_budget_pct": config.risk_budget_multiplier * 100,
            "max_positions": config.max_positions,
            "allow_entries": config.allow_new_entries,
            "allowed_strategies": [s.value for s in config.allowed_strategies],
        }

    def is_normal(self) -> bool:
        """Check if in normal mode."""
        return self.mode == CrisisMode.NORMAL

    def is_crisis(self) -> bool:
        """Check if in any crisis mode."""
        return self.mode != CrisisMode.NORMAL

    def force_mode(self, mode: CrisisMode, reason: str = "Manual override"):
        """Force a specific mode (for manual intervention)."""
        logger.warning(f"⚠️ FORCE MODE: {mode.value} - {reason}")
        self.mode = mode
        self.entered_at = datetime.now()
